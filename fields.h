// Copyright 2017 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef FIELDS_H_
#define FIELDS_H_

// Encodes/decodes uint32_t or byte array fields in header/sections.

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <vector>

#include "bit_reader.h"
#include "bits.h"
#include "compiler_specific.h"
#include "status.h"
#include "write_bits.h"

namespace pik {

// Uses a 2-bit "selector" to either represent a value directly, or the number
// of extra bits to send. This is faster to decode and denser than Exp-Golomb or
// Gamma codes when both small and large values occur.
//
// "distribution" is interpreted as an array of four bytes (least-significant
// first). Let b = Lookup(distribution, selector). If b & 0x80, then the value
// is b & 0x7F and no extra bits are sent. Otherwise, b indicates the number of
// subsequent value bits (1 to 32).
//
// Examples:
// Direct value: distribution 0x06A09088, value 32 => 10 (selector 2, b=0xA0).
// Extra bits: distribution 0x08060402, value 7 => 01 0111 (selector 1, b=4).
class U32Coder {
 public:
  static size_t MaxEncodedBits(const uint32_t distribution) {
    ValidateDistribution(distribution);
    size_t extra_bits = 0;
    for (int selector = 0; selector < 4; ++selector) {
      const size_t b = Lookup(distribution, selector);
      if (b & 0x80) continue;
      extra_bits = std::max(extra_bits, b);
    }
    return 2 + extra_bits;
  }

  static bool CanEncode(const uint32_t distribution, const uint32_t value,
                        size_t* PIK_RESTRICT encoded_bits) {
    ValidateDistribution(distribution);
    int selector;
    size_t extra_bits;
    const bool ok = ChooseEncoding(distribution, value, &selector, &extra_bits);
    *encoded_bits = ok ? 2 + extra_bits : 0;
    return ok;
  }

  static uint32_t Load(const uint32_t distribution,
                       BitReader* PIK_RESTRICT reader) {
    ValidateDistribution(distribution);
    const int selector = reader->ReadFixedBits<2>();
    const size_t b = Lookup(distribution, selector);
    if (b & 0x80) return b & 0x7F;
    return reader->ReadBits(b);
  }

  // Returns false if the value is too large to encode.
  static bool Store(const uint32_t distribution, const uint32_t value,
                    size_t* pos, uint8_t* storage) {
    ValidateDistribution(distribution);
    int selector;
    size_t extra_bits;
    if (!ChooseEncoding(distribution, value, &selector, &extra_bits)) {
      return false;
    }

    WriteBits(2, selector, pos, storage);
    if (extra_bits != 0) {
      WriteBits(extra_bits, value, pos, storage);
    }
    return true;
  }

 private:
  // Returns one byte from "distribution" at index "selector".
  static size_t Lookup(const uint32_t distribution, const int selector) {
    return (distribution >> (selector * 8)) & 0xFF;
  }

  // Ensures each b is either value|0x80, or 1..32 extra bits.
  static void ValidateDistribution(const uint32_t distribution) {
#if defined(PIK_ENABLE_ASSERT)
    for (int selector = 0; selector < 4; ++selector) {
      const size_t b = Lookup(distribution, selector);
      if (b & 0x80) continue;
      if (b == 0 || b > 32) {
        printf("Invalid distribution %8x[%d] == %zu\n", distribution, selector,
               b);
        PIK_ASSERT(false);
      }
    }
#endif
  }

  static bool ChooseEncoding(const uint32_t distribution, const uint32_t value,
                             int* PIK_RESTRICT selector,
                             size_t* PIK_RESTRICT extra_bits) {
    const size_t bits_required = 32 - NumZeroBitsAboveMSB(value);
    PIK_ASSERT(bits_required <= 32);

    // It is difficult to verify whether "distribution" is sorted, so check all
    // selectors and keep the one with the fewest extra bits.
    *extra_bits = 33;  // more than any valid encoding
    for (int s = 0; s < 4; ++s) {
      const size_t b = Lookup(distribution, s);
      if (b & 0x80) {
        if ((b & 0x7F) == value) {
          *selector = s;
          *extra_bits = 0;
        }
        continue;
      }

      if (bits_required > b) continue;

      if (b < *extra_bits) {
        *extra_bits = b;
        *selector = s;
      }
    }

    if (*extra_bits == 33) {
      *selector = 0;
      return PIK_FAILURE("No feasible selector found");
    }

    return true;
  }
};

// Coder for byte arrays: stores #bytes via U32Coder, then raw bytes.
class BytesCoder {
 public:
  static bool CanEncode(const std::vector<uint8_t>& value,
                        size_t* PIK_RESTRICT encoded_bits) {
    if (!U32Coder::CanEncode(kDistribution, value.size(), encoded_bits)) {
      return false;
    }
    *encoded_bits += value.size() * 8;
    return true;
  }

  static void Load(BitReader* PIK_RESTRICT reader,
                   std::vector<uint8_t>* PIK_RESTRICT value) {
    const uint32_t num_bytes = U32Coder::Load(kDistribution, reader);
    value->resize(num_bytes);

    // Read groups of bytes without calling FillBitBuffer every time.
    constexpr size_t kBytesPerGroup = 4;  // guaranteed by FillBitBuffer
    uint32_t i;
    for (i = 0; i + kBytesPerGroup <= value->size(); i += kBytesPerGroup) {
      reader->FillBitBuffer();
#if PIK_BYTE_ORDER_LITTLE
      const uint32_t buf = reader->PeekFixedBits<32>();
      reader->Advance(32);
      memcpy(value->data() + i, &buf, 4);
#else
      for (int idx_byte = 0; idx_byte < kBytesPerGroup; ++idx_byte) {
        value->data()[i + idx_byte] = reader->PeekFixedBits<8>();
        reader->Advance(8);
      }
#endif
    }

    reader->FillBitBuffer();
    for (; i < value->size(); ++i) {
      value->data()[i] = reader->PeekFixedBits<8>();
      reader->Advance(8);
    }
  }

  static bool Store(const std::vector<uint8_t>& value, size_t* PIK_RESTRICT pos,
                    uint8_t* storage) {
    if (!U32Coder::Store(kDistribution, value.size(), pos, storage))
      return false;

    size_t i = 0;
#if PIK_BYTE_ORDER_LITTLE
    // Write 4 bytes at a time
    uint32_t buf;
    for (; i + 4 <= value.size(); i += 4) {
      memcpy(&buf, value.data() + i, 4);
      WriteBits(32, buf, pos, storage);
    }
#endif

    // Write remaining bytes
    for (; i < value.size(); ++i) {
      WriteBits(8, value.data()[i], pos, storage);
    }
    return true;
  }

 private:
  static constexpr uint32_t kDistribution = 0x20181008;  // for #bytes
};

// Visitors for generating encoders/decoders for headers/sections that supply
// a VisitFields member function:

class ReadFieldsVisitor {
 public:
  ReadFieldsVisitor(BitReader* reader) : reader_(reader) {}

  void operator()(const uint32_t distribution, uint32_t* PIK_RESTRICT value) {
    *value = U32Coder::Load(distribution, reader_);
  }

  void operator()(std::vector<uint8_t>* PIK_RESTRICT value) {
    BytesCoder::Load(reader_, value);
  }

 private:
  BitReader* const reader_;
};

class CanEncodeFieldsVisitor {
 public:
  void operator()(const uint32_t distribution,
                  const uint32_t* PIK_RESTRICT value) {
    size_t encoded_bits;
    ok_ &= U32Coder::CanEncode(distribution, *value, &encoded_bits);
    encoded_bits_ += encoded_bits;
  }

  void operator()(const std::vector<uint8_t>* PIK_RESTRICT value) {
    size_t encoded_bits;
    ok_ &= BytesCoder::CanEncode(*value, &encoded_bits);
    encoded_bits_ += encoded_bits;
  }

  bool OK() const { return ok_; }
  size_t EncodedBits() const { return encoded_bits_; }

 private:
  bool ok_ = true;
  size_t encoded_bits_ = 0;
};

class WriteFieldsVisitor {
 public:
  WriteFieldsVisitor(size_t* pos, uint8_t* storage)
      : pos_(pos), storage_(storage) {}

  void operator()(const uint32_t distribution,
                  const uint32_t* PIK_RESTRICT value) {
    ok_ &= U32Coder::Store(distribution, *value, pos_, storage_);
  }

  void operator()(const std::vector<uint8_t>* PIK_RESTRICT value) {
    ok_ &= BytesCoder::Store(*value, pos_, storage_);
  }

  bool OK() const { return ok_; }

 private:
  size_t* pos_;
  uint8_t* storage_;
  bool ok_ = true;
};

}  // namespace pik

#endif  // FIELDS_H_
