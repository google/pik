// Copyright 2018 Google Inc. All Rights Reserved.
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
#include <array>
#include <memory>

#include "bit_reader.h"
#include "bits.h"
#include "brotli.h"
#include "compiler_specific.h"
#include "field_encodings.h"
#include "status.h"
#include "write_bits.h"

namespace pik {

// Chooses one of four encodings based on an a-priori "distribution":
// - raw: if IsRaw(distribution), send RawBits(distribution) = 1..32 raw bits;
//   This are values larger than ~32u, use kU32RawBits + #bits.
// - non-raw: send a 2-bit selector to choose byte b from "distribution",
//   least significant byte first. Then the value is encoded according to b:
//   -- direct: if b & 0x80, the value is b & 0x7F
//   -- offset: else if b & 0x40, the value is derived from (b & 7) + 1
//              extra bits plus an offset ((b >> 3) & 7) + 1.
//   -- extra: otherwise, the value is derived from b extra bits
//             (must be 1-32 extra bits)
// This is faster to decode and denser than Exp-Golomb or Gamma codes when both
// small and large values occur.
//
// Examples:
// Raw:    distribution 0xFFFFFFEF, value 32768 => 1000000000000000
// Direct: distribution 0x06A09088, value 32 => 10 (selector 2, b=0xA0).
// Extra:  distribution 0x08060402, value 7 => 01 0111 (selector 1, b=4).
// Offset: distribution 0x68584801, value 7 => 11 1 (selector 3, offset 5 + 1).
//
// Bit for bit example:
// An encoding mapping the following prefix code:
// 00 -> 0
// 01x -> 1..2
// 10xx -> 3..7
// 11xxxxxxxx -> 8..263
// Can be made with distrubition 0x7F514080. Dissecting this from hex digits
// left to right:
// 7: 0x40 flag for this byte and 2 bits of offset 8 for 8..263
// F: final bit of offset 8 and 3 bits setting extra to 7+1 for 8..263.
// 5: 0x40 flag for this byte and 2 bits of offset 3 for 3..7
// 1: One bit indicating window size 2 set for 3..7
// 4: 0x40 flag for this byte, no offset bits set, offset 0+1 for 1..2
// 0: no bits set in this flag, offset and extra bits set to 0 indicating an
//    offset 1 and extra 1 for 1..2
// 8: 0x80 flag set to indicate direct value for 0
// 0: bits of the direct value 0
class U32Coder {
 public:
  // Byte flag indicating direct value.
  static const uint32_t kDirect = 0x80;
  // Byte flag indicating extra bits with offset rather than pure extra bits.
  static const uint32_t kOffset = 0x40;

  static size_t MaxEncodedBits(const uint32_t distribution) {
    ValidateDistribution(distribution);
    if (IsRaw(distribution)) return RawBits(distribution);
    size_t extra_bits = 0;
    for (int selector = 0; selector < 4; ++selector) {
      const size_t b = Lookup(distribution, selector);
      if (b & kDirect) {
        continue;
      } else {
        extra_bits = std::max<size_t>(extra_bits, GetExtraBits(b));
      }
    }
    return 2 + extra_bits;
  }

  static Status CanEncode(const uint32_t distribution, const uint32_t value,
                          size_t* PIK_RESTRICT encoded_bits) {
    ValidateDistribution(distribution);
    int selector;
    size_t total_bits;
    const Status ok =
        ChooseEncoding(distribution, value, &selector, &total_bits);
    *encoded_bits = ok ? total_bits : 0;
    return ok;
  }

  static uint32_t Load(const uint32_t distribution,
                       BitReader* PIK_RESTRICT reader) {
    ValidateDistribution(distribution);
    if (IsRaw(distribution)) {
      return reader->ReadBits(RawBits(distribution));
    }
    const int selector = reader->ReadFixedBits<2>();
    const size_t b = Lookup(distribution, selector);
    if (b & kDirect) {
      return b & 0x7F;
    } else {
      uint32_t offset = GetOffset(b);
      uint32_t extra_bits = GetExtraBits(b);
      return reader->ReadBits(extra_bits) + offset;
    }
  }

  // Returns false if the value is too large to encode.
  static Status Store(const uint32_t distribution, const uint32_t value,
                      size_t* pos, uint8_t* storage) {
    int selector;
    size_t total_bits;
    PIK_RETURN_IF_ERROR(
        ChooseEncoding(distribution, value, &selector, &total_bits));

    if (IsRaw(distribution)) {
      WriteBits(RawBits(distribution), value, pos, storage);
      return true;
    }
    WriteBits(2, selector, pos, storage);

    const size_t b = Lookup(distribution, selector);
    if ((b & kDirect) == 0) {  // Nothing more to write for direct encoding
      uint32_t offset = GetOffset(b);
      PIK_ASSERT(value >= offset);
      WriteBits(total_bits - 2, value - offset, pos, storage);
    }

    return true;
  }

 private:
  static PIK_INLINE bool IsRaw(const uint32_t distribution) {
    return distribution > kU32RawBits;
  }

  static PIK_INLINE size_t RawBits(const uint32_t distribution) {
    PIK_ASSERT(IsRaw(distribution));
    return distribution - kU32RawBits;
  }

  // Returns one byte from "distribution" at index "selector".
  static PIK_INLINE size_t Lookup(const uint32_t distribution,
                                  const int selector) {
    PIK_ASSERT(!IsRaw(distribution));
    return (distribution >> (selector * 8)) & 0xFF;
  }

  static PIK_INLINE uint32_t GetOffset(const uint8_t b) {
    PIK_ASSERT(!(b & kDirect));
    if (b & kOffset) return ((b >> 3) & 7) + 1;
    return 0;
  }

  static PIK_INLINE uint32_t GetExtraBits(const uint8_t b) {
    PIK_ASSERT(!(b & kDirect));
    if (b & kOffset) return (b & 7) + 1;
    PIK_ASSERT(b != 0 && b <= 32);
    return b;
  }

  static void ValidateDistribution(const uint32_t distribution) {
#if PIK_ENABLE_ASSERT
    if (IsRaw(distribution)) return;  // raw 1..32: OK
    for (int selector = 0; selector < 4; ++selector) {
      const size_t b = Lookup(distribution, selector);
      if (b & kDirect) {
        continue;  // direct: OK
      } else if (b & kOffset) {
        continue;  // extra with offset: OK
      } else {
        // Forbid b = 0 because it requires an extra call to read/write 0 bits;
        // to encode a zero value, use b = kDirect instead.
        if (b == 0 || b > 32) {
          printf("Invalid distribution %8x[%d] == %zu\n", distribution,
                 selector, b);
          PIK_ASSERT(false);
        }
      }
    }
#endif
  }

  static Status ChooseEncoding(const uint32_t distribution,
                               const uint32_t value, int* PIK_RESTRICT selector,
                               size_t* PIK_RESTRICT total_bits) {
    const size_t bits_required = 32 - NumZeroBitsAboveMSB(value);
    PIK_ASSERT(bits_required <= 32);

    *selector = 0;
    *total_bits = 0;

    if (IsRaw(distribution)) {
      const size_t raw_bits = RawBits(distribution);
      if (bits_required > raw_bits) {
        return PIK_FAILURE("Insufficient raw bits");
      }
      *total_bits = raw_bits;
      return true;
    }

    // It is difficult to verify whether "distribution" is sorted, so check all
    // selectors and keep the one with the fewest total_bits.
    *total_bits = 64;  // more than any valid encoding
    for (int s = 0; s < 4; ++s) {
      const size_t b = Lookup(distribution, s);
      if (b & kDirect) {
        if ((b & 0x7F) == value) {
          *selector = s;
          *total_bits = 2;
          return true;  // Done, can't improve upon a direct encoding.
        }
        continue;
      }

      uint32_t extra_bits = GetExtraBits(b);
      if (b & kOffset) {
        uint32_t offset = GetOffset(b);
        if (value < offset || value >= offset + (1u << extra_bits)) continue;
      } else {
        if (bits_required > extra_bits) continue;
      }

      // Better than prior encoding, remember it:
      if (2 + extra_bits < *total_bits) {
        *selector = s;
        *total_bits = 2 + extra_bits;
      }
    }

    if (*total_bits == 64) return PIK_FAILURE("No feasible selector found");

    return true;
  }
};

// Coder for byte arrays: stores encoding and #bytes via U32Coder, then raw or
// Brotli-compressed bytes.
class BytesCoder {
  static constexpr uint32_t kDistSize = 0x20181008;
  static const int kBrotliQuality = 6;

 public:
  static Status CanEncode(Bytes encoding, const PaddedBytes& value,
                          size_t* PIK_RESTRICT encoded_bits) {
    PIK_ASSERT(encoding == Bytes::kRaw || encoding == Bytes::kBrotli);
    if (value.empty()) {
      return U32Coder::CanEncode(kU32Direct3Plus8, Bytes::kNone, encoded_bits);
    }

    PaddedBytes compressed;
    const PaddedBytes* store_what = &value;

    // Note: we will compress a second time when Store is called.
    if (encoding == Bytes::kBrotli) {
      PIK_RETURN_IF_ERROR(BrotliCompress(kBrotliQuality, value, &compressed));
      if (compressed.size() < value.size()) {
        store_what = &compressed;
      } else {
        encoding = Bytes::kRaw;
      }
    }

    size_t bits_encoding, bits_size;
    PIK_RETURN_IF_ERROR(
        U32Coder::CanEncode(kU32Direct3Plus8, encoding, &bits_encoding) &&
        U32Coder::CanEncode(kDistSize, store_what->size(), &bits_size));
    *encoded_bits = bits_encoding + bits_size + store_what->size() * 8;
    return true;
  }

  static Status Load(BitReader* PIK_RESTRICT reader,
                     PaddedBytes* PIK_RESTRICT value) {
    const Bytes encoding =
        static_cast<Bytes>(U32Coder::Load(kU32Direct3Plus8, reader));
    if (encoding == Bytes::kNone) {
      value->clear();
      return true;
    }
    if (encoding != Bytes::kRaw && encoding != Bytes::kBrotli) {
      return PIK_FAILURE("Unrecognized Bytes encoding");
    }

    const uint32_t num_bytes = U32Coder::Load(kDistSize, reader);
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

    if (encoding == Bytes::kBrotli) {
      const size_t kMaxOutput = 1ULL << 32;
      size_t bytes_read = 0;
      PaddedBytes decompressed;
      if (PIK_UNLIKELY(!BrotliDecompress(*value, kMaxOutput, &bytes_read,
                                         &decompressed))) {
        return false;
      }
      if (bytes_read != value->size()) {
        PIK_NOTIFY_ERROR("Read too few");
      }
      value->swap(decompressed);
    }
    return true;
  }

  static Status Store(Bytes encoding, const PaddedBytes& value,
                      size_t* PIK_RESTRICT pos, uint8_t* storage) {
    PIK_ASSERT(encoding == Bytes::kRaw || encoding == Bytes::kBrotli);
    if (value.empty()) {
      return U32Coder::Store(kU32Direct3Plus8, Bytes::kNone, pos, storage);
    }

    PaddedBytes compressed;
    const PaddedBytes* store_what = &value;

    if (encoding == Bytes::kBrotli) {
      PIK_RETURN_IF_ERROR(BrotliCompress(kBrotliQuality, value, &compressed));
      if (compressed.size() < value.size()) {
        store_what = &compressed;
      } else {
        encoding = Bytes::kRaw;
      }
    }

    PIK_RETURN_IF_ERROR(
        U32Coder::Store(kU32Direct3Plus8, encoding, pos, storage) &&
        U32Coder::Store(kDistSize, store_what->size(), pos, storage));

    size_t i = 0;
#if PIK_BYTE_ORDER_LITTLE
    // Write 4 bytes at a time
    uint32_t buf;
    for (; i + 4 <= store_what->size(); i += 4) {
      memcpy(&buf, store_what->data() + i, 4);
      WriteBits(32, buf, pos, storage);
    }
#endif

    // Write remaining bytes
    for (; i < store_what->size(); ++i) {
      WriteBits(8, store_what->data()[i], pos, storage);
    }
    return true;
  }
};

// Visitors for generating encoders/decoders for headers/sections with an
// associated non-member VisitFields function, which calls either U32 or
// Bytes. We do not overload operator() because U32 requires an extra parameter
// and a function name is easier to find by searching.

class ReadFieldsVisitor {
 public:
  ReadFieldsVisitor(BitReader* reader) : reader_(reader) {}

  void U32(const uint32_t distribution, uint32_t* PIK_RESTRICT value) {
    *value = U32Coder::Load(distribution, reader_);
  }

  template <typename T>
  void Enum(const uint32_t distribution, T* PIK_RESTRICT value) {
    uint32_t bits;
    U32(distribution, &bits);
    *value = static_cast<T>(bits);
  }

  void Bytes(const Bytes unused_encoding, PaddedBytes* PIK_RESTRICT value) {
    ok_ &= BytesCoder::Load(reader_, value);
  }

  template <typename T>
  void EnsureContainerSize(uint32_t size, T* container) {
    // Sets the container size to the given size in case of reading. The size
    // must have been read from a previously visited field.
    container->resize(size);
  }

  Status OK() const { return ok_; }

 private:
  bool ok_ = true;
  BitReader* const reader_;
};

class CanEncodeFieldsVisitor {
 public:
  void U32(const uint32_t distribution,
           const uint32_t* PIK_RESTRICT value) {
    size_t encoded_bits;
    ok_ &= U32Coder::CanEncode(distribution, *value, &encoded_bits);
    encoded_bits_ += encoded_bits;
  }

  template <typename T>
  void Enum(const uint32_t distribution, T* PIK_RESTRICT value) {
    uint32_t bits = static_cast<uint32_t>(*value);
    U32(distribution, &bits);
  }

  void Bytes(const Bytes encoding, const PaddedBytes* PIK_RESTRICT value) {
    size_t encoded_bits = 0;
    ok_ &= BytesCoder::CanEncode(encoding, *value, &encoded_bits);
    encoded_bits_ += encoded_bits;
  }

  template <typename T>
  void EnsureContainerSize(uint32_t size, const T* container) {
    // Nothing to do: This only has an effect for reading.
    PIK_ASSERT(container->size() == size);
  }

  Status OK() const { return ok_; }
  size_t EncodedBits() const { return encoded_bits_; }

 private:
  bool ok_ = true;
  size_t encoded_bits_ = 0;
};

class WriteFieldsVisitor {
 public:
  WriteFieldsVisitor(size_t* pos, uint8_t* storage)
      : pos_(pos), storage_(storage) {}

  void U32(const uint32_t distribution,
           const uint32_t* PIK_RESTRICT value) {
    ok_ &= U32Coder::Store(distribution, *value, pos_, storage_);
  }

  template <typename T>
  void Enum(const uint32_t distribution, T* PIK_RESTRICT value) {
    const uint32_t bits = static_cast<uint32_t>(*value);
    U32(distribution, &bits);
  }

  void Bytes(const Bytes encoding, const PaddedBytes* PIK_RESTRICT value) {
    ok_ &= BytesCoder::Store(encoding, *value, pos_, storage_);
  }

  template <typename T>
  void EnsureContainerSize(uint32_t size, const T* container) {
    // Nothing to do: This only has an effect for reading.
    PIK_ASSERT(container->size() == size);
  }

  Status OK() const { return ok_; }

 private:
  size_t* pos_;
  uint8_t* storage_;
  bool ok_ = true;
};

// T provides a non-const VisitFields (allows ReadFieldsVisitor to load fields),
// so we need to cast const T to non-const for visitors that don't actually need
// non-const (e.g. WriteFieldsVisitor).
template <class Visitor, class T>
Status VisitFieldsConst(Visitor* visitor, const T& t) {
  // Note: only called for Visitor that don't actually change T.
  return VisitFields(visitor, const_cast<T*>(&t));
}

template <class T>
Status CanEncodeFields(const T& t, size_t* PIK_RESTRICT encoded_bits) {
  CanEncodeFieldsVisitor visitor;
  PIK_RETURN_IF_ERROR(VisitFieldsConst(&visitor, t));
  const Status ok = visitor.OK();
  *encoded_bits = ok ? visitor.EncodedBits() : 0;
  return ok;
}

template <class T>
Status LoadFields(BitReader* reader, T* PIK_RESTRICT t) {
  ReadFieldsVisitor visitor(reader);
  PIK_RETURN_IF_ERROR(VisitFields(&visitor, t));
  return visitor.OK();
}

template <class T>
Status StoreFields(const T& t, size_t* pos, uint8_t* storage) {
  WriteFieldsVisitor visitor(pos, storage);
  PIK_RETURN_IF_ERROR(VisitFieldsConst(&visitor, t));
  return visitor.OK();
}

// A "section" is optional metadata that is only read from/written to the
// compressed stream if needed. Adding sections is the only way to extend
// this container format. Old code is forwards compatible because unknown
// sections can be skipped. Old files are backwards compatible because they
// indicate future sections are not present using a bit field in the header.
//
// To ensure interoperability, there will be no opaque fields nor proprietary
// section definitions. To introduce a new section and thereby advance the
// "version number" of this file format, add a unique_ptr member to Sections and
// append a call to "visitor" in VisitSections.

// Indicates which sections are present in the stream. This is required for
// backward compatibility (reading old files).
class SectionBits {
 public:
  static constexpr size_t kMaxSections = 32;

  void Set(const int idx) {
    PIK_CHECK(idx < kMaxSections);
    const uint32_t bit = 1U << idx;
    PIK_CHECK((bits_ & bit) == 0);  // not already set
    bits_ |= bit;
  }

  // Calls visitor(i), i = index of any non-zero (present) sections.
  template <class Visitor>
  void Foreach(const Visitor& visitor) const {
    uint32_t remaining = bits_;
    while (remaining != 0) {
      const int idx = NumZeroBitsBelowLSBNonzero(remaining);
      visitor(idx);
      remaining &= remaining - 1;  // clear lowest
    }
  }

  bool TestAndReset(const int idx) {
    PIK_CHECK(idx < kMaxSections);
    const uint32_t bit = 1U << idx;
    if ((bits_ & bit) == 0) {
      return false;
    }
    bits_ &= ~bit;  // clear lowest
    return true;
  }

  Status CanEncode(size_t* PIK_RESTRICT encoded_bits) const {
    return U32Coder::CanEncode(kDistribution, bits_, encoded_bits);
  }

  void Load(BitReader* reader) {
    bits_ = U32Coder::Load(kDistribution, reader);
  }

  Status Store(size_t* PIK_RESTRICT pos, uint8_t* storage) const {
    return U32Coder::Store(kDistribution, bits_, pos, storage);
  }

 private:
  static constexpr uint32_t kDistribution = 0x200C0480;

  uint32_t bits_ = 0;
};

// Stores size [bits] of all sections indicated by SectionBits. This is
// required for forward compatibility (skipping unknown sections).
template <size_t kNumKnown>
class SectionSizes {
 public:
  uint32_t Get(const int idx_section) const {
    PIK_CHECK(idx_section < SectionBits::kMaxSections);
    return sizes_[idx_section];
  }

  void Set(const int idx_section, const uint32_t size) {
    PIK_CHECK(idx_section < SectionBits::kMaxSections);
    sizes_[idx_section] = size;
  }

  Status CanEncode(const SectionBits& bits,
                   size_t* PIK_RESTRICT encoded_bits) const {
    bool ok = true;
    *encoded_bits = 0;
    bits.Foreach([this, &ok, encoded_bits](const int idx) {
      if (idx < kNumKnown) return;
      size_t size_bits;
      ok &= U32Coder::CanEncode(kDistribution, sizes_[idx], &size_bits);
      *encoded_bits += size_bits;
    });
    if (!ok) *encoded_bits = 0;
    return ok;
  }

  void Load(const SectionBits& bits, BitReader* reader) {
    std::fill(sizes_.begin(), sizes_.end(), 0);
    bits.Foreach([this, reader](const int idx) {
      if (idx < kNumKnown) return;
      sizes_[idx] = U32Coder::Load(kDistribution, reader);
    });
  }

  Status Store(const SectionBits& bits, size_t* pos, uint8_t* storage) const {
    bool ok = true;
    bits.Foreach([this, &ok, pos, storage](const int idx) {
      if (idx < kNumKnown) return;
      ok &= U32Coder::Store(kDistribution, sizes_[idx], pos, storage);
    });
    return ok;
  }

 private:
  // Up to 32 bit, EXIF limited to 64K bytes, palettes < 512 bytes.
  static constexpr uint32_t kDistribution = 0x20130F0C;

  std::array<uint32_t, SectionBits::kMaxSections> sizes_;  // [bits]
};

template <size_t kNumKnown>
class CanEncodeSectionVisitor {
 public:
  template <class T>
  void operator()(const std::unique_ptr<T>* PIK_RESTRICT ptr) {
    ++idx_section_;
    if (*ptr == nullptr) return;
    section_bits_.Set(idx_section_);

    CanEncodeFieldsVisitor field_visitor;
    ok_ &= VisitFields(&field_visitor, ptr->get());
    ok_ &= field_visitor.OK();
    const size_t encoded_bits = field_visitor.EncodedBits();
    section_sizes_.Set(idx_section_, encoded_bits);
    field_bits_ += encoded_bits;
  }

  Status CanEncode(size_t* PIK_RESTRICT encoded_bits) {
    size_t bits_bits, sizes_bits;
    ok_ &= section_bits_.CanEncode(&bits_bits);
    ok_ &= section_sizes_.CanEncode(section_bits_, &sizes_bits);
    *encoded_bits = ok_ ? bits_bits + sizes_bits + field_bits_ : 0;
    return ok_;
  }

  const SectionBits& GetBits() const { return section_bits_; }
  SectionSizes<kNumKnown>& GetSizes() { return section_sizes_; }

 private:
  int idx_section_ = -1;  // pre-incremented
  SectionBits section_bits_;
  SectionSizes<kNumKnown> section_sizes_;
  bool ok_ = true;
  size_t field_bits_ = 0;
};

// Analogous to VisitFieldsConst.
template <class Visitor, class T>
void VisitSectionsConst(Visitor* visitor, const T& t) {
  // Note: only called for Visitor that don't actually change T.
  VisitSections(visitor, const_cast<T*>(&t));
}

// Reads bits, sizes, and fields.
template <size_t kNumKnown>
class ReadSectionVisitor {
 public:
  explicit ReadSectionVisitor(BitReader* reader) : reader_(reader) {
    section_bits_.Load(reader_);
    section_sizes_.Load(section_bits_, reader_);
  }

  template <class T>
  void operator()(std::unique_ptr<T>* PIK_RESTRICT ptr) {
    ++idx_section_;
    if (!section_bits_.TestAndReset(idx_section_)) {
      return;
    }

    ptr->reset(new T);
    ReadFieldsVisitor field_reader(reader_);
    ok_ &= VisitFields(&field_reader, ptr->get());
  }

  void SkipUnknown() {
    // Not visited => unknown; skip past them in the bitstream.
    section_bits_.Foreach([this](const int idx) {
      printf("Warning: skipping unknown section %d (size %u)\n", idx,
             section_sizes_.Get(idx));
      reader_->SkipBits(section_sizes_.Get(idx));
    });
  }

  Status OK() const { return ok_; }

 private:
  int idx_section_ = -1;      // pre-incremented
  SectionBits section_bits_;  // Cleared after loading/skipping each section.
  SectionSizes<kNumKnown> section_sizes_;
  BitReader* PIK_RESTRICT reader_;  // not owned
  bool ok_ = true;
};

// Writes fields.
class WriteSectionVisitor {
 public:
  WriteSectionVisitor(size_t* pos, uint8_t* storage) : writer_(pos, storage) {}

  template <class T>
  void operator()(const std::unique_ptr<T>* PIK_RESTRICT ptr) {
    ++idx_section_;
    if (*ptr != nullptr) {
      ok_ &= VisitFields(&writer_, ptr->get());
    }
  }

  Status OK() const { return ok_ && writer_.OK(); }

 private:
  int idx_section_ = -1;  // pre-incremented
  WriteFieldsVisitor writer_;
  bool ok_ = true;
};

template <class T>
Status CanEncodeSectionsT(const T& t, size_t* PIK_RESTRICT encoded_bits) {
  CanEncodeSectionVisitor<T::kNumKnown> section_visitor;
  VisitSectionsConst(&section_visitor, t);
  return section_visitor.CanEncode(encoded_bits);
}

template <class T>
Status LoadSectionsT(BitReader* reader, T* PIK_RESTRICT t) {
  ReadSectionVisitor<T::kNumKnown> section_reader(reader);
  VisitSections(&section_reader, t);
  section_reader.SkipUnknown();
  return section_reader.OK() && reader->Healthy();
}

template <class T>
Status StoreSectionsT(const T& t, size_t* PIK_RESTRICT pos, uint8_t* storage) {
  CanEncodeSectionVisitor<T::kNumKnown> can_encode;
  VisitSectionsConst(&can_encode, t);
  const SectionBits& section_bits = can_encode.GetBits();
  PIK_RETURN_IF_ERROR(section_bits.Store(pos, storage));
  PIK_RETURN_IF_ERROR(can_encode.GetSizes().Store(section_bits, pos, storage));

  WriteSectionVisitor section_writer(pos, storage);
  VisitSectionsConst(&section_writer, t);
  return section_writer.OK();
}

}  // namespace pik

#endif  // FIELDS_H_
