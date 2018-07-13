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
#include <vector>

#include "bit_reader.h"
#include "bits.h"
#include "compiler_specific.h"
#include "status.h"
#include "write_bits.h"

namespace pik {

// Chooses one of three encodings based on an a-priori "distribution":
// - raw: if IsRaw(distribution), send RawBits(distribution) = 1..32 raw bits;
// - direct: send a 2-bit "selector" to choose a 7-bit value from distribution;
// - extra: send a 2-bit selector to choose the number of extra bits (1..32)
//   from distribution, then send that many bits.
// This is faster to decode and denser than Exp-Golomb or Gamma codes when both
// small and large values occur.
//
// If !IsRaw(distribution), it is interpreted as an array of four bytes
// (least-significant first). Let b = Lookup(distribution, selector).
// If b & 0x80, then the value is b & 0x7F; otherwise, b indicates the number of
// extra bits (1 to 32) encoding the value.
//
// Examples:
// Raw:    distribution 0xFFFFFFEF, value 32768 => 1000000000000000
// Direct: distribution 0x06A09088, value 32 => 10 (selector 2, b=0xA0).
// Extra:  distribution 0x08060402, value 7 => 01 0111 (selector 1, b=4).
class U32Coder {
 public:
  static size_t MaxEncodedBits(const uint32_t distribution) {
    ValidateDistribution(distribution);
    if (IsRaw(distribution)) return RawBits(distribution);
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
    size_t total_bits;
    const bool ok = ChooseEncoding(distribution, value, &selector, &total_bits);
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
    if (b & 0x80) return b & 0x7F;  // direct
    return reader->ReadBits(b);     // extra
  }

  // Returns false if the value is too large to encode.
  static bool Store(const uint32_t distribution, const uint32_t value,
                    size_t* pos, uint8_t* storage) {
    int selector;
    size_t total_bits;
    if (!ChooseEncoding(distribution, value, &selector, &total_bits)) {
      return false;
    }

    if (IsRaw(distribution)) {
      WriteBits(RawBits(distribution), value, pos, storage);
      return true;
    }

    WriteBits(2, selector, pos, storage);
    if (total_bits > 2) {
      WriteBits(total_bits - 2, value, pos, storage);
    }
    return true;
  }

 private:
  // Note: to avoid dependencies on this header, we allow users to hard-code
  // ~32u. This value is convenient because the maximum number of raw bits is 32
  // and ~32u + 32 == ~0u, which ensures RawBits can never exceed 32 and also
  // allows "distribution" to be sign-extended from an 8-bit immediate.
  static PIK_INLINE bool IsRaw(const uint32_t distribution) {
    return distribution > ~32u;
  }

  static PIK_INLINE size_t RawBits(const uint32_t distribution) {
    PIK_ASSERT(IsRaw(distribution));
    return distribution - ~32u;
  }

  // Returns one byte from "distribution" at index "selector".
  static PIK_INLINE size_t Lookup(const uint32_t distribution,
                                  const int selector) {
    PIK_ASSERT(!IsRaw(distribution));
    return (distribution >> (selector * 8)) & 0xFF;
  }

  static void ValidateDistribution(const uint32_t distribution) {
#if defined(PIK_ENABLE_ASSERT)
    if (IsRaw(distribution)) return;  // raw 1..32: OK
    for (int selector = 0; selector < 4; ++selector) {
      const size_t b = Lookup(distribution, selector);
      if (b & 0x80) continue;  // direct: OK
      // Forbid b = 0 because it requires an extra call to read/write 0 bits;
      // to encode a zero value, use b = 0x80 instead.
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
      if (b & 0x80) {
        if ((b & 0x7F) == value) {
          *selector = s;
          *total_bits = 2;
          return true;  // Done, can't improve upon a direct encoding.
        }
        continue;
      }  // Else not direct, try extra:

      if (bits_required > b) continue;

      // Better than prior encoding, remember it:
      if (2 + b < *total_bits) {
        *selector = s;
        *total_bits = 2 + b;
      }
    }

    if (*total_bits == 64) return PIK_FAILURE("No feasible selector found");

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

  void Bytes(std::vector<uint8_t>* PIK_RESTRICT value) {
    BytesCoder::Load(reader_, value);
  }

 private:
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

  void Bytes(const std::vector<uint8_t>* PIK_RESTRICT value) {
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

  void U32(const uint32_t distribution,
                  const uint32_t* PIK_RESTRICT value) {
    ok_ &= U32Coder::Store(distribution, *value, pos_, storage_);
  }

  void Bytes(const std::vector<uint8_t>* PIK_RESTRICT value) {
    ok_ &= BytesCoder::Store(*value, pos_, storage_);
  }

  bool OK() const { return ok_; }

 private:
  size_t* pos_;
  uint8_t* storage_;
  bool ok_ = true;
};

// T provides a non-const VisitFields (allows ReadFieldsVisitor to load fields),
// so we need to cast const T to non-const for visitors that don't actually need
// non-const (e.g. WriteFieldsVisitor).
template <class Visitor, class T>
void VisitFieldsConst(Visitor* visitor, const T& t) {
  // Note: only called for Visitor that don't actually change T.
  VisitFields(visitor, const_cast<T*>(&t));
}

template <class T>
bool CanEncodeFields(const T& t, size_t* PIK_RESTRICT encoded_bits) {
  CanEncodeFieldsVisitor visitor;
  VisitFieldsConst(&visitor, t);
  const bool ok = visitor.OK();
  *encoded_bits = ok ? visitor.EncodedBits() : 0;
  return ok;
}

template <class T>
void LoadFields(BitReader* reader, T* PIK_RESTRICT t) {
  ReadFieldsVisitor visitor(reader);
  VisitFields(&visitor, t);
}

template <class T>
bool StoreFields(const T& t, size_t* pos, uint8_t* storage) {
  WriteFieldsVisitor visitor(pos, storage);
  VisitFieldsConst(&visitor, t);
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

  bool CanEncode(size_t* PIK_RESTRICT encoded_bits) const {
    return U32Coder::CanEncode(kDistribution, bits_, encoded_bits);
  }

  void Load(BitReader* reader) {
    bits_ = U32Coder::Load(kDistribution, reader);
  }

  bool Store(size_t* PIK_RESTRICT pos, uint8_t* storage) const {
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

  bool CanEncode(const SectionBits& bits,
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
  };

  void Load(const SectionBits& bits, BitReader* reader) {
    std::fill(sizes_.begin(), sizes_.end(), 0);
    bits.Foreach([this, reader](const int idx) {
      if (idx < kNumKnown) return;
      sizes_[idx] = U32Coder::Load(kDistribution, reader);
    });
  }

  bool Store(const SectionBits& bits, size_t* pos, uint8_t* storage) const {
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
    VisitFields(&field_visitor, ptr->get());
    ok_ &= field_visitor.OK();
    const size_t encoded_bits = field_visitor.EncodedBits();
    section_sizes_.Set(idx_section_, encoded_bits);
    field_bits_ += encoded_bits;
  }

  bool CanEncode(size_t* PIK_RESTRICT encoded_bits) {
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
    VisitFields(&field_reader, ptr->get());
  }

  void SkipUnknown() {
    // Not visited => unknown; skip past them in the bitstream.
    section_bits_.Foreach([this](const int idx) {
      printf("Warning: skipping unknown section %d (size %u)\n", idx,
             section_sizes_.Get(idx));
      reader_->SkipBits(section_sizes_.Get(idx));
    });
  }

 private:
  int idx_section_ = -1;      // pre-incremented
  SectionBits section_bits_;  // Cleared after loading/skipping each section.
  SectionSizes<kNumKnown> section_sizes_;
  BitReader* PIK_RESTRICT reader_;  // not owned
};

// Writes fields.
class WriteSectionVisitor {
 public:
  WriteSectionVisitor(size_t* pos, uint8_t* storage) : writer_(pos, storage) {}

  template <class T>
  void operator()(const std::unique_ptr<T>* PIK_RESTRICT ptr) {
    ++idx_section_;
    if (*ptr != nullptr) {
      VisitFields(&writer_, ptr->get());
    }
  }

  bool OK() const { return writer_.OK(); }

 private:
  int idx_section_ = -1;  // pre-incremented
  WriteFieldsVisitor writer_;
};

template <class T>
bool CanEncodeSectionsT(const T& t, size_t* PIK_RESTRICT encoded_bits) {
  CanEncodeSectionVisitor<T::kNumKnown> section_visitor;
  VisitSectionsConst(&section_visitor, t);
  return section_visitor.CanEncode(encoded_bits);
}

template <class T>
bool LoadSectionsT(BitReader* reader, T* PIK_RESTRICT t) {
  ReadSectionVisitor<T::kNumKnown> section_reader(reader);
  VisitSections(&section_reader, t);
  section_reader.SkipUnknown();
  return true;
}

template <class T>
bool StoreSectionsT(const T& t, size_t* PIK_RESTRICT pos, uint8_t* storage) {
  CanEncodeSectionVisitor<T::kNumKnown> can_encode;
  VisitSectionsConst(&can_encode, t);
  const SectionBits& section_bits = can_encode.GetBits();
  if (!section_bits.Store(pos, storage)) return false;
  if (!can_encode.GetSizes().Store(section_bits, pos, storage)) return false;

  WriteSectionVisitor section_writer(pos, storage);
  VisitSectionsConst(&section_writer, t);
  return section_writer.OK();
}

}  // namespace pik

#endif  // FIELDS_H_
