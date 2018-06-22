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

#include "sections.h"

#include <algorithm>

#include "fields.h"

namespace pik {
namespace {

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
      if (idx < Sections::kNumKnown) return;
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
      if (idx < Sections::kNumKnown) return;
      sizes_[idx] = U32Coder::Load(kDistribution, reader);
    });
  }

  bool Store(const SectionBits& bits, size_t* pos, uint8_t* storage) const {
    bool ok = true;
    bits.Foreach([this, &ok, pos, storage](const int idx) {
      if (idx < Sections::kNumKnown) return;
      ok &= U32Coder::Store(kDistribution, sizes_[idx], pos, storage);
    });
    return ok;
  }

 private:
  // Up to 32 bit, EXIF limited to 64K bytes, palettes < 512 bytes.
  static constexpr uint32_t kDistribution = 0x20130F0C;

  std::array<uint32_t, SectionBits::kMaxSections> sizes_;  // [bits]
};

class CanEncodeSectionVisitor {
 public:
  template <class T>
  void operator()(const std::unique_ptr<T>* PIK_RESTRICT ptr) {
    ++idx_section_;
    if (*ptr == nullptr) return;
    section_bits_.Set(idx_section_);

    CanEncodeFieldsVisitor field_visitor;
    (*ptr)->VisitFields(&field_visitor);
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
  const SectionSizes& GetSizes() const { return section_sizes_; }

 private:
  int idx_section_ = -1;  // pre-incremented
  SectionBits section_bits_;
  SectionSizes section_sizes_;
  bool ok_ = true;
  size_t field_bits_ = 0;
};

// Analogous to VisitFieldsConst.
template <class T, class Visitor>
void VisitSectionsConst(const T& t, Visitor* visitor) {
  // Note: only called for Visitor that don't actually change T.
  const_cast<T*>(&t)->VisitSections(visitor);
}

// Reads bits, sizes, and fields.
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
    (*ptr)->VisitFields(&field_reader);
  }

  void SkipUnknown() {
    // Not visited => unknown; skip past them in the bitstream.
    section_bits_.Foreach([this](const int idx) {
      printf("Warning: skipping unknown section %d %u\n", idx,
             section_sizes_.Get(idx));
      reader_->SkipBits(section_sizes_.Get(idx));
    });
  }

 private:
  int idx_section_ = -1;      // pre-incremented
  SectionBits section_bits_;  // Cleared after loading/skipping each section.
  SectionSizes section_sizes_;
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
      (*ptr)->VisitFields(&writer_);
    }
  }

  bool OK() const { return writer_.OK(); }

 private:
  int idx_section_ = -1;  // pre-incremented
  WriteFieldsVisitor writer_;
};

}  // namespace

bool CanEncode(const Sections& sections, size_t* PIK_RESTRICT encoded_bits) {
  CanEncodeSectionVisitor section_visitor;
  VisitSectionsConst(sections, &section_visitor);
  return section_visitor.CanEncode(encoded_bits);
}

bool LoadSections(BitReader* reader, Sections* PIK_RESTRICT sections) {
  ReadSectionVisitor section_reader(reader);
  sections->VisitSections(&section_reader);
  section_reader.SkipUnknown();
  return true;
}

bool StoreSections(const Sections& sections, size_t* PIK_RESTRICT pos,
                   uint8_t* storage) {
  CanEncodeSectionVisitor can_encode;
  VisitSectionsConst(sections, &can_encode);
  const SectionBits& section_bits = can_encode.GetBits();
  if (!section_bits.Store(pos, storage)) return false;
  if (!can_encode.GetSizes().Store(section_bits, pos, storage)) return false;

  WriteSectionVisitor section_writer(pos, storage);
  VisitSectionsConst(sections, &section_writer);
  return section_writer.OK();
}

void TestUnsupportedSection() {
  for (const size_t extra_bits : {256, 513, 1023}) {
    std::unique_ptr<Palette> palette(new Palette);
    palette->num_colors_minus_one = 3;
    palette->num_alpha = 1;
    palette->colors.assign({10, 11, 12, 20, 21, 22, 30, 31, 32});
    palette->alpha.assign({0xFF});

    Sections sections;
    sections.palette = std::move(palette);

    // Get normal bits/sizes
    CanEncodeSectionVisitor can_encode;
    VisitSectionsConst(sections, &can_encode);
    SectionBits section_bits = can_encode.GetBits();
    SectionSizes section_sizes = can_encode.GetSizes();

    // Add a section unknown to the visitor (e.g. from future code versions)
    const int idx = 31;  // currently unused
    section_bits.Set(idx);
    section_sizes.Set(idx, extra_bits);

    // Write to bit stream
    size_t pos = 0;
    uint8_t storage[999] = {0};
    PIK_CHECK(section_bits.Store(&pos, storage));
    PIK_CHECK(section_sizes.Store(section_bits, &pos, storage));
    WriteSectionVisitor section_writer(&pos, storage);
    VisitSectionsConst(sections, &section_writer);
    PIK_CHECK(section_writer.OK());
    // .. including bogus bits for the imaginary section
    size_t i = 0;
    for (; i + 56 <= extra_bits; i += 56) {
      WriteBits(56, 0, &pos, storage);
    }
    for (; i < extra_bits; ++i) {
      WriteBits(1, 1, &pos, storage);
    }
    const size_t pos_after = pos;

    // Ensure LoadSections skips the unrecognized section
    WriteBits(20, 0xA55A, &pos, storage);  // sentinel
    BitReader reader(storage, sizeof(storage));
    Sections sections2;
    PIK_CHECK(LoadSections(&reader, &sections2));
    PIK_CHECK(reader.BitsRead() == pos_after);
    PIK_CHECK(reader.ReadBits(20) == 0xA55A);

    // As a sanity check, verify supported section fields match
    PIK_CHECK(sections2.palette->num_colors_minus_one == 3);
    PIK_CHECK(sections2.palette->colors.back() == 32);
    PIK_CHECK(sections2.palette->num_alpha == 1);
    PIK_CHECK(sections2.palette->alpha[0] == 0xFF);
  }
}

}  // namespace pik
