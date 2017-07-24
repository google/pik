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

#include "header.h"

#include <stdint.h>
#include <string.h>
#include <algorithm>
#include <array>

#include "arch_specific.h"
#include "bits.h"
#include "compiler_specific.h"

namespace pik {

// Coder for 32-bit integers. Encodes the number of value bits using 2-bit
// selectors. This is faster to decode and denser than Exp-Golomb or Gamma
// codes when the approximate distribution is known, and both small and
// large magnitudes occur.
class U32Coder {
 public:
  static constexpr size_t MaxCompressedBits() {
    return 2 + sizeof(uint32_t) * 8;
  }

  // The four bytes of "selector_bits" (sorted in ascending order from
  // low to high bit indices) indicate how many bits for each selector, or
  // zero if the selector represents the value directly.
  static uint32_t Load(const uint32_t selector_bits,
                       BitSource* const PIK_RESTRICT source) {
    if (source->CanRead32()) {
      source->Read32();
    }
    const int selector = source->Extract<2>();
    const int num_bits = NumBits(selector_bits, selector);
    if (num_bits == 0) {
      return selector;
    }
    if (source->CanRead32()) {
      source->Read32();
    }
    return source->ExtractVariableCount(num_bits);
  }

  // Returns false if the value is too large to encode.
  static bool Store(const uint32_t value, const uint32_t selector_bits,
                    BitSink* const PIK_RESTRICT sink) {
    // Encode i < 4 using selector i if NumBits(i) == 0.
    if (value < 4) {
      if (NumBits(selector_bits, value) == 0) {
        if (sink->CanWrite32()) {
          sink->Write32();
        }
        sink->Insert<2>(value);
        return true;
      }
    }

    const int bits_required = 32 - PIK_LZCNT32(value);
    // kBits (except the zeros) are sorted in ascending order,
    // so choose the first selector that provides enough bits.
    for (int selector = 0; selector < 4; ++selector) {
      const int num_bits = NumBits(selector_bits, selector);
      // Only the prior special case can make use of bits = zero. We must
      // prevent value = 0 from using selector i > 0, which will decode to i.
      if (num_bits == 0 || num_bits < bits_required) {
        continue;
      }

      // Found selector; use it.
      if (sink->CanWrite32()) {
        sink->Write32();
      }
      sink->Insert<2>(selector);
      if (sink->CanWrite32()) {
        sink->Write32();
      }
      sink->InsertVariableCount(num_bits, value);
      return true;
    }

    PIK_NOTIFY_ERROR("No feasible selector found");
    return false;
  }

 private:
  static int NumBits(const uint32_t selector_bits, const int selector) {
    return (selector_bits >> (selector * 8)) & 0xFF;
  }
};

// Reads fields (uint32_t or Bytes) from a BitSource.
class FieldReader {
 public:
  FieldReader(BitSource* source) : source_(source) {}
  void operator()(const uint32_t selector_bits,
                  uint32_t* const PIK_RESTRICT value) {
    *value = U32Coder::Load(selector_bits, source_);
  }

  void operator()(Bytes* const PIK_RESTRICT value) {
    const uint32_t num_bytes = U32Coder::Load(kU32Selectors, source_);
    value->resize(num_bytes);

    // Read four bytes at a time, reducing the relative cost of CanRead32.
    uint32_t i;
    for (i = 0; i + 4 <= value->size(); i += 4) {
      if (source_->CanRead32()) {
        source_->Read32();
      }
      for (int idx_byte = 0; idx_byte < 4; ++idx_byte) {
        value->data()[i + idx_byte] = source_->Extract<8>();
      }
    }

    if (source_->CanRead32()) {
      source_->Read32();
    }
    for (; i < value->size(); ++i) {
      value->data()[i] = source_->Extract<8>();
    }
  }

 private:
  BitSource* const source_;
};

// Establishes an upper bound on the compressed size, using the actual
// size of any Bytes fields.
class FieldMaxSize {
 public:
  void operator()(const uint32_t selector_bits,
                  const uint32_t* const PIK_RESTRICT value) {
    max_bits_ += U32Coder::MaxCompressedBits();
  }

  void operator()(const Bytes* const PIK_RESTRICT value) {
    max_bits_ += U32Coder::MaxCompressedBits();  // num_bytes
    max_bits_ += value->size() * 8;
  }

  size_t MaxBytes() const { return (max_bits_ + 7) / 8; }

 private:
  size_t max_bits_ = 0;
};

// Writes fields to a BitSink.
class FieldWriter {
 public:
  FieldWriter(BitSink* sink) : sink_(sink) {}

  void operator()(const uint32_t selector_bits,
                  const uint32_t* const PIK_RESTRICT value) {
    ok_ &= U32Coder::Store(*value, selector_bits, sink_);
  }

  void operator()(const Bytes* const PIK_RESTRICT value) {
    ok_ &= U32Coder::Store(value->size(), kU32Selectors, sink_);

    // Write four bytes at a time, reducing the relative cost of CanWrite32.
    uint32_t i;
    for (i = 0; i + 4 <= value->size(); i += 4) {
      if (sink_->CanWrite32()) {
        sink_->Write32();
      }
      for (int idx_byte = 0; idx_byte < 4; ++idx_byte) {
        sink_->Insert<8>(value->data()[i + idx_byte]);
      }
    }

    if (sink_->CanWrite32()) {
      sink_->Write32();
    }
    for (; i < value->size(); ++i) {
      sink_->Insert<8>(value->data()[i]);
    }
  }

  bool OK() const { return ok_; }

 private:
  BitSink* const sink_;
  bool ok_ = true;
};

// Stores and verifies the 4-byte file signature.
class Magic {
 public:
  static constexpr size_t CompressedSize() { return 4; }

  static bool Verify(BitSource* const PIK_RESTRICT source) {
    if (source->CanRead32()) {
      source->Read32();
    }
    for (int i = 0; i < 4; ++i) {
      if (source->Extract<8>() != String()[i]) {
        return false;
      }
    }
    return true;
  }

  static void Store(BitSink* const PIK_RESTRICT sink) {
    if (sink->CanWrite32()) {
      sink->Write32();
    }
    for (int i = 0; i < 4; ++i) {
      sink->Insert<8>(String()[i]);
    }
  }

 private:
  static const unsigned char* String() {
    // \n causes files opened in text mode to be rejected, and \xCC detects
    // 7-bit transfers (it is also an uppercase I with accent in ISO-8859-1).
    return reinterpret_cast<const unsigned char*>("P\xCCK\n");
  }
};

size_t MaxCompressedHeaderSize() {
  size_t size = Magic::CompressedSize();

  // Fields
  FieldMaxSize max_size;
  Header header;
  header.VisitFields(&max_size);
  size += max_size.MaxBytes();
  return size + 8;  // plus BitSink safety margin.
}

Status LoadHeader(BitSource* const PIK_RESTRICT source,
                  Header* const PIK_RESTRICT header) {
  if (!Magic::Verify(source)) {
    return Status::WRONG_MAGIC;
  }

  FieldReader reader(source);
  header->VisitFields(&reader);
  return Status::OK;
}

Status StoreHeader(const Header& header_const,
                   BitSink* const PIK_RESTRICT sink) {
  // VisitFields requires a non-const pointer, but we do not actually
  // modify the underlying memory.
  Header* const PIK_RESTRICT header = const_cast<Header*>(&header_const);

  Magic::Store(sink);

  FieldWriter writer(sink);
  header->VisitFields(&writer);
  return writer.OK() ? Status::OK : Status::RANGE_EXCEEEDED;
}

// Indicates which sections are present in the stream. This is required for
// backward compatibility (reading old files).
class SectionBits {
 public:
  static constexpr size_t kMaxSections = 32;

  void Set(const int idx) {
    PIK_CHECK(idx < kMaxSections);
    bits_ |= 1U << idx;
  }

  template <class Visitor>
  void Foreach(const Visitor& visitor) const {
    uint32_t remaining = bits_;
    while (remaining != 0) {
      const int idx = PIK_TZCNT32(remaining);
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

  void Load(BitSource* source) { bits_ = U32Coder::Load(kSelectors, source); }
  bool Store(BitSink* sink) const {
    return U32Coder::Store(bits_, kSelectors, sink);
  }

 private:
  static constexpr uint32_t kSelectors = kU32Selectors;
  uint32_t bits_ = 0;
};

// Stores size [bytes] of all sections indicated by SectionBits. This is
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

  void Load(const SectionBits& bits, BitSource* source) {
    std::fill(sizes_.begin(), sizes_.end(), 0);
    bits.Foreach([this, source](const int idx) {
      sizes_[idx] = U32Coder::Load(kSelectors, source);
    });
  }

  bool Store(const SectionBits& bits, BitSink* sink) const {
    bool ok = true;
    bits.Foreach([this, &ok, sink](const int idx) {
      ok &= U32Coder::Store(sizes_[idx], kSelectors, sink);
    });
    return ok;
  }

 private:
  static constexpr uint32_t kSelectors = kU32Selectors;

  std::array<uint32_t, SectionBits::kMaxSections> sizes_;  // [bytes]
};

// Reads sections (and bits/sizes) from BitSource.
class SectionReader {
 public:
  explicit SectionReader(BitSource* const PIK_RESTRICT source) {
    section_bits_.Load(source);
    section_sizes_.Load(section_bits_, source);

    // Sections are byte-aligned to simplify skipping over them.
    section_bytes_ = source->Finalize();
  }

  template <class T>
  void operator()(std::unique_ptr<T>* const PIK_RESTRICT ptr) {
    ++idx_section_;
    if (!section_bits_.TestAndReset(idx_section_)) {
      return;
    }

    // Load from byte-aligned storage.
    BitSource source(section_bytes_);
    section_bytes_ += section_sizes_.Get(idx_section_);
    ptr->reset(new T);
    FieldReader reader(&source);
    (*ptr)->VisitFields(&reader);
    const uint8_t* const PIK_RESTRICT end = source.Finalize();
    PIK_CHECK(section_bytes_ == end);
  }

  // Returns end pointer.
  const uint8_t* const PIK_RESTRICT Finalize() {
    // Skip any remaining/unknown sections.
    section_bits_.Foreach(
        [this](const int idx) { section_bytes_ += section_sizes_.Get(idx); });
    return section_bytes_;
  }

 private:
  // A bit is cleared after loading/skipping that section.
  SectionBits section_bits_;
  SectionSizes section_sizes_;
  int idx_section_ = -1;
  // Points to the start of the current section.
  const uint8_t* PIK_RESTRICT section_bytes_;
};

class SectionMaxSize {
 public:
  template <class T>
  void operator()(const std::unique_ptr<T>* const PIK_RESTRICT ptr) {
    ++idx_section_;
    PIK_CHECK(idx_section_ < SectionBits::kMaxSections);

    if (*ptr != nullptr) {
      FieldMaxSize max_size;
      (*ptr)->VisitFields(&max_size);
      max_bytes_ += max_size.MaxBytes();
    }
  }

  size_t MaxBytes() const { return max_bytes_; }

 private:
  int idx_section_ = -1;
  size_t max_bytes_ = 0;
};

// Writes sections (and bits/sizes) to BitSink.
class SectionWriter {
 public:
  SectionWriter(const size_t max_bytes)
      : section_storage_(max_bytes), section_bytes_(&section_storage_[0]) {}

  template <class T>
  void operator()(const std::unique_ptr<T>* const PIK_RESTRICT ptr) {
    ++idx_section_;

    if (!ok_ || *ptr == nullptr) {
      return;
    }
    section_bits_.Set(idx_section_);

    BitSink sink(section_bytes_);
    FieldWriter writer(&sink);
    (*ptr)->VisitFields(&writer);
    ok_ &= writer.OK();
    uint8_t* const PIK_RESTRICT end = sink.Finalize();
    section_sizes_.Set(idx_section_, end - section_bytes_);
    section_bytes_ = end;
  }

  // Returns end pointer or nullptr on failure.
  uint8_t* const PIK_RESTRICT Finalize(BitSink* const PIK_RESTRICT sink) {
    ok_ &= section_bits_.Store(sink);
    ok_ &= section_sizes_.Store(section_bits_, sink);
    if (!ok_) {
      return nullptr;
    }
    uint8_t* const PIK_RESTRICT end = sink->Finalize();

    // Append section_bytes_ to end.
    const size_t total_bytes = section_bytes_ - &section_storage_[0];
    PIK_CHECK(total_bytes <= section_storage_.size());
    memcpy(end, &section_storage_[0], total_bytes);
    return end + total_bytes;
  }

 private:
  Bytes section_storage_;
  // Points to the start of the current section.
  uint8_t* PIK_RESTRICT section_bytes_;
  int idx_section_ = -1;
  SectionBits section_bits_;
  SectionSizes section_sizes_;
  bool ok_ = true;
};

size_t MaxCompressedSectionsSize(const Sections& sections_const) {
  // VisitSections requires a non-const pointer, but we do not actually
  // modify the underlying memory.
  Sections* const PIK_RESTRICT sections =
      const_cast<Sections*>(&sections_const);
  SectionMaxSize max_size;
  sections->VisitSections(&max_size);
  return max_size.MaxBytes() + 8;  // plus BitSink safety margin.
}

const uint8_t* const PIK_RESTRICT
LoadSections(BitSource* const PIK_RESTRICT source,
             Sections* const PIK_RESTRICT sections) {
  SectionReader reader(source);
  sections->VisitSections(&reader);
  return reader.Finalize();
}

uint8_t* const PIK_RESTRICT StoreSections(const Sections& sections_const,
                                          const size_t max_bytes,
                                          BitSink* const PIK_RESTRICT sink) {
  // VisitSections requires a non-const pointer, but we do not actually
  // modify the underlying memory.
  Sections* const PIK_RESTRICT sections =
      const_cast<Sections*>(&sections_const);
  SectionWriter section_writer(max_bytes);
  sections->VisitSections(&section_writer);
  return section_writer.Finalize(sink);
}

}  // namespace pik
