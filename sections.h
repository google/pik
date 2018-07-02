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

#ifndef SECTIONS_H_
#define SECTIONS_H_

// Optional metadata with backward and forward compatibility.

#include <stddef.h>
#include <stdint.h>
#include <memory>
#include <vector>

#include "bit_reader.h"
#include "compiler_specific.h"

namespace pik {

#pragma pack(push, 1)

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

// Alpha channel (lossless compression).
struct Alpha {
  enum { kModeBrotli, kModeTransform, kModeInvalid };

  uint32_t mode = kModeBrotli;
  uint32_t bytes_per_alpha = 1;
  std::vector<uint8_t> encoded;  // interpretation depends on mode
};

template <class Visitor>
void VisitFields(Visitor* PIK_RESTRICT visitor, Alpha* PIK_RESTRICT alpha) {
  (*visitor)(0x04828180, &alpha->mode);
  (*visitor)(0x84828180, &alpha->bytes_per_alpha);
  (*visitor)(&alpha->encoded);
}

// Superset of PNG PLTE+tRNS.
struct Palette {
  enum { kEncodingRaw = 0 };

  // Whether color/alpha are compressed.
  uint32_t encoding = kEncodingRaw;

  // 1 or 2 (little-endian) byte per entry AND channel (RGB).
  uint32_t bytes_per_color = 1;

  // E.g. "255" instead of 256 to save one bit.
  uint32_t num_colors_minus_one = 0;

  // How many alpha entries (often 0 or 1, hence separate from colors).
  uint32_t num_alpha = 0;

  // "num_colors_minus_one+1" times 8/16-bit {R, G, B}; will not be reordered.
  std::vector<uint8_t> colors;

  // "num_alpha" times 8/16-bit opacity; will not be reordered.
  std::vector<uint8_t> alpha;
};

template <class Visitor>
void VisitFields(Visitor* PIK_RESTRICT visitor, Palette* PIK_RESTRICT palette) {
  (*visitor)(0x04828180, &palette->encoding);
  (*visitor)(0x84828180, &palette->bytes_per_color);
  (*visitor)(0x0C0A0806, &palette->num_colors_minus_one);
  (*visitor)(0x0C088180, &palette->num_alpha);
  (*visitor)(&palette->colors);
  (*visitor)(&palette->alpha);
}

struct ICC {
  std::vector<uint8_t> profile;
};

template <class Visitor>
void VisitFields(Visitor* PIK_RESTRICT visitor, ICC* PIK_RESTRICT icc) {
  (*visitor)(&icc->profile);
}

struct EXIF {
  std::vector<uint8_t> metadata;
};

template <class Visitor>
void VisitFields(Visitor* PIK_RESTRICT visitor, EXIF* PIK_RESTRICT exif) {
  (*visitor)(&exif->metadata);
}

struct XMP {
  std::vector<uint8_t> metadata;
};

template <class Visitor>
void VisitFields(Visitor* PIK_RESTRICT visitor, XMP* PIK_RESTRICT xmp) {
  (*visitor)(&xmp->metadata);
}

struct Sections {
  // Number of known sections at the time the bitstream was frozen. No need to
  // encode size if idx_section < kNumKnown because the decoder already knows
  // how to to read them. Do not change this after freezing!
  static constexpr size_t kNumKnown = 5;

  // Valid/present if non-null.
  std::unique_ptr<Alpha> alpha;
  std::unique_ptr<Palette> palette;
  std::unique_ptr<ICC> icc;
  std::unique_ptr<EXIF> exif;
  std::unique_ptr<XMP> xmp;
  // Add new section member before this comment.
};

template <class Visitor>
void VisitSections(Visitor* PIK_RESTRICT visitor,
                   Sections* PIK_RESTRICT sections) {
  // Sections are read/written in this order, so do not rearrange.
  (*visitor)(&sections->alpha);
  (*visitor)(&sections->palette);
  (*visitor)(&sections->icc);
  (*visitor)(&sections->exif);
  (*visitor)(&sections->xmp);
  // Add new section visitor before this comment.
}

#pragma pack(pop)

// Returns whether "sections" can be encoded (i.e. all fields have a valid
// representation). If so, "*encoded_bits" is the exact number of bits required.
bool CanEncode(const Sections& sections, size_t* PIK_RESTRICT encoded_bits);

bool LoadSections(BitReader* source, Sections* PIK_RESTRICT sections);

bool StoreSections(const Sections& sections, size_t* PIK_RESTRICT pos,
                   uint8_t* storage);

// For use by test - requires access to internal data structures.
void TestUnsupportedSection();

}  // namespace pik

#endif  // SECTIONS_H_
