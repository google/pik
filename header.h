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

#ifndef HEADER_H_
#define HEADER_H_

// Per-image header with backward and forward-compatible extension capability
// and compressed integer fields.

#include <stddef.h>
#include <stdint.h>
#include <memory>
#include <vector>

#include "bit_reader.h"
#include "compiler_specific.h"

namespace pik {

#pragma pack(push, 1)

// Header preceding all (sub-)images.
struct Header {
  enum ImageType {
    kImageTypePreview = 0,
    kImageTypePyramid = 1,
    kImageTypeFrame = 2,  // main image or animation frame
    // Used to terminate container of unbounded stream of Images.
    // All other Header fields are default-initialized.
    kImageTypeSentinel = 3
  };

  // What comes after the header + sections.
  enum Bitstream {
    // Subsequent header fields are (only) valid in this mode.
    kBitstreamDefault = 0,

    // Special bitstream for JPEG input images
    kBitstreamBrunsli = 1,
  };

  // Optional postprocessing steps.
  enum Flags {
    kSmoothDCPred = 1,

    // Decoder should apply edge-preserving filter in opsin space. Useful for
    // deblocking/deringing in lower-quality settings.
    kDenoise = 2,

    // Additional smoothing; helpful for medium/low-quality.
    kGaborishTransform = 4,

    // Dither opsin values before rounding to 8-bit SRGB. Generally beneficial
    // except for high-quality bitstreams (<= distance 1).
    kDither = 8,

    // Gradient map used to predict smooth areas.
    kGradientMap = 16,

    // Experimental, go/pik-block-strategy
    kBlockStrategy = 32,
  };

  enum Expand { kExpandNone = 8, kExpandLinear = 9, kExpandPower = 10 };

  // If kImageTypeSentinel, all other header fields are default-initialized.
  uint32_t image_type = kImageTypeFrame;

  uint32_t xsize = 0;  // pixels, not necessarily a multiple of kBlockWidth
  uint32_t ysize = 0;
  uint32_t num_components = 0;

  // If != kBitstreamDefault, all subsequent fields are default-initialized.
  uint32_t bitstream = kBitstreamDefault;

  uint32_t flags = 0;

  uint32_t original_bit_depth = 8;
  uint32_t expand = kExpandNone;  // if original_bit_depth != 8
  uint32_t expand_param = 0;      // if expand != kExpandNone

  uint32_t quant_template = 0;
};

// For loading/storing fields from/to the compressed stream.
template <class Visitor>
void VisitFields(Visitor* PIK_RESTRICT visitor, Header* PIK_RESTRICT header) {
  visitor->U32(0x83828180, &header->image_type);
  if (header->image_type == Header::kImageTypeSentinel) return;

  // Almost all camera images are less than 8K * 8K. We also allow the
  // full 32-bit range for completeness.
  visitor->U32(0x200D0B09, &header->xsize);
  visitor->U32(0x200D0B09, &header->ysize);
  // 2-bit encodings for 1 and 3 common cases; a dozen components for
  // remote-sensing data, or thousands for hyperspectral images.
  visitor->U32(0x10048381, &header->num_components);

  visitor->U32(0x06828180, &header->bitstream);
  if (header->bitstream != Header::kBitstreamDefault) return;

  visitor->U32(0x20181008, &header->flags);

  visitor->U32(0x058E8C88, &header->original_bit_depth);
  if (header->original_bit_depth != 8) {
    visitor->U32(0x038A8988, &header->expand);
    if (header->expand != Header::kExpandNone) {
      visitor->U32(~32u + 32, &header->expand_param);
    }
  }

  // Direct 2-bit encoding for quant template ids 0, 1, 2, 3.
  visitor->U32(0x83828180, &header->quant_template);

  // To extend: add a section, or add fields conditional on a NEW flag:
  // if (flag) visitor->U32(..).
}

// Returns whether the struct can be encoded (i.e. all fields have a valid
// representation). If so, "*encoded_bits" is the exact number of bits required.
bool CanEncode(const Header& header, size_t* PIK_RESTRICT encoded_bits);

bool LoadHeader(BitReader* reader, Header* PIK_RESTRICT header);

bool StoreHeader(const Header& header, size_t* pos, uint8_t* storage);

// Optional per-image extensions (backward- and forward-compatible):

// Alpha channel (lossless compression).
struct Alpha {
  enum { kModeBrotli, kModeTransform, kModeInvalid };

  uint32_t mode = kModeBrotli;
  uint32_t bytes_per_alpha = 1;
  std::vector<uint8_t> encoded;  // interpretation depends on mode
};

template <class Visitor>
void VisitFields(Visitor* PIK_RESTRICT visitor, Alpha* PIK_RESTRICT alpha) {
  visitor->U32(0x04828180, &alpha->mode);
  visitor->U32(0x84828180, &alpha->bytes_per_alpha);
  visitor->Bytes(&alpha->encoded);
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
  visitor->U32(0x04828180, &palette->encoding);
  visitor->U32(0x84828180, &palette->bytes_per_color);
  visitor->U32(0x0C0A0806, &palette->num_colors_minus_one);
  visitor->U32(0x0C088180, &palette->num_alpha);
  visitor->Bytes(&palette->colors);
  visitor->Bytes(&palette->alpha);
}

struct Sections {
  // Number of known sections at the time the bitstream was frozen. No need to
  // encode size if idx_section < kNumKnown because the decoder already knows
  // how to to read them. Do not change this after freezing!
  static constexpr size_t kNumKnown = 2;

  // Valid/present if non-null.
  std::unique_ptr<Alpha> alpha;
  std::unique_ptr<Palette> palette;
  // Add new section member before this comment.
};

template <class Visitor>
void VisitSections(Visitor* PIK_RESTRICT visitor,
                   Sections* PIK_RESTRICT sections) {
  // Sections are read/written in this order, so do not rearrange.
  (*visitor)(&sections->alpha);
  (*visitor)(&sections->palette);
  // Add new section visitor before this comment.
}

// Returns whether "sections" can be encoded (i.e. all fields have a valid
// representation). If so, "*encoded_bits" is the exact number of bits required.
bool CanEncode(const Sections& sections, size_t* PIK_RESTRICT encoded_bits);

bool LoadSections(BitReader* source, Sections* PIK_RESTRICT sections);

bool StoreSections(const Sections& sections, size_t* PIK_RESTRICT pos,
                   uint8_t* storage);

#pragma pack(pop)

}  // namespace pik

#endif  // HEADER_H_
