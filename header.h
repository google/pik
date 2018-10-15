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
#include "field_encodings.h"
#include "padded_bytes.h"

namespace pik {

enum class ImageType : uint32_t {
  kPreview = 0,  // preview or pyramid
  kMain,         // main image or animation rect(s)
  // Used to terminate container of unbounded stream of Images.
  // All other Header fields are default-initialized.
  kSentinel
  // Future extensions: [3, 6]
};

// What comes after the header + sections.
enum class Bitstream : uint32_t {
  // Subsequent header fields are (only) valid in this mode.
  kDefault = 0,

  // Special bitstream for JPEG input images
  kBrunsli,

  // Future extensions: [2, 6]
};

// Header preceding all (sub-)images.
struct Header {
  // Optional postprocessing steps. These flags are the source of truth;
  // Override must set/clear them rather than change their meaning.
  enum Flags {
    // Additional smoothing; helpful for medium/low-quality.
    // Builds a value from [0..7] indicating the strength of the transform.
    // 0 when nothing is done, 4 at default smoothing.
    // kGaborishTransform0 is the lsb value.
    // kGaborishTransform2 is the msb value.
    kGaborishTransform0 = 1,
    kGaborishTransform1 = 2,
    kGaborishTransform2 = 4,
    kGaborishTransformMask = 7,
    kGaborishTransformShift = 0,

    // Inject noise into decoded output.
    kNoise = 8,

    // Predictor for DCT coefficients.
    kSmoothDCPred = 16,

    // Gradient map used to predict smooth areas.
    kGradientMap = 32,

    // Constrained filter in decoder, useful for deringing.
    kAdaptiveReconstruction = 64,

    // Experimental, go/pik-block-strategy
    kUseAcStrategy = 128,

    // Image is compressed with grayscale optimizations. Only used for parsing
    // of pik file, may not be used to determine decompressed color format or
    // ICC color profile.
    kGrayscaleOpt = 256,
  };

  // If kSentinel, all other header fields are default-initialized.
  ImageType image_type = ImageType::kMain;

  uint32_t xsize = 0;  // pixels, not necessarily a multiple of kBlockDim
  uint32_t ysize = 0;

  uint32_t resampling_factor2 = 2;

  // If != kDefault, all subsequent fields are default-initialized.
  Bitstream bitstream = Bitstream::kDefault;

  uint32_t flags = 0;
};

// For loading/storing fields from/to the compressed stream.
template <class Visitor>
Status VisitFields(Visitor* PIK_RESTRICT visitor, Header* PIK_RESTRICT header) {
  visitor->Enum(kU32Direct3Plus4, &header->image_type);
  if (header->image_type == ImageType::kSentinel) return true;

  // Almost all camera images are less than 8K * 8K. We also allow the
  // full 32-bit range for completeness.
  visitor->U32(0x200D0B09, &header->xsize);
  visitor->U32(0x200D0B09, &header->ysize);
  visitor->U32(kU32Direct2348, &header->resampling_factor2);

  visitor->Enum(kU32Direct3Plus4, &header->bitstream);
  if (header->bitstream != Bitstream::kDefault) return true;

  visitor->U32(0x20181008, &header->flags);

  // To extend: add a section, or add fields conditional on a NEW flag:
  // if (flag) visitor->U32(..).

  return true;
}

// Returns whether the struct can be encoded (i.e. all fields have a valid
// representation). If so, "*encoded_bits" is the exact number of bits required.
Status CanEncode(const Header& header, size_t* PIK_RESTRICT encoded_bits);

Status LoadHeader(BitReader* reader, Header* PIK_RESTRICT header);

Status StoreHeader(const Header& header, size_t* pos, uint8_t* storage);

// Optional per-image extensions (backward- and forward-compatible):

// Alpha channel (lossless compression).
struct Alpha {
  enum {
    kModeBrotli = 0,
    kModeTransform
    // Future extensions: [2, 6]
  };

  uint32_t mode = kModeBrotli;
  uint32_t bytes_per_alpha = 1;
  PaddedBytes encoded;  // interpretation depends on mode
};

template <class Visitor>
Status VisitFields(Visitor* PIK_RESTRICT visitor, Alpha* PIK_RESTRICT alpha) {
  visitor->U32(kU32Direct3Plus4, &alpha->mode);
  visitor->U32(0x84828180u, &alpha->bytes_per_alpha);
  visitor->Bytes(Bytes::kRaw, &alpha->encoded);
  return true;
}

// Superset of PNG PLTE+tRNS.
struct Palette {
  enum { kEncodingRaw = 0 };

  // 1 or 2 (little-endian) byte per entry AND channel (RGB).
  uint32_t bytes_per_color = 1;

  // E.g. "255" instead of 256 to save one bit.
  uint32_t num_colors_minus_one = 0;

  // How many alpha entries (often 0 or 1, hence separate from colors).
  uint32_t num_alpha = 0;

  // "num_colors_minus_one+1" times 8/16-bit {R, G, B}; will not be reordered.
  PaddedBytes colors;

  // "num_alpha" times 8/16-bit opacity; will not be reordered.
  PaddedBytes alpha;
};

template <class Visitor>
Status VisitFields(Visitor* PIK_RESTRICT visitor,
                   Palette* PIK_RESTRICT palette) {
  visitor->U32(0x84828180u, &palette->bytes_per_color);
  visitor->U32(0x0C0A0806, &palette->num_colors_minus_one);
  visitor->U32(0x0C088180, &palette->num_alpha);
  visitor->Bytes(Bytes::kBrotli, &palette->colors);
  visitor->Bytes(Bytes::kBrotli, &palette->alpha);
  return true;
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
Status CanEncode(const Sections& sections, size_t* PIK_RESTRICT encoded_bits);

Status LoadSections(BitReader* source, Sections* PIK_RESTRICT sections);

Status StoreSections(const Sections& sections, size_t* PIK_RESTRICT pos,
                     uint8_t* storage);

}  // namespace pik

#endif  // HEADER_H_
