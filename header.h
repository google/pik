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

// Container file format with backward and forward-compatible extension
// capability and compression of integer fields.

#include <stddef.h>
#include <stdint.h>
#include <memory>
#include <vector>

#include "bit_buffer.h"
#include "compiler_specific.h"
#include "status.h"

namespace pik {

#pragma pack(push, 1)

// U32Coder selector_bits for a uniform distribution over 32-bit lengths.
constexpr uint32_t kU32Selectors = 0x20181008;

using Bytes = std::vector<uint8_t>;

// Header applicable to all image types. Only minimally extensible via flags;
// use Sections to add additional functionality in future.
struct Header {
  // What comes after the header + sections.
  enum {
    kBitstreamDefault = 0,
    kBitstreamLossless = 1,

    // Special bitstream for JPEG input images
    kBitstreamBrunsli = 2,
  };

  // Loosely associated bit flags that together describe the pixel format and
  // postprocessing of the kDefault bitstream.
  enum Flags {
    // There is an additional alpha plane, compressed without loss. None of the
    // other components are premultiplied.
    kAlpha = 1,

    // Decoder should apply edge-preserving filter in opsin space. Useful for
    // deblocking/deringing in lower-quality settings.
    kDenoise = 2,

    kSmoothDCPred = 4,

    // Additional smoothing; helpful for medium/low-quality.
    kGaborishTransform = 8,

    // Dither opsin values before rounding to 8-bit SRGB. Generally beneficial
    // except for high-quality bitstreams (<= distance 1).
    kDither = 16
  };

  // For loading/storing fields from/to the compressed stream. Accepts Bytes or
  // uint32_t preceded by U32Coder's selector_bits.
  template <class Visitor>
  void VisitFields(Visitor* const PIK_RESTRICT visitor) {
    // Almost all camera images are less than 8K * 8K. We also allow the
    // full 32-bit range for completeness.
    (*visitor)(0x200E0B09, &xsize);
    (*visitor)(0x200E0B09, &ysize);
    // 2-bit encodings for 1 and 3 common cases; a dozen components for
    // remote-sensing data, or thousands for hyperspectral images.
    (*visitor)(0x00100004, &num_components);
    (*visitor)(0x06000000, &bitstream);
    (*visitor)(kU32Selectors, &flags);
    // Direct 2-bit encoding for quant template ids 0, 1, 2, 3.
    (*visitor)(0x00000000, &quant_template);

    // To extend: add a section, or add fields conditional on a NEW flag:
    // if (flag) (*visitor)(..).
  }

  uint32_t xsize = 0;
  uint32_t ysize = 0;
  uint32_t num_components = 0;
  uint32_t bitstream = kBitstreamDefault;
  uint32_t flags = 0;
  uint32_t quant_template = 0;
  // TODO(janwas): hash?
};

// Returns an upper bound on the number of bytes needed to store a Header.
size_t MaxCompressedHeaderSize();

// Returns end of "from".
const uint8_t* LoadHeader(const uint8_t* from,
                Header* const PIK_RESTRICT header);

uint8_t* StoreHeader(const Header& header, uint8_t* to);

// A "section" is optional metadata that is only read from/written to the
// compressed stream if needed. Adding sections is the only way to extend
// this container format. Old code is forwards compatible because unknown
// sections can be skipped. Old files are backwards compatible because they
// indicate future sections are not present using a bit field in the header.
//
// To ensure interoperability, there will be no opaque fields nor proprietary
// section definitions. To introduce a new section and thereby advance the
// "version number" of this file format, add a unique_ptr member to Header and
// append a call to "visitor" in Header::VisitSections.

struct ICC {
  template <class Visitor>
  void VisitFields(Visitor* const PIK_RESTRICT visitor) {
    (*visitor)(&profile);
  }

  Bytes profile;
};

struct EXIF {
  template <class Visitor>
  void VisitFields(Visitor* const PIK_RESTRICT visitor) {
    (*visitor)(&metadata);
  }

  Bytes metadata;
};

// Superset of PNG PLTE+tRNS.
struct Palette {
  enum { kEncodingRaw = 0 };

  template <class Visitor>
  void VisitFields(Visitor* const PIK_RESTRICT visitor) {
    (*visitor)(0x04000000, &encoding);
    (*visitor)(0x00000000, &bytes_per_color);
    (*visitor)(0x0C0A0806, &num_colors_minus_one);
    (*visitor)(0x0C080000, &num_alpha);
    (*visitor)(&colors);
    (*visitor)(&alpha);
  }

  // Whether color/alpha are compressed.
  uint32_t encoding = kEncodingRaw;

  // 1 or 2 (little-endian) byte per entry AND channel (RGB).
  uint32_t bytes_per_color = 1;

  // E.g. "255" instead of 256 to save one bit.
  uint32_t num_colors_minus_one = 0;

  // How many alpha entries (often 0 or 1, hence separate from colors).
  uint32_t num_alpha = 0;

  // "num_colors_minus_one+1" times 8/16-bit {R, G, B}; will not be reordered.
  Bytes colors;

  // "num_alpha" times 8/16-bit opacity; will not be reordered.
  Bytes alpha;
};

struct Sections {
  template <class Visitor>
  void VisitSections(Visitor* const PIK_RESTRICT visitor) {
    // Sections are read/written in this order, so do not rearrange.
    (*visitor)(&icc);
    (*visitor)(&exif);
    (*visitor)(&palette);
    // Add new sections before this comment.
  }

  // Valid/present if non-null.
  std::unique_ptr<ICC> icc;
  std::unique_ptr<EXIF> exif;
  std::unique_ptr<Palette> palette;
};

// Returns an upper bound on the number of bytes needed to store "sections".
size_t MaxCompressedSectionsSize(const Sections& sections);

const uint8_t* const PIK_RESTRICT
LoadSections(SIMD_NAMESPACE::BitSource* const PIK_RESTRICT source,
             Sections* const PIK_RESTRICT sections);

// "max_bytes" is the return value of MaxCompressedSectionsSize.
uint8_t* const PIK_RESTRICT
StoreSections(const Sections& sections, const size_t max_bytes,
              SIMD_NAMESPACE::BitSink* const PIK_RESTRICT sink);

#pragma pack(pop)

}  // namespace pik

#endif  // HEADER_H_
