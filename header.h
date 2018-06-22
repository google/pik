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
#include <vector>

#include "bit_reader.h"
#include "compiler_specific.h"

namespace pik {

#pragma pack(push, 1)

// File header applicable to all image types.
struct Header {
  // What comes after the header + sections.
  enum {
    kBitstreamDefault = 0,
    kBitstreamLossless = 1,

    // Special bitstream for JPEG input images
    kBitstreamBrunsli = 2,
  };

  // Optional postprocessing steps for the kDefault bitstream.
  enum Flags {
    kSmoothDCPred = 1,

    // Decoder should apply edge-preserving filter in opsin space. Useful for
    // deblocking/deringing in lower-quality settings.
    kDenoise = 2,

    // Additional smoothing; helpful for medium/low-quality.
    kGaborishTransform = 4,

    // Dither opsin values before rounding to 8-bit SRGB. Generally beneficial
    // except for high-quality bitstreams (<= distance 1).
    kDither = 8
  };

  // For loading/storing fields from/to the compressed stream. Accepts Bytes or
  // uint32_t preceded by U32Coder's selector_bits.
  template <class Visitor>
  void VisitFields(Visitor* const PIK_RESTRICT visitor) {
    // Almost all camera images are less than 8K * 8K. We also allow the
    // full 32-bit range for completeness.
    (*visitor)(0x200D0B09, &xsize);
    (*visitor)(0x200D0B09, &ysize);
    // 2-bit encodings for 1 and 3 common cases; a dozen components for
    // remote-sensing data, or thousands for hyperspectral images.
    (*visitor)(0x10048381, &num_components);
    (*visitor)(0x06828180, &bitstream);
    (*visitor)(0x20181008, &flags);
    // Direct 2-bit encoding for quant template ids 0, 1, 2, 3.
    (*visitor)(0x83828180, &quant_template);

    // To extend: add a section, or add fields conditional on a NEW flag:
    // if (flag) (*visitor)(..).
  }

  uint32_t xsize = 0;
  uint32_t ysize = 0;
  uint32_t num_components = 0;
  uint32_t bitstream = kBitstreamDefault;
  uint32_t flags = 0;
  uint32_t quant_template = 0;
};

#pragma pack(pop)

// Returns whether "header" can be encoded (i.e. all fields have a valid
// representation). If so, "*encoded_bits" is the exact number of bits required.
bool CanEncode(const Header& header, size_t* PIK_RESTRICT encoded_bits);

bool LoadHeader(BitReader* source, Header* PIK_RESTRICT header);

bool StoreHeader(const Header& header, size_t* pos, uint8_t* storage);

}  // namespace pik

#endif  // HEADER_H_
