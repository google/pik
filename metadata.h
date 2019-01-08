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

#ifndef METADATA_H_
#define METADATA_H_

// Image metadata stored in FileHeader and CodecInOut.

#include <stdint.h>

#include "color_encoding.h"
#include "field_encodings.h"
#include "padded_bytes.h"
#include "pik_params.h"
#include "status.h"

namespace pik {

// Optional metadata about the original image source.
struct Transcoded {
  Transcoded();
  constexpr const char* Name() const { return "Transcoded"; }

  template <class Visitor>
  Status VisitFields(Visitor* PIK_RESTRICT visitor) {
    if (visitor->AllDefault(*this, &needs_visit)) return true;

    visitor->U32(0x05A09088, 8, &original_bit_depth);
    PIK_RETURN_IF_ERROR(visitor->VisitNested(&original_color_encoding));
    visitor->U32(0x84828180u, 0, &original_bytes_per_alpha);

    return true;
  }

  bool needs_visit;

  uint32_t original_bit_depth;            // = CodecInOut.dec_bit_depth
  ColorEncoding original_color_encoding;  // = io->dec_c_original in the encoder
  // TODO(user): This should use bits instead of bytes, 1-bit alpha channel
  //             images exist and may be desired by users using this feature.
  // Alpha bytes per channel of original image (not necessarily the same as
  // the encoding used in the pik file).
  uint32_t original_bytes_per_alpha = 0;
};

struct Metadata {
  Metadata();
  constexpr const char* Name() const { return "Metadata"; }

  template <class Visitor>
  Status VisitFields(Visitor* PIK_RESTRICT visitor) {
    if (visitor->AllDefault(*this, &needs_visit)) return true;

    PIK_RETURN_IF_ERROR(visitor->VisitNested(&transcoded));

    // 100, 250, 4000 are common; don't anticipate more than 10,000.
    visitor->U32(0x08D08582, kDefaultIntensityTarget / 50, &target_nits_div50);

    visitor->Bytes(BytesEncoding::kBrotli, &exif);
    visitor->Bytes(BytesEncoding::kBrotli, &iptc);
    visitor->Bytes(BytesEncoding::kBrotli, &xmp);

    return true;
  }

  bool needs_visit;

  Transcoded transcoded;

  uint32_t target_nits_div50;

  PaddedBytes exif;
  PaddedBytes iptc;
  PaddedBytes xmp;
};

}  // namespace pik

#endif  // METADATA_H_
