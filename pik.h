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

#ifndef PIK_H_
#define PIK_H_

#include <string>

#include "image.h"
#include "pik_info.h"
#include "status.h"
#include "padded_bytes.h"
#include "pik_params.h"

namespace pik {

// The input image is an 8-bit sRGB image.
bool PixelsToPik(const CompressParams& params, const MetaImageB& image,
                 PaddedBytes* compressed, PikInfo* aux_out);
bool PixelsToPik(const CompressParams& params, const Image3B& image,
                 PaddedBytes* compressed, PikInfo* aux_out);

// The input image is a linear (gamma expanded) sRGB image.
bool PixelsToPik(const CompressParams& params, const MetaImageF& linear,
                 PaddedBytes* compressed, PikInfo* aux_out);
bool PixelsToPik(const CompressParams& params, const Image3F& linear,
                 PaddedBytes* compressed, PikInfo* aux_out);

// The input image is an opsin dynamics image.
bool OpsinToPik(const CompressParams& params, const Image3F& opsin,
                PaddedBytes* compressed, PikInfo* aux_out);


// The output image is an 8-bit sRGB image.
bool PikToPixels(const DecompressParams& params, const PaddedBytes& compressed,
                 MetaImageB* image, PikInfo* aux_out);
bool PikToPixels(const DecompressParams& params, const PaddedBytes& compressed,
                 Image3B* image, PikInfo* aux_out);

// The output image is a 16-bit sRGB image.
bool PikToPixels(const DecompressParams& params, const PaddedBytes& compressed,
                 MetaImageU* image, PikInfo* aux_out);
bool PikToPixels(const DecompressParams& params, const PaddedBytes& compressed,
                 Image3U* image, PikInfo* aux_out);

// The output image is a linear (gamma expanded) sRGB image.
bool PikToPixels(const DecompressParams& params, const PaddedBytes& compressed,
                 MetaImageF* image, PikInfo* aux_out);
bool PikToPixels(const DecompressParams& params, const PaddedBytes& compressed,
                 Image3F* image, PikInfo* aux_out);
}  // namespace pik

#endif  // PIK_H_
