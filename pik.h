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

#include "data_parallel.h"
#include "guetzli/jpeg_data.h"
#include "header.h"
#include "image.h"
#include "padded_bytes.h"
#include "pik_info.h"
#include "pik_params.h"
#include "status.h"

namespace pik {

// The input image is an 8-bit sRGB image.
bool PixelsToPik(const CompressParams& params, const MetaImageB& image,
                 ThreadPool* pool, PaddedBytes* compressed,
                 PikInfo* aux_out = nullptr);
bool PixelsToPik(const CompressParams& params, const Image3B& image,
                 ThreadPool* pool, PaddedBytes* compressed,
                 PikInfo* aux_out = nullptr);

// The input image is a linear (gamma expanded) sRGB image.
bool PixelsToPik(const CompressParams& params, const MetaImageF& linear,
                 ThreadPool* pool, PaddedBytes* compressed,
                 PikInfo* aux_out = nullptr);
bool PixelsToPik(const CompressParams& params, const Image3F& linear,
                 ThreadPool* pool, PaddedBytes* compressed,
                 PikInfo* aux_out = nullptr);

// The input image is an opsin dynamics image.
bool OpsinToPik(const CompressParams& params, const Header& header,
                const MetaImageF& opsin,
                ThreadPool* pool, PaddedBytes* compressed,
                PikInfo* aux_out = nullptr);

// The input image is a (partially decoded) JPEG image.
bool JpegToPik(const CompressParams& params, const guetzli::JPEGData& jpeg,
               ThreadPool* pool, PaddedBytes* compressed,
               PikInfo* aux_out = nullptr);

// The output image is an 8-bit sRGB image.
bool PikToPixels(const DecompressParams& params, const PaddedBytes& compressed,
                 ThreadPool* pool, MetaImageB* image,
                 PikInfo* aux_out = nullptr);
bool PikToPixels(const DecompressParams& params, const PaddedBytes& compressed,
                 ThreadPool* pool, Image3B* image, PikInfo* aux_out = nullptr);

// The output image is a 16-bit sRGB image.
bool PikToPixels(const DecompressParams& params, const PaddedBytes& compressed,
                 ThreadPool* pool, MetaImageU* image,
                 PikInfo* aux_out = nullptr);
bool PikToPixels(const DecompressParams& params, const PaddedBytes& compressed,
                 ThreadPool* pool, Image3U* image, PikInfo* aux_out = nullptr);

// The output image is a floating-point sRGB image.
bool PikToPixels(const DecompressParams& params, const PaddedBytes& compressed,
                 ThreadPool* pool, MetaImageF* image,
                 PikInfo* aux_out = nullptr);
bool PikToPixels(const DecompressParams& params, const PaddedBytes& compressed,
                 ThreadPool* pool, Image3F* image, PikInfo* aux_out = nullptr);

}  // namespace pik

#endif  // PIK_H_
