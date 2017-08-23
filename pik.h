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

namespace pik {

struct CompressParams {
  // Only used for benchmarking (comparing vs libjpeg)
  int jpeg_quality = 100;
  bool jpeg_chroma_subsampling = false;
  bool clear_metadata = false;

  float butteraugli_distance = -1.0f;
  float target_bitrate = 0.0f;
  // 0.0 means search for the adaptive quantization map that matches the
  // butteraugli distance, positive values mean quantize everywhere with that
  // value.
  float uniform_quant = 0.0f;
  // If true, will use a compression method that is reasonably fast and aims to
  // find a trade-off between quality and file size that optimizes the
  // quality-adjusted-bits-per-pixel metric.
  bool fast_mode = false;

};

struct DecompressParams {
};

// The input image is an 8-bit sRGB image.
bool PixelsToPik(const CompressParams& params, const Image3B& planes,
                 PaddedBytes* compressed, PikInfo* aux_out);

// The input image is a linear (gamma expanded) sRGB image.
bool PixelsToPik(const CompressParams& params, const Image3F& linear,
                 PaddedBytes* compressed, PikInfo* aux_out);

// The input image is an opsin dynamics image.
bool OpsinToPik(const CompressParams& params, const Image3F& opsin,
                PaddedBytes* compressed, PikInfo* aux_out);


// The output image is an 8-bit sRGB image.
bool PikToPixels(const DecompressParams& params, const PaddedBytes& compressed,
                 Image3B* planes, PikInfo* aux_out);

// The output image is a 16-bit sRGB image.
bool PikToPixels(const DecompressParams& params, const PaddedBytes& compressed,
                 Image3U* planes, PikInfo* aux_out);

// The output image is a linear (gamma expanded) sRGB image.
bool PikToPixels(const DecompressParams& params, const PaddedBytes& compressed,
                 Image3F* planes, PikInfo* aux_out);
}  // namespace pik

#endif  // PIK_H_
