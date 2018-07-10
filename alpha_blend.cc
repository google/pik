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

#include "alpha_blend.h"
#include "compiler_specific.h"
#include "gamma_correct.h"

namespace pik {

ImageF AlphaBlend(const ImageF& img, const ImageU& alpha,
                  int bit_depth, uint8_t background) {
  const float bg_lin = Srgb8ToLinearTable()[background];
  const uint16_t opaque = 0xffff >> (16 - bit_depth);
  ImageF out(img.xsize(), img.ysize());
  for (int y = 0; y < img.ysize(); ++y) {
    const uint16_t* const PIK_RESTRICT row_a = alpha.Row(y);
    const float* const PIK_RESTRICT row_i = img.Row(y);
    float* const PIK_RESTRICT row_o = out.Row(y);
    for (int x = 0; x < img.xsize(); ++x) {
      const uint16_t a = row_a[x];
      if (a == 0) {
        row_o[x] = bg_lin;
      } else if (a == opaque) {
        row_o[x] = row_i[x];
      } else {
        const float w_fg = a * 1.0f / opaque;
        const float w_bg = 1.0f - w_fg;
        const float fg = w_fg * LinearToSrgb8Direct(row_i[x]);
        const float bg = w_bg * background;
        row_o[x] = Srgb8ToLinearDirect(fg + bg);
      }
    }
  }
  return out;
}

Image3F AlphaBlend(const MetaImageF& img, uint8_t background) {
  if (!img.HasAlpha()) {
    return CopyImage(img.GetColor());
  }
  std::array<ImageF, 3> planes;
  for (int c = 0; c < 3; ++c) {
    planes[c] = AlphaBlend(img.GetColor().Plane(c), img.GetAlpha(),
                           img.AlphaBitDepth(), background);
  }
  return Image3F(planes);
}

}  // namespace pik
