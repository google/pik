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

#include "butteraugli_distance.h"

#include <stddef.h>
#include <algorithm>

#undef PROFILER_ENABLED
#define PROFILER_ENABLED 1
#include "butteraugli/butteraugli.h"
#include "gamma_correct.h"
#include "profiler.h"

namespace pik {
namespace {

float ButteraugliDistanceLinearSRGB(const Image3F& rgb0, const Image3F& rgb1,
                                    float hf_asymmetry, ImageF* distmap_out) {
  ImageF distmap;
  PIK_CHECK(butteraugli::ButteraugliDiffmap(rgb0, rgb1, hf_asymmetry, distmap));
  if (distmap_out != nullptr) {
    *distmap_out = CopyImage(distmap);
  }
  return butteraugli::ButteraugliScoreFromDiffmap(distmap);
}

// color is linear, but blending happens in gamma-compressed space using
// (gamma-compressed) grayscale background color, alpha image represents
// weights of the sRGB colors in the [0 .. (1 << bit_depth) - 1] interval,
// output image is in linear space.
void AlphaBlend(const ImageF& in, float background_linear255,
                const ImageU& alpha, const uint16_t opaque, ImageF* out) {
  const float background = LinearToSrgb8Direct(background_linear255);

  for (size_t y = 0; y < out->ysize(); ++y) {
    const uint16_t* PIK_RESTRICT row_a = alpha.ConstRow(y);
    const float* PIK_RESTRICT row_i = in.ConstRow(y);
    float* PIK_RESTRICT row_o = out->Row(y);
    for (size_t x = 0; x < out->xsize(); ++x) {
      const uint16_t a = row_a[x];
      if (a == 0) {
        row_o[x] = background_linear255;
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
}

const Image3F* AlphaBlend(const CodecInOut& io, const Image3F& linear,
                          float background_linear255, Image3F* copy) {
  // No alpha => all opaque.
  if (!io.HasAlpha()) return &linear;

  *copy = Image3F(linear.xsize(), linear.ysize());
  const uint16_t opaque = (1U << io.AlphaBits()) - 1;
  AlphaBlend(linear.Plane(0), background_linear255, io.alpha(), opaque,
             copy->MutablePlane(0));
  AlphaBlend(linear.Plane(1), background_linear255, io.alpha(), opaque,
             copy->MutablePlane(1));
  AlphaBlend(linear.Plane(2), background_linear255, io.alpha(), opaque,
             copy->MutablePlane(2));
  return copy;
}

}  // namespace

float ButteraugliDistance(const CodecInOut* rgb0, const CodecInOut* rgb1,
                          float hf_asymmetry, ImageF* distmap_out) {
  PROFILER_FUNC;
  // Convert to linear sRGB (unless already in that space)
  const Image3F* linear_srgb0 = &rgb0->color();
  Image3F linear_srgb_copy0;
  if (!rgb0->IsLinearSRGB()) {
    const ColorEncoding& c = rgb0->Context()->c_linear_srgb[rgb0->IsGray()];
    PIK_CHECK(rgb0->CopyTo(c, &linear_srgb_copy0));
    linear_srgb0 = &linear_srgb_copy0;
  }
  const Image3F* linear_srgb1 = &rgb1->color();
  Image3F linear_srgb_copy1;
  if (!rgb1->IsLinearSRGB()) {
    const ColorEncoding& c = rgb1->Context()->c_linear_srgb[rgb1->IsGray()];
    PIK_CHECK(rgb1->CopyTo(c, &linear_srgb_copy1));
    linear_srgb1 = &linear_srgb_copy1;
  }

  // No alpha: skip blending, only need a single call to Butteraugli.
  if (!rgb0->HasAlpha() && !rgb1->HasAlpha()) {
    return ButteraugliDistanceLinearSRGB(*linear_srgb0, *linear_srgb1,
                                         hf_asymmetry, distmap_out);
  }

  // Blend on black and white backgrounds

  const float black = 0.0f;
  Image3F copy_black0, copy_black1;
  const Image3F* blended_black0 =
      AlphaBlend(*rgb0, *linear_srgb0, black, &copy_black0);
  const Image3F* blended_black1 =
      AlphaBlend(*rgb1, *linear_srgb1, black, &copy_black1);

  const float white = 255.0f;
  Image3F copy_white0, copy_white1;
  const Image3F* blended_white0 =
      AlphaBlend(*rgb0, *linear_srgb0, white, &copy_white0);
  const Image3F* blended_white1 =
      AlphaBlend(*rgb1, *linear_srgb1, white, &copy_white1);

  ImageF distmap_black, distmap_white;
  const float dist_black = ButteraugliDistanceLinearSRGB(
      *blended_black0, *blended_black1, hf_asymmetry, &distmap_black);
  const float dist_white = ButteraugliDistanceLinearSRGB(
      *blended_white0, *blended_white1, hf_asymmetry, &distmap_white);

  // distmap and return values are the max of distmap_black/white.
  if (distmap_out != nullptr) {
    const size_t xsize = rgb0->xsize();
    const size_t ysize = rgb0->ysize();
    *distmap_out = ImageF(xsize, ysize);
    for (size_t y = 0; y < ysize; ++y) {
      const float* PIK_RESTRICT row_black = distmap_black.ConstRow(y);
      const float* PIK_RESTRICT row_white = distmap_white.ConstRow(y);
      float* PIK_RESTRICT row_out = distmap_out->Row(y);
      for (size_t x = 0; x < xsize; ++x) {
        row_out[x] = std::max(row_black[x], row_white[x]);
      }
    }
  }
  return std::max(dist_black, dist_white);
}

}  // namespace pik
