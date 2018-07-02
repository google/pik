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

#ifndef OPSIN_INVERSE_H_
#define OPSIN_INVERSE_H_

#include <stdint.h>
#include <algorithm>
#include <cstdlib>
#include <vector>

#include "compiler_specific.h"
#include "data_parallel.h"
#include "image.h"
#include "opsin_params.h"
#include "simd_helpers.h"

namespace pik {

// Inverts the pixel-wise RGB->XYB conversion in OpsinDynamicsImage() (including
// the gamma mixing and simple gamma), without clamping. "inverse_matrix" points
// to 9 broadcasted vectors, which are the 3x3 entries of the (row-major)
// opsin absorbance matrix inverse. Pre-multiplying its entries by c is
// equivalent to multiplying linear_* by c afterwards.
template <class D, class V>
PIK_INLINE void XybToRgbWithoutClamp(D d, const V opsin_x, const V opsin_y,
                                     const V opsin_b,
                                     const V* PIK_RESTRICT inverse_matrix,
                                     V* const PIK_RESTRICT linear_r,
                                     V* const PIK_RESTRICT linear_g,
                                     V* const PIK_RESTRICT linear_b) {
  using namespace SIMD_NAMESPACE;

#if SIMD_TARGET_VALUE == SIMD_NONE
  const auto inv_scale_x = set1(d, kInvScaleR);
  const auto inv_scale_y = set1(d, kInvScaleG);
  const auto neg_bias_r = set1(d, kNegOpsinAbsorbanceBiasRGB[0]);
  const auto neg_bias_g = set1(d, kNegOpsinAbsorbanceBiasRGB[1]);
  const auto neg_bias_b = set1(d, kNegOpsinAbsorbanceBiasRGB[2]);
#else
  const auto neg_bias_rgb = load_dup128(d, kNegOpsinAbsorbanceBiasRGB);
  SIMD_ALIGN const float inv_scale_lanes[4] = {kInvScaleR, kInvScaleG};
  const auto inv_scale = load_dup128(d, inv_scale_lanes);
  const auto inv_scale_x = broadcast<0>(inv_scale);
  const auto inv_scale_y = broadcast<1>(inv_scale);
  const auto neg_bias_r = broadcast<0>(neg_bias_rgb);
  const auto neg_bias_g = broadcast<1>(neg_bias_rgb);
  const auto neg_bias_b = broadcast<2>(neg_bias_rgb);
#endif

  // Color space: XYB -> RGB
  const auto gamma_r = inv_scale_x * (opsin_y + opsin_x);
  const auto gamma_g = inv_scale_y * (opsin_y - opsin_x);
  const auto gamma_b = opsin_b;

  // Undo gamma compression: linear = gamma^3 for efficiency.
  const auto gamma_r2 = gamma_r * gamma_r;
  const auto gamma_g2 = gamma_g * gamma_g;
  const auto gamma_b2 = gamma_b * gamma_b;
  const auto mixed_r = mul_add(gamma_r2, gamma_r, neg_bias_r);
  const auto mixed_g = mul_add(gamma_g2, gamma_g, neg_bias_g);
  const auto mixed_b = mul_add(gamma_b2, gamma_b, neg_bias_b);

  // Unmix (multiply by 3x3 inverse_matrix)
  *linear_r = inverse_matrix[0] * mixed_r;
  *linear_g = inverse_matrix[3] * mixed_r;
  *linear_b = inverse_matrix[6] * mixed_r;
  const auto tmp_r = inverse_matrix[1] * mixed_g;
  const auto tmp_g = inverse_matrix[4] * mixed_g;
  const auto tmp_b = inverse_matrix[7] * mixed_g;
  *linear_r = mul_add(inverse_matrix[2], mixed_b, *linear_r);
  *linear_g = mul_add(inverse_matrix[5], mixed_b, *linear_g);
  *linear_b = mul_add(inverse_matrix[8], mixed_b, *linear_b);
  *linear_r += tmp_r;
  *linear_g += tmp_g;
  *linear_b += tmp_b;
}

// Also clamps the resulting pixel values to [0.0, 255.0].
template <class D, class V>
PIK_INLINE void XybToRgb(D d, const V opsin_x, const V opsin_y, const V opsin_b,
                         const V* PIK_RESTRICT inverse_matrix,
                         V* const PIK_RESTRICT linear_r,
                         V* const PIK_RESTRICT linear_g,
                         V* const PIK_RESTRICT linear_b) {
  XybToRgbWithoutClamp(d, opsin_x, opsin_y, opsin_b, inverse_matrix, linear_r,
                       linear_g, linear_b);
  *linear_r = Clamp0To255(d, *linear_r);
  *linear_g = Clamp0To255(d, *linear_g);
  *linear_b = Clamp0To255(d, *linear_b);
}

// "dither" enables 2x2 dithering, but only if SIMD_TARGET_VALUE != SIMD_NONE
// and the output is U8 (first overload).
void CenteredOpsinToSrgb(const Image3F& opsin, const bool dither,
                         ThreadPool* pool, Image3B* srgb);
void CenteredOpsinToSrgb(const Image3F& opsin, const bool dither,
                         ThreadPool* pool, Image3U* srgb);
void CenteredOpsinToSrgb(const Image3F& opsin, const bool dither,
                         ThreadPool* pool, Image3F* srgb);

Image3B OpsinDynamicsInverse(const Image3F& opsin);
Image3F LinearFromOpsin(const Image3F& opsin);

}  // namespace pik

#endif  // OPSIN_INVERSE_H_
