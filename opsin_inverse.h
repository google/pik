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
#include "image.h"
#include "opsin_params.h"
#include "simd/simd.h"

namespace pik {

// V is scalar<float> or vec<float>.

template <class D, class V>
PIK_INLINE V XyToR(D d, const V x, const V y) {
  return set1(d, kInvScaleR) * (y + x);
}

template <class D, class V>
PIK_INLINE V XyToG(D d, const V x, const V y) {
  return set1(d, kInvScaleG) * (y - x);
}

template <class V>
PIK_INLINE V SimpleGammaInverse(const V v) {
  return v * v * v;
}

template <class D, class V>
PIK_INLINE V MixedToRed(D d, const V r, const V g, const V b) {
  return (set1(d, kOpsinAbsorbanceInverseMatrix[0]) * r +
          set1(d, kOpsinAbsorbanceInverseMatrix[1]) * g +
          set1(d, kOpsinAbsorbanceInverseMatrix[2]) * b);
}

template <class D, class V>
PIK_INLINE V MixedToGreen(D d, const V r, const V g, const V b) {
  return (set1(d, kOpsinAbsorbanceInverseMatrix[3]) * r +
          set1(d, kOpsinAbsorbanceInverseMatrix[4]) * g +
          set1(d, kOpsinAbsorbanceInverseMatrix[5]) * b);
}

template <class D, class V>
PIK_INLINE V MixedToBlue(D d, const V r, const V g, const V b) {
  return (set1(d, kOpsinAbsorbanceInverseMatrix[6]) * r +
          set1(d, kOpsinAbsorbanceInverseMatrix[7]) * g +
          set1(d, kOpsinAbsorbanceInverseMatrix[8]) * b);
}

template <class D, class V>
PIK_INLINE V Clamp0To255(D d, const V x) {
  return clamp(x, setzero(d), set1(d, 255.0f));
}

// Inverts the pixel-wise RGB->XYB conversion in OpsinDynamicsImage() (including
// the gamma mixing and simple gamma) and clamps the resulting pixel values
// between 0.0 and 255.0.
template <class D, class V>
PIK_INLINE void XybToRgb(D d, const V x, const V y, const V b,
                         V* const PIK_RESTRICT red, V* const PIK_RESTRICT green,
                         V* const PIK_RESTRICT blue) {
  const auto r_mix = SimpleGammaInverse(XyToR(d, x, y));
  const auto g_mix = SimpleGammaInverse(XyToG(d, x, y));
  const auto b_mix = SimpleGammaInverse(b);
  *red = Clamp0To255(d, MixedToRed(d, r_mix, g_mix, b_mix));
  *green = Clamp0To255(d, MixedToGreen(d, r_mix, g_mix, b_mix));
  *blue = Clamp0To255(d, MixedToBlue(d, r_mix, g_mix, b_mix));
}

void CenteredOpsinToSrgb(const Image3F& opsin, Image3B* srgb);
void CenteredOpsinToSrgb(const Image3F& opsin, Image3U* srgb);
void CenteredOpsinToSrgb(const Image3F& opsin, Image3F* srgb);

Image3B OpsinDynamicsInverse(const Image3F& opsin);
Image3F LinearFromOpsin(const Image3F& opsin);

}  // namespace pik

#endif  // OPSIN_INVERSE_H_
