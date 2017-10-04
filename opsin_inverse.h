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

// V is vec1<float> or vec<float>.

template <typename V>
PIK_INLINE V XyToR(const V x, const V y) {
  return set1(V(), kInvScaleR) * (y + x);
}

template <typename V>
PIK_INLINE V XyToG(const V x, const V y) {
  return set1(V(), kInvScaleG) * (y - x);
}

template <typename V>
PIK_INLINE V SimpleGammaInverse(const V v) {
  return v * v * v;
}

template <typename V>
PIK_INLINE V MixedToRed(const V r, const V g, const V b) {
  return (set1(V(), kOpsinAbsorbanceInverseMatrix[0]) * r +
          set1(V(), kOpsinAbsorbanceInverseMatrix[1]) * g +
          set1(V(), kOpsinAbsorbanceInverseMatrix[2]) * b);
}

template <typename V>
PIK_INLINE V MixedToGreen(const V r, const V g, const V b) {
  return (set1(V(), kOpsinAbsorbanceInverseMatrix[3]) * r +
          set1(V(), kOpsinAbsorbanceInverseMatrix[4]) * g +
          set1(V(), kOpsinAbsorbanceInverseMatrix[5]) * b);
}

template <typename V>
PIK_INLINE V MixedToBlue(const V r, const V g, const V b) {
  return (set1(V(), kOpsinAbsorbanceInverseMatrix[6]) * r +
          set1(V(), kOpsinAbsorbanceInverseMatrix[7]) * g +
          set1(V(), kOpsinAbsorbanceInverseMatrix[8]) * b);
}

template <typename V>
PIK_INLINE V Clamp0To255(const V x) {
  return clamp(x, setzero(V()), set1(V(), 255.0f));
}

// Inverts the pixel-wise RGB->XYB conversion in OpsinDynamicsImage() (including
// the gamma mixing and simple gamma) and clamps the resulting pixel values
// between 0.0 and 255.0.
template <typename V>
PIK_INLINE void XybToRgb(const V x, const V y, const V b,
                         V* const PIK_RESTRICT red, V* const PIK_RESTRICT green,
                         V* const PIK_RESTRICT blue) {
  const V r_mix = SimpleGammaInverse(XyToR(x, y));
  const V g_mix = SimpleGammaInverse(XyToG(x, y));
  const V b_mix = SimpleGammaInverse(b);
  *red = Clamp0To255(MixedToRed(r_mix, g_mix, b_mix));
  *green = Clamp0To255(MixedToGreen(r_mix, g_mix, b_mix));
  *blue = Clamp0To255(MixedToBlue(r_mix, g_mix, b_mix));
}

Image3B OpsinDynamicsInverse(const Image3F& opsin);
Image3F LinearFromOpsin(const Image3F& opsin);

}  // namespace pik

#endif  // OPSIN_INVERSE_H_
