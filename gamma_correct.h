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

#ifndef GAMMA_CORRECT_H_
#define GAMMA_CORRECT_H_

#include <stdint.h>
#include <algorithm>
#include <cmath>

#include "arch_specific.h"
#include "compiler_specific.h"
#include "image.h"

namespace pik {

const float* Srgb8ToLinearTable();

const uint8_t* LinearToSrgb8Table();
const uint8_t* LinearToSrgb8TablePlusQuarter();
const uint8_t* LinearToSrgb8TableMinusQuarter();

PIK_INLINE uint8_t LinearToSrgb8(const uint8_t* lut, float val) {
  val = std::min(255.0f, std::max(0.0f, val));
  return lut[static_cast<int>(val * 16.0f + 0.5f)];
}

// Naive/direct computation used to initialize lookup table.
PIK_INLINE float Srgb8ToLinearDirect(float val) {
  if (val < 0.0) return 0.0;
  if (val <= 10.31475) return val / 12.92;
  if (val >= 255.0) return 255.0;
  return 255.0 * std::pow(((val / 255.0) + 0.055) / 1.055, 2.4);
}

// Naive/direct computation used to initialize lookup table. In/out: 0-255.
PIK_INLINE float LinearToSrgb8Direct(float val) {
  if (val < 0.0) return 0.0;
  if (val >= 255.0) return 255.0;
  if (val <= 10.31475 / 12.92) return val * 12.92;
  return 255.0 * (std::pow(val / 255.0, 1.0 / 2.4) * 1.055 - 0.055);
}

ImageF LinearFromSrgb(const ImageB& srgb);
Image3F LinearFromSrgb(const Image3B& srgb);
Image3F LinearFromSrgb(const Image3U& srgb);

ImageB Srgb8FromLinear(const ImageF& linear);
Image3B Srgb8FromLinear(const Image3F& linear);

// Good enough approximation for 16-bit precision in floating point range 0-255.
template <class D, class V>
V Pow24Poly(D d, V z) {
  // Max error: 0.0033, just enough for 16-bit precision in range 0.05-255.0
  return (((((((set1(d, -4.68139386898368e-16f) * z +
                set1(d, 4.90086432807652e-13f)) *
                   z +
               set1(d, -2.47340675718632e-10f)) *
                  z +
              set1(d, 1.19290078259837e-07f)) *
                 z +
             set1(d, 2.52620611718157e-05f)) *
                z +
            set1(d, 0.000444842939032242f)) *
               z +
           set1(d, 0.000799090310465544f)) *
              z +
          set1(d, -0.000163653719937429f)) /
         (((set1(d, 1.56013541641187e-07f) * z +
            set1(d, 7.53337144487887e-06f)) *
               z +
           set1(d, 4.70604936708696e-05f)) *
              z +
          set1(d, 3.22659288940486e-05f));
}

template <class D, typename V>
V LinearToSrgbPoly(D d, V z) {
  const V linear = z * set1(d, 12.92f);
  const V poly = Pow24Poly(d, z);
  const V ret = select(linear, poly, z > set1(d, 10.31475f / 12.92f));
  return clamp(ret, setzero(d), set1(d, 255.0f));
}

// Returns sRGB as floating-point (same range but not rounded to integer).
ImageF SrgbFFromLinear(const ImageF& linear);
Image3F SrgbFFromLinear(const Image3F& linear);

}  // namespace pik

#endif  // GAMMA_CORRECT_H_
