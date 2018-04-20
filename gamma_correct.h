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
#include "gamma_correct_poly.h"
#include "image.h"
#include "simd_helpers.h"

namespace pik {

const float* Srgb8ToLinearTable();

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
PIK_INLINE double LinearToSrgb8Direct(double val) {
  // Tests fail if this is computed with single-precision.
  using T = double;
  if (val < T(0.0)) return T(0.0);
  if (val >= T(255.0)) return T(255.0);
  if (val <= T(10.31475 / 12.92)) return val * T(12.92);
  const T val01 = val / T(255.0);
  return T(255.0) * (std::pow(val01, T(1.0 / 2.4)) * T(1.055) - T(0.055));
}

ImageF LinearFromSrgb(const ImageB& srgb);
Image3F LinearFromSrgb(const Image3B& srgb);
Image3F LinearFromSrgb(const Image3U& srgb);

ImageB Srgb8FromLinear(const ImageF& linear);
Image3B Srgb8FromLinear(const Image3F& linear);

// Returns sRGB as floating-point (same range but not rounded to integer).
ImageF SrgbFFromLinear(const ImageF& linear);
Image3F SrgbFFromLinear(const Image3F& linear);

}  // namespace pik

#endif  // GAMMA_CORRECT_H_
