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

#include "gamma_correct.h"

#include <stddef.h>
#include <cmath>

#include "compiler_specific.h"
#include "profiler.h"
#include "vector256.h"

namespace pik {
namespace {
template<typename V>
V Pow24Poly(V z) {
  // Max error: 0.0033, just enough for 16-bit precision in range 0.05-255.0
  return
      (((((((V(-4.68139386898368e-16f) * z + V(4.90086432807652e-13f)) * z +
      V(-2.47340675718632e-10f)) * z + V(1.19290078259837e-07f)) * z +
      V(2.52620611718157e-05f)) * z + V(0.000444842939032242f)) * z +
      V(0.000799090310465544f)) * z + V(-0.000163653719937429f)) /
      (((V(1.56013541641187e-07f) * z + V(7.53337144487887e-06f)) * z +
      V(4.70604936708696e-05f)) * z + V(3.22659288940486e-05f));
}

float LinearToSrgbPolyImpl(float z) {
  if (z <= 0.0f) return 0.0f;
  if (z >= 255) return 255;
  if (z <= 10.31475f / 12.92f) return z * 12.92f;
  return Pow24Poly(z);
}

template<typename V>
V LinearToSrgbPolyImpl(V z) {
  // When needed, z should be clamped in 0-255. Not done so far for extra speed.
  const V linear = z * V(12.92f);
  const V poly = Pow24Poly(z);
  return Select(linear, poly, z > V(10.31475f / 12.92f));
}
}  // namespace

template<typename V>
V LinearToSrgbPoly(V z) {
  return LinearToSrgbPolyImpl(z);
}

template float LinearToSrgbPoly(float z);

template PIK_TARGET_NAME::V8x32F LinearToSrgbPoly(PIK_TARGET_NAME::V8x32F z);


const float* NewSrgb8ToLinearTable() {
  float* table = new float[256];
  for (int i = 0; i < 256; ++i) {
    table[i] = Srgb8ToLinearDirect(i);
  }
  return table;
}

const float* Srgb8ToLinearTable() {
  static const float* const kSrgb8ToLinearTable = NewSrgb8ToLinearTable();
  return kSrgb8ToLinearTable;
}

const uint8_t* NewLinearToSrgb8Table(float bias) {
  uint8_t* table = new uint8_t[4096];
  for (int i = 0; i < 4096; ++i) {
    table[i] = std::round(
        std::min(255.0f, std::max(0.0f, LinearToSrgb8Direct(i / 16.0) + bias)));
  }
  return table;
}

const uint8_t* LinearToSrgb8Table() {
  static const uint8_t* const kLinearToSrgb8Table = NewLinearToSrgb8Table(0.0f);
  return kLinearToSrgb8Table;
}

const uint8_t* LinearToSrgb8TablePlusQuarter() {
  static const uint8_t* const kLinearToSrgb8TablePlusQuarter =
      NewLinearToSrgb8Table(0.25);
  return kLinearToSrgb8TablePlusQuarter;
}

const uint8_t* LinearToSrgb8TableMinusQuarter() {
  static const uint8_t* const kLinearToSrgb8TableMinusQuarter =
      NewLinearToSrgb8Table(-0.25);
  return kLinearToSrgb8TableMinusQuarter;
}

ImageF LinearFromSrgb(const ImageB& srgb) {
  PROFILER_FUNC;
  const float* lut = Srgb8ToLinearTable();
  const size_t xsize = srgb.xsize();
  const size_t ysize = srgb.ysize();
  ImageF linear(xsize, ysize);
  for (size_t y = 0; y < ysize; ++y) {
    const uint8_t* const PIK_RESTRICT row = srgb.Row(y);
    float* const PIK_RESTRICT row_linear = linear.Row(y);
    for (size_t x = 0; x < xsize; ++x) {
      row_linear[x] = lut[row[x]];
    }
  }
  return linear;
}

Image3F LinearFromSrgb(const Image3B& srgb) {
  return Image3F(LinearFromSrgb(srgb.plane(0)), LinearFromSrgb(srgb.plane(1)),
                 LinearFromSrgb(srgb.plane(2)));
}

ImageB Srgb8FromLinear(const ImageF& linear) {
  PROFILER_FUNC;
  const size_t xsize = linear.xsize();
  const size_t ysize = linear.ysize();
  ImageB srgb(xsize, ysize);
  for (size_t y = 0; y < ysize; ++y) {
    const float* const PIK_RESTRICT row_linear = linear.Row(y);
    uint8_t* const PIK_RESTRICT row = srgb.Row(y);
    for (size_t x = 0; x < xsize; ++x) {
      row[x] = static_cast<uint8_t>(
          std::round(LinearToSrgb8Direct(row_linear[x])));
    }
  }
  return srgb;
}

Image3B Srgb8FromLinear(const Image3F& linear) {
  return Image3B(Srgb8FromLinear(linear.plane(0)),
                 Srgb8FromLinear(linear.plane(1)),
                 Srgb8FromLinear(linear.plane(2)));
}

ImageF SrgbFFromLinear(const ImageF& linear) {
  PROFILER_FUNC;
  const size_t xsize = linear.xsize();
  const size_t ysize = linear.ysize();
  ImageF srgb(xsize, ysize);
  for (size_t y = 0; y < ysize; ++y) {
    const float* const PIK_RESTRICT row_linear = linear.Row(y);
    float* const PIK_RESTRICT row = srgb.Row(y);
    for (size_t x = 0; x < xsize; ++x) {
      row[x] = LinearToSrgb8Direct(row_linear[x]);
    }
  }
  return srgb;
}

Image3F SrgbFFromLinear(const Image3F& linear) {
  return Image3F(SrgbFFromLinear(linear.plane(0)),
                 SrgbFFromLinear(linear.plane(1)),
                 SrgbFFromLinear(linear.plane(2)));
}

}  // namespace pik
