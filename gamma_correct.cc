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

namespace pik {
namespace {
float LinearToSrgb8Poly(float z) {
  if (z <= 0.0) return 0.0;
  if (z >= 255) return 255;
  if (z <= 10.31475 / 12.92) return z * 12.92;
  // The approximation requires double precision to work.
  double r1, r2, r3, r4, r5, r6, num, den;
  z = (z - (255.77) * 0.5) / (254.23 * 0.5);
  r6 = -0.0109341700384508 + 2 * z * 0.00131133595817393;
  r5 = 0.285747961922774 + 2 * z * r6 - 0.00131133595817393;
  r4 = 7.99170966186712 + 2 * z * r5 - r6;
  r3 = 55.9346675991248 + 2 * z * r4 - r5;
  r2 = 192.043847048084 + 2 * z * r3 - r4;
  r1 = 385.048686956293 + 2 * z * r2 - r3;
  num = 241.245857093411 + z * r1 - r2;
  r3 = 0.175381684697298 + 2 * z * 0.0175634104108948;
  r2 = 0.712908802449088 + 2 * z * r3 - 0.0175634104108948;
  r1 = 1.55508391793947 + 2 * z * r2 - r3;
  den = 1 + z * r1 - r2;
  return num / den;
}
}  // namespace

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

ImageU Srgb16FromLinear(const ImageF& linear) {
  PROFILER_FUNC;
  const size_t xsize = linear.xsize();
  const size_t ysize = linear.ysize();
  ImageU srgb(xsize, ysize);
  for (size_t y = 0; y < ysize; ++y) {
    const float* const PIK_RESTRICT row_linear = linear.Row(y);
    uint16_t* const PIK_RESTRICT row = srgb.Row(y);
    for (size_t x = 0; x < xsize; ++x) {
      // Multiply by 257.0 to convert range 0-255.0 into 0-65535.0
      // The Linear to sRGB conversion uses floating point math.
      row[x] = static_cast<uint16_t>(
          std::round(LinearToSrgb8Poly(row_linear[x]) * 257.0));
    }
  }
  return srgb;
}

Image3U Srgb16FromLinear(const Image3F& linear) {
  return Image3U(Srgb16FromLinear(linear.plane(0)),
                 Srgb16FromLinear(linear.plane(1)),
                 Srgb16FromLinear(linear.plane(2)));
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
