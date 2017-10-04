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

#include "opsin_inverse.h"

#include <array>

#include "gamma_correct.h"
#include "simd/simd.h"

namespace pik {

Image3B OpsinDynamicsInverse(const Image3F& opsin) {
  Image3B srgb(opsin.xsize(), opsin.ysize());
  for (size_t y = 0; y < opsin.ysize(); ++y) {
    auto row_in = opsin.Row(y);
    auto row_out = srgb.Row(y);
    using namespace SIMD_NAMESPACE;
    using V = vec<float>;
    constexpr size_t N = NumLanes<V>();
    for (size_t x = 0; x < opsin.xsize(); x += N) {
      V r, g, b;
      XybToRgb(load(V(), row_in[0] + x), load(V(), row_in[1] + x),
               load(V(), row_in[2] + x), &r, &g, &b);
      const uint8_t u8 = 0;
      // TODO(janwas): 8-bit precision would suffice.
      const auto u8_r = convert_to(u8, i32_from_f32(LinearToSrgbPoly(r)));
      const auto u8_g = convert_to(u8, i32_from_f32(LinearToSrgbPoly(g)));
      const auto u8_b = convert_to(u8, i32_from_f32(LinearToSrgbPoly(b)));

      store(u8_r, &row_out[0][x]);
      store(u8_g, &row_out[1][x]);
      store(u8_b, &row_out[2][x]);
    }
  }
  return srgb;
}

Image3F LinearFromOpsin(const Image3F& opsin) {
  Image3F srgb(opsin.xsize(), opsin.ysize());
  using namespace SIMD_NAMESPACE;
  using V = vec<float>;
  constexpr size_t N = NumLanes<V>();
  for (size_t y = 0; y < opsin.ysize(); ++y) {
    auto row_in = opsin.Row(y);
    auto row_out = srgb.Row(y);
    for (size_t x = 0; x < opsin.xsize(); x += N) {
      const V vx = load(V(), row_in[0] + x);
      const V vy = load(V(), row_in[1] + x);
      const V vb = load(V(), row_in[2] + x);
      V r, g, b;
      XybToRgb(vx, vy, vb, &r, &g, &b);
      store(r, row_out[0] + x);
      store(g, row_out[1] + x);
      store(b, row_out[2] + x);
    }
  }
  return srgb;
}

}  // namespace pik
