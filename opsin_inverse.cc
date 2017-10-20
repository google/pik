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

void CenteredOpsinToSrgb(const Image3F& opsin, Image3B* srgb) {
  using namespace SIMD_NAMESPACE;
  const Full<float> d;
  const auto lut_scale = set1(d, 16.0f);
  const uint8_t* PIK_RESTRICT lut_plus = LinearToSrgb8TablePlusQuarter();
  const uint8_t* PIK_RESTRICT lut_minus = LinearToSrgb8TableMinusQuarter();
  *srgb = Image3B(opsin.xsize(), opsin.ysize());
  for (int y = 0; y < srgb->ysize(); ++y) {
    auto row_in = opsin.Row(y);
    auto row_out = srgb->Row(y);
    for (int x = 0; x < srgb->xsize(); x += d.N) {
      SIMD_ALIGN int buf[3][d.N];
      const auto valx = load(d, &row_in[0][x]) + set1(d, kXybCenter[0]);
      const auto valy = load(d, &row_in[1][x]) + set1(d, kXybCenter[1]);
      const auto valb = load(d, &row_in[2][x]) + set1(d, kXybCenter[2]);
      Full<float>::V out_r, out_g, out_b;
      XybToRgb(d, valx, valy, valb, &out_r, &out_g, &out_b);
      const Full<int32_t> di;
      store(i32_from_f32(out_r * lut_scale), di, &buf[0][0]);
      store(i32_from_f32(out_g * lut_scale), di, &buf[1][0]);
      store(i32_from_f32(out_b * lut_scale), di, &buf[2][0]);
      const int xy = x + y;
      for (int k = 0; k < d.N; ++k) {
        const uint8_t* PIK_RESTRICT lut = (xy + k) % 2 ? lut_plus : lut_minus;
        row_out[0][x + k] = lut[buf[0][k]];
        row_out[1][x + k] = lut[buf[1][k]];
        row_out[2][x + k] = lut[buf[2][k]];
      }
    }
  }
}

void CenteredOpsinToSrgb(const Image3F& opsin, Image3U* srgb) {
  using namespace SIMD_NAMESPACE;
  const Full<float> d;
  const auto scale_to_16bit = set1(d, 257.0f);
  *srgb = Image3U(opsin.xsize(), opsin.ysize());
  for (int y = 0; y < srgb->ysize(); ++y) {
    auto row_in = opsin.Row(y);
    auto row_out = srgb->Row(y);
    for (int x = 0; x < srgb->xsize(); x += d.N) {
      const auto valx = load(d, &row_in[0][x]) + set1(d, kXybCenter[0]);
      const auto valy = load(d, &row_in[1][x]) + set1(d, kXybCenter[1]);
      const auto valb = load(d, &row_in[2][x]) + set1(d, kXybCenter[2]);
      Full<float>::V out_r, out_g, out_b;
      XybToRgb(d, valx, valy, valb, &out_r, &out_g, &out_b);
      out_r = LinearToSrgbPoly(d, out_r) * scale_to_16bit;
      out_g = LinearToSrgbPoly(d, out_g) * scale_to_16bit;
      out_b = LinearToSrgbPoly(d, out_b) * scale_to_16bit;
      // Half-vectors of half-width lanes.
      const Part<uint16_t, d.N> d16;
      const auto u16_r = convert_to(d16, i32_from_f32(out_r));
      const auto u16_g = convert_to(d16, i32_from_f32(out_g));
      const auto u16_b = convert_to(d16, i32_from_f32(out_b));
      store(u16_r, d16, &row_out[0][x]);
      store(u16_g, d16, &row_out[1][x]);
      store(u16_b, d16, &row_out[2][x]);
    }
  }
}

void CenteredOpsinToSrgb(const Image3F& opsin, Image3F* srgb) {
  using namespace SIMD_NAMESPACE;
  const Full<float> d;
  *srgb = Image3F(opsin.xsize(), opsin.ysize());
  for (int y = 0; y < srgb->ysize(); ++y) {
    auto row_in = opsin.Row(y);
    auto row_out = srgb->Row(y);
    for (int x = 0; x < srgb->xsize(); x += d.N) {
      const auto valx = load(d, &row_in[0][x]) + set1(d, kXybCenter[0]);
      const auto valy = load(d, &row_in[1][x]) + set1(d, kXybCenter[1]);
      const auto valb = load(d, &row_in[2][x]) + set1(d, kXybCenter[2]);
      Full<float>::V out_r, out_g, out_b;
      XybToRgb(d, valx, valy, valb, &out_r, &out_g, &out_b);
      store(out_r, d, &row_out[0][x]);
      store(out_g, d, &row_out[1][x]);
      store(out_b, d, &row_out[2][x]);
    }
  }
}

Image3B OpsinDynamicsInverse(const Image3F& opsin) {
  Image3B srgb(opsin.xsize(), opsin.ysize());
  for (size_t y = 0; y < opsin.ysize(); ++y) {
    auto row_in = opsin.Row(y);
    auto row_out = srgb.Row(y);
    using namespace SIMD_NAMESPACE;
    const Full<float> d;
    for (size_t x = 0; x < opsin.xsize(); x += d.N) {
      Full<float>::V r, g, b;
      XybToRgb(d, load(d, row_in[0] + x), load(d, row_in[1] + x),
               load(d, row_in[2] + x), &r, &g, &b);
      const Part<uint8_t, d.N> d8;
      // TODO(janwas): 8-bit precision would suffice.
      const auto u8_r = convert_to(d8, i32_from_f32(LinearToSrgbPoly(d, r)));
      const auto u8_g = convert_to(d8, i32_from_f32(LinearToSrgbPoly(d, g)));
      const auto u8_b = convert_to(d8, i32_from_f32(LinearToSrgbPoly(d, b)));
      store(u8_r, d8, &row_out[0][x]);
      store(u8_g, d8, &row_out[1][x]);
      store(u8_b, d8, &row_out[2][x]);
    }
  }
  return srgb;
}

Image3F LinearFromOpsin(const Image3F& opsin) {
  Image3F srgb(opsin.xsize(), opsin.ysize());
  using namespace SIMD_NAMESPACE;
  const Full<float> d;
  for (size_t y = 0; y < opsin.ysize(); ++y) {
    auto row_in = opsin.Row(y);
    auto row_out = srgb.Row(y);
    for (size_t x = 0; x < opsin.xsize(); x += d.N) {
      const auto vx = load(d, row_in[0] + x);
      const auto vy = load(d, row_in[1] + x);
      const auto vb = load(d, row_in[2] + x);
      Full<float>::V r, g, b;
      XybToRgb(d, vx, vy, vb, &r, &g, &b);
      store(r, d, row_out[0] + x);
      store(g, d, row_out[1] + x);
      store(b, d, row_out[2] + x);
    }
  }
  return srgb;
}

}  // namespace pik
