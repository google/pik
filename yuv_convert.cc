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

#include "yuv_convert.h"

#include <type_traits>
#include <stdint.h>

#include "gamma_correct.h"

namespace pik {

template <typename T> static PIK_INLINE T Clamp(T a, T b, T val) {
  return std::max(a, std::min(b, val));
}

// Requires b > 0
static PIK_INLINE int DivRound(int64_t a, int64_t b) {
  const int64_t bias = a < 0 ? -(b >> 1) : (b >> 1);
  return (a + bias) / b;
}

// Constants defining the Rec 709 YUV <-> RGB transforms, see
// https://en.wikipedia.org/wiki/YCbCr
constexpr float kKr = 0.2126f;
constexpr float kKg = 0.7152f;
constexpr float kKb = 0.0722f;
constexpr float kKrC = 1.0f - kKr;
constexpr float kKbC = 1.0f - kKb;

//
// YUV Rec 709 -> RGB conversion
//

// YUV Rec 709 to RGB transform matrix.
// Before applying this linear transform, the (16, 128, 128) offset vector must
// be subtracted from the input (Y, U, V) vector.
// The output vector components are in the [0.0, 1.0] range.
constexpr float kYUVRec709ToRGB[9] = {
  1.0f / 219.0f, 0.0f, kKrC / 112.0f,
  1.0f / 219.0f, -kKbC * kKb / (112.0f * kKg), -kKrC * kKr / (112.0f * kKg),
  1.0f / 219.0f, kKbC / 112.0f, 0.0f,
};

// The same matrix as above, each entry expressed as the ratio of two 64-bit
// integers.
constexpr int64_t kYUVRec709ToRGBNum[9] = {
  667520000LL,          0LL,  1027745976LL,
  667520000LL, -122251567LL,  -305507263LL,
  667520000LL, 1211001672LL,           0LL,
};
constexpr int64_t kYUVRec709ToRGBDenom = 146186880000LL;

// Converts the input Rec 709 (y, u, v) pixel to RGB with range [0.0, 1.0].
void YUVRec709PixelToRGB(uint8_t y, uint8_t u, uint8_t v,
                         float* PIK_RESTRICT r,
                         float* PIK_RESTRICT g,
                         float* PIK_RESTRICT b) {
  *r = Clamp(0.0f, 1.0f,
             (kYUVRec709ToRGB[0] * (y - 16.0f) +
              kYUVRec709ToRGB[2] * (v - 128.0f)));
  *g = Clamp(0.0f, 1.0f,
             (kYUVRec709ToRGB[3] * (y - 16.0f) +
              kYUVRec709ToRGB[4] * (u - 128.0f) +
              kYUVRec709ToRGB[5] * (v - 128.0f)));
  *b = Clamp(0.0f, 1.0f,
             (kYUVRec709ToRGB[6] * (y - 16.0f) +
              kYUVRec709ToRGB[7] * (u - 128.0f)));
}

// Same as above, but the output R, G, B values are either in the range 0..255
// (for T = uint8_t) or 0..65535 (for T = uint16_t).
template <typename T>
void YUVRec709PixelToRGB(uint8_t y, uint8_t u, uint8_t v,
                         T* PIK_RESTRICT r,
                         T* PIK_RESTRICT g,
                         T* PIK_RESTRICT b) {
  static_assert(std::is_unsigned<T>::value, "T must be unsigned");
  const int maxv = (1 << (8 * sizeof(T))) - 1;
  const int64_t r_num = (kYUVRec709ToRGBNum[0] * (y - 16) +
                         kYUVRec709ToRGBNum[2] * (v - 128));
  const int64_t g_num = (kYUVRec709ToRGBNum[3] * (y - 16) +
                         kYUVRec709ToRGBNum[4] * (u - 128) +
                         kYUVRec709ToRGBNum[5] * (v - 128));
  const int64_t b_num = (kYUVRec709ToRGBNum[6] * (y - 16) +
                         kYUVRec709ToRGBNum[7] * (u - 128));
  *r = Clamp(0, maxv, DivRound(maxv * r_num, kYUVRec709ToRGBDenom));
  *g = Clamp(0, maxv, DivRound(maxv * g_num, kYUVRec709ToRGBDenom));
  *b = Clamp(0, maxv, DivRound(maxv * b_num, kYUVRec709ToRGBDenom));
};

template <typename T>
void YUVRec709ImageToRGB(const Image3B& yuv, Image3<T>* rgb) {
  for (int y = 0; y < yuv.ysize(); ++y) {
    auto row_yuv = yuv.Row(y);
    auto row_rgb = rgb->Row(y);
    for (int x = 0; x < yuv.xsize(); ++x) {
      YUVRec709PixelToRGB(row_yuv[0][x], row_yuv[1][x], row_yuv[2][x],
                          &row_rgb[0][x], &row_rgb[1][x], &row_rgb[2][x]);
    }
  }
}

Image3B RGB8ImageFromYUVRec709(const Image3B& yuv) {
  Image3B rgb(yuv.xsize(), yuv.ysize());
  YUVRec709ImageToRGB(yuv, &rgb);
  return rgb;
}

Image3U RGB16ImageFromYUVRec709(const Image3B& yuv) {
  Image3U rgb(yuv.xsize(), yuv.ysize());
  YUVRec709ImageToRGB(yuv, &rgb);
  return rgb;
}

Image3F RGBLinearImageFromYUVRec709(const Image3B& yuv) {
  Image3F linear(yuv.xsize(), yuv.ysize());
  for (int y = 0; y < yuv.ysize(); ++y) {
    const auto row_yuv = yuv.ConstRow(y);
    auto row_linear = linear.Row(y);
    for (int x = 0; x < yuv.xsize(); ++x) {
      float r, g, b;
      YUVRec709PixelToRGB(row_yuv[0][x], row_yuv[1][x], row_yuv[2][x],
                          &r, &g, &b);
      row_linear[0][x] = Srgb8ToLinearDirect(r * 255.0);
      row_linear[1][x] = Srgb8ToLinearDirect(g * 255.0);
      row_linear[2][x] = Srgb8ToLinearDirect(b * 255.0);
    }
  }
  return linear;
}

//
// RGB -> YUV Rec 709 conversion
//

// RGB to YUV Rec 709 to transform matrix.
// After applying this linear transform, the (16, 128, 128) offset vector must
// be added to the output (Y, U, V) vector.
// The input vector components are in the [0.0, 1.0] range.
// The Y output component is in the range [0.0, 219.0] (or [16.0, 235.0] after
// adding the offset), and the U, V output components are in the [-112.0, 112.0]
// range (or [16.0, 240.0] after adding the offset).
constexpr float kRGBToYUVRec709[9] = {
  219.0f * kKr, 219.0f * kKg, 219.0f * kKb,
  -112 * kKr / kKbC, -112.0f * kKg / kKbC, 112.0f,
  112.0f, -112.0f * kKg / kKrC, -112.0f * kKb / kKrC,
};

// The same matrix as above, each entry expressed as the ratio of two 64-bit
// integers.
constexpr int64_t kRGBToYUVRec709Num[9] = {
   4251744579171LL, 14303140747992LL,  1443913257837LL,
  -2343617360000LL, -7884078720000LL, 10227696080000LL,
  10227696080000LL, -9289875840000LL,  -937820240000LL,
};
constexpr int64_t kRGBToYUVRec709Denom = 91318715000LL;

template <typename T>
void RGBPixelToYUVRec709(T r, T g, T b,
                         uint8_t* PIK_RESTRICT y,
                         uint8_t* PIK_RESTRICT u,
                         uint8_t* PIK_RESTRICT v) {
  static_assert(std::is_unsigned<T>::value, "T must be unsigned");
  const int maxv = (1 << (8 * sizeof(T))) - 1;
  const int64_t y_num = (kRGBToYUVRec709Num[0] * r +
                         kRGBToYUVRec709Num[1] * g +
                         kRGBToYUVRec709Num[2] * b);
  const int64_t u_num = (kRGBToYUVRec709Num[3] * r +
                         kRGBToYUVRec709Num[4] * g +
                         kRGBToYUVRec709Num[5] * b);
  const int64_t v_num = (kRGBToYUVRec709Num[6] * r +
                         kRGBToYUVRec709Num[7] * g +
                         kRGBToYUVRec709Num[8] * b);
  const int64_t denom = maxv * kRGBToYUVRec709Denom;
  *y = DivRound(y_num, denom) + 16;
  *u = DivRound(u_num, denom) + 128;
  *v = DivRound(v_num, denom) + 128;
}

template <typename T>
Image3B RGBImageToYUVRec709(const Image3<T>& rgb) {
  Image3B yuv(rgb.xsize(), rgb.ysize());
  for (int y = 0; y < rgb.ysize(); ++y) {
    const auto row_rgb = rgb.ConstRow(y);
    auto row_yuv = yuv.Row(y);
    for (int x = 0; x < rgb.xsize(); ++x) {
      RGBPixelToYUVRec709(row_rgb[0][x], row_rgb[1][x], row_rgb[2][x],
                          &row_yuv[0][x], &row_yuv[1][x], &row_yuv[2][x]);
    }
  }
  return yuv;
}

Image3B YUVRec709ImageFromRGB8(const Image3B& rgb) {
  return RGBImageToYUVRec709(rgb);
}

Image3B YUVRec709ImageFromRGB16(const Image3U& rgb) {
  return RGBImageToYUVRec709(rgb);
}

}  // namespace pik
