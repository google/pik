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

#define PROFILER_ENABLED 1
#include "gamma_correct.h"
#include "profiler.h"
#include "tile_flow.h"

namespace pik {
namespace {

SIMD_NAMESPACE::Full<float>::V inverse_matrix[9];

// Called from non-local static initializer for convenience.
int InitInverseMatrix() {
  const SIMD_NAMESPACE::Full<float> d;
  const float* PIK_RESTRICT inverse = GetOpsinAbsorbanceInverseMatrix();
  for (size_t i = 0; i < 9; ++i) {
    inverse_matrix[i] = set1(d, inverse[i]);
  }

  return 0;
}

int dummy = InitInverseMatrix();

struct LinearToSRGB_U8 {
  using D = SIMD_NAMESPACE::Full<float>;
  using V = D::V;

  PIK_INLINE void operator()(const V linear_r, const V linear_g,
                             const V linear_b, const V dither,
                             const V unused_mul,
                             uint8_t* PIK_RESTRICT out_srgb_r,
                             uint8_t* PIK_RESTRICT out_srgb_g,
                             uint8_t* PIK_RESTRICT out_srgb_b) const {
    using namespace SIMD_NAMESPACE;
    D d;
    Full<uint32_t> du;
    Full<int32_t> di;

    // The convert_to below take care of clamping.
    V srgb_r, srgb_g, srgb_b;
    LinearToSrgb8PolyWithoutClamp(linear_r, linear_g, linear_b, &srgb_r,
                                  &srgb_g, &srgb_b);

    // Quarter-vectors.
    const Part<uint8_t, d.N> d8;
    const auto srgb_r8 = convert_to(d8, nearest_int(srgb_r + dither));
    const auto srgb_g8 = convert_to(d8, nearest_int(srgb_g + dither));
    const auto srgb_b8 = convert_to(d8, nearest_int(srgb_b + dither));

    store(srgb_r8, d8, out_srgb_r);
    store(srgb_g8, d8, out_srgb_g);
    store(srgb_b8, d8, out_srgb_b);
  }
};

struct LinearToSRGB_U16 {
  using D = SIMD_NAMESPACE::Full<float>;
  using V = D::V;

  // "mul_srgb" is 257.
  PIK_INLINE void operator()(const V linear_r, const V linear_g,
                             const V linear_b, const V unused_add,
                             const V mul_srgb,
                             uint16_t* PIK_RESTRICT out_srgb_r,
                             uint16_t* PIK_RESTRICT out_srgb_g,
                             uint16_t* PIK_RESTRICT out_srgb_b) const {
    using namespace SIMD_NAMESPACE;
    D d;

    // The convert_to below take care of clamping.
    const auto srgb_r = LinearToSrgb8PolyWithoutClamp(linear_r) * mul_srgb;
    const auto srgb_g = LinearToSrgb8PolyWithoutClamp(linear_g) * mul_srgb;
    const auto srgb_b = LinearToSrgb8PolyWithoutClamp(linear_b) * mul_srgb;

    // Half-vectors.
    const Part<uint16_t, d.N> d16;
    const auto srgb_r16 = convert_to(d16, nearest_int(srgb_r));
    const auto srgb_g16 = convert_to(d16, nearest_int(srgb_g));
    const auto srgb_b16 = convert_to(d16, nearest_int(srgb_b));

    store(srgb_r16, d16, out_srgb_r);
    store(srgb_g16, d16, out_srgb_g);
    store(srgb_b16, d16, out_srgb_b);
  }
};

struct LinearToSRGB_F32 {
  using D = SIMD_NAMESPACE::Full<float>;
  using V = D::V;

  PIK_INLINE void operator()(const V linear_r, const V linear_g,
                             const V linear_b, const V unused_add,
                             const V unused_mul, float* PIK_RESTRICT out_srgb_r,
                             float* PIK_RESTRICT out_srgb_g,
                             float* PIK_RESTRICT out_srgb_b) const {
    using namespace SIMD_NAMESPACE;
    D d;

    const auto srgb_r = LinearToSrgb8PolyWithoutClamp(linear_r);
    const auto srgb_g = LinearToSrgb8PolyWithoutClamp(linear_g);
    const auto srgb_b = LinearToSrgb8PolyWithoutClamp(linear_b);

    store(srgb_r, d, out_srgb_r);
    store(srgb_g, d, out_srgb_g);
    store(srgb_b, d, out_srgb_b);
  }
};

// Called via TileFlow (matches TFFunc signature). Used for U16 and F32.
template <class LinearToSRGB, typename T>
PIK_INLINE void CenteredOpsinToSrgbFunc(const void*,
                                        const ConstImageViewF* linear,
                                        const OutputRegion& output_region,
                                        const MutableImageViewF* srgb) {
  PROFILER_ZONE("|| Opsin->SRGB");
  using namespace SIMD_NAMESPACE;
  const Full<float> d;

  const auto center_xyb = load_dup128(d, kXybCenter);

  // 257 for U16 mul.
  const auto mul_srgb = broadcast<3>(center_xyb);

  for (uint32_t y = 0; y < output_region.ysize; ++y) {
    // Faster than adding via ByteOffset at end of loop.
    const float* PIK_RESTRICT row_linear_x = linear[0].ConstRow(y);
    const float* PIK_RESTRICT row_linear_y = linear[1].ConstRow(y);
    const float* PIK_RESTRICT row_linear_b = linear[2].ConstRow(y);

    T* PIK_RESTRICT row_srgb_r = reinterpret_cast<T*>(srgb[0].Row(y));
    T* PIK_RESTRICT row_srgb_g = reinterpret_cast<T*>(srgb[1].Row(y));
    T* PIK_RESTRICT row_srgb_b = reinterpret_cast<T*>(srgb[2].Row(y));

    for (uint32_t x = 0; x < output_region.xsize; x += d.N) {
      const auto in_linear_x =
          load(d, row_linear_x + x) + broadcast<0>(center_xyb);
      const auto in_linear_y =
          load(d, row_linear_y + x) + broadcast<1>(center_xyb);
      const auto in_linear_b =
          load(d, row_linear_b + x) + broadcast<2>(center_xyb);
      decltype(d)::V linear_r, linear_g, linear_b;
      XybToRgb(d, in_linear_x, in_linear_y, in_linear_b, inverse_matrix,
               &linear_r, &linear_g, &linear_b);

      LinearToSRGB()(linear_r, linear_g, linear_b, setzero(d), mul_srgb,
                     row_srgb_r + x, row_srgb_g + x, row_srgb_b + x);
    }
  }
}

// Flips lane signs by rotating the vector blocks by one lane.
template <class V>
static PIK_INLINE V ToggleDither(const V dither) {
  return SIMD_NAMESPACE::combine_shift_right_bytes<4>(dither, dither);
}

PIK_INLINE void CenteredOpsinToSrgb8Func(const void*,
                                         const ConstImageViewF* opsin,
                                         const OutputRegion& output_region,
                                         const MutableImageViewF* srgb) {
  PROFILER_ZONE("|| Opsin->SRGB8");
  using namespace SIMD_NAMESPACE;
  const Full<float> d;

  const auto center_xyb = load_dup128(d, kXybCenter);

  // First row of a 2x2 dither matrix:  -+ -+ .. -+
  SIMD_ALIGN constexpr float lanes[4] = {-0.25f, +0.25f, -0.25f, +0.25f};
  auto dither = load_dup128(d, lanes);
  if (output_region.y & 1) {
    dither = ToggleDither(dither);
  }

  for (uint32_t y = 0; y < output_region.ysize; ++y) {
    // Faster than adding via ByteOffset at end of loop.
    const float* PIK_RESTRICT row_opsin_x = opsin[0].ConstRow(y);
    const float* PIK_RESTRICT row_opsin_y = opsin[1].ConstRow(y);
    const float* PIK_RESTRICT row_opsin_b = opsin[2].ConstRow(y);

    uint8_t* PIK_RESTRICT row_srgb_r =
        reinterpret_cast<uint8_t*>(srgb[0].Row(y));
    uint8_t* PIK_RESTRICT row_srgb_g =
        reinterpret_cast<uint8_t*>(srgb[1].Row(y));
    uint8_t* PIK_RESTRICT row_srgb_b =
        reinterpret_cast<uint8_t*>(srgb[2].Row(y));

    for (uint32_t x = 0; x < output_region.xsize; x += d.N) {
      const auto opsin_x = load(d, row_opsin_x + x) + broadcast<0>(center_xyb);
      const auto opsin_y = load(d, row_opsin_y + x) + broadcast<1>(center_xyb);
      const auto opsin_b = load(d, row_opsin_b + x) + broadcast<2>(center_xyb);

      decltype(dither) linear_r, linear_g, linear_b;
      XybToRgbWithoutClamp(d, opsin_x, opsin_y, opsin_b, inverse_matrix,
                           &linear_r, &linear_g, &linear_b);

      LinearToSRGB_U8()(linear_r, linear_g, linear_b, dither, dither,
                        row_srgb_r + x, row_srgb_g + x, row_srgb_b + x);
    }

    dither = ToggleDither(dither);
  }
}

template <class LinearToSRGB, typename T>
void CenteredOpsinToSrgbT(const Image3F& opsin, ThreadPool* pool,
                          Image3<T>* srgb) {
  const size_t xsize = opsin.xsize();
  const size_t ysize = opsin.ysize();
  *srgb = Image3<T>(xsize, ysize);

  TFBuilder builder;
  TFNode* src_opsin = builder.AddSource("src_opsin", 3, TFType::kF32);
  builder.SetSource(src_opsin, &opsin);

  const TFType type = TFTypeUtils::FromT(T());
  const TFFunc func = (type == TFType::kU8)
                          ? &CenteredOpsinToSrgb8Func
                          : &CenteredOpsinToSrgbFunc<LinearToSRGB, T>;
  TFNode* sink = builder.Add("opsin->srgb", Borders(), Scale(), {src_opsin}, 3,
                             type, func);
  builder.SetSink(sink, srgb);

  const auto graph =
      builder.Finalize(ImageSize::Make(xsize, ysize), ImageSize{64, 64}, pool);
  graph->Run();
}

}  // namespace

void CenteredOpsinToSrgb(const Image3F& opsin, ThreadPool* pool,
                         Image3B* srgb) {
  CenteredOpsinToSrgbT<LinearToSRGB_U8>(opsin, pool, srgb);
}

void CenteredOpsinToSrgb(const Image3F& opsin, ThreadPool* pool,
                         Image3U* srgb) {
  CenteredOpsinToSrgbT<LinearToSRGB_U16>(opsin, pool, srgb);
}
void CenteredOpsinToSrgb(const Image3F& opsin, ThreadPool* pool,
                         Image3F* srgb) {
  CenteredOpsinToSrgbT<LinearToSRGB_F32>(opsin, pool, srgb);
}

Image3B OpsinDynamicsInverse(const Image3F& opsin) {
  using namespace SIMD_NAMESPACE;
  const Full<float> d;
  using V = Full<float>::V;

  Image3B srgb(opsin.xsize(), opsin.ysize());
  for (size_t y = 0; y < opsin.ysize(); ++y) {
    auto row_in = opsin.Row(y);
    auto row_out = srgb.Row(y);

    for (size_t x = 0; x < opsin.xsize(); x += d.N) {
      V r, g, b;
      XybToRgb(d, load(d, row_in[0] + x), load(d, row_in[1] + x),
               load(d, row_in[2] + x), inverse_matrix, &r, &g, &b);
      const Part<uint8_t, d.N> d8;
      const auto u8_r = convert_to(d8, nearest_int(LinearToSrgb8Poly(d, r)));
      const auto u8_g = convert_to(d8, nearest_int(LinearToSrgb8Poly(d, g)));
      const auto u8_b = convert_to(d8, nearest_int(LinearToSrgb8Poly(d, b)));
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
  using V = Full<float>::V;

  for (size_t y = 0; y < opsin.ysize(); ++y) {
    auto row_in = opsin.Row(y);
    auto row_out = srgb.Row(y);
    for (size_t x = 0; x < opsin.xsize(); x += d.N) {
      const auto vx = load(d, row_in[0] + x);
      const auto vy = load(d, row_in[1] + x);
      const auto vb = load(d, row_in[2] + x);
      V r, g, b;
      XybToRgb(d, vx, vy, vb, inverse_matrix, &r, &g, &b);
      store(r, d, row_out[0] + x);
      store(g, d, row_out[1] + x);
      store(b, d, row_out[2] + x);
    }
  }
  return srgb;
}

}  // namespace pik
