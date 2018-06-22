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

// For U16/F32, or when dithering is disabled (cparam or lack of SIMD).
struct Dither_None {
  using D = SIMD_NAMESPACE::Full<float>;
  using V = D::V;

  static V Init(size_t) { return undefined(D()); }
  static V Eval(const V srgb, const V dither) { return srgb; }
  static V Toggle(const V dither) { return dither; }
};

// Dithering (2x2) helps for larger distances but not at 1 or below.
// Only possible with actual SIMD (requires combine_shift_right_bytes).
#if SIMD_TARGET_VALUE != SIMD_NONE
struct Dither_2x2 {
  using D = SIMD_NAMESPACE::Full<float>;
  using V = D::V;

  static V Init(size_t y) {
    // First row of a 2x2 dither matrix:  -+ -+ .. -+
    SIMD_ALIGN constexpr float lanes[4] = {-0.25f, +0.25f, -0.25f, +0.25f};
    auto dither = load_dup128(D(), lanes);
    if (y & 1) {
      dither = Toggle(dither);
    }
    return dither;
  }

  static V Eval(const V srgb, const V dither) {
    return srgb + dither;
  }

  static V Toggle(const V dither) {
    // Flips lane signs by rotating the vector blocks by one lane.
    return SIMD_NAMESPACE::combine_shift_right_bytes<4>(dither, dither);
  }
};
#else
using Dither_2x2 = Dither_None;
#endif

template <class Dither>
struct LinearToSRGB_U8 {
  using D = SIMD_NAMESPACE::Full<float>;
  using V = D::V;

  static V ExtraArg(size_t y) { return Dither::Init(y); }
  static V UpdateExtraArg(const V v) { return Dither::Toggle(v); }

  PIK_INLINE void operator()(const V linear_r, const V linear_g,
                             const V linear_b, const V dither,
                             uint8_t* PIK_RESTRICT out_srgb_r,
                             uint8_t* PIK_RESTRICT out_srgb_g,
                             uint8_t* PIK_RESTRICT out_srgb_b) const {
    using namespace SIMD_NAMESPACE;
    constexpr D d;

    // The convert_to below take care of clamping.
    V srgb_r, srgb_g, srgb_b;
    LinearToSrgb8PolyWithoutClamp(linear_r, linear_g, linear_b, &srgb_r,
                                  &srgb_g, &srgb_b);

    // Quarter-vectors.
    constexpr Part<uint8_t, d.N> d8;
    const auto srgb_r8 =
        convert_to(d8, nearest_int(Dither::Eval(srgb_r, dither)));
    const auto srgb_g8 =
        convert_to(d8, nearest_int(Dither::Eval(srgb_g, dither)));
    const auto srgb_b8 =
        convert_to(d8, nearest_int(Dither::Eval(srgb_b, dither)));

    store(srgb_r8, d8, out_srgb_r);
    store(srgb_g8, d8, out_srgb_g);
    store(srgb_b8, d8, out_srgb_b);
  }
};

// Same as U8, but multiplies result by 257 to expand to 16-bit.
struct LinearToSRGB_U16 {
  using D = SIMD_NAMESPACE::Full<float>;
  using V = D::V;

  static V ExtraArg(size_t) { return set1(D(), kXybCenter[3]); }
  static V UpdateExtraArg(const V v) { return v; }

  // "mul_srgb" is 257.
  PIK_INLINE void operator()(const V linear_r, const V linear_g,
                             const V linear_b, const V mul_srgb,
                             uint16_t* PIK_RESTRICT out_srgb_r,
                             uint16_t* PIK_RESTRICT out_srgb_g,
                             uint16_t* PIK_RESTRICT out_srgb_b) const {
    using namespace SIMD_NAMESPACE;
    constexpr D d;

    // The convert_to below take care of clamping.
    V srgb_r, srgb_g, srgb_b;
    LinearToSrgb8PolyWithoutClamp(linear_r, linear_g, linear_b, &srgb_r,
                                  &srgb_g, &srgb_b);

    // Half-vectors.
    constexpr Part<uint16_t, d.N> d16;
    const auto srgb_r16 = convert_to(d16, nearest_int(srgb_r * mul_srgb));
    const auto srgb_g16 = convert_to(d16, nearest_int(srgb_g * mul_srgb));
    const auto srgb_b16 = convert_to(d16, nearest_int(srgb_b * mul_srgb));

    store(srgb_r16, d16, out_srgb_r);
    store(srgb_g16, d16, out_srgb_g);
    store(srgb_b16, d16, out_srgb_b);
  }
};

struct LinearToSRGB_F32 {
  using D = SIMD_NAMESPACE::Full<float>;
  using V = D::V;

  static V ExtraArg(size_t) { return undefined(D()); }
  static V UpdateExtraArg(const V v) { return v; }

  PIK_INLINE void operator()(const V linear_r, const V linear_g,
                             const V linear_b, const V unused,
                             float* PIK_RESTRICT out_srgb_r,
                             float* PIK_RESTRICT out_srgb_g,
                             float* PIK_RESTRICT out_srgb_b) const {
    using namespace SIMD_NAMESPACE;
    D d;

    V srgb_r, srgb_g, srgb_b;
    LinearToSrgb8Poly(d, linear_r, linear_g, linear_b, &srgb_r, &srgb_g,
                      &srgb_b);

    store(srgb_r, d, out_srgb_r);
    store(srgb_g, d, out_srgb_g);
    store(srgb_b, d, out_srgb_b);
  }
};

// Called via TileFlow (matches TFFunc signature). Used for U16 and F32.
template <class LinearToSRGB, typename T>
PIK_INLINE void CenteredOpsinToSrgbFunc(
    const void*, const ConstImageViewF* PIK_RESTRICT linear,
    const OutputRegion& output_region,
    const MutableImageViewF* PIK_RESTRICT srgb) {
  using namespace SIMD_NAMESPACE;
  const Full<float> d;

  const auto center_x = set1(d, kXybCenter[0]);
  const auto center_y = set1(d, kXybCenter[1]);
  const auto center_b = set1(d, kXybCenter[2]);
  // dither for U8; 257 for U16; unused for F32.
  auto extra_arg = LinearToSRGB::ExtraArg(output_region.y);

  for (uint32_t y = 0; y < output_region.ysize; ++y) {
    // Faster than adding via ByteOffset at end of loop.
    const float* PIK_RESTRICT row_linear_x = linear[0].ConstRow(y);
    const float* PIK_RESTRICT row_linear_y = linear[1].ConstRow(y);
    const float* PIK_RESTRICT row_linear_b = linear[2].ConstRow(y);

    T* PIK_RESTRICT row_srgb_r = reinterpret_cast<T*>(srgb[0].Row(y));
    T* PIK_RESTRICT row_srgb_g = reinterpret_cast<T*>(srgb[1].Row(y));
    T* PIK_RESTRICT row_srgb_b = reinterpret_cast<T*>(srgb[2].Row(y));

    for (uint32_t x = 0; x < output_region.xsize; x += d.N) {
      const auto in_linear_x = load(d, row_linear_x + x) + center_x;
      const auto in_linear_y = load(d, row_linear_y + x) + center_y;
      const auto in_linear_b = load(d, row_linear_b + x) + center_b;
      decltype(d)::V linear_r, linear_g, linear_b;
      XybToRgb(d, in_linear_x, in_linear_y, in_linear_b, inverse_matrix,
               &linear_r, &linear_g, &linear_b);

      LinearToSRGB()(linear_r, linear_g, linear_b, extra_arg, row_srgb_r + x,
                     row_srgb_g + x, row_srgb_b + x);
    }

    extra_arg = LinearToSRGB::UpdateExtraArg(extra_arg);
  }
}

// TODO(janwas): available for merging into another TF graph if possible.
template <class LinearToSRGB, typename T>
void CenteredOpsinToSrgbT_TF(const Image3F& opsin, ThreadPool* pool,
                             Image3<T>* srgb) {
  PROFILER_FUNC;
  const size_t xsize = opsin.xsize();
  const size_t ysize = opsin.ysize();
  *srgb = Image3<T>(xsize, ysize);

  TFBuilder builder;
  TFNode* src_opsin = builder.AddSource("src_opsin", 3, TFType::kF32);
  builder.SetSource(src_opsin, &opsin);

  const TFType type = TFTypeUtils::FromT(T());
  const TFFunc func = &CenteredOpsinToSrgbFunc<LinearToSRGB, T>;
  TFNode* sink = builder.Add("opsin->srgb", Borders(), Scale(), {src_opsin}, 3,
                             type, func);
  builder.SetSink(sink, srgb);

  const auto graph =
      builder.Finalize(ImageSize::Make(xsize, ysize), ImageSize{64, 64}, pool);
  graph->Run();
}

template <class LinearToSRGB, typename T>
void CenteredOpsinToSrgbT(const Image3F& opsin, ThreadPool* pool,
                          Image3<T>* srgb) {
  PROFILER_FUNC;
  const size_t xsize = opsin.xsize();
  const size_t ysize = opsin.ysize();
  *srgb = Image3<T>(xsize, ysize);

  using namespace SIMD_NAMESPACE;
  const Full<float> d;

  const auto center_x = set1(d, kXybCenter[0]);
  const auto center_y = set1(d, kXybCenter[1]);
  const auto center_b = set1(d, kXybCenter[2]);

  pool->Run(0, ysize, [&](const int task, const int thread) {
    const size_t y = task;
    // dither for U8; 257 for U16; unused for F32.
    auto extra_arg = LinearToSRGB::ExtraArg(y);

    // Faster than adding via ByteOffset at end of loop.
    const float* PIK_RESTRICT row_linear_x = opsin.ConstPlaneRow(0, y);
    const float* PIK_RESTRICT row_linear_y = opsin.ConstPlaneRow(1, y);
    const float* PIK_RESTRICT row_linear_b = opsin.ConstPlaneRow(2, y);

    T* PIK_RESTRICT row_srgb_r = reinterpret_cast<T*>(srgb->PlaneRow(0, y));
    T* PIK_RESTRICT row_srgb_g = reinterpret_cast<T*>(srgb->PlaneRow(1, y));
    T* PIK_RESTRICT row_srgb_b = reinterpret_cast<T*>(srgb->PlaneRow(2, y));

    for (size_t x = 0; x < xsize; x += d.N) {
      const auto in_linear_x = load(d, row_linear_x + x) + center_x;
      const auto in_linear_y = load(d, row_linear_y + x) + center_y;
      const auto in_linear_b = load(d, row_linear_b + x) + center_b;
      decltype(d)::V linear_r, linear_g, linear_b;
      XybToRgb(d, in_linear_x, in_linear_y, in_linear_b, inverse_matrix,
               &linear_r, &linear_g, &linear_b);

      LinearToSRGB()(linear_r, linear_g, linear_b, extra_arg, row_srgb_r + x,
                     row_srgb_g + x, row_srgb_b + x);
    }
  });
}

}  // namespace

void CenteredOpsinToSrgb(const Image3F& opsin, const bool dither,
                         ThreadPool* pool, Image3B* srgb) {
  if (dither) {
    CenteredOpsinToSrgbT<LinearToSRGB_U8<Dither_2x2>>(opsin, pool, srgb);
  } else {
    CenteredOpsinToSrgbT<LinearToSRGB_U8<Dither_None>>(opsin, pool, srgb);
  }
}

void CenteredOpsinToSrgb(const Image3F& opsin, const bool dither,
                         ThreadPool* pool, Image3U* srgb) {
  CenteredOpsinToSrgbT<LinearToSRGB_U16>(opsin, pool, srgb);
}
void CenteredOpsinToSrgb(const Image3F& opsin, const bool dither,
                         ThreadPool* pool, Image3F* srgb) {
  CenteredOpsinToSrgbT<LinearToSRGB_F32>(opsin, pool, srgb);
}

Image3B OpsinDynamicsInverse(const Image3F& opsin) {
  using namespace SIMD_NAMESPACE;
  constexpr Full<float> d;
  using V = Full<float>::V;

  Image3B srgb(opsin.xsize(), opsin.ysize());
  for (size_t y = 0; y < opsin.ysize(); ++y) {
    const float* PIK_RESTRICT row_xyb0 = opsin.PlaneRow(0, y);
    const float* PIK_RESTRICT row_xyb1 = opsin.PlaneRow(1, y);
    const float* PIK_RESTRICT row_xyb2 = opsin.PlaneRow(2, y);
    uint8_t* PIK_RESTRICT row_srgb0 = srgb.PlaneRow(0, y);
    uint8_t* PIK_RESTRICT row_srgb1 = srgb.PlaneRow(1, y);
    uint8_t* PIK_RESTRICT row_srgb2 = srgb.PlaneRow(2, y);

    for (size_t x = 0; x < opsin.xsize(); x += d.N) {
      V r, g, b;
      XybToRgb(d, load(d, row_xyb0 + x), load(d, row_xyb1 + x),
               load(d, row_xyb2 + x), inverse_matrix, &r, &g, &b);
      constexpr Part<uint8_t, d.N> d8;
      const auto u8_r = convert_to(d8, nearest_int(LinearToSrgb8Poly(d, r)));
      const auto u8_g = convert_to(d8, nearest_int(LinearToSrgb8Poly(d, g)));
      const auto u8_b = convert_to(d8, nearest_int(LinearToSrgb8Poly(d, b)));
      store(u8_r, d8, &row_srgb0[x]);
      store(u8_g, d8, &row_srgb1[x]);
      store(u8_b, d8, &row_srgb2[x]);
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
    const float* PIK_RESTRICT row_xyb0 = opsin.PlaneRow(0, y);
    const float* PIK_RESTRICT row_xyb1 = opsin.PlaneRow(1, y);
    const float* PIK_RESTRICT row_xyb2 = opsin.PlaneRow(2, y);
    float* PIK_RESTRICT row_srgb0 = srgb.PlaneRow(0, y);
    float* PIK_RESTRICT row_srgb1 = srgb.PlaneRow(1, y);
    float* PIK_RESTRICT row_srgb2 = srgb.PlaneRow(2, y);

    for (size_t x = 0; x < opsin.xsize(); x += d.N) {
      const auto vx = load(d, row_xyb0 + x);
      const auto vy = load(d, row_xyb1 + x);
      const auto vb = load(d, row_xyb2 + x);
      V r, g, b;
      XybToRgb(d, vx, vy, vb, inverse_matrix, &r, &g, &b);
      store(r, d, row_srgb0 + x);
      store(g, d, row_srgb1 + x);
      store(b, d, row_srgb2 + x);
    }
  }
  return srgb;
}

}  // namespace pik
