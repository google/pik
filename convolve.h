// Copyright 2018 Google Inc. All Rights Reserved.
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

#ifndef CONVOLVE_H_
#define CONVOLVE_H_

// Fast SIMD 2D convolution.

#include <stddef.h>
#include <stdint.h>
#include <cassert>

#include "compiler_specific.h"
#include "data_parallel.h"
#include "image.h"
#include "profiler.h"
#include "simd_helpers.h"
#include "status.h"
#include "tile_flow.h"

namespace pik {
namespace SIMD_NAMESPACE {

// Non-vectorized implementations for validation.
class Slow {
 public:
  template <int kRadius, class Wrap, class ImageOrView>
  static inline void ConvolveSeparable(const ImageOrView& in,
                                       const size_t xsize, const size_t ysize,
                                       const float* PIK_RESTRICT horz_weights,
                                       const float* PIK_RESTRICT vert_weights,
                                       ImageF* out) {
    for (size_t y = 0; y < ysize; ++y) {
      float* const PIK_RESTRICT row_out = out->Row(y);
      for (size_t x = 0; x < xsize; ++x) {
        row_out[x] = ConvolveSeparablePixel<kRadius, Wrap>(
            in, xsize, ysize, x, y, horz_weights, vert_weights);
      }
    }
  }

  // Weights are interpreted differently for the 3x3 case; our scalar code
  // must match that behavior. Index = Manhattan distance from center: 0 =
  // center, 1 = 4-neighborhood, 2 = diagonal neighborhood.
  template <class Wrap, class ImageOrView>
  static inline void Convolve3x3(const ImageOrView& in, const size_t xsize,
                                 const size_t ysize, const float weights[3],
                                 ImageF* out) {
    PIK_CHECK(xsize == out->xsize() && ysize == out->ysize());
    for (size_t y = 0; y < ysize; ++y) {
      const float* const PIK_RESTRICT row_t = in.ConstRow(Wrap()(y - 1, ysize));
      const float* const PIK_RESTRICT row_m = in.ConstRow(y);
      const float* const PIK_RESTRICT row_b = in.ConstRow(Wrap()(y + 1, ysize));
      float* const PIK_RESTRICT row_out = out->Row(y);

      for (size_t x = 0; x < xsize; ++x) {
        float mul = row_m[x] * weights[0];
        const int64_t xm1 = Wrap()(x - 1, xsize);
        const int64_t xp1 = Wrap()(x + 1, xsize);
        const float tl = row_t[xm1];
        const float ml = row_m[xm1];
        const float bl = row_b[xm1];
        const float tr = row_t[xp1];
        const float mr = row_m[xp1];
        const float br = row_b[xp1];
        mul += (row_t[x] + row_b[x] + ml + mr) * weights[1];
        mul += (tl + tr + bl + br) * weights[2];
        row_out[x] = static_cast<float>(mul);
      }
    }
  }

  // Slow N*R^2 algorithm in case weights are not separable, but avoids
  // bounds-checking overhead for interior pixels. Weights are the lower-right
  // quadrant of the kernel and need not be pre-normalized.
  template <int64_t kRadius, class Wrap, class ImageOrView>
  static inline void ConvolveSymmetric(
      const ImageOrView& in, const size_t xsize, const size_t ysize,
      const float (&weights)[(kRadius + 1) * (kRadius + 1)], ImageF* out) {
    // Normalize all weights (expand quadrant into entire kernel)
    float sum = 0.0f;
    for (int64_t ky = -kRadius; ky <= kRadius; ky++) {
      const int64_t wy = std::abs(ky);
      for (int64_t kx = -kRadius; kx <= kRadius; kx++) {
        const int64_t wx = std::abs(kx);
        sum += weights[wy * (kRadius + 1) + wx];
      }
    }
    const float mul = 1.0 / sum;
    float normalized[(kRadius + 1) * (kRadius + 1)];
    for (size_t i = 0; i < (kRadius + 1) * (kRadius + 1); ++i) {
      normalized[i] = weights[i] * mul;
    }

    int64_t iy = 0;
    for (; iy < kRadius; iy++) {
      ConvolveSymmetricRow<kRadius, Wrap>(in, xsize, ysize, iy, normalized,
                                          out);
    }
    for (; iy < ysize - kRadius; iy++) {
      ConvolveSymmetricRow<kRadius, WrapUnchanged>(in, xsize, ysize, iy,
                                                   normalized, out);
    }
    for (; iy < ysize; iy++) {
      ConvolveSymmetricRow<kRadius, Wrap>(in, xsize, ysize, iy, normalized,
                                          out);
    }
  }

 private:
  template <int kRadius, class Wrap, class ImageOrView>
  static inline float ConvolveSeparablePixel(
      const ImageOrView& in, const size_t xsize, const size_t ysize,
      const size_t x, const size_t y, const float* PIK_RESTRICT horz_weights,
      const float* PIK_RESTRICT vert_weights) {
    float mul = 0.0f;
    for (int dy = -kRadius; dy <= kRadius; ++dy) {
      const float wy = vert_weights[std::abs(dy)];
      const size_t sy = Wrap()(y + dy, ysize);
      PIK_CHECK(sy < ysize);
      const float* const PIK_RESTRICT row = in.ConstRow(sy);
      for (int dx = -kRadius; dx <= kRadius; ++dx) {
        const float wx = horz_weights[std::abs(dx)];
        const size_t sx = Wrap()(x + dx, xsize);
        PIK_CHECK(sx < xsize);
        mul += row[sx] * wx * wy;
      }
    }
    return mul;
  }

  // "weights": see above.
  template <class Wrap>
  static inline float Convolve3x3Pixel(const ImageF& in, const size_t xsize,
                                       const size_t ysize, const size_t x,
                                       const size_t y, const float weights[3]) {
    const float* const PIK_RESTRICT row_t = in.ConstRow(Wrap()(y - 1, ysize));
    const float* const PIK_RESTRICT row_m = in.ConstRow(y);
    const float* const PIK_RESTRICT row_b = in.ConstRow(Wrap()(y + 1, ysize));

    float mul = row_m[x] * weights[0];
    const int64_t xm1 = Wrap()(x - 1, xsize);
    const int64_t xp1 = Wrap()(x + 1, xsize);
    const float tl = row_t[xm1];
    const float ml = row_m[xm1];
    const float bl = row_b[xm1];
    const float tr = row_t[xp1];
    const float mr = row_m[xp1];
    const float br = row_b[xp1];
    mul += (row_t[x] + row_b[x] + ml + mr) * weights[1];
    mul += (tl + tr + bl + br) * weights[2];
    return static_cast<float>(mul);
  }

  // Slow N*R^2 algorithm in case weights are not separable.
  template <int64_t kRadius, class CheckX, class WrapY, class ImageOrView>
  static inline float ConvolveSymmetricPixel(
      const ImageOrView& in, const size_t xsize, const size_t ysize,
      const int64_t ix, const int64_t iy,
      const float (&weights)[(kRadius + 1) * (kRadius + 1)]) {
    float sum = 0.0;

    // ix: image; kx: kernel; wx: weight
    for (int64_t ky = -kRadius; ky <= kRadius; ky++) {
      const int64_t wy = std::abs(ky);
      const int64_t y = WrapY()(iy + ky, ysize);
      const float* PIK_RESTRICT row_in = in.ConstRow(y);

      for (int64_t kx = -kRadius; kx <= kRadius; kx++) {
        const int64_t wx = std::abs(kx);
        const int64_t x = CheckX()(ix + kx, xsize);

        sum += row_in[x] * weights[wy * (kRadius + 1) + wx];
      }
    }
    return sum;
  }

  template <int64_t kRadius, class WrapY, class ImageOrView>
  static inline void ConvolveSymmetricRow(
      const ImageOrView& in, const size_t xsize, const size_t ysize,
      const int64_t iy, const float (&weights)[(kRadius + 1) * (kRadius + 1)],
      ImageF* PIK_RESTRICT out) {
    float* PIK_RESTRICT row_out = out->Row(iy);
    int64_t ix = 0;
    for (; ix < kRadius; ix++) {
      row_out[ix] = ConvolveSymmetricPixel<kRadius, WrapMirror, WrapY>(
          in, xsize, ysize, ix, iy, weights);
    }
    for (; ix < xsize - kRadius; ix++) {
      row_out[ix] = ConvolveSymmetricPixel<kRadius, WrapUnchanged, WrapY>(
          in, xsize, ysize, ix, iy, weights);
    }
    for (; ix < xsize; ix++) {
      row_out[ix] = ConvolveSymmetricPixel<kRadius, WrapMirror, WrapY>(
          in, xsize, ysize, ix, iy, weights);
    }
  }
};

#if SIMD_TARGET_VALUE != SIMD_NONE

// For code-folding.
struct Kernel {
  struct Box3 {
    static constexpr int64_t kRadius = 1;
    using D = Full<float>;
    static PIK_INLINE D::V HorzWeights() {
      constexpr float w0 = 0.111111111f;
      constexpr float w1 = 0.111111111f;
      constexpr float w2 = 0.111111111f;
      SIMD_ALIGN constexpr float lanes[4] = {w0, w1, w2, 0.0f};
      return D::V(load_dup128(D(), lanes).raw);
    }

    static PIK_INLINE D::V VertWeights() { return HorzWeights(); }
  };

  // Approximation of the Laplacian. Use with Direct3Laplacian.
  struct Laplacian3 {
    static constexpr int64_t kRadius = 1;
    using D = Full<float>;
    static PIK_INLINE D::V HorzWeights() {
      constexpr float w0 = -4.0f;
      constexpr float w1 = 1.0f;  // unused
      constexpr float w2 = 0.0f;  // unused
      SIMD_ALIGN constexpr float lanes[4] = {w0, w1, w2, 0.0f};
      return D::V(load_dup128(D(), lanes).raw);
    }

    static PIK_INLINE D::V VertWeights() { return HorzWeights(); }
  };

  // Computed by research/convolve_weights.py's cubic spline approximations of
  // prolate spheroidal wave functions.
  struct Lowpass3 {
    static constexpr int64_t kRadius = 1;
    using D = Full<float>;
    static PIK_INLINE D::V HorzWeights() {
      constexpr float w0 = 0.36208932f;
      constexpr float w1 = 0.12820096f;
      constexpr float w2 = 0.03127668f;
      SIMD_ALIGN constexpr float lanes[4] = {w0, w1, w2, 0.0f};
      return D::V(load_dup128(D(), lanes).raw);
    }

    static PIK_INLINE D::V VertWeights() { return HorzWeights(); }
  };

  struct Lowpass5 {
    static constexpr int64_t kRadius = 2;
    using D = Full<float>;
    static PIK_INLINE D::V HorzWeights() {
      constexpr float w0 = 0.41714928f;
      constexpr float w1 = 0.25539268f;
      constexpr float w2 = 0.03603267f;
      SIMD_ALIGN constexpr float lanes[4] = {w0, w1, w2, 0.0f};
      return D::V(load_dup128(D(), lanes).raw);
    }

    static PIK_INLINE D::V VertWeights() { return HorzWeights(); }
  };

  struct Lowpass7 {
    static constexpr int64_t kRadius = 3;
    using D = Full<float>;
    static PIK_INLINE D::V HorzWeights() {
      constexpr float w0 = 0.27847368f;
      constexpr float w1 = 0.22449034f;
      constexpr float w2 = 0.11221872f;
      constexpr float w3 = 0.02405410f;
      SIMD_ALIGN constexpr float lanes[4] = {w0, w1, w2, w3};
      return D::V(load_dup128(D(), lanes).raw);
    }

    static PIK_INLINE D::V VertWeights() { return HorzWeights(); }
  };

  struct Gaussian5 {
    static constexpr int64_t kRadius = 2;
    using D = Full<float>;
    static PIK_INLINE D::V HorzWeights() {
      // Sigma = 2
      constexpr float w0 = 0.250301f;
      constexpr float w1 = 0.221461f;
      constexpr float w2 = 0.153388f;
      SIMD_ALIGN constexpr float lanes[4] = {w0, w1, w2, 0.0f};
      return D::V(load_dup128(D(), lanes).raw);
    }

    static PIK_INLINE D::V VertWeights() { return HorzWeights(); }
  };

  struct Gaussian5S1 {
    static constexpr int64_t kRadius = 2;
    using D = Full<float>;
    static PIK_INLINE D::V HorzWeights() {
      // Sigma = 1
      constexpr float w0 = 0.38774f;
      constexpr float w1 = 0.24477f;
      constexpr float w2 = 0.06136f;
      SIMD_ALIGN constexpr float lanes[4] = {w0, w1, w2, 0.0f};
      return D::V(load_dup128(D(), lanes).raw);
    }

    static PIK_INLINE D::V VertWeights() { return HorzWeights(); }
  };

  struct Gaussian7 {
    static constexpr int64_t kRadius = 3;
    using D = Full<float>;
    static PIK_INLINE D::V HorzWeights() {
      // Sigma = 2
      constexpr float w0 = 0.214607f;
      constexpr float w1 = 0.189879f;
      constexpr float w2 = 0.131514f;
      constexpr float w3 = 0.071303f;
      SIMD_ALIGN constexpr float lanes[4] = {w0, w1, w2, w3};
      return D::V(load_dup128(D(), lanes).raw);
    }

    static PIK_INLINE D::V VertWeights() { return HorzWeights(); }
  };

  struct Box5 {
    static constexpr int64_t kRadius = 2;
    using D = Full<float>;
    static PIK_INLINE D::V HorzWeights() {
      constexpr float w = 0.2f;
      SIMD_ALIGN constexpr float lanes[4] = {w, w, w, 0.0f};
      return D::V(load_dup128(D(), lanes).raw);
    }

    static PIK_INLINE D::V VertWeights() { return HorzWeights(); }
  };

  struct Box7 {
    static constexpr int64_t kRadius = 3;
    using D = Full<float>;
    static PIK_INLINE D::V HorzWeights() {
      constexpr float w = 0.142857143f;
      SIMD_ALIGN constexpr float lanes[4] = {w, w, w, 0.0f};
      return D::V(load_dup128(D(), lanes).raw);
    }

    static PIK_INLINE D::V VertWeights() { return HorzWeights(); }
  };
};

// Synthesizes left/right neighbors from a vector of center pixels.
class Neighbors {
  using V = Full<float>::V;
  static const Full<float> d;

 public:
  // Returns l[i] == c[i - 1].
  static PIK_INLINE V L1(const V c, const V p) {
#if SIMD_TARGET_VALUE == SIMD_AVX2
    // c = PONM'LKJI, p = Hxxx'xxxx
    const V L_H = concat_lo_hi(c, p);
    return combine_shift_right_bytes<12>(c, L_H);  // ONML'KJIH
#else
    // c = LKJI, p = Hxxx
    return combine_shift_right_bytes<12>(c, p);  // KJIH
#endif
  }

  // Returns l[i] == c[Mirror(i - 1)].
  static PIK_INLINE V FirstL1(const V c) {
#if SIMD_TARGET_VALUE == SIMD_AVX2
    SIMD_ALIGN constexpr int lanes[8] = {0, 0, 1, 2, 3, 4, 5, 6};
    const auto indices = set_table_indices(d, lanes);
    // c = PONM'LKJI
    return table_lookup_lanes(c, indices);  // ONML'KJII
#else
    // c = LKJI
    return V(_mm_shuffle_ps(c.raw, c.raw, _MM_SHUFFLE(2, 1, 0, 0)));  // KJII
#endif
  }

  // Returns l[i] == c[Mirror(i - 2)].
  static PIK_INLINE V FirstL2(const V c) {
#if SIMD_TARGET_VALUE == SIMD_AVX2
    SIMD_ALIGN constexpr int lanes[8] = {1, 0, 0, 1, 2, 3, 4, 5};
    const auto indices = set_table_indices(d, lanes);
    // c = PONM'LKJI
    return table_lookup_lanes(c, indices);  // NMLK'JIIJ
#else
    // c = LKJI
    return V(_mm_shuffle_ps(c.raw, c.raw, _MM_SHUFFLE(1, 0, 0, 1)));  // JIIJ
#endif
  }

  // Returns r[i] == c[i + 1].
  static PIK_INLINE V R1(const V c, const V n) {
#if SIMD_TARGET_VALUE == SIMD_AVX2
    // c = PONM'LKJI, n = xxxx'xxxQ
    const V Q_M = concat_lo_hi(n, c);             // Right-aligned (lower lane)
    return combine_shift_right_bytes<4>(Q_M, c);  // QPON'MLKJ
#else
    // c = LKJI, n = xxxM
    return combine_shift_right_bytes<4>(n, c);  // MLKJ
#endif
  }

  // Returns r[i] == c[i + 1].
  static PIK_INLINE V LastR1(const V c) {
#if SIMD_TARGET_VALUE == SIMD_AVX2
    SIMD_ALIGN constexpr uint32_t lanes[8] = {1, 2, 3, 4, 5, 6, 7, 7};
    const auto indices = load(Full<uint32_t>(), lanes);
    // c = PONM'LKJI
    return V(_mm256_permutevar8x32_ps(c.raw, indices.raw));  // PPON'MLKJ
#else
    // c = LKJI
    const auto L = broadcast<3>(c);
    return combine_shift_right_bytes<4>(L, c);  // LLKJ
#endif
  }
};

// Requires kRadius valid (mirrored or neighbor) columns on either side of
// [0, xsize). It is also safe to load entire vectors. This behavior is
// required by TFNode with Borders(>0). In other cases, this assumption requires
// ConvolveT to PadImage, so LeftRightInvalid would be more efficient.
struct LeftRightValid {};

// No valid values outside [0, xsize), but the strategy may still safely load
// the preceding vector, and/or round xsize up to the vector lane count. This
// avoids needing PadImage.
struct LeftRightInvalid {};

// LeftRightInvalid requires xsize >= Full<float>::N + kConvolveMaxRadius.
static constexpr size_t kConvolveMaxRadius = 3;

// For use by set_table_indices.
static inline const int32_t* MirrorLanes(const size_t mod) {
  Full<float> d;
#if SIMD_TARGET_VALUE == SIMD_AVX2
  // last  part  mirrored
  // 01234567| 76543210   loadedReg 76543210 mirroredReg 01234567
  // 01234567|8 8765432   loadedReg 87654321 mirroredReg 23456788
  // 01234567|89 987654   loadedReg 98765432 mirroredReg 45678998
  // 01234567|89A A9876   loadedReg A9876543 mirroredReg 6789AA98
  // 01234567|89AB BA98
  // 01234567|89ABC CBA
  // 01234567|89ABCD DC
  // 01234567|89ABCDE E   loadedReg EDCBA987 mirroredReg EEDCBA98
  SIMD_ALIGN static constexpr int32_t idx_lanes[d.N * d.N] = {
      7, 6, 5, 4, 3, 2, 1, 0,  // 0
      7, 7, 6, 5, 4, 3, 2, 1,  // 1
      6, 7, 7, 6, 5, 4, 3, 2,  // 2
      5, 6, 7, 7, 6, 5, 4, 3,  // 3
      4, 5, 6, 7, 7, 6, 5, 4,  // 4
      3, 4, 5, 6, 7, 7, 6, 5,  // 5
      2, 3, 4, 5, 6, 7, 7, 6,  // 6
      1, 2, 3, 4, 5, 6, 7, 7,  // 7
  };
#else
  // 0123| 3210   loadedReg 3210 mirroredReg 0123
  // 0123|4 432   loadedReg 4321 mirroredReg 2344
  // 0123|45 54   loadedReg 5432 mirroredReg 4554
  // 0123|456 6   loadedReg 6543 mirroredReg 6654

  SIMD_ALIGN static constexpr int32_t idx_lanes[d.N * d.N] = {
      3, 2, 1, 0,  // 0
      3, 3, 2, 1,  // 1
      2, 3, 3, 2,  // 2
      1, 2, 3, 3,  // 3
  };
#endif
  return idx_lanes + mod * d.N;
}

// 3x3 convolution by symmetric kernel with a single scan through the input.
// Uses Kernel::HorzWeights()[i] for Manhattan distance i = [0, 3).
class Direct3 {
  using V = Full<float>::V;
  static const Full<float> d;

 public:
  static constexpr int64_t kRadius = 1;
  static constexpr size_t kNumWeights = 3;

  template <class Kernel>
  static void UnpackWeights(float* PIK_RESTRICT weights) {
    store(broadcast<0>(Kernel::HorzWeights()), d, weights + 0 * d.N);
    store(broadcast<1>(Kernel::HorzWeights()), d, weights + 1 * d.N);
    store(broadcast<2>(Kernel::HorzWeights()), d, weights + 2 * d.N);
  }

  // Weights: Manhattan distance 0 = center, 1 = 4-neighborhood, 2 = diagonal.
  // Only accesses pixels in [0, xsize).
  template <size_t kSizeModN, class WrapRow>
  static PIK_INLINE void ConvolveRow(LeftRightInvalid,
                                     const float* const PIK_RESTRICT row_m,
                                     const size_t xsize, const int64_t stride,
                                     const WrapRow& wrap_row,
                                     const float* PIK_RESTRICT weights,
                                     float* const PIK_RESTRICT row_out) {
    // t, m, b = top, middle, bottom row;
    const float* const PIK_RESTRICT row_t = wrap_row(row_m - stride, stride);
    const float* const PIK_RESTRICT row_b = wrap_row(row_m + stride, stride);

    // l, c, r = left, center, right. Leftmost vector: need FirstL1.
    {
      const V tc = load_unaligned(d, row_t + 0);
      const V mc = load_unaligned(d, row_m + 0);
      const V bc = load_unaligned(d, row_b + 0);
      const V sum_tb = tc + bc;
      const V tl = Neighbors::FirstL1(tc);
      const V tr = load_unaligned(d, row_t + 0 + 1);
      const V sum_t2 = tl + tr;
      const V ml = Neighbors::FirstL1(mc);
      const V mr = load_unaligned(d, row_m + 0 + 1);
      const V sum_lr = ml + mr;
      const V bl = Neighbors::FirstL1(bc);
      const V br = load_unaligned(d, row_b + 0 + 1);
      const V sum_b2 = bl + br;
      const V conv =
          WeightedSum012(mc, sum_tb + sum_lr, sum_t2 + sum_b2, weights);
      store(conv, d, row_out + 0);
    }

    // Loop as long as we can load enough new values:
    size_t x = d.N;
    for (; x + d.N + kRadius <= xsize; x += d.N) {
      const auto conv = ConvolveValid(row_t, row_m, row_b, x, weights);
      store(conv, d, row_out + x);
    }

    // For final (partial) vector:
    const V tc = load_unaligned(d, row_t + x);
    const V mc = load_unaligned(d, row_m + x);
    const V bc = load_unaligned(d, row_b + x);
    const V sum_tb = tc + bc;

    V tr, mr, br;
    if (kSizeModN == 0) {
      // The above loop didn't handle the last vector because it needs an
      // additional right neighbor (generated via mirroring).
      auto mirror = set_table_indices(d, MirrorLanes(d.N - 1));
      tr = table_lookup_lanes(tc, mirror);
      mr = table_lookup_lanes(mc, mirror);
      br = table_lookup_lanes(bc, mirror);
    } else {
      auto mirror = set_table_indices(d, MirrorLanes((xsize % d.N) - 1));
      // Loads last valid value into uppermost lane and mirrors.
      tr = table_lookup_lanes(load_unaligned(d, row_t + xsize - d.N), mirror);
      mr = table_lookup_lanes(load_unaligned(d, row_m + xsize - d.N), mirror);
      br = table_lookup_lanes(load_unaligned(d, row_b + xsize - d.N), mirror);
    }

    const V tl = load_unaligned(d, row_t + x - 1);
    const V sum_t2 = tl + tr;
    const V ml = load_unaligned(d, row_m + x - 1);
    const V sum_lr = ml + mr;
    const V bl = load_unaligned(d, row_b + x - 1);
    const V sum_b2 = bl + br;
    const V conv =
        WeightedSum012(mc, sum_tb + sum_lr, sum_t2 + sum_b2, weights);
    store(conv, d, row_out + x);
  }

  // Weights: Manhattan distance 0 = center, 1 = 4-neighborhood, 2 = diagonal.
  template <size_t kSizeModN, class WrapRow>
  static PIK_INLINE void ConvolveRow(LeftRightValid,
                                     const float* const PIK_RESTRICT row_m,
                                     const size_t xsize, const int64_t stride,
                                     const WrapRow& wrap_row,
                                     const float* PIK_RESTRICT weights,
                                     float* const PIK_RESTRICT row_out) {
    // t, m, b = top, middle, bottom row;
    const float* const PIK_RESTRICT row_t = wrap_row(row_m - stride, stride);
    const float* const PIK_RESTRICT row_b = wrap_row(row_m + stride, stride);

    // l, c, r = left, center, right.
    for (size_t x = 0; x < xsize; x += d.N) {
      const V conv = ConvolveValid(row_t, row_m, row_b, x, weights);
      store(conv, d, row_out + x);
    }
  }

 private:
  // Returns sum{x_i * w_i}.
  template <class V>
  static PIK_INLINE V WeightedSum012(const V x0, const V x1, const V x2,
                                     const float* PIK_RESTRICT weights) {
    const V w0 = load(d, weights + 0 * d.N);

    const V w1 = load(d, weights + 1 * d.N);
    const V mul0 = x0 * w0;

    const V w2 = load(d, weights + 2 * d.N);
    const V mul1 = x1 * w1;

    const V mul2 = mul_add(x2, w2, mul0);
    return mul1 + mul2;
  }

  static PIK_INLINE V ConvolveValid(const float* PIK_RESTRICT row_t,
                                    const float* PIK_RESTRICT row_m,
                                    const float* PIK_RESTRICT row_b,
                                    const int64_t x,
                                    const float* PIK_RESTRICT weights) {
    const V tc = load_unaligned(d, row_t + x);
    const V mc = load_unaligned(d, row_m + x);
    const V bc = load_unaligned(d, row_b + x);
    const V sum_tb = tc + bc;
    const V tl = load_unaligned(d, row_t + x - 1);
    const V tr = load_unaligned(d, row_t + x + 1);
    const V sum_t2 = tl + tr;
    const V ml = load_unaligned(d, row_m + x - 1);
    const V mr = load_unaligned(d, row_m + x + 1);
    const V sum_lr = ml + mr;
    const V bl = load_unaligned(d, row_b + x - 1);
    const V br = load_unaligned(d, row_b + x + 1);
    const V sum_b2 = bl + br;
    return WeightedSum012(mc, sum_tb + sum_lr, sum_t2 + sum_b2, weights);
  }
};

// Same as Direct3, but assumes weights[1] == 1 and weights[2] == 0.
class Direct3Laplacian {
  using V = Full<float>::V;
  static const Full<float> d;

 public:
  static constexpr int64_t kRadius = 1;
  static constexpr size_t kNumWeights = 1;

  template <class Kernel>
  static void UnpackWeights(float* PIK_RESTRICT weights) {
    store(broadcast<0>(Kernel::HorzWeights()), d, weights + 0 * d.N);
  }

  // Only accesses pixels in [0, xsize).
  template <size_t kSizeModN, class WrapRow>
  static PIK_INLINE void ConvolveRow(LeftRightInvalid,
                                     const float* const PIK_RESTRICT row_m,
                                     const size_t xsize, const int64_t stride,
                                     const WrapRow& wrap_row,
                                     const float* PIK_RESTRICT weights,
                                     float* const PIK_RESTRICT row_out) {
    // t, m, b = top, middle, bottom row;
    const float* const PIK_RESTRICT row_t = wrap_row(row_m - stride, stride);
    const float* const PIK_RESTRICT row_b = wrap_row(row_m + stride, stride);
    // p, c, n = previous, current, next register.
    auto tp = load_unaligned(d, row_t);
    auto tc = load_unaligned(d, row_t + d.N);
    auto mp = load_unaligned(d, row_m);
    auto mc = load_unaligned(d, row_m + d.N);
    auto bp = load_unaligned(d, row_b);
    auto bc = load_unaligned(d, row_b + d.N);

    const V w0 = load(d, weights);

    // Leftmost vector: "previous" is actually center.
    const V sum_lr = Neighbors::FirstL1(mp) + Neighbors::R1(mp, mc);
    const V conv = mul_add(mp, w0, tp + bp + sum_lr);
    store(conv, d, row_out + 0);

    // Loop while at least 1 value to load (higher lanes may be uninitialized)
    size_t x = d.N;
    for (; x + d.N + 1 <= xsize; x += d.N) {
      const V tn = load_unaligned(d, row_t + x + d.N);
      const V mn = load_unaligned(d, row_m + x + d.N);
      const V bn = load_unaligned(d, row_b + x + d.N);

      // Here, this is faster than the unaligned loads in Direct3!
      const V sum_lr = Neighbors::L1(mc, mp) + Neighbors::R1(mc, mn);
      const V conv = mul_add(mc, w0, tc + bc + sum_lr);
      store(conv, d, row_out + x);

      tp = tc;
      tc = tn;
      mp = mc;
      mc = mn;
      bp = bc;
      bc = bn;
    }

    // Not a whole vector => need to pad "center" via mirroring.
    if ((xsize % d.N) != 0) {
      // Last valid value in uppermost lane.
      const V t_last = load_unaligned(d, row_t + xsize - d.N);
      const V m_last = load_unaligned(d, row_m + xsize - d.N);
      const V b_last = load_unaligned(d, row_b + xsize - d.N);

      const auto mirror = set_table_indices(d, MirrorLanes(xsize % d.N));
      tc = table_lookup_lanes(t_last, mirror);
      mc = table_lookup_lanes(m_last, mirror);
      bc = table_lookup_lanes(b_last, mirror);
    }

    // Write the last vector, of which [1, d.N] lanes are valid.
    {
      const V sum_lr = Neighbors::L1(mc, mp) + Neighbors::LastR1(mc);
      const V conv = mul_add(mc, w0, tc + bc + sum_lr);
      store(conv, d, row_out + x);
    }
  }

  // Weights: Manhattan distance 0 = center, 1 = 4-neighborhood, 2 = diagonal.
  template <size_t kSizeModN, class WrapRow>
  static PIK_INLINE void ConvolveRow(LeftRightValid,
                                     const float* const PIK_RESTRICT row_m,
                                     const size_t xsize, const int64_t stride,
                                     const WrapRow& wrap_row,
                                     const float* PIK_RESTRICT weights,
                                     float* const PIK_RESTRICT row_out) {
    // t, m, b = top, middle, bottom row;
    const float* const PIK_RESTRICT row_t = wrap_row(row_m - stride, stride);
    const float* const PIK_RESTRICT row_b = wrap_row(row_m + stride, stride);
    // p, c, n = previous, current, next register.
    auto tp = load_unaligned(d, row_t - d.N);
    auto tc = load_unaligned(d, row_t);
    auto mp = load_unaligned(d, row_m - d.N);
    auto mc = load_unaligned(d, row_m);
    auto bp = load_unaligned(d, row_b - d.N);
    auto bc = load_unaligned(d, row_b);

    const V w0 = load(d, weights);

    // Loop until all output produced. WARNING: padding is uninitialized!
    for (size_t x = 0; x < xsize; x += d.N) {
      const V tn = load_unaligned(d, row_t + x + d.N);
      const V mn = load_unaligned(d, row_m + x + d.N);
      const V bn = load_unaligned(d, row_b + x + d.N);

      const V sum_lr = Neighbors::L1(mc, mp) + Neighbors::R1(mc, mn);
      const V conv = mul_add(mc, w0, tc + bc + sum_lr);
      store(conv, d, row_out + x);

      tp = tc;
      tc = tn;
      mp = mc;
      mc = mn;
      bp = bc;
      bc = bn;
    }
  }
};

// 5x5 convolution by separable kernel with a single scan through the input.
class Separated5 {
  using V = Full<float>::V;
  static const Full<float> d;

 public:
  static constexpr int64_t kRadius = 2;
  static constexpr size_t kNumWeights = 6;

  template <class Kernel>
  static void UnpackWeights(float* PIK_RESTRICT weights) {
    store(broadcast<0>(Kernel::HorzWeights()), d, weights + 0 * d.N);
    store(broadcast<1>(Kernel::HorzWeights()), d, weights + 1 * d.N);
    store(broadcast<2>(Kernel::HorzWeights()), d, weights + 2 * d.N);
    store(broadcast<0>(Kernel::VertWeights()), d, weights + 3 * d.N);
    store(broadcast<1>(Kernel::VertWeights()), d, weights + 4 * d.N);
    store(broadcast<2>(Kernel::VertWeights()), d, weights + 5 * d.N);
  }

  template <size_t kSizeModN, class WrapRow>
  static PIK_INLINE void ConvolveRow(LeftRightInvalid,
                                     const float* const PIK_RESTRICT row_m,
                                     const size_t xsize, const int64_t stride,
                                     const WrapRow& wrap_row,
                                     const float* PIK_RESTRICT weights,
                                     float* const PIK_RESTRICT row_out) {
    const int64_t neg_stride = -stride;  // allows LEA addressing.
    const float* const PIK_RESTRICT row_t2 =
        wrap_row(row_m + 2 * neg_stride, stride);
    const float* const PIK_RESTRICT row_t1 =
        wrap_row(row_m + 1 * neg_stride, stride);
    const float* const PIK_RESTRICT row_b1 =
        wrap_row(row_m + 1 * stride, stride);
    const float* const PIK_RESTRICT row_b2 =
        wrap_row(row_m + 2 * stride, stride);

    const V w0 = load(d, weights + 3 * d.N);
    const V w1 = load(d, weights + 4 * d.N);
    const V w2 = load(d, weights + 5 * d.N);

    {
      const V conv0 = HorzConvolveFirst(row_m, weights) * w0;

      const V conv1t = HorzConvolveFirst(row_t1, weights);
      const V conv1b = HorzConvolveFirst(row_b1, weights);
      const V conv1 = mul_add(conv1t + conv1b, w1, conv0);

      const V conv2t = HorzConvolveFirst(row_t2, weights);
      const V conv2b = HorzConvolveFirst(row_b2, weights);
      const V conv2 = mul_add(conv2t + conv2b, w2, conv1);
      store(conv2, d, row_out + 0);
    }

    // Main loop: load inputs without padding
    size_t x = d.N;
    for (; x + d.N + kRadius <= xsize; x += d.N) {
      const V conv0 = HorzConvolve(row_m + x, weights) * w0;

      const V conv1t = HorzConvolve(row_t1 + x, weights);
      const V conv1b = HorzConvolve(row_b1 + x, weights);
      const V conv1 = mul_add(conv1t + conv1b, w1, conv0);

      const V conv2t = HorzConvolve(row_t2 + x, weights);
      const V conv2b = HorzConvolve(row_b2 + x, weights);
      const V conv2 = mul_add(conv2t + conv2b, w2, conv1);
      store(conv2, d, row_out + x);
    }

    // Last full vector to write (the above loop handled mod >= 2)
    if (kSizeModN < 2) {
      const V conv0 =
          HorzConvolveLast<kSizeModN>(row_m, x, xsize, weights) * w0;

      const V conv1t = HorzConvolveLast<kSizeModN>(row_t1, x, xsize, weights);
      const V conv1b = HorzConvolveLast<kSizeModN>(row_b1, x, xsize, weights);
      const V conv1 = mul_add(conv1t + conv1b, w1, conv0);

      const V conv2t = HorzConvolveLast<kSizeModN>(row_t2, x, xsize, weights);
      const V conv2b = HorzConvolveLast<kSizeModN>(row_b2, x, xsize, weights);
      const V conv2 = mul_add(conv2t + conv2b, w2, conv1);
      store(conv2, d, row_out + x);
      x += d.N;
    }

    // If mod = 0, the above vector was the last.
    if (kSizeModN != 0) {
      for (; x < xsize; ++x) {
        float mul = 0.0f;
        for (int64_t dy = -kRadius; dy <= kRadius; ++dy) {
          const float wy = weights[3 * d.N + std::abs(dy) * d.N];
          const float* clamped_row = wrap_row(row_m + dy * stride, stride);
          for (int64_t dx = -kRadius; dx <= kRadius; ++dx) {
            const float wx = weights[std::abs(dx) * d.N];
            const int64_t clamped_x = Mirror(x + dx, xsize);
            mul += clamped_row[clamped_x] * wx * wy;
          }
        }
        row_out[x] = mul;
      }
    }
  }

  template <size_t kSizeModN, class WrapRow>
  static PIK_INLINE void ConvolveRow(LeftRightValid,
                                     const float* const PIK_RESTRICT row_m,
                                     const size_t xsize, const int64_t stride,
                                     const WrapRow& wrap_row,
                                     const float* PIK_RESTRICT weights,
                                     float* const PIK_RESTRICT row_out) {
    const int64_t neg_stride = -stride;  // allows LEA addressing.
    const float* const PIK_RESTRICT row_t2 =
        wrap_row(row_m + 2 * neg_stride, stride);
    const float* const PIK_RESTRICT row_t1 =
        wrap_row(row_m + 1 * neg_stride, stride);
    const float* const PIK_RESTRICT row_b1 =
        wrap_row(row_m + 1 * stride, stride);
    const float* const PIK_RESTRICT row_b2 =
        wrap_row(row_m + 2 * stride, stride);

    const V w0 = load(d, weights + 3 * d.N);
    const V w1 = load(d, weights + 4 * d.N);
    const V w2 = load(d, weights + 5 * d.N);

    // Loop until all output produced. WARNING: padding is uninitialized!
    for (size_t x = 0; x < xsize; x += d.N) {
      const V conv0 = HorzConvolve(row_m + x, weights) * w0;

      const V conv1t = HorzConvolve(row_t1 + x, weights);
      const V conv1b = HorzConvolve(row_b1 + x, weights);
      const V conv1 = mul_add(conv1t + conv1b, w1, conv0);

      const V conv2t = HorzConvolve(row_t2 + x, weights);
      const V conv2b = HorzConvolve(row_b2 + x, weights);
      const V conv2 = mul_add(conv2t + conv2b, w2, conv1);

      store(conv2, d, row_out + x);
    }
  }

 private:
  // Same as HorzConvolve for the first/last vector in a row.
  static PIK_INLINE V
  HorzConvolveFirst(const float* const PIK_RESTRICT pos,
                    const float* PIK_RESTRICT horz_weights) {
    const V c = load_unaligned(d, pos);
    const V mul0 = c * load(d, horz_weights + 0 * d.N);

    const V r1 = load_unaligned(d, pos + 1);
    const V r2 = load_unaligned(d, pos + 2);

    const V sum1 = Neighbors::FirstL1(c) + r1;
    const V mul1 = mul_add(sum1, load(d, horz_weights + 1 * d.N), mul0);
    const V sum2 = Neighbors::FirstL2(c) + r2;
    const V mul2 = mul_add(sum2, load(d, horz_weights + 2 * d.N), mul1);
    return mul2;
  }

  template <size_t kSizeModN>
  static PIK_INLINE V HorzConvolveLast(const float* const PIK_RESTRICT row,
                                       const size_t x, const size_t xsize,
                                       const float* PIK_RESTRICT horz_weights) {
    const V c = load_unaligned(d, row + x);
    const V mul0 = c * load(d, horz_weights + 0 * d.N);

    const V l1 = load_unaligned(d, row + x - 1);
    const V l2 = load_unaligned(d, row + x - 2);

    V r1, r2;
    if (kSizeModN == 0) {
      r2 = table_lookup_lanes(c, set_table_indices(d, MirrorLanes(d.N - 2)));
      r1 = table_lookup_lanes(c, set_table_indices(d, MirrorLanes(d.N - 1)));
    } else {  // == 1
      const auto last = load_unaligned(d, row + xsize - d.N);
      r2 = table_lookup_lanes(last, set_table_indices(d, MirrorLanes(d.N - 1)));
      r1 = last;
    }

    // Sum of pixels with Manhattan distance i, multiplied by weights[i].
    const V sum1 = l1 + r1;
    const V mul1 = mul_add(sum1, load(d, horz_weights + 1 * d.N), mul0);
    const V sum2 = l2 + r2;
    const V mul2 = mul_add(sum2, load(d, horz_weights + 2 * d.N), mul1);
    return mul2;
  }

  // Requires kRadius valid pixels before/after pos.
  static PIK_INLINE V HorzConvolve(const float* const PIK_RESTRICT pos,
                                   const float* PIK_RESTRICT horz_weights) {
    const V c = load_unaligned(d, pos);
    const V mul0 = c * load(d, horz_weights + 0 * d.N);

    // Loading anew is faster than combining vectors.
    const V l1 = load_unaligned(d, pos - 1);
    const V r1 = load_unaligned(d, pos + 1);
    const V l2 = load_unaligned(d, pos - 2);
    const V r2 = load_unaligned(d, pos + 2);
    // Sum of pixels with Manhattan distance i, multiplied by weights[i].
    const V sum1 = l1 + r1;
    const V mul1 = mul_add(sum1, load(d, horz_weights + 1 * d.N), mul0);
    const V sum2 = l2 + r2;
    const V mul2 = mul_add(sum2, load(d, horz_weights + 2 * d.N), mul1);
    return mul2;
  }
};

// Avoids all bounds checks, but requires ImageView input, e.g. TileFlow nodes
// with Borders(kRadius) and TFWrap::kMirror for sources.
struct BorderAlreadyValid {};

// Avoids PadImage, but requires a strategy that supports LeftRightInvalid.
struct BorderNeverUsed {};

// Slow: Convolve calls PadImage and requires bounds checks.
struct BorderNeedsInit {};

// Adapters for zero-cost switching between ThreadPool and non-threaded loop.
struct ExecutorLoop {};
struct ExecutorPool {
  explicit ExecutorPool(ThreadPool* pool) : pool(pool) {}

  ThreadPool* pool;  // not owned
};

// Single entry point for convolution. A class template avoids repeating the
// arguments in each function.
// "Strategy" (Direct*/Separable*) decides kernel size and how to evaluate it.
// "Kernel" (kernel::*) defines the weights.
// "Border" is Border{AlreadyValid/NeverUsed/NeedsInit}.
template <class Strategy, class Kernel, class Border = BorderNeverUsed>
class ConvolveT {
  static constexpr int64_t kRadius = Strategy::kRadius;
  static_assert(kRadius == Kernel::kRadius, "kRadius mismatch");

 public:
  // Wrapper for common case (no ThreadPool) to avoid ::template syntax.
  static PIK_INLINE void Run(const ImageF& in, const ImageF* out) {
    RunExecutor(in, out, ExecutorLoop());
  }

  // Not supported for BorderAlreadyValid because Image rows are vector-aligned.
  // "Executor": ExecutorPool uses a ThreadPool; ExecutorLoop just loops.
  template <class Executor>
  static PIK_INLINE void RunExecutor(const ImageF& in, const ImageF* out,
                                     const Executor executor) {
    PIK_CHECK(SameSize(in, *out));
    const size_t xsize = in.xsize();
    const size_t ysize = in.ysize();
    PIK_CHECK(xsize >= Full<float>::N);  // For BorderNeverUsed.

    RunImpl(Border(), in, xsize, ysize, out, executor);
  }

  // Compatible with TFFunc. Border is typically BorderAlreadyValid if the
  // TFNode has Borders(>0).
  template <int kPlanes>
  static void RunPlanes(const void*, const ConstImageViewF* in,
                        const OutputRegion& output_region,
                        const MutableImageViewF* out) {
    for (int c = 0; c < kPlanes; ++c) {
      RunImpl(Border(), in[c], output_region.xsize, output_region.ysize,
              out + c, ExecutorLoop());
    }
  }

 private:
  // Similar to image.h Wrap but for row pointers.
  class WrapRowMirror {
   public:
    template <class ImageOrView>
    WrapRowMirror(const ImageOrView& image, const size_t ysize)
        : first_row_(image.ConstRow(0)), last_row_(image.ConstRow(ysize - 1)) {}

    const float* const PIK_RESTRICT operator()(
        const float* const PIK_RESTRICT row, const int64_t stride) const {
      if (row < first_row_) {
        const int64_t num_before = first_row_ - row;
        // Mirrored; one row before => row 0, two before = row 1, ...
        return first_row_ + num_before - stride;
      }
      if (row > last_row_) {
        const int64_t num_after = row - last_row_;
        // Mirrored; one row after => last row, two after = last - 1, ...
        return last_row_ - num_after + stride;
      }
      return row;
    }

   private:
    const float* const PIK_RESTRICT first_row_;
    const float* const PIK_RESTRICT last_row_;
  };

  // Used in the interior, where we have kRadius valid rows before and after.
  struct WrapRowIdentity {
    PIK_INLINE const float* const PIK_RESTRICT operator()(
        const float* const PIK_RESTRICT row, const int64_t stride) const {
      return row;
    }
  };

  template <size_t kSizeModN, class LeftRight, class WrapRow>
  static PIK_INLINE void RunRow(const float* PIK_RESTRICT in,
                                const size_t xsize, const int64_t stride,
                                const WrapRow& wrap_row,
                                const float* PIK_RESTRICT weights,
                                const float* PIK_RESTRICT out) {
    // LeftRight value instead of template arg enables overload resolution.
    Strategy::template ConvolveRow<kSizeModN>(LeftRight(), in, xsize, stride,
                                              wrap_row, weights,
                                              const_cast<float*>(out));
  }

  // Threaded.
  template <size_t kSizeModN, class LeftRight, class ImageOrConstView,
            class ImageOrMutableView>
  static PIK_INLINE void RunInterior(const ExecutorPool executor,
                                     const ImageOrConstView& in,
                                     const size_t xsize, const int64_t ybegin,
                                     const int64_t yend, const int64_t stride,
                                     const float* PIK_RESTRICT weights,
                                     const ImageOrMutableView* out) {
    // There is no interior if ysize <= 2 * kRadius.
    if (ybegin >= yend) return;

    executor.pool->Run(
        ybegin, yend,
        [&in, xsize, stride, weights, out](const int y, const int thread) {
          RunRow<kSizeModN, LeftRight>(in.ConstRow(y), xsize, stride,
                                       WrapRowIdentity(), weights, out->Row(y));
        });
  }

  // No thread, just loop.
  template <size_t kSizeModN, class LeftRight, class ImageOrConstView,
            class ImageOrMutableView>
  static PIK_INLINE void RunInterior(ExecutorLoop, const ImageOrConstView& in,
                                     const size_t xsize, const int64_t ybegin,
                                     const int64_t yend, const int64_t stride,
                                     const float* PIK_RESTRICT weights,
                                     const ImageOrMutableView* out) {
    const float* row_in = in.ConstRow(ybegin);
    const float* row_out = out->Row(ybegin);  // RunRow casts to float*.
    for (int64_t y = ybegin; y < yend; ++y) {
      RunRow<kSizeModN, LeftRight>(row_in, xsize, stride, WrapRowIdentity(),
                                   weights, row_out);
      row_in += in.bytes_per_row() / sizeof(float);
      row_out += out->bytes_per_row() / sizeof(float);
    }
  }

  template <size_t kSizeModN, class LeftRight, class ImageOrConstView,
            class ImageOrMutableView, class Executor>
  static PIK_INLINE void RunWithBoundsChecks(const ImageOrConstView& in,
                                             const size_t xsize,
                                             const int64_t ysize,
                                             const float* PIK_RESTRICT weights,
                                             const ImageOrMutableView* out,
                                             const Executor executor) {
    const int64_t stride = in.bytes_per_row() / sizeof(float);
    const Full<float> d;
    const WrapRowMirror wrap_row(in, ysize);

    for (int64_t y = 0; y < kRadius; ++y) {
      RunRow<kSizeModN, LeftRight>(in.ConstRow(y), xsize, stride, wrap_row,
                                   weights, out->Row(y));
    }

    RunInterior<kSizeModN, LeftRight>(executor, in, xsize, kRadius,
                                      ysize - kRadius, stride, weights, out);

    for (int64_t y = ysize - kRadius; y < ysize; ++y) {
      RunRow<kSizeModN, LeftRight>(in.ConstRow(y), xsize, stride, wrap_row,
                                   weights, out->Row(y));
    }
  }

  // Ensures each row has an additional vector's worth of valid values on the
  // right AND left borders (residing in otherwise unused padding area reserved
  // by BytesPerRow), initialized via mirroring with replication.
  template <template <typename> class ImageOrView, typename T>
  static void PadImage(const size_t xsize, const size_t ysize,
                       const ImageOrView<T>* image) {
    PIK_ASSERT(xsize > kRadius && ysize > kRadius);
    static_assert(kRadius * sizeof(T) <= kMaxVectorSize, "Not enough padding");

    for (size_t y = 0; y < ysize; ++y) {
      // Even if the image is const, we're allowed to overwrite its padding.
      T* const PIK_RESTRICT row = const_cast<T*>(image->ConstRow(y));

      for (int64_t i = 0; i < kRadius; ++i) {
        row[xsize + i] = row[Mirror(xsize + i, xsize)];
        row[-1 - i] = row[i];
      }
    }
  }

  // Slow path: padding and bounds checks.
  template <class ImageOrConstView, class ImageOrMutableView, class Executor>
  static PIK_INLINE void RunImpl(BorderNeedsInit, const ImageOrConstView& in,
                                 const size_t xsize, const size_t ysize,
                                 const ImageOrMutableView* out,
                                 const Executor executor) {
    PROFILER_ZONE("Convolve slow");
    // Each RunRow requires that 2*kRadius+1 rows already be padded. Padding
    // the entire image pollutes the cache. We could pre-pad 2*kRadius rows and
    // then one row per RunRow, but callers who care about speed should anyway
    // use the other, faster Border modes.
    PadImage(xsize, ysize, &in);

    const Full<float> d;
    SIMD_ALIGN float weights[Strategy::kNumWeights * d.N];
    Strategy::template UnpackWeights<Kernel>(weights);

    switch (xsize % d.N) {
      case 0:
        return RunWithBoundsChecks<0, LeftRightValid>(in, xsize, ysize, weights,
                                                      out, executor);
      case 1:
        return RunWithBoundsChecks<1, LeftRightValid>(in, xsize, ysize, weights,
                                                      out, executor);
      default:  // Only need <= kRadius
        return RunWithBoundsChecks<2, LeftRightValid>(in, xsize, ysize, weights,
                                                      out, executor);
    }
  }

  // Fast: already have extra columns AND rows => no bounds checks. Only
  // possible with *ImageView because Image rows must be vector-aligned.
  template <class Executor>
  static PIK_INLINE void RunImpl(BorderAlreadyValid, const ConstImageViewF& in,
                                 const size_t xsize, const size_t ysize,
                                 const MutableImageViewF* out,
                                 const Executor executor) {
    PROFILER_ZONE("Convolve tile");
    const int64_t stride = in.bytes_per_row() / sizeof(float);
    const Full<float> d;
    SIMD_ALIGN float weights[Strategy::kNumWeights * d.N];
    Strategy::template UnpackWeights<Kernel>(weights);

    switch (xsize % d.N) {
      case 0:
        return RunInterior<0, LeftRightValid>(executor, in, xsize, 0, ysize,
                                              stride, weights, out);
      case 1:
        return RunInterior<1, LeftRightValid>(executor, in, xsize, 0, ysize,
                                              stride, weights, out);
      default:  // Only need <= kRadius
        return RunInterior<2, LeftRightValid>(executor, in, xsize, 0, ysize,
                                              stride, weights, out);
    }
  }

  // Fast: no padding, but bounds checks.
  template <class ImageOrConstView, class ImageOrMutableView, class Executor>
  static PIK_INLINE void RunImpl(BorderNeverUsed, const ImageOrConstView& in,
                                 const size_t xsize, const size_t ysize,
                                 const ImageOrMutableView* out,
                                 const Executor executor) {
    PROFILER_ZONE("Convolve fast");
    const Full<float> d;
    SIMD_ALIGN float weights[Strategy::kNumWeights * d.N];
    Strategy::template UnpackWeights<Kernel>(weights);

    switch (xsize % d.N) {
      case 0:
        return RunWithBoundsChecks<0, LeftRightInvalid>(in, xsize, ysize,
                                                        weights, out, executor);
      case 1:
        return RunWithBoundsChecks<1, LeftRightInvalid>(in, xsize, ysize,
                                                        weights, out, executor);
      default:  // Only need <= kRadius
        return RunWithBoundsChecks<2, LeftRightInvalid>(in, xsize, ysize,
                                                        weights, out, executor);
    }
  }
};

#endif  // SIMD_TARGET_VALUE != SIMD_NONE

}  // namespace SIMD_NAMESPACE
}  // namespace pik

#endif  // CONVOLVE_H_
