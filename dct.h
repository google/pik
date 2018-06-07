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

#ifndef DCT_H_
#define DCT_H_

#include "image.h"
#include "simd/simd.h"
#include "tile_flow.h"

namespace pik {

// Computes the in-place 8x8 DCT of block.
// Requires that block is 32-bytes aligned.
//
// The DCT used is a scaled variant of DCT-II, which is orthonormal:
//
// G(u,v) =
//     (1/4)alpha(u)alpha(v)*
//     sum_{x,y}(g(x,y)*cos((2x+1)uM_PI/16)*cos((2y+1)vM_PI/16))
//
// where alpha(u) is 1/sqrt(2) if u = 0 and 1 otherwise, g_{x,y} is the pixel
// value at coordiantes (x,y) and G(u,v) is the DCT coefficient at spatial
// frequency (u,v).
void ComputeBlockDCTFloat(float block[64]);

// Computes the in-place 8x8 inverse DCT of block.
// Requires that block is 32-bytes aligned.
void ComputeBlockIDCTFloat(float block[64]);

// Returns a 8*N x 8*M image where each 8x8 block is produced with
// ComputeTransposedScaledBlockIDCTFloat() from the corresponding 64x1 block of
// the coefficient image.
// REQUIRES: coeffs.xsize() == 64*N, coeffs.ysize() == M
Image3F TransposedScaledIDCT(const Image3F& coeffs);

TFNode* AddTransposedScaledIDCT(const TFPorts in_xyb, bool zero_dc,
                                TFBuilder* builder);

// Returns a 64*N x M image where each 64x1 block is produced with
// ComputeTransposedScaledBlockDCTFloat() from the corresponding 8x8 block of
// the image. Note that the whole coefficient image is scaled by 1/64
// afterwards, so that this is exactly the inverse of TransposedScaledIDCT().
// REQUIRES: coeffs.xsize() == 8*N, coeffs.ysize() == 8*M
Image3F TransposedScaledDCT(const Image3F& img);

// Final scaling factors of outputs/inputs in the Arai, Agui, and Nakajima
// algorithm computing the DCT/IDCT.
// The algorithm is described in the book JPEG: Still Image Data Compression
// Standard, section 4.3.5.
static const float kIDCTScales[8] = {
    0.3535533906f, 0.4903926402f, 0.4619397663f, 0.4157348062f,
    0.3535533906f, 0.2777851165f, 0.1913417162f, 0.0975451610f};

static const float kRecipIDCTScales[8] = {
    1.0 / 0.3535533906f, 1.0 / 0.4903926402f, 1.0 / 0.4619397663f,
    1.0 / 0.4157348062f, 1.0 / 0.3535533906f, 1.0 / 0.2777851165f,
    1.0 / 0.1913417162f, 1.0 / 0.0975451610f};

// See "Steerable Discrete Cosine Transform", Fracastoro G., Fosson S., Magli
// E., https://arxiv.org/pdf/1610.09152.pdf
void RotateDCT(float angle, float block[64]);

using DCTDesc =
    SIMD_NAMESPACE::Part<float, SIMD_MIN(8, SIMD_NAMESPACE::Full<float>::N)>;

#if SIMD_TARGET_VALUE == SIMD_AVX2

// Each vector holds one row of the input/output block.
template <class V>
PIK_INLINE void TransposeBlock_AVX2(V& i0, V& i1, V& i2, V& i3, V& i4, V& i5,
                                    V& i6, V& i7) {
  // Surprisingly, this straightforward implementation (24 cycles on port5) is
  // faster than load128+insert and load_dup128+concat_hi_lo+blend.
  const auto q0 = interleave_lo(i0, i2);
  const auto q1 = interleave_lo(i1, i3);
  const auto q2 = interleave_hi(i0, i2);
  const auto q3 = interleave_hi(i1, i3);
  const auto q4 = interleave_lo(i4, i6);
  const auto q5 = interleave_lo(i5, i7);
  const auto q6 = interleave_hi(i4, i6);
  const auto q7 = interleave_hi(i5, i7);

  const auto r0 = interleave_lo(q0, q1);
  const auto r1 = interleave_hi(q0, q1);
  const auto r2 = interleave_lo(q2, q3);
  const auto r3 = interleave_hi(q2, q3);
  const auto r4 = interleave_lo(q4, q5);
  const auto r5 = interleave_hi(q4, q5);
  const auto r6 = interleave_lo(q6, q7);
  const auto r7 = interleave_hi(q6, q7);

  i0 = concat_lo_lo(r4, r0);
  i1 = concat_lo_lo(r5, r1);
  i2 = concat_lo_lo(r6, r2);
  i3 = concat_lo_lo(r7, r3);
  i4 = concat_hi_hi(r4, r0);
  i5 = concat_hi_hi(r5, r1);
  i6 = concat_hi_hi(r6, r2);
  i7 = concat_hi_hi(r7, r3);
}

#endif  // SIMD_TARGET_VALUE == SIMD_AVX2

PIK_INLINE void TransposeBlock(float* PIK_RESTRICT block) {
#if SIMD_TARGET_VALUE == SIMD_AVX2
  const DCTDesc d;
  static_assert(d.N == 8, "Wrong vector size, must match block width");
  auto i0 = load(d, block + 0 * d.N);
  auto i1 = load(d, block + 1 * d.N);
  auto i2 = load(d, block + 2 * d.N);
  auto i3 = load(d, block + 3 * d.N);
  auto i4 = load(d, block + 4 * d.N);
  auto i5 = load(d, block + 5 * d.N);
  auto i6 = load(d, block + 6 * d.N);
  auto i7 = load(d, block + 7 * d.N);
  TransposeBlock_AVX2(i0, i1, i2, i3, i4, i5, i6, i7);
  store(i0, d, block + 0 * d.N);
  store(i1, d, block + 1 * d.N);
  store(i2, d, block + 2 * d.N);
  store(i3, d, block + 3 * d.N);
  store(i4, d, block + 4 * d.N);
  store(i5, d, block + 5 * d.N);
  store(i6, d, block + 6 * d.N);
  store(i7, d, block + 7 * d.N);
#elif SIMD_TARGET_VALUE == SIMD_NONE
  // https://en.wikipedia.org/wiki/In-place_matrix_transposition#Square_matrices
  for (int n = 0; n < 8 - 1; ++n) {
    for (int m = n + 1; m < 8; ++m) {
      // Swap
      const float tmp = block[m * 8 + n];
      block[m * 8 + n] = block[n * 8 + m];
      block[n * 8 + m] = tmp;
    }
  }
#else  // generic 128-bit
  const SIMD_NAMESPACE::Part<float, 4> d;
  const auto p0L = load(d, block + 0);
  const auto p0H = load(d, block + 4);
  const auto p1L = load(d, block + 8);
  const auto p1H = load(d, block + 12);
  const auto p2L = load(d, block + 16);
  const auto p2H = load(d, block + 20);
  const auto p3L = load(d, block + 24);
  const auto p3H = load(d, block + 28);
  const auto p4L = load(d, block + 32);
  const auto p4H = load(d, block + 36);
  const auto p5L = load(d, block + 40);
  const auto p5H = load(d, block + 44);
  const auto p6L = load(d, block + 48);
  const auto p6H = load(d, block + 52);
  const auto p7L = load(d, block + 56);
  const auto p7H = load(d, block + 60);

  const auto q0L = interleave_lo(p0L, p2L);
  const auto q0H = interleave_lo(p0H, p2H);
  const auto q1L = interleave_lo(p1L, p3L);
  const auto q1H = interleave_lo(p1H, p3H);
  const auto q2L = interleave_hi(p0L, p2L);
  const auto q2H = interleave_hi(p0H, p2H);
  const auto q3L = interleave_hi(p1L, p3L);
  const auto q3H = interleave_hi(p1H, p3H);
  const auto q4L = interleave_lo(p4L, p6L);
  const auto q4H = interleave_lo(p4H, p6H);
  const auto q5L = interleave_lo(p5L, p7L);
  const auto q5H = interleave_lo(p5H, p7H);
  const auto q6L = interleave_hi(p4L, p6L);
  const auto q6H = interleave_hi(p4H, p6H);
  const auto q7L = interleave_hi(p5L, p7L);
  const auto q7H = interleave_hi(p5H, p7H);

  const auto r0L = interleave_lo(q0L, q1L);
  const auto r0H = interleave_lo(q0H, q1H);
  const auto r1L = interleave_hi(q0L, q1L);
  const auto r1H = interleave_hi(q0H, q1H);
  const auto r2L = interleave_lo(q2L, q3L);
  const auto r2H = interleave_lo(q2H, q3H);
  const auto r3L = interleave_hi(q2L, q3L);
  const auto r3H = interleave_hi(q2H, q3H);
  const auto r4L = interleave_lo(q4L, q5L);
  const auto r4H = interleave_lo(q4H, q5H);
  const auto r5L = interleave_hi(q4L, q5L);
  const auto r5H = interleave_hi(q4H, q5H);
  const auto r6L = interleave_lo(q6L, q7L);
  const auto r6H = interleave_lo(q6H, q7H);
  const auto r7L = interleave_hi(q6L, q7L);
  const auto r7H = interleave_hi(q6H, q7H);

  store(r0L, d, block + 0);
  store(r4L, d, block + 4);
  store(r1L, d, block + 8);
  store(r5L, d, block + 12);
  store(r2L, d, block + 16);
  store(r6L, d, block + 20);
  store(r3L, d, block + 24);
  store(r7L, d, block + 28);
  store(r0H, d, block + 32);
  store(r4H, d, block + 36);
  store(r1H, d, block + 40);
  store(r5H, d, block + 44);
  store(r2H, d, block + 48);
  store(r6H, d, block + 52);
  store(r3H, d, block + 56);
  store(r7H, d, block + 60);
#endif
}

// Adapters for Column[I]DCT source/destination:

// Block: contiguous
class FromBlock {
 public:
  explicit FromBlock(const float* block) : block_(block) {}

  PIK_INLINE DCTDesc::V Load(const size_t row, size_t i) const {
    return load(DCTDesc(), block_ + row * 8 + i);
  }

 private:
  const float* block_;
};
class ToBlock {
 public:
  explicit ToBlock(float* block) : block_(block) {}

  PIK_INLINE void Store(const DCTDesc::V& v, const size_t row,
                        const size_t i) const {
    store(v, DCTDesc(), block_ + row * 8 + i);
  }

 private:
  float* block_;
};
class ScaleToBlock {
 public:
  explicit ScaleToBlock(float* block)
      : block_(block), mul_(set1(DCTDesc(), 1.0f / 64)) {}

  PIK_INLINE void Store(const DCTDesc::V& v, const size_t row,
                        const size_t i) const {
    store(v * mul_, DCTDesc(), block_ + row * 8 + i);
  }

 private:
  float* block_;
  DCTDesc::V mul_;
};

// Lines: 8x8 within a larger image
class FromLines {
 public:
  FromLines(const float* top_left, size_t stride)
      : top_left_(top_left), stride_(stride) {}

  PIK_INLINE DCTDesc::V Load(const size_t row, const size_t i) const {
    return load(DCTDesc(), top_left_ + row * stride_ + i);
  }

 private:
  const float* top_left_;
  size_t stride_;  // move to next line by adding this to pointer
};
class ToLines {
 public:
  ToLines(float* top_left, size_t stride)
      : top_left_(top_left), stride_(stride) {}

  PIK_INLINE void Store(const DCTDesc::V& v, const size_t row,
                        const size_t i) const {
    store(v, DCTDesc(), top_left_ + row * stride_ + i);
  }

 private:
  float* top_left_;
  size_t stride_;  // move to next line by adding this to pointer
};

#if SIMD_TARGET_VALUE == SIMD_AVX2

// Each vector holds one row of the input/output block.
template <class V>
PIK_INLINE void ColumnDCT_AVX2(V& i0, V& i1, V& i2, V& i3, V& i4, V& i5, V& i6,
                               V& i7) {
  const DCTDesc d;

  const auto c1 = set1(d, 0.707106781186548f);
  const auto c2 = set1(d, -0.382683432365090f);
  const auto c3 = set1(d, 1.30656296487638f);
  const auto c4 = set1(d, 0.541196100146197f);

  const auto t00 = i0 + i7;
  const auto t01 = i0 - i7;
  const auto t02 = i3 + i4;
  const auto t03 = i3 - i4;
  const auto t04 = i2 + i5;
  const auto t05 = i2 - i5;
  const auto t06 = i1 + i6;
  const auto t07 = i1 - i6;
  const auto t08 = t00 + t02;
  const auto t09 = t00 - t02;
  const auto t10 = t06 + t04;
  const auto t11 = t06 - t04;
  const auto t12 = t07 + t05;
  const auto t13 = t01 + t07;
  const auto t14 = t05 + t03;
  const auto t15 = t11 + t09;
  const auto t16 = t13 - t14;
  const auto t17 = c1 * t15;
  const auto t18 = c1 * t12;
  const auto t19 = c2 * t16;
  const auto t20 = t01 + t18;
  const auto t21 = t01 - t18;
  const auto t22 = mul_add(c3, t13, t19);
  const auto t23 = mul_add(c4, t14, t19);
  i0 = t08 + t10;
  i1 = t20 + t22;
  i2 = t09 + t17;
  i3 = t21 - t23;
  i4 = t08 - t10;
  i5 = t21 + t23;
  i6 = t09 - t17;
  i7 = t20 - t22;
}

// Each vector holds one row of the input/output block.
template <class V>
PIK_INLINE void ColumnIDCT_AVX2(V& i0, V& i1, V& i2, V& i3, V& i4, V& i5, V& i6,
                                V& i7) {
  const DCTDesc d;

  const auto c1 = set1(d, 1.41421356237310f);
  const auto c2 = set1(d, 0.76536686473018f);
  const auto c3 = set1(d, 2.61312592975275f);
  const auto c4 = set1(d, 1.08239220029239f);

  const auto t00 = i0 + i4;
  const auto t01 = i0 - i4;
  const auto t02 = i2 + i6;
  const auto t03 = i2 - i6;
  const auto t04 = i1 + i7;
  const auto t05 = i1 - i7;
  const auto t06 = i5 + i3;
  const auto t07 = i5 - i3;
  const auto t08 = t04 + t06;
  const auto t09 = t04 - t06;
  const auto t10 = t00 + t02;
  const auto t11 = t00 - t02;
  const auto t12 = t05 + t07;
  const auto t13 = c2 * t12;
  const auto t14 = SIMD_NAMESPACE::ext::mul_subtract(c1, t03, t02);
  const auto t15 = t01 + t14;
  const auto t16 = t01 - t14;
  const auto t17 = SIMD_NAMESPACE::ext::mul_subtract(c3, t05, t13);
  const auto t18 = mul_add(c4, t07, t13);
  const auto t19 = t08 - t17;
  const auto t20 = mul_add(c1, t09, t19);
  const auto t21 = t18 - t20;
  i0 = t10 + t08;
  i1 = t15 - t19;
  i2 = t16 + t20;
  i3 = t11 + t21;
  i4 = t11 - t21;
  i5 = t16 - t20;
  i6 = t15 + t19;
  i7 = t10 - t08;
}

#else

template <class From, class To>
PIK_INLINE void ColumnDCT(const From& from, const To& to) {
  using namespace SIMD_NAMESPACE;
  const DCTDesc d;

  const auto c1 = set1(d, 0.707106781186548f);
  const auto c2 = set1(d, -0.382683432365090f);
  const auto c3 = set1(d, 1.30656296487638f);
  const auto c4 = set1(d, 0.541196100146197f);

  for (size_t i = 0; i < 8; i += d.N) {
    const auto i0 = from.Load(0, i);
    const auto i1 = from.Load(1, i);
    const auto i2 = from.Load(2, i);
    const auto i3 = from.Load(3, i);
    const auto i4 = from.Load(4, i);
    const auto i5 = from.Load(5, i);
    const auto i6 = from.Load(6, i);
    const auto i7 = from.Load(7, i);
    const auto t00 = i0 + i7;
    const auto t01 = i0 - i7;
    const auto t02 = i3 + i4;
    const auto t03 = i3 - i4;
    const auto t04 = i2 + i5;
    const auto t05 = i2 - i5;
    const auto t06 = i1 + i6;
    const auto t07 = i1 - i6;
    const auto t08 = t00 + t02;
    const auto t09 = t00 - t02;
    const auto t10 = t06 + t04;
    const auto t11 = t06 - t04;
    const auto t12 = t07 + t05;
    const auto t13 = t01 + t07;
    const auto t14 = t05 + t03;
    const auto t15 = t11 + t09;
    const auto t16 = t13 - t14;
    const auto t17 = c1 * t15;
    const auto t18 = c1 * t12;
    const auto t19 = c2 * t16;
    const auto t20 = t01 + t18;
    const auto t21 = t01 - t18;
    const auto t22 = mul_add(c3, t13, t19);
    const auto t23 = mul_add(c4, t14, t19);
    to.Store(t08 + t10, 0, i);
    to.Store(t20 + t22, 1, i);
    to.Store(t09 + t17, 2, i);
    to.Store(t21 - t23, 3, i);
    to.Store(t08 - t10, 4, i);
    to.Store(t21 + t23, 5, i);
    to.Store(t09 - t17, 6, i);
    to.Store(t20 - t22, 7, i);
  }
}

template <class From, class To, class DC_Op>
PIK_INLINE void ColumnIDCT(const From& from, const To& to, const DC_Op dc_op) {
  using namespace SIMD_NAMESPACE;
  const DCTDesc d;

  const auto c1 = set1(d, 1.41421356237310f);
  const auto c2 = set1(d, 0.76536686473018f);
  const auto c3 = set1(d, 2.61312592975275f);
  const auto c4 = set1(d, 1.08239220029239f);

  for (size_t i = 0; i < 8; i += d.N) {
    // Apply dc_op to the first value (= DC)
    const auto i0 = (i == 0) ? dc_op(from.Load(0, i)) : from.Load(0, i);
    const auto i1 = from.Load(1, i);
    const auto i2 = from.Load(2, i);
    const auto i3 = from.Load(3, i);
    const auto i4 = from.Load(4, i);
    const auto i5 = from.Load(5, i);
    const auto i6 = from.Load(6, i);
    const auto i7 = from.Load(7, i);
    const auto t00 = i0 + i4;
    const auto t01 = i0 - i4;
    const auto t02 = i2 + i6;
    const auto t03 = i2 - i6;
    const auto t04 = i1 + i7;
    const auto t05 = i1 - i7;
    const auto t06 = i5 + i3;
    const auto t07 = i5 - i3;
    const auto t08 = t04 + t06;
    const auto t09 = t04 - t06;
    const auto t10 = t00 + t02;
    const auto t11 = t00 - t02;
    const auto t12 = t05 + t07;
    const auto t13 = c2 * t12;
    const auto t14 = SIMD_NAMESPACE::ext::mul_subtract(c1, t03, t02);
    const auto t15 = t01 + t14;
    const auto t16 = t01 - t14;
    const auto t17 = SIMD_NAMESPACE::ext::mul_subtract(c3, t05, t13);
    const auto t18 = mul_add(c4, t07, t13);
    const auto t19 = t08 - t17;
    const auto t20 = mul_add(c1, t09, t19);
    const auto t21 = t18 - t20;
    to.Store(t10 + t08, 0, i);
    to.Store(t15 - t19, 1, i);
    to.Store(t16 + t20, 2, i);
    to.Store(t11 + t21, 3, i);
    to.Store(t11 - t21, 4, i);
    to.Store(t16 - t20, 5, i);
    to.Store(t15 + t19, 6, i);
    to.Store(t10 - t08, 7, i);
  }
}

#endif  // SIMD_TARGET_VALUE == SIMD_AVX2

// Same as ComputeBlockDCTFloat(), but the output is further transformed with
// the following:
//   block'[8 * ky + kx] =
//     block[8 * kx + ky] * 64.0 * kIDCTScales[kx] * kIDCTScales[ky]
template <class From, class To>
static PIK_INLINE void ComputeTransposedScaledBlockDCTFloat(const From& from,
                                                            const To& to) {
#if SIMD_TARGET_VALUE == SIMD_AVX2
  auto i0 = from.Load(0, 0);
  auto i1 = from.Load(1, 0);
  auto i2 = from.Load(2, 0);
  auto i3 = from.Load(3, 0);
  auto i4 = from.Load(4, 0);
  auto i5 = from.Load(5, 0);
  auto i6 = from.Load(6, 0);
  auto i7 = from.Load(7, 0);

  ColumnDCT_AVX2(i0, i1, i2, i3, i4, i5, i6, i7);
  TransposeBlock_AVX2(i0, i1, i2, i3, i4, i5, i6, i7);
  ColumnDCT_AVX2(i0, i1, i2, i3, i4, i5, i6, i7);

  to.Store(i0, 0, 0);
  to.Store(i1, 1, 0);
  to.Store(i2, 2, 0);
  to.Store(i3, 3, 0);
  to.Store(i4, 4, 0);
  to.Store(i5, 5, 0);
  to.Store(i6, 6, 0);
  to.Store(i7, 7, 0);
#else
  SIMD_ALIGN float block[64];
  ColumnDCT(from, ToBlock(block));
  TransposeBlock(block);
  ColumnDCT(FromBlock(block), to);
#endif
}

// Sets the DC lane to zero.
struct DC_Zero {
  template <class V>
  V operator()(const V v) const {
    using namespace SIMD_NAMESPACE;
    const Full<float> df;
    const Full<int32_t> di;
    // Negated so upper lanes can default-initialize to 0.
    SIMD_ALIGN static constexpr int32_t neg_mask_lanes[di.N] = {-1};
    const auto neg_mask = cast_to(df, load(di, neg_mask_lanes));
    return andnot(neg_mask, v);
  }
};

struct DC_Unchanged {
  template <class V>
  constexpr V operator()(const V v) const {
    return v;
  }
};

// Same as ComputeBlockIDCTFloat(), but the input is first transformed with
// the following:
//   block'[8 * ky + kx] =
//     block[8 * kx + ky] / (kIDCTScales[kx] * kIDCTScales[ky])
template <class From, class To, class DC_Op>
static PIK_INLINE void ComputeTransposedScaledBlockIDCTFloat(
    const From& from, const To& to, const DC_Op dc_op) {
#if SIMD_TARGET_VALUE == SIMD_AVX2
  auto i0 = dc_op(from.Load(0, 0));
  auto i1 = from.Load(1, 0);
  auto i2 = from.Load(2, 0);
  auto i3 = from.Load(3, 0);
  auto i4 = from.Load(4, 0);
  auto i5 = from.Load(5, 0);
  auto i6 = from.Load(6, 0);
  auto i7 = from.Load(7, 0);

  ColumnIDCT_AVX2(i0, i1, i2, i3, i4, i5, i6, i7);
  TransposeBlock_AVX2(i0, i1, i2, i3, i4, i5, i6, i7);
  ColumnIDCT_AVX2(i0, i1, i2, i3, i4, i5, i6, i7);

  to.Store(i0, 0, 0);
  to.Store(i1, 1, 0);
  to.Store(i2, 2, 0);
  to.Store(i3, 3, 0);
  to.Store(i4, 4, 0);
  to.Store(i5, 5, 0);
  to.Store(i6, 6, 0);
  to.Store(i7, 7, 0);
#else
  SIMD_ALIGN float block[64];
  ColumnIDCT(from, ToBlock(block), dc_op);
  TransposeBlock(block);
  ColumnIDCT(FromBlock(block), to, DC_Unchanged());
#endif
}

// Requires that block is 32-bytes aligned.
static PIK_INLINE void ComputeTransposedScaledBlockDCTFloat(float block[64]) {
  ComputeTransposedScaledBlockDCTFloat(FromBlock(block), ToBlock(block));
}

// Requires that block is 32-bytes aligned.
template <class DC_Op>
static PIK_INLINE void ComputeTransposedScaledBlockIDCTFloat(
    float block[64], const DC_Op dc_op) {
  ComputeTransposedScaledBlockIDCTFloat(FromBlock(block), ToBlock(block),
                                        dc_op);
}

}  // namespace pik

#endif  // DCT_H_
