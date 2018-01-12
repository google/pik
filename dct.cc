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

#include "dct.h"

#include "arch_specific.h"
#include "compiler_specific.h"
#include "simd/simd.h"

namespace pik {

using namespace SIMD_NAMESPACE;
constexpr size_t kMaxLanes = Full<float>::N;
using D = Part<float, SIMD_MIN(8, kMaxLanes)>;

void TransposeBlock(float block[64]) {
#if SIMD_TARGET_VALUE == SIMD_AVX2
  const D d;
  const auto p0 = load(d, block + 0);
  const auto p1 = load(d, block + 8);
  const auto p2 = load(d, block + 16);
  const auto p3 = load(d, block + 24);
  const auto p4 = load(d, block + 32);
  const auto p5 = load(d, block + 40);
  const auto p6 = load(d, block + 48);
  const auto p7 = load(d, block + 56);
  const auto q0 = interleave_lo(p0, p2);
  const auto q1 = interleave_lo(p1, p3);
  const auto q2 = interleave_hi(p0, p2);
  const auto q3 = interleave_hi(p1, p3);
  const auto q4 = interleave_lo(p4, p6);
  const auto q5 = interleave_lo(p5, p7);
  const auto q6 = interleave_hi(p4, p6);
  const auto q7 = interleave_hi(p5, p7);
  const auto r0 = interleave_lo(q0, q1);
  const auto r1 = interleave_hi(q0, q1);
  const auto r2 = interleave_lo(q2, q3);
  const auto r3 = interleave_hi(q2, q3);
  const auto r4 = interleave_lo(q4, q5);
  const auto r5 = interleave_hi(q4, q5);
  const auto r6 = interleave_lo(q6, q7);
  const auto r7 = interleave_hi(q6, q7);
  const auto s0 = concat_lo_lo(r4, r0);
  const auto s1 = concat_lo_lo(r5, r1);
  const auto s2 = concat_lo_lo(r6, r2);
  const auto s3 = concat_lo_lo(r7, r3);
  const auto s4 = concat_hi_hi(r4, r0);
  const auto s5 = concat_hi_hi(r5, r1);
  const auto s6 = concat_hi_hi(r6, r2);
  const auto s7 = concat_hi_hi(r7, r3);
  store(s0, d, block + 0);
  store(s1, d, block + 8);
  store(s2, d, block + 16);
  store(s3, d, block + 24);
  store(s4, d, block + 32);
  store(s5, d, block + 40);
  store(s6, d, block + 48);
  store(s7, d, block + 56);
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
  const Part<float, 4> d;
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

PIK_INLINE void ColumnIDCT(float block[64]) {
  const D d;
  for (size_t i = 0; i < 8; i += d.N) {
    const auto i0 = load(d, block + 0 + i);
    const auto i1 = load(d, block + 8 + i);
    const auto i2 = load(d, block + 16 + i);
    const auto i3 = load(d, block + 24 + i);
    const auto i4 = load(d, block + 32 + i);
    const auto i5 = load(d, block + 40 + i);
    const auto i6 = load(d, block + 48 + i);
    const auto i7 = load(d, block + 56 + i);
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
    const auto t14 = mul_sub(c1, t03, t02);
    const auto t15 = t01 + t14;
    const auto t16 = t01 - t14;
    const auto t17 = mul_sub(c3, t05, t13);
    const auto t18 = mul_add(c4, t07, t13);
    const auto t19 = t17 - t08;
    const auto t20 = mul_sub(c1, t09, t19);
    const auto t21 = t18 - t20;
    store(t10 + t08, d, block + 0 + i);
    store(t15 + t19, d, block + 8 + i);
    store(t16 + t20, d, block + 16 + i);
    store(t11 + t21, d, block + 24 + i);
    store(t11 - t21, d, block + 32 + i);
    store(t16 - t20, d, block + 40 + i);
    store(t15 - t19, d, block + 48 + i);
    store(t10 - t08, d, block + 56 + i);
  }
}

PIK_INLINE void ColumnDCT(float block[64]) {
  const D d;
  for (size_t i = 0; i < 8; i += d.N) {
    const auto i0 = load(d, block + 0 + i);
    const auto i1 = load(d, block + 8 + i);
    const auto i2 = load(d, block + 16 + i);
    const auto i3 = load(d, block + 24 + i);
    const auto i4 = load(d, block + 32 + i);
    const auto i5 = load(d, block + 40 + i);
    const auto i6 = load(d, block + 48 + i);
    const auto i7 = load(d, block + 56 + i);
    const auto c1 = set1(d, 0.707106781186548f);
    const auto c2 = set1(d, 0.382683432365090f);
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
    const auto t22 = mul_sub(c3, t13, t19);
    const auto t23 = mul_sub(c4, t14, t19);
    store(t08 + t10, d, block + 0 + i);
    store(t20 + t22, d, block + 8 + i);
    store(t09 + t17, d, block + 16 + i);
    store(t21 - t23, d, block + 24 + i);
    store(t08 - t10, d, block + 32 + i);
    store(t21 + t23, d, block + 40 + i);
    store(t09 - t17, d, block + 48 + i);
    store(t20 - t22, d, block + 56 + i);
  }
}

void ComputeTransposedScaledBlockDCTFloat(float block[64]) {
  ColumnDCT(block);
  TransposeBlock(block);
  ColumnDCT(block);
};

void ComputeTransposedScaledBlockIDCTFloat(float block[64]) {
  ColumnIDCT(block);
  TransposeBlock(block);
  ColumnIDCT(block);
}

void ComputeBlockDCTFloat(float block[64]) {
  ComputeTransposedScaledBlockDCTFloat(block);
  TransposeBlock(block);
  for (int y = 0; y < 8; ++y) {
    for (int x = 0; x < 8; ++x) {
      block[8 * y + x] /= 64.0f * kIDCTScales[y] * kIDCTScales[x];
    }
  }
}

void ComputeBlockIDCTFloat(float block[64]) {
  for (int y = 0; y < 8; ++y) {
    for (int x = 0; x < 8; ++x) {
      block[8 * y + x] *= kIDCTScales[y] * kIDCTScales[x];
    }
  }
  TransposeBlock(block);
  ComputeTransposedScaledBlockIDCTFloat(block);
}

}  // namespace pik
