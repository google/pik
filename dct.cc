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

PIK_INLINE void TransposeBlock(float block[64]) {
  const Full<float, AVX2> d;
  // TODO(user) Add non-AVX2 fallback.
  const auto p0 = load(d, &block[0]);
  const auto p1 = load(d, &block[8]);
  const auto p2 = load(d, &block[16]);
  const auto p3 = load(d, &block[24]);
  const auto p4 = load(d, &block[32]);
  const auto p5 = load(d, &block[40]);
  const auto p6 = load(d, &block[48]);
  const auto p7 = load(d, &block[56]);
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
  using V = Full<float, AVX2>::V;
  store(V(_mm256_permute2f128_ps(r0, r4, 0x20)), d, &block[0]);
  store(V(_mm256_permute2f128_ps(r1, r5, 0x20)), d, &block[8]);
  store(V(_mm256_permute2f128_ps(r2, r6, 0x20)), d, &block[16]);
  store(V(_mm256_permute2f128_ps(r3, r7, 0x20)), d, &block[24]);
  store(V(_mm256_permute2f128_ps(r0, r4, 0x31)), d, &block[32]);
  store(V(_mm256_permute2f128_ps(r1, r5, 0x31)), d, &block[40]);
  store(V(_mm256_permute2f128_ps(r2, r6, 0x31)), d, &block[48]);
  store(V(_mm256_permute2f128_ps(r3, r7, 0x31)), d, &block[56]);
}

PIK_INLINE void ColumnIDCT(float block[64]) {
  const Full<float, AVX2> d;
  // TODO(user) Add non-AVX2 fallback.
  const auto i0 = load(d, &block[0]);
  const auto i1 = load(d, &block[8]);
  const auto i2 = load(d, &block[16]);
  const auto i3 = load(d, &block[24]);
  const auto i4 = load(d, &block[32]);
  const auto i5 = load(d, &block[40]);
  const auto i6 = load(d, &block[48]);
  const auto i7 = load(d, &block[56]);
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
  store(t10 + t08, d, &block[0]);
  store(t15 + t19, d, &block[8]);
  store(t16 + t20, d, &block[16]);
  store(t11 + t21, d, &block[24]);
  store(t11 - t21, d, &block[32]);
  store(t16 - t20, d, &block[40]);
  store(t15 - t19, d, &block[48]);
  store(t10 - t08, d, &block[56]);
}

PIK_INLINE void ColumnDCT(float block[64]) {
  const Full<float, AVX2> d;
  // TODO(user) Add non-AVX2 fallback.
  const auto i0 = load(d, &block[0]);
  const auto i1 = load(d, &block[8]);
  const auto i2 = load(d, &block[16]);
  const auto i3 = load(d, &block[24]);
  const auto i4 = load(d, &block[32]);
  const auto i5 = load(d, &block[40]);
  const auto i6 = load(d, &block[48]);
  const auto i7 = load(d, &block[56]);
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
  store(t08 + t10, d, &block[0]);
  store(t20 + t22, d, &block[8]);
  store(t09 + t17, d, &block[16]);
  store(t21 - t23, d, &block[24]);
  store(t08 - t10, d, &block[32]);
  store(t21 + t23, d, &block[40]);
  store(t09 - t17, d, &block[48]);
  store(t20 - t22, d, &block[56]);
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
