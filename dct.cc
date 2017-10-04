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
using V = vec256<float>;

PIK_INLINE void TransposeBlock(float block[64]) {
  // TODO(user) Add non-AVX2 fallback.
  const V p0 = load(V(), &block[0]);
  const V p1 = load(V(), &block[8]);
  const V p2 = load(V(), &block[16]);
  const V p3 = load(V(), &block[24]);
  const V p4 = load(V(), &block[32]);
  const V p5 = load(V(), &block[40]);
  const V p6 = load(V(), &block[48]);
  const V p7 = load(V(), &block[56]);
  const V q0 = interleave_lo(p0, p2);
  const V q1 = interleave_lo(p1, p3);
  const V q2 = interleave_hi(p0, p2);
  const V q3 = interleave_hi(p1, p3);
  const V q4 = interleave_lo(p4, p6);
  const V q5 = interleave_lo(p5, p7);
  const V q6 = interleave_hi(p4, p6);
  const V q7 = interleave_hi(p5, p7);
  const V r0 = interleave_lo(q0, q1);
  const V r1 = interleave_hi(q0, q1);
  const V r2 = interleave_lo(q2, q3);
  const V r3 = interleave_hi(q2, q3);
  const V r4 = interleave_lo(q4, q5);
  const V r5 = interleave_hi(q4, q5);
  const V r6 = interleave_lo(q6, q7);
  const V r7 = interleave_hi(q6, q7);
  store(V(_mm256_permute2f128_ps(r0, r4, 0x20)), &block[0]);
  store(V(_mm256_permute2f128_ps(r1, r5, 0x20)), &block[8]);
  store(V(_mm256_permute2f128_ps(r2, r6, 0x20)), &block[16]);
  store(V(_mm256_permute2f128_ps(r3, r7, 0x20)), &block[24]);
  store(V(_mm256_permute2f128_ps(r0, r4, 0x31)), &block[32]);
  store(V(_mm256_permute2f128_ps(r1, r5, 0x31)), &block[40]);
  store(V(_mm256_permute2f128_ps(r2, r6, 0x31)), &block[48]);
  store(V(_mm256_permute2f128_ps(r3, r7, 0x31)), &block[56]);
}

PIK_INLINE void ColumnIDCT(float block[64]) {
  // TODO(user) Add non-AVX2 fallback.
  const V i0 = load(V(), &block[0]);
  const V i1 = load(V(), &block[8]);
  const V i2 = load(V(), &block[16]);
  const V i3 = load(V(), &block[24]);
  const V i4 = load(V(), &block[32]);
  const V i5 = load(V(), &block[40]);
  const V i6 = load(V(), &block[48]);
  const V i7 = load(V(), &block[56]);
  const V c1 = set1(V(), 1.41421356237310f);
  const V c2 = set1(V(), 0.76536686473018f);
  const V c3 = set1(V(), 2.61312592975275f);
  const V c4 = set1(V(), 1.08239220029239f);
  const V t00 = i0 + i4;
  const V t01 = i0 - i4;
  const V t02 = i2 + i6;
  const V t03 = i2 - i6;
  const V t04 = i1 + i7;
  const V t05 = i1 - i7;
  const V t06 = i5 + i3;
  const V t07 = i5 - i3;
  const V t08 = t04 + t06;
  const V t09 = t04 - t06;
  const V t10 = t00 + t02;
  const V t11 = t00 - t02;
  const V t12 = t05 + t07;
  const V t13 = c2 * t12;
  const V t14 = mul_sub(c1, t03, t02);
  const V t15 = t01 + t14;
  const V t16 = t01 - t14;
  const V t17 = mul_sub(c3, t05, t13);
  const V t18 = mul_add(c4, t07, t13);
  const V t19 = t17 - t08;
  const V t20 = mul_sub(c1, t09, t19);
  const V t21 = t18 - t20;
  store(t10 + t08, &block[0]);
  store(t15 + t19, &block[8]);
  store(t16 + t20, &block[16]);
  store(t11 + t21, &block[24]);
  store(t11 - t21, &block[32]);
  store(t16 - t20, &block[40]);
  store(t15 - t19, &block[48]);
  store(t10 - t08, &block[56]);
}

PIK_INLINE void ColumnDCT(float block[64]) {
  // TODO(user) Add non-AVX2 fallback.
  const V i0 = load(V(), &block[0]);
  const V i1 = load(V(), &block[8]);
  const V i2 = load(V(), &block[16]);
  const V i3 = load(V(), &block[24]);
  const V i4 = load(V(), &block[32]);
  const V i5 = load(V(), &block[40]);
  const V i6 = load(V(), &block[48]);
  const V i7 = load(V(), &block[56]);
  const V c1 = set1(V(), 0.707106781186548f);
  const V c2 = set1(V(), 0.382683432365090f);
  const V c3 = set1(V(), 1.30656296487638f);
  const V c4 = set1(V(), 0.541196100146197f);
  const V t00 = i0 + i7;
  const V t01 = i0 - i7;
  const V t02 = i3 + i4;
  const V t03 = i3 - i4;
  const V t04 = i2 + i5;
  const V t05 = i2 - i5;
  const V t06 = i1 + i6;
  const V t07 = i1 - i6;
  const V t08 = t00 + t02;
  const V t09 = t00 - t02;
  const V t10 = t06 + t04;
  const V t11 = t06 - t04;
  const V t12 = t07 + t05;
  const V t13 = t01 + t07;
  const V t14 = t05 + t03;
  const V t15 = t11 + t09;
  const V t16 = t13 - t14;
  const V t17 = c1 * t15;
  const V t18 = c1 * t12;
  const V t19 = c2 * t16;
  const V t20 = t01 + t18;
  const V t21 = t01 - t18;
  const V t22 = mul_sub(c3, t13, t19);
  const V t23 = mul_sub(c4, t14, t19);
  store(t08 + t10, &block[0]);
  store(t20 + t22, &block[8]);
  store(t09 + t17, &block[16]);
  store(t21 - t23, &block[24]);
  store(t08 - t10, &block[32]);
  store(t21 + t23, &block[40]);
  store(t09 - t17, &block[48]);
  store(t20 - t22, &block[56]);
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
