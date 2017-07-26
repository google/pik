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

#include "compiler_specific.h"
#include "vector256.h"
#include "vector128.h"

namespace pik {

PIK_INLINE void TransposeBlock(float block[64]) {
  using namespace PIK_TARGET_NAME;
  using V = V8x32F;
  const V p0 = Load<V>(&block[0]);
  const V p1 = Load<V>(&block[8]);
  const V p2 = Load<V>(&block[16]);
  const V p3 = Load<V>(&block[24]);
  const V p4 = Load<V>(&block[32]);
  const V p5 = Load<V>(&block[40]);
  const V p6 = Load<V>(&block[48]);
  const V p7 = Load<V>(&block[56]);
  const V q0(_mm256_unpacklo_ps(p0, p2));
  const V q1(_mm256_unpacklo_ps(p1, p3));
  const V q2(_mm256_unpackhi_ps(p0, p2));
  const V q3(_mm256_unpackhi_ps(p1, p3));
  const V q4(_mm256_unpacklo_ps(p4, p6));
  const V q5(_mm256_unpacklo_ps(p5, p7));
  const V q6(_mm256_unpackhi_ps(p4, p6));
  const V q7(_mm256_unpackhi_ps(p5, p7));
  const V r0(_mm256_unpacklo_ps(q0, q1));
  const V r1(_mm256_unpackhi_ps(q0, q1));
  const V r2(_mm256_unpacklo_ps(q2, q3));
  const V r3(_mm256_unpackhi_ps(q2, q3));
  const V r4(_mm256_unpacklo_ps(q4, q5));
  const V r5(_mm256_unpackhi_ps(q4, q5));
  const V r6(_mm256_unpacklo_ps(q6, q7));
  const V r7(_mm256_unpackhi_ps(q6, q7));
  Store(_mm256_permute2f128_ps(r0, r4, 0x20), &block[ 0]);
  Store(_mm256_permute2f128_ps(r1, r5, 0x20), &block[ 8]);
  Store(_mm256_permute2f128_ps(r2, r6, 0x20), &block[16]);
  Store(_mm256_permute2f128_ps(r3, r7, 0x20), &block[24]);
  Store(_mm256_permute2f128_ps(r0, r4, 0x31), &block[32]);
  Store(_mm256_permute2f128_ps(r1, r5, 0x31), &block[40]);
  Store(_mm256_permute2f128_ps(r2, r6, 0x31), &block[48]);
  Store(_mm256_permute2f128_ps(r3, r7, 0x31), &block[56]);
}

PIK_INLINE void ColumnIDCT(float block[64]) {
  // TODO(user) Add non-AVX fallback.
  using namespace PIK_TARGET_NAME;
  int i;
#if defined __AVX__
  using V = V8x32F;
#else
  using V = V4x32F;
  for (i = 0; i<2; ++i) {
#endif
  const V i0 = Load<V>(&block[0]);
  const V i1 = Load<V>(&block[8]);
  const V i2 = Load<V>(&block[16]);
  const V i3 = Load<V>(&block[24]);
  const V i4 = Load<V>(&block[32]);
  const V i5 = Load<V>(&block[40]);
  const V i6 = Load<V>(&block[48]);
  const V i7 = Load<V>(&block[56]);
  const V c1(1.41421356237310f);
  const V c2(0.76536686473018f);
  const V c3(2.61312592975275f);
  const V c4(1.08239220029239f);
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
#if defined __AVX__
  const V t14(_mm256_fmsub_ps(c1, t03, t02));
#else
  const V t14(_mm_sub_ps(_mm_mul_ps(c1, t03), t02));
#endif
  const V t15 = t01 + t14;
  const V t16 = t01 - t14;
#if defined __AVX__
  const V t17(_mm256_fmsub_ps(c3, t05, t13));
  const V t18(_mm256_fmadd_ps(c4, t07, t13));
#else
  const V t17(_mm_sub_ps(_mm_mul_ps(c3, t05), t13));
  const V t18(_mm_add_ps(_mm_mul_ps(c4, t07), t13));
#endif
  const V t19 = t17 - t08;
#if defined __AVX__
  const V t20(_mm256_fmsub_ps(c1, t09, t19));
#else
  const V t20(_mm_sub_ps(_mm_mul_ps(c1, t09), t19));
#endif
  const V t21 = t18 - t20;
  Store(t10 + t08, &block[ 0]);
  Store(t15 + t19, &block[ 8]);
  Store(t16 + t20, &block[16]);
  Store(t11 + t21, &block[24]);
  Store(t11 - t21, &block[32]);
  Store(t16 - t20, &block[40]);
  Store(t15 - t19, &block[48]);
  Store(t10 - t08, &block[56]);
#if defined __AVX__
#else
  block += 4;
  }
#endif
}

PIK_INLINE void ColumnDCT(float block[64]) {
  // TODO(user) Add non-AVX fallback.
  using namespace PIK_TARGET_NAME;
  int i;
#if defined __AVX__
  using V = V8x32F;
#else
  using V = V4x32F;
  for(i=0;i<2;++i) {
#endif
  const V i0 = Load<V>(&block[0]);
  const V i1 = Load<V>(&block[8]);
  const V i2 = Load<V>(&block[16]);
  const V i3 = Load<V>(&block[24]);
  const V i4 = Load<V>(&block[32]);
  const V i5 = Load<V>(&block[40]);
  const V i6 = Load<V>(&block[48]);
  const V i7 = Load<V>(&block[56]);
  const V c1(0.707106781186548f);
  const V c2(0.382683432365090f);
  const V c3(1.30656296487638f);
  const V c4(0.541196100146197f);
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
#if defined __AVX__
  const V t22(_mm256_fmsub_ps(c3, t13, t19));
  const V t23(_mm256_fmsub_ps(c4, t14, t19));
#else
  const V t22(_mm_sub_ps(_mm_mul_ps(c3, t13), t19));
  const V t23(_mm_sub_ps(_mm_mul_ps(c4, t14), t19));
#endif
  Store(t08 + t10, &block[ 0]);
  Store(t20 + t22, &block[ 8]);
  Store(t09 + t17, &block[16]);
  Store(t21 - t23, &block[24]);
  Store(t08 - t10, &block[32]);
  Store(t21 + t23, &block[40]);
  Store(t09 - t17, &block[48]);
  Store(t20 - t22, &block[56]);
#if defined __AVX__
#else
  block += 4;
  }
#endif
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
