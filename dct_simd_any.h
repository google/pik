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

#ifndef DCT_SIMD_ANY_H_
#define DCT_SIMD_ANY_H_

#include "block.h"
#include "compiler_specific.h"
#include "simd/simd.h"

namespace pik {

// DCT building blocks that does not require specific SIMD vector length.

template <class From, class To>
SIMD_ATTR PIK_INLINE void CopyBlock8(const From& from, const To& to) {
  const BlockDesc d;
  for (size_t i = 0; i < 8; i += d.N) {
    const auto i0 = from.Load(0, i);
    const auto i1 = from.Load(1, i);
    const auto i2 = from.Load(2, i);
    const auto i3 = from.Load(3, i);
    const auto i4 = from.Load(4, i);
    const auto i5 = from.Load(5, i);
    const auto i6 = from.Load(6, i);
    const auto i7 = from.Load(7, i);
    to.Store(i0, 0, i);
    to.Store(i1, 1, i);
    to.Store(i2, 2, i);
    to.Store(i3, 3, i);
    to.Store(i4, 4, i);
    to.Store(i5, 5, i);
    to.Store(i6, 6, i);
    to.Store(i7, 7, i);
  }
}

template <class V>
SIMD_ATTR PIK_INLINE void ColumnDCT8(V& i0, V& i1, V& i2, V& i3, V& i4, V& i5,
                                     V& i6, V& i7) {
  const BlockDesc d;

  const auto c1 = set1(d, 0.707106781186548f);  // 1 / sqrt(2)
  const auto c2 = set1(d, 0.382683432365090f);  // cos(3 * pi / 8)
  const auto c3 = set1(d, 1.30656296487638f);   // 1 / (2 * cos(3 * pi / 8))
  const auto c4 = set1(d, 0.541196100146197f);  // sqrt(2) * cos(3 * pi / 8)

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
  const auto t16 = t14 - t13;
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

// "A low multiplicative complexity fast recursive DCT-2 algorithm"
// Maxim Vashkevich, Alexander Pertrovsky, 27 Jul 2012
template <class V>
SIMD_ATTR PIK_INLINE void ColumnDCT16(V& i00, V& i01, V& i02, V& i03, V& i04,
                                      V& i05, V& i06, V& i07, V& i08, V& i09,
                                      V& i10, V& i11, V& i12, V& i13, V& i14,
                                      V& i15) {
  const BlockDesc d;

  const auto c1_16 = set1(d, 1.9615705608064609f);   // 2 * cos(1 * pi / 16)
  const auto c2_16 = set1(d, 1.8477590650225735f);   // 2 * cos(2 * pi / 16)
  const auto c3_16 = set1(d, 1.6629392246050905f);   // 2 * cos(3 * pi / 16)
  const auto c4_16 = set1(d, 1.4142135623730951f);   // 2 * cos(4 * pi / 16)
  const auto c5_16 = set1(d, 1.1111404660392046f);   // 2 * cos(5 * pi / 16)
  const auto c6_16 = set1(d, 0.7653668647301797f);   // 2 * cos(6 * pi / 16)
  const auto c7_16 = set1(d, 0.39018064403225666f);  // 2 * cos(7 * pi / 16)

  const auto t00 = i00 + i15;
  const auto t01 = i01 + i14;
  const auto t02 = i02 + i13;
  const auto t03 = i03 + i12;
  const auto t04 = i04 + i11;
  const auto t05 = i05 + i10;
  const auto t06 = i06 + i09;
  const auto t07 = i07 + i08;
  const auto t08 = i00 - i15;
  const auto t09 = i01 - i14;
  const auto t10 = i02 - i13;
  const auto t11 = i03 - i12;
  const auto t12 = i04 - i11;
  const auto t13 = i05 - i10;
  const auto t14 = i06 - i09;
  const auto t15 = i07 - i08;
  const auto t16 = t00 + t07;
  const auto t17 = t01 + t06;
  const auto t18 = t02 + t05;
  const auto t19 = t03 + t04;
  const auto t20 = t00 - t07;
  const auto t21 = t01 - t06;
  const auto t22 = t02 - t05;
  const auto t23 = t03 - t04;
  const auto t24 = t16 + t19;
  const auto t25 = t17 + t18;
  const auto t26 = t16 - t19;
  const auto t27 = t17 - t18;
  i00 = t24 + t25;
  i08 = t24 - t25;
  const auto t30 = t26 - t27;
  const auto t31 = t27 * c4_16;
  i04 = t30 + t31;
  i12 = t30 - t31;
  const auto t34 = t20 - t23;
  const auto t35 = t21 - t22;
  const auto t36 = t22 * c4_16;
  const auto t37 = t23 * c4_16;
  const auto t38 = t34 + t36;
  const auto t39 = t35 + t37;
  const auto t40 = t34 - t36;
  const auto t41 = t35 - t37;
  const auto t42 = t38 - t39;
  const auto t43 = t39 * c2_16;
  i02 = t42 + t43;
  i14 = t42 - t43;
  const auto t46 = t40 - t41;
  const auto t47 = t41 * c6_16;
  i06 = t46 + t47;
  i10 = t46 - t47;
  const auto t50 = t08 - t15;
  const auto t51 = t09 - t14;
  const auto t52 = t10 - t13;
  const auto t53 = t11 - t12;
  const auto t54 = t12 * c4_16;
  const auto t55 = t13 * c4_16;
  const auto t56 = t14 * c4_16;
  const auto t57 = t15 * c4_16;
  const auto t58 = t50 + t54;
  const auto t59 = t51 + t55;
  const auto t60 = t52 + t56;
  const auto t61 = t53 + t57;
  const auto t62 = t50 - t54;
  const auto t63 = t51 - t55;
  const auto t64 = t52 - t56;
  const auto t65 = t53 - t57;
  const auto t66 = t58 - t61;
  const auto t67 = t59 - t60;
  const auto t68 = t60 * c2_16;
  const auto t69 = t61 * c2_16;
  const auto t70 = t66 + t68;
  const auto t71 = t67 + t69;
  const auto t72 = t66 - t68;
  const auto t73 = t67 - t69;
  const auto t74 = t70 - t71;
  const auto t75 = t71 * c1_16;
  i01 = t74 + t75;
  i15 = t74 - t75;
  const auto t78 = t72 - t73;
  const auto t79 = t73 * c7_16;
  i07 = t78 + t79;
  i09 = t78 - t79;
  const auto t82 = t62 - t65;
  const auto t83 = t63 - t64;
  const auto t84 = t64 * c6_16;
  const auto t85 = t65 * c6_16;
  const auto t86 = t82 + t84;
  const auto t87 = t83 + t85;
  const auto t88 = t82 - t84;
  const auto t89 = t83 - t85;
  const auto t90 = t86 - t87;
  const auto t91 = t87 * c3_16;
  i03 = t90 + t91;
  i13 = t90 - t91;
  const auto t94 = t88 - t89;
  const auto t95 = t89 * c5_16;
  i05 = t94 + t95;
  i11 = t94 - t95;
}

template <class From, class To>
SIMD_ATTR PIK_INLINE void ColumnDCT8(const From& from, const To& to) {
  const BlockDesc d;

  for (size_t i = 0; i < 8; i += d.N) {
    auto i0 = from.Load(0, i);
    auto i1 = from.Load(1, i);
    auto i2 = from.Load(2, i);
    auto i3 = from.Load(3, i);
    auto i4 = from.Load(4, i);
    auto i5 = from.Load(5, i);
    auto i6 = from.Load(6, i);
    auto i7 = from.Load(7, i);
    ColumnDCT8(i0, i1, i2, i3, i4, i5, i6, i7);
    to.Store(i0, 0, i);
    to.Store(i1, 1, i);
    to.Store(i2, 2, i);
    to.Store(i3, 3, i);
    to.Store(i4, 4, i);
    to.Store(i5, 5, i);
    to.Store(i6, 6, i);
    to.Store(i7, 7, i);
  }
}

template <class From, class To>
SIMD_ATTR PIK_INLINE void ColumnDCT16(const From& from, const To& to) {
  const BlockDesc d;

  for (size_t i = 0; i < 16; i += d.N) {
    auto i00 = from.Load(0, i);
    auto i01 = from.Load(1, i);
    auto i02 = from.Load(2, i);
    auto i03 = from.Load(3, i);
    auto i04 = from.Load(4, i);
    auto i05 = from.Load(5, i);
    auto i06 = from.Load(6, i);
    auto i07 = from.Load(7, i);
    auto i08 = from.Load(8, i);
    auto i09 = from.Load(9, i);
    auto i10 = from.Load(10, i);
    auto i11 = from.Load(11, i);
    auto i12 = from.Load(12, i);
    auto i13 = from.Load(13, i);
    auto i14 = from.Load(14, i);
    auto i15 = from.Load(15, i);
    ColumnDCT16(i00, i01, i02, i03, i04, i05, i06, i07, i08, i09, i10, i11, i12,
        i13, i14, i15);
    to.Store(i00, 0, i);
    to.Store(i01, 1, i);
    to.Store(i02, 2, i);
    to.Store(i03, 3, i);
    to.Store(i04, 4, i);
    to.Store(i05, 5, i);
    to.Store(i06, 6, i);
    to.Store(i07, 7, i);
    to.Store(i08, 8, i);
    to.Store(i09, 9, i);
    to.Store(i10, 10, i);
    to.Store(i11, 11, i);
    to.Store(i12, 12, i);
    to.Store(i13, 13, i);
    to.Store(i14, 14, i);
    to.Store(i15, 15, i);
  }
}

// NB: ColumnIDCT8(ColumnDCT8(I)) = 8.0 * I
template <class V>
SIMD_ATTR PIK_INLINE void ColumnIDCT8(V& i0, V& i1, V& i2, V& i3, V& i4, V& i5,
                                      V& i6, V& i7) {
  const BlockDesc d;

  const auto c1 = set1(d, 1.41421356237310f);  // sqrt(2)
  const auto c2 = set1(d, 2.61312592975275f);  // 1 / cos(3 * pi / 8)
  const auto c3 = set1(d, 0.76536686473018f);  // 2 * cos(3 * pi / 8)
  const auto c4 = set1(d, 1.08239220029239f);  // 2 * sqrt(2) * cos(3 * pi / 8)

  const auto t00 = i0 + i4;
  const auto t01 = i0 - i4;
  const auto t02 = i6 + i2;
  const auto t03 = i6 - i2;
  const auto t04 = i7 + i1;
  const auto t05 = i7 - i1;
  const auto t06 = i5 + i3;
  const auto t07 = i5 - i3;
  const auto t08 = t04 + t06;
  const auto t09 = t04 - t06;
  const auto t10 = t00 + t02;
  const auto t11 = t00 - t02;
  const auto t12 = t07 - t05;
  const auto t13 = c3 * t12;
  const auto t14 = mul_add(c1, t03, t02);
  const auto t15 = t01 - t14;
  const auto t16 = t01 + t14;
  const auto t17 = mul_add(c2, t05, t13);
  const auto t18 = mul_add(c4, t07, t13);
  const auto t19 = t08 + t17;
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

// "A low multiplicative complexity fast recursive DCT-2 algorithm"
// Maxim Vashkevich, Alexander Pertrovsky, 27 Jul 2012
template <class V>
SIMD_ATTR PIK_INLINE void ColumnIDCT16(V& i00, V& i01, V& i02, V& i03, V& i04,
                                       V& i05, V& i06, V& i07, V& i08, V& i09,
                                       V& i10, V& i11, V& i12, V& i13, V& i14,
                                       V& i15) {
  const BlockDesc d;

  const auto c1_16 = set1(d, 0.5097955791041592f);  // 0.5 / cos(1 * pi / 16)
  const auto c2_16 = set1(d, 0.541196100146197f);   // 0.5 / cos(2 * pi / 16)
  const auto c3_16 = set1(d, 0.6013448869350453f);  // 0.5 / cos(3 * pi / 16)
  const auto c4_16 = set1(d, 0.7071067811865475f);  // 0.5 / cos(4 * pi / 16)
  const auto c5_16 = set1(d, 0.8999762231364156f);  // 0.5 / cos(5 * pi / 16)
  const auto c6_16 = set1(d, 1.3065629648763764f);  // 0.5 / cos(6 * pi / 16)
  const auto c7_16 = set1(d, 2.5629154477415055f);  // 0.5 / cos(7 * pi / 16)

  const auto t00 = i00 + i08;
  const auto t01 = i00 - i08;
  const auto t02 = i04 + i12;
  const auto t03 = i04 - i12;
  const auto t04 = t03 * c4_16;
  const auto t05 = t02 + t04;
  const auto t06 = t00 + t05;
  const auto t07 = t01 + t04;
  const auto t08 = t00 - t05;
  const auto t09 = t01 - t04;
  const auto t10 = i02 + i14;
  const auto t11 = i02 - i14;
  const auto t12 = t11 * c2_16;
  const auto t13 = t10 + t12;
  const auto t14 = i06 + i10;
  const auto t15 = i06 - i10;
  const auto t16 = t15 * c6_16;
  const auto t17 = t14 + t16;
  const auto t18 = t13 + t17;
  const auto t19 = t12 + t16;
  const auto t20 = t13 - t17;
  const auto t21 = t12 - t16;
  const auto t22 = t20 * c4_16;
  const auto t23 = t21 * c4_16;
  const auto t24 = t18 + t23;
  const auto t25 = t19 + t22;
  const auto t26 = t06 + t24;
  const auto t27 = t07 + t25;
  const auto t28 = t09 + t22;
  const auto t29 = t08 + t23;
  const auto t30 = t06 - t24;
  const auto t31 = t07 - t25;
  const auto t32 = t09 - t22;
  const auto t33 = t08 - t23;
  const auto t34 = i01 + i15;
  const auto t35 = i01 - i15;
  const auto t36 = t35 * c1_16;
  const auto t37 = t34 + t36;
  const auto t38 = i07 + i09;
  const auto t39 = i07 - i09;
  const auto t40 = t39 * c7_16;
  const auto t41 = t38 + t40;
  const auto t42 = t37 + t41;
  const auto t43 = t36 + t40;
  const auto t44 = t37 - t41;
  const auto t45 = t36 - t40;
  const auto t46 = t44 * c2_16;
  const auto t47 = t45 * c2_16;
  const auto t48 = t42 + t47;
  const auto t49 = t43 + t46;
  const auto t50 = i03 + i13;
  const auto t51 = i03 - i13;
  const auto t52 = t51 * c3_16;
  const auto t53 = t50 + t52;
  const auto t54 = i05 + i11;
  const auto t55 = i05 - i11;
  const auto t56 = t55 * c5_16;
  const auto t57 = t54 + t56;
  const auto t58 = t53 + t57;
  const auto t59 = t52 + t56;
  const auto t60 = t53 - t57;
  const auto t61 = t52 - t56;
  const auto t62 = t60 * c6_16;
  const auto t63 = t61 * c6_16;
  const auto t64 = t58 + t63;
  const auto t65 = t59 + t62;
  const auto t66 = t48 + t64;
  const auto t67 = t49 + t65;
  const auto t68 = t46 + t62;
  const auto t69 = t47 + t63;
  const auto t70 = t48 - t64;
  const auto t71 = t49 - t65;
  const auto t72 = t46 - t62;
  const auto t73 = t47 - t63;
  const auto t74 = t70 * c4_16;
  const auto t75 = t71 * c4_16;
  const auto t76 = t72 * c4_16;
  const auto t77 = t73 * c4_16;
  const auto t78 = t66 + t77;
  const auto t79 = t67 + t76;
  const auto t80 = t68 + t75;
  const auto t81 = t69 + t74;
  i00 = t26 + t78;
  i01 = t27 + t79;
  i02 = t28 + t80;
  i03 = t29 + t81;
  i04 = t33 + t74;
  i05 = t32 + t75;
  i06 = t31 + t76;
  i07 = t30 + t77;
  i15 = t26 - t78;
  i14 = t27 - t79;
  i13 = t28 - t80;
  i12 = t29 - t81;
  i11 = t33 - t74;
  i10 = t32 - t75;
  i09 = t31 - t76;
  i08 = t30 - t77;
}

template <class From, class To>
SIMD_ATTR PIK_INLINE void ColumnIDCT8(const From& from, const To& to) {
  const BlockDesc d;

  for (size_t i = 0; i < 8; i += d.N) {
    auto i0 = from.Load(0, i);
    auto i1 = from.Load(1, i);
    auto i2 = from.Load(2, i);
    auto i3 = from.Load(3, i);
    auto i4 = from.Load(4, i);
    auto i5 = from.Load(5, i);
    auto i6 = from.Load(6, i);
    auto i7 = from.Load(7, i);
    ColumnIDCT8(i0, i1, i2, i3, i4, i5, i6, i7);
    to.Store(i0, 0, i);
    to.Store(i1, 1, i);
    to.Store(i2, 2, i);
    to.Store(i3, 3, i);
    to.Store(i4, 4, i);
    to.Store(i5, 5, i);
    to.Store(i6, 6, i);
    to.Store(i7, 7, i);
  }
}

template <class From, class To>
SIMD_ATTR PIK_INLINE void ColumnIDCT16(const From& from, const To& to) {
  const BlockDesc d;

  for (size_t i = 0; i < 16; i += d.N) {
    auto i00 = from.Load(0, i);
    auto i01 = from.Load(1, i);
    auto i02 = from.Load(2, i);
    auto i03 = from.Load(3, i);
    auto i04 = from.Load(4, i);
    auto i05 = from.Load(5, i);
    auto i06 = from.Load(6, i);
    auto i07 = from.Load(7, i);
    auto i08 = from.Load(8, i);
    auto i09 = from.Load(9, i);
    auto i10 = from.Load(10, i);
    auto i11 = from.Load(11, i);
    auto i12 = from.Load(12, i);
    auto i13 = from.Load(13, i);
    auto i14 = from.Load(14, i);
    auto i15 = from.Load(15, i);
    ColumnIDCT16(i00, i01, i02, i03, i04, i05, i06, i07, i08, i09, i10, i11,
        i12, i13, i14, i15);
    to.Store(i00, 0, i);
    to.Store(i01, 1, i);
    to.Store(i02, 2, i);
    to.Store(i03, 3, i);
    to.Store(i04, 4, i);
    to.Store(i05, 5, i);
    to.Store(i06, 6, i);
    to.Store(i07, 7, i);
    to.Store(i08, 8, i);
    to.Store(i09, 9, i);
    to.Store(i10, 10, i);
    to.Store(i11, 11, i);
    to.Store(i12, 12, i);
    to.Store(i13, 13, i);
    to.Store(i14, 14, i);
    to.Store(i15, 15, i);
  }
}

}  // namespace pik

#endif  // THIRD_PARTY_DCT_SIMD_ANY_H_
