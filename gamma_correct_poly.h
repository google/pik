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

#ifndef GAMMA_CORRECT_POLY_H_
#define GAMMA_CORRECT_POLY_H_

// WARNING: this is a "restricted" header because it is included from
// translation units compiled with different flags. This header and its
// dependencies must not define any function unless it is static inline and/or
// within namespace SIMD_NAMESPACE. See arch_specific.h for details.

#include "rational_polynomial.h"
#include "simd_helpers.h"

namespace pik {
namespace SIMD_NAMESPACE {

// Efficient approximation of LinearToSrgb8Direct.
template <class D = Full<float>>
class LinearToSrgb8Polynomial {
  using T = typename D::T;

  // 4/3 is a good tradeoff: the degree is low enough that single precision
  // is sufficient, and the max abs error on [0, 255] is 3.02E-2, which is good
  // enough because we only need the correct 8-bit pixel value after rounding.
  using Polynomial = RationalPolynomial<D, 4, 3>;

  // Returns polynomial approximator of LinearToSrgb8Direct(x01 * 255) / 255.
  static Polynomial InitPoly() {
    // Computed via af_cheb_rational (k=100).
    const T p[4 + 1] = {-5.0940643078E-06, 9.6089389850E-03, 4.9360562207E-01,
                        1.9957751314E+00, 3.9809614017E-01};

    const T q[3 + 1] = {4.6257541759E-04, 8.3720417588E-02, 1.2275796918E+00,
                        1.5849762703E+00};
    return Polynomial(p, q);
  }

 public:
  LinearToSrgb8Polynomial() : polynomial_(InitPoly()) {}

  // x = [0, 255]. Output is not clamped! Maximum error: see above.
  template <class V>
  PIK_INLINE V operator()(const V x) const {
    const D d;
    const V linear = x * set1(d, 12.92);
    // Range reduction: 2 extra constants, but half the max error vs 5/4.
    const V x01 = x * set1(d, 1.0 / 255);

    const V poly = polynomial_(x01) * set1(d, 255.0);
    return select(linear, poly, x01 > set1(d, 0.00313080495356));
  }

 private:
  Polynomial polynomial_;
};

// Same logic as LinearToSrgb8Polynomial but wrapped in a function (faster).
// x = [0, 255]. Output is not clamped! Maximum error: see above.
template <class V>
V LinearToSrgb8PolyWithoutClamp(const V x) {
  // Computed via af_cheb_rational (k=100); replicated 4x.
  SIMD_ALIGN constexpr float p[(4 + 1) * 4] = {
      -5.094064307E-6f, -5.094064307E-6f, -5.094064307E-6f, -5.094064307E-6f,
      9.6089389850E-3f, 9.6089389850E-3f, 9.6089389850E-3f, 9.6089389850E-3f,
      4.9360562207E-1f, 4.9360562207E-1f, 4.9360562207E-1f, 4.9360562207E-1f,
      1.9957751314E+0f, 1.9957751314E+0f, 1.9957751314E+0f, 1.9957751314E+0f,
      3.9809614017E-1f, 3.9809614017E-1f, 3.9809614017E-1f, 3.9809614017E-1f};

  SIMD_ALIGN constexpr float q[(3 + 1) * 4] = {
      4.6257541759E-4f, 4.6257541759E-4f, 4.6257541759E-4f, 4.6257541759E-4f,
      8.3720417588E-2f, 8.3720417588E-2f, 8.3720417588E-2f, 8.3720417588E-2f,
      1.2275796918E+0f, 1.2275796918E+0f, 1.2275796918E+0f, 1.2275796918E+0f,
      1.5849762703E+0f, 1.5849762703E+0f, 1.5849762703E+0f, 1.5849762703E+0f};

  const Full<float> d;
  const V linear = x * set1(d, 12.92f);
  // Range reduction: 2 extra constants, but half the max error vs 5/4.
  const V x01 = x * set1(d, 1.0f / 255);

  const V poly = EvalRationalPolynomial(x01, p, q) * set1(d, 255.0);
  return select(linear, poly, x01 > set1(d, 0.00313080495356));
}

// Evaluate at three locations (more efficient).
template <class V>
void LinearToSrgb8PolyWithoutClamp(const V x0, const V x1, const V x2,
                                   V* PIK_RESTRICT y0, V* PIK_RESTRICT y1,
                                   V* PIK_RESTRICT y2) {
  using namespace SIMD_NAMESPACE;
  // Computed via af_cheb_rational (k=100); replicated 4x.
  SIMD_ALIGN constexpr float p[(4 + 1) * 4] = {
      -5.094064307E-6f, -5.094064307E-6f, -5.094064307E-6f, -5.094064307E-6f,
      9.6089389850E-3f, 9.6089389850E-3f, 9.6089389850E-3f, 9.6089389850E-3f,
      4.9360562207E-1f, 4.9360562207E-1f, 4.9360562207E-1f, 4.9360562207E-1f,
      1.9957751314E+0f, 1.9957751314E+0f, 1.9957751314E+0f, 1.9957751314E+0f,
      3.9809614017E-1f, 3.9809614017E-1f, 3.9809614017E-1f, 3.9809614017E-1f};

  SIMD_ALIGN constexpr float q[(3 + 1) * 4] = {
      4.6257541759E-4f, 4.6257541759E-4f, 4.6257541759E-4f, 4.6257541759E-4f,
      8.3720417588E-2f, 8.3720417588E-2f, 8.3720417588E-2f, 8.3720417588E-2f,
      1.2275796918E+0f, 1.2275796918E+0f, 1.2275796918E+0f, 1.2275796918E+0f,
      1.5849762703E+0f, 1.5849762703E+0f, 1.5849762703E+0f, 1.5849762703E+0f};

  const Full<float> d;
#if SIMD_TARGET_VALUE == SIMD_NONE
  const auto shrink = set1(d, 1.0f / 255);
  const auto linear = set1(d, 12.92f);
  const auto expand = set1(d, 255.0f);
  const auto min_poly_x = set1(d, 0.00313080495356f);
#else
  SIMD_ALIGN constexpr float lanes[4] = {1.0f / 255, 12.92f, 255.0f,
                                         0.00313080495356f};
  const auto constants = load_dup128(d, lanes);
  const auto shrink = broadcast<0>(constants);
  const auto linear = broadcast<1>(constants);
  const auto expand = broadcast<2>(constants);
  const auto min_poly_x = broadcast<3>(constants);
#endif

  // Range reduction: 2 extra constants, but half the max error vs 5/4.
  const V x0_01 = x0 * shrink;
  const V x1_01 = x1 * shrink;
  const V x2_01 = x2 * shrink;

  EvalRationalPolynomialTriple(x0_01, x1_01, x2_01, p, q, y0, y1, y2);


  *y0 = select(x0 * linear, *y0 * expand, x0_01 > min_poly_x);
  *y1 = select(x1 * linear, *y1 * expand, x1_01 > min_poly_x);
  *y2 = select(x2 * linear, *y2 * expand, x2_01 > min_poly_x);
}

template <class D, typename V>
V LinearToSrgb8Poly(D d, V z) {
  return Clamp0To255(d, LinearToSrgb8PolyWithoutClamp(z));
}

template <class D, typename V>
void LinearToSrgb8Poly(D d, const V x0, const V x1, const V x2,
                       V* PIK_RESTRICT y0, V* PIK_RESTRICT y1,
                       V* PIK_RESTRICT y2) {
  *y0 = LinearToSrgb8Poly(d, x0);
  *y1 = LinearToSrgb8Poly(d, x1);
  *y2 = LinearToSrgb8Poly(d, x2);
}

}  // namespace SIMD_NAMESPACE
}  // namespace pik

#endif  // GAMMA_CORRECT_POLY_H_
