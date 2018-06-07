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

#ifndef SIMD_HELPERS_H_
#define SIMD_HELPERS_H_

#include "simd/simd.h"

// 4 instances of a given literal value, useful as input to load_dup128.
#define PIK_REP4(literal) literal, literal, literal, literal

namespace pik {

template <class V>
SIMD_INLINE float FirstLane(const V v) {
  return get_part(SIMD_NAMESPACE::Part<float, 1>(), v);
}

template <class V>
SIMD_INLINE float GetLane(const V v, size_t i) {
  SIMD_NAMESPACE::Full<float> d;
  SIMD_ALIGN float lanes[d.N];
  store(v, d, lanes);
  return lanes[i];
}

template <typename T>
SIMD_INLINE T* SIMD_RESTRICT ByteOffset(T* SIMD_RESTRICT base,
                                        const intptr_t byte_offset) {
  const uintptr_t base_addr = reinterpret_cast<uintptr_t>(base);
  return reinterpret_cast<T*>(base_addr + byte_offset);
}

template <class D, class V>
static SIMD_INLINE V Clamp0To255(D d, const V x) {
  const auto clamped = min(x, set1(d, 255.0f));
  // If negative, replace with zero (faster than floating-point max()).
  return select(clamped, setzero(d), condition_from_sign(clamped));
}

// [0, max_value]
template <class D, class V>
static SIMD_INLINE V Clamp0ToMax(D d, const V x, const V max_value) {
  const auto clamped = min(x, max_value);
  // If negative, replace with zero (faster than floating-point max()).
  return select(clamped, setzero(d), condition_from_sign(clamped));
}

// One Newton-Raphson iteration.
template <class V>
static SIMD_INLINE V ReciprocalNR(const V x) {
  const auto rcp = approximate_reciprocal(x);
  const auto sum = rcp + rcp;
  const auto x_rcp = x * rcp;
  return nmul_add(x_rcp, rcp, sum);
}

// Primary template: default to actual division.
template <typename T, class V>
struct FastDivision {
  V operator()(const V n, const V d) const { return n / d; }
};
// Partial specialization for float vectors.
template <class V>
struct FastDivision<float, V> {
  V operator()(const V n, const V d) const { return n * ReciprocalNR(d); }
};

}  // namespace pik

#endif  // SIMD_HELPERS_H_
