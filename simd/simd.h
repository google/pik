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

#ifndef SIMD_SIMD_H_
#define SIMD_SIMD_H_

// SIMD library facade: delegates to platform-specific headers (currently
// scalar/SSE4/AVX2/ARMv8, later PPC8). To ensure performance portability,
// the library only includes operations that are efficient on all platforms.
//
// WARNING: this header is included from translation units compiled with
// different flags. To prevent ODR violations, all functions defined here or
// in dependent headers must be inlined and/or within namespace SIMD_NAMESPACE.
// The namespace name varies depending on compile flags, so this header requires
// textual inclusion.

#include <stddef.h>  // size_t
#include "simd/port.h"
#include "simd/util.h"  // must come after port.h

// Summary of available types (T is the lane type, e.g. float):
// Vec<T, N[, Target]> or Desc<T, N[, Target]>::V are aliases for a full vector
//   of at least 128 bits, or N (=2^j) lane part, or scalar.
//
// PB[B]xN[N] are aliases for vectors/parts with a given lane type and count:
//   P = lane type prefix: unsigned (u), signed (i), or floating-point (f);
//   B[B] = number of bits per lane;
//   N[N] = number of lanes: N[N] = 2^j, B[B] * N[N] <= 128.

namespace pik {
namespace SIMD_NAMESPACE {

// Platform-specific specializations with "type" alias; use via Vec<> below.
template <typename T, size_t N, class Target>
struct VecT;

// Alias for a vector/part/scalar: Vec<uint32_t, 1> = u32x1;
// Vec<float, 1, NONE> = scalar<float>. This is the return type of initializers
// such as setzero(). Parts and full vectors are distinct types on x86 to avoid
// inadvertent conversions. By contrast, PPC parts are merely aliases for full
// vectors to avoid wrapper overhead.
template <typename T, size_t N, class Target = SIMD_TARGET>
using Vec = typename VecT<T, N, Target>::type;

// Descriptor: properties that uniquely identify a Vec. Used to select an
// overloaded function, typically via Full/Part/Scalar aliases below.
template <typename LaneT, size_t kLanes, class TargetT = SIMD_TARGET>
struct Desc {
  using T = LaneT;
  static constexpr size_t N = kLanes;
  using Target = TargetT;

  using V = Vec<T, N, Target>;

  static_assert((N & (N - 1)) == 0, "N must be a power of two");
  static_assert(N == 1 || TargetT::value != SIMD_NONE, "Scalar must have N=1");
};

// Shorthand for a full vector.
template <typename T, class Target = SIMD_TARGET>
using Full = Desc<T, Target::template NumLanes<T>(), Target>;

// Shorthand for a part (or full) vector. N=2^j. Note that MinTarget selects
// a 128-bit Target when T and N are small enough (avoids additional AVX2
// versions of SSE4 initializers/loads).
template <typename T, size_t N>
using Part = Desc<T, N, MinTarget<T, N>>;

// Shorthand for a scalar; note that scalar<T> is the actual data class.
template <typename T>
using Scalar = Desc<T, 1, NONE>;

// Returns size [bytes] of the valid lanes, not necessarily the same as the
// underlying raw register.
template <class D>
constexpr size_t vec_size() {
  return D::N * sizeof(typename D::T);
}

// Returns a name for the vector/part/scalar in PB[B][xN[N]] format (see above).
// Useful for understanding which instantiation of a generic test failed.
template <class D>
inline const char* vec_name() {
  using T = typename D::T;
  constexpr size_t N = D::N;
  constexpr int kTarget = D::Target::value;

  // Avoids depending on <type_traits>.
  const bool is_float = T(2.25) != T(2);
  const bool is_signed = T(-1) < T(0);
  constexpr char prefix = is_float ? 'f' : (is_signed ? 'i' : 'u');

  constexpr size_t bits = sizeof(T) * 8;
  constexpr char bits10 = '0' + (bits / 10);
  constexpr char bits1 = '0' + (bits % 10);

  // Scalars: omit the xN suffix.
  if (kTarget == SIMD_NONE) {
    static constexpr char name1[8] = {prefix, bits1};
    static constexpr char name2[8] = {prefix, bits10, bits1};
    return sizeof(T) == 1 ? name1 : name2;
  }

  constexpr char N1 = (N < 10) ? '\0' : '0' + (N % 10);
  constexpr char N10 = (N < 10) ? '0' + (N % 10) : '0' + (N / 10);

  static constexpr char name1[8] = {prefix, bits1, 'x', N10, N1};
  static constexpr char name2[8] = {prefix, bits10, bits1, 'x', N10, N1};
  return sizeof(T) == 1 ? name1 : name2;
}

// Ensures these dependencies are added to deps.mk. The inclusions have no
// effect because the headers are empty #ifdef SIMD_DEPS.
#ifdef SIMD_DEPS
#include "simd/x86_avx2.h"
#include "simd/x86_sse4.h"
#include "simd/arm64_neon.h"
#endif

// Also used by x86_avx2.h => must be included first.
#if SIMD_ENABLE_SSE4
#include "simd/x86_sse4.h"
#endif

#if SIMD_ENABLE_SSE4 && SIMD_ENABLE_AVX2
#include "simd/x86_avx2.h"
#endif

#if SIMD_ENABLE_ARM8
#include "simd/arm64_neon.h"
#endif

// Always available
#include "simd/scalar.h"

}  // namespace SIMD_NAMESPACE
}  // namespace pik

#endif  // SIMD_SIMD_H_
