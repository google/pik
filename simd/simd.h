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

// Performance-portable SIMD API for SSE4/AVX2/ARMv8, later AVX-512 and PPC8.
// Each operation is efficient on all platforms.

// WARNING: this header may be included from translation units compiled with
// different flags. To prevent ODR violations, all functions defined here or
// in dependent headers must be inlined and/or within namespace SIMD_NAMESPACE.
// The namespace name varies depending on compile flags, so this header requires
// textual inclusion.
#include <stddef.h>  // size_t
#include "simd/port.h"
#include "simd/util.h"

// Ensures an array is aligned and suitable for load()/store() functions.
// Example: SIMD_ALIGN T lanes[V::N];
#define SIMD_ALIGN alignas(32)

namespace pik {
#ifdef SIMD_NAMESPACE
namespace SIMD_NAMESPACE {
#endif

// SIMD operations are implemented as overloaded functions selected using a
// "descriptor" D := Desc<T, N[, Target]>. For example: `D::V setzero(D)`.
// T is the lane type, N the number of lanes, Target is an instruction set
// (e.g. SSE4), defaulting to the best available. The return type D::V is either
// a full vector of at least 128 bits, an N-lane (=2^j) part, or a scalar.

// Specialized in platform-specific headers; see Desc::V.
template <typename T, size_t N, class Target>
struct VecT;

// Descriptor: properties that uniquely identify a vector/part/scalar. Used to
// select overloaded functions; see Full/Part/Scalar aliases below.
template <typename LaneT, size_t kLanes, class TargetT>
struct Desc {
  Desc() {}

  using T = LaneT;
  static constexpr size_t N = kLanes;
  using Target = TargetT;

  // Alias for the actual vector data, e.g. scalar<float> for <float, 1, NONE>,
  // returned by initializers such as setzero(). Parts and full vectors are
  // distinct types on x86 to avoid inadvertent conversions. By contrast, PPC
  // parts are merely aliases for full vectors to avoid wrapper overhead.
  using V = typename VecT<T, N, Target>::type;

  static_assert((N & (N - 1)) == 0, "N must be a power of two");
  static_assert(N <= Target::template NumLanes<T>(), "N too large");
};

// Avoid having to specify SIMD_TARGET in every Part<>/Full<>. Option 1: macro,
// required in attr mode, where we want a different default target for each
// expansion of target-specific code (via foreach_target.h's includes).
#define SIMD_FULL(T) Full<T, SIMD_TARGET>
#define SIMD_PART(T, N) Part<T, N, SIMD_TARGET>

#if SIMD_USE_ATTR
#define SIMD_DEFAULT_TARGET
#else
// Option 2 (normal mode): default argument; sufficient because this entire
// header is included from each target-specific translation unit.
#define SIMD_DEFAULT_TARGET = SIMD_TARGET
#endif

// Shorthand for a full vector.
template <typename T, class Target SIMD_DEFAULT_TARGET>
using Full = Desc<T, Target::template NumLanes<T>(), Target>;

// Shorthand for a part (or full) vector. N=2^j. Note that PartTarget selects
// a 128-bit Target when T and N are small enough (avoids additional AVX2
// versions of SSE4 initializers/loads).
template <typename T, size_t N, class Target SIMD_DEFAULT_TARGET>
using Part = Desc<T, N, PartTarget<T, N, Target>>;

// Shorthand for a scalar; note that scalar<T> is the actual data class.
template <typename T>
using Scalar = Desc<T, 1, NONE>;

// Convenient shorthand for VecT. Chooses the smallest possible Target.
template <typename T, size_t N, class Target>
using VT = typename VecT<T, N, PartTarget<T, N, Target>>::type;

// Type tags for get_half(Upper(), v) etc.
struct Upper {};
struct Lower {};
#define SIMD_HALF Lower()

// Returns a name for the vector/part/scalar. The type prefix is u/i/f for
// unsigned/signed/floating point, followed by the number of bits per lane;
// then 'x' followed by the number of lanes. Example: u8x16. This is useful for
// understanding which instantiation of a generic test failed.
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

// Include all headers #if SIMD_DEPS to ensure they are added to deps.mk.
// This has no other effect because the headers are empty #if SIMD_DEPS.

// Also used by x86_avx2.h => must be included first.
#if SIMD_DEPS || (SIMD_ENABLE_SSE4 || SIMD_ENABLE_AVX2)
#include "simd/x86_sse4.h"
#endif

#if SIMD_DEPS || SIMD_ENABLE_AVX2
#include "simd/x86_avx2.h"
#endif

#if SIMD_DEPS || SIMD_ENABLE_ARM8
#include "simd/arm64_neon.h"
#endif

// Always available
#include "simd/scalar.h"

#ifdef SIMD_NAMESPACE
}  // namespace SIMD_NAMESPACE
#endif
}  // namespace pik

#endif  // SIMD_SIMD_H_
