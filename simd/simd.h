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
// SSE4/AVX2/scalar, later PPC8 and ARMv8). To ensure performance portability,
// the library only includes operations that are efficient on all platforms.
//
// WARNING: this header may be included from translation units compiled with
// different flags. To prevent ODR violations, this header and its
// dependencies must only define functions if they are static inline and/or
// within namespace SIMD_NAMESPACE. The namespace name varies depending on
// compile flags, so this header requires textual inclusion.

#include <stddef.h>  // size_t
#include <stdint.h>
#include "simd/port.h"
#include "simd/util.h"  // must come after port.h

// Summary of available types (T is the lane type, e.g. float):
// vec1<T>: single-element vector for loop remainders/measuring SIMD speedup;
// vec32<T>: quarter of a 128-bit vector, limited operations;
// vec64<T>: half of a 128-bit vector, limited operations;
// vec128<T>: 128-bit vector;
// vec256<T>: 256-bit vector, only available to AVX2-specific programs;
// vec<T>: alias template for the best instruction set from SIMD_ENABLE;
// vec<T, SSE4>: alias template for a specific instruction set (SSE4);
// PB[B]xN[N]: fixed-type, fixed-width type aliases, where
//   P = lane type prefix: unsigned (u), signed (i), or floating-point (f);
//   B[B] = number of bits per lane;
//   N[N] = number of lanes (such that B[B] * N[N] is one of the above sizes).

namespace pik {
namespace SIMD_NAMESPACE {

// "Member types and constants" for vec<T, Target>; must be non-members because
// we do not control the specializations (vector classes) on PPC.

// True if V is one of our vector types, e.g. i32x4 or i32vec or vec<int32_t>.
// (Primary template; will be specialized in platform-specific headers.)
template <class V>
struct IsVec {
  static constexpr bool value = false;
};

// (Primary template; will be specialized for PPC vectors.)
template <class V>
struct LaneT {
  using type = typename V::T;  // member type in x86 and scalar wrappers.
};
// The vector's lane type. Lane<i32x4> = int32_t.
template <class V>
using Lane = typename LaneT<V>::type;

// How many lanes in the vector. NumLanes<i32x4>() = 4.
// (Primary template; specialized for platform-specific half-vectors.)
template <class V>
struct NumLanes {
  using T = Lane<V>;
  static constexpr size_t value = sizeof(V) / sizeof(T);
  static_assert(value != 0, "NumLanes cannot be zero");
  constexpr operator size_t() const { return value; }
};

// Ensures these dependencies are added to deps.mk. The inclusions have no
// effect because the headers are empty #ifdef SIMD_DEPS.
#ifdef SIMD_DEPS
#include "simd/x86_avx2.h"
#include "simd/x86_sse4.h"
#endif

// Must be included before x86_avx2.h (this is its half-vector type)
#if SIMD_ENABLE_SSE4
#include "simd/x86_sse4.h"
#endif

#if SIMD_ENABLE_SSE4 && SIMD_ENABLE_AVX2
#include "simd/x86_avx2.h"
#endif

#if SIMD_ENABLE_NEON
#include "simd/arm64_neon.h"
#endif

// Always available
#include "simd/scalar.h"

// (specializations for vec<>, see below)
template <class Target>
struct VecT {
#if SIMD_ENABLE_ANY
  template <typename T>
  using type = vec128<T>;
#endif
};
template <>
struct VecT<None> {
  template <typename T>
  using type = vec1<T>;
};
#if SIMD_ENABLE_AVX2
template <>
struct VecT<AVX2> {
  template <typename T>
  using type = vec256<T>;
};
#endif
// Alias of a vector class with lane type T and instruction set Target,
// typically obtained from SIMD_TARGET or dispatch::Run. The default Target is
// the 'best' (i.e. with the widest vectors) of all SIMD_ENABLE bits.
template <typename T, class Target = SIMD_TARGET>
using vec = typename VecT<Target>::template type<T>;

// Generic SIMD algorithms:

// Returns the closest value to v within [lo, hi].
template <typename V>
SIMD_INLINE V clamp(const V v, const V lo, const V hi) {
  return min(max(lo, v), hi);
}

// Returns a vector with lane i=0..N-1 set to "first" + i. Unique per-lane
// values are required to detect lane-crossing bugs.
template <class V>
SIMD_INLINE V Iota(const Lane<V> first = 0) {
  constexpr size_t N = NumLanes<V>();
  SIMD_ALIGN Lane<V> lanes[N];
  for (size_t i = 0; i < N; ++i) {
    lanes[i] = first + i;
  }
  return load(V(), lanes);
}

// Returns a name for V in PB[B]xN[N] format (see above).
// Useful for understanding which instantiation of a generic test failed.
template <class V>
SIMD_INLINE const char* vec_name() {
  using T = Lane<V>;
  // Avoids depending on <type_traits>.
  const bool is_float = T(2.25) != T(2);
  const bool is_signed = T(-1) < T(0);
  constexpr char prefix = is_float ? 'f' : (is_signed ? 'i' : 'u');

  constexpr size_t bits = sizeof(T) * 8;
  constexpr char bits10 = '0' + (bits / 10);
  constexpr char bits1 = '0' + (bits % 10);

  constexpr size_t N = NumLanes<V>();
  constexpr char N1 = (N < 10) ? '\0' : '0' + (N % 10);
  constexpr char N10 = (N < 10) ? '0' + (N % 10) : '0' + (N / 10);

  // 8-bit lanes (the only single-digit bit width)
  if (sizeof(T) == 1) {
    static constexpr char name[8] = {prefix, bits1, 'x', N10, N1};
    return name;
  }

  static constexpr char name[8] = {prefix, bits10, bits1, 'x', N10, N1};
  return name;
}

// Cache control

SIMD_INLINE void stream(const uint32_t t, uint32_t* SIMD_RESTRICT aligned) {
#if SIMD_ARCH_X86
  _mm_stream_si32(reinterpret_cast<int*>(aligned), t);
#else
  CopyBytes(t, aligned);
#endif
}

SIMD_INLINE void stream(const uint64_t t, uint64_t* SIMD_RESTRICT aligned) {
#if SIMD_ARCH_X86
  _mm_stream_si64(reinterpret_cast<long long*>(aligned), t);
#else
  CopyBytes(t, aligned);
#endif
}

// Ensures previous weakly-ordered stores are visible. No effect on non-x86.
SIMD_INLINE void store_fence() {
#if SIMD_ARCH_X86
  _mm_sfence();
#endif
}

// Begins loading the cache line containing "p".
template <typename T>
SIMD_INLINE void prefetch(const T* p) {
#if SIMD_ARCH_X86
  _mm_prefetch(p, _MM_HINT_T0);
#elif SIMD_ARCH_ARM
  __pld(p);
#endif
}

// Invalidates and flushes the cache line containing "p". No effect on non-x86.
SIMD_INLINE void flush_cacheline(const void* p) {
#if SIMD_ARCH_X86
  _mm_clflush(p);
#endif
}

}  // namespace SIMD_NAMESPACE
}  // namespace pik

#endif  // SIMD_SIMD_H_
