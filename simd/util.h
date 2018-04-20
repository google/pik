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

// Replacements for standard library functionality required from "restricted"
// headers/source files where it is unsafe to include system headers.

#ifndef SIMD_UTIL_H_
#define SIMD_UTIL_H_

#include "simd/port.h"

namespace pik {
#ifdef SIMD_NAMESPACE
namespace SIMD_NAMESPACE {
#endif

// std::min/max.

#define SIMD_MIN(a, b) ((a) < (b) ? (a) : (b))
#define SIMD_MAX(a, b) ((a) < (b) ? (b) : (a))

// memcpy/memset.

// The source/destination must not overlap/alias.
template <size_t kBytes, typename From, typename To>
SIMD_INLINE void CopyBytes(const From* from, To* to) {
  const uint8_t* SIMD_RESTRICT from_bytes =
      reinterpret_cast<const uint8_t*>(from);
  uint8_t* SIMD_RESTRICT to_bytes = reinterpret_cast<uint8_t*>(to);
  for (size_t i = 0; i < kBytes; ++i) {
    to_bytes[i] = from_bytes[i];
  }
}

// The source/destination must not overlap/alias.
template <size_t kBytes, typename From, typename To>
SIMD_INLINE void CopyBytesWithOffset(const From* from, const int offset,
                                     To* to) {
  const uint8_t* SIMD_RESTRICT from_bytes =
      reinterpret_cast<const uint8_t*>(from) + offset;
  uint8_t* SIMD_RESTRICT to_bytes = reinterpret_cast<uint8_t*>(to);
  for (size_t i = 0; i < kBytes; ++i) {
    to_bytes[i] = from_bytes[i];
  }
}

template <typename T>
SIMD_INLINE void SetBytes(const uint8_t byte, T* t) {
  uint8_t* bytes = reinterpret_cast<uint8_t*>(t);
  for (size_t i = 0; i < sizeof(T); ++i) {
    bytes[i] = byte;
  }
}

// numeric_limits<T>

template <typename T>
constexpr bool IsFloat() {
  return T(1.25) != T(1);
}

template <typename T>
constexpr bool IsSigned() {
  return T(0) > T(-1);
}

// Largest/smallest representable integer values.
template <typename T>
constexpr T LimitsMax() {
  return IsSigned<T>() ? T((1ULL << (sizeof(T) * 8 - 1)) - 1)
                       : static_cast<T>(~0ull);
}
template <typename T>
constexpr T LimitsMin() {
  return IsSigned<T>() ? T(-1) - LimitsMax<T>() : T(0);
}

// Random numbers

struct RandomState {
  uint64_t state;
  uint64_t inc;
};

// *Really* minimal PCG32 code / (c) 2014 M.E. O'Neill / pcg-random.org
// Licensed under Apache License 2.0 (NO WARRANTY, etc. see website)
SIMD_INLINE uint32_t Random32(RandomState* rng) {
  uint64_t oldstate = rng->state;
  // Advance internal state
  rng->state = oldstate * 6364136223846793005ULL + (rng->inc | 1);
  // Calculate output function (XSH RR), uses old state for max ILP
  uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
  uint32_t rot = oldstate >> 59u;
  return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

// Value to string

// Returns end of string (position of '\0').
template <typename T>
inline char* ToString(T value, char* to) {
  char reversed[64];
  char* pos = reversed;
  int64_t before;
  do {
    before = value;
    value /= 10;
    const int64_t mod = before - value * 10;
    *pos++ = "9876543210123456789"[9 + mod];
  } while (value != 0);
  if (before < 0) *pos++ = '-';

  // Reverse the string
  const int num_chars = pos - reversed;
  for (int i = 0; i < num_chars; ++i) {
    to[i] = pos[-1 - i];
  }
  to[num_chars] = '\0';
  return to + num_chars;
}

template <>
inline char* ToString<float>(const float value, char* to) {
  const int64_t truncated = static_cast<int64_t>(value);
  char* end = ToString(truncated, to);
  *end++ = '.';
  int64_t frac = static_cast<int64_t>((value - truncated) * 1E8);
  if (frac < 0) frac = -frac;
  return ToString(frac, end);
}

template <>
inline char* ToString<double>(const double value, char* to) {
  const int64_t truncated = static_cast<int64_t>(value);
  char* end = ToString(truncated, to);
  *end++ = '.';
  int64_t frac = static_cast<int64_t>((value - truncated) * 1E16);
  if (frac < 0) frac = -frac;
  return ToString(frac, end);
}

template <>
inline char* ToString<const char*>(const char* value, char* to) {
  const char* p = value;
  while (*p != '\0') {
    *to++ = *p++;
  }
  *to = '\0';
  return to;
}

// String comparison

template <typename T1, typename T2>
inline bool BytesEqual(const T1* p1, const T2* p2, const size_t size) {
  const uint8_t* bytes1 = reinterpret_cast<const uint8_t*>(p1);
  const uint8_t* bytes2 = reinterpret_cast<const uint8_t*>(p2);
  for (size_t i = 0; i < size; ++i) {
    if (bytes1[i] != bytes2[i]) return false;
  }
  return true;
}

inline bool StringsEqual(const char* s1, const char* s2) {
  while (*s1 == *s2++) {
    if (*s1++ == '\0') return true;
  }
  return false;
}

// Cache control

SIMD_INLINE void stream(const uint32_t t, uint32_t* SIMD_RESTRICT aligned) {
#if SIMD_ARCH == SIMD_ARCH_X86
  _mm_stream_si32(reinterpret_cast<int*>(aligned), t);
#else
  CopyBytes<4>(&t, aligned);
#endif
}

SIMD_INLINE void stream(const uint64_t t, uint64_t* SIMD_RESTRICT aligned) {
#if SIMD_ARCH == SIMD_ARCH_X86
  _mm_stream_si64(reinterpret_cast<long long*>(aligned), t);
#else
  CopyBytes<8>(&t, aligned);
#endif
}

// Delays subsequent loads until prior loads are visible. On Intel CPUs, also
// serves as a full fence (waits for all prior instructions to complete).
// No effect on non-x86.
SIMD_INLINE void load_fence() {
#if SIMD_ARCH == SIMD_ARCH_X86
  _mm_lfence();
#endif
}

// Ensures previous weakly-ordered stores are visible. No effect on non-x86.
SIMD_INLINE void store_fence() {
#if SIMD_ARCH == SIMD_ARCH_X86
  _mm_sfence();
#endif
}

// Begins loading the cache line containing "p".
template <typename T>
SIMD_INLINE void prefetch(const T* p) {
#if SIMD_ARCH == SIMD_ARCH_X86
  _mm_prefetch(p, _MM_HINT_T0);
#elif SIMD_ARCH == SIMD_ARCH_ARM
  __pld(p);
#endif
}

// Invalidates and flushes the cache line containing "p". No effect on non-x86.
SIMD_INLINE void flush_cacheline(const void* p) {
#if SIMD_ARCH == SIMD_ARCH_X86
  _mm_clflush(p);
#endif
}

// Call during spin loops to potentially reduce contention/power consumption.
SIMD_INLINE void pause() {
#if SIMD_ARCH == SIMD_ARCH_X86
  _mm_pause();
#endif
}

#ifdef SIMD_NAMESPACE
}  // namespace SIMD_NAMESPACE
#endif
}  // namespace pik

#endif  // SIMD_UTIL_H_
