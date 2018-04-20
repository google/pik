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

#ifndef SIMD_PORT_H_
#define SIMD_PORT_H_

#include <stddef.h>
#include <stdint.h>

// Detects compiler/architecture and defines instruction sets.

// Compiler-specific keywords
#ifdef _MSC_VER
#define SIMD_RESTRICT __restrict
#define SIMD_INLINE __forceinline
#define SIMD_NOINLINE __declspec(noinline)
#define SIMD_LIKELY(expr) expr
#define SIMD_TRAP __debugbreak
#define SIMD_TARGET_ATTR(feature_str)
#define SIMD_DIAGNOSTICS(tokens) __pragma(warning(tokens))
#define SIMD_DIAGNOSTICS_OFF(msc, gcc) SIMD_DIAGNOSTICS(msc)

#elif defined(__GNUC__) || defined(__clang__)
#define SIMD_RESTRICT __restrict__
#define SIMD_INLINE \
  inline __attribute__((always_inline)) __attribute__((flatten))
#define SIMD_NOINLINE inline __attribute__((noinline))
#define SIMD_LIKELY(expr) __builtin_expect(!!(expr), 1)
#define SIMD_TRAP __builtin_trap
#define SIMD_TARGET_ATTR(feature_str) __attribute__((target(feature_str)))
#define SIMD_PRAGMA(tokens) _Pragma (#tokens)
#define SIMD_DIAGNOSTICS(tokens) SIMD_PRAGMA(GCC diagnostic tokens)
#define SIMD_DIAGNOSTICS_OFF(msc, gcc) SIMD_DIAGNOSTICS(gcc)

#else
#error "Unsupported compiler"
#endif

// Architectures
#define SIMD_ARCH_X86 8
#define SIMD_ARCH_PPC 9
#define SIMD_ARCH_ARM 0xA
#if defined(__x86_64__) || defined(_M_X64)
#define SIMD_ARCH SIMD_ARCH_X86
#elif defined(__powerpc64__) || defined(_M_PPC)
#define SIMD_ARCH SIMD_ARCH_PPC
#elif defined(__aarch64__)
#define SIMD_ARCH SIMD_ARCH_ARM
#else
#error "Unsupported platform"
#endif

// Instruction set bits are a compact encoding of zero or more targets used in
// dispatch::Run and the SIMD_ENABLE macro.
#define SIMD_NONE 0
#define SIMD_AVX2 2
#define SIMD_SSE4 4
#define SIMD_AVX512 16
#define SIMD_PPC 1  // v2.07 or 3
#define SIMD_ARM8 8

// Default to portable mode (only scalar.h). This macro should only be set by
// the build system and tested below.
#ifndef SIMD_ENABLE
#define SIMD_ENABLE 0
#endif

// Disable attr mode unless users request it (requires recent compiler).
#ifndef SIMD_USE_ATTR
#define SIMD_USE_ATTR 0
#endif

// If MSVC 2015 || SIMD_USE_ATTR, intrinsics are accessible if we're on the
// same platform.
#define SIMD_HAVE_AVX2 (SIMD_ARCH == SIMD_ARCH_X86)
#define SIMD_HAVE_SSE4 (SIMD_ARCH == SIMD_ARCH_X86)
#define SIMD_HAVE_AVX512 (SIMD_ARCH == SIMD_ARCH_X86)
#define SIMD_HAVE_ARM8 (SIMD_ARCH == SIMD_ARCH_ARM)

// .. otherwise, disallow intrinsics if -m flags are not specified.
#if !defined(_MSC_VER) && !SIMD_USE_ATTR

#if !defined(__SSE4_2__) || !defined(__AES__)
#undef SIMD_HAVE_SSE4
#define SIMD_HAVE_SSE4 0
#endif  // SSE4

#if !defined(__AVX2__) || !defined(__FMA__)
#undef SIMD_HAVE_AVX2
#define SIMD_HAVE_AVX2 0
#endif  // AVX2

#if !defined(__ARM_NEON)
#undef SIMD_HAVE_ARM8
#define SIMD_HAVE_ARM8 0
#endif  // AVX2

#endif

// Set ENABLE_XX shortcuts (for internal use).
#define SIMD_ENABLE_SSE4 (SIMD_ENABLE & SIMD_SSE4) && SIMD_HAVE_SSE4
#define SIMD_ENABLE_AVX2 (SIMD_ENABLE & SIMD_AVX2) && SIMD_HAVE_AVX2
#define SIMD_ENABLE_AVX512 (SIMD_ENABLE & SIMD_AVX512) && SIMD_HAVE_AVX512
#define SIMD_ENABLE_ARM8 (SIMD_ENABLE & SIMD_ARM8) && SIMD_HAVE_ARM8

// Detects "best available" instruction set and includes their headers. NOTE:
// system headers cannot be included from within SIMD_NAMESPACE due to conflicts
// with other headers. ODR violations are avoided if all their functions (static
// inline in Clang's library and extern inline in GCC's) are inlined.

#if SIMD_ENABLE_AVX2
#include <immintrin.h>
#define SIMD_TARGET AVX2

#elif SIMD_ENABLE_SSE4
#include <smmintrin.h>
#include <wmmintrin.h>  // AES
#define SIMD_TARGET SSE4

#elif SIMD_ARCH == SIMD_ARCH_X86
// No instruction set enabled, but we still need the header for flush_cacheline.
#include <emmintrin.h>
#endif

#if SIMD_ENABLE_ARM8
#include <arm_neon.h>
#define SIMD_TARGET ARM8
#endif

// Nothing enabled => portable mode, only use scalar.h.
#ifndef SIMD_TARGET
#define SIMD_TARGET NONE
#endif

// Define macros based on the SIMD_TARGET _when those macros are expanded_.
#define SIMD_CONCAT_IMPL(a, b) a##b
#define SIMD_CONCAT(a, b) SIMD_CONCAT_IMPL(a, b)

#define SIMD_ATTR_ARM8 SIMD_TARGET_ATTR("armv8-a+crypto")
#define SIMD_ATTR_SSE4 SIMD_TARGET_ATTR("sse4.2,aes,pclmul")
#define SIMD_ATTR_AVX2 SIMD_TARGET_ATTR("avx,avx2,fma")
#define SIMD_ATTR_NONE
#define SIMD_ATTR SIMD_CONCAT(SIMD_ATTR_, SIMD_TARGET)


#if SIMD_USE_ATTR
// In attr mode, SIMD_TARGET is redefined for each expansion.
#undef SIMD_TARGET

#else
#define SIMD_NAMESPACE SIMD_CONCAT(N_, SIMD_TARGET)
#undef SIMD_TARGET_ATTR
#define SIMD_TARGET_ATTR(feature_str)

#endif

#define SIMD_TARGET_VALUE SIMD_CONCAT(SIMD_, SIMD_TARGET)

namespace pik {

// Instruction set tag names used to specialize VecT - results in more
// understandable mangled names than using SIMD_SSE4=4 directly. Their names
// must match the SIMD_TARGET definitions above.
#if SIMD_ARCH == SIMD_ARCH_X86
struct SSE4 {
  static constexpr int value = SIMD_SSE4;
  template <typename T>
  static constexpr size_t NumLanes() {
    return 16 / sizeof(T);
  }
};
struct AVX2 {
  static constexpr int value = SIMD_AVX2;
  template <typename T>
  static constexpr size_t NumLanes() {
    return 32 / sizeof(T);
  }
};
struct AVX512 {
  static constexpr int value = SIMD_AVX512;
  template <typename T>
  static constexpr size_t NumLanes() {
    return 64 / sizeof(T);
  }
};
#elif SIMD_ARCH == SIMD_ARCH_PPC
struct PPC8 {
  static constexpr int value = SIMD_PPC;
  template <typename T>
  static constexpr size_t NumLanes() {
    return 16 / sizeof(T);
  }
};
#elif SIMD_ARCH == SIMD_ARCH_ARM
struct ARM8 {
  static constexpr int value = SIMD_ARM8;
  template <typename T>
  static constexpr size_t NumLanes() {
    return 16 / sizeof(T);
  }
};
#endif

struct NONE {
  static constexpr int value = SIMD_NONE;
  template <typename T>
  static constexpr size_t NumLanes() {
    return 1;
  }
};

// Default: no change to Target. kBlocks = ceil(size / 16).
template <size_t kBlocks, class Target>
struct PartTargetT {
  using type = Target;
};
// Never override NONE.
template <>
struct PartTargetT<1, NONE> {
  using type = NONE;
};

// On X86, it is cheaper to use small vectors (prefixes of larger registers)
// when possible; this also reduces the number of overloaded functions.
#if SIMD_ENABLE_SSE4
template <class Target>
struct PartTargetT<1, Target> {
  using type = SSE4;
};
#endif
#if SIMD_ENABLE_AVX2
template <class Target>
struct PartTargetT<2, Target> {
  using type = AVX2;
};
#endif

template <typename T, size_t N, class Target>
using PartTarget =
    typename PartTargetT<(N * sizeof(T) + 15) / 16, Target>::type;

// Unfortunately the GCC/Clang intrinsics do not accept int64_t*.
using GatherIndex64 = long long int;
static_assert(sizeof(GatherIndex64) == 8, "Must be 64-bit type");

}  // namespace pik

#endif  // SIMD_PORT_H_
