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

// Detects compiler/architecture and defines instruction sets.

// Ensures an array is aligned and suitable for load()/store() functions.
// Example: SIMD_ALIGN T lanes[V::N];
#define SIMD_ALIGN alignas(32)

// SIMD_TARGET_ATTR prerequisites on Clang/GCC: __has_attribute(target) does not
// guarantee intrinsics are usable, hence we must check for specific versions.
#if defined(__clang__)
// Apple 8.2 == Clang 3.9
#ifdef __apple_build_version__
#if __clang_major__ > 8 || (__clang_major__ == 8 && __clang_minor__ >= 2)
#define SIMD_TARGET_ATTR
#endif
// llvm.org Clang 3.9
#elif __clang_major__ > 3 || (__clang_major__ == 3 && __clang_minor__ >= 9)
#define SIMD_TARGET_ATTR
#endif
// GCC 4.9
#elif defined(__GNUC__) && \
    (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 9))
#define SIMD_TARGET_ATTR
#endif

// Compiler-specific keywords
#ifdef _MSC_VER
#define SIMD_RESTRICT __restrict
#define SIMD_INLINE __forceinline
#define SIMD_NOINLINE __declspec(noinline)
#define SIMD_LIKELY(expr) expr
#define SIMD_TRAP __debugbreak
#define SIMD_TARGET_ATTR(feature_str)

#elif defined(__GNUC__) || defined(__clang__)
#define SIMD_RESTRICT __restrict__
#define SIMD_INLINE inline __attribute__((always_inline))
#define SIMD_NOINLINE inline __attribute__((noinline))
#define SIMD_LIKELY(expr) __builtin_expect(!!(expr), 1)
#define SIMD_TRAP __builtin_trap
// Only define if prequisites are met (see usage below)
#ifdef SIMD_TARGET_ATTR
#undef SIMD_TARGET_ATTR
#define SIMD_TARGET_ATTR(feature_str) __attribute__((target(feature_str)))
#endif

#else
#error "Unsupported compiler"
#endif

// Architectures
#define SIMD_ARCH_X86 0
#define SIMD_ARCH_PPC 0
#define SIMD_ARCH_ARM 0

#if defined(__x86_64__) || defined(_M_X64)
#undef SIMD_ARCH_X86
#define SIMD_ARCH_X86 1

#elif defined(__powerpc64__) || defined(_M_PPC)
#undef SIMD_ARCH_PPC
#define SIMD_ARCH_PPC 1

#elif defined(__aarch64__)
#undef SIMD_ARCH_ARM
#define SIMD_ARCH_ARM 1

#else
#error "Unsupported platform"
#endif

// Instruction set bits are a compact encoding of zero or more targets used in
// dispatch::Run and the SIMD_ENABLE macro.
#define SIMD_NONE 0
#if SIMD_ARCH_X86
#define SIMD_AVX2 2
#define SIMD_SSE4 4
#define SIMD_AVX512 16
#endif
#if SIMD_ARCH_PPC
#define SIMD_PPC 1  // v2.07 or 3
#endif
#if SIMD_ARCH_ARM
#define SIMD_ARM 8  // v8
#endif

// Default to portable mode (only scalar.h). This macro should only be set by
// the build system and tested below; users check #if SIMD_ENABLE_* to decide
// whether the instruction set is actually usable on this compiler.
#ifndef SIMD_ENABLE
#define SIMD_ENABLE 0
#endif

// Compiler "supports" SIMD_TARGET_ATTR => use target-specific attributes
// instead of per-file -mavx2; no need to check for predefined macros.
#ifdef SIMD_TARGET_ATTR
#define SIMD_ENABLE_SSE4 (SIMD_ENABLE & SIMD_SSE4)
#define SIMD_ENABLE_AVX2 (SIMD_ENABLE & SIMD_AVX2)
#define SIMD_ENABLE_NEON (SIMD_ENABLE & SIMD_ARM)
#define SIMD_ATTR_SSE4 SIMD_TARGET_ATTR("sse4.2,aes,pclmul")
#define SIMD_ATTR_AVX2 SIMD_TARGET_ATTR("avx,avx2,fma")

// Older compiler: can only use an instruction set if the extra flags are set.
#else
#define SIMD_ATTR_SSE4
#define SIMD_ATTR_AVX2

#if defined(__SSE4_2__) && defined(__AES__)
#define SIMD_ENABLE_SSE4 (SIMD_ENABLE & SIMD_SSE4)
#else
#define SIMD_ENABLE_SSE4 0
#endif

#if defined(__AVX2__) && defined(__FMA__)
#define SIMD_ENABLE_AVX2 (SIMD_ENABLE & SIMD_AVX2)
#else
#define SIMD_ENABLE_AVX2 0
#endif

#if defined(SIMD_ARCH_ARM) && defined(__ARM_NEON)
#define SIMD_ENABLE_NEON (SIMD_ENABLE & SIMD_ARM)
#else
#define SIMD_ENABLE_NEON 0
#endif

#endif

// Whether any SIMD instruction set is active (used to exclude tests not
// supported by scalar.h).
#define SIMD_ENABLE_ANY (SIMD_ENABLE_SSE4 | SIMD_ENABLE_AVX2 | SIMD_ENABLE_NEON)

// Detects "best available" instruction set and includes their headers;
// defines SIMD_NAMESPACE, SIMD_TARGET for vec<T, SIMD_TARGET> and SIMD_ATTR
// for user-defined functions. NOTE: headers cannot be included from
// SIMD_NAMESPACE due to conflicts with other system headers. ODR violations
// are avoided if all their functions (static inline in Clang's library and
// extern inline in GCC's) are inlined.
#if SIMD_ARCH_X86
#if SIMD_ENABLE_AVX2
#include <immintrin.h>
#define SIMD_NAMESPACE avx2
#define SIMD_TARGET AVX2
#define SIMD_ATTR SIMD_ATTR_AVX2

#elif SIMD_ENABLE_SSE4
#include <smmintrin.h>
#include <wmmintrin.h>
#define SIMD_NAMESPACE sse4
#define SIMD_TARGET SSE4
#define SIMD_ATTR SIMD_ATTR_SSE4

#else
// No instruction set enabled, but we still need "SSE2" for cache-control.
#include <emmintrin.h>
#endif
#endif

#if SIMD_ENABLE_NEON
#include <arm_neon.h>
#define SIMD_NAMESPACE neon
#define SIMD_TARGET ARM8
#define SIMD_ATTR
#endif

#ifndef SIMD_TARGET
#define SIMD_NAMESPACE none
#define SIMD_TARGET None
#define SIMD_ATTR
#endif

namespace pik {

// Instruction set tag names used to specialize VecT - results in more
// understandable mangled names than using SIMD_SSE4=4 directly.
#if SIMD_ARCH_X86
struct SSE4 {
  static constexpr int value = SIMD_SSE4;
};
struct AVX2 {
  static constexpr int value = SIMD_AVX2;
};
struct AVX512 {
  static constexpr int value = SIMD_AVX512;
};
#elif SIMD_ARCH_PPC
struct PPC {
  static constexpr int value = SIMD_PPC;
};
#elif SIMD_ARCH_ARM
struct ARM8 {
  static constexpr int value = SIMD_ARM;
};
#endif

struct None {
  static constexpr int value = SIMD_NONE;
};

}  // namespace pik

#endif  // SIMD_PORT_H_
