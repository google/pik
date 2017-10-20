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

// Ensures an array is aligned and suitable for load()/store() functions.
// Example: SIMD_ALIGN T lanes[V::N];
#define SIMD_ALIGN alignas(32)

// SIMD_TARGET_ATTR prerequisites on Clang/GCC: __has_attribute(target) does not
// guarantee intrinsics are usable, hence we must check for specific versions.
#if defined(__clang__)
// Apple 8.2 == Clang 3.9
#ifdef __apple_build_version__
#if __clang_major__ > 8 || (__clang_major__ == 8 && __clang_minor__ >= 2)
#define SIMD_HAVE_ATTR
#endif
// llvm.org Clang 3.9
#elif __clang_major__ > 3 || (__clang_major__ == 3 && __clang_minor__ >= 9)
#define SIMD_HAVE_ATTR
#endif
// GCC 4.9
#elif defined(__GNUC__) && \
    (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 9))
#define SIMD_HAVE_ATTR
#endif

// TODO(janwas): re-enable once PIK functions have SIMD_ATTR annotations.
#if 0 && defined(SIMD_HAVE_ATTR)
#define SIMD_USE_ATTR 1
#else
#define SIMD_USE_ATTR 0
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
#if SIMD_USE_ATTR
#define SIMD_TARGET_ATTR(feature_str) __attribute__((target(feature_str)))
#else
#define SIMD_TARGET_ATTR(feature_str)
#endif

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
#if SIMD_ARCH == SIMD_ARCH_X86
#define SIMD_AVX2 2
#define SIMD_SSE4 4
#define SIMD_AVX512 16
#elif SIMD_ARCH == SIMD_ARCH_PPC
#define SIMD_PPC 1  // v2.07 or 3
#elif SIMD_ARCH == SIMD_ARCH_ARM
#define SIMD_ARM8 8
#endif

// Target-specific attributes required by each instruction set, only required
// #if SIMD_USE_ATTR.
#define SIMD_ATTR_NONE
#define SIMD_ATTR_SSE4 SIMD_TARGET_ATTR("sse4.2,aes,pclmul")
#define SIMD_ATTR_AVX2 SIMD_TARGET_ATTR("avx,avx2,fma")
#define SIMD_ATTR_ARM8

// Default to portable mode (only scalar.h). This macro should only be set by
// the build system and tested below; users check #if SIMD_ENABLE_* to decide
// whether the instruction set is actually usable on this compiler.
#ifndef SIMD_ENABLE
#define SIMD_ENABLE 0
#endif

#if SIMD_ARCH == SIMD_ARCH_X86

// Enabled := (SIMD_USE_ATTR || -m flags) && SIMD_ENABLE bit set.

#if SIMD_USE_ATTR || (defined(__SSE4_2__) && defined(__AES__))
#define SIMD_ENABLE_SSE4 (SIMD_ENABLE & SIMD_SSE4)
#else
#define SIMD_ENABLE_SSE4 0
#endif

#if SIMD_USE_ATTR || (defined(__AVX2__) && defined(__FMA__))
#define SIMD_ENABLE_AVX2 (SIMD_ENABLE & SIMD_AVX2)
#else
#define SIMD_ENABLE_AVX2 0
#endif

#define SIMD_ENABLE_AVX512 0

#elif SIMD_ARCH == SIMD_ARCH_ARM

#if SIMD_USE_ATTR || (defined(__ARM_NEON))
#define SIMD_ENABLE_ARM8 (SIMD_ENABLE & SIMD_ARM8)
#else
#define SIMD_ENABLE_ARM8 0
#endif

#endif  // SIMD_ARCH

// Detects "best available" instruction set and includes their headers.
// NOTE: system headers cannot be included from within SIMD_NAMESPACE because
// of conflicts with other headers. ODR violations are avoided if all their
// functions (static inline in Clang's library and extern inline in GCC's) are
// inlined. SIMD_TARGET is for vec<T, SIMD_TARGET> and also used to define
// SIMD_ATTR SIMD_BITS is the maximum vector size [bits], or zero if only
// scalar.h is available.

#if SIMD_ENABLE_AVX2
#include <immintrin.h>
#define SIMD_NAMESPACE avx2
#define SIMD_TARGET AVX2
#define SIMD_BITS 256

#elif SIMD_ENABLE_SSE4
#include <smmintrin.h>
#include <wmmintrin.h>
#define SIMD_NAMESPACE sse4
#define SIMD_TARGET SSE4
#define SIMD_BITS 128

#elif SIMD_ARCH == SIMD_ARCH_X86
// No instruction set enabled, but we still need "SSE2" for cache-control.
#include <emmintrin.h>
#endif

#if SIMD_ENABLE_ARM8
#include <arm_neon.h>
#define SIMD_NAMESPACE arm
#define SIMD_TARGET ARM8
#define SIMD_BITS 128
#endif

// Nothing enabled => only use scalar.h.
#ifndef SIMD_TARGET
#define SIMD_NAMESPACE none
#define SIMD_TARGET NONE
#define SIMD_BITS 0
#endif

#define SIMD_CONCAT(first, second) first##second
// Required due to macro expansion rules.
#define SIMD_EXPAND_CONCAT(first, second) SIMD_CONCAT(first, second)

// Evaluates to nothing if !SIMD_USE_ATTR.
#define SIMD_ATTR SIMD_EXPAND_CONCAT(SIMD_ATTR_, SIMD_TARGET)

// SIMD_TARGET expands to one of the structs below (for specializing templates);
// for the preprocessor, use this instead: #if SIMD_TARGET_VALUE == SIMD_SSE4.
#define SIMD_TARGET_VALUE SIMD_EXPAND_CONCAT(SIMD_, SIMD_TARGET)

namespace pik {

// Instruction set tag names used to specialize VecT - results in more
// understandable mangled names than using SIMD_SSE4=4 directly. Must match
// the SIMD_TARGET definitions above, and the suffixes of their SIMD_*.
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

// Default to SIMD_TARGET.
template <bool kFitsIn128>
struct MinTargetT {
  using type = SIMD_TARGET;
};
// On x86, use XMM (less overhead) for all <= 128 bit vectors.
#if SIMD_ARCH == SIMD_ARCH_X86 && SIMD_ENABLE_SSE4
template <>
struct MinTargetT<true> {
  using type = SSE4;
};
#endif
// Chooses the smallest/cheapest target, e.g. for Part<uint32_t, 1>.
template <typename T, size_t N>
using MinTarget = typename MinTargetT<(N * sizeof(T)) <= 16>::type;

}  // namespace pik

#endif  // SIMD_PORT_H_
