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

#ifndef ARCH_SPECIFIC_H_
#define ARCH_SPECIFIC_H_

// WARNING: this is a "restricted" header because it is included from
// translation units compiled with different flags. This header and its
// dependencies must not define any function unless it is static inline and/or
// within namespace PIK_TARGET_NAME.
//
// Background: older GCC/Clang require flags such as -mavx2 before AVX2 SIMD
// intrinsics can be used. These intrinsics are only used within blocks that
// first verify CPU capabilities. However, the flag also allows the compiler to
// generate AVX2 code in other places. This can violate the One Definition Rule,
// which requires multiple instances of a function with external linkage
// (e.g. extern inline in a header) to be "equivalent". To prevent the resulting
// crashes on non-AVX2 CPUs, any header (transitively) included from a
// translation unit compiled with different flags is "restricted". This means
// all function definitions must have internal linkage (e.g. static inline), or
// reside in namespace PIK_TARGET_NAME, which expands to a name unique to the
// current compiler flags.
//
// Most C system headers are safe to include, but C++ headers should generally
// be avoided because they often do not specify static linkage and cannot
// reliably be wrapped in a namespace.

#include "compiler_specific.h"

#include <stdint.h>

namespace pik {

#if defined(__x86_64__) || defined(_M_X64)
#define PIK_ARCH_X64 1
#else
#define PIK_ARCH_X64 0
#endif

#ifdef __aarch64__
#define PIK_ARCH_AARCH64 1
#else
#define PIK_ARCH_AARCH64 0
#endif

#if defined(__powerpc64__) || defined(_M_PPC)
#define PIK_ARCH_PPC 1
#else
#define PIK_ARCH_PPC 0
#endif

// Target := instruction set extension(s) such as SSE41. A translation unit can
// only provide a single target-specific implementation because they require
// different compiler flags.

// Either the build system specifies the target by defining PIK_TARGET_NAME
// (which is necessary for Portable on X64, and SSE41 on MSVC), or we'll choose
// the most efficient one that can be compiled given the current flags:
#ifndef PIK_TARGET_NAME

// To avoid excessive code size and dispatch overhead, we only support a few
// groups of extensions, e.g. FMA+BMI2+AVX+AVX2 =: "AVX2". These names must
// match the PIK_TARGET_* suffixes below.
#ifdef __AVX2__
#define PIK_TARGET_NAME AVX2
#elif defined(__SSE4_1__)
#define PIK_TARGET_NAME SSE41
#else
#define PIK_TARGET_NAME Portable
#endif

#endif  // PIK_TARGET_NAME

#define PIK_CONCAT(first, second) first##second
// Required due to macro expansion rules.
#define PIK_EXPAND_CONCAT(first, second) PIK_CONCAT(first, second)
// Appends PIK_TARGET_NAME to "identifier_prefix".
#define PIK_ADD_TARGET_SUFFIX(identifier_prefix) \
  PIK_EXPAND_CONCAT(identifier_prefix, PIK_TARGET_NAME)

// PIK_TARGET expands to an integer constant (template argument).
// This ensures your code will work correctly when compiler flags are changed,
// and benefit from subsequently added targets/specializations.
#define PIK_TARGET PIK_ADD_TARGET_SUFFIX(PIK_TARGET_)

// Associate targets with integer literals so the preprocessor can compare them
// with PIK_TARGET. Do not instantiate templates with these values - use
// PIK_TARGET instead. Must be unique powers of two, see TargetBits. Always
// defined even if unavailable on this PIK_ARCH_* to allow calling TargetName.
// The suffixes must match the PIK_TARGET_NAME identifiers.
#define PIK_TARGET_Portable 1
#define PIK_TARGET_SSE41 2
#define PIK_TARGET_AVX2 4

// Bit array for one or more PIK_TARGET_*. Used to indicate which target(s) are
// supported or were called by InstructionSets::RunAll.
using TargetBits = unsigned;

namespace PIK_TARGET_NAME {

// Calls func(bit_value) for every nonzero bit in "bits".
template <class Func>
void ForeachTarget(TargetBits bits, const Func& func) {
  while (bits != 0) {
    const TargetBits lowest = bits & (~bits + 1);
    func(lowest);
    bits &= ~lowest;
  }
}

}  // namespace PIK_TARGET_NAME

// Returns a brief human-readable string literal identifying one of the above
// bits, or nullptr if zero, multiple, or unknown bits are set.
const char* TargetName(const TargetBits target_bit);

#if PIK_ARCH_X64

// This constant avoids image.h depending on vector256.h.
enum { kVectorSize = 32 };  // AVX-2

// Calls CPUID instruction with eax=level and ecx=count and returns the result
// in abcd array where abcd = {eax, ebx, ecx, edx} (hence the name abcd).
void Cpuid(const uint32_t level, const uint32_t count,
           uint32_t* PIK_RESTRICT abcd);

// Returns the APIC ID of the CPU on which we're currently running.
uint32_t ApicId();

// Returns nominal CPU clock frequency for converting tsc_timer cycles to
// seconds. This is unaffected by CPU throttling ("invariant"). Thread-safe.
double InvariantCyclesPerSecond();

float X64_Reciprocal12(const float x);

#endif  // PIK_ARCH_X64

}  // namespace pik

#endif  // ARCH_SPECIFIC_H_
