// Copyright 2016 Google Inc. All Rights Reserved.
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

#ifndef COMPILER_SPECIFIC_H_
#define COMPILER_SPECIFIC_H_

// WARNING: compiled with different flags => must not define/instantiate any
// inline functions, nor include any headers that do - see instruction_sets.h.

#include <stdint.h>

// #if is shorter and safer than #ifdef. *_VERSION are zero if not detected,
// otherwise 100 * major + minor version. Note that other packages check for
// #ifdef COMPILER_MSVC, so we cannot use that same name.

#ifdef _MSC_VER
#define PIK_COMPILER_MSVC _MSC_VER
#else
#define PIK_COMPILER_MSVC 0
#endif

#ifdef __GNUC__
#define PIK_COMPILER_GCC (__GNUC__ * 100 + __GNUC_MINOR__)
#else
#define PIK_COMPILER_GCC 0
#endif

#ifdef __clang__
// For reasons unknown, Forge currently explicitly defines these to 0.0.
#define PIK_COMPILER_CLANG 1  // (__clang_major__ * 100 + __clang_minor__)
// Clang pretends to be GCC for compatibility.
#undef PIK_COMPILER_GCC
#define PIK_COMPILER_GCC 0
#else
#define PIK_COMPILER_CLANG 0
#endif

#if PIK_COMPILER_MSVC
#define PIK_RESTRICT __restrict
#elif PIK_COMPILER_GCC
#define PIK_RESTRICT __restrict__
#else
#define PIK_RESTRICT
#endif

#if PIK_COMPILER_MSVC
#define PIK_INLINE __forceinline
#define PIK_NOINLINE __declspec(noinline)
#else
#define PIK_INLINE inline __attribute__((always_inline))
#define PIK_NOINLINE __attribute__((noinline))
#endif

#if PIK_COMPILER_MSVC
#define PIK_NORETURN __declspec(noreturn)
#elif PIK_COMPILER_GCC
#define PIK_NORETURN __attribute__((noreturn))
#endif

#if PIK_COMPILER_MSVC
#define PIK_UNREACHABLE __assume(false)
#elif PIK_COMPILER_CLANG || PIK_COMPILER_GCC >= 405
#define PIK_UNREACHABLE __builtin_unreachable()
#else
#define PIK_UNREACHABLE
#endif

#if PIK_COMPILER_MSVC
// Unsupported, __assume is not the same.
#define PIK_LIKELY(expr) expr
#define PIK_UNLIKELY(expr) expr
#else
#define PIK_LIKELY(expr) __builtin_expect(!!(expr), 1)
#define PIK_UNLIKELY(expr) __builtin_expect(!!(expr), 0)
#endif

#if PIK_COMPILER_MSVC
#include <intrin.h>

#pragma intrinsic(_ReadWriteBarrier)
#define PIK_COMPILER_FENCE _ReadWriteBarrier()
#elif PIK_COMPILER_GCC
#define PIK_COMPILER_FENCE asm volatile("" : : : "memory")
#else
#define PIK_COMPILER_FENCE
#endif

// Returns a void* pointer which the compiler then assumes is N-byte aligned.
// Example: float* PIK_RESTRICT aligned = (float*)PIK_ASSUME_ALIGNED(in, 32);
//
// The assignment semantics are required by GCC/Clang. ICC provides an in-place
// __assume_aligned, whereas MSVC's __assume appears unsuitable.
#if PIK_COMPILER_GCC
    #define PIK_ASSUME_ALIGNED(ptr, align) __builtin_assume_aligned((ptr), (align))
#elif PIK_COMPILER_CLANG
    #if __has_builtin(__builtin_assume_aligned)
        #define PIK_ASSUME_ALIGNED(ptr, align) __builtin_assume_aligned((ptr), (align))
    #else
        // Early versions of Clang did not support __builtin_assume_aligned.
        #define PIK_ASSUME_ALIGNED(ptr, align) ptr
    #endif // __has_builtin(__builtin_assume_aligned)
#else
    #define PIK_ASSUME_ALIGNED(ptr, align) ptr /* not supported */
#endif

#endif  // COMPILER_SPECIFIC_H_
