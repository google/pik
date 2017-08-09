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

#ifndef BITS_H_
#define BITS_H_

#include <stdint.h>
#include "compiler_specific.h"

// WARNING: this file currently only exports macros, so it is exempt from
// target-specific namespaces (see arch_specific.h).

#ifdef __LZCNT__
    #if PIK_COMPILER_CLANG
        #define PIK_LZCNT32 __lzcnt32
        #define PIK_LZCNT64 __lzcnt64
    #else
        #define PIK_LZCNT32 _lzcnt_u32
        #define PIK_LZCNT64 _lzcnt_u64
    #endif
#else
    #define PIK_LZCNT32 NumZeroBitsAboveMSB32
    #define PIK_LZCNT64 NumZeroBitsAboveMSB64
#endif

#ifdef __BMI__
#define PIK_TZCNT32 _tzcnt_u32
#define PIK_TZCNT64 _tzcnt_u64
#else
#define PIK_TZCNT32 NumZeroBitsBelowLSB32
#define PIK_TZCNT64 NumZeroBitsBelowLSB64
#endif

// The underlying instructions are actually undefined for x == 0.
// These functions are also used in AVX2 builds to verify PIK_LZCNT32 etc.
static PIK_INLINE int NumZeroBitsAboveMSB32(const uint32_t x) {
  return (x == 0) ? 32 : __builtin_clz(x);
}
static PIK_INLINE int NumZeroBitsAboveMSB64(const uint64_t x) {
  return (x == 0) ? 64 : __builtin_clzl(x);
}
static PIK_INLINE int NumZeroBitsBelowLSB32(const uint32_t x) {
  return (x == 0) ? 32 : __builtin_ctz(x);
}
static PIK_INLINE int NumZeroBitsBelowLSB64(const uint64_t x) {
  return (x == 0) ? 64 : __builtin_ctzl(x);
}

#endif  // BITS_H_
