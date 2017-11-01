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

// Undefined results for x == 0.
static PIK_INLINE int NumZeroBitsAboveMSB32Nonzero(const uint32_t x) {
  return __builtin_clz(x);
}
static PIK_INLINE int NumZeroBitsAboveMSB64Nonzero(const uint64_t x) {
  return __builtin_clzl(x);
}
static PIK_INLINE int NumZeroBitsBelowLSB32Nonzero(const uint32_t x) {
  return __builtin_ctz(x);
}
static PIK_INLINE int NumZeroBitsBelowLSB64Nonzero(const uint64_t x) {
  return __builtin_ctzl(x);
}

// Returns bit width for x == 0.
static PIK_INLINE int NumZeroBitsAboveMSB32(const uint32_t x) {
  return (x == 0) ? 32 : NumZeroBitsAboveMSB32Nonzero(x);
}
static PIK_INLINE int NumZeroBitsAboveMSB64(const uint64_t x) {
  return (x == 0) ? 64 : NumZeroBitsAboveMSB64Nonzero(x);
}
static PIK_INLINE int NumZeroBitsBelowLSB32(const uint32_t x) {
  return (x == 0) ? 32 : NumZeroBitsBelowLSB32Nonzero(x);
}
static PIK_INLINE int NumZeroBitsBelowLSB64(const uint64_t x) {
  return (x == 0) ? 64 : NumZeroBitsBelowLSB64Nonzero(x);
}

#endif  // BITS_H_
