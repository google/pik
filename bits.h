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

#ifdef _MSC_VER
#include <intrin.h>
#endif

#include <stdint.h>
#include "compiler_specific.h"

static PIK_INLINE int PopCount(const uint32_t x) {
#ifdef _MSC_VER
  return _mm_popcnt_u32(x);
#else
  return __builtin_popcount(x);
#endif
}

// Undefined results for x == 0.
static PIK_INLINE int NumZeroBitsAboveMSBNonzero(const uint32_t x) {
#ifdef _MSC_VER
  unsigned long index;
  _BitScanReverse(&index, x);
  return index;
#else
  return __builtin_clz(x);
#endif
}
static PIK_INLINE int NumZeroBitsAboveMSBNonzero(const uint64_t x) {
#ifdef _MSC_VER
  unsigned long index;
  _BitScanReverse64(&index, x);
  return index;
#else
  return __builtin_clzl(x);
#endif
}
static PIK_INLINE int NumZeroBitsBelowLSBNonzero(const uint32_t x) {
#ifdef _MSC_VER
  unsigned long index;
  _BitScanForward(&index, x);
  return index;
#else
  return __builtin_ctz(x);
#endif
}
static PIK_INLINE int NumZeroBitsBelowLSBNonzero(const uint64_t x) {
#ifdef _MSC_VER
  unsigned long index;
  _BitScanForward64(&index, x);
  return index;
#else
  return __builtin_ctzl(x);
#endif
}

// Returns bit width for x == 0.
static PIK_INLINE int NumZeroBitsAboveMSB(const uint32_t x) {
  return (x == 0) ? 32 : NumZeroBitsAboveMSBNonzero(x);
}
static PIK_INLINE int NumZeroBitsAboveMSB(const uint64_t x) {
  return (x == 0) ? 64 : NumZeroBitsAboveMSBNonzero(x);
}
static PIK_INLINE int NumZeroBitsBelowLSB(const uint32_t x) {
  return (x == 0) ? 32 : NumZeroBitsBelowLSBNonzero(x);
}
static PIK_INLINE int NumZeroBitsBelowLSB(const uint64_t x) {
  return (x == 0) ? 64 : NumZeroBitsBelowLSBNonzero(x);
}

// Returns base-2 logarithm, rounded down.
static PIK_INLINE int FloorLog2Nonzero(const uint32_t x) {
  return 31 ^ NumZeroBitsAboveMSBNonzero(x);
}
static PIK_INLINE int FloorLog2Nonzero(const uint64_t x) {
  return 63 ^ NumZeroBitsAboveMSBNonzero(x);
}

// Returns base-2 logarithm, rounded up.
static PIK_INLINE int CeilLog2Nonzero(const uint32_t x) {
  const int floor_log2 = FloorLog2Nonzero(x);
  if ((x & (x - 1)) == 0) return floor_log2;  // power of two
  return floor_log2 + 1;
}

static PIK_INLINE int CeilLog2Nonzero(const uint64_t x) {
  const int floor_log2 = FloorLog2Nonzero(x);
  if ((x & (x - 1)) == 0) return floor_log2;  // power of two
  return floor_log2 + 1;
}

#endif  // BITS_H_
