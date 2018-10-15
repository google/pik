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

#ifndef DCT_H_
#define DCT_H_

#include "block.h"
#include "compiler_specific.h"
#include "dct_simd_4.h"
#include "dct_simd_8.h"
#include "dct_simd_any.h"
#include "simd/simd.h"
#include "status.h"

namespace pik {

// Final scaling factors of outputs/inputs in the Arai, Agui, and Nakajima
// algorithm computing the DCT/IDCT.
// The algorithm is described in the book JPEG: Still Image Data Compression
// Standard, section 4.3.5.
/* Python snippet to produce these tables:
from mpmath import *
N = 8
def iscale(u):
  eps = sqrt(mpf(0.5)) if u == 0 else mpf(1.0)
  return sqrt(mpf(2) / mpf(N)) * eps * cos(mpf(u) * pi / mpf(2 * N))
def scale(u):
  return mpf(1) / (mpf(N) * iscale(i))
mp.dps = 18
print(", ".join([str(scale(i)) + 'f' for i in range(N)]))
print(", ".join([str(iscale(i)) + 'f' for i in range(N)]))
 */
static const float kDCTScales8[8] = {
  0.353553390593273762f, 0.254897789552079584f,
  0.270598050073098492f, 0.30067244346752264f,
  0.353553390593273762f, 0.449988111568207852f,
  0.653281482438188264f, 1.28145772387075309f
};

static const float kIDCTScales8[8] = {
  0.353553390593273762f, 0.490392640201615225f,
  0.461939766255643378f, 0.415734806151272619f,
  0.353553390593273762f, 0.277785116509801112f,
  0.191341716182544886f, 0.0975451610080641339f
};

static const float kIDCTScales16[16] = {
  0.25f, 0.177632042131274808f,
  0.180239955501736978f, 0.184731156892216368f,
  0.191341716182544886f, 0.200444985785954314f,
  0.212607523691814112f, 0.228686034616512494f,
  0.25f, 0.278654739432954475f,
  0.318189645143208485f, 0.375006192208515097f,
  0.461939766255643378f, 0.608977011699708658f,
  0.906127446352887843f, 1.80352839005774887f
};

static const float kDCTScales16[16] = {
  0.25f, 0.351850934381595615f,
  0.346759961330536865f, 0.33832950029358817f,
  0.326640741219094132f, 0.311806253246667808f,
  0.293968900604839679f, 0.273300466750439372f,
  0.25f, 0.224291896585659071f,
  0.196423739596775545f, 0.166663914619436624f,
  0.135299025036549246f, 0.102631131880589345f,
  0.0689748448207357531f, 0.0346542922997728657f
};

template<size_t N>
constexpr const float* DCTScales() {
  return (N == 8) ? kDCTScales8 : kDCTScales16;
}

template<size_t N>
constexpr const float* IDCTScales() {
  return (N == 8) ? kIDCTScales8 : kIDCTScales16;
}

// https://en.wikipedia.org/wiki/In-place_matrix_transposition#Square_matrices
template <size_t N, class From, class To>
PIK_INLINE void GenericTransposeBlockInplace(const From& from, const To& to) {
  // This does not guarantee anything, just saves from the most stupid mistakes.
  PIK_ASSERT(from.Address(0, 0) == to.Address(0, 0));
  for (size_t n = 0; n < N - 1; ++n) {
    for (size_t m = n + 1; m < N; ++m) {
      // Swap
      const float tmp = from.Read(m, n);
      to.Write(from.Read(n, m), m, n);
      to.Write(tmp, n, m);
    }
  }
}

template <size_t N, class From, class To>
PIK_INLINE void GenericTransposeBlock(const From& from, const To& to) {
  // This does not guarantee anything, just saves from the most stupid mistakes.
  PIK_ASSERT(from.Address(0, 0) != to.Address(0, 0));
  for (size_t n = 0; n < N; ++n) {
    for (size_t m = 0; m < N; ++m) {
      to.Write(from.Read(n, m), m, n);
    }
  }
}

template <class From, class To>
SIMD_ATTR PIK_INLINE void TransposeBlock8(const From& from, const To& to) {
#if SIMD_TARGET_VALUE == SIMD_AVX2
  TransposeBlock8_V8(from, to);
#elif SIMD_TARGET_VALUE == SIMD_NONE
  if (from.Address(0, 0) == to.Address(0, 0)) {
    GenericTransposeBlockInplace<8>(from, to);
  } else {
    GenericTransposeBlock<8>(from, to);
  }
#else  // generic 128-bit
  TransposeBlock8_V4(from, to);
#endif
}

template <class From, class To>
SIMD_ATTR PIK_INLINE void TransposeBlock16(const From& from, const To& to) {
  SIMD_ALIGN float tmp[8 * 8];
  TransposeBlock8(from, to);
  TransposeBlock8(from.View(0, 8), ToBlock<8>(tmp));
  TransposeBlock8(from.View(8, 0), to.View(0, 8));
  CopyBlock8(FromBlock<8>(tmp), to.View(8, 0));
  TransposeBlock8(from.View(8, 8), to.View(8, 8));
}

// Computes the in-place NxN transposed-scaled-DCT (tsDCT) of block.
// Requires that block is SIMD_ALIGN'ed.
//
// Final DCT coefficients could be obtained the following way:
//   unscaled(f)[x, y] = f[x, y] * DCTScales<N>[x] * DCTScales<N>[y]
//   untransposed(f)[x, y] = f[y, x]
//   DCT(input) = unscaled(untransposed(tsDCT(input)))
//
// NB: DCT denotes scaled variant of DCT-II, which is orthonormal.
//
// See also DCTSlow, ComputeDCT
template <size_t N, class From, class To>
static SIMD_ATTR PIK_INLINE void ComputeTransposedScaledDCT(const From& from,
                                                            const To& to) {
  if (N == 8) {
#if SIMD_TARGET_VALUE == SIMD_AVX2
  ComputeTransposedScaledDCT8_V8(from, to);
#elif SIMD_TARGET_VALUE == SIMD_NONE
  SIMD_ALIGN float block[8 * 8];
  ColumnDCT8(from, ToBlock<8>(block));
  TransposeBlock8(FromBlock<8>(block), ToBlock<8>(block));
  ColumnDCT8(FromBlock<8>(block), to);
#else
  ComputeTransposedScaledDCT8_V4(from, to);
#endif
  } else {
    SIMD_ALIGN float block[16 * 16];
    ColumnDCT16(from, ToBlock<16>(block));
    TransposeBlock16(FromBlock<16>(block), ToBlock<16>(block));
    ColumnDCT16(FromBlock<16>(block), to);
  }
}

// Computes the in-place NxN transposed-scaled-iDCT (tsIDCT)of block.
// Requires that block is SIMD_ALIGN'ed.
//
// Final DCT coefficients could be obtained the following way:
//   unscaled(f)[x, y] = f[x, y] * IDCTScales<N>[x] * IDCTScales<N>[y]
//   untransposed(f)[x, y] = f[y, x]
//   IDCT(input) = tsIDCT(untransposed(unscaled(input)))
//
// NB: IDCT denotes scaled variant of DCT-III, which is orthonormal.
//
// See also IDCTSlow, ComputeIDCT.
template <size_t N, class From, class To>
static SIMD_ATTR PIK_INLINE void ComputeTransposedScaledIDCT(const From& from,
                                                             const To& to) {
  if (N == 8) {
#if SIMD_TARGET_VALUE == SIMD_AVX2
  ComputeTransposedScaledIDCT8_V8(from, to);
#elif SIMD_TARGET_VALUE == SIMD_NONE
  SIMD_ALIGN float block[8 * 8];
  ColumnIDCT8(from, ToBlock<8>(block));
  TransposeBlock8(FromBlock<8>(block), ToBlock<8>(block));
  ColumnIDCT8(FromBlock<8>(block), to);
#else
  ComputeTransposedScaledIDCT8_V4(from, to);
#endif
  } else {
    SIMD_ALIGN float block[16 * 16];
    ColumnIDCT16(from, ToBlock<16>(block));
    TransposeBlock16(FromBlock<16>(block), ToBlock<16>(block));
    ColumnIDCT16(FromBlock<16>(block), to);
  }
}

// Similar to ComputeTransposedScaledDCT, but only DC coefficient is calculated.
template <size_t N, class From>
static SIMD_ATTR PIK_INLINE float ComputeScaledDC(const From& from) {
  static_assert(N == 8, "Currently only 8x8 is supported");

#if SIMD_TARGET_VALUE == SIMD_AVX2
  return ComputeScaledDC8_V8(from);
#elif SIMD_TARGET_VALUE == SIMD_NONE
  const BlockDesc d;
  auto sum = setzero(d);
  for (size_t iy = 0; iy < N; ++iy) {
    for (size_t ix = 0; ix < N; ix += d.N) {
      sum += from.Load(iy, ix);
    }
  }
  sum = ext::sum_of_lanes(sum);
  return get_part(SIMD_PART(float, 1)(), sum);
#else
  return ComputeScaledDC8_V4(from);
#endif
}

}  // namespace pik

#endif  // DCT_H_
