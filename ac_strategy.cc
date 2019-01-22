// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "ac_strategy.h"
#include "block.h"
#include "common.h"
#include "dct.h"
#include "dct_util.h"
#include "entropy_coder.h"
#include "image.h"
#include "opsin_params.h"
#include "profiler.h"
#include "simd/simd.h"

namespace pik {
namespace {
// True if we should try to find a non-trivial AC strategy.
const constexpr bool kChooseAcStrategy = true;

// Returns the value such that ComputeTransposedScaledDCT<N>() of a block with
// this value in position (x, y) and 0s everywhere else will have the average of
// absolute values of 1.
template <size_t N>
float DCTTotalScale(size_t x, size_t y) {
  return N * DCTScales<N>()[x] * DCTScales<N>()[y] * L1NormInv<N>()[x] *
         L1NormInv<N>()[y];
}
template <size_t N>
float DCTInvTotalScale(size_t x, size_t y) {
  return N * IDCTScales<N>()[x] * IDCTScales<N>()[y] * L1Norm<N>()[x] *
         L1Norm<N>()[y];
}

// Computes the lowest-frequency LFxLF-sized square in output, which is a
// DCTN-sized DCT block, by doing a NxN DCT on the input block.
template <size_t DCTN, size_t LF, size_t N>
SIMD_ATTR PIK_INLINE void ReinterpretingDCT(const float* input,
                                            const size_t input_stride,
                                            float* output,
                                            const size_t output_stride) {
  static_assert(LF == N,
                "ReinterpretingDCT should only be called with LF == N");
  SIMD_ALIGN float block[N * N] = {};
  for (size_t y = 0; y < N; y++) {
    for (size_t x = 0; x < N; x++) {
      block[y * N + x] = input[y * input_stride + x];
    }
  }
  ComputeTransposedScaledDCT<N>()(FromBlock<N>(block), ScaleToBlock<N>(block));
  for (size_t y = 0; y < LF; y++) {
    for (size_t x = 0; x < LF; x++) {
      output[y * output_stride + x] = block[y * N + x] *
                                      DCTTotalScale<N>(x, y) *
                                      DCTInvTotalScale<DCTN>(x, y);
    }
  }
}

// Inverse of ReinterpretingDCT.
template <size_t DCTN, size_t LF, size_t N>
SIMD_ATTR PIK_INLINE void ReinterpretingIDCT(const float* input,
                                             const size_t input_stride,
                                             float* output,
                                             const size_t output_stride) {
  SIMD_ALIGN float block[N * N] = {};
  for (size_t y = 0; y < LF; y++) {
    for (size_t x = 0; x < LF; x++) {
      block[y * N + x] = input[y * input_stride + x] *
                         DCTInvTotalScale<N>(x, y) * DCTTotalScale<DCTN>(x, y);
    }
  }
  ComputeTransposedScaledIDCT<N>()(FromBlock<N>(block), ToBlock<N>(block));

  for (size_t y = 0; y < N; y++) {
    for (size_t x = 0; x < N; x++) {
      output[y * output_stride + x] = block[y * N + x];
    }
  }
}

template <size_t S>
void DCT2TopBlock(const float* block, size_t stride, float* out) {
  static_assert(kBlockDim % S == 0, "S should be a divisor of kBlockDim");
  static_assert(S % 2 == 0, "S should be even");
  float temp[kBlockDim * kBlockDim];
  constexpr size_t num_2x2 = S / 2;
  for (size_t y = 0; y < num_2x2; y++) {
    for (size_t x = 0; x < num_2x2; x++) {
      float c00 = block[y * 2 * stride + x * 2];
      float c01 = block[y * 2 * stride + x * 2 + 1];
      float c10 = block[(y * 2 + 1) * stride + x * 2];
      float c11 = block[(y * 2 + 1) * stride + x * 2 + 1];
      float r00 = c00 + c01 + c10 + c11;
      float r01 = c00 + c01 - c10 - c11;
      float r10 = c00 - c01 + c10 - c11;
      float r11 = c00 - c01 - c10 + c11;
      r00 *= 0.25f;
      r01 *= 0.25f;
      r10 *= 0.25f;
      r11 *= 0.25f;
      temp[y * kBlockDim + x] = r00;
      temp[y * kBlockDim + num_2x2 + x] = r01;
      temp[(y + num_2x2) * kBlockDim + x] = r10;
      temp[(y + num_2x2) * kBlockDim + num_2x2 + x] = r11;
    }
  }
  for (size_t y = 0; y < S; y++) {
    for (size_t x = 0; x < S; x++) {
      out[y * kBlockDim + x] = temp[y * kBlockDim + x];
    }
  }
}

template <size_t S>
void IDCT2TopBlock(const float* block, size_t stride, float* out) {
  static_assert(kBlockDim % S == 0, "S should be a divisor of kBlockDim");
  static_assert(S % 2 == 0, "S should be even");
  float temp[kBlockDim * kBlockDim];
  constexpr size_t num_2x2 = S / 2;
  for (size_t y = 0; y < num_2x2; y++) {
    for (size_t x = 0; x < num_2x2; x++) {
      float c00 = block[y * kBlockDim + x];
      float c01 = block[y * kBlockDim + num_2x2 + x];
      float c10 = block[(y + num_2x2) * kBlockDim + x];
      float c11 = block[(y + num_2x2) * kBlockDim + num_2x2 + x];
      float r00 = c00 + c01 + c10 + c11;
      float r01 = c00 + c01 - c10 - c11;
      float r10 = c00 - c01 + c10 - c11;
      float r11 = c00 - c01 - c10 + c11;
      temp[y * 2 * stride + x * 2] = r00;
      temp[y * 2 * stride + x * 2 + 1] = r01;
      temp[(y * 2 + 1) * stride + x * 2] = r10;
      temp[(y * 2 + 1) * stride + x * 2 + 1] = r11;
    }
  }
  for (size_t y = 0; y < S; y++) {
    for (size_t x = 0; x < S; x++) {
      out[y * kBlockDim + x] = temp[y * kBlockDim + x];
    }
  }
}

}  // namespace

SIMD_ATTR void AcStrategy::TransformFromPixels(
    const float* pixels, size_t pixels_stride, float* coefficients,
    size_t coefficients_stride) const {
  if (block_ != 0) return;
  switch (strategy_) {
    case Type::IDENTITY: {
      SIMD_ALIGN float coeffs[kBlockDim * kBlockDim];
      for (size_t y = 0; y < 2; y++) {
        for (size_t x = 0; x < 2; x++) {
          float block_dc = 0;
          for (size_t iy = 0; iy < 4; iy++) {
            for (size_t ix = 0; ix < 4; ix++) {
              block_dc += pixels[(y * 4 + iy) * pixels_stride + x * 4 + ix];
            }
          }
          block_dc *= 1.0f / 16;
          for (size_t iy = 0; iy < 4; iy++) {
            for (size_t ix = 0; ix < 4; ix++) {
              if (ix == 1 && iy == 1) continue;
              coeffs[(y + iy * 2) * 8 + x + ix * 2] =
                  pixels[(y * 4 + iy) * pixels_stride + x * 4 + ix] -
                  pixels[(y * 4 + 1) * pixels_stride + x * 4 + 1];
            }
          }
          coeffs[(y + 2) * 8 + x + 2] = coeffs[y * 8 + x];
          coeffs[y * 8 + x] = block_dc;
        }
      }
      float block00 = coeffs[0];
      float block01 = coeffs[1];
      float block10 = coeffs[8];
      float block11 = coeffs[9];
      coeffs[0] = (block00 + block01 + block10 + block11) * 0.25f;
      coeffs[1] = (block00 + block01 - block10 - block11) * 0.25f;
      coeffs[8] = (block00 - block01 + block10 - block11) * 0.25f;
      coeffs[9] = (block00 - block01 - block10 + block11) * 0.25f;
      memcpy(coefficients, coeffs, kBlockDim * kBlockDim * sizeof(float));
      break;
    }
    case Type::DCT4X4_NOHF:
    case Type::DCT4X4: {
      SIMD_ALIGN float coeffs[kBlockDim * kBlockDim];
      for (size_t y = 0; y < 2; y++) {
        for (size_t x = 0; x < 2; x++) {
          float block[4 * 4];
          ComputeTransposedScaledDCT<4>()(
              FromLines<4>(pixels + y * 4 * pixels_stride + x * 4,
                           pixels_stride),
              ScaleToBlock<4>(block));
          for (size_t iy = 0; iy < 4; iy++) {
            for (size_t ix = 0; ix < 4; ix++) {
              coeffs[(y + iy * 2) * 8 + x + ix * 2] = block[iy * 4 + ix];
            }
          }
        }
      }
      float block00 = coeffs[0];
      float block01 = coeffs[1];
      float block10 = coeffs[8];
      float block11 = coeffs[9];
      coeffs[0] = (block00 + block01 + block10 + block11) * 0.25f;
      coeffs[1] = (block00 + block01 - block10 - block11) * 0.25f;
      coeffs[8] = (block00 - block01 + block10 - block11) * 0.25f;
      coeffs[9] = (block00 - block01 - block10 + block11) * 0.25f;
      memcpy(coefficients, coeffs, kBlockDim * kBlockDim * sizeof(float));
      break;
    }
    case Type::DCT2X2: {
      SIMD_ALIGN float coeffs[kBlockDim * kBlockDim];
      DCT2TopBlock<8>(pixels, pixels_stride, coeffs);
      DCT2TopBlock<4>(coeffs, kBlockDim, coeffs);
      DCT2TopBlock<2>(coeffs, kBlockDim, coeffs);
      memcpy(coefficients, coeffs, kBlockDim * kBlockDim * sizeof(float));
      break;
    }
    case Type::DCT16X16: {
      SIMD_ALIGN float output[4 * kBlockDim * kBlockDim];
      ComputeTransposedScaledDCT<2 * kBlockDim>()(
          FromLines<2 * kBlockDim>(pixels, pixels_stride),
          ScaleToBlock<2 * kBlockDim>(output));
      ScatterBlock<2 * kBlockDim, 2 * kBlockDim>(output, coefficients,
                                                 coefficients_stride);
      break;
    }
    case Type::DCT32X32: {
      SIMD_ALIGN float output[16 * kBlockDim * kBlockDim];
      ComputeTransposedScaledDCT<4 * kBlockDim>()(
          FromLines<4 * kBlockDim>(pixels, pixels_stride),
          ScaleToBlock<4 * kBlockDim>(output));
      ScatterBlock<4 * kBlockDim, 4 * kBlockDim>(output, coefficients,
                                                 coefficients_stride);
      break;
    }
    case Type::DCT_NOHF:
    case Type::DCT: {
      ComputeTransposedScaledDCT<kBlockDim>()(
          FromLines<kBlockDim>(pixels, pixels_stride),
          ScaleToBlock<kBlockDim>(coefficients));
      break;
    }
  }
}

SIMD_ATTR void AcStrategy::TransformToPixels(const float* coefficients,
                                             size_t coefficients_stride,
                                             float* pixels,
                                             size_t pixels_stride) const {
  if (block_ != 0) return;
  switch (strategy_) {
    case Type::IDENTITY: {
      SIMD_ALIGN float coeffs[kBlockDim * kBlockDim];
      memcpy(coeffs, coefficients, kBlockDim * kBlockDim * sizeof(float));
      float dcs[4] = {};
      float block00 = coeffs[0];
      float block01 = coeffs[1];
      float block10 = coeffs[8];
      float block11 = coeffs[9];
      dcs[0] = block00 + block01 + block10 + block11;
      dcs[1] = block00 + block01 - block10 - block11;
      dcs[2] = block00 - block01 + block10 - block11;
      dcs[3] = block00 - block01 - block10 + block11;
      for (size_t y = 0; y < 2; y++) {
        for (size_t x = 0; x < 2; x++) {
          float block_dc = dcs[y * 2 + x];
          float residual_sum = 0;
          for (size_t iy = 0; iy < 4; iy++) {
            for (size_t ix = 0; ix < 4; ix++) {
              if (ix == 0 && iy == 0) continue;
              residual_sum += coeffs[(y + iy * 2) * 8 + x + ix * 2];
            }
          }
          pixels[(4 * y + 1) * pixels_stride + 4 * x + 1] =
              block_dc - residual_sum * (1.0f / 16);
          for (size_t iy = 0; iy < 4; iy++) {
            for (size_t ix = 0; ix < 4; ix++) {
              if (ix == 1 && iy == 1) continue;
              pixels[(y * 4 + iy) * pixels_stride + x * 4 + ix] =
                  coeffs[(y + iy * 2) * 8 + x + ix * 2] +
                  pixels[(4 * y + 1) * pixels_stride + 4 * x + 1];
            }
          }
          pixels[y * 4 * pixels_stride + x * 4] =
              coeffs[(y + 2) * 8 + x + 2] +
              pixels[(4 * y + 1) * pixels_stride + 4 * x + 1];
        }
      }
      break;
    }
    case Type::DCT4X4_NOHF:
    case Type::DCT4X4: {
      SIMD_ALIGN float coeffs[kBlockDim * kBlockDim];
      memcpy(coeffs, coefficients, kBlockDim * kBlockDim * sizeof(float));
      float dcs[4] = {};
      float block00 = coeffs[0];
      float block01 = coeffs[1];
      float block10 = coeffs[8];
      float block11 = coeffs[9];
      dcs[0] = block00 + block01 + block10 + block11;
      dcs[1] = block00 + block01 - block10 - block11;
      dcs[2] = block00 - block01 + block10 - block11;
      dcs[3] = block00 - block01 - block10 + block11;
      for (size_t y = 0; y < 2; y++) {
        for (size_t x = 0; x < 2; x++) {
          float block[4 * 4];
          block[0] = dcs[y * 2 + x];
          for (size_t iy = 0; iy < 4; iy++) {
            for (size_t ix = 0; ix < 4; ix++) {
              if (ix == 0 && iy == 0) continue;
              block[iy * 4 + ix] = coeffs[(y + iy * 2) * 8 + x + ix * 2];
            }
          }
          ComputeTransposedScaledIDCT<4>()(
              FromBlock<4>(block),
              ToLines<4>(pixels + y * 4 * pixels_stride + x * 4,
                         pixels_stride));
        }
      }
      break;
    }
    case Type::DCT2X2: {
      SIMD_ALIGN float coeffs[kBlockDim * kBlockDim];
      memcpy(coeffs, coefficients, sizeof(float) * kBlockDim * kBlockDim);
      IDCT2TopBlock<2>(coeffs, kBlockDim, coeffs);
      IDCT2TopBlock<4>(coeffs, kBlockDim, coeffs);
      IDCT2TopBlock<8>(coeffs, kBlockDim, coeffs);
      for (size_t y = 0; y < kBlockDim; y++) {
        for (size_t x = 0; x < kBlockDim; x++) {
          pixels[y * pixels_stride + x] = coeffs[y * kBlockDim + x];
        }
      }
      break;
    }
    case Type::DCT16X16: {
      SIMD_ALIGN float output[4 * kBlockDim * kBlockDim];
      GatherBlock<2 * kBlockDim, 2 * kBlockDim>(coefficients,
                                                coefficients_stride, output);
      ComputeTransposedScaledIDCT<2 * kBlockDim>()(
          FromBlock<2 * kBlockDim>(output),
          ToLines<2 * kBlockDim>(pixels, pixels_stride));
      break;
    }
    case Type::DCT32X32: {
      SIMD_ALIGN float output[16 * kBlockDim * kBlockDim];
      GatherBlock<4 * kBlockDim, 4 * kBlockDim>(coefficients,
                                                coefficients_stride, output);
      ComputeTransposedScaledIDCT<4 * kBlockDim>()(
          FromBlock<4 * kBlockDim>(output),
          ToLines<4 * kBlockDim>(pixels, pixels_stride));
      break;
    }
    case Type::DCT_NOHF:
    case Type::DCT: {
      ComputeTransposedScaledIDCT<kBlockDim>()(
          FromBlock<kBlockDim>(coefficients),
          ToLines<kBlockDim>(pixels, pixels_stride));
      break;
    }
  }
}

SIMD_ATTR void AcStrategy::LowestFrequenciesFromDC(const float* PIK_RESTRICT dc,
                                                   size_t dc_stride, float* llf,
                                                   size_t llf_stride) const {
  if (block_) return;
  switch (strategy_) {
    case Type::DCT_NOHF:
    case Type::DCT:
      llf[0] = dc[0];
      break;
    case Type::DCT16X16: {
      float tmp[4] = {};
      ReinterpretingDCT<2 * kBlockDim, 2, 2>(dc, dc_stride, tmp, 2);
      for (size_t y = 0; y < 2; y++) {
        for (size_t x = 0; x < 2; x++) {
          llf[y * llf_stride + x] = tmp[y * 2 + x];
        }
      }
      break;
    }
    case Type::DCT32X32: {
      float tmp[16] = {};
      ReinterpretingDCT<4 * kBlockDim, 4, 4>(dc, dc_stride, tmp, 4);
      for (size_t y = 0; y < 4; y++) {
        for (size_t x = 0; x < 4; x++) {
          llf[y * llf_stride + x] = tmp[y * 4 + x];
        }
      }
      break;
    }
    case Type::DCT2X2:
    case Type::DCT4X4_NOHF:
    case Type::DCT4X4:
    case Type::IDENTITY:
      llf[0] = dc[0];
      break;
  };
}

SIMD_ATTR void AcStrategy::DCFromLowestFrequencies(
    const float* PIK_RESTRICT block, size_t block_stride, float* dc,
    size_t dc_stride) const {
  if (block_) return;
  switch (strategy_) {
    case Type::DCT_NOHF:
    case Type::DCT:
      dc[0] = block[0];
      break;
    case Type::DCT16X16: {
      float dest[4] = {};
      GatherBlock<2 * kBlockDim, 2 * kBlockDim, 2, 2>(block, block_stride,
                                                      dest);
      ReinterpretingIDCT<2 * kBlockDim, 2, 2>(dest, 2, dc, dc_stride);
      break;
    }
    case Type::DCT32X32: {
      float dest[16] = {};
      GatherBlock<4 * kBlockDim, 4 * kBlockDim, 4, 4>(block, block_stride,
                                                      dest);
      ReinterpretingIDCT<4 * kBlockDim, 4, 4>(dest, 4, dc, dc_stride);
      break;
    }
    case Type::DCT2X2:
    case Type::DCT4X4:
    case Type::DCT4X4_NOHF:
    case Type::IDENTITY:
      dc[0] = block[0];
      break;
  }
}

SIMD_ATTR void AcStrategy::DC2x2FromLowestFrequencies(
    const float* PIK_RESTRICT llf, size_t llf_stride, float* PIK_RESTRICT dc2x2,
    size_t dc2x2_stride) const {
  if (block_) return;
  constexpr size_t N = kBlockDim;
  switch (strategy_) {
    case Type::DCT_NOHF:
    case Type::DCT: {
      ReinterpretingIDCT<N, 1, 2>(llf, 0, dc2x2, dc2x2_stride);
      break;
    }
    case Type::DCT16X16: {
      float dest[16] = {};
      dest[0] = llf[0];
      dest[1] = llf[1];
      dest[4] = llf[llf_stride];
      dest[5] = llf[llf_stride + 1];
      ReinterpretingIDCT<2 * N, 2, 4>(dest, 4, dc2x2, dc2x2_stride);
      break;
    }
    case Type::DCT32X32: {
      float dest[64] = {};
      for (size_t iy = 0; iy < 4; iy++) {
        for (size_t ix = 0; ix < 4; ix++) {
          dest[iy * 8 + ix] = llf[iy * llf_stride + ix];
        }
      }
      ReinterpretingIDCT<4 * N, 4, 8>(dest, 8, dc2x2, dc2x2_stride);
      break;
    }
    case Type::DCT2X2:
    case Type::DCT4X4:
    case Type::DCT4X4_NOHF:
    case Type::IDENTITY:
      dc2x2[0] = llf[0];
      dc2x2[1] = llf[0];
      dc2x2[dc2x2_stride] = llf[0];
      dc2x2[dc2x2_stride + 1] = llf[0];
      break;
  }
}

SIMD_ATTR void AcStrategy::DC2x2FromLowFrequencies(const float* block,
                                                   size_t block_stride,
                                                   float* dc2x2,
                                                   size_t dc2x2_stride) const {
  if (block_) return;
  switch (strategy_) {
    case Type::DCT_NOHF:
    case Type::DCT:
      ReinterpretingIDCT<kBlockDim, 2, 2>(block, kBlockDim, dc2x2,
                                          dc2x2_stride);
      break;
    case Type::DCT16X16: {
      float dest[16] = {};
      GatherBlock<2 * kBlockDim, 2 * kBlockDim, 4, 4>(block, block_stride,
                                                      dest);
      ReinterpretingIDCT<2 * kBlockDim, 4, 4>(dest, 4, dc2x2, dc2x2_stride);
      break;
    }
    case Type::DCT32X32: {
      float dest[64] = {};
      GatherBlock<4 * kBlockDim, 4 * kBlockDim, 8, 8>(block, block_stride,
                                                      dest);
      ReinterpretingIDCT<4 * kBlockDim, 8, 8>(dest, 8, dc2x2, dc2x2_stride);
      break;
    }
    case Type::DCT2X2:
    case Type::DCT4X4:
    case Type::DCT4X4_NOHF:
    case Type::IDENTITY:
      float block00 = block[0];
      float block01 = block[1];
      float block10 = block[8];
      float block11 = block[9];
      dc2x2[0] = block00 + block01 + block10 + block11;
      dc2x2[1] = block00 + block01 - block10 - block11;
      dc2x2[dc2x2_stride] = block00 - block01 + block10 - block11;
      dc2x2[dc2x2_stride + 1] = block00 - block01 - block10 + block11;
      break;
  }
}

SIMD_ATTR void AcStrategy::LowFrequenciesFromDC2x2(const float* dc2x2,
                                                   size_t dc2x2_stride,
                                                   float* block,
                                                   size_t block_stride) const {
  if (block_) return;
  switch (strategy_) {
    case Type::DCT_NOHF:
    case Type::DCT:
      ReinterpretingDCT<kBlockDim, 2, 2>(dc2x2, dc2x2_stride, block,
                                         block_stride);
      break;
    case Type::DCT16X16: {
      float dest[16] = {};
      ReinterpretingDCT<2 * kBlockDim, 4, 4>(dc2x2, dc2x2_stride, dest, 4);
      for (size_t y = 0; y < 4; y++) {
        const size_t by = y / 2;
        const size_t iy = y & 1;
        const size_t yp = iy * 2 + by;
        for (size_t x = 0; x < 4; x++) {
          const size_t bx = x / 2;
          const size_t ix = x & 1;
          const size_t xp = ix * 2 + bx;
          block[yp * block_stride + xp] = dest[y * 4 + x];
        }
      }
      break;
    }
    case Type::DCT32X32: {
      float dest[64] = {};
      ReinterpretingDCT<4 * kBlockDim, 8, 8>(dc2x2, dc2x2_stride, dest, 8);
      for (size_t y = 0; y < 8; y++) {
        const size_t by = y / 4;
        const size_t iy = y & 3;
        const size_t yp = iy * 2 + by;
        for (size_t x = 0; x < 8; x++) {
          const size_t bx = x / 4;
          const size_t ix = x & 3;
          const size_t xp = ix * 2 + bx;
          block[yp * block_stride + xp] = dest[y * 8 + x];
        }
      }
      break;
    }
    case Type::DCT2X2:
    case Type::DCT4X4:
    case Type::DCT4X4_NOHF:
    case Type::IDENTITY:
      float block00 = dc2x2[0];
      float block01 = dc2x2[1];
      float block10 = dc2x2[dc2x2_stride];
      float block11 = dc2x2[dc2x2_stride + 1];
      block[0] = (block00 + block01 + block10 + block11) * 0.25f;
      block[1] = (block00 + block01 - block10 - block11) * 0.25f;
      block[block_stride] = (block00 - block01 + block10 - block11) * 0.25f;
      block[block_stride + 1] = (block00 - block01 - block10 + block11) * 0.25f;
  }
}

void AcStrategyImage::SetFromRaw(const Rect& rect, const ImageB& raw_layers) {
  PIK_ASSERT(SameSize(rect, raw_layers));
  size_t stride = layers_.PixelsPerRow();
  for (size_t y = 0; y < rect.ysize(); ++y) {
    uint8_t* PIK_RESTRICT row = rect.Row(&layers_, y);
    for (size_t x = 0; x < rect.xsize(); ++x) {
      row[x] = INVALID;
    }
  }
  for (size_t y = 0; y < rect.ysize(); ++y) {
    const uint8_t* PIK_RESTRICT row_in = raw_layers.Row(y);
    uint8_t* PIK_RESTRICT row = rect.Row(&layers_, y);
    for (size_t x = 0; x < rect.xsize(); ++x) {
      if (row[x] != INVALID) continue;
      uint8_t raw_strategy = row_in[x];
#ifdef ADDRESS_SANITIZER
      PIK_ASSERT(AcStrategy::IsRawStrategyValid(raw_strategy));
#endif
      AcStrategy acs = AcStrategy::FromRawStrategy(raw_strategy);
#ifdef ADDRESS_SANITIZER
      PIK_ASSERT(y + acs.covered_blocks_y() <= rect.ysize());
      PIK_ASSERT(x + acs.covered_blocks_x() <= rect.xsize());
#endif
      for (size_t iy = 0; iy < acs.covered_blocks_y(); iy++) {
        for (size_t ix = 0; ix < acs.covered_blocks_x(); ix++) {
          row[x + ix + iy * stride] =
              (raw_strategy << 4) | (iy * acs.covered_blocks_x() + ix);
        }
      }
    }
  }
}

size_t AcStrategyImage::CountBlocks(AcStrategy::Type type) const {
  size_t ret = 0;
  for (size_t y = 0; y < layers_.ysize(); y++) {
    const uint8_t* PIK_RESTRICT row = layers_.ConstRow(y);
    for (size_t x = 0; x < layers_.xsize(); x++) {
      if (row[x] == (static_cast<uint8_t>(type) << 4)) ret++;
    }
  }
  return ret;
}

SIMD_ATTR void FindBestAcStrategy(float butteraugli_target,
                                  const ImageF* quant_field, const Image3F& src,
                                  ThreadPool* pool,
                                  AcStrategyImage* ac_strategy,
                                  PikInfo* aux_out) {
  PROFILER_FUNC;
  size_t xsize_blocks = src.xsize() / kBlockDim;
  size_t ysize_blocks = src.ysize() / kBlockDim;
  Image3F coeffs = Image3F(xsize_blocks * kBlockDim * kBlockDim, ysize_blocks);
  TransposedScaledDCT(src, &coeffs);
  *ac_strategy = AcStrategyImage(xsize_blocks, ysize_blocks);
  if (!kChooseAcStrategy) {
    return;
  }
  std::vector<bool> disable_dct16(xsize_blocks * ysize_blocks);
  std::vector<bool> disable_dct32(xsize_blocks * ysize_blocks);
  const auto disable_large_transforms = [&](int bx, int by) SIMD_ATTR {
    // If we find a well-fitting DCT4x4 within the larger block,
    // we disable the larger block.
    {
      std::vector<double> blockval(4);
      // 4x4 DCT needs less focus on B channel, since at that resolution
      // blue needs to be correct only by average.
      static const double kColorWeights4x4[3] = {
          0.55560083410056468,
          1.0,
          0.3,
      };
      for (int ix = 0; ix < 4; ++ix) {
        double total_sum = 0;
        int offx = (ix & 1) * 4;
        int offy = (ix & 2) * 2;
        for (size_t c = 0; c < src.kNumPlanes; c++) {
          double sum = 0;
          for (size_t iy = 0; iy < 3; iy++) {
            const float* row0 =
                src.ConstPlaneRow(c, by * kBlockDim + offy + iy);
            const float* row1 =
                src.ConstPlaneRow(c, by * kBlockDim + offy + iy + 1);
            for (size_t dx = 0; dx < 3; dx++) {
              int x = bx * kBlockDim + offx + dx;
              sum += fabs(row0[x] - row0[x + 1]) + fabs(row0[x] - row1[x]);
            }
            {
              int x = bx * kBlockDim + offx + 3;
              sum += fabs(row0[x] - row1[x]);
            }
          }
          int iy = 3;
          const float* row0 = src.ConstPlaneRow(c, by * kBlockDim + offy + iy);
          for (size_t dx = 0; dx < 3; dx++) {
            int x = bx * kBlockDim + offx + dx;
            sum += fabs(row0[x] - row0[x + 1]);
          }
          total_sum += kColorWeights4x4[c] * sum;
        }
        blockval[ix] = total_sum;
      }
      double norm2 = 0.0;
      double norm4 = 0.0;
      double norm8 = 0.0;
      for (int ix = 0; ix < 4; ++ix) {
        double v = blockval[ix];
        v *= v;
        norm2 += v;
        v *= v;
        norm4 += v;
        v *= v;
        norm8 += v;
      }
      norm2 = std::pow(norm2 * (1.0 / 4), 0.5);
      norm4 = std::pow(norm4 * (1.0 / 4), 0.25);
      norm8 = std::pow(norm8 * (1.0 / 4), 0.125);
      norm2 += 0.03;

      double kMul1 = 0.8469907937870913;
      double loss_4x4 = kMul1 * norm8 / norm2;
      double kMul2 = -0.054845247834733254;
      loss_4x4 += kMul2 * norm4 / norm2;
      static const double loss_4x4_limit0 = 1.0745851804785773;
      if (loss_4x4 >= loss_4x4_limit0) {
        // Probably not multi-threading safe.
        disable_dct32[(by & ~3) * xsize_blocks + (bx & ~3)] = true;
        disable_dct16[(by & ~1) * xsize_blocks + (bx & ~1)] = true;
      }
    }
  };
  const auto find_block_strategy = [&](int bx, int by) SIMD_ATTR {
    // The quantized symbol distribution contracts with the increasing
    // butteraugli_target.
    const float discretization_factor =
        100 * (6.9654004856811754) / butteraugli_target;
    // A value below 1.0 to favor 8x8s when all things are equal.
    // 16x16 has wider reach of oscillations and this part of the
    // computation is not aware of visual masking. Inhomogeneous
    // visual masking will propagate accuracy further with 16x16 than
    // with 8x8 dcts.
    const float kFavor8x8Dct = 0.97022476956774629;
    float kFavor8x8DctOver32x32 = 0.75;
    if (butteraugli_target >= 6.0) {
      kFavor8x8DctOver32x32 = 0.86;
    }
    static const double kColorWeights[3] = {
        0.55560083410056468,
        2.5,
        2.0,
    };
    // DCT4X4
    {
      double blockval[4];
      double blockval_2x2[16];
      // 4x4 DCT needs less focus on B channel, since at that resolution
      // blue needs to be correct only by average.
      static const double kColorWeights4x4[3] = {
          0.55560083410056468,
          1.0,
          0.3,
      };
      // DCT4X4 collection
      for (int ix = 0; ix < 4; ++ix) {
        double total_sum = 0;
        int offx = (ix & 1) * 4;
        int offy = (ix & 2) * 2;
        for (size_t c = 0; c < src.kNumPlanes; c++) {
          double sum = 0;
          for (size_t iy = 0; iy < 3; iy++) {
            const float* row0 =
                src.ConstPlaneRow(c, by * kBlockDim + offy + iy);
            const float* row1 =
                src.ConstPlaneRow(c, by * kBlockDim + offy + iy + 1);
            for (size_t dx = 0; dx < 3; dx++) {
              int x = bx * kBlockDim + offx + dx;
              sum += fabs(row0[x] - row0[x + 1]) + fabs(row0[x] - row1[x]);
            }
            {
              int x = bx * kBlockDim + offx + 3;
              sum += fabs(row0[x] - row1[x]);
            }
          }
          int iy = 3;
          const float* row0 = src.ConstPlaneRow(c, by * kBlockDim + offy + iy);
          for (size_t dx = 0; dx < 3; dx++) {
            int x = bx * kBlockDim + offx + dx;
            sum += fabs(row0[x] - row0[x + 1]);
          }
          total_sum += kColorWeights4x4[c] * sum;
        }
        blockval[ix] = total_sum;
      }
      // DCT2X2 collection
      for (int ix = 0; ix < 16; ++ix) {
        double total_sum = 0;
        int offx = (ix & 3) * 2;
        int offy = ((ix >> 2) & 3) * 2;
        for (size_t c = 0; c < src.kNumPlanes; c++) {
          double sum = 0;
          const float* row0 =
              src.ConstPlaneRow(c, by * kBlockDim + offy);
          const float* row1 =
              src.ConstPlaneRow(c, by * kBlockDim + offy + 1);
          int x = bx * kBlockDim + offx;
          sum += fabs(row0[x] - row0[x + 1]);
          sum += fabs(row0[x] - row1[x]);
          sum += fabs(row0[x + 1] - row1[x + 1]);
          sum += fabs(row1[x] - row1[x + 1]);
          total_sum += kColorWeights4x4[c] * sum;
        }
        blockval_2x2[ix] = total_sum;
      }
      double norm2 = 0.0;
      double norm4 = 0.0;
      double norm8 = 0.0;
      for (int ix = 0; ix < 4; ++ix) {
        double v = blockval[ix];
        v *= v;
        norm2 += v;
        v *= v;
        norm4 += v;
        v *= v;
        norm8 += v;
      }
      norm2 = std::pow(norm2 * (1.0 / 4), 0.5);
      norm4 = std::pow(norm4 * (1.0 / 4), 0.25);
      norm8 = std::pow(norm8 * (1.0 / 4), 0.125);
      norm2 += 0.03;

      double norm2_2x2 = 0.0;
      double norm4_2x2 = 0.0;
      double norm8_2x2 = 0.0;
      for (int ix = 0; ix < 16; ++ix) {
        double v = blockval_2x2[ix];
        v *= v;
        norm2_2x2 += v;
        v *= v;
        norm4_2x2 += v;
        v *= v;
        norm8_2x2 += v;
      }
      norm2_2x2 = std::pow(norm2_2x2 * (1.0 / 16), 0.5);
      norm4_2x2 = std::pow(norm4_2x2 * (1.0 / 16), 0.25);
      norm8_2x2 = std::pow(norm8_2x2 * (1.0 / 16), 0.125);
      norm2_2x2 += 0.019728218434017869;

      double kMul1 = 0.84759906694969511;
      double loss_4x4 = kMul1 * norm8 / norm2;
      double loss_2x2 = kMul1 * norm8_2x2 / norm2_2x2;
      double kMulCross = 0.25055664541015626;
      loss_2x2 *= 1.0 - kMulCross;
      loss_2x2 += kMulCross * loss_4x4;
      double kMul2 = -0.012183673789682012;
      loss_4x4 += kMul2 * norm4 / norm2;
      loss_2x2 += kMul2 * norm4_2x2 / norm2_2x2;
      static const double loss_4x4_limit0 = 1.0772933332748065;
      static const double loss_2x2_mul = 1.2160334689154628;
      if (loss_4x4 >= loss_4x4_limit0) {
        if (loss_2x2 >= loss_2x2_mul * loss_4x4) {
          return AcStrategy::Type::DCT2X2;
        }
        return AcStrategy::Type::DCT4X4;
      }
      static const double loss_2x2_limit = 1.3804543168741275;
      static const double loss_2x2_limit_4x4 = 0.65007565370307108;
      if (loss_2x2 >= loss_2x2_limit && loss_4x4 >= loss_2x2_limit_4x4) {
        return AcStrategy::Type::DCT2X2;
      }
    }
    const auto apply_identity =
        [](const double butteraugli_target, const Image3F* img,
           const ImageF* quant_field, size_t block_width, size_t block_height,
           int bx, int by) SIMD_ATTR {
          int non_zeros = 0;
          int zeros = 0;
          const double low_limit = 1.0 / butteraugli_target;
          const double high_limit = 5.0 / butteraugli_target;
          for (size_t c = 0; c < img->kNumPlanes; c++) {
            double ave[4] = {0};
            for (size_t iy = 0; iy < block_height; iy++) {
              const float* row = img->ConstPlaneRow(c, by * block_height + iy);
              for (size_t ix = 0; ix < block_width; ix++) {
                int aveix = ((iy >> 2) * 2 + (ix >> 2)) & 3;
                ave[aveix] += row[bx * block_width + ix];
              }
            }
            ave[0] *= 1.0 / 16;
            ave[1] *= 1.0 / 16;
            ave[2] *= 1.0 / 16;
            ave[3] *= 1.0 / 16;
            for (size_t iy = 0; iy < block_height; iy++) {
              const float* row = img->ConstPlaneRow(c, by * block_height + iy);
              for (size_t ix = 0; ix < block_width; ix++) {
                int aveix = ((iy >> 2) * 2 + (ix >> 2)) & 3;
                float val = fabs(row[bx * block_width + ix] - ave[aveix]);
                if (val < low_limit) {
                  zeros++;
                }
                if (c == 1 && val > high_limit) {
                  non_zeros++;
                }
              }
            }
          }
          if (non_zeros == 0) {
            return 0;
          }
          return zeros;
        };
    int identity_score = apply_identity(butteraugli_target, &src, quant_field,
                                        kBlockDim, kBlockDim, bx, by);
    if (identity_score > 3 * 49) {
      return AcStrategy::Type::IDENTITY;
    }
    const double kPow = 0.98633286110451568;
    const double kPow2 = 0.016160818829927922;
    // DCT32
    if (!disable_dct32[by * xsize_blocks + bx] && bx + 3 < xsize_blocks &&
        by + 3 < ysize_blocks && (bx & 3) == 0 && (by & 3) == 0) {
      static const double kDiff = 0.739609387964272;
      double dct8x8_entropy = 0;
      for (size_t c = 0; c < coeffs.kNumPlanes; c++) {
        double entropy = 0;
        for (size_t iy = 0; iy < 4; iy++) {
          const float* row = coeffs.ConstPlaneRow(c, by + iy);
          int bx_actual = bx;
          for (size_t ix = 1; ix < kBlockDim * kBlockDim * 4; ix++) {
            // Skip the dc values at 0 and 64.
            if ((ix & 63) == 0) {
              bx_actual++;
              continue;
            }
            float mul = 1.0f / DequantMatrix(0, kQuantKindDCT8, c)[ix & 63];
            float val =
                mul * row[bx * kBlockDim * kBlockDim + ix];
            val *= quant_field->ConstRow(by + iy)[bx_actual];
            float v = fabsf(val) * discretization_factor;
            entropy += 1 + kDiff - pow(kPow, v) - kDiff * pow(kPow2, v);
          }
        }
        dct8x8_entropy += kColorWeights[c] * entropy;
      }
      float quant_inhomogeneity = 0;
      float max_quant = -1e30;
      for (int dy = 0; dy < 4; ++dy) {
        for (int dx = 0; dx < 4; ++dx) {
          float quant = quant_field->ConstRow(by + dy)[bx + dx];
          max_quant = std::max(max_quant, quant);
          quant_inhomogeneity -= quant;
        }
      }
      quant_inhomogeneity += 16 * max_quant;
      double kMulInho = (-47.780 * (-4.4869535189137961  )) / butteraugli_target;
      dct8x8_entropy += kMulInho * quant_inhomogeneity;
      double dct32x32_entropy = 0;
      for (size_t c = 0; c < src.kNumPlanes; c++) {
        double entropy = 0;
        SIMD_ALIGN float dct32x32[16 * kBlockDim * kBlockDim] = {};
        AcStrategy(AcStrategy::Type::DCT32X32, 0)
            .TransformFromPixels(
                src.PlaneRow(c, kBlockDim * by) + kBlockDim * bx,
                src.PixelsPerRow(), dct32x32, 4 * 64);
        for (size_t k = 0; k < 16 * kBlockDim * kBlockDim; k++) {
          if (k % 64 == 0) {
            // Leave out the lowest frequencies.
            continue;
          }
          float mul = 1.0f / DequantMatrix(0, kQuantKindDCT32Start, c)[k];
          float val = mul * dct32x32[k];
          val *= max_quant;
          float v = fabsf(val) * discretization_factor;
          entropy += 1 + kDiff - pow(kPow, v) - kDiff * pow(kPow2, v);
        }
        dct32x32_entropy += kColorWeights[c] * entropy;
      }
      if (dct32x32_entropy < kFavor8x8DctOver32x32 * dct8x8_entropy) {
        return AcStrategy::Type::DCT32X32;
      }
    }

    // DCT16
    if (!disable_dct16[by * xsize_blocks + bx] && bx + 1 < xsize_blocks &&
        by + 1 < ysize_blocks && (bx & 1) == 0 && (by & 1) == 0) {
      static const double kDiff = 0.30935949753915776;
      double dct8x8_entropy = 0;
      for (size_t c = 0; c < coeffs.kNumPlanes; c++) {
        double entropy = 0;
        for (size_t iy = 0; iy < 2; iy++) {
          const float* row = coeffs.ConstPlaneRow(c, by + iy);
          int bx_actual = bx;
          for (size_t ix = 1; ix < kBlockDim * kBlockDim * 2; ix++) {
            // Skip the dc values at 0 and 64.
            if (ix == 64) {
              bx_actual++;
              continue;
            }
            float mul = 1.0f / DequantMatrix(0, kQuantKindDCT8, c)[ix & 63];
            float val =
                mul * row[bx * kBlockDim * kBlockDim + ix];
            val *= quant_field->ConstRow(by + iy)[bx_actual];
            float v = fabsf(val) * discretization_factor;
            entropy += 1 + kDiff - pow(kPow, v) - kDiff * pow(kPow2, v);
          }
        }
        dct8x8_entropy += kColorWeights[c] * entropy;
      }
      float max_quant = std::max<double>(
          std::max<double>(quant_field->ConstRow(by)[bx],
                           quant_field->ConstRow(by)[bx + 1]),
          std::max<double>(quant_field->ConstRow(by + 1)[bx],
                           quant_field->ConstRow(by + 1)[bx + 1]));
      float quant_inhomogeneity =
          4 * max_quant -
          (quant_field->ConstRow(by)[bx] + quant_field->ConstRow(by)[bx + 1] +
           quant_field->ConstRow(by + 1)[bx] +
           quant_field->ConstRow(by + 1)[bx + 1]);
      double kMulInho = (-47.780 * (4.0553021854221809  )) / butteraugli_target;
      dct8x8_entropy += kMulInho * quant_inhomogeneity;
      double dct16x16_entropy = 0;
      for (size_t c = 0; c < src.kNumPlanes; c++) {
        double entropy = 0;
        SIMD_ALIGN float dct16x16[4 * kBlockDim * kBlockDim] = {};
        AcStrategy(AcStrategy::Type::DCT16X16, 0)
            .TransformFromPixels(
                src.PlaneRow(c, kBlockDim * by) + kBlockDim * bx,
                src.PixelsPerRow(), dct16x16, 2 * 64);
        for (size_t k = 0; k < 4 * kBlockDim * kBlockDim; k++) {
          if (k % 64 == 0) {
            // Leave out the lowest frequencies.
            continue;
          }
          float mul = 1.0f / DequantMatrix(0, kQuantKindDCT16Start, c)[k];
          float val = mul * dct16x16[k];
          val *= max_quant;
          float v = fabsf(val) * discretization_factor;
          entropy += 1 + kDiff - pow(kPow, v) - kDiff * pow(kPow2, v);
        }
        dct16x16_entropy += kColorWeights[c] * entropy;
      }
      if (dct16x16_entropy < kFavor8x8Dct * dct8x8_entropy) {
        return AcStrategy::Type::DCT16X16;
      }
    }
    return AcStrategy::Type::DCT;
  };
  ImageB raw_ac_strategy(xsize_blocks, ysize_blocks);
  RunOnPool(pool, 0, ysize_blocks, [&](int y, int _) {
    for (size_t x = 0; x < xsize_blocks; x++) {
      disable_large_transforms(x, y);
    }
  });
  RunOnPool(pool, 0, ysize_blocks, [&](int y, int _) {
    uint8_t* PIK_RESTRICT row = raw_ac_strategy.Row(y);
    for (size_t x = 0; x < xsize_blocks; x++) {
      row[x] = static_cast<uint8_t>(find_block_strategy(x, y));
    }
  });
  ac_strategy->SetFromRaw(Rect(raw_ac_strategy), raw_ac_strategy);
  if (aux_out != nullptr) {
    aux_out->num_dct2_blocks =
        ac_strategy->CountBlocks(AcStrategy::Type::DCT2X2);
    aux_out->num_dct4_blocks =
        ac_strategy->CountBlocks(AcStrategy::Type::DCT4X4);
    aux_out->num_dct16_blocks =
        ac_strategy->CountBlocks(AcStrategy::Type::DCT16X16);
    aux_out->num_dct32_blocks =
        ac_strategy->CountBlocks(AcStrategy::Type::DCT32X32);
  }
  if (ac_strategy->CountBlocks(AcStrategy::Type::DCT) ==
      xsize_blocks * ysize_blocks) {
    *ac_strategy = AcStrategyImage(xsize_blocks, ysize_blocks);
  }
  if (WantDebugOutput(aux_out)) {
    aux_out->DumpImage("ac_strategy_type", ac_strategy->ConstRaw());
  }
}

}  // namespace pik
