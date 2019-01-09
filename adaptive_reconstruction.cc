// Copyright 2018 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "adaptive_reconstruction.h"
#include <cstdint>
#include <cstring>

#include "ac_strategy.h"
#include "block.h"
#include "common.h"
#include "dct.h"
#include "dct_util.h"
#include "entropy_coder.h"
#include "epf.h"
#include "profiler.h"
#include "quantizer.h"
#include "simd/simd.h"

#ifndef PIK_AR_PRINT_STATS
#define PIK_AR_PRINT_STATS 0
#endif

namespace pik {
namespace {

Image3F DoDenoise(const Image3F& opsin, const Image3F& opsin_sharp,
                  const Quantizer& quantizer, const ImageI& raw_quant_field,
                  const AcStrategyImage& ac_strategy,
                  const EpfParams& epf_params, AdaptiveReconstructionAux* aux) {
  if (aux != nullptr) {
    aux->quant_scale = quantizer.Scale();
  }

  Image3F smoothed(opsin.xsize(), opsin.ysize());

  const float quant_scale = quantizer.Scale();
  if (epf_params.enable_adaptive) {
    Dispatch(TargetBitfield().Best(), EdgePreservingFilter(), opsin,
             opsin_sharp, &raw_quant_field, quant_scale, ac_strategy,
             epf_params, &smoothed, aux ? &aux->epf_stats : nullptr);
  } else {
    float stretch;
    Dispatch(TargetBitfield().Best(), EdgePreservingFilter(), opsin,
             opsin_sharp, epf_params, aux ? &aux->stretch : &stretch,
             &smoothed);
  }
  return smoothed;
}

using DF = SIMD_FULL(float);
using DI = SIMD_FULL(int32_t);
using DU = SIMD_FULL(uint32_t);
using VF = DF::V;
using VI = DI::V;
using VU = DU::V;

struct AdaptiveReconstructionStats {
  VU low_clamp_count;
  VU high_clamp_count;
  SIMD_ATTR AdaptiveReconstructionStats() {
    low_clamp_count = setzero(DU());
    high_clamp_count = setzero(DU());
  }
  SIMD_ATTR void Assimilate(const AdaptiveReconstructionStats& other) {
    high_clamp_count += other.high_clamp_count;
    low_clamp_count += other.low_clamp_count;
  }
};

// Clamp the difference between the coefficients of the filtered image and the
// coefficients of the original image (i.e. `correction`) to an interval whose
// size depends on the values in the non-smoothed image. The interval is
// scaled according to `interval_scale`.
// TODO(user): cleanup by removing `bias`, and all the related methods
// everywhere.
SIMD_ATTR PIK_INLINE void AsymmetricClamp(
    const VF interval_scale, const VF bias, const VF values, const float asym,
    float* PIK_RESTRICT block, float* PIK_RESTRICT min_ratio,
    float* PIK_RESTRICT lower_bounds, float* PIK_RESTRICT upper_bounds,
    AdaptiveReconstructionStats* PIK_RESTRICT stats) {
  const DF df;
  const auto half = set1(df, 0.5f);
  const auto small = set1(df, asym);
  const auto is_neg = values < set1(df, -0.5f) * interval_scale;
  const auto upper_bound = select(half, small, is_neg) * interval_scale;

  const auto is_pos = values > half * interval_scale;
  const auto neghalf = set1(df, -0.5f);
  const auto negsmall = set1(df, -asym);
  const auto lower_bound = select(neghalf, negsmall, is_pos) * interval_scale;

  const auto correction = load(df, block);
  const auto lo = lower_bound;
  const auto hi = upper_bound;
  // Store for later clamping. The mul-by-min_ratio cannot be used in some
  // cases (zero or sign changes).
  store(lo, df, lower_bounds);
  store(hi, df, upper_bounds);
  const auto clamped = min(max(lo, correction), hi);
  const auto prev_min_ratio = load(df, min_ratio);

  const SIMD_FULL(uint32_t) du;
  const auto zero = setzero(du);  // faster than comparing as float.
  // Can't divide by correction=0, and clamped=0 results in ratio=0.
  // TODO(janwas): confirm + take advantage of: either both, or neither are 0.
  const auto all_zero =
      (cast_to(du, correction) == zero) | (cast_to(du, clamped) == zero);
  // If sign(clamped) != sign(correction), we get a negative ratio.
  const auto sign = cast_to(df, set1(du, 0x80000000u));
  const auto changed_sign = (clamped ^ correction) & sign;
  const auto bad = condition_from_sign(cast_to(df, all_zero) | changed_sign);
  const auto clamp_ratio = select(clamped / correction, prev_min_ratio, bad);
  store(min(prev_min_ratio, clamp_ratio), df, min_ratio);

#if PIK_AR_PRINT_STATS
  const DU du;
  const auto one = set1(du, uint32_t(1));
  const auto is_low = correction < clamped;
  stats->low_clamp_count += cast_to(du, is_low) & one;
  const auto is_high = correction > clamped;
  stats->high_clamp_count += cast_to(du, is_high) & one;
#endif
}

// Clamps a block of the filtered image, pointed to by `opsin`, ensuring that it
// does not get too far away from the values in the corresponding block of the
// original image, pointed to by `original`. Instead of computing the difference
// of the DCT of the two images, we compute the DCT of the difference as DCT is
// a linear operator and this saves some work. `biases` contains the
// dequantization bias that was used for the current block.
SIMD_ATTR PIK_INLINE void UpdateMinRatioOfClampToOriginalDCT(
    const float* PIK_RESTRICT original, size_t stride,
    const float* PIK_RESTRICT biases, const size_t biases_stride,
    const float* PIK_RESTRICT dequant_matrix, const float inv_quant_ac,
    const float dc_mul, AcStrategy acs, const float* PIK_RESTRICT filt,
    float* PIK_RESTRICT min_ratio, float* PIK_RESTRICT lower_bounds,
    float* PIK_RESTRICT upper_bounds, float* PIK_RESTRICT block,
    AdaptiveReconstructionStats* PIK_RESTRICT stats) {
  const SIMD_FULL(float) df;
  const SIMD_PART(float, 1) d1;
  const SIMD_FULL(uint32_t) du;

  const size_t block_width = kBlockDim * acs.covered_blocks_x();
  const size_t block_height = kBlockDim * acs.covered_blocks_y();

  for (size_t iy = 0; iy < block_height; iy++) {
    for (size_t ix = 0; ix < block_width; ix += df.N) {
      const auto filt_v = load(df, filt + stride * iy + ix);
      const auto original_v = load(df, original + stride * iy + ix);
      store(filt_v - original_v, df, block + block_width * iy + ix);
    }
  }

  // Every "block row" contains covered_blocks_x blocks - thus, the next block
  // row starts after covered_block_x times the number of coefficients per
  // block floats.
  acs.TransformFromPixels(block, block_width, block,
                          acs.covered_blocks_x() * kBlockDim * kBlockDim);

  SIMD_ALIGN const constexpr uint32_t only_lane0_bits[df.N] = {~0u};
  const auto only_lane0 = cast_to(df, load(du, only_lane0_bits));

  // The higher the kAsymClampLimit, the more blurred image.
  const float kAsymClampLimit = 0.18644350819576438;
  float asym = kAsymClampLimit;
  SIMD_ALIGN float block2[AcStrategy::kMaxCoeffArea];
  {
    const SIMD_FULL(float) df;
    // Unnecessary to compute them here as we already know these values,
    // as they are available for ac in the decoded stream.
    // But for simplicity of getting this done, I'm recomputing for now.
    acs.TransformFromPixels(original, stride, block2, block_width * kBlockDim);

    auto sums = setzero(df);
    for (size_t by = 0; by < acs.covered_blocks_y(); by++) {
      for (size_t bx = 0; bx < acs.covered_blocks_x(); bx++) {
        const size_t block_offset =
            (by * acs.covered_blocks_x() + bx) * kBlockDim * kBlockDim;
        // First iteration: skip lowest-frequency coefficient.
        const auto interval_scale =
            set1(df, inv_quant_ac) * load(df, dequant_matrix + block_offset);
        const auto v = andnot(only_lane0, load(df, block2)) / interval_scale;
        sums += v * v;

        for (size_t k = df.N; k < kBlockDim * kBlockDim; k += df.N) {
          const auto interval_scale =
              set1(df, inv_quant_ac) *
              load(df, dequant_matrix + block_offset + k);
          const auto v = load(df, block2 + block_offset + k) / interval_scale;
          sums += v * v;
        }
      }
    }
    float sum = get_part(d1, ext::sum_of_lanes(sums));
    sum *= acs.InverseNumACCoefficients();
    sum = std::sqrt(sum);
    sum *= 0.054307033953246348f;
    sum += 0.044037713396095239f;
    // Smaller sum -> sharper image.
    asym = std::min<float>(sum, asym);
  }

  // TODO(janwas): template, make covered_blocks* constants
  for (size_t by = 0; by < acs.covered_blocks_y(); by++) {
    for (size_t bx = 0; bx < acs.covered_blocks_x(); bx++) {
      const size_t block_offset =
          (by * acs.covered_blocks_x() + bx) * kBlockDim * kBlockDim;
      const size_t biases_offset =
          by * biases_stride + bx * kBlockDim * kBlockDim;
      // First iteration: skip lowest-frequency coefficient.
      const auto bias = load(df, biases + biases_offset);
      const auto values = load(df, block2 + block_offset);
      const auto ac_mul =
          load(df, dequant_matrix + block_offset) * set1(df, inv_quant_ac);
      const float cur_dc_mul = acs.ARLowestFrequencyScale(bx, by) * dc_mul;
      const auto interval_scale =
          select(ac_mul, set1(df, cur_dc_mul), only_lane0);
      AsymmetricClamp(interval_scale, bias, values, asym, block + block_offset,
                      min_ratio + block_offset, lower_bounds + block_offset,
                      upper_bounds + block_offset, stats);

      for (size_t k = df.N; k < kBlockDim * kBlockDim; k += df.N) {
        const size_t ofs = block_offset + k;
        const auto bias = load(df, biases + biases_offset + k);
        const auto values = load(df, block2 + ofs);
        const auto interval_scale =
            load(df, dequant_matrix + ofs) * set1(df, inv_quant_ac);
        AsymmetricClamp(interval_scale, bias, values, asym, block + ofs,
                        min_ratio + ofs, lower_bounds + ofs, upper_bounds + ofs,
                        stats);
      }
    }
  }
}

// Clamp by multiplying block[k] by min_ratio[k], then IDCT.
// DoMul allows disabling the scaling for X as an experiment (disabled).
template <bool DoMul>
SIMD_ATTR PIK_INLINE void ClampAndIDCT(
    float* PIK_RESTRICT block, const size_t block_width,
    const size_t block_height, const float* PIK_RESTRICT min_ratio,
    const float* PIK_RESTRICT lower_bounds,
    const float* PIK_RESTRICT upper_bounds, const AcStrategy acs,
    const float* PIK_RESTRICT original, float* PIK_RESTRICT filt,
    size_t stride) {
#ifdef ADDRESS_SANITIZER
  for (size_t k = 0; k < AcStrategy::kMaxCoeffArea; ++k) {
    PIK_ASSERT(min_ratio[k] > 0.0f);
  }
#endif

  const SIMD_FULL(float) df;
  for (size_t k = 0; k < AcStrategy::kMaxCoeffArea; k += df.N) {
    const auto mul = DoMul ? load(df, min_ratio + k) : set1(df, 1.0f);
    const auto scaled = load(df, block + k) * mul;
    const auto clamped = min(max(load(df, lower_bounds + k), scaled),
                             load(df, upper_bounds + k));
    store(clamped, df, block + k);
  }

  // IDCT
  acs.TransformToPixels(block, block_width * kBlockDim, block, block_width);

  for (size_t iy = 0; iy < block_height; iy++) {
    for (size_t ix = 0; ix < block_width; ix += df.N) {
      const auto block_v = load(df, block + block_width * iy + ix);
      const auto original_v = load(df, original + stride * iy + ix);
      store(block_v + original_v, df, filt + stride * iy + ix);
    }
  }
}

void ComputeResidualSlow(const Image3F& in, const Image3F& smoothed,
                         Image3F* PIK_RESTRICT residual) {
  for (int c = 0; c < in.kNumPlanes; ++c) {
    for (size_t y = 0; y < in.ysize(); ++y) {
      const float* row_in = in.PlaneRow(c, y);
      const float* row_smoothed = smoothed.PlaneRow(c, y);
      float* PIK_RESTRICT row_out = residual->PlaneRow(c, y);
      for (size_t x = 0; x < in.xsize(); ++x) {
        row_out[x] = std::abs(row_in[x] - row_smoothed[x]);
      }
    }
  }
}

}  // namespace

SIMD_ATTR Image3F AdaptiveReconstruction(
    Image3F* in, const Image3F& non_smoothed, const Quantizer& quantizer,
    const ImageI& raw_quant_field, const AcStrategyImage& ac_strategy,
    const Image3F& biases, const EpfParams& epf_params,
    AdaptiveReconstructionAux* aux) {
  PROFILER_FUNC;
  // Input image should have an integer number of blocks.
  PIK_ASSERT(in->xsize() % kBlockDim == 0 && in->ysize() % kBlockDim == 0);
  const size_t xsize_blocks = in->xsize() / kBlockDim;
  const size_t ysize_blocks = in->ysize() / kBlockDim;

  for (size_t c = 0; c < 3; c++) {
    size_t base = DequantMatrixOffset(0, kQuantKindDCT16Start, c);
    (void)base;  // Unused if asserts disabled.
    PIK_ASSERT(DequantMatrixOffset(0, kQuantKindDCT16Start + 1, c) == base + 1);
    PIK_ASSERT(DequantMatrixOffset(0, kQuantKindDCT16Start + 2, c) == base + 2);
    PIK_ASSERT(DequantMatrixOffset(0, kQuantKindDCT16Start + 3, c) == base + 3);
  }

  // Modified below (clamped).
  Image3F filt = DoDenoise(*in, non_smoothed, quantizer, raw_quant_field,
                           ac_strategy, epf_params, aux);

  const size_t stride = filt.PlaneRow(0, 1) - filt.PlaneRow(0, 0);
  const size_t biases_stride = biases.PixelsPerRow();
  PIK_ASSERT(stride == in->PlaneRow(0, 1) - in->PlaneRow(0, 0));

  AdaptiveReconstructionStats stats;

  // Dequantization matrices.
  const float* PIK_RESTRICT dequant_matrices =
      quantizer.DequantMatrix(0, kQuantKindDCT8);
  float dc_mul[3];
  for (size_t c = 0; c < 3; c++) {
    dc_mul[c] = quantizer.inv_quant_dc() *
                dequant_matrices[DequantMatrixOffset(0, kQuantKindDCT8, c) *
                                 kDCTBlockSize];
  }

  for (size_t by = 0; by < ysize_blocks; ++by) {
    const int32_t* PIK_RESTRICT row_quant = raw_quant_field.ConstRow(by);
    const AcStrategyRow ac_strategy_row = ac_strategy.ConstRow(by);

    const float* PIK_RESTRICT row_original_x =
        non_smoothed.ConstPlaneRow(0, by * kBlockDim);
    const float* PIK_RESTRICT row_original_y =
        non_smoothed.ConstPlaneRow(1, by * kBlockDim);
    const float* PIK_RESTRICT row_original_b =
        non_smoothed.ConstPlaneRow(2, by * kBlockDim);

    const float* PIK_RESTRICT row_biases_x = biases.ConstPlaneRow(0, by);
    const float* PIK_RESTRICT row_biases_y = biases.ConstPlaneRow(1, by);
    const float* PIK_RESTRICT row_biases_b = biases.ConstPlaneRow(2, by);

    float* PIK_RESTRICT row_filt_x = filt.PlaneRow(0, by * kBlockDim);
    float* PIK_RESTRICT row_filt_y = filt.PlaneRow(1, by * kBlockDim);
    float* PIK_RESTRICT row_filt_b = filt.PlaneRow(2, by * kBlockDim);

    for (size_t bx = 0; bx < xsize_blocks; ++bx) {
      const int32_t qac = row_quant[bx];
      const float inv_quant_ac = quantizer.inv_quant_ac(qac);
      const AcStrategy acs = ac_strategy_row[bx];
      if (!acs.IsFirstBlock()) continue;

      // TODO(janwas): hoist/precompute
      const float* dequant_matrix_x =
          dequant_matrices +
          kDCTBlockSize * DequantMatrixOffset(0, acs.GetQuantKind(), /*c=*/0);
      const float* dequant_matrix_y =
          dequant_matrices +
          kDCTBlockSize * DequantMatrixOffset(0, acs.GetQuantKind(), /*c=*/1);
      const float* dequant_matrix_b =
          dequant_matrices +
          kDCTBlockSize * DequantMatrixOffset(0, acs.GetQuantKind(), /*c=*/2);

      const size_t block_ofs = bx * kBlockDim;
      const float* PIK_RESTRICT pos_original_x = row_original_x + block_ofs;
      const float* PIK_RESTRICT pos_original_y = row_original_y + block_ofs;
      const float* PIK_RESTRICT pos_original_b = row_original_b + block_ofs;
      float* PIK_RESTRICT pos_filt_x = row_filt_x + block_ofs;
      float* PIK_RESTRICT pos_filt_y = row_filt_y + block_ofs;
      float* PIK_RESTRICT pos_filt_b = row_filt_b + block_ofs;

      SIMD_ALIGN float min_ratio[AcStrategy::kMaxCoeffArea];
      const SIMD_FULL(float) df;
      for (size_t k = 0; k < AcStrategy::kMaxCoeffArea; k += df.N) {
        store(set1(df, 1.0f), df, min_ratio + k);
      }

      SIMD_ALIGN float block_x[AcStrategy::kMaxCoeffArea];
      SIMD_ALIGN float block_y[AcStrategy::kMaxCoeffArea];
      SIMD_ALIGN float block_b[AcStrategy::kMaxCoeffArea];
      SIMD_ALIGN float lo_x[AcStrategy::kMaxCoeffArea];
      SIMD_ALIGN float hi_x[AcStrategy::kMaxCoeffArea];
      SIMD_ALIGN float lo_y[AcStrategy::kMaxCoeffArea];
      SIMD_ALIGN float hi_y[AcStrategy::kMaxCoeffArea];
      SIMD_ALIGN float lo_b[AcStrategy::kMaxCoeffArea];
      SIMD_ALIGN float hi_b[AcStrategy::kMaxCoeffArea];

      UpdateMinRatioOfClampToOriginalDCT(
          pos_original_x, stride, row_biases_x + bx * kDCTBlockSize,
          biases_stride, dequant_matrix_x, inv_quant_ac, dc_mul[0], acs,
          pos_filt_x, min_ratio, lo_x, hi_x, block_x, &stats);
      UpdateMinRatioOfClampToOriginalDCT(
          pos_original_y, stride, row_biases_y + bx * kDCTBlockSize,
          biases_stride, dequant_matrix_y, inv_quant_ac, dc_mul[1], acs,
          pos_filt_y, min_ratio, lo_y, hi_y, block_y, &stats);
      UpdateMinRatioOfClampToOriginalDCT(
          pos_original_b, stride, row_biases_b + bx * kDCTBlockSize,
          biases_stride, dequant_matrix_b, inv_quant_ac, dc_mul[2], acs,
          pos_filt_b, min_ratio, lo_b, hi_b, block_b, &stats);

      const size_t block_width = kBlockDim * acs.covered_blocks_x();
      const size_t block_height = kBlockDim * acs.covered_blocks_y();
      ClampAndIDCT<true>(block_x, block_width, block_height, min_ratio, lo_x,
                         hi_x, acs, pos_original_x, pos_filt_x, stride);
      ClampAndIDCT<true>(block_y, block_width, block_height, min_ratio, lo_y,
                         hi_y, acs, pos_original_y, pos_filt_y, stride);
      ClampAndIDCT<true>(block_b, block_width, block_height, min_ratio, lo_b,
                         hi_b, acs, pos_original_b, pos_filt_b, stride);
    }  // bx
  }    // by

#if PIK_AR_PRINT_STATS
  fprintf(
      stderr,
      "Number of low-clamped, high-clamped, and total values: %8u %8u %8lu\n",
      get_part(DU(), ext::sum_of_lanes(stats.low_clamp_count)),
      get_part(DU(), ext::sum_of_lanes(stats.high_clamp_count)),
      in->xsize() * in->ysize());
#endif

  if (aux != nullptr) {
    if (aux->residual != nullptr) {
      ComputeResidualSlow(*in, filt, aux->residual);
    }
    if (aux->ac_quant != nullptr) {
      CopyImageTo(quantizer.RawQuantField(), aux->ac_quant);
    }
    if (aux->ac_quant != nullptr) {
      CopyImageTo(ac_strategy.ConstRaw(), aux->ac_strategy);
    }
  }
  return filt;
}

}  // namespace pik
