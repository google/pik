// Copyright 2018 Google Inc. All Rights Reserved.
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

#include "adaptive_reconstruction.h"

#include "af_edge_preserving_filter.h"
#include "block.h"
#include "common.h"
#include "context.h"  // kDCTBlockSize
#include "dct.h"
#include "simd/simd.h"

namespace pik {
namespace {

const float kEpfMulScale = 10000.0f;
// Multiplier for filter parameter (/kEpfMulScale)
const int32_t FLAGS_epf_mul = 256;

Image3F DoDenoise(const Quantizer& quantizer, ThreadPool* pool,
                  const Image3F& opsin) {
  const float scale = quantizer.Scale() * kEpfMulScale;
  const float sigma_mul =
      scale / (FLAGS_epf_mul << EdgePreservingFilter::kSigmaShift);
  Image3F smoothed(opsin.xsize(), opsin.ysize());
  Dispatch(TargetBitfield().Best(), EdgePreservingFilter(), opsin,
           &quantizer.RawQuantField(), sigma_mul, pool, &smoothed);
  return smoothed;
}

using DF = SIMD_FULL(float);
using DI = SIMD_FULL(int32_t);
using VF = DF::V;
using VI = DI::V;

// Clamp around the original coefficients: less for zeros,
// asymmetric clamping for non-zeros.
SIMD_ATTR PIK_INLINE VF AsymmetricClamp(const VF original, const VF smoothed) {
  const DF df;
  const auto half = set1(df, 0.5f);
  const auto small = set1(df, 0.2f);
  const auto is_neg = original < set1(df, -0.5f);
  const auto is_pos = original > half;
  const auto add = select(half, small, is_neg);
  const auto sub = select(half, small, is_pos);
  return min(max(original - sub, smoothed), original + add);
}

SIMD_ATTR VF YtoX(VF y, VI x_factor) {
  const auto scale = set1(DF(), kColorScaleX);
  const auto biased = convert_to(DF(), x_factor - set1(DI(), kColorOffsetX));
  return y * biased * scale;
}
SIMD_ATTR VF YtoB(VF y, VI b_factor) {
  const auto scale = set1(DF(), kColorScaleB);
  const auto biased = convert_to(DF(), b_factor - set1(DI(), kColorOffsetB));
  return y * biased * scale;
}
SIMD_ATTR VF YtoTag(TagX, VF y, VI factor) { return YtoX(y, factor); }
SIMD_ATTR VF YtoTag(TagB, VF y, VI factor) { return YtoB(y, factor); }

// quant_matrix[] = 1/dequant_matrix[]. quant_ac = raw*scale.
SIMD_ATTR PIK_INLINE void ScaledBlockDCT_Y(
    const float* PIK_RESTRICT opsin, size_t stride,
    const float* const PIK_RESTRICT quant_matrix, const float quant_ac,
    const float inv_dc_mul, float* PIK_RESTRICT block,
    float* PIK_RESTRICT block_unscaled) {
  ComputeTransposedScaledDCT<kBlockDim>(
      FromLines(opsin, stride), ScaleToBlock<kBlockDim>(block_unscaled));

  const SIMD_FULL(float) df;
  const SIMD_FULL(uint32_t) du;
  const auto qac = set1(df, quant_ac);

  // Merge DC into first vector - important because caller will load a vector
  // from "block", which leads to a stall if we write block[0] here.
  const auto unscaled = load(df, block_unscaled);
  const auto scaled_dc = unscaled * set1(df, inv_dc_mul);
  const auto scaled = unscaled * load(df, quant_matrix) * qac;
  SIMD_ALIGN const uint32_t mask_lanes[df.N] = {~0u};
  const auto mask_dc = cast_to(df, load(du, mask_lanes));
  const auto merged = select(scaled, scaled_dc, mask_dc);
  store(merged, df, block);

  for (size_t k = df.N; k < kDCTBlockSize; k += df.N) {
    const auto unscaled = load(df, block_unscaled + k);
    const auto scaled = unscaled * load(df, quant_matrix + k) * qac;
    store(scaled, df, block + k);
  }
}

// y_block_unscaled is a copy from BEFORE calling ScaledBlockDCT_Y.
template <class TagXB>
SIMD_ATTR PIK_INLINE void ScaledBlockDCT_XB(
    const float* PIK_RESTRICT opsin, size_t stride,
    const float* PIK_RESTRICT y_block_unscaled, const int32_t correlation,
    const float dc_color_factor, const float* const PIK_RESTRICT quant_matrix,
    const float quant_ac, const float inv_dc_mul, float* PIK_RESTRICT block) {
  ComputeTransposedScaledDCT<kBlockDim>(FromLines(opsin, stride),
                                        ScaleToBlock<kBlockDim>(block));
  const SIMD_FULL(float) df;
  const SIMD_FULL(int32_t) di;
  const auto qac = set1(df, quant_ac);
  const auto cmap = set1(di, correlation);

  auto scaled = load(df, block);
  const auto y_unscaled = load(df, y_block_unscaled);
  auto scaled_dc = nmul_add(set1(df, dc_color_factor), y_unscaled, scaled);
  scaled_dc *= set1(df, inv_dc_mul);
  scaled -= YtoTag(TagXB(), y_unscaled, cmap);
  scaled *= load(df, quant_matrix) * qac;

  // Merge DC into first vector - important because caller will load a vector
  // from "block", which leads to a stall if we write block[0] here.
  SIMD_ALIGN const int32_t mask_lanes[df.N] = {-1};
  const auto mask_dc = cast_to(df, load(di, mask_lanes));
  const auto merged = select(scaled, scaled_dc, mask_dc);
  store(merged, df, block);

  for (size_t k = df.N; k < kDCTBlockSize; k += df.N) {
    auto scaled = load(df, block + k);
    const auto y_unscaled = load(df, y_block_unscaled + k);
    scaled -= YtoTag(TagXB(), y_unscaled, cmap);
    scaled *= load(df, quant_matrix + k) * qac;
    store(scaled, df, block + k);
  }
}

SIMD_ATTR PIK_INLINE void ClampBlockToOriginalDCT_Y(
    const float* PIK_RESTRICT original, size_t stride,
    const float* PIK_RESTRICT dequant_matrix,
    const float* PIK_RESTRICT quant_matrix, const float quant_ac,
    const float inv_quant_ac, const float dc_mul, const float inv_dc_mul,
    float* PIK_RESTRICT opsin, float* PIK_RESTRICT original_block_unscaled,
    float* PIK_RESTRICT block, float* PIK_RESTRICT y_block_unscaled) {
  SIMD_ALIGN float original_block[kDCTBlockSize];
  ScaledBlockDCT_Y(original, stride, quant_matrix, quant_ac, inv_dc_mul,
                   original_block, original_block_unscaled);

  ScaledBlockDCT_Y(opsin, stride, quant_matrix, quant_ac, inv_dc_mul, block,
                   y_block_unscaled);

  const SIMD_FULL(float) df;
  for (size_t k = 0; k < kDCTBlockSize; k += df.N) {
    const auto original = load(df, original_block + k);
    const auto smoothed = load(df, block + k);
    auto clamped = AsymmetricClamp(original, smoothed);

    // Scale back to original range.
    clamped *= load(df, dequant_matrix + k) * set1(df, inv_quant_ac);
    store(clamped, df, block + k);
  }

  block[0] = original_block[0] * dc_mul;

  // IDCT
  ComputeTransposedScaledIDCT<kBlockDim>(FromBlock<kBlockDim>(block),
                                         ToLines(opsin, stride));
}

// y_block[_unscaled] are for ColorCorrelationMap.
template <class TagXB>
SIMD_ATTR PIK_INLINE void ClampBlockToOriginalDCT_XB(
    const float* PIK_RESTRICT original, size_t stride,
    const float* PIK_RESTRICT original_y_block_unscaled,
    const float* PIK_RESTRICT y_block,
    const float* PIK_RESTRICT y_block_unscaled, const int32_t cmap,
    const float dc_color_factor, const float* PIK_RESTRICT dequant_matrix,
    const float* PIK_RESTRICT quant_matrix, const float quant_ac,
    const float inv_quant_ac, const float dc_mul, const float inv_dc_mul,
    float* PIK_RESTRICT opsin) {
  SIMD_ALIGN float original_block[kDCTBlockSize];
  ScaledBlockDCT_XB<TagXB>(original, stride, original_y_block_unscaled, cmap,
                           dc_color_factor, quant_matrix, quant_ac, inv_dc_mul,
                           original_block);

  SIMD_ALIGN float block[kDCTBlockSize];
  ScaledBlockDCT_XB<TagXB>(opsin, stride, y_block_unscaled, cmap,
                           dc_color_factor, quant_matrix, quant_ac, inv_dc_mul,
                           block);

  const SIMD_FULL(float) df;
  const SIMD_FULL(int32_t) di;
  for (size_t k = 0; k < kDCTBlockSize; k += df.N) {
    const auto original = load(df, original_block + k);
    const auto smoothed = load(df, block + k);
    auto clamped = AsymmetricClamp(original, smoothed);

    // Scale back to original range.
    clamped *= load(df, dequant_matrix + k) * set1(df, inv_quant_ac);
    clamped += YtoTag(TagXB(), load(df, y_block + k), set1(di, cmap));
    store(clamped, df, block + k);
  }

  block[0] = original_block[0] * dc_mul + dc_color_factor * y_block[0];

  // IDCT
  ComputeTransposedScaledIDCT<kBlockDim>(FromBlock<kBlockDim>(block),
                                         ToLines(opsin, stride));
}

}  // namespace

SIMD_ATTR Image3F AdaptiveReconstruction(const Quantizer& quantizer,
                                         const ColorCorrelationMap& cmap,
                                         ThreadPool* pool, const Image3F& in,
                                         const ImageB& ac_strategy,
                                         OpsinIntraTransform* transform) {
  const size_t xsize_blocks = in.xsize() / kBlockDim;
  const size_t ysize_blocks = in.ysize() / kBlockDim;

  Rect full(in);
  const Image3F& opsin = transform->IntraToOpsin(in, full, pool);
  // Modified below (clamped).
  Image3F filt = DoDenoise(quantizer, pool, opsin);
  transform->OpsinToIntraInPlace(&filt, full, pool);

  const size_t stride = filt.PlaneRow(0, 1) - filt.PlaneRow(0, 0);
  PIK_ASSERT(stride == in.PlaneRow(0, 1) - in.PlaneRow(0, 0));
  const float dc_color_factor_x = ColorCorrelationMap::YtoX(1, cmap.ytox_dc);
  const float dc_color_factor_b = ColorCorrelationMap::YtoB(1, cmap.ytob_dc);

  const float* PIK_RESTRICT dequant_matrices =
      quantizer.DequantMatrix(0, kQuantKindDCT8);

  SIMD_ALIGN float quant_matrices[kDCTBlockSize * 3 * kNumQuantKinds];
  for (size_t i = 0; i < kDCTBlockSize * 3 * kNumQuantKinds; ++i) {
    quant_matrices[i] = 1.0f / dequant_matrices[i];
  }

  // DC quantization always uses DCT8 values.
  const float dc_mul_x = dequant_matrices[0] * quantizer.inv_quant_dc();
  const float dc_mul_y =
      dequant_matrices[kDCTBlockSize] * quantizer.inv_quant_dc();
  const float dc_mul_b =
      dequant_matrices[2 * kDCTBlockSize] * quantizer.inv_quant_dc();
  const float inv_dc_mul_x = 1.0f / dc_mul_x;
  const float inv_dc_mul_y = 1.0f / dc_mul_y;
  const float inv_dc_mul_b = 1.0f / dc_mul_b;

  pool->Run(0, ysize_blocks, [&](const int task, const int thread) {
    const size_t by = task;

    SIMD_ALIGN float original_y_block_unscaled[kDCTBlockSize];
    SIMD_ALIGN float y_block[kDCTBlockSize];
    // Copy before ClampBlockToOriginalDCT_Y rescales y_block.
    SIMD_ALIGN float y_block_unscaled[kDCTBlockSize];

    const int32_t* PIK_RESTRICT row_cmap_x =
        cmap.ytox_map.ConstRow(by / kColorTileDimInBlocks);
    const int32_t* PIK_RESTRICT row_cmap_b =
        cmap.ytob_map.ConstRow(by / kColorTileDimInBlocks);

    const float* PIK_RESTRICT row_in_x = in.PlaneRow(0, by * kBlockDim);
    const float* PIK_RESTRICT row_in_y = in.PlaneRow(1, by * kBlockDim);
    const float* PIK_RESTRICT row_in_b = in.PlaneRow(2, by * kBlockDim);
    float* PIK_RESTRICT row_filt_x = filt.PlaneRow(0, by * kBlockDim);
    float* PIK_RESTRICT row_filt_y = filt.PlaneRow(1, by * kBlockDim);
    float* PIK_RESTRICT row_filt_b = filt.PlaneRow(2, by * kBlockDim);

    const int32_t* PIK_RESTRICT row_quant = quantizer.RawQuantField().Row(by);

    for (size_t bx = 0; bx < xsize_blocks; ++bx) {
      const size_t kind = GetQuantKindFromAcStrategy(ac_strategy, bx, by);
      const int32_t qac = row_quant[bx];
      const float quant_ac = quantizer.Scale() * qac;
      const float inv_quant_ac = quantizer.inv_quant_ac(qac);

      ClampBlockToOriginalDCT_Y(
          row_in_y + bx * kBlockDim, stride,
          &dequant_matrices[(3 * kind + 1) * kDCTBlockSize],
          &quant_matrices[(3 * kind + 1) * kDCTBlockSize], quant_ac,
          inv_quant_ac, dc_mul_y, inv_dc_mul_y, row_filt_y + bx * kBlockDim,
          original_y_block_unscaled, y_block, y_block_unscaled);

      ClampBlockToOriginalDCT_XB<TagX>(
          row_in_x + bx * kBlockDim, stride, original_y_block_unscaled, y_block,
          y_block_unscaled, row_cmap_x[bx / kColorTileDimInBlocks],
          dc_color_factor_x, &dequant_matrices[3 * kind * kDCTBlockSize],
          &quant_matrices[3 * kind * kDCTBlockSize], quant_ac, inv_quant_ac,
          dc_mul_x, inv_dc_mul_x, row_filt_x + bx * kBlockDim);

      ClampBlockToOriginalDCT_XB<TagB>(
          row_in_b + bx * kBlockDim, stride, original_y_block_unscaled, y_block,
          y_block_unscaled, row_cmap_b[bx / kColorTileDimInBlocks],
          dc_color_factor_b, &dequant_matrices[(3 * kind + 2) * kDCTBlockSize],
          &quant_matrices[(3 * kind + 2) * kDCTBlockSize], quant_ac,
          inv_quant_ac, dc_mul_b, inv_dc_mul_b, row_filt_b + bx * kBlockDim);
    }
  });
  return filt;
}

}  // namespace pik
