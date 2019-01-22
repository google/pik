// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "ac_predictions.h"
#include "ac_strategy.h"
#include "codec.h"
#include "compressed_image_fwd.h"
#include "data_parallel.h"
#include "opsin_inverse.h"

#undef PROFILER_ENABLED
#define PROFILER_ENABLED 1
#include "bits.h"
#include "block.h"
#include "common.h"
#include "convolve.h"
#include "dct.h"
#include "dct_util.h"
#include "deconvolve.h"
#include "entropy_coder.h"
#include "image.h"
#include "profiler.h"
#include "quantizer.h"
#include "resample.h"
#include "upscaler.h"

namespace pik {
namespace {
// Adds or subtracts block to/from "add_to",
// except elements 0,H,V,D. May overwrite parts of "block".
template <bool add>
SIMD_ATTR void AddBlockExcept0HVDTo(const float* PIK_RESTRICT block,
                                    float* PIK_RESTRICT add_to) {
  constexpr int N = kBlockDim;

  const SIMD_PART(float, SIMD_MIN(SIMD_FULL(float)::N, 8)) d;

#if SIMD_TARGET_VALUE == SIMD_NONE
  // Fallback because SIMD version assumes at least two lanes.
  block[0] = 0.0f;
  block[1] = 0.0f;
  block[N] = 0.0f;
  block[N + 1] = 0.0f;
  if (!add) {
    for (size_t i = 0; i < N * N; ++i) {
      add_to[i] -= block[i];
    }
  } else {
    for (size_t i = 0; i < N * N; ++i) {
      add_to[i] += block[i];
    }
  }
#else
  // Negated to enable default zero-initialization of upper lanes.
  SIMD_ALIGN uint32_t mask2[d.N] = {~0u, ~0u};
  const auto only_01 = load(d, reinterpret_cast<float*>(mask2));

  // First block row: don't add block[0, 1].
  auto prev = load(d, add_to + 0);
  auto coefs = load(d, block + 0);
  auto masked_coefs = andnot(only_01, coefs);
  auto sum = add ? prev + masked_coefs : prev - masked_coefs;
  store(sum, d, add_to + 0);
  // Handle remnants of DCT row (for 128-bit SIMD, or N > 8)
  for (size_t ix = d.N; ix < N; ix += d.N) {
    prev = load(d, add_to + ix);
    coefs = load(d, block + ix);
    sum = add ? prev + coefs : prev - coefs;
    store(sum, d, add_to + ix);
  }

  // Second block row: don't add block[V, D].
  prev = load(d, add_to + N);
  coefs = load(d, block + N);
  masked_coefs = andnot(only_01, coefs);
  sum = add ? prev + masked_coefs : prev - masked_coefs;
  store(sum, d, add_to + N);
  // Handle remnants of DCT row (for 128-bit SIMD, or N > 8)
  for (size_t ix = d.N; ix < N; ix += d.N) {
    prev = load(d, add_to + N + ix);
    coefs = load(d, block + N + ix);
    sum = add ? prev + coefs : prev - coefs;
    store(sum, d, add_to + N + ix);
  }

  for (size_t i = 2 * N; i < N * N; i += d.N) {
    prev = load(d, add_to + i);
    coefs = load(d, block + i);
    sum = add ? prev + coefs : prev - coefs;
    store(sum, d, add_to + i);
  }
#endif
}

// Un-color-correlates, quantizes, dequantizes and color-correlates the
// specified coefficients inside the given block, using or storing the y-channel
// values in y_block. Used by predictors to compute the decoder-side values to
// compute predictions on. Coefficients are specified as a bit array.
template <size_t c>
SIMD_ATTR PIK_INLINE void ComputeDecoderCoefficients(
    const float cmap_factor, const Quantizer& quantizer, const int32_t quant_ac,
    const float inv_quant_ac, const uint8_t quant_kind, const float* block_src,
    uint64_t coefficients, float* block, float* y_block) {
  constexpr size_t N = kBlockDim;
  for (uint64_t bits = coefficients; bits != 0; bits &= bits - 1) {
    size_t idx = NumZeroBitsBelowLSBNonzero(bits);
    block[idx] = block_src[idx];
  }
  if (c != 1) {
    for (uint64_t bits = coefficients; bits != 0; bits &= bits - 1) {
      size_t idx = NumZeroBitsBelowLSBNonzero(bits);
      block[idx] -= y_block[idx] * cmap_factor;
    }
  }
  int16_t qblock[N * N];
  quantizer.QuantizeBlockCoefficients(quant_ac, quant_kind, c, block, qblock,
                                      coefficients);
  const float* PIK_RESTRICT dequant_matrix =
      quantizer.DequantMatrix(c, quant_kind);
  for (uint64_t bits = coefficients; bits != 0; bits &= bits - 1) {
    size_t idx = NumZeroBitsBelowLSBNonzero(bits);
    block[idx] =
        AdjustQuantBias<c>(qblock[idx]) * (dequant_matrix[idx] * inv_quant_ac);
  }
  if (c != 1) {
    // Restore color correlation in the coefficients, as they will be
    // un-correlated later.
    for (uint64_t bits = coefficients; bits != 0; bits &= bits - 1) {
      size_t idx = NumZeroBitsBelowLSBNonzero(bits);
      block[idx] += y_block[idx] * cmap_factor;
    }
  } else {
    for (uint64_t bits = coefficients; bits != 0; bits &= bits - 1) {
      size_t idx = NumZeroBitsBelowLSBNonzero(bits);
      y_block[idx] = block[idx];
    }
  }
}

static constexpr float k4x4BlurStrength = 2.0007879236394901;

namespace lf_kernel {
struct LFPredictionBlur {
  PIK_INLINE const Weights3x3& Weights() const {
    static constexpr float w0 = 0.41459272584128337;
    static constexpr float w1 = 0.25489157325704559;
    static constexpr float w2 = 0.046449679523692139;
    static const Weights3x3 weights = {
        {SIMD_REP4(w2)}, {SIMD_REP4(w1)}, {SIMD_REP4(w2)},
        {SIMD_REP4(w1)}, {SIMD_REP4(w0)}, {SIMD_REP4(w1)},
        {SIMD_REP4(w2)}, {SIMD_REP4(w1)}, {SIMD_REP4(w2)}};
    return weights;
  }
};
}  // namespace lf_kernel

// Subtract predictions, compute decoder-side coefficients, add predictions and
// compute 2x2 block.
template <size_t c>
SIMD_ATTR void ComputeDecoderBlockAnd2x2DC(
    bool is_border, bool predict_lf, bool predict_hf, AcStrategy acs,
    const size_t residuals_stride, const size_t pred_stride,
    const size_t lf2x2_stride, const size_t bx, const Quantizer& quantizer,
    int32_t quant_ac, const float* PIK_RESTRICT cmap_factor,
    const float* PIK_RESTRICT pred[3], float* PIK_RESTRICT residuals[3],
    float* PIK_RESTRICT lf2x2_row[3], const float* PIK_RESTRICT dc[3],
    float* PIK_RESTRICT y_residuals_dec) {
  PROFILER_FUNC;
  constexpr size_t N = kBlockDim;
  float decoder_coeffs[AcStrategy::kMaxCoeffArea] = {};
  if (!is_border) {
    for (size_t iby = 0; iby < acs.covered_blocks_y(); iby++) {
      for (size_t ibx = 0; ibx < acs.covered_blocks_x(); ibx++) {
        float* current =
            residuals[c] + N * N * (bx + ibx - 1) + residuals_stride * iby;
        const float* pred_current =
            pred[c] + 2 * (bx + ibx) + pred_stride * 2 * iby;
        float* y_current =
            y_residuals_dec + N * N * (acs.covered_blocks_x() * iby + ibx);
        float* decoder_current =
            decoder_coeffs + N * N * (acs.covered_blocks_x() * iby + ibx);

        if (predict_lf) {
          // Remove prediction
          current[1] -= pred_current[1];
          current[N] -= pred_current[pred_stride];
          current[N + 1] -= pred_current[pred_stride + 1];
        }

        // Quantization roundtrip
        const size_t kind =
            acs.GetQuantKind(iby * acs.covered_blocks_x() + ibx);
        const float inv_quant_ac = quantizer.inv_quant_ac(quant_ac);
        // 0x302 has bits 1, 8, 9 set.
        ComputeDecoderCoefficients<c>(cmap_factor[c], quantizer, quant_ac,
                                      inv_quant_ac, kind, current, 0x302,
                                      decoder_current, y_current);

        decoder_current[0] = current[0];
        if (predict_lf) {
          // Add back prediction
          decoder_current[1] += pred_current[1];
          decoder_current[N] += pred_current[pred_stride];
          decoder_current[N + 1] += pred_current[pred_stride + 1];
        }
      }
    }
  } else {
    decoder_coeffs[0] = dc[c][bx];
    if (predict_lf) {
      decoder_coeffs[1] = pred[c][2 * bx + 1];
      decoder_coeffs[N] = pred[c][pred_stride + 2 * bx];
      decoder_coeffs[N + 1] = pred[c][pred_stride + 2 * bx + 1];
    }
  }
  if (predict_hf) {
    acs.DC2x2FromLowFrequencies(decoder_coeffs, N * N * acs.covered_blocks_x(),
                                lf2x2_row[c] + 2 * bx, lf2x2_stride);
  }
}

// Copies the lowest-frequency coefficients from DC- to AC-sized image.
SIMD_ATTR void CopyLlf(const Image3F& llf, Image3F* PIK_RESTRICT ac64) {
  PROFILER_FUNC;
  constexpr size_t N = kBlockDim;
  constexpr size_t block_size = N * N;
  const size_t xsize = llf.xsize() - 2;
  const size_t ysize = llf.ysize() - 2;

  // Copy (reinterpreted) DC values to 0-th block values.
  for (size_t c = 0; c < ac64->kNumPlanes; c++) {
    for (size_t by = 0; by < ysize; ++by) {
      const float* llf_row = llf.ConstPlaneRow(c, by + 1);
      float* ac_row = ac64->PlaneRow(c, by);
      for (size_t bx = 0; bx < xsize; bx++) {
        ac_row[block_size * bx] = llf_row[bx + 1];
      }
    }
  }
}

typedef std::array<std::array<float, 4>, 3> Ub4Kernel;

SIMD_ATTR void ComputeUb4Kernel(const float sigma, Ub4Kernel* out) {
  for (int j = 0; j < 3; ++j) {
    for (int k = 0; k < 4; ++k) {
      out->at(j)[k] = 0.0f;
    }
  }
  std::vector<float> kernel = GaussianKernel(4, sigma);
  for (int k = 0; k < 4; ++k) {
    const int split0 = 4 - k;
    const int split1 = 8 - k;
    for (int j = 0; j < split0; ++j) {
      out->at(0)[k] += kernel[j];
    }
    for (int j = split0; j < split1; ++j) {
      out->at(1)[k] += kernel[j];
    }
    for (int j = split1; j < kernel.size(); ++j) {
      out->at(2)[k] += kernel[j];
    }
  }
}

// Adds to "add_to" (DCT) an image defined by the following transformations:
//  1) Upsample image 4x4 with nearest-neighbor
//  2) Blur with a Gaussian kernel of radius 4 and given sigma
//  3) perform TransposedScaledDCT()
//  4) Zero out the top 2x2 corner of each DCT block
//  5) Negates the prediction if add is false (so the encoder subtracts, and
//  the decoder adds)
template <bool add>
SIMD_ATTR void UpSample4x4BlurDCT(const Rect& dc_rect, const ImageF& img,
                                  const Ub4Kernel& kernel,
                                  const AcStrategyImage& ac_strategy,
                                  const Rect& acs_rect, ImageF* add_to) {
  PROFILER_FUNC;
  constexpr size_t N = kBlockDim;
  constexpr size_t block_size = N * N;

  // TODO(robryk): There's no good reason to compute the full DCT here. It's
  // fine if the output is in pixel space, we just need to zero out top 2x2
  // DCT coefficients. We can do that by computing a "partial DCT" and
  // subtracting (we can have two outputs: a positive pixel-space output and a
  // negative DCT-space output).

  // TODO(robryk): Failing that, merge the blur and DCT into a single linear
  // operation, if feasible.

  const size_t bx0 = dc_rect.x0();
  const size_t bxs = dc_rect.xsize();
  PIK_CHECK(bxs >= 1);
  const size_t bx1 = bx0 + bxs;
  const size_t bx_max = DivCeil(add_to->xsize(), block_size);
  const size_t by0 = dc_rect.y0();
  const size_t bys = dc_rect.ysize();
  PIK_CHECK(bys >= 1);
  const size_t by1 = by0 + bys;
  const size_t by_max = add_to->ysize();
  PIK_CHECK(bx1 <= bx_max && by1 <= by_max);
  const size_t xs = bxs * 2;
  const size_t ys = bys * 2;

  using D = SIMD_PART(float, SIMD_MIN(SIMD_FULL(float)::N, 8));
  using V = D::V;
  const D d;
  V vw0[4] = {set1(d, kernel[0][0]), set1(d, kernel[0][1]),
              set1(d, kernel[0][2]), set1(d, kernel[0][3])};
  V vw1[4] = {set1(d, kernel[1][0]), set1(d, kernel[1][1]),
              set1(d, kernel[1][2]), set1(d, kernel[1][3])};
  V vw2[4] = {set1(d, kernel[2][0]), set1(d, kernel[2][1]),
              set1(d, kernel[2][2]), set1(d, kernel[2][3])};

  ImageF blur_x(xs * 4, ys + 2);
  for (size_t y = 0; y < ys + 2; ++y) {
    const float* PIK_RESTRICT row = img.ConstRow(y + 1);
    float* const PIK_RESTRICT row_out = blur_x.Row(y);
    for (int x = 0; x < xs; ++x) {
      const float v0 = row[x + 1];
      const float v1 = row[x + 2];
      const float v2 = row[x + 3];
      for (int ix = 0; ix < 4; ++ix) {
        row_out[4 * x + ix] =
            v0 * kernel[0][ix] + v1 * kernel[1][ix] + v2 * kernel[2][ix];
      }
    }
  }

  {
    PROFILER_ZONE("dct upsample");
    for (size_t by = 0; by < bys; ++by) {
      const D d;
      SIMD_ALIGN float block[AcStrategy::kMaxCoeffArea];
      SIMD_ALIGN float temp_block[AcStrategy::kMaxCoeffArea];
      const size_t out_stride = add_to->PixelsPerRow();
      const size_t blur_stride = blur_x.PixelsPerRow();

      float* PIK_RESTRICT row_out = add_to->Row(by0 + by);
      AcStrategyRow ac_strategy_row = ac_strategy.ConstRow(acs_rect, by0 + by);
      for (int bx = 0; bx < bxs; ++bx) {
        AcStrategy acs = ac_strategy_row[bx0 + bx];
        if (!acs.IsFirstBlock()) continue;
        if (!acs.PredictHF()) continue;
        for (int idy = 0; idy < acs.covered_blocks_y(); idy++) {
          const float* PIK_RESTRICT row0d = blur_x.ConstRow(2 * (by + idy));
          const float* PIK_RESTRICT row1d = row0d + blur_stride;
          const float* PIK_RESTRICT row2d = row1d + blur_stride;
          const float* PIK_RESTRICT row3d = row2d + blur_stride;
          for (int idx = 0; idx < acs.covered_blocks_x(); idx++) {
            float* PIK_RESTRICT block_ptr =
                block + AcStrategy::kMaxCoeffBlocks * block_size * idy +
                8 * idx;
            for (int ix = 0; ix < 8; ix += d.N) {
              const auto val0 = load(d, &row0d[(bx + idx) * 8 + ix]);
              const auto val1 = load(d, &row1d[(bx + idx) * 8 + ix]);
              const auto val2 = load(d, &row2d[(bx + idx) * 8 + ix]);
              const auto val3 = load(d, &row3d[(bx + idx) * 8 + ix]);
              for (int iy = 0; iy < 4; ++iy) {
                // A mul_add pair is faster but causes 1E-5 difference.
                const auto vala =
                    val0 * vw0[iy] + val1 * vw1[iy] + val2 * vw2[iy];
                const auto valb =
                    val1 * vw0[iy] + val2 * vw1[iy] + val3 * vw2[iy];
                store(vala, d, &block_ptr[iy * AcStrategy::kMaxBlockDim + ix]);
                store(valb, d,
                      &block_ptr[iy * AcStrategy::kMaxBlockDim +
                                 AcStrategy::kMaxBlockDim * 4 + ix]);
              }
            }
          }
        }

        acs.TransformFromPixels(block, AcStrategy::kMaxBlockDim, temp_block,
                                AcStrategy::kMaxCoeffBlocks * block_size);
        for (size_t iby = 0; iby < acs.covered_blocks_y(); iby++) {
          for (size_t ibx = 0; ibx < acs.covered_blocks_x(); ibx++) {
            const float* PIK_RESTRICT in =
                temp_block +
                block_size * (AcStrategy::kMaxCoeffBlocks * iby + ibx);
            float* PIK_RESTRICT out =
                row_out + block_size * (bx0 + bx + ibx) + out_stride * iby;
            AddBlockExcept0HVDTo<add>(in, out);
          }
        }
      }
    }
  }
}

}  // namespace

// Compute the lowest-frequency coefficients in the DCT block (1x1 for DCT8,
// 2x2 for DCT16, etc.)
SIMD_ATTR void ComputeLlf(const Image3F& dc, const AcStrategyImage& ac_strategy,
                          const Rect& acs_rect, Image3F* PIK_RESTRICT llf) {
  PROFILER_FUNC;
  const size_t xsize = dc.xsize();
  const size_t ysize = dc.ysize();
  const size_t dc_stride = dc.PixelsPerRow();
  const size_t llf_stride = llf->PixelsPerRow();

  // Copy (reinterpreted) DC values to 0-th block values.
  for (size_t c = 0; c < llf->kNumPlanes; c++) {
    for (size_t by = 0; by < ysize; ++by) {
      const bool is_border_y = by == 0 || by == ysize - 1;
      AcStrategyRow ac_strategy_row =
          ac_strategy.ConstRow(acs_rect, is_border_y ? 0 : by - 1);
      const float* dc_row = dc.ConstPlaneRow(c, by);
      float* llf_row = llf->PlaneRow(c, by);
      for (size_t bx = 0; bx < xsize; bx++) {
        const bool is_border = is_border_y || (bx == 0 || bx == xsize - 1);
        AcStrategy acs = is_border ? AcStrategy(AcStrategy::Type::DCT, 0)
                                   : ac_strategy_row[bx - 1];
        acs.LowestFrequenciesFromDC(dc_row + bx, dc_stride, llf_row + bx,
                                    llf_stride);
      }
    }
  }
}

// The LF prediction works as follows:
// - Blur the initial DC2x2 image (see ComputeSharpDc2x2FromLlf)
// - Compute the same-size DCT of the resulting blurred image
SIMD_ATTR void PredictLf(const AcStrategyImage& ac_strategy,
                         const Rect& acs_rect, const Image3F& llf,
                         ImageF* tmp2x2, Image3F* lf2x2) {
  PROFILER_FUNC;
  const size_t xsize = llf.xsize();
  const size_t ysize = llf.ysize();
  const size_t llf_stride = llf.PixelsPerRow();
  const size_t lf2x2_stride = lf2x2->PixelsPerRow();
  const size_t tmp2x2_stride = tmp2x2->PixelsPerRow();

  // Plane-wise transforms require 2*4DC*4 = 128KiB active memory. Would be
  // further subdivided into 2 or more stripes to reduce memory pressure.
  for (size_t c = 0; c < lf2x2->kNumPlanes; c++) {
    ImageF* lf2x2_plane = lf2x2->MutablePlane(c);

    // Computes the initial DC2x2 from the lowest-frequency coefficients.
    for (size_t by = 0; by < ysize; ++by) {
      const bool is_border_y = by == 0 || by == ysize - 1;
      AcStrategyRow ac_strategy_row =
          ac_strategy.ConstRow(acs_rect, is_border_y ? 0 : by - 1);
      float* tmp2x2_row = tmp2x2->Row(2 * by);
      const float* llf_row = llf.PlaneRow(c, by);
      for (size_t bx = 0; bx < xsize; bx++) {
        const bool is_border = is_border_y || (bx == 0 || bx == xsize - 1);
        AcStrategy acs = is_border ? AcStrategy(AcStrategy::Type::DCT, 0)
                                   : ac_strategy_row[bx - 1];
        acs.DC2x2FromLowestFrequencies(llf_row + bx, llf_stride,
                                       tmp2x2_row + 2 * bx, tmp2x2_stride);
      }
    }

    // Smooth out DC2x2.
    if (xsize * 2 < kConvolveMinWidth) {
      using Convolution = slow::General3x3Convolution<1, WrapMirror>;
      Convolution::Run(*tmp2x2, xsize * 2, ysize * 2,
                       lf_kernel::LFPredictionBlur(), lf2x2_plane);
    } else {
      const BorderNeverUsed border;
      // Parallel doesn't help here for moderate-sized images.
      const ExecutorLoop executor;
      ConvolveT<strategy::Symmetric3>::Run(border, executor, *tmp2x2,
                                           lf_kernel::LFPredictionBlur(),
                                           lf2x2_plane);
    }

    // Compute LF coefficients
    for (size_t by = 0; by < ysize; ++by) {
      const bool is_border_y = by == 0 || by == ysize - 1;
      AcStrategyRow ac_strategy_row =
          ac_strategy.ConstRow(acs_rect, is_border_y ? 0 : by - 1);
      float* lf2x2_row = lf2x2_plane->Row(2 * by);
      for (size_t bx = 0; bx < xsize; bx++) {
        const bool is_border = is_border_y || (bx == 0 || bx == xsize - 1);
        AcStrategy acs = is_border ? AcStrategy(AcStrategy::Type::DCT, 0)
                                   : ac_strategy_row[bx - 1];
        acs.LowFrequenciesFromDC2x2(lf2x2_row + 2 * bx, lf2x2_stride,
                                    lf2x2_row + 2 * bx, lf2x2_stride);
      }
    }
  }
}

// Predict dc2x2 from DC values.
// - Use the LF block (but not the lowest frequency block) as a predictor
// - Update those values with the actual residuals, and re-compute a 2x
//   upsampled image out of that as an input for HF predictions.
SIMD_ATTR void PredictLfForEncoder(bool predict_lf, bool predict_hf,
                                   const Image3F& dc,
                                   const AcStrategyImage& ac_strategy,
                                   const ColorCorrelationMap& cmap,
                                   const Quantizer& quantizer,
                                   Image3F* PIK_RESTRICT ac64, Image3F* dc2x2) {
  PROFILER_FUNC;
  const size_t xsize = dc.xsize();
  const size_t ysize = dc.ysize();
  const size_t ac_stride = ac64->PixelsPerRow();
  const size_t dc2x2_stride = dc2x2->PixelsPerRow();
  // TODO(user): should not be allocated, when predict_lf == false.
  Image3F lf2x2(xsize * 2, ysize * 2);
  {
    Image3F llf(xsize, ysize);
    ComputeLlf(dc, ac_strategy, Rect(ac_strategy.ConstRaw()), &llf);
    if (predict_lf) {
      // dc2x2 plane is borrowed for temporary storage.
      PredictLf(ac_strategy, Rect(ac_strategy.ConstRaw()), llf,
                dc2x2->MutablePlane(0), &lf2x2);
    }
    CopyLlf(llf, ac64);
  }
  const size_t lf2x2_stride = lf2x2.PixelsPerRow();

  // Compute decoder-side coefficients, 2x scaled DC image, and subtract
  // predictions.
  for (size_t by = 0; by < ysize; ++by) {
    float y_residuals_dec[AcStrategy::kMaxCoeffArea];
    const bool is_border_y = by == 0 || by == ysize - 1;
    // The following variables will not be used if we are on a border.
    AcStrategyRow acr = ac_strategy.ConstRow(is_border_y ? 0 : by - 1);
    float* ac_row[3] = {ac64->PlaneRow(0, is_border_y ? 0 : by - 1),
                        ac64->PlaneRow(1, is_border_y ? 0 : by - 1),
                        ac64->PlaneRow(2, is_border_y ? 0 : by - 1)};

    const float* dc_row[3] = {dc.ConstPlaneRow(0, by), dc.ConstPlaneRow(1, by),
                              dc.ConstPlaneRow(2, by)};
    const float* lf2x2_row[3] = {lf2x2.ConstPlaneRow(0, 2 * by),
                                 lf2x2.ConstPlaneRow(1, 2 * by),
                                 lf2x2.ConstPlaneRow(2, 2 * by)};
    float* dc2x2_row[3] = {dc2x2->PlaneRow(0, 2 * by),
                           dc2x2->PlaneRow(1, 2 * by),
                           dc2x2->PlaneRow(2, 2 * by)};
    const int32_t* row_quant =
        quantizer.RawQuantField().ConstRow(is_border_y ? 0 : by - 1);
    for (size_t bx = 0; bx < xsize; bx++) {
      const bool is_border = is_border_y || (bx == 0 || bx == xsize - 1);
      AcStrategy acs =
          is_border ? AcStrategy(AcStrategy::Type::DCT, 0) : acr[bx - 1];
      if (!acs.IsFirstBlock()) continue;
      size_t tx = (is_border ? 0 : bx - 1) / kColorTileDimInBlocks;
      size_t ty = (is_border ? 0 : by - 1) / kColorTileDimInBlocks;
      float cmap_factor[3] = {
          ColorCorrelationMap::YtoX(1.0f, cmap.ytox_map.ConstRow(ty)[tx]), 0.0f,
          ColorCorrelationMap::YtoB(1.0f, cmap.ytob_map.ConstRow(ty)[tx])};
      const int32_t quant = bx == 0 ? row_quant[0] : row_quant[bx - 1];
      ComputeDecoderBlockAnd2x2DC<1>(
          is_border, predict_lf, predict_hf, acs, ac_stride, lf2x2_stride,
          dc2x2_stride, bx, quantizer, quant, cmap_factor, lf2x2_row, ac_row,
          dc2x2_row, dc_row, y_residuals_dec);
      ComputeDecoderBlockAnd2x2DC<0>(
          is_border, predict_lf, predict_hf, acs, ac_stride, lf2x2_stride,
          dc2x2_stride, bx, quantizer, quant, cmap_factor, lf2x2_row, ac_row,
          dc2x2_row, dc_row, y_residuals_dec);
      ComputeDecoderBlockAnd2x2DC<2>(
          is_border, predict_lf, predict_hf, acs, ac_stride, lf2x2_stride,
          dc2x2_stride, bx, quantizer, quant, cmap_factor, lf2x2_row, ac_row,
          dc2x2_row, dc_row, y_residuals_dec);
    }
  }
}

// Similar to PredictLfForEncoder.
SIMD_ATTR void UpdateLfForDecoder(const Rect& tile, bool predict_lf,
                                  bool predict_hf,
                                  const AcStrategyImage& ac_strategy,
                                  const Rect& acs_rect, const ImageF& llf_plane,
                                  ImageF* ac64_plane, ImageF* dc2x2_plane,
                                  ImageF* lf2x2_plane) {
  constexpr size_t N = kBlockDim;
  constexpr size_t block_size = N * N;
  const size_t bx0 = tile.x0();
  const size_t bx1 = bx0 + tile.xsize();
  const size_t by0 = tile.y0();
  const size_t by1 = by0 + tile.ysize();
  const size_t ac_stride = ac64_plane->PixelsPerRow();
  const size_t dc2x2_stride = predict_hf ? dc2x2_plane->PixelsPerRow() : 0;
  const size_t lf2x2_stride = predict_lf ? lf2x2_plane->PixelsPerRow() : 0;

  for (size_t by = by0; by < by1; ++by) {
    const float* llf_row = llf_plane.ConstRow(by + 1);
    float* ac_row = ac64_plane->Row(by);
    for (size_t bx = bx0; bx < bx1; bx++) {
      ac_row[block_size * bx] = llf_row[bx + 1];
    }
  }

  // Compute decoder-side coefficients, 2x scaled DC image, and subtract
  // predictions.
  // Add predictions and compute 2x scaled image to feed to HF predictor
  if (predict_lf) {
    for (size_t by = by0; by < by1; ++by) {
      AcStrategyRow ac_strategy_row = ac_strategy.ConstRow(acs_rect, by);
      float* ac_row = ac64_plane->Row(by);
      const float* lf2x2_row = lf2x2_plane->ConstRow(2 * (by + 1));
      for (size_t bx = bx0; bx < bx1; bx++) {
        AcStrategy acs = ac_strategy_row[bx];
        if (!acs.IsFirstBlock()) continue;
        if (predict_lf) {
          for (size_t iy = 0; iy < acs.covered_blocks_y(); iy++) {
            for (size_t ix = 0; ix < acs.covered_blocks_x(); ix++) {
              float* ac_pos = ac_row + ac_stride * iy + (bx + ix) * block_size;
              const float* lf2x2_pos =
                  lf2x2_row + lf2x2_stride * iy * 2 + (bx + 1 + ix) * 2;
              ac_pos[1] += lf2x2_pos[1];
              ac_pos[N] += lf2x2_pos[lf2x2_stride];
              ac_pos[N + 1] += lf2x2_pos[lf2x2_stride + 1];
            }
          }
        }
      }
    }
  }

  if (predict_hf) {
    for (size_t by = by0; by < by1; ++by) {
      AcStrategyRow ac_strategy_row = ac_strategy.ConstRow(acs_rect, by);
      float* dc2x2_row = dc2x2_plane->Row(2 * (by + 1));
      const float* ac_row = ac64_plane->Row(by);
      for (size_t bx = bx0; bx < bx1; bx++) {
        AcStrategy acs = ac_strategy_row[bx];
        if (!acs.IsFirstBlock()) continue;
        acs.DC2x2FromLowFrequencies(ac_row + block_size * bx, ac_stride,
                                    dc2x2_row + 2 * (bx + 1), dc2x2_stride);
      }
    }
  }
}

SIMD_ATTR void ComputePredictionResiduals(const Image3F& pred2x2,
                                          const AcStrategyImage& ac_strategy,
                                          Image3F* PIK_RESTRICT coeffs) {
  Rect dc_rect(0, 0, pred2x2.xsize() / 2 - 2, pred2x2.ysize() / 2 - 2);
  Rect acs_rect(0, 0, ac_strategy.xsize(), ac_strategy.ysize());
  Ub4Kernel kernel;
  ComputeUb4Kernel(k4x4BlurStrength, &kernel);
  for (int c = 0; c < coeffs->kNumPlanes; ++c) {
    UpSample4x4BlurDCT</*add=*/false>(dc_rect, pred2x2.Plane(c), kernel,
                                      ac_strategy, acs_rect,
                                      coeffs->MutablePlane(c));
  }
}

void AddPredictions(const Image3F& pred2x2, const AcStrategyImage& ac_strategy,
                    const Rect& acs_rect, Image3F* PIK_RESTRICT dcoeffs) {
  PROFILER_FUNC;
  Rect dc_rect(0, 0, pred2x2.xsize() / 2 - 2, pred2x2.ysize() / 2 - 2);
  Ub4Kernel kernel;
  ComputeUb4Kernel(k4x4BlurStrength, &kernel);
  for (int c = 0; c < dcoeffs->kNumPlanes; ++c) {
    // Updates dcoeffs _except_ 0HVD.
    UpSample4x4BlurDCT</*add=*/true>(dc_rect, pred2x2.Plane(c), kernel,
                                     ac_strategy, acs_rect,
                                     dcoeffs->MutablePlane(c));
  }
}

}  // namespace pik
