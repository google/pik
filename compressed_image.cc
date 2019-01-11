// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "compressed_image.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>

#undef PROFILER_ENABLED
#define PROFILER_ENABLED 1
#include "ac_predictions.h"
#include "ac_strategy.h"
#include "ans_decode.h"
#include "block.h"
#include "butteraugli_distance.h"
#include "common.h"
#include "compiler_specific.h"
#include "compressed_dc.h"
#include "compressed_image_fwd.h"
#include "convolve.h"
#include "dc_predictor.h"
#include "dct.h"
#include "dct_util.h"
#include "deconvolve.h"
#include "entropy_coder.h"
#include "fields.h"
#include "gaborish.h"
#include "gauss_blur.h"
#include "gradient_map.h"
#include "headers.h"
#include "huffman_decode.h"
#include "huffman_encode.h"
#include "image.h"
#include "lossless16.h"
#include "lossless8.h"
#include "opsin_image.h"
#include "opsin_inverse.h"
#include "opsin_params.h"
#include "profiler.h"
#include "quantizer.h"
#include "resample.h"
#include "simd/simd.h"
#include "status.h"
#include "upscaler.h"

namespace pik {

namespace {

void ZeroDcValues(Image3F* image) {
  const constexpr size_t N = kBlockDim;
  const size_t xsize_blocks = image->xsize() / (N * N);
  const size_t ysize_blocks = image->ysize();
  for (size_t c = 0; c < image->kNumPlanes; c++) {
    for (size_t by = 0; by < ysize_blocks; by++) {
      float* PIK_RESTRICT stored_values = image->PlaneRow(c, by);
      for (size_t bx = 0; bx < xsize_blocks; bx++) {
        stored_values[bx * N * N] = 0;
      }
    }
  }
}

}  // namespace

constexpr float kIdentityAvgParam = 0.25;

// This struct allow to remove the X and B channels of XYB images, and
// reconstruct them again from only the Y channel, when the image is grayscale.
struct GrayXyb {
  static const constexpr int kM = 16;  // Amount of line pieces.

  GrayXyb() { Compute(); }

  void YToXyb(float y, float* x, float* b) const {
    int i = (int)((y - ysub) * ymul * kM);
    i = std::min(std::max(0, i), kM - 1);
    *x = y * y_to_x_slope[i] + y_to_x_constant[i];
    *b = y * y_to_b_slope[i] + y_to_b_constant[i];
  }

  void RemoveXB(Image3F* image) const {
    for (size_t y = 0; y < image->ysize(); ++y) {
      float* PIK_RESTRICT row_x = image->PlaneRow(0, y);
      float* PIK_RESTRICT row_b = image->PlaneRow(2, y);
      for (size_t x = 0; x < image->xsize(); x++) {
        row_x[x] = 0;
        row_b[x] = 0;
      }
    }
  }

  void RestoreXB(Image3F* image) const {
    for (size_t y = 0; y < image->ysize(); ++y) {
      const float* PIK_RESTRICT row_y = image->PlaneRow(1, y);
      float* PIK_RESTRICT row_x = image->PlaneRow(0, y);
      float* PIK_RESTRICT row_b = image->PlaneRow(2, y);
      for (size_t x = 0; x < image->xsize(); x++) {
        YToXyb(row_y[x], &row_x[x], &row_b[x]);
      }
    }
  }

 private:
  void Compute() {
    static const int kN = 1024;
    std::vector<float> x(kN);
    std::vector<float> y(kN);
    std::vector<float> z(kN);
    for (int i = 0; i < kN; i++) {
      float gray = (float)(256.0f * i / kN);
      LinearToXyb(gray, gray, gray, &x[i], &y[i], &z[i]);
    }

    float min = y[0];
    float max = y[kN - 1];
    int m = 0;
    int border[kM + 1];
    for (int i = 0; i < kN; i++) {
      if (y[i] >= y[0] + (max - min) * m / kM) {
        border[m] = i;
        m++;
      }
    }
    border[kM] = kN;

    ysub = min;
    ymul = 1.0 / (max - min);

    for (int i = 0; i < kM; i++) {
      LinearRegression(y.data() + border[i], x.data() + border[i],
                       border[i + 1] - border[i], &y_to_x_constant[i],
                       &y_to_x_slope[i]);
      LinearRegression(y.data() + border[i], z.data() + border[i],
                       border[i + 1] - border[i], &y_to_b_constant[i],
                       &y_to_b_slope[i]);
    }
  }

  // finds a and b such that y ~= b*x + a
  void LinearRegression(const float* x, const float* y, size_t size, double* a,
                        double* b) {
    double mx = 0, my = 0;    // mean
    double mx2 = 0, my2 = 0;  // second moment
    double mxy = 0;
    for (size_t i = 0; i < size; i++) {
      double inv = 1.0 / (i + 1);

      double dx = x[i] - mx;
      double xn = dx * inv;
      mx += xn;
      mx2 += dx * xn * i;

      double dy = y[i] - my;
      double yn = dy * inv;
      my += yn;
      my2 += dy * yn * i;

      mxy += i * xn * yn - mxy * inv;
    }

    double sx = std::sqrt(mx2 / (size - 1));
    double sy = std::sqrt(my2 / (size - 1));

    double sumxy = mxy * size + my * mx * size;
    double r = (sumxy - size * mx * my) / ((size - 1.0) * sx * sy);

    *b = r * sy / sx;
    *a = my - *b * mx;
  }

  double y_to_x_slope[kM];
  double y_to_x_constant[kM];
  double y_to_b_slope[kM];
  double y_to_b_constant[kM];

  double ysub;
  double ymul;
};

static const std::unique_ptr<GrayXyb> kGrayXyb(new GrayXyb);

SIMD_ATTR void InitializePassEncCache(const PassHeader& pass_header,
                                      const Image3F& opsin_full,
                                      const AcStrategyImage& ac_strategy,
                                      const Quantizer& quantizer,
                                      const ColorCorrelationMap& cmap,
                                      ThreadPool* pool,
                                      PassEncCache* pass_enc_cache) {
  PROFILER_FUNC;
  constexpr size_t N = kBlockDim;
  constexpr int block_size = N * N;
  pass_enc_cache->use_gradient = pass_header.flags & PassHeader::kGradientMap;
  pass_enc_cache->grayscale_opt = pass_header.flags & PassHeader::kGrayscaleOpt;
  const size_t xsize_blocks = opsin_full.xsize() / N;
  const size_t ysize_blocks = opsin_full.ysize() / N;

  pass_enc_cache->coeffs = Image3F(xsize_blocks * block_size, ysize_blocks);
  Image3F dc = Image3F(xsize_blocks, ysize_blocks);

  auto compute_dc = [&](int by, int _) {
    for (int c = 0; c < 3; ++c) {
      for (size_t bx = 0; bx < xsize_blocks; ++bx) {
        AcStrategy acs = ac_strategy.ConstRow(by)[bx];
        acs.TransformFromPixels(
            opsin_full.ConstPlaneRow(c, by * N) + bx * N,
            opsin_full.PixelsPerRow(),
            pass_enc_cache->coeffs.PlaneRow(c, by) + bx * block_size,
            pass_enc_cache->coeffs.PixelsPerRow());
        acs.DCFromLowestFrequencies(
            pass_enc_cache->coeffs.ConstPlaneRow(c, by) + bx * block_size,
            pass_enc_cache->coeffs.PixelsPerRow(), dc.PlaneRow(c, by) + bx,
            dc.PixelsPerRow());
      }
    }
  };

  RunOnPool(pool, 0, ysize_blocks, compute_dc);

  if (pass_enc_cache->use_gradient) {
    ComputeGradientMap(dc, pass_enc_cache->grayscale_opt, quantizer, pool,
                       &pass_enc_cache->gradient);
  }

  constexpr int cY = 1;  // Y color channel.

  {
    ImageF dec_dc_Y = QuantizeRoundtripDC(quantizer, cY, dc.Plane(cY));

    if (pass_enc_cache->grayscale_opt) {
      kGrayXyb->RemoveXB(&dc);
    } else {
      ApplyColorCorrelationDC</*decode=*/false>(cmap, dec_dc_Y, &dc);
    }

    pass_enc_cache->dc = QuantizeCoeffsDC(dc, quantizer);
    pass_enc_cache->dc_dec =
        Image3F(pass_enc_cache->dc.xsize(), pass_enc_cache->dc.ysize());
    for (size_t c = 0; c < 3; c++) {
      const float mul = quantizer.DequantMatrix(c, kQuantKindDCT8)[0] *
                        quantizer.inv_quant_dc();
      for (size_t y = 0; y < pass_enc_cache->dc.ysize(); y++) {
        const int16_t* PIK_RESTRICT row_in =
            pass_enc_cache->dc.ConstPlaneRow(c, y);
        float* PIK_RESTRICT row_out = pass_enc_cache->dc_dec.PlaneRow(c, y);
        for (size_t x = 0; x < pass_enc_cache->dc.xsize(); x++) {
          row_out[x] = row_in[x] * mul;
        }
      }
    }
    if (!pass_enc_cache->grayscale_opt) {
      ApplyColorCorrelationDC</*decode=*/true>(cmap, dec_dc_Y,
                                               &pass_enc_cache->dc_dec);
    } else {
      kGrayXyb->RestoreXB(&pass_enc_cache->dc_dec);
    }

    if (pass_enc_cache->use_gradient) {
      ApplyGradientMap(pass_enc_cache->gradient, quantizer,
                       &pass_enc_cache->dc_dec);
    }
  }
}

SIMD_ATTR void InitializeEncCache(const PassHeader& pass_header,
                                  const GroupHeader& group_header,
                                  const PassEncCache& pass_enc_cache,
                                  const Rect& group_rect, EncCache* enc_cache) {
  PROFILER_FUNC;
  constexpr size_t N = kBlockDim;
  constexpr int block_size = N * N;
  PIK_ASSERT(!enc_cache->initialized);

  const size_t full_xsize_blocks = pass_enc_cache.dc_dec.xsize();
  const size_t full_ysize_blocks = pass_enc_cache.dc_dec.ysize();
  const size_t x0_blocks = group_rect.x0() / N;
  const size_t y0_blocks = group_rect.y0() / N;

  enc_cache->xsize_blocks = group_rect.xsize() / N;
  enc_cache->ysize_blocks = group_rect.ysize() / N;
  enc_cache->predict_lf = pass_header.predict_lf;
  enc_cache->predict_hf = pass_header.predict_hf;
  enc_cache->grayscale_opt = pass_enc_cache.grayscale_opt;

  enc_cache->dc_dec =
      Image3F(enc_cache->xsize_blocks + 2, enc_cache->ysize_blocks + 2);
  for (size_t c = 0; c < 3; c++) {
    for (size_t y = 0; y < enc_cache->ysize_blocks + 2; y++) {
      const size_t y_src = SourceCoord(y + y0_blocks, full_ysize_blocks);
      const float* row_src = pass_enc_cache.dc_dec.ConstPlaneRow(c, y_src);
      float* row_dc = enc_cache->dc_dec.PlaneRow(c, y);
      for (size_t x = 0; x < enc_cache->xsize_blocks + 2; x++) {
        const size_t x_src = SourceCoord(x + x0_blocks, full_xsize_blocks);
        row_dc[x] = row_src[x_src];
      }
    }
  }

  const Rect coeff_rect(x0_blocks * block_size, y0_blocks,
                        enc_cache->xsize_blocks * block_size,
                        enc_cache->ysize_blocks);

  enc_cache->coeffs = CopyImage(coeff_rect, pass_enc_cache.coeffs);

  enc_cache->initialized = true;
}

SIMD_ATTR void ComputeCoefficients(const Quantizer& quantizer,
                                   const ColorCorrelationMap& cmap,
                                   ThreadPool* pool, EncCache* enc_cache,
                                   MultipassManager* manager,
                                   const PikInfo* aux_out) {
  PROFILER_FUNC;
  constexpr size_t N = kBlockDim;
  constexpr size_t block_size = N * N;
  const size_t xsize_blocks = enc_cache->xsize_blocks;
  const size_t ysize_blocks = enc_cache->ysize_blocks;
  PIK_ASSERT(enc_cache->initialized);

  manager->StripInfoBeforePredictions(enc_cache);

  enc_cache->quant_field = CopyImage(quantizer.RawQuantField());
  ImageI& quant_field = enc_cache->quant_field;

  // TODO(user): it would be better to find & apply correlation here, when
  // quantization is chosen.

  Image3F coeffs_init;
  if (aux_out && aux_out->testing_aux.ac_prediction != nullptr) {
    coeffs_init = CopyImage(enc_cache->coeffs);
  }

  constexpr int cY = 1;

  {
    PROFILER_ZONE("enc predictions");

    Image3F pred2x2(enc_cache->dc_dec.xsize() * 2,
                    enc_cache->dc_dec.ysize() * 2);
    PredictLfForEncoder(enc_cache->predict_lf, enc_cache->predict_hf,
                        enc_cache->dc_dec, enc_cache->ac_strategy, cmap,
                        quantizer, &enc_cache->coeffs, &pred2x2);
    if (enc_cache->predict_hf) {
      ComputePredictionResiduals(pred2x2, enc_cache->ac_strategy,
                                 &enc_cache->coeffs);
    }
  }

  if (aux_out && aux_out->testing_aux.ac_prediction != nullptr) {
    Subtract(coeffs_init, enc_cache->coeffs,
             aux_out->testing_aux.ac_prediction);
    ZeroDcValues(aux_out->testing_aux.ac_prediction);
  }

  {
    Image3F coeffs_ac = CopyImage(enc_cache->coeffs);

    ImageF dec_ac_Y(xsize_blocks * block_size, ysize_blocks);
    for (size_t by = 0; by < ysize_blocks; ++by) {
      const float* PIK_RESTRICT row_in = coeffs_ac.ConstPlaneRow(cY, by);
      float* PIK_RESTRICT row_out = dec_ac_Y.Row(by);
      AcStrategyRow ac_strategy_row = enc_cache->ac_strategy.ConstRow(by);
      for (size_t bx = 0; bx < xsize_blocks; ++bx) {
        const int32_t quant_ac = quant_field.Row(by)[bx];
        quantizer.QuantizeRoundtripBlockAC<cY>(
            quant_ac, ac_strategy_row[bx].GetQuantKind(),
            row_in + bx * block_size, row_out + bx * block_size);
      }
    }

    UnapplyColorCorrelationAC(cmap, dec_ac_Y, &coeffs_ac);

    enc_cache->ac = Image3S(xsize_blocks * block_size, ysize_blocks);
    for (int c = 0; c < 3; ++c) {
      for (size_t by = 0; by < ysize_blocks; ++by) {
        const float* PIK_RESTRICT row_in = coeffs_ac.PlaneRow(c, by);
        int16_t* PIK_RESTRICT row_out = enc_cache->ac.PlaneRow(c, by);
        const int32_t* row_quant = quant_field.ConstRow(by);
        AcStrategyRow ac_strategy_row = enc_cache->ac_strategy.ConstRow(by);
        for (size_t bx = 0; bx < xsize_blocks; ++bx) {
          const float* PIK_RESTRICT block_in = &row_in[bx * block_size];
          int16_t* PIK_RESTRICT block_out = &row_out[bx * block_size];
          quantizer.QuantizeBlockAC(row_quant[bx],
                                    ac_strategy_row[bx].GetQuantKind(), c,
                                    block_in, block_out);
        }
      }
    }
  }
}

PaddedBytes EncodeToBitstream(const EncCache& enc_cache, const Rect& rect,
                              const Quantizer& quantizer,
                              const NoiseParams& noise_params,
                              const ColorCorrelationMap& cmap, bool fast_mode,
                              MultipassHandler* handler, PikInfo* info) {
  PROFILER_FUNC;
  constexpr size_t N = kBlockDim;
  constexpr size_t block_size = N * N;
  PIK_ASSERT(quantizer.block_dim() == N);
  PIK_ASSERT(rect.x0() % kTileDim == 0);
  PIK_ASSERT(rect.xsize() % N == 0);
  PIK_ASSERT(rect.y0() % kTileDim == 0);
  PIK_ASSERT(rect.ysize() % N == 0);
  const size_t xsize_blocks = rect.xsize() / N;
  const size_t ysize_blocks = rect.ysize() / N;
  const size_t xsize_tiles = DivCeil(xsize_blocks, kTileDimInBlocks);
  const size_t ysize_tiles = DivCeil(ysize_blocks, kTileDimInBlocks);
  const Rect group_acs_qf_area_rect(rect.x0() / N, rect.y0() / N, xsize_blocks,
                                    ysize_blocks);
  const Rect tile_rect(rect.x0() / kTileDim, rect.y0() / kTileDim, xsize_tiles,
                       ysize_tiles);

  PikImageSizeInfo* ac_info = info ? &info->layers[kLayerAC] : nullptr;
  std::string noise_code = EncodeNoise(noise_params);

  const Rect ac_rect(8 * rect.x0(), rect.y0() / 8, 8 * rect.xsize(),
                     rect.ysize() / 8);

  int32_t order[kOrderContexts * block_size];
  std::vector<uint8_t> context_map;
  ComputeCoeffOrder(enc_cache.ac, ac_rect, order);

  std::string order_code = EncodeCoeffOrders(order, info);

  std::vector<std::vector<Token>> all_tokens(2);
  std::vector<Token>& ac_tokens = all_tokens[0];
  std::vector<Token>& ac_strategy_and_quant_field_tokens = all_tokens[1];

  for (size_t y = 0; y < ysize_tiles; y++) {
    for (size_t x = 0; x < xsize_tiles; x++) {
      const Rect tile_rect(x * kTileDimInBlocks, y * kTileDimInBlocks,
                           kTileDimInBlocks, kTileDimInBlocks, xsize_blocks,
                           ysize_blocks);
      TokenizeCoefficients(order, tile_rect, enc_cache.ac, &ac_tokens);
    }
  }

  TokenizeAcStrategy(group_acs_qf_area_rect, enc_cache.ac_strategy,
                     handler->HintAcStrategy(),
                     &ac_strategy_and_quant_field_tokens);

  TokenizeQuantField(group_acs_qf_area_rect, enc_cache.quant_field,
                     handler->HintQuantField(), enc_cache.ac_strategy,
                     &ac_strategy_and_quant_field_tokens);

  std::vector<ANSEncodingData> codes;
  std::string histo_code = "";
  if (fast_mode) {
    histo_code =
        BuildAndEncodeHistogramsFast(all_tokens, &codes, &context_map, ac_info);
  } else {
    histo_code = BuildAndEncodeHistograms(kNumContexts, all_tokens, &codes,
                                          &context_map, ac_info);
  }

  // TODO(user): consider either merging to AC or encoding separately.
  std::string ac_strategy_and_quant_field_code = WriteTokens(
      ac_strategy_and_quant_field_tokens, codes, context_map, ac_info);

  std::string ac_code = WriteTokens(ac_tokens, codes, context_map, ac_info);

  if (info) {
    info->layers[kLayerHeader].total_size += noise_code.size();
  }

  PaddedBytes out(noise_code.size() + order_code.size() + histo_code.size() +
                  ac_strategy_and_quant_field_code.size() + ac_code.size());
  size_t byte_pos = 0;
  Append(noise_code, &out, &byte_pos);
  Append(order_code, &out, &byte_pos);
  Append(histo_code, &out, &byte_pos);
  Append(ac_strategy_and_quant_field_code, &out, &byte_pos);
  Append(ac_code, &out, &byte_pos);

  // TODO(veluca): fix this with DC supergroups.
  float output_size_estimate = out.size() - ac_code.size() - histo_code.size();
  std::vector<std::array<size_t, 256>> counts(kNumContexts);
  size_t extra_bits = 0;
  for (const auto& token_list : all_tokens) {
    for (const auto& token : token_list) {
      counts[token.context][token.symbol]++;
      extra_bits += token.nbits;
    }
  }
  float entropy_coded_bits = 0;
  for (size_t ctx = 0; ctx < kNumContexts; ctx++) {
    size_t total =
        std::accumulate(counts[ctx].begin(), counts[ctx].end(), size_t(0));
    if (total == 0) continue;  // Prevent div by zero.
    double entropy = 0;
    for (size_t i = 0; i < 256; i++) {
      double p = 1.0 * counts[ctx][i] / total;
      if (p > 1e-4) {
        entropy -= p * std::log(p);
      }
    }
    entropy_coded_bits += entropy * total / std::log(2);
  }
  output_size_estimate +=
      static_cast<float>(extra_bits + entropy_coded_bits) / kBitsPerByte;
  if (info != nullptr) info->entropy_estimate = output_size_estimate;
  return out;
}

class Dequant {
 public:
  void Init(const ColorCorrelationMap& cmap, const Quantizer& quantizer) {
    dequant_matrices_ = quantizer.DequantMatrix(0, kQuantKindDCT8);
    inv_global_scale_ = quantizer.InvGlobalScale();
  }

  // Dequantizes and inverse color-transforms one tile, i.e. the window
  // `rect` (in block units) within the entire output image `dec_cache->ac`.
  // Reads the rect `rect16` (in block units) in `img_ac16`. Reads and write
  // only to the `block_group_rect` part of biases/ac_strategy/quant_field.
  SIMD_ATTR void DoAC(const Rect& rect16, const Image3S& img_ac16,
                      const Rect& rect, const Rect& block_group_rect,
                      const ImageI& img_ytox, const ImageI& img_ytob,
                      DecCache* PIK_RESTRICT dec_cache,
                      PassDecCache* PIK_RESTRICT pass_dec_cache) const {
    PIK_ASSERT(SameSize(rect, rect16));
    constexpr size_t N = kBlockDim;
    constexpr size_t block_size = N * N;
    const size_t xsize = rect.xsize();  // [blocks]
    const size_t ysize = rect.ysize();
    PIK_ASSERT(img_ac16.xsize() % block_size == 0);
    PIK_ASSERT(xsize <= img_ac16.xsize() / block_size);
    PIK_ASSERT(ysize <= img_ac16.ysize());
    PIK_ASSERT(SameSize(img_ytox, img_ytob));

    using D = SIMD_FULL(float);
    constexpr D d;
    constexpr SIMD_PART(int16_t, D::N) d16;
    constexpr SIMD_PART(int32_t, D::N) d32;

    // Rect representing the current tile inside the current group, in an image
    // in which each block is 1x1.
    const Rect block_tile_group_rect(block_group_rect.x0() + rect.x0(),
                                     block_group_rect.y0() + rect.y0(),
                                     rect.xsize(), rect.ysize());

    const size_t x0_cmap = rect.x0() / kColorTileDimInBlocks;
    const size_t y0_cmap = rect.y0() / kColorTileDimInBlocks;
    const size_t x0_dct = rect.x0() * block_size;
    const size_t x0_dct_group = block_tile_group_rect.x0() * block_size;
    const size_t x0_dct16 = rect16.x0() * block_size;

    for (size_t by = 0; by < ysize; ++by) {
      const int16_t* PIK_RESTRICT row_y16 =
          img_ac16.PlaneRow(1, by + rect16.y0()) + x0_dct16;
      const int* PIK_RESTRICT row_quant_field =
          block_tile_group_rect.ConstRow(pass_dec_cache->raw_quant_field, by);
      float* PIK_RESTRICT row_y =
          dec_cache->ac.PlaneRow(1, rect.y0() + by) + x0_dct;
      float* PIK_RESTRICT row_y_biases =
          pass_dec_cache->biases.PlaneRow(1, block_tile_group_rect.y0() + by) +
          x0_dct_group;
      AcStrategyRow ac_strategy_row =
          pass_dec_cache->ac_strategy.ConstRow(block_tile_group_rect, by);
      for (size_t bx = 0; bx < xsize; ++bx) {
        const auto scaled_dequant =
            set1(d, SafeDiv(inv_global_scale_, row_quant_field[bx]));

        size_t kind = ac_strategy_row[bx].GetQuantKind();
        for (size_t k = 0; k < block_size; k += d.N) {
          const size_t x = bx * block_size + k;

          // Y-channel quantization matrix for the given kind.
          const auto dequant = load(
              d,
              &dequant_matrices_[DequantMatrixOffset(0, kind, 1) * block_size] +
                  k);
          const auto y_mul = dequant * scaled_dequant;

          const auto quantized_y16 = load(d16, row_y16 + x);
          const auto quantized_y =
              convert_to(d, convert_to(d32, quantized_y16));

          const auto debiased_y = AdjustQuantBias<1>(quantized_y);
          store(quantized_y - debiased_y, d, row_y_biases + x);
          const auto dequant_y = debiased_y * y_mul;
          store(dequant_y, d, row_y + x);
        }
      }
    }

    for (int c = 0; c < 3; c += 2) {  // === for c in {0, 2}
      const ImageI& img_cmap = (c == 0) ? img_ytox : img_ytob;
      for (size_t by = 0; by < ysize; ++by) {
        const size_t ty = by / kColorTileDimInBlocks;
        const int16_t* PIK_RESTRICT row_xb16 =
            img_ac16.PlaneRow(c, by + rect16.y0()) + x0_dct16;
        const int* PIK_RESTRICT row_quant_field =
            block_tile_group_rect.ConstRow(pass_dec_cache->raw_quant_field, by);
        const int* PIK_RESTRICT row_cmap =
            img_cmap.ConstRow(y0_cmap + ty) + x0_cmap;
        const float* PIK_RESTRICT row_y =
            dec_cache->ac.ConstPlaneRow(1, rect.y0() + by) + x0_dct;
        float* PIK_RESTRICT row_xb =
            dec_cache->ac.PlaneRow(c, rect.y0() + by) + x0_dct;
        float* PIK_RESTRICT row_xb_biases =
            pass_dec_cache->biases.PlaneRow(c,
                                            block_tile_group_rect.y0() + by) +
            x0_dct_group;

        AcStrategyRow ac_strategy_row =
            pass_dec_cache->ac_strategy.ConstRow(block_tile_group_rect, by);
        for (size_t bx = 0; bx < xsize; ++bx) {
          const auto scaled_dequant =
              set1(d, SafeDiv(inv_global_scale_, row_quant_field[bx]));

          size_t kind = ac_strategy_row[bx].GetQuantKind();
          const float* dequant_matrix =
              &dequant_matrices_[DequantMatrixOffset(0, kind, c) * block_size];
          const size_t tx = bx / kColorTileDimInBlocks;
          const int32_t cmap = row_cmap[tx];

          if (c == 0) {
            const auto y_mul = set1(d, ColorCorrelationMap::YtoX(1.0f, cmap));

            for (size_t k = 0; k < block_size; k += d.N) {
              const size_t x = bx * block_size + k;

              const auto xb_mul = load(d, dequant_matrix + k) * scaled_dequant;

              const auto quantized_xb16 = load(d16, row_xb16 + x);
              const auto quantized_xb =
                  convert_to(d, convert_to(d32, quantized_xb16));

              const auto out_y = load(d, row_y + x);

              const auto debiased_xb = AdjustQuantBias<0>(quantized_xb);
              store(quantized_xb - debiased_xb, d, row_xb_biases + x);
              const auto dequant_xb = debiased_xb * xb_mul;

              store(mul_add(y_mul, out_y, dequant_xb), d, row_xb + x);
            }
          } else {
            const auto y_mul = set1(d, ColorCorrelationMap::YtoB(1.0f, cmap));

            for (size_t k = 0; k < block_size; k += d.N) {
              const size_t x = bx * block_size + k;

              const auto xb_mul = load(d, dequant_matrix + k) * scaled_dequant;

              const auto quantized_xb16 = load(d16, row_xb16 + x);
              const auto quantized_xb =
                  convert_to(d, convert_to(d32, quantized_xb16));

              const auto out_y = load(d, row_y + x);

              const auto debiased_xb = AdjustQuantBias<2>(quantized_xb);
              store(quantized_xb - debiased_xb, d, row_xb_biases + x);
              const auto dequant_xb = debiased_xb * xb_mul;

              store(mul_add(y_mul, out_y, dequant_xb), d, row_xb + x);
            }
          }
        }
      }
    }
  }

 private:
  static PIK_INLINE float SafeDiv(float num, int32_t div) {
    return div == 0 ? 1E10f : num / div;
  }

  // AC dequant
  const float* PIK_RESTRICT dequant_matrices_;
  float inv_global_scale_;
};

// Temporary storage; one per thread, for one tile.
struct DecoderBuffers {
  void InitOnce() {
    constexpr size_t N = kBlockDim;
    constexpr size_t block_size = N * N;
    // This thread already allocated its buffers.
    if (num_nzeroes.xsize() != 0) return;

    // Allocate enough for a whole tile - partial tiles on the right/bottom
    // border just use a subset. The valid size is passed via Rect.
    const size_t xsize_blocks = kTileDimInBlocks;
    const size_t ysize_blocks = kTileDimInBlocks;

    quantized_ac = Image3S(xsize_blocks * block_size, ysize_blocks);

    num_nzeroes = Image3I(xsize_blocks, ysize_blocks);
  }

  // Decode
  Image3S quantized_ac;

  // DequantAC
  Image3I num_nzeroes;
};

bool DecodeCoefficientsAndDequantize(
    const PassHeader& pass_header, const GroupHeader& header,
    const Rect& group_rect, MultipassHandler* handler,
    const size_t xsize_blocks, const size_t ysize_blocks,
    const PaddedBytes& compressed, BitReader* reader,
    const ColorCorrelationMap& cmap, DecCache* dec_cache,
    PassDecCache* pass_dec_cache, const Quantizer& quantizer) {
  PROFILER_FUNC;
  constexpr size_t N = kBlockDim;
  constexpr size_t block_size = N * N;

  PIK_ASSERT(group_rect.x0() % kBlockDim == 0);
  PIK_ASSERT(group_rect.y0() % kBlockDim == 0);
  const size_t x0_blocks = DivCeil(group_rect.x0(), kBlockDim);
  const size_t y0_blocks = DivCeil(group_rect.y0(), kBlockDim);
  const Rect group_acs_qf_rect(x0_blocks, y0_blocks, xsize_blocks,
                               ysize_blocks);

  const size_t xsize_tiles = DivCeil(xsize_blocks, kTileDimInBlocks);
  const size_t ysize_tiles = DivCeil(ysize_blocks, kTileDimInBlocks);
  const size_t num_tiles = xsize_tiles * ysize_tiles;

  DecoderBuffers tmp;
  tmp.InitOnce();

  int coeff_order[kOrderContexts * block_size];
  for (size_t c = 0; c < kOrderContexts; ++c) {
    DecodeCoeffOrder(&coeff_order[c * block_size], reader);
  }
  PIK_RETURN_IF_ERROR(reader->JumpToByteBoundary());

  ANSCode code;
  std::vector<uint8_t> context_map;
  // Histogram data size is small and does not require parallelization.
  PIK_RETURN_IF_ERROR(
      DecodeHistograms(reader, kNumContexts, 256, &code, &context_map));
  PIK_RETURN_IF_ERROR(reader->JumpToByteBoundary());

  ANSSymbolReader ac_strategy_and_quant_field_decoder(&code);
  ANSSymbolReader strategy_decoder(&code);
  if (!DecodeAcStrategy(reader, &ac_strategy_and_quant_field_decoder,
                        context_map, group_acs_qf_rect,
                        &pass_dec_cache->ac_strategy,
                        handler->HintAcStrategy())) {
    return PIK_FAILURE("Failed to decode AcStrategy.");
  }

  if (!DecodeQuantField(
          reader, &ac_strategy_and_quant_field_decoder, context_map,
          group_acs_qf_rect, pass_dec_cache->ac_strategy,
          &pass_dec_cache->raw_quant_field, handler->HintQuantField())) {
    return PIK_FAILURE("Failed to decode QuantField.");
  }
  if (!ac_strategy_and_quant_field_decoder.CheckANSFinalState()) {
    return PIK_FAILURE("QuantField: ANS checksum failure.");
  }
  PIK_RETURN_IF_ERROR(reader->JumpToByteBoundary());

  Dequant dequant;
  dequant.Init(cmap, quantizer);
  dec_cache->ac = Image3F(xsize_blocks * block_size, ysize_blocks);

  ANSSymbolReader ac_decoder(&code);
  for (size_t task = 0; task < num_tiles; ++task) {
    const size_t tile_x = task % xsize_tiles;
    const size_t tile_y = task / xsize_tiles;
    const Rect rect(tile_x * kTileDimInBlocks, tile_y * kTileDimInBlocks,
                    kTileDimInBlocks, kTileDimInBlocks, xsize_blocks,
                    ysize_blocks);
    const Rect quantized_rect(0, 0, rect.xsize(), rect.ysize());

    if (!DecodeAC(context_map, coeff_order, reader, &ac_decoder,
                  &tmp.quantized_ac, rect, &tmp.num_nzeroes)) {
      return PIK_FAILURE("Failed to decode AC.");
    }

    dequant.DoAC(quantized_rect, tmp.quantized_ac, rect, group_acs_qf_rect,
                 cmap.ytox_map, cmap.ytob_map, dec_cache, pass_dec_cache);
  }
  if (!ac_decoder.CheckANSFinalState()) {
    return PIK_FAILURE("ANS checksum failure.");
  }
  PIK_RETURN_IF_ERROR(reader->JumpToByteBoundary());

  return true;
}

bool DecodeFromBitstream(const PassHeader& pass_header,
                         const GroupHeader& header,
                         const PaddedBytes& compressed, BitReader* reader,
                         const Rect& group_rect, MultipassHandler* handler,
                         const size_t xsize_blocks, const size_t ysize_blocks,
                         const ColorCorrelationMap& cmap,
                         NoiseParams* noise_params, const Quantizer& quantizer,
                         DecCache* dec_cache, PassDecCache* pass_dec_cache) {
  PIK_RETURN_IF_ERROR(DecodeNoise(reader, noise_params));

  return DecodeCoefficientsAndDequantize(
      pass_header, header, group_rect, handler, xsize_blocks, ysize_blocks,
      compressed, reader, cmap, dec_cache, pass_dec_cache, quantizer);
}

void DequantImageAC(const Quantizer& quantizer, const ColorCorrelationMap& cmap,
                    const Image3S& quantized_ac, ThreadPool* pool,
                    DecCache* dec_cache, PassDecCache* pass_dec_cache,
                    const Rect& group_rect) {
  PROFILER_ZONE("dequant");
  constexpr size_t N = kBlockDim;
  constexpr size_t block_size = N * N;

  // Caller must have allocated/filled quantized_dc/ac.
  PIK_CHECK(quantized_ac.xsize() ==
                quantizer.RawQuantField().xsize() * block_size &&
            quantized_ac.ysize() == quantizer.RawQuantField().ysize());

  const size_t xsize_blocks = quantizer.RawQuantField().xsize();
  const size_t ysize_blocks = quantizer.RawQuantField().ysize();
  const size_t xsize_tiles = DivCeil(xsize_blocks, kTileDimInBlocks);
  const size_t ysize_tiles = DivCeil(ysize_blocks, kTileDimInBlocks);

  Dequant dequant;
  dequant.Init(cmap, quantizer);
  dec_cache->ac = Image3F(xsize_blocks * block_size, ysize_blocks);

  std::vector<DecoderBuffers> decoder_buf(NumThreads(pool));

  PIK_ASSERT(group_rect.x0() % kBlockDim == 0 &&
             group_rect.y0() % kBlockDim == 0 &&
             group_rect.xsize() % kBlockDim == 0 &&
             group_rect.ysize() % kBlockDim == 0);

  const Rect block_group_rect(
      group_rect.x0() / kBlockDim, group_rect.y0() / kBlockDim,
      group_rect.xsize() / kBlockDim, group_rect.ysize() / kBlockDim);

  const size_t num_tiles = xsize_tiles * ysize_tiles;
  const auto dequant_tile = [&](const int task, const int thread) {
    DecoderBuffers& tmp = decoder_buf[thread];
    tmp.InitOnce();

    const size_t tile_x = task % xsize_tiles;
    const size_t tile_y = task / xsize_tiles;
    const Rect rect(tile_x * kTileDimInBlocks, tile_y * kTileDimInBlocks,
                    kTileDimInBlocks, kTileDimInBlocks, xsize_blocks,
                    ysize_blocks);

    dequant.DoAC(rect, quantized_ac, rect, block_group_rect, cmap.ytox_map,
                 cmap.ytob_map, dec_cache, pass_dec_cache);
  };
  RunOnPool(pool, 0, num_tiles, dequant_tile, "DequantImage");
}

static SIMD_ATTR void InverseIntegralTransform(
    const size_t xsize_blocks, const size_t ysize_blocks,
    const Image3F& ac_image, const AcStrategyImage& ac_strategy,
    const Rect& acs_rect, Image3F* PIK_RESTRICT idct) {
  PROFILER_ZONE("IDCT");

  constexpr size_t N = kBlockDim;
  constexpr size_t block_size = N * N;
  const size_t idct_stride = idct->PixelsPerRow();

  for (int c = 0; c < 3; ++c) {
    const ImageF& ac_plane = ac_image.Plane(c);
    const size_t ac_per_row = ac_plane.PixelsPerRow();
    ImageF* PIK_RESTRICT idct_plane = idct->MutablePlane(c);

    for (size_t by = 0; by < ysize_blocks; ++by) {
      const float* PIK_RESTRICT ac_row = ac_plane.ConstRow(by);
      const AcStrategyRow& acs_row = ac_strategy.ConstRow(acs_rect, by);
      float* PIK_RESTRICT idct_row = idct_plane->Row(by * N);

      for (size_t bx = 0; bx < xsize_blocks; ++bx) {
        const float* PIK_RESTRICT ac_pos = ac_row + bx * block_size;
        const AcStrategy& acs = acs_row[bx];
        float* PIK_RESTRICT idct_pos = idct_row + bx * N;

        acs.TransformToPixels(ac_pos, ac_per_row, idct_pos, idct_stride);
      }
    }
  }
}

Image3F ReconOpsinImage(const PassHeader& pass_header,
                        const GroupHeader& header, const Quantizer& quantizer,
                        const Rect& block_group_rect, DecCache* dec_cache,
                        PassDecCache* pass_dec_cache, PikInfo* pik_info) {
  PROFILER_ZONE("ReconOpsinImage");
  constexpr size_t N = kBlockDim;
  const size_t xsize_blocks = block_group_rect.xsize();
  const size_t ysize_blocks = block_group_rect.ysize();
  const size_t xsize_tiles = DivCeil(xsize_blocks, kTileDimInBlocks);
  const size_t ysize_tiles = DivCeil(ysize_blocks, kTileDimInBlocks);
  const bool predict_lf = pass_header.predict_lf;
  const bool predict_hf = pass_header.predict_hf;

  // TODO(veluca): this should probably happen upon dequantization of DC. Also,
  // we should consider doing something similar for AC.
  if (pass_header.flags & PassHeader::kGrayscaleOpt) {
    PROFILER_ZONE("GrayscaleRestoreXB");
    kGrayXyb->RestoreXB(&dec_cache->dc);
  }

  if (pik_info && pik_info->testing_aux.ac_prediction != nullptr) {
    PROFILER_ZONE("Copy ac_prediction");
    *pik_info->testing_aux.ac_prediction = CopyImage(dec_cache->ac);
  }

  // Sets dcoeffs.0 from DC (for DCT blocks) and updates HVD.
  // TODO(user): do not allocate when !predict_hf
  Image3F pred2x2(dec_cache->dc.xsize() * 2, dec_cache->dc.ysize() * 2);
  Image3F* PIK_RESTRICT ac64 = &dec_cache->ac;

  // Currently llf is temporary storage, but it will be more persistent
  // in tile-wise processing.
  Image3F llf(xsize_blocks + 2, ysize_blocks + 2);
  ComputeLlf(dec_cache->dc, pass_dec_cache->ac_strategy, block_group_rect,
             &llf);

  Image3F lf2x2;
  if (predict_lf) {
    lf2x2 = Image3F((xsize_blocks + 2) * 2, (ysize_blocks + 2) * 2);
    // dc2x2 plane is borrowed for temporary storage.
    PredictLf(pass_dec_cache->ac_strategy, block_group_rect, llf,
              pred2x2.MutablePlane(0), &lf2x2);
  }
  FillImage(0.0f, &pred2x2);

  // Compute the border of pred2x2.
  if (predict_hf) {
    AcStrategy acs(AcStrategy::Type::DCT, 0);
    const size_t pred2x2_stride = pred2x2.PixelsPerRow();
    if (predict_lf) {
      const size_t lf2x2_stride = lf2x2.PixelsPerRow();
      float block[N * N] = {};
      for (size_t c = 0; c < 3; c++) {
        for (size_t x : {0UL, xsize_blocks + 1}) {
          for (size_t y = 0; y < ysize_blocks + 2; y++) {
            const float* row_llf = llf.ConstPlaneRow(c, y);
            const float* row_lf2x2 = lf2x2.ConstPlaneRow(c, 2 * y);
            float* row_pred2x2 = pred2x2.PlaneRow(c, 2 * y);
            block[0] = row_llf[x];
            block[1] = row_lf2x2[2 * x + 1];
            block[N] = row_lf2x2[lf2x2_stride + 2 * x];
            block[N + 1] = row_lf2x2[lf2x2_stride + 2 * x + 1];
            acs.DC2x2FromLowFrequencies(block, 0 /*not used*/,
                                        row_pred2x2 + 2 * x, pred2x2_stride);
          }
        }
        for (size_t y : {0UL, ysize_blocks + 1}) {
          const float* row_llf = llf.ConstPlaneRow(c, y);
          const float* row_lf2x2 = lf2x2.ConstPlaneRow(c, 2 * y);
          float* row_pred2x2 = pred2x2.PlaneRow(c, 2 * y);
          for (size_t x = 0; x < xsize_blocks + 2; x++) {
            block[0] = row_llf[x];
            block[1] = row_lf2x2[2 * x + 1];
            block[N] = row_lf2x2[lf2x2_stride + 2 * x];
            block[N + 1] = row_lf2x2[lf2x2_stride + 2 * x + 1];
            acs.DC2x2FromLowFrequencies(block, 0 /*not used*/,
                                        row_pred2x2 + 2 * x, pred2x2_stride);
          }
        }
      }
    } else {
      const size_t llf_stride = llf.PixelsPerRow();
      for (size_t c = 0; c < 3; c++) {
        for (size_t x : {0UL, xsize_blocks + 1}) {
          for (size_t y = 0; y < ysize_blocks + 2; y++) {
            const float* row_llf = llf.ConstPlaneRow(c, y);
            float* row_pred2x2 = pred2x2.PlaneRow(c, 2 * y);
            acs.DC2x2FromLowestFrequencies(row_llf + x, llf_stride,
                                           row_pred2x2 + 2 * x, pred2x2_stride);
          }
        }
        for (size_t y : {0UL, ysize_blocks + 1}) {
          const float* row_llf = llf.ConstPlaneRow(c, y);
          float* row_pred2x2 = pred2x2.PlaneRow(c, 2 * y);
          for (size_t x = 0; x < xsize_blocks + 2; x++) {
            acs.DC2x2FromLowestFrequencies(row_llf + x, llf_stride,
                                           row_pred2x2 + 2 * x, pred2x2_stride);
          }
        }
      }
    }
  }

  // tile_stage is used to make calculation dispatching simple; each pixel
  // corresponds to tile. Each bit corresponds to stage:
  // * 0-th bit for calculation or lf2x2 / pred2x2 & initial LF AC update;
  ImageB tile_stage(xsize_tiles + 1, ysize_tiles + 1);

  for (size_t c = 0; c < dec_cache->ac.kNumPlanes; c++) {
    // Reset tile stages.
    for (size_t ty = 0; ty < ysize_tiles + 1; ++ty) {
      uint8_t* tile_stage_row = tile_stage.Row(ty);
      for (size_t tx = 0; tx < xsize_tiles + 1; ++tx) {
        bool is_sentinel = (ty >= ysize_tiles) || (tx >= xsize_tiles);
        tile_stage_row[tx] = is_sentinel ? 255 : 0;
      }
    }

    const ImageF& llf_plane = llf.Plane(c);
    ImageF* ac64_plane = ac64->MutablePlane(c);
    ImageF* pred2x2_plane = predict_hf ? pred2x2.MutablePlane(c) : nullptr;
    ImageF* lf2x2_plane = predict_lf ? lf2x2.MutablePlane(c) : nullptr;
    for (size_t ty = 0; ty < ysize_tiles; ++ty) {
      for (size_t tx = 0; tx < xsize_tiles; ++tx) {
        for (size_t lfty = ty; lfty < ty + 2; ++lfty) {
          uint8_t* tile_stage_row = tile_stage.Row(lfty);
          for (size_t lftx = tx; lftx < tx + 2; ++lftx) {
            if ((tile_stage_row[lftx] & 1) != 0) continue;
            const Rect tile(lftx * kTileDimInBlocks, lfty * kTileDimInBlocks,
                            kTileDimInBlocks, kTileDimInBlocks, xsize_blocks,
                            ysize_blocks);
            UpdateLfForDecoder(tile, predict_lf, predict_hf,
                               pass_dec_cache->ac_strategy, block_group_rect,
                               llf_plane, ac64_plane, pred2x2_plane,
                               lf2x2_plane);
            tile_stage_row[lftx] |= 1;
          }
        }
        if (predict_hf) {
          // TODO(user): invoke AddPredictions for (tx, ty) tile here.
        }
      }
    }
  }

  if (predict_hf) {
    // TODO(user): make UpSample4x4BlurDCT tile-wise-able.
    AddPredictions(pred2x2, pass_dec_cache->ac_strategy, block_group_rect,
                   &dec_cache->ac);
  }

  Image3F idct(xsize_blocks * N, ysize_blocks * N);
  InverseIntegralTransform(xsize_blocks, ysize_blocks, dec_cache->ac,
                           pass_dec_cache->ac_strategy, block_group_rect,
                           &idct);

  if (pik_info && pik_info->testing_aux.ac_prediction != nullptr) {
    PROFILER_ZONE("Subtract ac_prediction");
    Subtract(dec_cache->ac, *pik_info->testing_aux.ac_prediction,
             pik_info->testing_aux.ac_prediction);
    ZeroDcValues(pik_info->testing_aux.ac_prediction);
  }

  return idct;
}

Image3F FinalizePassDecoding(Image3F&& idct, const PassHeader& pass_header,
                             const Quantizer& quantizer,
                             PassDecCache* pass_dec_cache, PikInfo* pik_info) {
  Image3F copy;
  const Image3F* non_smoothed = &idct;
  if (pass_header.gaborish != GaborishStrength::kOff) {
    // AdaptiveReconstruction needs a copy before smoothing.
    // TODO(janwas): remove
    if (pass_header.have_adaptive_reconstruction) {
      PROFILER_ZONE("CopyForAR");
      copy = CopyImage(idct);
      non_smoothed = &copy;

      idct = ConvolveGaborish(std::move(idct), pass_header.gaborish,
                              /*pool=*/nullptr);
    }
  }

  if (pass_header.have_adaptive_reconstruction) {
    AdaptiveReconstructionAux* ar_aux =
        pik_info ? &pik_info->adaptive_reconstruction_aux : nullptr;
    idct = AdaptiveReconstruction(
        &idct, *non_smoothed, quantizer, pass_dec_cache->raw_quant_field,
        pass_dec_cache->ac_strategy, pass_dec_cache->biases,
        pass_header.epf_params, ar_aux);
  }

  idct = ConvolveGaborish(std::move(idct), pass_header.gaborish,
                          /*pool=*/nullptr);
  return std::move(idct);
}

}  // namespace pik
