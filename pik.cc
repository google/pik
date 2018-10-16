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

#include "pik.h"

#include <limits.h>  // PATH_MAX
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#define PROFILER_ENABLED 1
#include "adaptive_quantization.h"
#include "adaptive_reconstruction.h"
#include "arch_specific.h"
#include "bits.h"
#include "brunsli_v2_decode.h"
#include "brunsli_v2_encode.h"
#include "butteraugli_comparator.h"
#include "byte_order.h"
#include "common.h"
#include "compiler_specific.h"
#include "container.h"
#include "context.h"
#include "convolve.h"
#include "dct.h"
#include "dct_util.h"
#include "entropy_coder.h"
#include "external_image.h"
#include "fast_log.h"
#include "gamma_correct.h"
#include "guetzli/jpeg_data.h"
#include "guetzli/jpeg_data_decoder.h"
#include "image_io.h"
#include "jpeg_quant_tables.h"
#include "opsin_image.h"
#include "opsin_inverse.h"
#include "pik_alpha.h"
#include "profiler.h"
#include "resize.h"
#include "simd/targets.h"

bool FLAGS_log_search_state = false;
// If true, prints the quantization maps at each iteration.
bool FLAGS_dump_quant_state = false;

namespace pik {
namespace {

// True if we should try to find a non-trivial AC strategy.
const constexpr bool kChooseAcStrategy = false;

// Quantization weights for ID-coded blocks.
static const float kQuant64Identity[64] = {
    0.7,
    0.7,
    0.7,
    0.7,
    0.7,
    0.588994181958768,
    0.637412731903807,
    0.658265403732354,
    0.664184545211087,
    0.489557940162741,
    0.441809196703547,
    0.466519636930879,
    0.508335971631046,
    0.501624976604618,
    0.447903166596090,
    0.487108568611555,
    0.457498177649116,
    0.452848682971061,
    0.490734792977206,
    0.443057415317534,
    0.392141167751584,
    0.358316869050728,
    0.418207067977596,
    0.467858769794496,
    0.443549876397044,
    0.427476997048851,
    0.356915568577037,
    0.303161982522844,
    0.363256217129902,
    0.352611929358839,
    0.418999909922867,
    0.398691116916629,
    0.377526872102006,
    0.352326109479318,
    0.308206968728254,
    0.280093699425257,
    0.304147527622546,
    0.359202623799680,
    0.339645164723791,
    0.349585237941005,
    0.386203903613304,
    0.353582036654603,
    0.305955548639682,
    0.365259530060446,
    0.333159510814334,
    0.363133568767434,
    0.334161790012618,
    0.389194124900511,
    0.349326306148990,
    0.390310895605386,
    0.408666924454222,
    0.335930464190049,
    0.359313000261458,
    0.381109877480420,
    0.392933763109596,
    0.359529015172913,
    0.347676628893596,
    0.370974565818013,
    0.350361463992334,
    0.338064798002449,
    0.336743523710490,
    0.296631529585931,
    0.304517245589665,
    0.302956514467806,
};

// Quantization weights for DCT-coded blocks.
static const float kQuant64Dct[64] = {
    0.0,
    0.7,
    0.7,
    0.7,
    0.7,
    0.588994181958768,
    0.637412731903807,
    0.658265403732354,
    0.664184545211087,
    0.489557940162741,
    0.441809196703547,
    0.466519636930879,
    0.508335971631046,
    0.501624976604618,
    0.447903166596090,
    0.487108568611555,
    0.457498177649116,
    0.452848682971061,
    0.490734792977206,
    0.443057415317534,
    0.392141167751584,
    0.358316869050728,
    0.418207067977596,
    0.467858769794496,
    0.443549876397044,
    0.427476997048851,
    0.356915568577037,
    0.303161982522844,
    0.363256217129902,
    0.352611929358839,
    0.418999909922867,
    0.398691116916629,
    0.377526872102006,
    0.352326109479318,
    0.308206968728254,
    0.280093699425257,
    0.304147527622546,
    0.359202623799680,
    0.339645164723791,
    0.349585237941005,
    0.386203903613304,
    0.353582036654603,
    0.305955548639682,
    0.365259530060446,
    0.333159510814334,
    0.363133568767434,
    0.334161790012618,
    0.389194124900511,
    0.349326306148990,
    0.390310895605386,
    0.408666924454222,
    0.335930464190049,
    0.359313000261458,
    0.381109877480420,
    0.392933763109596,
    0.359529015172913,
    0.347676628893596,
    0.370974565818013,
    0.350361463992334,
    0.338064798002449,
    0.336743523710490,
    0.296631529585931,
    0.304517245589665,
    0.302956514467806,
};

// Can't rely on C++14 yet.
template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

static float GetIntensityMultiplier(const CompressParams& params) {
  return params.intensity_target * kIntensityMultiplier;
}

template <class Image>
size_t TargetSize(const CompressParams& params, const Image& img) {
  if (params.target_size > 0) {
    return params.target_size;
  }
  if (params.target_bitrate > 0.0) {
    return 0.5 + params.target_bitrate * img.xsize() * img.ysize() / 8;
  }
  return 0;
}

ImageF TileDistMap(const ImageF& distmap, int tile_size, int margin) {
  PROFILER_FUNC;
  const int tile_xsize = (distmap.xsize() + tile_size - 1) / tile_size;
  const int tile_ysize = (distmap.ysize() + tile_size - 1) / tile_size;
  ImageF tile_distmap(tile_xsize, tile_ysize);
  for (int tile_y = 0; tile_y < tile_ysize; ++tile_y) {
    int y_min = std::max<int>(0, tile_size * tile_y - margin);
    int y_max =
        std::min<int>(distmap.ysize(), tile_size * (tile_y + 1) + margin);
    for (int tile_x = 0; tile_x < tile_xsize; ++tile_x) {
      int x_min = std::max<int>(0, tile_size * tile_x - margin);
      int x_max =
          std::min<int>(distmap.xsize(), tile_size * (tile_x + 1) + margin);
      float max_dist = 0.0;
      for (int y = y_min; y < y_max; ++y) {
        float ymul = 1.0;
        static const float kBorderMul = 0.98f;
        static const float kCornerMul = 0.7f;
        if (margin != 0 && (y == y_min || y == y_max - 1)) {
          ymul = kBorderMul;
        }
        const float* const PIK_RESTRICT row = distmap.Row(y);
        for (int x = x_min; x < x_max; ++x) {
          float xmul = ymul;
          if (margin != 0 && (x == x_min || x == x_max - 1)) {
            if (xmul == 1.0) {
              xmul = kBorderMul;
            } else {
              xmul = kCornerMul;
            }
          }
          float v = xmul * row[x];
          max_dist = std::max(max_dist, v);
        }
      }
      tile_distmap.Row(tile_y)[tile_x] = max_dist;
    }
  }
  return tile_distmap;
}

ImageF DistToPeakMap(const ImageF& field, float peak_min, int local_radius,
                     float peak_weight) {
  ImageF result(field.xsize(), field.ysize());
  FillImage(-1.0f, &result);
  for (int y0 = 0; y0 < field.ysize(); ++y0) {
    for (int x0 = 0; x0 < field.xsize(); ++x0) {
      int x_min = std::max(0, x0 - local_radius);
      int y_min = std::max(0, y0 - local_radius);
      int x_max = std::min<int>(field.xsize(), x0 + 1 + local_radius);
      int y_max = std::min<int>(field.ysize(), y0 + 1 + local_radius);
      float local_max = peak_min;
      for (int y = y_min; y < y_max; ++y) {
        for (int x = x_min; x < x_max; ++x) {
          local_max = std::max(local_max, field.Row(y)[x]);
        }
      }
      if (field.Row(y0)[x0] >
          (1.0f - peak_weight) * peak_min + peak_weight * local_max) {
        for (int y = y_min; y < y_max; ++y) {
          for (int x = x_min; x < x_max; ++x) {
            float dist = std::max(std::abs(y - y0), std::abs(x - x0));
            float cur_dist = result.Row(y)[x];
            if (cur_dist < 0.0 || cur_dist > dist) {
              result.Row(y)[x] = dist;
            }
          }
        }
      }
    }
  }
  return result;
}

bool AdjustQuantVal(float* const PIK_RESTRICT q, const float d,
                    const float factor, const float quant_max) {
  if (*q >= 0.999f * quant_max) return false;
  const float inv_q = 1.0f / *q;
  const float adj_inv_q = inv_q - factor / (d + 1.0f);
  *q = 1.0f / std::max(1.0f / quant_max, adj_inv_q);
  return true;
}

void DumpHeatmap(const PikInfo* info, const std::string& label,
                 const ImageF& image, float good_threshold,
                 float bad_threshold) {
  Image3B heatmap =
      butteraugli::CreateHeatMapImage(image, good_threshold, bad_threshold);
  char filename[200];
  snprintf(filename, sizeof(filename), "%s%05d", label.c_str(),
           info->num_butteraugli_iters);
  info->DumpImage(filename, heatmap);
}

void DumpHeatmaps(const PikInfo* info, float ba_target,
                  const ImageF& quant_field, const ImageF& tile_heatmap) {
  if (!WantDebugOutput(info)) return;
  ImageF inv_qmap(quant_field.xsize(), quant_field.ysize());
  for (size_t y = 0; y < quant_field.ysize(); ++y) {
    const float* PIK_RESTRICT row_q = quant_field.ConstRow(y);
    float* PIK_RESTRICT row_inv_q = inv_qmap.Row(y);
    for (size_t x = 0; x < quant_field.xsize(); ++x) {
      row_inv_q[x] = 1.0f / row_q[x];  // never zero
    }
  }
  DumpHeatmap(info, "quant_heatmap", inv_qmap, 4.0f * ba_target,
              6.0f * ba_target);
  DumpHeatmap(info, "tile_heatmap", tile_heatmap, ba_target, 1.5f * ba_target);
}

SIMD_ATTR void FindBestAcStrategy(float butteraugli_target, ThreadPool* pool,
                                  EncCache* cache) {
  cache->ac_strategy.layers = ImageB(cache->xsize_blocks, cache->ysize_blocks);
  if (kChooseAcStrategy) {
    const auto find_block_strategy = [butteraugli_target, cache](int bx,
                                                                 int by) {
      // TODO(user): parameter/function optimization. Consider other choice
      // strategies.
      constexpr float kIdWeight = 1.5e-1;
      const float k16x16Weight = 4 / std::sqrt(butteraugli_target);
      constexpr float kWeights[3] = {1. / kXybRange[0], 1. / kXybRange[1],
                                     1. / kXybRange[2]};
      constexpr float kL2Weight = 1.0;
      constexpr float kL4Weight = -1.0;
      constexpr float kDiscretizationFactor = 256;
      const auto estimate_entropy = [kWeights](const Image3F* img,
                                               size_t block_width,
                                               size_t block_height, int bx,
                                               int by, const float* kQuant) {
        float l2 = 0;
        float l4 = 0;
        for (size_t c = 0; c < img->kNumPlanes; c++) {
          size_t k = 0;
          for (size_t iy = 0; iy < block_height; iy++) {
            const float* row = img->ConstPlaneRow(c, by * block_height + iy);
            for (size_t ix = 0; ix < block_width; ix++) {
              float val = row[bx * block_width + ix] * kWeights[c] * kQuant[k];
              val *= val;
              l2 += val;
              val *= val;
              l4 += val;
              k++;
            }
          }
        }
        l2 = std::sqrt(l2);
        l4 = std::pow(l4, 0.25);
        return kL2Weight * l2 + kL4Weight * l4;
      };
      float dct_entropy = estimate_entropy(
          &cache->coeffs_init, kBlockDim * kBlockDim, 1, bx, by, kQuant64Dct);
      float id_entropy = estimate_entropy(cache->src, kBlockDim, kBlockDim, bx,
                                          by, kQuant64Identity);
      if (kIdWeight * id_entropy < dct_entropy) {
        return AcStrategyType::IDENTITY;
      }
      if (bx + 1 < cache->xsize_blocks && by + 1 < cache->ysize_blocks &&
          bx % 2 == 0 && by % 2 == 0) {
        float dct8x8_entropy = 0;
        for (size_t c = 0; c < cache->coeffs_init.kNumPlanes; c++) {
          for (size_t iy = 0; iy < 2; iy++) {
            const float* row = cache->coeffs_init.ConstPlaneRow(c, by + iy);
            for (size_t ix = 0; ix < kBlockDim * kBlockDim * 2; ix++) {
              float val = row[bx * kBlockDim * kBlockDim + ix] * kWeights[c];
              int v = fabsf(val) * kDiscretizationFactor;
              if (v) dct8x8_entropy += log2(v);
            }
          }
        }
        float dct16x16_entropy = 0;
        for (size_t c = 0; c < cache->src->kNumPlanes; c++) {
          SIMD_ALIGN float dct16x16[4 * kBlockDim * kBlockDim] = {};
          ComputeTransposedScaledDCT<2 * kBlockDim>(
              FromLines(
                  cache->src->PlaneRow(c, kBlockDim * by) + kBlockDim * bx,
                  cache->src->PixelsPerRow()),
              ScaleToBlock<2 * kBlockDim>(dct16x16));
          for (size_t k = 0; k < 4 * kBlockDim * kBlockDim; k++) {
            // TODO(user): meaningful kQuant estimate.
            float val = dct16x16[k] * kWeights[c];
            int v = fabsf(val) * kDiscretizationFactor;
            if (v) dct16x16_entropy += log2(v);
          }
          /*for (size_t iy = 0; iy < 2 * kBlockDim; iy++) {
            for (size_t ix = 0; ix < 2 * kBlockDim; ix++) {
              float val = dct16x16[iy * 2 * kBlockDim + ix];
              val = fabsf(val);
              fprintf(stderr, "%02x ", int(val / 0.02 * 16));
            }
            fprintf(stderr, "\n");
          }
          fprintf(stderr, "vs dct8:\n");
          for (size_t iy = 0; iy < 2 * kBlockDim; iy++) {
            for (size_t ix = 0; ix < 2 * kBlockDim; ix++) {
              size_t byp = by + iy / kBlockDim;
              size_t bxp = bx + ix / kBlockDim;
              size_t oy = iy % kBlockDim;
              size_t ox = ix % kBlockDim;
              float val = cache->coeffs_init.PlaneRow(
                  c, byp)[bxp * kBlockDim * kBlockDim + oy * kBlockDim + ox];
              val = fabsf(val);
              fprintf(stderr, "%02x ", int(val / 0.02 * 16));
            }
            fprintf(stderr, "\n");
          }
          fprintf(stderr, "\n");*/
        }
        // fprintf(stderr, "e16: %f e8: %f\n", dct16x16_entropy,
        // dct8x8_entropy); fprintf(stderr, "\n");

        if (k16x16Weight * dct16x16_entropy < dct8x8_entropy)
          return AcStrategyType::DCT16X16;
      }
      return AcStrategyType::DCT;
    };
    pool->Run(0, cache->ac_strategy.layers.ysize(),
              [&find_block_strategy, cache](int y, int _) {
                uint8_t* PIK_RESTRICT row = cache->ac_strategy.layers.Row(y);
                for (size_t x = 0; x < cache->ac_strategy.layers.xsize(); x++) {
                  row[x] = find_block_strategy(x, y);
                }
              });
    for (size_t y = 0; y + 1 < cache->ac_strategy.layers.ysize(); y += 2) {
      uint8_t* PIK_RESTRICT row = cache->ac_strategy.layers.Row(y);
      uint8_t* PIK_RESTRICT next_row = cache->ac_strategy.layers.Row(y + 1);
      for (size_t x = 0; x + 1 < cache->ac_strategy.layers.xsize(); x += 2) {
        if (row[x] == AcStrategyType::DCT16X16) {
          row[x + 1] = AcStrategyType::NONE;
          next_row[x] = AcStrategyType::NONE;
          next_row[x + 1] = AcStrategyType::NONE;
        }
      }
    }
    // TODO(user): don't set this if the strategy is trivial. For this to
    // work, we need to set the flag in the header after the call to
    // FindBestAcStrategy.
    cache->ac_strategy.use_ac_strategy = true;
  } else {
    FillImage((uint8_t)AcStrategyType::DCT, &cache->ac_strategy.layers);
    if (cache->ac_strategy.use_ac_strategy) {
      StrategyResampleAll(&cache->ac_strategy.layers);
    }
  }
}

void FindBestQuantization(const Image3F& opsin_orig, const Image3F& opsin_arg,
                          const CompressParams& cparams,
                          const Header& header_arg, float butteraugli_target,
                          const ColorCorrelationMap& cmap, ThreadPool* pool,
                          Quantizer* quantizer, PikInfo* aux_out,
                          OpsinIntraTransform* transform, double rescale) {
  Header header = header_arg;
  if (header.flags & Header::kGaborishTransformMask) {
    // In the search we will pretend that we have normal Gaborish.
    header.flags &= ~Header::kGaborishTransformMask;
    header.flags |= Header::kGaborishTransform2;
  }

  const float intensity_multiplier = GetIntensityMultiplier(cparams);
  const float intensity_multiplier3 = std::cbrt(intensity_multiplier);

  Rect full(opsin_orig);
  ButteraugliComparator comparator(
      transform->IntraToOpsin(opsin_orig, full, pool), cparams.hf_asymmetry,
      intensity_multiplier);
  const float butteraugli_target_dc = std::min<float>(
      butteraugli_target, pow(butteraugli_target, 0.44468493042213048));
  const float kInitialQuantDC =
      intensity_multiplier3 * (0.61651408278886166) / butteraugli_target_dc;
  const float kQuantAC =
      intensity_multiplier3 * (1.1314004245104872) / butteraugli_target;

  ImageF intensity_ac =
      IntensityAcEstimate(opsin_orig.Plane(1), intensity_multiplier3, pool);

  ImageF quant_field;
  quant_field = ScaleImage(
      kQuantAC * (float)rescale,
      AdaptiveQuantizationMap(
          transform->IntraToOpsin(opsin_orig, full, pool).Plane(1),
          intensity_ac, cparams,
          kBlockDim  // TODO(user): what happens when kBlockDim changes?
          ));
  ImageF tile_distmap;
  ImageF initial_quant_field = CopyImage(quant_field);

  EncCache cache;
  ComputeInitialCoefficients(header, opsin_arg, pool, &cache);
  FindBestAcStrategy(cparams.butteraugli_distance, pool, &cache);
  for (int i = 0; i < cparams.max_butteraugli_iters; ++i) {
    if (FLAGS_dump_quant_state) {
      printf("\nQuantization field:\n");
      for (int y = 0; y < quant_field.ysize(); ++y) {
        for (int x = 0; x < quant_field.xsize(); ++x) {
          printf(" %.5f", quant_field.Row(y)[x]);
        }
        printf("\n");
      }
    }

    if (quantizer->SetQuantField(kInitialQuantDC, QuantField(quant_field))) {
      DecCache dec_cache;
      ComputeCoefficients(*quantizer, cmap, pool, &cache);
      // Override quant field with the one seen by decoder.
      quantizer->SetRawQuantField(CopyImage(cache.quant_field));
      dec_cache.ac_strategy.use_ac_strategy = cache.ac_strategy.use_ac_strategy;
      dec_cache.ac_strategy.layers = CopyImage(cache.ac_strategy.layers);
      dec_cache.quantized_dc = std::move(cache.dc);
      dec_cache.quantized_ac = std::move(cache.ac);
      dec_cache.gradient = CopyImage(cache.gradient);
      DequantImage(*quantizer, cmap, pool, &dec_cache);
      if (i == cparams.max_butteraugli_iters - 1) {
        header = header_arg;
      }
      Image3F recon = ReconOpsinImage(header, *quantizer, pool, &dec_cache);

      if (header.flags & Header::kAdaptiveReconstruction) {
        recon = AdaptiveReconstruction(*quantizer, cmap, pool, recon,
                                       dec_cache.ac_strategy.layers, transform);
      }

      PROFILER_ZONE("enc Butteraugli");
      Image3F linear(recon.xsize(), recon.ysize());
      CenteredOpsinToOpsin(recon, pool, &linear);
      linear.ShrinkTo(opsin_orig.xsize(), opsin_orig.ysize());
      transform->IntraToOpsinInPlace(&linear, Rect(linear), pool);
      OpsinToLinear(linear, pool, &linear);
      comparator.Compare(linear);
      static const int kMargins[100] = {0, 0, 0, 1, 2, 1, 1, 1, 0};
      tile_distmap = TileDistMap(comparator.distmap(), 8, kMargins[i]);
      if (WantDebugOutput(aux_out)) {
        DumpHeatmaps(aux_out, butteraugli_target, quant_field, tile_distmap);
        ++aux_out->num_butteraugli_iters;
      }
      if (FLAGS_log_search_state) {
        float minval, maxval;
        ImageMinMax(quant_field, &minval, &maxval);
        printf("\nButteraugli iter: %d/%d\n", i, cparams.max_butteraugli_iters);
        printf("Butteraugli distance: %f\n", comparator.distance());
        printf("quant range: %f ... %f  DC quant: %f\n", minval, maxval,
               kInitialQuantDC);
        if (FLAGS_dump_quant_state) {
          quantizer->DumpQuantizationMap();
        }
      }
    }
    double kPow[8] = {
        1.0283976124098497,
        1.0416081835470576,
        0.72155909844986499,
        0.35609018139702503,
        0.15,
        0.0,
        0.0,
        0.0,
    };
    double kPowMod[8] = {
        0.0054931613192667691,
        -0.00054442182655455116,
        -0.0053781489009911282,
        0.0056946014681933353,
        0.0,
        0.0,
        0.0,
        0.0,
    };
    if (i == 5) {
      // Don't allow optimization to make the quant field a lot worse than
      // what the initial guess was. This allows the AC field to have enough
      // precision to reduce the oscillations due to the dc reconstruction.
      double kInitMul = 0.6;
      if (header.flags & Header::kSmoothDCPred) {
        // We need to be more aggressive in not letting the adaptive
        // quantization go down to avoid long range oscillations that
        // come naturally from smooth dc.
        kInitMul = 0.9;
      }
      const double kOneMinusInitMul = 1.0 - kInitMul;
      for (int y = 0; y < quant_field.ysize(); ++y) {
        float* const PIK_RESTRICT row_q = quant_field.Row(y);
        const float* const PIK_RESTRICT row_init = initial_quant_field.Row(y);
        for (int x = 0; x < quant_field.xsize(); ++x) {
          double clamp = kOneMinusInitMul * row_q[x] + kInitMul * row_init[x];
          if (row_q[x] < clamp) {
            row_q[x] = clamp;
          }
        }
      }
    }

    double cur_pow = 0.0;
    if (i < 7) {
      cur_pow = kPow[i] + (butteraugli_target - 1.0) * kPowMod[i];
    }
    // pow(x, 0) == 1, so skip pow.
    if (cur_pow == 0.0) {
      for (int y = 0; y < quant_field.ysize(); ++y) {
        const float* const PIK_RESTRICT row_dist = tile_distmap.Row(y);
        float* const PIK_RESTRICT row_q = quant_field.Row(y);
        for (int x = 0; x < quant_field.xsize(); ++x) {
          const float diff = row_dist[x] / butteraugli_target;
          if (diff >= 1.0f) {
            row_q[x] *= diff;
          }
        }
      }
    } else {
      for (int y = 0; y < quant_field.ysize(); ++y) {
        const float* const PIK_RESTRICT row_dist = tile_distmap.Row(y);
        float* const PIK_RESTRICT row_q = quant_field.Row(y);
        for (int x = 0; x < quant_field.xsize(); ++x) {
          const float diff = row_dist[x] / butteraugli_target;
          if (diff < 1.0f) {
            row_q[x] *= pow(diff, cur_pow);
          } else {
            row_q[x] *= diff;
          }
        }
      }
    }
  }
  quantizer->SetQuantField(kInitialQuantDC, QuantField(quant_field));
}

void FindBestQuantizationHQ(const Image3F& opsin_orig, const Image3F& opsin,
                            const CompressParams& cparams, const Header& header,
                            float butteraugli_target,
                            const ColorCorrelationMap& cmap, ThreadPool* pool,
                            Quantizer* quantizer, PikInfo* aux_out,
                            OpsinIntraTransform* transform, double rescale) {
  const bool slow = cparams.guetzli_mode;
  const float intensity_multiplier = GetIntensityMultiplier(cparams);
  const float intensity_multiplier3 = std::cbrt(intensity_multiplier);
  Rect full(0, 0, opsin_orig.xsize(), opsin_orig.ysize());
  const Image3F& untras = transform->IntraToOpsin(opsin_orig, full, pool);
  ButteraugliComparator comparator(untras, cparams.hf_asymmetry,
                                   intensity_multiplier);
  ImageF intensity_ac =
      IntensityAcEstimate(opsin_orig.Plane(1), intensity_multiplier3, pool);
  ImageF quant_field =
      ScaleImage(intensity_multiplier3 * (float)rescale * (slow ? 1.2f : 1.5f),
                 AdaptiveQuantizationMap(untras.Plane(1), intensity_ac, cparams,
                                         kBlockDim));
  ImageF best_quant_field = CopyImage(quant_field);
  float best_butteraugli = 1000.0f;
  ImageF tile_distmap;
  static const int kMaxOuterIters = 2;
  int outer_iter = 0;
  int butteraugli_iter = 0;
  int search_radius = 0;
  float quant_ceil = 5.0f;
  float quant_dc = intensity_multiplier3 * (slow ? 1.2f : 1.6f);
  float best_quant_dc = quant_dc;
  int num_stalling_iters = 0;
  int max_iters = slow ? cparams.max_butteraugli_iters_guetzli_mode
                       : cparams.max_butteraugli_iters;
  EncCache cache;
  ComputeInitialCoefficients(header, opsin, pool, &cache);
  FindBestAcStrategy(butteraugli_target, pool, &cache);
  for (;;) {
    if (FLAGS_dump_quant_state) {
      printf("\nQuantization field:\n");
      for (int y = 0; y < quant_field.ysize(); ++y) {
        for (int x = 0; x < quant_field.xsize(); ++x) {
          printf(" %.5f", quant_field.Row(y)[x]);
        }
        printf("\n");
      }
    }
    float qmin, qmax;
    ImageMinMax(quant_field, &qmin, &qmax);
    ++butteraugli_iter;
    if (quantizer->SetQuantField(quant_dc, QuantField(quant_field))) {
      ComputeCoefficients(*quantizer, cmap, pool, &cache);
      // Override quant field with the one seen by
      // decoder.
      quantizer->SetRawQuantField(CopyImage(cache.quant_field));
      DecCache dec_cache;
      dec_cache.ac_strategy.use_ac_strategy = cache.ac_strategy.use_ac_strategy;
      dec_cache.ac_strategy.layers = CopyImage(cache.ac_strategy.layers);
      dec_cache.quantized_dc = std::move(cache.dc);
      dec_cache.quantized_ac = std::move(cache.ac);
      dec_cache.gradient = CopyImage(cache.gradient);
      DequantImage(*quantizer, cmap, pool, &dec_cache);
      Image3F recon = ReconOpsinImage(header, *quantizer, pool, &dec_cache);

      if (header.flags & Header::kAdaptiveReconstruction) {
        recon = AdaptiveReconstruction(*quantizer, cmap, pool, recon,
                                       dec_cache.ac_strategy.layers, transform);
      }

      PROFILER_ZONE("enc Butteraugli");
      Image3F linear(recon.xsize(), recon.ysize());
      CenteredOpsinToOpsin(recon, pool, &linear);
      linear.ShrinkTo(opsin_orig.xsize(), opsin_orig.ysize());
      transform->IntraToOpsinInPlace(&linear, Rect(linear), pool);
      OpsinToLinear(linear, pool, &linear);
      comparator.Compare(linear);
      bool best_quant_updated = false;
      if (comparator.distance() <= best_butteraugli) {
        best_quant_field = CopyImage(quant_field);
        best_butteraugli = std::max(comparator.distance(), butteraugli_target);
        best_quant_updated = true;
        best_quant_dc = quant_dc;
        num_stalling_iters = 0;
      } else if (outer_iter == 0) {
        ++num_stalling_iters;
      }
      tile_distmap = TileDistMap(comparator.distmap(), 8, 0);
      if (WantDebugOutput(aux_out)) {
        DumpHeatmaps(aux_out, butteraugli_target, quant_field, tile_distmap);
      }
      if (aux_out) {
        ++aux_out->num_butteraugli_iters;
      }
      if (FLAGS_log_search_state) {
        float minval, maxval;
        ImageMinMax(quant_field, &minval, &maxval);
        printf("\nButteraugli iter: %d/%d%s\n", butteraugli_iter, max_iters,
               best_quant_updated ? " (*)" : "");
        printf("Butteraugli distance: %f\n", comparator.distance());
        printf(
            "quant range: %f ... %f  DC quant: "
            "%f\n",
            minval, maxval, quant_dc);
        printf("search radius: %d\n", search_radius);
        if (FLAGS_dump_quant_state) {
          quantizer->DumpQuantizationMap();
        }
      }
    }
    if (butteraugli_iter >= max_iters) {
      break;
    }
    bool changed = false;
    while (!changed && comparator.distance() > butteraugli_target) {
      for (int radius = 0; radius <= search_radius && !changed; ++radius) {
        ImageF dist_to_peak_map =
            DistToPeakMap(tile_distmap, butteraugli_target, radius, 0.0);
        for (int y = 0; y < quant_field.ysize(); ++y) {
          float* const PIK_RESTRICT row_q = quant_field.Row(y);
          const float* const PIK_RESTRICT row_dist = dist_to_peak_map.Row(y);
          for (int x = 0; x < quant_field.xsize(); ++x) {
            if (row_dist[x] >= 0.0f) {
              static const float kAdjSpeed[kMaxOuterIters] = {0.1f, 0.04f};
              const float factor = (slow ? kAdjSpeed[outer_iter] : 0.2f) *
                                   tile_distmap.Row(y)[x];
              if (AdjustQuantVal(&row_q[x], row_dist[x], factor, quant_ceil)) {
                changed = true;
              }
            }
          }
        }
      }
      if (!changed || num_stalling_iters >= (slow ? 3 : 1)) {
        // Try to extend the search parameters.
        if ((search_radius < 4) &&
            (qmax < 0.99f * quant_ceil || quant_ceil >= 3.0f + search_radius)) {
          ++search_radius;
          continue;
        }
        if (quant_dc < 0.4f * quant_ceil - 0.8f) {
          quant_dc += 0.2f;
          changed = true;
          continue;
        }
        if (quant_ceil < 8.0f) {
          quant_ceil += 0.5f;
          continue;
        }
        break;
      }
    }
    if (!changed) {
      if (!slow || ++outer_iter == kMaxOuterIters) break;
      static const float kQuantScale = 0.75f;
      for (int y = 0; y < quant_field.ysize(); ++y) {
        for (int x = 0; x < quant_field.xsize(); ++x) {
          quant_field.Row(y)[x] *= kQuantScale;
        }
      }
      num_stalling_iters = 0;
    }
  }
  quantizer->SetQuantField(best_quant_dc, QuantField(best_quant_field));
}

template <typename V, typename R>
inline void FindIndexOfSumMaximum(const V* array, const size_t len, R* idx,
                                  V* sum) {
  PIK_ASSERT(len > 0);
  V maxval = 0;
  V val = 0;
  R maxidx = 0;
  for (size_t i = 1; i < len; ++i) {
    val += array[i];
    if (val > maxval) {
      maxval = val;
      maxidx = i;
    }
  }
  *idx = maxidx;
  *sum = maxval;
}

template <int MAIN_CHANNEL, int SIDE_CHANNEL, int SCALE, int OFFSET>
void FindBestCorrelation(const Image3F& dct, ImageI* PIK_RESTRICT map,
                         ImageF* PIK_RESTRICT tmp_map, int* PIK_RESTRICT dc,
                         float acceptance) {
  constexpr int N = kBlockDim;
  constexpr int block_size = N * N;
  constexpr float kScale = SCALE;
  constexpr float kZeroThresh = kScale * kZeroBiasHQ[SIDE_CHANNEL];
  // Always use DCT8 quantization values for DC.
  const float* const PIK_RESTRICT kDequantMatrix =
      &DequantMatrix<N>(0, kQuantKindDCT8)[block_size * SIDE_CHANNEL];
  float qm[block_size];
  for (int k = 0; k < block_size; ++k) {
    qm[k] = 1.0f / kDequantMatrix[k];
  }
  int32_t d_num_zeros_global[256] = {0};
  for (int ty = 0; ty < map->ysize(); ++ty) {
    int* PIK_RESTRICT row_out = map->Row(ty);
    float* PIK_RESTRICT row_tmp_out = tmp_map->Row(ty);
    for (int tx = 0; tx < map->xsize(); ++tx) {
      const int y0 = ty * kColorTileDimInBlocks;
      const int x0 = tx * kColorTileDimInBlocks * block_size;
      const int y1 = std::min<int>(y0 + kColorTileDimInBlocks, dct.ysize());
      const int x1 =
          std::min<int>(x0 + kColorTileDimInBlocks * block_size, dct.xsize());
      int32_t d_num_zeros[257] = {0};
      for (size_t y = y0; y < y1; ++y) {
        const float* const PIK_RESTRICT row_m =
            dct.ConstPlaneRow(MAIN_CHANNEL, y);
        const float* const PIK_RESTRICT row_s =
            dct.ConstPlaneRow(SIDE_CHANNEL, y);
        for (size_t x = x0; x < x1; ++x) {
          if (x % block_size == 0) continue;
          const float scaled_m = row_m[x] * qm[x % block_size];
          const float scaled_s =
              kScale * row_s[x] * qm[x % block_size] + OFFSET * scaled_m;
          // Increment num_zeros[idx] if
          //   std::abs(scaled_s - (idx - OFFSET) *
          //   scaled_m) < kZeroThresh
          if (std::abs(scaled_m) < 1e-8) {
            // Range is too narrow, all-or-nothing
            // strategy should be OK.
            if (std::abs(scaled_s) < kZeroThresh) {
              d_num_zeros[0]++;
            }
          } else {
            float from;
            float to;
            if (scaled_m > 0) {
              from = (scaled_s - kZeroThresh) / scaled_m;
              to = (scaled_s + kZeroThresh) / scaled_m;
            } else {
              from = (scaled_s + kZeroThresh) / scaled_m;
              to = (scaled_s - kZeroThresh) / scaled_m;
            }
            if (from < 0.0f) {
              from = 0.0f;
            }
            if (to > 255.0f) {
              to = 255.0f;
            }
            // Instead of clamping the both values
            // we just check that range is sane.
            if (from <= to) {
              d_num_zeros[(int)std::ceil(from)]++;
              d_num_zeros[(int)std::floor(to + 1)]--;
            }
          }
        }
      }
      int best = 0;
      int32_t best_sum = 0;
      FindIndexOfSumMaximum(d_num_zeros, 256, &best, &best_sum);
      for (size_t i = 0; i < 256; ++i) {
        d_num_zeros_global[i] += d_num_zeros[i];
      }
      row_out[tx] = best;
      row_tmp_out[tx] = (float)best_sum / ((x1 - x0) * (y1 - y0));
    }
  }

  int global_best = 0;
  int32_t global_sum = 0;
  FindIndexOfSumMaximum(d_num_zeros_global, 256, &global_best, &global_sum);
  float global_normalized_sum = (float)global_sum / (dct.xsize() * dct.ysize());
  float normalized_acceptance =
      acceptance * kColorTileDimInBlocks * kColorTileDimInBlocks * block_size;
  for (int ty = 0; ty < map->ysize(); ++ty) {
    int* PIK_RESTRICT row_out = map->Row(ty);
    float* PIK_RESTRICT row_tmp_out = tmp_map->Row(ty);
    for (int tx = 0; tx < map->xsize(); ++tx) {
      // Revert to the global factor used for dc if
      // the number of zeros is almost the same.
      if (row_tmp_out[tx] <= global_normalized_sum + normalized_acceptance) {
        row_out[tx] = global_best;
      }
    }
  }
  *dc = global_best;
}

// Proposes a distance to try for a given bpp target. This could depend
// on the entropy in the image, too, but let's start with something.
static double ApproximateButteraugliDistanceForBPP(double bpp) {
  return 1.704 * pow(bpp, -0.804);
}

void CompressToTargetSize(const Image3F& opsin_orig, const Image3F& opsin,
                          const CompressParams& cparams,
                          const NoiseParams& noise_params, const Header& header,
                          size_t target_size, const ColorCorrelationMap& cmap,
                          ThreadPool* pool, Quantizer* quantizer,
                          PikInfo* aux_out, OpsinIntraTransform* transform,
                          double rescale) {
  const double num_pixels = opsin.xsize() * opsin.ysize();
  double dist = ApproximateButteraugliDistanceForBPP(target_size / num_pixels);
  double best_dist;
  double best_loss = 1e99;
  float best_quant_dc;
  ImageF best_quant_ac;
  EncCache cache;
  ComputeInitialCoefficients(header, opsin, pool, &cache);
  for (int i = 0; i < 12; ++i) {
    FindBestQuantization(opsin_orig, opsin, cparams, header, dist, cmap, pool,
                         quantizer, aux_out, transform, rescale);
    FindBestAcStrategy(dist, pool, &cache);
    ComputeCoefficients(*quantizer, cmap, pool, &cache);
    PaddedBytes candidate = EncodeToBitstream(
        cache, *quantizer, cache.gradient, noise_params, cmap, false, nullptr);
    const double ratio = static_cast<double>(candidate.size()) / target_size;
    const double loss = std::max(ratio, 1.0 / std::max(ratio, 1e-30));
    if (best_loss > loss) {
      best_dist = dist;
      best_loss = loss;
      quantizer->GetQuantField(&best_quant_dc, &best_quant_ac);
    }
    dist *= ratio;
  }
  printf("Choosing butteraugli distance %.15g\n", best_dist);
  quantizer->SetQuantField(best_quant_dc, QuantField(best_quant_ac));
}

Status JpegToPikLossless(const guetzli::JPEGData& jpg, PaddedBytes* compressed,
                         PikInfo* aux_out) {
  Container container;
  size_t container_bits;
  PIK_CHECK(CanEncode(container, &container_bits));

  Header header;
  header.bitstream = Bitstream::kBrunsli;
  size_t header_bits;
  PIK_CHECK(CanEncode(header, &header_bits));

  const size_t encoded_bits = container_bits + header_bits;
  compressed->resize((encoded_bits + 7) / 8 + BrunsliV2MaximumEncodedSize(jpg));
  size_t pos = 0;
  PIK_CHECK(StoreContainer(container, &pos, compressed->data()));
  PIK_CHECK(StoreHeader(header, &pos, compressed->data()));
  WriteZeroesToByteBoundary(&pos, compressed->data());
  if (!BrunsliV2EncodeJpegData(jpg, pos / 8, compressed)) {
    return PIK_FAILURE("Invalid jpeg input.");
  }
  return true;
}

Status BrunsliToPixels(const PaddedBytes& compressed, size_t pos,
                       CodecInOut* io) {
  guetzli::JPEGData jpeg;
  if (!BrunsliV2DecodeJpegData(compressed.data() + pos, compressed.size() - pos,
                               &jpeg)) {
    return PIK_FAILURE("Brunsli v2 decoding error");
  }
  std::vector<uint8_t> rgb = DecodeJpegToRGB(jpeg);
  if (rgb.empty()) {
    return PIK_FAILURE("JPEG decoding error.");
  }

  const bool is_gray = false;
  const bool has_alpha = false;
  const uint8_t* end = rgb.data() + rgb.size();
  return io->SetFromSRGB(jpeg.width, jpeg.height, is_gray, has_alpha,
                         rgb.data(), end);
}

// For encoder.
uint32_t HeaderFlagsFromParams(const CompressParams& params,
                               const CodecInOut* io) {
  uint32_t flags = 0;

  const float dist = params.butteraugli_distance;

  // We don't add noise at low butteraugli distances because the original noise
  // is stored within the compressed image and adding noise makes things worse.
  if (ApplyOverride(params.noise, dist >= kMinButteraugliForNoise)) {
    flags |= Header::kNoise;
  }

  if (ApplyOverride(params.gradient, dist >= kMinButteraugliForGradient)) {
    flags |= Header::kGradientMap;
  }

  if (ApplyOverride(params.adaptive_reconstruction,
                    dist >= kMinButteraugliForAdaptiveReconstruction)) {
    flags |= Header::kAdaptiveReconstruction;
  }

  // Smooth prediction (8x upsampling) is more expensive when adaptive
  // reconstruction is enabled, and worse than adaptive if we can only choose
  // one of them. Therefore, only use it if forced via override.
  if (ApplyOverride(params.smooth, false)) {
    flags |= Header::kSmoothDCPred;
  }

  uint32_t default_gaborish = (flags & Header::kSmoothDCPred) ? 4 : 0;
  if (flags & Header::kGradientMap) {
    if (dist < 2.6) {
      default_gaborish = 3;  // 75%
    } else if (dist < 3.5) {
      default_gaborish = 2;  // 50%
    } else {
      default_gaborish = 1;  // 25%
    }
  }
  if (params.gaborish != -1) default_gaborish = params.gaborish;
  PIK_ASSERT(default_gaborish <= Header::kGaborishTransformMask);
  flags |= default_gaborish << Header::kGaborishTransformShift;

  if (params.use_ac_strategy) {
    if (flags & Header::kSmoothDCPred) {
      flags |= Header::kUseAcStrategy;
    }
    // TODO(user): investigate if non-smooth could also be made eligible.
  }
  // TODO(user): we should only set the flag if we actually end up with a
  // non-trivial strategy.
  if (kChooseAcStrategy) {
    flags |= Header::kUseAcStrategy;
  }

  if (io->IsGray()) {
    flags |= Header::kGrayscaleOpt;
  }

  return flags;
}

void OverrideFlag(const Override o, const uint32_t flag,
                  uint32_t* PIK_RESTRICT flags) {
  if (o == Override::kOn) {
    *flags |= flag;
  } else if (o == Override::kOff) {
    *flags &= ~flag;
  }
}

void OverrideFlags(const DecompressParams& params,
                   Header* PIK_RESTRICT header) {
  OverrideFlag(params.noise, Header::kNoise, &header->flags);
  OverrideFlag(params.gradient, Header::kGradientMap, &header->flags);
  OverrideFlag(params.adaptive_reconstruction, Header::kAdaptiveReconstruction,
               &header->flags);
}

}  // namespace

std::shared_ptr<Quantizer> GetCompressQuantizer(
    const CompressParams& params, size_t xsize_blocks, size_t ysize_blocks,
    const Image3F& opsin_orig, const Image3F& opsin,
    const NoiseParams& noise_params, const Header& header,
    const ColorCorrelationMap& cmap, ThreadPool* pool, PikInfo* aux_out,
    OpsinIntraTransform* transform, double rescale) {
  const float dist = params.butteraugli_distance;
  int quant_template =
      (dist >= kMinButteraugliForDefaultQuant) ? kQuantDefault : kQuantHQ;
  std::shared_ptr<Quantizer> quantizer = std::make_shared<Quantizer>(
      kBlockDim, quant_template, xsize_blocks, ysize_blocks);
  const float intensity_multiplier = GetIntensityMultiplier(params);
  const float intensity_multiplier3 = std::cbrt(intensity_multiplier);
  if (params.fast_mode) {
    PROFILER_ZONE("enc fast quant");
    const float butteraugli_target = params.butteraugli_distance;
    const float butteraugli_target_dc = std::min<float>(
        butteraugli_target, pow(butteraugli_target, 0.32595755694491835));
    const float butteraugli_target_ac = std::min<float>(
        butteraugli_target, pow(butteraugli_target, 0.94005123937290269));
    const float kQuantDC =
        intensity_multiplier3 * (0.51651408278886166) / butteraugli_target_dc;
    const float kQuantAC =
        intensity_multiplier3 * (1.7170154843665173) / butteraugli_target_ac;
    Rect full(opsin_orig);
    ImageF intensity_ac = IntensityAcEstimate(
        transform->IntraToOpsin(opsin_orig, full, pool).Plane(1),
        intensity_multiplier3, pool);
    ImageF qf = AdaptiveQuantizationMap(
        opsin_orig.Plane(1), intensity_ac, params,
        kBlockDim  // TODO(user): what happens when kBlockDim changes?
    );
    quantizer->SetQuantField(
        kQuantDC, QuantField(ScaleImage(kQuantAC * (float)rescale, qf)));
  } else if (params.target_size > 0 || params.target_bitrate > 0.0) {
    size_t target_size = TargetSize(params, opsin);
    PROFILER_ZONE("enc compressToTarget");
    CompressToTargetSize(opsin_orig, opsin, params, noise_params, header,
                         target_size, cmap, pool, quantizer.get(), aux_out,
                         transform, rescale);
  } else if (params.uniform_quant > 0.0) {
    PROFILER_ZONE("enc SetQuant");
    quantizer->SetQuant(params.uniform_quant * rescale);
  } else {
    // Normal PIK encoding to a butteraugli score.
    PROFILER_ZONE("enc find best2");
    if (params.guetzli_mode) {
      FindBestQuantizationHQ(opsin_orig, opsin, params, header,
                             params.butteraugli_distance, cmap, pool,
                             quantizer.get(), aux_out, transform, rescale);
    } else {
      FindBestQuantization(opsin_orig, opsin, params, header,
                           params.butteraugli_distance, cmap, pool,
                           quantizer.get(), aux_out, transform, rescale);
    }
  }
  return quantizer;
}

Status PixelsToPikFrame(
    CompressParams params, const CodecInOut* io,
    const GetQuantizer& get_quantizer, PaddedBytes* compressed, size_t& pos,
    PikInfo* aux_out, OpsinIntraTransform* transform,
    const std::function<void(const Header& header, Sections&& sections,
                             size_t image_start)>& pass_info) {
  size_t target_size = TargetSize(params, io->color());
  size_t opsin_target_size =
      (compressed->size() < target_size ? target_size - compressed->size() : 1);
  if (params.target_size > 0 || params.target_bitrate > 0.0) {
    params.target_size = opsin_target_size;
  } else if (params.butteraugli_distance < 0) {
    return PIK_FAILURE("Expected non-negative distance");
  }

  Header header;
  header.xsize = io->xsize();
  header.ysize = io->ysize();
  header.resampling_factor2 = params.resampling_factor2;
  header.flags = HeaderFlagsFromParams(params, io);

  size_t header_bits;
  PIK_RETURN_IF_ERROR(CanEncode(header, &header_bits));

  Sections sections;
  if (io->HasAlpha()) {
    PROFILER_ZONE("enc alpha");
    PIK_RETURN_IF_ERROR(
        AlphaToPik(params, io->alpha(), io->AlphaBits(), &sections.alpha));
  }
  size_t sections_bits;
  PIK_RETURN_IF_ERROR(CanEncode(sections, &sections_bits));

  compressed->resize(DivCeil(pos + header_bits + sections_bits, kBitsPerByte));
  PIK_RETURN_IF_ERROR(StoreHeader(header, &pos, compressed->data()));
  PIK_RETURN_IF_ERROR(StoreSections(sections, &pos, compressed->data()));
  WriteZeroesToByteBoundary(&pos, compressed->data());

  if (aux_out != nullptr) {
    aux_out->layers[kLayerHeader].total_size += (header_bits + 7) / 8;
    aux_out->layers[kLayerSections].total_size += (sections_bits + 7) / 8;
  }
  if (pass_info) {
    pass_info(header, std::move(sections), compressed->size());
  }
  Image3F opsin_orig = OpsinDynamicsImage(io);  // parallel
  Rect full(opsin_orig);
  transform->OpsinToIntraInPlace(&opsin_orig, full, &io->Context()->pool);
  if (header.resampling_factor2 != 2) {
    opsin_orig = DownsampleImage(opsin_orig, header.resampling_factor2);
  }

  constexpr size_t N = kBlockDim;
  PROFILER_ZONE("enc OpsinToPik uninstrumented");
  const size_t xsize = opsin_orig.xsize();
  const size_t ysize = opsin_orig.ysize();
  if (xsize == 0 || ysize == 0) return PIK_FAILURE("Empty image");
  const size_t xsize_blocks = DivCeil(xsize, N);
  const size_t ysize_blocks = DivCeil(ysize, N);
  ThreadPool* pool = &io->Context()->pool;
  Image3F opsin = PadImageToMultiple(opsin_orig, N);
  CenterOpsinValues(&opsin);
  NoiseParams noise_params;

  if (header.flags & Header::kNoise) {
    PROFILER_ZONE("enc GetNoiseParam");
    // Don't start at zero amplitude since adding noise is expensive -- it
    // significantly slows down decoding, and this is unlikely to completely go
    // away even with advanced optimizations. After the
    // kNoiseModelingRampUpDistanceRange we have reached the full level, i.e.
    // noise is no longer represented by the compressed image, so we can add
    // full noise by the noise modeling itself.
    static const double kNoiseModelingRampUpDistanceRange = 0.6;
    static const double kNoiseLevelAtStartOfRampUp = 0.25;
    // TODO(user) test and properly select quality_coef with smooth filter
    float quality_coef = 1.0f;
    const double rampup =
        (params.butteraugli_distance - kMinButteraugliForNoise) /
        kNoiseModelingRampUpDistanceRange;
    if (rampup < 1.0) {
      quality_coef = kNoiseLevelAtStartOfRampUp +
                     (1.0 - kNoiseLevelAtStartOfRampUp) * rampup;
    }
    GetNoiseParameter(opsin, &noise_params, quality_coef);
  }
  if (header.flags & Header::kGaborishTransformMask) {
    double strength = 1.0;
    if (params.butteraugli_distance > 2.5) {
      strength += 0.05 * (params.butteraugli_distance - 2.5);
      if (strength >= 1.5) {
        strength = 1.5;
      }
    }
    GaborishInverse(opsin, strength);
  }
  ColorCorrelationMap cmap(xsize, ysize);
  if (!params.fast_mode &&
      (params.butteraugli_distance >= 0.0 || params.target_bitrate > 0.0 ||
       params.target_size > 0)) {
    PROFILER_ZONE("enc YTo* correlation");

    Image3F dct = TransposedScaledDCT<N>(
        transform->IntraToOpsin(opsin, full, pool), pool);

    ImageF tmp(DivCeil(xsize, kColorTileDim), DivCeil(ysize, kColorTileDim));

    // These two coefficients are eligible for optimization.
    // Perhaps, they also could be made quality-dependent.
    // Prefer global until 25% more (full) tile coefficients become zero.
    float y_to_b_acceptance = 0.25f;
    // Prefer local until 62.5% less (full) tile coefficients become zero.
    float y_to_x_acceptance = -0.625f;

    FindBestCorrelation</* from Y */ 1, /* to B */ 2, kColorFactorB,
                        kColorOffsetB>(dct, &cmap.ytob_map, &tmp, &cmap.ytob_dc,
                                       y_to_b_acceptance);
    FindBestCorrelation</* from Y */ 1, /* to X */ 0, kColorFactorX,
                        kColorOffsetX>(dct, &cmap.ytox_map, &tmp, &cmap.ytox_dc,
                                       y_to_x_acceptance);
  }

  std::shared_ptr<Quantizer> quantizer =
      get_quantizer(params, xsize_blocks, ysize_blocks, opsin_orig, opsin,
                    noise_params, header, cmap, pool, aux_out, transform);
  EncCache cache;
  ComputeInitialCoefficients(header, opsin, pool, &cache);
  FindBestAcStrategy(params.butteraugli_distance, pool, &cache);
  ComputeCoefficients(*quantizer, cmap, pool, &cache, aux_out);
  PaddedBytes compressed_data =
      EncodeToBitstream(cache, *quantizer, cache.gradient, noise_params, cmap,
                        params.fast_mode, aux_out);

  {
    size_t old_size = compressed->size();
    compressed->resize(compressed->size() + compressed_data.size());
    memcpy(compressed->data() + old_size, compressed_data.data(),
           compressed_data.size());
    pos += compressed_data.size() * kBitsPerByte;
  }
  io->enc_size = compressed->size();

  return true;
}

Status PixelsToPik(const CompressParams& params, const CodecInOut* io,
                   PaddedBytes* compressed, PikInfo* aux_out) {
  if (io->xsize() == 0 || io->ysize() == 0) {
    return PIK_FAILURE("Empty image");
  }
  if (!io->HasOriginalBitsPerSample()) {
    return PIK_FAILURE(
        "Pik requires specifying original bit depth "
        "of the pixels to encode as metadata.");
  }
  Container container;
  container.original_bit_depth = io->original_bits_per_sample();
  container.original_color_encoding = io->dec_c_original;
  container.target_nits_div50 = params.intensity_target / 50;
  container.simple =
      container.original_bit_depth == 8 &&
      container.original_color_encoding.IsSRGB() &&
      container.target_nits_div50 == kDefaultIntensityTarget / 50;
  (void)io->Context()->cms.MaybeRemoveProfile(
      &container.original_color_encoding);

  size_t container_bits;
  PIK_CHECK(CanEncode(container, &container_bits));

  // TODO(janwas): enable (behind flag)
  constexpr bool kAllowMetadata = false;

  ContainerSections container_sections;
  if (kAllowMetadata && io->metadata.HasAny()) {
    container_sections.metadata.reset(new Metadata(io->metadata));
  }
  size_t container_sections_bits;
  PIK_CHECK(CanEncode(container_sections, &container_sections_bits));

  compressed->resize(
      DivCeil(container_bits + container_sections_bits, kBitsPerByte));
  size_t pos = 0;
  PIK_RETURN_IF_ERROR(StoreContainer(container, &pos, compressed->data()));
  PIK_RETURN_IF_ERROR(
      StoreSections(container_sections, &pos, compressed->data()));

  NoopOpsinIntraTransform transform;
  return PixelsToPikFrame(
      params, io,
      [](const CompressParams& params, size_t xsize_blocks, size_t ysize_blocks,
         const Image3F& opsin_orig, const Image3F& opsin,
         const NoiseParams& noise_params, const Header& header,
         const ColorCorrelationMap& cmap, ThreadPool* pool, PikInfo* aux_out,
         OpsinIntraTransform* transform) {
        return GetCompressQuantizer(params, xsize_blocks, ysize_blocks,
                                    opsin_orig, opsin, noise_params, header,
                                    cmap, pool, aux_out, transform);
      },
      compressed, pos, aux_out, &transform);
}

Status JpegToPik(CodecContext* codec_context, const CompressParams& params,
                 const guetzli::JPEGData& jpeg, PaddedBytes* compressed,
                 PikInfo* aux_out) {
  if (params.butteraugli_distance <= 0.0) {
    return JpegToPikLossless(jpeg, compressed, aux_out);
  }

  // Create CodecInOut so we can call the regular PIK encoder.
  const std::vector<uint8_t>& interleaved = DecodeJpegToRGB(jpeg);
  CodecInOut io(codec_context);
  const bool is_gray = false;
  const bool has_alpha = false;
  return io.SetFromSRGB(jpeg.width, jpeg.height, is_gray, has_alpha,
                        interleaved.data(),
                        interleaved.data() + interleaved.size()) &&
         PixelsToPik(params, &io, compressed, aux_out);
}

Status ValidateHeaderFields(const Header& header,
                            const DecompressParams& params) {
  if (header.xsize == 0 || header.ysize == 0) {
    return PIK_FAILURE("Empty image.");
  }

  static const uint32_t kMaxWidth = (1 << 25) - 1;
  if (header.xsize > kMaxWidth) {
    return PIK_FAILURE("Image too wide.");
  }

  uint64_t num_pixels = static_cast<uint64_t>(header.xsize) * header.ysize;
  if (num_pixels > params.max_num_pixels) {
    return PIK_FAILURE("Image too big.");
  }

  return true;
}

// Allows decoding just the header or sections without loading the entire file.
class Decoder {
 public:
  // To avoid the complexity of file I/O and buffering, we assume the bitstream
  // is loaded (or for large images/sequences: mapped into) memory.
  Decoder(const uint8_t* compressed, const size_t compressed_size)
      : reader_(compressed, compressed_size) {}

  Status ReadContainer() {
    PIK_CHECK(valid_ == 0);
    PIK_RETURN_IF_ERROR(LoadContainer(&reader_, &container_));
    PIK_RETURN_IF_ERROR(LoadSections(&reader_, &container_sections_));
    valid_ |= kContainer;
    return true;
  }

  Status ReadHeader() {
    PIK_CHECK(valid_ == kContainer);
    PIK_RETURN_IF_ERROR(LoadHeader(&reader_, &header_));
    if (header_.bitstream == Bitstream::kDefault) {
      PIK_RETURN_IF_ERROR(LoadSections(&reader_, &sections_));
      reader_.JumpToByteBoundary();
    } else {
      // sections_ is default-initialized and thus already "valid".
    }

    valid_ |= kHeader;
    return true;
  }

  const Container& GetContainer() const {
    PIK_CHECK(valid_ & kContainer);
    return container_;
  }

  const ContainerSections& GetContainerSections() const {
    PIK_CHECK(valid_ & kContainer);
    return container_sections_;
  }

  const Header& GetHeader() const {
    PIK_CHECK(valid_ & kHeader);
    return header_;
  }

  const Sections& GetSections() const {
    PIK_CHECK(valid_ & kHeader);
    return sections_;
  }

  // TODO(janwas): remove once Brunsli is integrated.
  BitReader& GetReader() { return reader_; }

 private:
  enum Valid {
    kContainer = 1,
    kHeader = 1,
  };

  BitReader reader_;
  uint64_t valid_ = 0;
  Container container_;
  // Workaround for ubsan alignment error.
  alignas(8) ContainerSections container_sections_;
  // TODO(janwas): vector of these
  Header header_;
  Sections sections_;
};

Status PikFrameToPixels(const DecompressParams& params,
                        const PaddedBytes& compressed,
                        const Container& container,
                        const ContainerSections& container_sections,
                        const Header& header, const Sections& sections,
                        BitReader* reader, CodecInOut* io, PikInfo* aux_out,
                        OpsinIntraTransform* transform) {
  size_t data_start = reader->Position();

  if (container_sections.metadata != nullptr) {
    io->metadata = *container_sections.metadata;
  }
  if (header.bitstream != Bitstream::kDefault) {
    return PIK_FAILURE("Unsupported bitstream");
  }
  ThreadPool* pool = &io->Context()->pool;
  // (Only valid for Bitstream::kDefault!)
  PIK_RETURN_IF_ERROR(ValidateHeaderFields(header, params));

  // Used when writing the output file unless DecoderHints overrides it.
  io->SetOriginalBitsPerSample(container.original_bit_depth);
  io->dec_c_original = container.original_color_encoding;
  if (io->dec_c_original.icc.empty()) {
    // Removed by MaybeRemoveProfile; fail unless we successfully restore it.
    PIK_RETURN_IF_ERROR(
        io->Context()->cms.SetProfileFromFields(&io->dec_c_original));
  }

  const size_t resampling_factor2 = header.resampling_factor2;
  if (resampling_factor2 != 2 && resampling_factor2 != 3 &&
      resampling_factor2 != 4 && resampling_factor2 != 8) {
    return PIK_FAILURE("Pik decoding failed: invalid resampling factor");
  }

  ImageSize opsin_size = DownsampledImageSize(
      ImageSize::Make(header.xsize, header.ysize), resampling_factor2);
  const size_t xsize_blocks = DivCeil<size_t>(opsin_size.xsize, kBlockDim);
  const size_t ysize_blocks = DivCeil<size_t>(opsin_size.ysize, kBlockDim);

  Quantizer quantizer(kBlockDim, 0, xsize_blocks, ysize_blocks);
  NoiseParams noise_params;
  ColorCorrelationMap cmap(opsin_size.xsize, opsin_size.ysize);
  DecCache dec_cache;
  dec_cache.eager_dequant = true;
  {
    PROFILER_ZONE("dec_bitstr");
    if (!DecodeFromBitstream(header, compressed, reader, xsize_blocks,
                             ysize_blocks, pool, &cmap, &noise_params,
                             &quantizer, &dec_cache)) {
      return PIK_FAILURE("Pik decoding failed.");
    }
  }
  // DequantImage is not invoked, because coefficients are eagerly dequantized
  // in DecodeFromBitstream.
  Image3F opsin = ReconOpsinImage(header, quantizer, pool, &dec_cache, aux_out);

  if (header.flags & Header::kAdaptiveReconstruction) {
    opsin = AdaptiveReconstruction(quantizer, cmap, pool, opsin,
                                   dec_cache.ac_strategy.layers, transform);
  }

  if (header.flags & Header::kNoise) {
    PROFILER_ZONE("add_noise");
    AddNoise(noise_params, &opsin);
  }

  CenteredOpsinToOpsin(opsin, pool, &opsin);

  Rect full(opsin);
  transform->IntraToOpsinInPlace(&opsin, full, pool);

  if (resampling_factor2 != 2) {
    opsin =
        UpsampleImage(opsin, header.xsize, header.ysize, resampling_factor2);
  }

  Image3F linear(header.xsize, header.ysize);
  OpsinToLinear(opsin, pool, &linear);
  if (header.flags & Header::kGrayscaleOpt) {
    for (size_t y = 0; y < linear.ysize(); ++y) {
      float* PIK_RESTRICT row_r = linear.PlaneRow(0, y);
      float* PIK_RESTRICT row_g = linear.PlaneRow(1, y);
      float* PIK_RESTRICT row_b = linear.PlaneRow(2, y);
      for (size_t x = 0; x < linear.xsize(); x++) {
        float gray = row_r[x] * 0.299 + row_g[x] * 0.587 + row_b[x] * 0.114;
        row_r[x] = row_g[x] = row_b[x] = gray;
      }
    }
  }

  const ColorEncoding& c =
      io->Context()->c_linear_srgb[io->dec_c_original.IsGray()];
  io->SetFromImage(std::move(linear), c);
  if (sections.alpha != nullptr) {
    ImageU alpha(header.ysize, header.ysize);
    PIK_RETURN_IF_ERROR(PikToAlpha(params, *sections.alpha, &alpha));
    const size_t alpha_bits = sections.alpha->bytes_per_alpha * kBitsPerByte;
    io->SetAlpha(std::move(alpha), alpha_bits);
  }

  io->enc_size = reader->Position() - data_start;

  return true;
}

Status PikToPixels(const DecompressParams& params,
                   const PaddedBytes& compressed, CodecInOut* io,
                   PikInfo* aux_out) {
  PROFILER_ZONE("PikToPixels uninstrumented");

  Decoder decoder(compressed.data(), compressed.size());
  PIK_RETURN_IF_ERROR(decoder.ReadContainer());
  PIK_RETURN_IF_ERROR(decoder.ReadHeader());

  const Container& container = decoder.GetContainer();

  const ContainerSections& container_sections = decoder.GetContainerSections();

  Header header = decoder.GetHeader();  // copy so we can override flags
  OverrideFlags(params, &header);

  if (header.bitstream == Bitstream::kBrunsli) {
    // TODO(janwas): prepend sections, ValidateHeader, avoid padding
    decoder.GetReader().JumpToByteBoundary();
    return BrunsliToPixels(compressed, decoder.GetReader().Position(), io);
  }

  const Sections& sections = decoder.GetSections();
  NoopOpsinIntraTransform transform;
  PIK_RETURN_IF_ERROR(PikFrameToPixels(
      params, compressed, container, container_sections, header, sections,
      &decoder.GetReader(), io, aux_out, &transform));

  if (params.check_decompressed_size &&
      decoder.GetReader().Position() != compressed.size()) {
    return PIK_FAILURE("Pik compressed data size mismatch.");
  }
  return true;
}

}  // namespace pik
