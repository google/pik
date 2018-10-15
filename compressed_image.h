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

#ifndef COMPRESSED_IMAGE_H_
#define COMPRESSED_IMAGE_H_

#include <stddef.h>
#include <stdint.h>

#include "bit_reader.h"
#include "common.h"
#include "data_parallel.h"
#include "header.h"
#include "image.h"
#include "noise.h"
#include "padded_bytes.h"
#include "pik_info.h"
#include "pik_params.h"
#include "quantizer.h"

namespace pik {

void CenterOpsinValues(Image3F* img);

// Tile is the rectangular grid of blocks that share color correlation
// parameters ("factor_x/b" such that residual_b = blue - Y * factor_b).
constexpr size_t kColorTileDim = 64;

static_assert(kColorTileDim % kBlockDim == 0,
              "Color tile dim should be divisible by block dim");
constexpr size_t kColorTileDimInBlocks = kColorTileDim / kBlockDim;

static_assert(kGroupWidthInBlocks % kColorTileDimInBlocks == 0,
              "Group dim should be divisible by color tile dim");

constexpr const int32_t kColorFactorX = 256;
constexpr const int32_t kColorOffsetX = 128;
constexpr const float kColorScaleX = 1.0f / kColorFactorX;

constexpr const int32_t kColorFactorB = 128;
constexpr const int32_t kColorOffsetB = 0;
constexpr const float kColorScaleB = 1.0f / kColorFactorB;

// For dispatching to ColorCorrelationMap::YtoTag overloads.
struct TagX {};
struct TagB {};

// WARNING: keep in sync with separate SIMD YtoTag implementation.
struct ColorCorrelationMap {
  ColorCorrelationMap(size_t xsize, size_t ysize)  // pixels
      : ytox_dc(128),
        ytob_dc(120),
        ytox_map(DivCeil(xsize, kColorTileDim), DivCeil(ysize, kColorTileDim)),
        ytob_map(DivCeil(xsize, kColorTileDim), DivCeil(ysize, kColorTileDim)) {
    FillImage(128, &ytox_map);
    FillImage(120, &ytob_map);
  }

  // |y| is scaled by some number calculated from x_factor; consequently,
  // passing 1.0f in place of |y| result will be the scaling factor.
  constexpr static float YtoX(float y, int32_t x_factor) {
    return y * (x_factor - kColorOffsetX) * kColorScaleX;
  }

  // |y| is scaled by some number calculated from b_factor; consequently,
  // passing 1.0f in place of |y| result will be the scaling factor.
  constexpr static float YtoB(float y, int32_t b_factor) {
    return y * (b_factor - kColorOffsetB) * kColorScaleB;
  }

  constexpr static float YtoTag(TagX, float y, int32_t factor) {
    return YtoX(y, factor);
  }
  constexpr static float YtoTag(TagB, float y, int32_t factor) {
    return YtoB(y, factor);
  }

  int32_t ytox_dc;
  int32_t ytob_dc;
  ImageI ytox_map;
  ImageI ytob_map;
};

struct AcStrategy {
  ImageB layers;  // <AcStrategyType>
  bool use_ac_strategy = false;
};

void UnapplyColorCorrelation(const ColorCorrelationMap& cmap,
                             const ImageF& y_plane, Image3F* coeffs);

struct GradientMap;

// Per-thread cache items kept in EncCache.
// Used to reduce memory pressure (let threads work with warm memory).
struct EncCachePerThread {
  bool initialized = false;

  // Temporary ile with (original - predicted) values.
  ImageF diff;
};

// Working area for ComputeCoefficients; avoids duplicated work when called
// multiple times.
struct EncCache {
  bool initialized = false;

  bool is_smooth;
  bool use_gradient;
  bool grayscale_opt = false;
  size_t xsize_blocks;
  size_t ysize_blocks;

  // Original image; image should not be modified or discarded
  // until the last invocation of ComputeCoefficients.
  const Image3F* src;

  // DCT [with optional preprocessing that depends only on DC]
  Image3F coeffs_init;

  Image3F dc_init;

  // Working value, copied from coeffs_init.
  Image3F coeffs;

  // QuantDcKey() value for which cached results are valid.
  uint32_t last_quant_dc_key = ~0u;

  // ComputePredictionResiduals
  Image3F dc_dec;

  // ComputePredictionResiduals
  Image3F coeffs_dec;

  // ComputePredictionResiduals_Smooth
  Image3F dc_sharp;

  // Gradient map. Empty if not used.
  Image3F gradient;

  // AC strategy.
  AcStrategy ac_strategy;

  // Output values
  Image3S dc;
  Image3S ac;          // 64 coefs per block, first (DC) is ignored.
  ImageI quant_field;  // Final values, to be encoded in stream.

  std::vector<EncCachePerThread> buffers;
};

// Toy AC strategy.
void StrategyResampleAll(ImageB* image);

// Reference to |opsin| might be stored -> image should not be modified or
// discarded until the last invocation of ComputeCoefficients.
void ComputeInitialCoefficients(const Header& header, const Image3F& opsin,
                                ThreadPool* pool, EncCache* cache);

void ComputeCoefficients(const Quantizer& quantizer,
                         const ColorCorrelationMap& cmap, ThreadPool* pool,
                         EncCache* cache, const PikInfo* aux_out = nullptr);

// The gradient may be image with dimension 0 if disabled.
PaddedBytes EncodeToBitstream(const EncCache& cache, const Quantizer& quantizer,
                              const Image3F& gradient,
                              const NoiseParams& noise_params,
                              const ColorCorrelationMap& cmap, bool fast_mode,
                              PikInfo* info = nullptr);

struct DecCache {
  // If false, DequantImage should be invoked before ReconOpsinImage; otherwise
  // it is assumed that someone else (i.e. DecodeFromBitstream) did it already.
  bool eager_dequant = false;

  // Only used if !eager_dequant
  Image3S quantized_dc;
  Image3S quantized_ac;

  // Dequantized output produced by DecodeFromBitstream (if eager_dequant) or
  // DequantImage.
  Image3F dc;
  Image3F ac;

  AcStrategy ac_strategy;

  Image3F gradient;
};

// "compressed" is the same range from which reader was constructed, and allows
// seeking to tiles and constructing per-thread BitReader.
// Writes to (cache->eager_dequant ? cache->dc/ac : cache->quantized_dc/ac).
bool DecodeFromBitstream(const Header& header, const PaddedBytes& compressed,
                         BitReader* reader, const size_t xsize_blocks,
                         const size_t ysize_blocks, ThreadPool* pool,
                         ColorCorrelationMap* cmap, NoiseParams* noise_params,
                         Quantizer* quantizer, DecCache* cache);

// Dequantizes AC and DC coefficients.
void DequantImage(const Quantizer& quantizer, const ColorCorrelationMap& cmap,
                  ThreadPool* pool, DecCache* cache);

// Optionally does DC preconditioning, performs IDCT, and
// optionally applies image post-processing.
Image3F ReconOpsinImage(const Header& header, const Quantizer& quantizer,
                        ThreadPool* pool, DecCache* cache,
                        PikInfo* pik_info = nullptr);

void GaborishInverse(Image3F& opsin, double strength);
Image3F ConvolveGaborish(Image3F&& in, double strength, ThreadPool* pool);

ImageF IntensityAcEstimate(const ImageF& image, float multiplier,
                           ThreadPool* pool);

}  // namespace pik

#endif  // COMPRESSED_IMAGE_H_
