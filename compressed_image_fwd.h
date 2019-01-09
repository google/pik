// Copyright 2018 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef COMPRESSED_IMAGE_FWD_H_
#define COMPRESSED_IMAGE_FWD_H_

#include "ac_strategy.h"
#include "common.h"
#include "data_parallel.h"
#include "gauss_blur.h"
#include "image.h"

namespace pik {

struct GradientMap {
  Image3F gradient;  // corners of the gradient map tiles

  // Size of the DC image
  size_t xsize_dc;
  size_t ysize_dc;

  // Size of the gradient map (amount of corner points of tiles, one larger than
  // amount of tiles in x and y direction)
  size_t xsize;
  size_t ysize;

  bool grayscale;
};

// Working area for ComputeCoefficients; avoids duplicated work when called
// multiple times.
struct EncCache {
  bool initialized = false;

  // Enable new Lossless codec for DC. This flag exists only temporarily
  // as long as both old and new implementation co-exist, and eventually
  // only the new implementation should remain.
  bool use_new_dc = false;

  bool use_gradient;
  bool grayscale_opt = false;
  size_t xsize_blocks;
  size_t ysize_blocks;

  // Original image.
  Image3F src;

  // DCT [with optional preprocessing that depends only on DC]
  Image3F coeffs_init;

  Image3F dc_init;

  // Working value, copied from coeffs_init.
  Image3F coeffs;

  // QuantDcKey() value for which cached results are valid.
  uint32_t last_quant_dc_key = ~0u;

  // ComputePredictionResiduals
  Image3F dc_dec;

  // Gradient map, if used.
  GradientMap gradient;

  // AC strategy.
  AcStrategyImage ac_strategy;

  // Every cell with saliency > threshold will be considered as 'salient'.
  float saliency_threshold;
  // Debug parameter: If true, drop non-salient AC part in progressive encoding.
  bool saliency_debug_skip_nonsalient;

  // Enable/disable predictions. Set in ComputeInitialCoefficients from the
  // group header. Current usage is only in progressive mode.
  bool predict_lf;
  bool predict_hf;

  // Output values
  Image3S dc;
  Image3S ac;          // 64 coefs per block, first (DC) is ignored.
  ImageI quant_field;  // Final values, to be encoded in stream.
};

struct DecCache {
  // Enable new Lossless codec for DC. This flag exists only temporarily
  // as long as both old and new implementation co-exist, and eventually
  // only the new implementation should remain.
  bool use_new_dc = false;

  // Only used in encoder loop.
  Image3S quantized_dc;
  Image3S quantized_ac;

  // Dequantized output produced by DecodeFromBitstream or DequantImage.
  Image3F dc;
  Image3F ac;

  GradientMap gradient;
};

// Information that is used at the pass level. All the images here should be
// accessed through a group rect (either with block units or pixel units).
struct PassDecCache {
  // Bias that was used for dequantization of the corresponding coefficient.
  // Note that the code that stores the biases relies on the fact that DC biases
  // are 0.
  Image3F biases;

  // Raw quant field to be used for adaptive reconstruction.
  ImageI raw_quant_field;

  AcStrategyImage ac_strategy;
};

template <size_t N>
std::vector<float> DCfiedGaussianKernel(float sigma) {
  std::vector<float> result(3, 0.0);
  std::vector<float> hires = GaussianKernel<float>(N, sigma);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < hires.size(); j++) {
      result[(i + j) / N] += hires[j] / N;
    }
  }
  return result;
}

}  // namespace pik

#endif  // COMPRESSED_IMAGE_FWD_H_
