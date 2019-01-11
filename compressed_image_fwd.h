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

// Contains global information that are computed once per pass.
struct PassEncCache {
  // DCT coefficients for the full image
  Image3F coeffs;

  Image3F dc_dec;
  Image3S dc;

  // Enable new Lossless codec for DC. This flag exists only temporarily
  // as long as both old and new implementation co-exist, and eventually
  // only the new implementation should remain.
  bool use_new_dc = false;

  bool use_gradient;
  bool grayscale_opt = false;
  // Gradient map, if used.
  GradientMap gradient;
};

// Working area for ComputeCoefficients
struct EncCache {
  bool initialized = false;

  bool grayscale_opt = false;

  size_t xsize_blocks;
  size_t ysize_blocks;

  // ComputePredictionResiduals
  Image3F dc_dec;

  // Working value, copied from coeffs_init.
  Image3F coeffs;

  // AC strategy.
  AcStrategyImage ac_strategy;

  // Every cell with saliency > threshold will be considered as 'salient'.
  float saliency_threshold;
  // Debug parameter: If true, drop non-salient AC part in progressive encoding.
  bool saliency_debug_skip_nonsalient;

  // Enable/disable predictions. Set in ComputeInitialCoefficients from the
  // pass header. Current usage is only in progressive mode.
  bool predict_lf;
  bool predict_hf;

  // Output values
  Image3S ac;          // 64 coefs per block, first (DC) is ignored.
  ImageI quant_field;  // Final values, to be encoded in stream.
};

struct DecCache {
  // Dequantized output produced by DecodeFromBitstream, DequantImage or
  // ExtractGroupDC.
  // TODO(veluca): replace the DC with a pointer + a rect to avoid copies.
  Image3F dc;
  Image3F ac;
};

// Information that is used at the pass level. All the images here should be
// accessed through a group rect (either with block units or pixel units).
struct PassDecCache {
  // Enable new Lossless codec for DC. This flag exists only temporarily
  // as long as both old and new implementation co-exist, and eventually
  // only the new implementation should remain.
  bool use_new_dc = false;

  bool grayscale;

  // Bias that was used for dequantization of the corresponding coefficient.
  // Note that the code that stores the biases relies on the fact that DC biases
  // are 0.
  Image3F biases;

  // Full DC of the pass. Note that this will be split in *AC* group sized
  // chunks for AC predictions (DC group size != AC group size).
  Image3F dc;

  GradientMap gradient;

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
