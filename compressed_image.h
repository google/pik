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
#include <string>

#include "bit_reader.h"
#include "common.h"
#include "header.h"
#include "image.h"
#include "noise.h"
#include "padded_bytes.h"
#include "pik_info.h"
#include "pik_params.h"
#include "quantizer.h"
#include "tile_flow.h"

namespace pik {

Image3F AlignImage(const Image3F& in, const size_t N);

void CenterOpsinValues(Image3F* img);

struct ColorTransform {
  ColorTransform(size_t xsize, size_t ysize)
      : ytob_dc(120),
        ytox_dc(128),
        ytob_map(DivCeil(xsize, kTileWidth), DivCeil(ysize, kTileHeight), 120),
        ytox_map(DivCeil(xsize, kTileWidth), DivCeil(ysize, kTileHeight), 128) {}
  int ytob_dc;
  int ytox_dc;
  Image<int> ytob_map;
  Image<int> ytox_map;
};

struct QuantizedCoeffs {
  Image3S dc;
  Image3S ac;  // 64 coefs per block, first (DC) is ignored.
};

void ComputePredictionResiduals(const Quantizer& quantizer, int flags,
                                Image3F* coeffs);

void ApplyColorTransform(const ColorTransform& ctan, const float factor,
                         const ImageF& y_plane, Image3F* coeffs);

// Working area for ComputeCoefficients; avoids duplicated work when called
// multiple times.
struct EncCache {
  bool have_coeffs_init = false;
  // DCT [with optional preprocessing that depends only on DC]
  Image3F coeffs_init;

  // Working value, copied from coeffs_init.
  Image3F coeffs;

  bool have_pred = false;

  // ComputePredictionResiduals
  Image3F dc_dec;

  // ComputePredictionResiduals_Smooth
  Image3F dc_sharp;
  Image3F pred_smooth;
};

QuantizedCoeffs ComputeCoefficients(const CompressParams& params,
                                    const Header& header, const Image3F& opsin,
                                    const Quantizer& quantizer,
                                    const ColorTransform& ctan,
                                    ThreadPool* pool,
                                    EncCache* cache,
                                    const PikInfo* aux_out = nullptr);

PaddedBytes EncodeToBitstream(const QuantizedCoeffs& qcoeffs,
                              const Quantizer& quantizer,
                              const NoiseParams& noise_params,
                              const ColorTransform& ctan, bool fast_mode,
                              PikInfo* info = nullptr);

struct DecCache {
  Image3S dc_quant;
  Image3F dc;

  bool eager_dc_dequant = false;
  float mul_dc[3];
  float ytox;
  float ytob;

  Image3S ac_quant;
};

// "compressed" is the same range from which reader was constructed, and allows
// seeking to tiles and constructing per-thread BitReader.
bool DecodeFromBitstream(const PaddedBytes& compressed, BitReader* reader,
                         const size_t xsize, const size_t ysize,
                         ThreadPool* pool, ColorTransform* ctan,
                         NoiseParams* noise_params, Quantizer* quantizer,
                         DecCache* cache);

Image3F ReconOpsinImage(const Header& header, const Quantizer& quantizer,
                        const ColorTransform& ctan, ThreadPool* pool,
                        DecCache* cache, PikInfo* pik_info = nullptr);

void GaborishInverse(Image3F& opsin);
Image3F ConvolveGaborish(const Image3F& in, ThreadPool* pool);
Image3F ConvolveGaborishTF(const Image3F& in, ThreadPool* pool);

}  // namespace pik

#endif  // COMPRESSED_IMAGE_H_
