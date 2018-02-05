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

#include "common.h"
#include "image.h"
#include "noise.h"
#include "pik_info.h"
#include "pik_params.h"
#include "quantizer.h"

namespace pik {

Image3F AlignImage(const Image3F& in, const size_t N);

void CenterOpsinValues(Image3F* img);

struct ColorTransform {
  ColorTransform(size_t xsize, size_t ysize)
      : ytob_dc(120), ytox_dc(128),
        ytob_map(DivCeil(xsize, kTileSize),
                 DivCeil(ysize, kTileSize), 120),
        ytox_map(DivCeil(xsize, kTileSize),
                 DivCeil(ysize, kTileSize), 128) {}
  int ytob_dc;
  int ytox_dc;
  Image<int> ytob_map;
  Image<int> ytox_map;
};

struct QuantizedCoeffs {
  Image3W dct;
};

void ComputePredictionResiduals(const Quantizer& quantizer,
                                Image3F* coeffs,
                                Image3F* predicted_coeffs);

void ApplyColorTransform(const ColorTransform& ctan,
                         const float factor,
                         const ImageF& y_plane,
                         Image3F* coeffs);

QuantizedCoeffs ComputeCoefficients(const CompressParams& params,
                                    const Image3F& opsin,
                                    const Quantizer& quantizer,
                                    const ColorTransform& ctan,
                                    const PikInfo* aux_out);

std::string EncodeToBitstream(const QuantizedCoeffs& qcoeffs,
                              const Quantizer& quantizer,
                              const NoiseParams& noise_params,
                              const ColorTransform& ctan,
                              bool fast_mode,
                              PikInfo* info);

bool DecodeFromBitstream(const uint8_t* data, const size_t data_size,
                         const size_t xsize, const size_t ysize,
                         ColorTransform* ctan,
                         NoiseParams* noise_params,
                         Quantizer* quantizer,
                         QuantizedCoeffs* qcoeffs,
                         size_t* compressed_size);

// Last optional argument receives (if non-null) the scaled DCT coefficients
// before the final IDCT.
Image3F ReconOpsinImage(const QuantizedCoeffs& qcoeffs,
                        const Quantizer& quantizer, const ColorTransform& ctan,
                        Image3F* transposed_scaled_dct = nullptr);

}  // namespace pik

#endif  // COMPRESSED_IMAGE_H_
