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

#ifndef QUANTIZER_H_
#define QUANTIZER_H_

#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>
#include <array>
#include <cmath>
#include <string>
#include <vector>

#include "bit_reader.h"
#include "compiler_specific.h"
#include "dct.h"
#include "image.h"
#include "pik_info.h"

namespace pik {

class Quantizer {
 public:
  Quantizer(int quant_xsize, int quant_ysize);

  void Copy(const Quantizer& other) {
    PIK_CHECK(other.initialized_);
    global_scale_ = other.global_scale_;
    quant_dc_ = other.quant_dc_;
    inv_global_scale_ = other.inv_global_scale_;
    inv_quant_dc_ = other.inv_quant_dc_;
    initialized_ = true;
    quant_img_ac_ = CopyImage(other.quant_img_ac_);
    scale_ = CopyImage3(other.scale_);
  }

  bool SetQuantField(const float quant_dc, const ImageF& qf);
  void GetQuantField(float* quant_dc, ImageF* qf);

  void SetQuant(float quant) {
    SetQuantField(quant, ImageF(quant_xsize_, quant_ysize_, quant));
  }

  float inv_quant_patch() const { return inv_quant_patch_; }
  float inv_quant_dc() const { return inv_quant_dc_; }
  float inv_quant_ac(int quant_x, int quant_y) const {
    return inv_global_scale_ / quant_img_ac_.Row(quant_y)[quant_x];
  }

  const Image3F& scale() const { return scale_; }

  void QuantizeBlock(int quant_x, int quant_y, int c,
                     const float* PIK_RESTRICT block_in,
                     int16_t* PIK_RESTRICT block_out) const {
    const float* const PIK_RESTRICT scale =
        &scale_.Row(quant_y)[c][quant_x * 64];
    for (int k = 0; k < 64; ++k) {
      const float val = block_in[k] * scale[k];
      static const float kZeroBias[3] = { 0.65f, 0.6f, 0.7f };
      const float thres = kZeroBias[c];
      block_out[k] = (k > 0 && std::abs(val) < thres) ? 0 : std::round(val);
    }
  }

  // Returns the sum of squared errors in pixel space (i.e. after applying
  // ComputeTransposedScaledBlockIDCTFloat() on error) that would result from
  // quantizing and dequantizing the given block.
  float PixelErrorSquared(int qx, int qy, int c,
                          const float* PIK_RESTRICT block_in) const {
    const float* const PIK_RESTRICT scale = &scale_.Row(qy)[c][qx * 64];
    int16_t coeffs[64];
    QuantizeBlock(qx, qy, c, block_in, coeffs);
    float sumsq = 0.0f;
    for (int ky = 0, k = 0; ky < 8; ++ky) {
      for (int kx = 0; kx < 8; ++kx, ++k) {
        const float idct_scale = 1.0f / (kIDCTScales[kx] * kIDCTScales[ky]);
        const float e = (block_in[k] - coeffs[k] / scale[k]) * idct_scale;
        sumsq += e * e;
      }
    }
    return sumsq;
  }

  std::string Encode(PikImageSizeInfo* info) const;

  bool Decode(BitReader* br);

  void DumpQuantizationMap() const;

 private:
  const int quant_xsize_;
  const int quant_ysize_;
  int global_scale_;
  int quant_patch_;
  int quant_dc_;
  Image<int> quant_img_ac_;
  float inv_global_scale_;
  float inv_quant_patch_;
  float inv_quant_dc_;
  // Scaled quantization multipliers, one for each channel.
  Image3F scale_;
  bool initialized_ = false;
};

const float* DequantMatrix();

Image3W QuantizeCoeffs(const Image3F& in, const Quantizer& quantizer);
Image3F DequantizeCoeffs(const Image3W& in, const Quantizer& quantizer);

}  // namespace pik

#endif  // QUANTIZER_H_
