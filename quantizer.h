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
#include "image.h"

namespace pik {

struct AdaptiveQuantParams {
  float initial_quant_val_dc;
  float initial_quant_val_ac;
};

class Quantizer {
 public:
  Quantizer(int xsize, int ysize,
            const int coeffs_per_block,
            const float* dequant_matrix);

  bool SetQuantField(const float quant_dc, const ImageF& qf);
  void GetQuantField(float* quant_dc, ImageF* qf);

  void SetQuant(float quant) {
    SetQuantField(quant, ImageF(quant_xsize_, quant_ysize_, quant));
  }

  float inv_quant_dc() const { return inv_quant_dc_; }
  float inv_quant_ac(int quant_x, int quant_y) const {
    return inv_global_scale_ / quant_img_ac_.Row(quant_y)[quant_x];
  }

  void QuantizeBlock(int quant_x, int quant_y,
                     int c, int k_start, int k_end,
                     const float* PIK_RESTRICT block_in,
                     int16_t* PIK_RESTRICT block_out) const {
    const float* const PIK_RESTRICT scale =
        &scale_.Row(quant_y)[c][quant_x * coeffs_per_block_];
    for (int k = k_start; k < k_end; ++k) {
      const float val = block_in[k] * scale[k];
      static const float kZeroBias[3] = { 0.65f, 0.6f, 0.7f };
      const float thres = kZeroBias[c];
      block_out[k] = (k > 0 && std::abs(val) < thres) ? 0 : std::round(val);
    }
  }

  std::string Encode() const;
  size_t EncodedSize() const;

  bool Decode(BitReader* br);

  void DumpQuantizationMap() const;

 private:
  const int quant_xsize_;
  const int quant_ysize_;
  const int coeffs_per_block_;
  const float* const dequant_matrix_;
  uint16_t global_scale_;
  int quant_dc_;
  Image<int> quant_img_ac_;
  float inv_global_scale_;
  float inv_quant_dc_;
  // Scaled quantization multipliers, one for each channel.
  Image3F scale_;
  bool initialized_ = false;
};

}  // namespace pik

#endif  // QUANTIZER_H_
