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

#include "quantizer.h"

#include <stdlib.h>
#include <vector>

#include "arch_specific.h"
#include "compiler_specific.h"
#include "dct.h"
#include "opsin_codec.h"
#include "vector256.h"

namespace pik {

static const int kGlobalScaleDenom = 1 << 16;
static const int kQuantMax = 256;
static const int kDefaultQuant = 64;

int ClampVal(int val) {
  return std::min(kQuantMax, std::max(1, val));
}

Quantizer::Quantizer(int quant_xsize, int quant_ysize, int coeffs_per_block,
                     const float* dequant_matrix) :
    quant_xsize_(quant_xsize),
    quant_ysize_(quant_ysize),
    coeffs_per_block_(coeffs_per_block),
    dequant_matrix_(dequant_matrix),
    global_scale_(kGlobalScaleDenom / kDefaultQuant),
    quant_dc_(kDefaultQuant),
    quant_img_ac_(quant_xsize_, quant_ysize_, kDefaultQuant),
    scale_(quant_xsize_ * coeffs_per_block_, quant_ysize_),
    initialized_(false) {
}

bool Quantizer::SetQuantField(const float quant_dc, const ImageF& qf) {
  bool changed = false;
  float range_min, range_max;
  ImageMinMax(qf, &range_min, &range_max);
  range_max = std::max(range_max, quant_dc);
  range_min = std::min(range_min, quant_dc);
  float range_dynamic = std::min(16.0f, range_max / range_min);
  // We want range_max to map to 8 * sqrt(range_dynamic) and therefore
  // range_min would map to 8 / sqrt(range_dynamic).
  float log_global_scale =
      std::log(range_max / (8.0 * sqrt(range_dynamic))) / std::log(2);
  int global_scale_shift = static_cast<int>(-log_global_scale + 0.5);
  int new_global_scale = kGlobalScaleDenom >> global_scale_shift;
  if (new_global_scale != global_scale_) {
    global_scale_ = new_global_scale;
    changed = true;
  }
  const float scale = global_scale_ * 1.0f / kGlobalScaleDenom;
  const float inv_scale = 1.0f / scale;
  int val = ClampVal(quant_dc * inv_scale + 0.5f);
  if (val != quant_dc_) {
    quant_dc_ = val;
    changed = true;
  }
  for (int y = 0; y < quant_ysize_; ++y) {
    for (int x = 0; x < quant_xsize_; ++x) {
      int val = ClampVal(qf.Row(y)[x] * inv_scale + 0.5f);
      if (val != quant_img_ac_.Row(y)[x]) {
        quant_img_ac_.Row(y)[x] = val;
        changed = true;
      }
    }
  }
  if (!initialized_) {
    changed = true;
  }
  if (changed) {
    const float* const PIK_RESTRICT kDequantMatrix = dequant_matrix_;
    std::vector<float> quant_matrix(3 * coeffs_per_block_);
    for (int i = 0; i < quant_matrix.size(); ++i) {
      quant_matrix[i] = 1.0f / (64.0f * kDequantMatrix[i]);
    }
    const float qdc = scale * quant_dc_;
    for (int y = 0; y < quant_ysize_; ++y) {
      auto row_q = quant_img_ac_.Row(y);
      auto row_scale = scale_.Row(y);
      for (int x = 0; x < quant_xsize_; ++x) {
        const int offset = x * coeffs_per_block_;
        const float qac = scale * row_q[x];
        for (int c = 0; c < 3; ++c) {
          const float* const PIK_RESTRICT qm =
              &quant_matrix[c * coeffs_per_block_];
          for (int k = 0; k < coeffs_per_block_; ++k) {
            row_scale[c][offset + k] = qac * qm[k];
          }
          row_scale[c][offset] = qdc * qm[0];
        }
      }
    }
    inv_global_scale_ = 1.0f / scale;
    inv_quant_dc_ = 1.0f / qdc;
    initialized_ = true;
  }
  return changed;
}

void Quantizer::GetQuantField(float* quant_dc, ImageF* qf) {
  const float scale = global_scale_ * 1.0f / kGlobalScaleDenom;
  *quant_dc = scale * quant_dc_;
  *qf = ImageF(quant_xsize_, quant_ysize_);
  for (int y = 0; y < quant_ysize_; ++y) {
    for (int x = 0; x < quant_xsize_; ++x) {
      qf->Row(y)[x] = scale * quant_img_ac_.Row(y)[x];
    }
  }
}

std::string Quantizer::Encode() const {
  return (std::string(1, global_scale_ >> 8) +
          std::string(1, global_scale_ & 0xff) + std::string(1, quant_dc_ - 1) +
          EncodePlane(quant_img_ac_, 1, kQuantMax));
}

size_t Quantizer::EncodedSize() const {
  return (3 + EncodedPlaneSize(quant_img_ac_, 1, kQuantMax));
}

size_t Quantizer::Decode(const uint8_t* data, size_t len) {
  size_t pos = 0;
  global_scale_ = data[pos++] << 8;
  global_scale_ += data[pos++];
  quant_dc_ = data[pos++] + 1;
  pos += DecodePlane(data + pos, len - pos, 1, kQuantMax, &quant_img_ac_);
  inv_global_scale_ = kGlobalScaleDenom * 1.0 / global_scale_;
  inv_quant_dc_ = inv_global_scale_ / quant_dc_;
  initialized_ = true;
  return pos;
}

void Quantizer::DumpQuantizationMap() const {
  printf("Global scale: %d (%.7f)\nDC quant: %d\n", global_scale_,
         global_scale_ * 1.0 / kGlobalScaleDenom, quant_dc_);
  printf("AC quantization Map:\n");
  for (size_t y = 0; y < quant_img_ac_.ysize(); ++y) {
    for (size_t x = 0; x < quant_img_ac_.xsize(); ++x) {
      printf(" %3d", quant_img_ac_.Row(y)[x]);
    }
    printf("\n");
  }
}

}  // namespace pik
