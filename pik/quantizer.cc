// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/quantizer.h"

#include <stdio.h>
#include <algorithm>
#include <sstream>
#include <vector>
#include "pik/ac_strategy.h"

#undef PROFILER_ENABLED
#define PROFILER_ENABLED 1
#include "pik/arch_specific.h"
#include "pik/common.h"
#include "pik/compiler_specific.h"
#include "pik/dct.h"
#include "pik/dct_util.h"
#include "pik/profiler.h"
#include "pik/quant_weights.h"
#include "pik/simd/simd.h"

namespace pik {

static const int kDefaultQuant = 64;

Quantizer::Quantizer(const DequantMatrices* dequant, int quant_xsize,
                     int quant_ysize)
    : Quantizer(dequant, quant_xsize, quant_ysize, kDefaultQuant,
                kGlobalScaleDenom / kDefaultQuant) {}

Quantizer::Quantizer(const DequantMatrices* dequant, int quant_xsize,
                     int quant_ysize, int quant_dc, int global_scale)
    : quant_xsize_(quant_xsize),
      quant_ysize_(quant_ysize),
      global_scale_(global_scale),
      quant_dc_(quant_dc),
      quant_img_ac_(quant_xsize_, quant_ysize_),
      dequant_(dequant) {
  RecomputeFromGlobalScale();

  FillImage(kDefaultQuant, &quant_img_ac_);

  memcpy(zero_bias_, kZeroBiasDefault, sizeof(kZeroBiasDefault));
}

// TODO(veluca): reclaim the unused bit in global_scale encoding.
std::string Quantizer::Encode(PikImageSizeInfo* info) const {
  std::stringstream ss;
  int global_scale = global_scale_ - 1;
  ss << std::string(1, global_scale >> 8);
  ss << std::string(1, global_scale & 0xff);
  ss << std::string(1, quant_dc_ - 1);
  if (info) {
    info->total_size += 3;
  }
  return ss.str();
}

bool Quantizer::Decode(BitReader* br) {
  int global_scale = br->ReadBits(8) << 8;
  global_scale |= br->ReadBits(8);
  global_scale_ = (global_scale & 0x7FFF) + 1;
  quant_dc_ = br->ReadBits(8) + 1;
  RecomputeFromGlobalScale();
  inv_quant_dc_ = inv_global_scale_ / quant_dc_;
  return true;
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

// Works in "DC image", i.e. transforms every pixel.
Image3S QuantizeCoeffsDC(const Image3F& dc, const Quantizer& quantizer) {
  const size_t xsize_blocks = dc.xsize();
  const size_t ysize_blocks = dc.ysize();
  Image3S out(xsize_blocks, ysize_blocks);
  for (int c = 0; c < 3; ++c) {
    for (size_t by = 0; by < ysize_blocks; ++by) {
      const float* PIK_RESTRICT row_in = dc.PlaneRow(c, by);
      int16_t* PIK_RESTRICT row_out = out.PlaneRow(c, by);
      for (size_t bx = 0; bx < xsize_blocks; ++bx) {
        row_out[bx] = quantizer.QuantizeDC(c, row_in[bx]);
      }
    }
  }
  return out;
}

ImageF QuantizeRoundtripDC(const Quantizer& quantizer, int c,
                           const ImageF& dc) {
  // All coordinates are blocks.
  const int xsize_blocks = dc.xsize();
  const int ysize_blocks = dc.ysize();
  ImageF out(xsize_blocks, ysize_blocks);

  // Always use DCT8 quantization kind for DC
  const float mul =
      quantizer.DequantMatrix(kQuantKindDCT8, c)[0] * quantizer.inv_quant_dc();
  for (size_t by = 0; by < ysize_blocks; ++by) {
    const float* PIK_RESTRICT row_in = dc.ConstRow(by);
    float* PIK_RESTRICT row_out = out.Row(by);
    for (size_t bx = 0; bx < xsize_blocks; ++bx) {
      row_out[bx] = quantizer.QuantizeDC(c, row_in[bx]) * mul;
    }
  }
  return out;
}

}  // namespace pik
