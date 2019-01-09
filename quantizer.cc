// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "quantizer.h"

#include <stdio.h>
#include <algorithm>
#include <sstream>
#include <vector>

#undef PROFILER_ENABLED
#define PROFILER_ENABLED 1
#include "arch_specific.h"
#include "cache_aligned.h"
#include "common.h"
#include "compiler_specific.h"
#include "dct.h"
#include "dct_util.h"
#include "profiler.h"
#include "quant_weights.h"
#include "simd/simd.h"

namespace pik {

static const int kDefaultQuant = 64;

const float* NewDequantMatrices() {
  constexpr int N = kBlockDim;
  constexpr int block_size = N * N;
  constexpr int table_size = 3 * block_size;
  const float* idct4_scales = IDCTScales<N / 2>();
  const float* idct_scales = IDCTScales<N>();
  const float* idct16_scales = IDCTScales<2 * N>();
  const float* idct32_scales = IDCTScales<4 * N>();
  float* table = static_cast<float*>(CacheAligned::Allocate(
      kNumQuantTables * kNumQuantKinds * table_size * sizeof(float)));

  static_assert(kNumQuantTables == 1,
                "Update this function when adding quantization tables.");

  static_assert(kNumQuantKinds == 23,
                "Update this function when adding new quantization kinds.");

  // DCT8 (quant_kind 0)
  {
    const double* quant_weights = GetQuantWeightsDCT8();
    for (size_t c = 0; c < 3; c++) {
      for (size_t i = 0; i < N * N; i++) {
        double weight = quant_weights[c * block_size + i];
        const size_t x = i % N;
        const size_t y = i / N;
        const float idct_scale = idct_scales[x] * idct_scales[y] / block_size;
        table[DequantMatrixOffset(0, kQuantKindDCT8, c) * block_size + i] =
            idct_scale / weight;
      }
    }
  }

  // Identity (quant_kind 1)
  {
    const double* quant_weights = GetQuantWeightsIdentity();
    for (size_t c = 0; c < 3; c++) {
      for (size_t i = 0; i < N * N; i++) {
        double weight = quant_weights[c * block_size + i];
        table[DequantMatrixOffset(0, kQuantKindID, c) * block_size + i] =
            1.0f / weight;
      }
    }
  }

  // DCT4 (quant_kind 2)
  {
    double weight01[3] = {};
    double weight11[3] = {};
    const double* quant_weights = GetQuantWeightsDCT4(weight01, weight11);
    for (size_t c = 0; c < 3; c++) {
      for (size_t i = 0; i < N * N; i++) {
        const size_t x = i % N;
        const size_t y = i / N;
        double weight =
            quant_weights[c * (N / 2 * N / 2) + (y / 2 * N / 2) + x / 2];
        float idct_scale =
            idct4_scales[x / 2] * idct4_scales[y / 2] / (N / 2 * N / 2);
        if (i == 1 || i == N) idct_scale *= weight01[c];
        if (i == N + 1) idct_scale *= weight11[c];
        table[DequantMatrixOffset(0, kQuantKindDCT4, c) * block_size + i] =
            idct_scale / weight;
      }
    }
  }

  // DCT16 (quant_kind 3 to 6)
  {
    const double* quant_weights = GetQuantWeightsDCT16();
    for (size_t c = 0; c < 3; c++) {
      float dct16_table[4 * N * N] = {};
      for (size_t i = 0; i < 4 * N * N; i++) {
        double weight = quant_weights[c * 4 * block_size + i];
        const size_t x = i & (2 * N - 1);
        const size_t y = i / (2 * N);
        const float idct_scale =
            idct16_scales[x] * idct16_scales[y] / (4 * block_size);
        dct16_table[i] = idct_scale / weight;
      }
      ScatterBlock<2 * N, 2 * N>(
          dct16_table,
          &table[DequantMatrixOffset(0, kQuantKindDCT16Start, c) * block_size],
          2 * N * N);
    }
  }

  // DCT32 (quant_kind 7 to 22)
  {
    const double* quant_weights = GetQuantWeightsDCT32();
    for (size_t c = 0; c < 3; c++) {
      float dct32_table[16 * N * N] = {};
      for (size_t i = 0; i < 16 * N * N; i++) {
        double weight = quant_weights[c * 16 * block_size + i];
        const size_t x = i & (4 * N - 1);
        const size_t y = i / (4 * N);
        const float idct_scale =
            idct32_scales[x] * idct32_scales[y] / (16 * block_size);
        dct32_table[i] = idct_scale / weight;
      }
      ScatterBlock<4 * N, 4 * N>(
          dct32_table,
          &table[DequantMatrixOffset(0, kQuantKindDCT32Start, c) * block_size],
          4 * N * N);
    }
  }
  return table;
}

// Returns aligned memory.
const float* DequantMatrix(int id, size_t quant_kind, int c) {
  const constexpr size_t N = kBlockDim;
  PIK_ASSERT(quant_kind < kNumQuantKinds);
  static const float* const kDequantMatrix = NewDequantMatrices();
  return &kDequantMatrix[DequantMatrixOffset(id, quant_kind, c) * N * N];
}

Quantizer::Quantizer(size_t block_dim, int template_id, int quant_xsize,
                     int quant_ysize)
    : Quantizer(block_dim, template_id, quant_xsize, quant_ysize, kDefaultQuant,
                kGlobalScaleDenom / kDefaultQuant) {}

Quantizer::Quantizer(size_t block_dim, int template_id, int quant_xsize,
                     int quant_ysize, int quant_dc, int global_scale)
    : block_dim_(block_dim),
      quant_xsize_(quant_xsize),
      quant_ysize_(quant_ysize),
      template_id_(template_id % kNumQuantTables),
      global_scale_(global_scale),
      quant_dc_(quant_dc),
      quant_img_ac_(quant_xsize_, quant_ysize_),
      initialized_(false) {
  // DCT16 still has block_dim = 8.
  PIK_ASSERT(block_dim_ == 8);
  RecomputeFromGlobalScale();

  const size_t block_size = block_dim_ * block_dim_;
  quant_matrix_.resize(3 * kNumQuantKinds * block_size);

  FillImage(kDefaultQuant, &quant_img_ac_);

  PIK_ASSERT(template_id < kNumQuantTables);
  memcpy(zero_bias_, kZeroBiasDefault, sizeof(kZeroBiasDefault));
}

const float* Quantizer::DequantMatrix(int c, size_t quant_kind) const {
  return pik::DequantMatrix(template_id_, quant_kind, c);
}

std::string Quantizer::Encode(PikImageSizeInfo* info) const {
  std::stringstream ss;
  static_assert(kNumQuantTables <= 2, "template_id is supposed to be 1 bit");
  int global_scale_and_template_id = (global_scale_ - 1) | (template_id_ << 15);
  ss << std::string(1, global_scale_and_template_id >> 8);
  ss << std::string(1, global_scale_and_template_id & 0xff);
  ss << std::string(1, quant_dc_ - 1);
  if (info) {
    info->total_size += 3;
  }
  return ss.str();
}

bool Quantizer::Decode(BitReader* br) {
  int global_scale_and_template_id = br->ReadBits(8) << 8;
  global_scale_and_template_id += br->ReadBits(8);
  quant_dc_ = br->ReadBits(8) + 1;
  global_scale_ = (global_scale_and_template_id & 0x7FFF) + 1;
  template_id_ = global_scale_and_template_id >> 15;
  RecomputeFromGlobalScale();
  inv_quant_dc_ = inv_global_scale_ / quant_dc_;
  initialized_ = true;
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
      quantizer.DequantMatrix(c, kQuantKindDCT8)[0] * quantizer.inv_quant_dc();
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
