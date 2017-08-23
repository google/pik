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

#include "compressed_image.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <array>

#include "bit_reader.h"
#include "cache_aligned.h"
#include "compiler_specific.h"
#include "dc_predictor.h"
#include "dct.h"
#include "gamma_correct.h"
#include "image_io.h"
#include "opsin_codec.h"
#include "opsin_image.h"
#include "opsin_inverse.h"
#include "opsin_params.h"
#include "status.h"
#include "vector256.h"

namespace pik {

namespace {

static const int kBlockSize2 = 2 * kBlockSize;
static const int kBlockSize3 = 3 * kBlockSize;
static const int kLastCol = kBlockEdge - 1;
static const int kLastRow = kLastCol * kBlockEdge;
static const int kCoeffsPerBlock = kBlockSize;

static const float kDCBlurSigma = 3.0f;

int DivCeil(int a, int b) {
  return (a + b - 1) / b;
}

// Expects that values are in the interval [-range, range].
ImageB NormalizeAndClip(const ImageF& in, float range) {
  ImageB out(in.xsize(), in.ysize());
  const float scale = 255 / (2.0f * range);
  for (int y = 0; y < in.ysize(); ++y) {
    auto row_in = in.Row(y);
    auto row_out = out.Row(y);
    for (int x = 0; x < in.xsize(); ++x) {
      if (row_in[x] < -range || row_in[x] > range) {
        static int num_printed = 0;
        if (++num_printed < 100) {
          printf("Value %f outside range [%f, %f]\n", row_in[x], -range, range);
        }
      }
      row_out[x] = std::min(255, std::max(
          0, static_cast<int>(scale * (row_in[x] + range) + 0.5)));
    }
  }
  return out;
}

void DumpOpsin(const PikInfo* info, const Image3F& in,
               const std::string& label) {
  if (!info || info->debug_prefix.empty()) return;
  WriteImage(ImageFormatPNG(),
             Image3B(NormalizeAndClip(in.plane(0), 1.25f * kXybRange[0]),
                     NormalizeAndClip(in.plane(1), 1.50f * kXybRange[1]),
                     NormalizeAndClip(in.plane(2), 1.50f * kXybRange[2])),
             info->debug_prefix + label + ".png");
}

static const float kQuantizeMul[3] = { 2.631f, 0.780f, 0.125f };

// kQuantWeights[3 * k_zz + c] is the relative weight of the k_zz coefficient
// (in the zig-zag order) in component c. Higher weights correspond to finer
// quantization intervals and more bits spent in encoding.
static const float kQuantWeights[kBlockSize3] = {
  3.0000000f, 1.9500000f, 3.7000000f, 2.0000000f, 1.4000000f, 2.0000000f,
  2.0000000f, 1.4000000f, 2.0000000f, 1.4000000f, 1.4000000f, 1.2000000f,
  1.6000000f, 1.4000000f, 1.4000000f, 1.3000000f, 1.4000000f, 1.2000000f,
  1.0000000f, 1.0000000f, 1.0000000f, 1.0000000f, 1.0000000f, 1.0000000f,
  1.0000000f, 1.0000000f, 1.0000000f, 1.0000000f, 1.0000000f, 1.0000000f,
  0.8000000f, 0.8000000f, 0.8000000f, 0.8000000f, 0.8000000f, 0.8000000f,
  0.8000000f, 0.8000000f, 0.8000000f, 0.8000000f, 0.8000000f, 0.8000000f,
  0.8000000f, 0.8000000f, 0.8000000f, 0.8000000f, 0.8000000f, 0.8000000f,
  0.8000000f, 0.8000000f, 0.8000000f, 0.8000000f, 0.8000000f, 0.8000000f,
  0.8000000f, 0.8000000f, 0.8000000f, 0.8000000f, 0.8000000f, 0.8000000f,
  0.8000000f, 0.8000000f, 0.8000000f, 0.8000000f, 0.8000000f, 0.8000000f,
  0.8000000f, 0.8000000f, 0.8000000f, 0.8000000f, 0.8000000f, 0.8000000f,
  0.8000000f, 0.8000000f, 0.8000000f, 0.8000000f, 0.8000000f, 0.8000000f,
  0.8000000f, 0.8000000f, 0.8000000f, 0.8000000f, 0.8000000f, 0.8000000f,
  0.6200000f, 0.6200000f, 0.6200000f, 0.6200000f, 0.6200000f, 0.6200000f,
  0.6200000f, 0.6200000f, 0.6200000f, 0.6200000f, 0.6200000f, 0.6200000f,
  0.6200000f, 0.6200000f, 0.6200000f, 0.6200000f, 0.6200000f, 0.6200000f,
  0.6200000f, 0.6200000f, 0.6200000f, 0.6200000f, 0.6200000f, 0.6200000f,
  0.6200000f, 0.6200000f, 0.6200000f, 0.6200000f, 0.6200000f, 0.6200000f,
  0.6200000f, 0.6200000f, 0.6200000f, 0.6200000f, 0.6200000f, 0.6200000f,
  0.6200000f, 0.6200000f, 0.6200000f, 0.6200000f, 0.6200000f, 0.6200000f,
  0.6200000f, 0.6200000f, 0.6200000f, 0.6000000f, 0.6000000f, 0.6000000f,
  0.6000000f, 0.6000000f, 0.6000000f, 0.6000000f, 0.6000000f, 0.6000000f,
  0.6000000f, 0.6000000f, 0.6000000f, 0.6000000f, 0.6000000f, 0.6000000f,
  0.6000000f, 0.6000000f, 0.6000000f, 0.6000000f, 0.6000000f, 0.6000000f,
  0.6000000f, 0.6000000f, 0.6000000f, 0.6000000f, 0.6000000f, 0.6000000f,
  0.6000000f, 0.6000000f, 0.6000000f, 0.6000000f, 0.6000000f, 0.6000000f,
  0.6000000f, 0.6000000f, 0.6000000f, 0.6000000f, 0.6000000f, 0.6000000f,
  0.6000000f, 0.6000000f, 0.6000000f, 0.6000000f, 0.6000000f, 0.6000000f,
  0.6000000f, 0.6000000f, 0.6000000f, 0.6000000f, 0.6000000f, 0.6000000f,
  0.6000000f, 0.6000000f, 0.6000000f, 0.6000000f, 0.6000000f, 0.6000000f,
  0.6000000f, 0.6000000f, 0.6000000f, 0.6000000f, 0.6000000f, 0.6000000f,
};

float getmodifier(int i) {
  char buf[10];
  snprintf(buf, 10, "VAR%d", i);
  char* p = getenv(buf);
  if (p == nullptr) return 0.0;
  return static_cast<float>(atof(p));
}

const float* NewDequantMatrix() {
  float* table = static_cast<float*>(
      CacheAligned::Allocate(3 * kCoeffsPerBlock * sizeof(float)));
  for (int c = 0; c < 3; ++c) {
    for (int k_zz = 0; k_zz < kBlockSize; ++k_zz) {
      int k = kNaturalCoeffOrder[k_zz];
      int idx = k_zz * 3 + c;
      float idct_scale =
          kIDCTScales[k % kBlockEdge] * kIDCTScales[k / kBlockEdge] / 64.0f;
      float weight = kQuantWeights[idx];
      float modify = getmodifier(idx);
      PIK_CHECK(modify > -0.5 * weight);
      weight += modify;
      weight *= kQuantizeMul[c];
      table[c * kCoeffsPerBlock + k] = idct_scale / weight;
    }
  }
  return table;
}

const float* DequantMatrix() {
  static const float* const kDequantMatrix = NewDequantMatrix();
  return kDequantMatrix;
}

std::vector<float> GaussianKernel(int radius, float sigma) {
  std::vector<float> kernel(2 * radius + 1);
  const float scaler = -1.0 / (2 * sigma * sigma);
  for (int i = -radius; i <= radius; ++i) {
    kernel[i + radius] = std::exp(scaler * i * i);
  }
  return kernel;
}

void ComputeBlockBlurWeights(const float sigma,
                             float* const PIK_RESTRICT w0,
                             float* const PIK_RESTRICT w1,
                             float* const PIK_RESTRICT w2) {
  std::vector<float> kernel = GaussianKernel(kBlockEdge, sigma);
  float weight = 0.0f;
  for (int i = 0; i < kernel.size(); ++i) {
    weight += kernel[i];
  }
  float scale = 1.0f / weight;
  for (int k = 0; k < kBlockEdge; ++k) {
    const int split0 = kBlockEdge - k;
    const int split1 = 2 * kBlockEdge - k;
    for (int j = 0; j < split0; ++j) {
      w0[k] += kernel[j];
    }
    for (int j = split0; j < split1; ++j) {
      w1[k] += kernel[j];
    }
    for (int j = split1; j < kernel.size(); ++j) {
      w2[k] += kernel[j];
    }
    w0[k] *= scale;
    w1[k] *= scale;
    w2[k] *= scale;
  }
}

Image3F ComputeDCBlurX(const Image3W& dct_coeffs,
                       const float sigma,
                       const float inv_quant_dc,
                       const float ytob_dc,
                       const float* const PIK_RESTRICT dequant_matrix) {
  float w_prev[kBlockEdge] = { 0.0f };
  float w_cur[kBlockEdge] = { 0.0f };
  float w_next[kBlockEdge] = { 0.0f };
  ComputeBlockBlurWeights(sigma, w_prev, w_cur, w_next);
  const float inv_scale[3] = {
    dequant_matrix[0] * inv_quant_dc,
    dequant_matrix[kBlockSize] * inv_quant_dc,
    dequant_matrix[kBlockSize2] * inv_quant_dc
  };
  const int block_xsize = dct_coeffs.xsize() / kBlockSize;
  const int block_ysize = dct_coeffs.ysize();
  Image3F out(block_xsize * kBlockEdge, block_ysize);
  for (int by = 0; by < block_ysize; ++by) {
    auto row_dc = dct_coeffs.Row(by);
    for (int c = 0; c < 3; ++c) {
      std::vector<float> row_tmp(block_xsize + 2);
      for (int bx = 0; bx < block_xsize; ++bx) {
        const int offset = bx * kBlockSize;
        float dc = row_dc[c][offset] * inv_scale[c];
        if (c == 2) {
          dc += row_dc[1][offset] * inv_scale[1] * ytob_dc;
        }
        row_tmp[bx + 1] = dc;
      }
      row_tmp[0] = row_tmp[1 + std::min(1, block_xsize - 1)];
      row_tmp[block_xsize + 1] = row_tmp[1 + std::max(0, block_xsize - 2)];
      float* const PIK_RESTRICT row_out = out.Row(by)[c];
      for (int bx = 0; bx < block_xsize; ++bx) {
        const float dc0 = row_tmp[bx];
        const float dc1 = row_tmp[bx + 1];
        const float dc2 = row_tmp[bx + 2];
        const int offset = bx * kBlockEdge;
        for (int ix = 0; ix < kBlockEdge; ++ix) {
          row_out[offset + ix] =
              dc0 * w_prev[ix] + dc1 * w_cur[ix] + dc2 * w_next[ix];
        }
      }
    }
  }
  return out;
}

PIK_INLINE float ComputeBlurredBlock(const Image3F& blur_x, int c, int offsetx,
                                     int y_up, int y_cur, int y_down,
                                     const float* const PIK_RESTRICT w_up,
                                     const float* const PIK_RESTRICT w_cur,
                                     const float* const PIK_RESTRICT w_down,
                                     float* const PIK_RESTRICT out) {
  const float* const PIK_RESTRICT row0 = &blur_x.Row(y_up)[c][offsetx];
  const float* const PIK_RESTRICT row1 = &blur_x.Row(y_cur)[c][offsetx];
  const float* const PIK_RESTRICT row2 = &blur_x.Row(y_down)[c][offsetx];
  float avg = 0.0f;
  for (int ix = 0; ix < kBlockEdge; ++ix) {
    const float val0 = row0[ix];
    const float val1 = row1[ix];
    const float val2 = row2[ix];
    for (int iy = 0; iy < kBlockEdge; ++iy) {
      const float val = val0 * w_up[iy] + val1 * w_cur[iy] + val2 * w_down[iy];
      out[iy * kBlockEdge + ix] = val;
      avg += val;
    }
  }
  avg /= 64.0f;
  return avg;
}

}  // namespace

CompressedImage::CompressedImage(int xsize, int ysize, PikInfo* info)
    : xsize_(xsize), ysize_(ysize),
      block_xsize_(DivCeil(xsize, kBlockEdge)),
      block_ysize_(DivCeil(ysize, kBlockEdge)),
      tile_xsize_(DivCeil(xsize, kTileEdge)),
      tile_ysize_(DivCeil(ysize, kTileEdge)),
      num_blocks_(block_xsize_ * block_ysize_),
      quantizer_(block_xsize_, block_ysize_, kCoeffsPerBlock, DequantMatrix()),
      dct_coeffs_(block_xsize_ * kBlockSize, block_ysize_),
      ytob_dc_(120),
      ytob_ac_(tile_xsize_, tile_ysize_, 120),
      pik_info_(info) {
}

// static
CompressedImage CompressedImage::FromOpsinImage(
    const Image3F& opsin, PikInfo* info) {
  CompressedImage img(opsin.xsize(), opsin.ysize(), info);
  const size_t xsize = kBlockEdge * img.block_xsize_;
  const size_t ysize = kBlockEdge * img.block_ysize_;
  img.opsin_image_.reset(new Image3F(xsize, ysize));
  int y = 0;
  for (; y < opsin.ysize(); ++y) {
    for (int c = 0; c < 3; ++c) {
      const float* const PIK_RESTRICT row_in = &opsin.Row(y)[c][0];
      float* const PIK_RESTRICT row_out = &img.opsin_image_->Row(y)[c][0];
      const float center = kXybCenter[c];
      int x = 0;
      for (; x < opsin.xsize(); ++x) {
        row_out[x] = row_in[x] - center;
      }
      const int lastcol = opsin.xsize() - 1;
      const float lastval = row_out[lastcol];
      for (; x < xsize; ++x) {
        row_out[x] = lastval;
      }
    }
  }
  const int lastrow = opsin.ysize() - 1;
  for (; y < ysize; ++y) {
    for (int c = 0; c < 3; ++c) {
      const float* const PIK_RESTRICT row_in =
          &img.opsin_image_->Row(lastrow)[c][0];
      float* const PIK_RESTRICT row_out = &img.opsin_image_->Row(y)[c][0];
      memcpy(row_out, row_in, xsize * sizeof(row_out[0]));
    }
  }
  DumpOpsin(info, *img.opsin_image_, "opsin_orig");
  return img;
}

// We modify the standard DCT of the intensity channel by further decorrelating
// the 1st and 3rd AC coefficients in the first row and first column. The
// unscaled prediction coefficient corresponds to the 1-d DCT of a linear slope.
static const float kACPredScale = 0.25f;
static const float kACPred31 = kACPredScale * 0.104536f;

void CompressedImage::QuantizeBlock(int block_x, int block_y) {
  const int offsetx = block_x * kBlockEdge;
  const int offsety = block_y * kBlockEdge;
  alignas(32) float block[kBlockSize3];
  for (int c = 0; c < 3; ++c) {
    float* const PIK_RESTRICT cblock = &block[kBlockSize * c];
    for (int iy = 0; iy < kBlockEdge; ++iy) {
      memcpy(&cblock[iy * kBlockEdge],
             &opsin_image_->Row(offsety + iy)[c][offsetx],
             kBlockEdge * sizeof(cblock[0]));
    }
  }
  if (opsin_overlay_.get() != nullptr) {
    const float* const PIK_RESTRICT overlay =
        &opsin_overlay_->Row(block_y)[3 * block_x * kBlockSize];
    for (int k = 0; k < kBlockSize3; ++k) {
      block[k] -= overlay[k];
    }
  }
  for (int c = 0; c < 3; ++c) {
    ComputeTransposedScaledBlockDCTFloat(&block[kBlockSize * c]);
  }
  // Remove some correlation between the 1st and 3rd AC coefficients in the
  // first column and first row.
  block[kBlockSize + 3] -= kACPred31 * block[kBlockSize + 1];
  block[kBlockSize + 24] -= kACPred31 * block[kBlockSize + 8];
  auto row_out = dct_coeffs_.Row(block_y);
  const int offset = block_x * kBlockSize;
  int16_t* const PIK_RESTRICT iblocky = &row_out[1][offset];
  const float inv_quant_dc = 64.0f * quantizer_.inv_quant_dc();
  const float inv_quant_ac = 64.0f * quantizer_.inv_quant_ac(block_x, block_y);
  const float* PIK_RESTRICT kDequantMatrix = DequantMatrix();
  quantizer_.QuantizeBlock(block_x, block_y, 1, 0, kBlockSize,
                           &block[kBlockSize], iblocky);
  for (int k = 0; k < kBlockSize; ++k) {
    block[kBlockSize + k] = iblocky[k] * kDequantMatrix[kBlockSize + k] *
        inv_quant_ac;
  }
  block[kBlockSize] = iblocky[0] * kDequantMatrix[kBlockSize] * inv_quant_dc;
  {
    const int tile_x = block_x / kTileToBlockRatio;
    const int tile_y = block_y / kTileToBlockRatio;
    const float ytob_ac = YToBAC(tile_x, tile_y);
    for (int k = 0; k < kBlockSize; ++k) {
      block[kBlockSize2 + k] -= ytob_ac * block[kBlockSize + k];
    }
    block[kBlockSize2] -= (YToBDC() - ytob_ac) * block[kBlockSize];
  }
  for (int c = 0; c < 3; ++c) {
    if (c == 1) {
      // y channel was already quantized
      continue;
    }
    quantizer_.QuantizeBlock(
        block_x, block_y, c, 0, kBlockSize,
        &block[c * kBlockSize], &row_out[c][offset]);
  }
}

void CompressedImage::QuantizeDC() {
  const float inv_quant_dc = 64.0f * quantizer_.inv_quant_dc();
  const float* PIK_RESTRICT kDequantMatrix = DequantMatrix();
  const float inv_scale[3] = {
    kDequantMatrix[0] * inv_quant_dc,
    kDequantMatrix[kBlockSize] * inv_quant_dc,
    kDequantMatrix[kBlockSize2] * inv_quant_dc
  };
  const float scale[3] = {
    1.0f / inv_scale[0], 1.0f / inv_scale[1], 1.0f / inv_scale[2]
  };
  for (int block_y = 0; block_y < block_ysize_; ++block_y) {
    for (int block_x = 0; block_x < block_xsize_; ++block_x) {
      const int offsetx = block_x * kBlockEdge;
      const int offsety = block_y * kBlockEdge;
      float dc[3] = { 0 };
      for (int c = 0; c < 3; ++c) {
        for (int ix = 0; ix < kBlockEdge; ++ix) {
          for (int iy = 0; iy < kBlockEdge; ++iy) {
            dc[c] += opsin_image_->Row(offsety + iy)[c][offsetx + ix];
          }
        }
      }
      auto row_out = dct_coeffs_.Row(block_y);
      const int offset = block_x * kBlockSize;
      row_out[0][offset] = std::round(dc[0] * scale[0]);
      row_out[1][offset] = std::round(dc[1] * scale[1]);
      dc[2] -= YToBDC() * row_out[1][offset] * inv_scale[1];
      row_out[2][offset] = std::round(dc[2] * scale[2]);
    }
  }
}

void CompressedImage::ComputeOpsinOverlay() {
  opsin_overlay_.reset(new ImageF(block_xsize_ * kBlockSize3, block_ysize_));
  const float inv_quant_dc = quantizer_.inv_quant_dc();
  Image3F dc_blur_x = ComputeDCBlurX(coeffs(), kDCBlurSigma, inv_quant_dc,
                                     YToBDC(), DequantMatrix());
  float w_up[kBlockSize] = { 0.0f };
  float w_cur[kBlockSize] = { 0.0f };
  float w_down[kBlockSize] = { 0.0f };
  ComputeBlockBlurWeights(kDCBlurSigma, w_up, w_cur, w_down);
  for (int by = 0; by < block_ysize_; ++by) {
    int by_u = block_ysize_ == 1 ? 0 : by == 0 ? 1 : by - 1;
    int by_d = block_ysize_ == 1 ? 0 : by + 1 < block_ysize_ ? by + 1 : by - 1;
    float* const PIK_RESTRICT row = opsin_overlay_->Row(by);
    for (int bx = 0, i = 0; bx < block_xsize_; ++bx) {
      const int offsetx = bx * kBlockEdge;
      for (int c = 0; c < 3; ++c, i += kBlockSize) {
        float avg = ComputeBlurredBlock(dc_blur_x, c, offsetx, by_u, by, by_d,
                                        w_up, w_cur, w_down, &row[i]);
        for (int k = 0; k < kBlockSize; ++k) {
          row[i + k] -= avg;
        }
      }
    }
  }
}

void CompressedImage::Quantize() {
  QuantizeDC();
  ComputeOpsinOverlay();
  for (int block_y = 0; block_y < block_ysize_; ++block_y) {
    for (int block_x = 0; block_x < block_xsize_; ++block_x) {
      QuantizeBlock(block_x, block_y);
    }
  }
}

std::string CompressedImage::Encode() const {
  PIK_CHECK(ytob_dc_ >= 0);
  PIK_CHECK(ytob_dc_ < 256);
  std::string ytob_code =
      std::string(1, ytob_dc_) + EncodePlane(ytob_ac_, 0, 255);
  std::string quant_code = quantizer_.Encode();
  PikImageSizeInfo* dc_info = pik_info_ ? &pik_info_->dc_image : nullptr;
  PikImageSizeInfo* ac_info = pik_info_ ? &pik_info_->ac_image : nullptr;
  std::string dc_code = EncodeImage(PredictDC(dct_coeffs_), 1, dc_info);
  std::string ac_code = EncodeAC(dct_coeffs_, ac_info);
  if (pik_info_) {
    pik_info_->ytob_image_size = ytob_code.size();
    pik_info_->quant_image_size = quant_code.size();
  }
  return PadTo4Bytes(ytob_code + quant_code + dc_code + ac_code);
}

std::string CompressedImage::EncodeFast() const {
  PIK_CHECK(ytob_dc_ >= 0);
  PIK_CHECK(ytob_dc_ < 256);
  std::string ytob_code =
      std::string(1, ytob_dc_) + EncodePlane(ytob_ac_, 0, 255);
  std::string quant_code = quantizer_.Encode();
  PikImageSizeInfo* dc_info = pik_info_ ? &pik_info_->dc_image : nullptr;
  PikImageSizeInfo* ac_info = pik_info_ ? &pik_info_->ac_image : nullptr;
  std::string dc_code = EncodeImage(PredictDC(dct_coeffs_), 1, dc_info);
  std::string ac_code = EncodeACFast(dct_coeffs_, ac_info);
  if (pik_info_) {
    pik_info_->ytob_image_size = ytob_code.size();
    pik_info_->quant_image_size = quant_code.size();
  }
  return PadTo4Bytes(ytob_code + quant_code + dc_code + ac_code);
}

bool CompressedImage::Decode(const uint8_t* compressed,
                             const size_t compressed_size) {
  if (compressed_size == 0) {
    return PIK_FAILURE("Empty compressed data.");
  }
  if (compressed_size % 4 != 0) {
    return PIK_FAILURE("Invalid padding.");
  }
  BitReader br(compressed, compressed_size);
  ytob_dc_ = br.ReadBits(8);
  if (!DecodePlane(&br, 0, 255, &ytob_ac_)) {
    return PIK_FAILURE("DecodePlane failed.");
  }
  if (!quantizer_.Decode(&br)) {
    return PIK_FAILURE("quantizer Decode failed.");
  }
  if (!DecodeImage(&br, kBlockSize, &dct_coeffs_)) {
    return PIK_FAILURE("DecodeImage failed.");
  }
  if (!DecodeAC(&br, &dct_coeffs_)) {
    return PIK_FAILURE("DecodeAC failed.");
  }
  if (br.Position() != compressed_size) {
    return PIK_FAILURE("Pik compressed data size mismatch.");
  }
  UnpredictDC(&dct_coeffs_);
  return true;
}

void CompressedImage::DequantizeBlock(const int block_x, const int block_y,
                                      float* const PIK_RESTRICT block) const {

  using namespace PIK_TARGET_NAME;
  const int tile_y = block_y / kTileToBlockRatio;
  auto row = dct_coeffs_.Row(block_y);
  const int tile_x = block_x / kTileToBlockRatio;
  const int offset = block_x * kBlockSize;
  const float inv_quant_dc = quantizer_.inv_quant_dc();
  const float inv_quant_ac = quantizer_.inv_quant_ac(block_x, block_y);
  const float* PIK_RESTRICT kDequantMatrix = DequantMatrix();
  for (int c = 0; c < 3; ++c) {
    const int16_t* const PIK_RESTRICT iblock = &row[c][offset];
    const float* const PIK_RESTRICT muls = &kDequantMatrix[c * kBlockSize];
    float* const PIK_RESTRICT cur_block = &block[c * kBlockSize];
    for (int k = 0; k < kBlockSize; ++k) {
      cur_block[k] = iblock[k] * (muls[k] * inv_quant_ac);
    }
    cur_block[0] = iblock[0] * (muls[0] * inv_quant_dc);
  }
  using V = V8x32F;
  const float kYToBAC = YToBAC(tile_x, tile_y);
  for (int k = 0; k < kBlockSize; k += V::N) {
    const V y = Load<V>(block + k + kBlockSize);
    const V b = Load<V>(block + k + kBlockSize2) + V(kYToBAC) * y;
    Store(b, block + k + kBlockSize2);
  }
  block[kBlockSize2] += (YToBDC() - kYToBAC) * block[kBlockSize];
  block[kBlockSize + 3] += kACPred31 * block[kBlockSize + 1];
  block[kBlockSize + 24] += kACPred31 * block[kBlockSize + 8];
}

namespace {

void ColorTransformOpsinToSrgb(const float* const PIK_RESTRICT block,
                               int block_x, int block_y,
                               Image3B* const PIK_RESTRICT srgb) {
  using namespace PIK_TARGET_NAME;
  const uint8_t* lut_plus = LinearToSrgb8TablePlusQuarter();
  const uint8_t* lut_minus = LinearToSrgb8TableMinusQuarter();
  // TODO(user) Combine these two for loops and get rid of rgb[].
  alignas(32) int rgb[kBlockSize3];
  using V = V8x32F;
  for (int k = 0; k < kBlockSize; k += V::N) {
    const V x = Load<V>(block + k) + V(kXybCenter[0]);
    const V y = Load<V>(block + k + kBlockSize) + V(kXybCenter[1]);
    const V b = Load<V>(block + k + kBlockSize2) + V(kXybCenter[2]);
    const V lut_scale(16.0f);
    V out_r, out_g, out_b;
    XybToRgb(x, y, b, &out_r, &out_g, &out_b);
    Store(RoundToInt(out_r * lut_scale), rgb + k);
    Store(RoundToInt(out_g * lut_scale), rgb + k + kBlockSize);
    Store(RoundToInt(out_b * lut_scale), rgb + k + kBlockSize2);
  }
  const int yoff = kBlockEdge * block_y;
  const int xoff = kBlockEdge * block_x;
  for (int iy = 0; iy < kBlockEdge; ++iy) {
    auto row = srgb->Row(iy + yoff);
    for (int ix = 0; ix < kBlockEdge; ++ix) {
      const int px = ix + xoff;
      const int k = kBlockEdge * iy + ix;
      const uint8_t* lut = (ix + iy) % 2 ? lut_plus : lut_minus;
      row[0][px] = lut[rgb[k + 0]];
      row[1][px] = lut[rgb[k + kBlockSize]];
      row[2][px] = lut[rgb[k + kBlockSize2]];
    }
  }
}

void ColorTransformOpsinToSrgb(const float* const PIK_RESTRICT block,
                               int block_x, int block_y,
                               Image3U* const PIK_RESTRICT srgb) {
  using namespace PIK_TARGET_NAME;
  // TODO(user) Combine these two for loops and get rid of rgb[].
  alignas(32) int rgb[kBlockSize3];
  using V = V8x32F;
  for (int k = 0; k < kBlockSize; k += V::N) {
    const V x = Load<V>(block + k) + V(kXybCenter[0]);
    const V y = Load<V>(block + k + kBlockSize) + V(kXybCenter[1]);
    const V b = Load<V>(block + k + kBlockSize2) + V(kXybCenter[2]);
    const V scale_to_16bit(257.0f);
    V out_r, out_g, out_b;
    XybToRgb(x, y, b, &out_r, &out_g, &out_b);

    out_r = LinearToSrgbPoly(out_r) * scale_to_16bit;
    out_g = LinearToSrgbPoly(out_g) * scale_to_16bit;
    out_b = LinearToSrgbPoly(out_b) * scale_to_16bit;

    Store(RoundToInt(out_r), rgb + k);
    Store(RoundToInt(out_g), rgb + k + kBlockSize);
    Store(RoundToInt(out_b), rgb + k + kBlockSize2);
  }

  const int yoff = kBlockEdge * block_y;
  const int xoff = kBlockEdge * block_x;
  for (int iy = 0; iy < kBlockEdge; ++iy) {
    auto row = srgb->Row(iy + yoff);
    for (int ix = 0; ix < kBlockEdge; ++ix) {
      const int px = ix + xoff;
      const int k = kBlockEdge * iy + ix;
      row[0][px] = rgb[k + 0];
      row[1][px] = rgb[k + kBlockSize];
      row[2][px] = rgb[k + kBlockSize2];
    }
  }
}

void ColorTransformOpsinToSrgb(const float* const PIK_RESTRICT block,
                               int block_x, int block_y,
                               Image3F* const PIK_RESTRICT srgb) {
  using namespace PIK_TARGET_NAME;
  // TODO(user) Combine these two for loops and get rid of rgb[].
  alignas(32) float rgb[kBlockSize3];
  using V = V8x32F;
  for (int k = 0; k < kBlockSize; k += V::N) {
    const V x = Load<V>(block + k) + V(kXybCenter[0]);
    const V y = Load<V>(block + k + kBlockSize) + V(kXybCenter[1]);
    const V b = Load<V>(block + k + kBlockSize2) + V(kXybCenter[2]);
    V out_r, out_g, out_b;
    XybToRgb(x, y, b, &out_r, &out_g, &out_b);
    Store(out_r, rgb + k);
    Store(out_g, rgb + k + kBlockSize);
    Store(out_b, rgb + k + kBlockSize2);
  }
  const int yoff = kBlockEdge * block_y;
  const int xoff = kBlockEdge * block_x;
  for (int iy = 0; iy < kBlockEdge; ++iy) {
    auto row = srgb->Row(iy + yoff);
    for (int ix = 0; ix < kBlockEdge; ++ix) {
      const int px = ix + xoff;
      const int k = kBlockEdge * iy + ix;
      row[0][px] = rgb[k + 0];
      row[1][px] = rgb[k + kBlockSize];
      row[2][px] = rgb[k + kBlockSize2];
    }
  }
}

}  // namespace

template <class Image3T>
Image3T GetPixels(const CompressedImage& img) {
  const int block_xsize = img.block_xsize();
  const int block_ysize = img.block_ysize();
  Image3T out(block_xsize * kBlockEdge, block_ysize * kBlockEdge);
  const float inv_quant_dc = img.quantizer().inv_quant_dc();
  Image3F dc_blur_x = ComputeDCBlurX(img.coeffs(), kDCBlurSigma, inv_quant_dc,
                                     img.YToBDC(), DequantMatrix());
  float w_up[kBlockSize] = { 0.0f };
  float w_cur[kBlockSize] = { 0.0f };
  float w_down[kBlockSize] = { 0.0f };
  ComputeBlockBlurWeights(kDCBlurSigma, w_up, w_cur, w_down);
  alignas(32) float block_out[kBlockSize3];
  for (int by = 0; by < block_ysize; ++by) {
    int by_u = block_ysize == 1 ? 0 : by == 0 ? 1 : by - 1;
    int by_d = block_ysize == 1 ? 0 : by + 1 < block_ysize ? by + 1 : by - 1;
    for (int bx = 0; bx < block_xsize; ++bx) {
      img.DequantizeBlock(bx, by, block_out);
      const int offsetx = bx * kBlockEdge;
      for (int c = 0; c < 3; ++c) {
        ComputeTransposedScaledBlockIDCTFloat(&block_out[kBlockSize * c]);
        alignas(32) float dc_blur[kBlockSize];
        float avg = ComputeBlurredBlock(dc_blur_x, c, offsetx, by_u, by, by_d,
                                        w_up, w_cur, w_down, dc_blur);
        for (int k = 0; k < kBlockSize; ++k) {
          block_out[kBlockSize * c + k] += dc_blur[k] - avg;
        }
      }
      ColorTransformOpsinToSrgb(block_out, bx, by, &out);
    }
  }
  out.ShrinkTo(img.xsize(), img.ysize());
  return out;
}

Image3B CompressedImage::ToSRGB() const {
  return GetPixels<Image3B>(*this);
}

Image3U CompressedImage::ToSRGB16() const {
  return GetPixels<Image3U>(*this);
}

Image3F CompressedImage::ToLinear() const {
  return GetPixels<Image3F>(*this);
}

}  // namespace pik
