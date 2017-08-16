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

static const int kBlockEdge = 8;
static const int kBlockSize = kBlockEdge * kBlockEdge;
static const int kBlockSize2 = 2 * kBlockSize;
static const int kBlockSize3 = 3 * kBlockSize;
static const int kLastCol = kBlockEdge - 1;
static const int kLastRow = kLastCol * kBlockEdge;
static const int kCoeffsPerBlock = kBlockSize;
static const int kQuantBlockRes = 1;

int DivCeil(int a, int b) {
  return (a + b - 1) / b;
}

PIK_INLINE void PredictBlock(const float* const PIK_RESTRICT prev_block,
                             const float* const PIK_RESTRICT top_block,
                             float* const PIK_RESTRICT prediction) {
  float sum = 0.0;
  for (int i = 0; i < kBlockEdge; ++i) {
    sum += top_block[kLastRow + i];
    sum += prev_block[kBlockEdge * i + kLastCol];
  }
  sum /= (2 * kBlockEdge);
  for (int k = 0; k < kBlockSize; ++k) {
    prediction[k] = sum;
  }
}

PIK_INLINE void AddBlock(const float* const PIK_RESTRICT a,
                         float* const PIK_RESTRICT b) {
  for (int i = 0; i < kBlockSize; ++i) b[i] += a[i];
}

PIK_INLINE void SubtractBlock(const float* const PIK_RESTRICT a,
                              float* const PIK_RESTRICT b) {
  for (int i = 0; i < kBlockSize; ++i) b[i] -= a[i];
}

Image3F Subtract(const Image3F& in, const float centers[3]) {
  Image3F out(in.xsize(), in.ysize());
  for (int y = 0; y < in.ysize(); ++y) {
    for (int c = 0; c < 3; ++c) {
      for (int x = 0; x < in.xsize(); ++x) {
        out.Row(y)[c][x] = in.Row(y)[c][x] - centers[c];
      }
    }
  }
  return out;
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

void DumpOpsinDiff(const PikInfo* info, const Image3F& in,
                   const std::string& label) {
  if (!info || info->debug_prefix.empty()) return;
  WriteImage(ImageFormatPNG(),
             Image3B(NormalizeAndClip(in.plane(0), 2.0f * kXybRange[0]),
                     NormalizeAndClip(in.plane(1), 2.0f * kXybRange[1]),
                     NormalizeAndClip(in.plane(2), 2.0f * kXybRange[2])),
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

}  // namespace

CompressedImage::CompressedImage(int xsize, int ysize, PikInfo* info)
    : xsize_(xsize), ysize_(ysize),
      block_xsize_(DivCeil(xsize, kBlockEdge)),
      block_ysize_(DivCeil(ysize, kBlockEdge)),
      quant_xsize_(DivCeil(block_xsize_, kQuantBlockRes)),
      quant_ysize_(DivCeil(block_ysize_, kQuantBlockRes)),
      num_blocks_(block_xsize_ * block_ysize_),
      quantizer_(quant_xsize_, quant_ysize_, kCoeffsPerBlock, DequantMatrix()),
      dct_coeffs_(block_xsize_ * kBlockSize, block_ysize_),
      ytob_dc_(120),
      ytob_ac_(DivCeil(xsize, kYToBRes), DivCeil(ysize, kYToBRes), 120),
      pik_info_(info) {
}

int CompressedImage::quant_tile_size() const {
  return kQuantBlockRes * kBlockEdge;
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
  const int quant_x = block_x / kQuantBlockRes;
  const int quant_y = block_y / kQuantBlockRes;
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
  const float inv_quant_ac = 64.0f * quantizer_.inv_quant_ac(quant_x, quant_y);
  const float* PIK_RESTRICT kDequantMatrix = DequantMatrix();
  quantizer_.QuantizeBlock(quant_x, quant_y, 1, 0, kBlockSize,
                           &block[kBlockSize], iblocky);
  for (int k = 0; k < kBlockSize; ++k) {
    block[kBlockSize + k] = iblocky[k] * kDequantMatrix[kBlockSize + k] *
        inv_quant_ac;
  }
  block[kBlockSize] = iblocky[0] * kDequantMatrix[kBlockSize] * inv_quant_dc;
  {
    const int tile_x = block_x * kBlockEdge / kYToBRes;
    const int tile_y = block_y * kBlockEdge / kYToBRes;
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
        quant_x, quant_y, c, 0, kBlockSize,
        &block[c * kBlockSize], &row_out[c][offset]);
  }
}

void CompressedImage::Quantize() {
  for (int block_y = 0; block_y < block_ysize_; ++block_y) {
    for (int block_x = 0; block_x < block_xsize_; ++block_x) {
      QuantizeBlock(block_x, block_y);
    }
  }
}

Image3B CompressedImage::ToSRGB() const {
  Image3B out(block_xsize_ * kBlockEdge, block_ysize_ * kBlockEdge);
  alignas(32) float block_out[kBlockSize3];
  for (int block_y = 0; block_y < block_ysize_; ++block_y) {
    for (int block_x = 0; block_x < block_xsize_; ++block_x) {
      UpdateBlock(block_x, block_y, block_out);
      UpdateSRGB(block_out, block_x, block_y, &out);
    }
  }
  out.ShrinkTo(xsize_, ysize_);
  return out;
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

void CompressedImage::UpdateBlock(const int block_x, const int block_y,
                                  float* const PIK_RESTRICT block) const {
  using namespace PIK_TARGET_NAME;
  const int quant_y = block_y / kQuantBlockRes;
  const int tile_y = block_y * kBlockEdge / kYToBRes;
  auto row = dct_coeffs_.Row(block_y);
  const int quant_x = block_x / kQuantBlockRes;
  const int tile_x = block_x * kBlockEdge / kYToBRes;
  const int offset = block_x * kBlockSize;
  const float inv_quant_dc = quantizer_.inv_quant_dc();
  const float inv_quant_ac = quantizer_.inv_quant_ac(quant_x, quant_y);
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
  for (int c = 0; c < 3; ++c) {
    ComputeTransposedScaledBlockIDCTFloat(&block[kBlockSize * c]);
  }
}

void CompressedImage::UpdateSRGB(const float* const PIK_RESTRICT block,
                                 int block_x, int block_y,
                                 Image3B* const PIK_RESTRICT srgb) const {
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

Image3F CompressedImage::ToLinear() const {
  Image3F out(block_xsize_ * kBlockEdge, block_ysize_ * kBlockEdge);
  for (int block_y = 0; block_y < block_ysize_; ++block_y) {
    for (int block_x = 0; block_x < block_xsize_; ++block_x) {
      alignas(32) float block[kBlockSize3];
      UpdateBlock(block_x, block_y, block);
      const int yoff = kBlockEdge * block_y;
      const int xoff = kBlockEdge * block_x;
      for (int iy = 0; iy < kBlockEdge; ++iy) {
        auto row = out.Row(iy + yoff);
        for (int ix = 0; ix < kBlockEdge; ++ix) {
          const int px = ix + xoff;
          const int k = kBlockEdge * iy + ix;
          float x = block[k + 0] + kXybCenter[0];
          float y = block[k + kBlockSize] + kXybCenter[1];
          float z = block[k + kBlockSize2] + kXybCenter[2];
          XybToRgb(x, y, z, &row[0][px], &row[1][px], &row[2][px]);
        }
      }
    }
  }
  out.ShrinkTo(xsize_, ysize_);
  return out;
}

}  // namespace pik
