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
#include "compiler_specific.h"
#include "dc_predictor.h"
#include "dct_util.h"
#include "gauss_blur.h"
#include "opsin_codec.h"
#include "opsin_params.h"
#include "status.h"

namespace pik {

namespace {

static const int kBlockEdge = 8;
static const int kBlockSize = kBlockEdge * kBlockEdge;

int DivCeil(int a, int b) {
  return (a + b - 1) / b;
}

}  // namespace

Image3F AlignImage(const Image3F& in, const size_t N) {
  const size_t block_xsize = DivCeil(in.xsize(), N);
  const size_t block_ysize = DivCeil(in.ysize(), N);
  const size_t xsize = N * block_xsize;
  const size_t ysize = N * block_ysize;
  Image3F out(xsize, ysize);
  int y = 0;
  for (; y < in.ysize(); ++y) {
    for (int c = 0; c < 3; ++c) {
      const float* const PIK_RESTRICT row_in = &in.Row(y)[c][0];
      float* const PIK_RESTRICT row_out = &out.Row(y)[c][0];
      memcpy(row_out, row_in, in.xsize() * sizeof(row_in[0]));
      const int lastcol = in.xsize() - 1;
      const float lastval = row_out[lastcol];
      for (int x = in.xsize(); x < xsize; ++x) {
        row_out[x] = lastval;
      }
    }
  }
  const int lastrow = in.ysize() - 1;
  for (; y < ysize; ++y) {
    for (int c = 0; c < 3; ++c) {
      const float* const PIK_RESTRICT row_in = out.ConstPlaneRow(c, lastrow);
      float* const PIK_RESTRICT row_out = out.PlaneRow(c, y);
      memcpy(row_out, row_in, xsize * sizeof(row_out[0]));
    }
  }
  return out;
}

void CenterOpsinValues(Image3F* img) {
  for (int y = 0; y < img->ysize(); ++y) {
    auto row = img->Row(y);
    for (int x = 0; x < img->xsize(); ++x) {
      for (int c = 0; c < 3; ++c) {
        row[c][x] -= kXybCenter[c];
      }
    }
  }
}

void YToBTransform(const float factor, Image3F* opsin) {
  for (int y = 0; y < opsin->ysize(); ++y) {
    auto row = opsin->Row(y);
    for (int x = 0; x < opsin->xsize(); ++x) {
      row[2][x] += factor * row[1][x];
    }
  }
}

void Adjust2x2ACFromDC(const Image3F& dc, const int direction,
                       Image3F* coeffs) {
  constexpr float kWeight0 = 0.027630534023046f;
  constexpr float kWeight1 = 0.133676439523697f;
  constexpr float kWeight2 = 0.035697385668755f;
  constexpr float kKernel0[3] = { 1.0f, 0.0f, -1.0f };
  constexpr float kKernel1[3] = { kWeight0, kWeight1, kWeight0 };
  constexpr float kKernel2[3] = { kWeight2, 0.0f, -kWeight2 };
  const std::vector<float> kernel0(kKernel0, kKernel0 + 3);
  const std::vector<float> kernel1(kKernel1, kKernel1 + 3);
  const std::vector<float> kernel2(kKernel2, kKernel2 + 3);
  Image3F tmp0 = ConvolveXSampleAndTranspose(dc, kernel0, 1);
  Image3F tmp1 = ConvolveXSampleAndTranspose(dc, kernel1, 1);
  Image3F ac01 = ConvolveXSampleAndTranspose(tmp1, kernel0, 1);
  Image3F ac10 = ConvolveXSampleAndTranspose(tmp0, kernel1, 1);
  Image3F ac11 = ConvolveXSampleAndTranspose(tmp0, kernel2, 1);
  for (int by = 0; by < dc.ysize(); ++by) {
    auto row01 = ac01.ConstRow(by);
    auto row10 = ac10.ConstRow(by);
    auto row11 = ac11.ConstRow(by);
    auto row_out = coeffs->Row(by);
    for (int bx = 0, x = 0; bx < dc.xsize(); ++bx, x += kBlockSize) {
      for (int c = 0; c < 3; ++c) {
        row_out[c][x + 1] += direction * row01[c][bx];
        row_out[c][x + 8] += direction * row10[c][bx];
        row_out[c][x + 9] += direction * row11[c][bx];
      }
    }
  }
}

QuantizedCoeffs ComputeCoefficients(const Image3F& opsin,
                                    const Quantizer& quantizer) {
  Image3F coeffs = TransposedScaledDCT(opsin);
  QuantizedCoeffs qcoeffs = QuantizeCoeffs(coeffs, quantizer);
  Image3F dcoeffs = DequantizeCoeffs(qcoeffs, quantizer);
  Adjust2x2ACFromDC(DCImage(dcoeffs), -1, &coeffs);
  qcoeffs = QuantizeCoeffs(coeffs, quantizer);
  dcoeffs = DequantizeCoeffs(qcoeffs, quantizer);
  Adjust2x2ACFromDC(DCImage(dcoeffs), 1, &dcoeffs);
  Image3F pred = GetPixelSpaceImageFrom2x2Corners(dcoeffs);
  pred = UpSample4x4BlurDCT(pred, 1.5f);
  SubtractFrom(pred, &coeffs);
  return QuantizeCoeffs(coeffs, quantizer);
}

std::string EncodeToBitstream(const QuantizedCoeffs& qcoeffs,
                              const Quantizer& quantizer,
                              int ytob,
                              bool fast_mode,
                              PikInfo* info) {
  PIK_CHECK(ytob >= 0);
  PIK_CHECK(ytob < 256);
  std::string ytob_code = std::string(1, ytob);
  PikImageSizeInfo* quant_info = info ? &info->layers[0] : nullptr;
  PikImageSizeInfo* dc_info = info ? &info->layers[1] : nullptr;
  PikImageSizeInfo* ac_info = info ? &info->layers[2] : nullptr;
  std::string quant_code = quantizer.Encode(quant_info);
  std::string dc_code = EncodeImage(PredictDC(qcoeffs), 1, dc_info);
  std::string ac_code = fast_mode ?
      EncodeACFast(qcoeffs, ac_info) :
      EncodeAC(qcoeffs, ac_info);
  return PadTo4Bytes(ytob_code + quant_code + dc_code + ac_code);
}

bool DecodeFromBitstream(const uint8_t* data, const size_t data_size,
                         const size_t xsize, const size_t ysize,
                         int* ytob,
                         Quantizer* quantizer,
                         QuantizedCoeffs* qcoeffs,
                         size_t* compressed_size) {
  if (data_size == 0) {
    return PIK_FAILURE("Empty compressed data.");
  }
  *qcoeffs = Image3W(DivCeil(xsize, kBlockEdge) * kBlockSize,
                     DivCeil(ysize, kBlockEdge));
  BitReader br(data, data_size & ~3);
  *ytob = br.ReadBits(8);
  if (!quantizer->Decode(&br)) {
    return PIK_FAILURE("quantizer Decode failed.");
  }
  if (!DecodeImage(&br, kBlockSize, qcoeffs)) {
    return PIK_FAILURE("DecodeImage failed.");
  }
  if (!DecodeAC(&br, qcoeffs)) {
    return PIK_FAILURE("DecodeAC failed.");
  }
  *compressed_size = br.Position();
  UnpredictDC(qcoeffs);
  return true;
}

Image3F ReconOpsinImage(const QuantizedCoeffs& qcoeffs,
                        const Quantizer& quantizer) {
  Image3F dcoeffs = DequantizeCoeffs(qcoeffs, quantizer);
  Adjust2x2ACFromDC(DCImage(dcoeffs), 1, &dcoeffs);
  Image3F pred = GetPixelSpaceImageFrom2x2Corners(dcoeffs);
  pred = UpSample4x4BlurDCT(pred, 1.5f);
  AddTo(pred, &dcoeffs);
  return TransposedScaledIDCT(dcoeffs);
}

}  // namespace pik
