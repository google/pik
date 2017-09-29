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
#include "vector128.h"
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

const double *GetQuantizeMul() {
  static double kQuantizeMul[3] = {
    1.9189204419575077,
    0.87086518648437961,
    0.14416093417099549,
  };
  return &kQuantizeMul[0];
};

// kQuantWeights[3 * k_zz + c] is the relative weight of the k_zz coefficient
// (in the zig-zag order) in component c. Higher weights correspond to finer
// quantization intervals and more bits spent in encoding.
const double *GetQuantWeights() {
  static double kQuantWeights[kBlockSize3] = {
    3.1116384958312873,
    1.9493486886858318,
    3.7356702523108076,
    1.6376603398927516,
    0.90329541944008185,
    1.1924294798808717,
    1.6751086981040682,
    0.91086559902706354,
    1.3230594582478616,
    1.2083767049877585,
    1.3817411526714352,
    1.2533650395458487,
    1.3869170538521951,
    1.4695312986673121,
    0.99293595066355556,
    1.0717982557177361,
    0.77157226728475437,
    0.90014274926175286,
    0.98789123070249052,
    1.0068666175402106,
    0.96006472092122286,
    0.97546450225501924,
    1.0028287751393217,
    0.96327362036476849,
    0.94519992131731234,
    1.0397303129870175,
    0.8768538162652274,
    0.72259226198101179,
    0.77345511996219551,
    0.69686799736446703,
    0.75026596827865244,
    0.74593198246655135,
    0.72383355746284095,
    0.85527787563242419,
    0.77235080374314469,
    0.89085410178838442,
    0.7633330445134987,
    0.80404240082171852,
    0.75631861848435222,
    0.7969244311243745,
    0.80335731742726979,
    0.67359559842028449,
    0.72969875072028578,
    0.7860618793942612,
    0.54861701802111473,
    0.69747238728941174,
    0.79474549435934105,
    0.64569824585883318,
    0.74856364387131902,
    0.79953121556785256,
    0.5817162482472058,
    0.7400626220783687,
    0.75611094606974305,
    0.68609545905365177,
    0.87633313121025569,
    0.82522696002735152,
    0.68952511265573835,
    0.81698325161338492,
    0.77119264991997505,
    0.65664011847964321,
    0.7113943789197511,
    0.71654686115855182,
    0.57133610422951886,
    0.53784460976696491,
    0.67886793875123497,
    0.41012386366984932,
    0.88436458133484919,
    0.78697858505117113,
    0.66757231829162755,
    0.79864079537213051,
    0.83330073556484574,
    0.56230115407003167,
    0.73371714121200982,
    0.784821179161595,
    0.63683336803749147,
    0.5128389662260503,
    0.76510370223002389,
    0.6071016402671725,
    0.73760066230479648,
    0.58811067041995513,
    0.46565306264232459,
    0.49953580940678222,
    0.57710614035253449,
    0.40765842682471454,
    0.4303752490677128,
    0.69532449310310929,
    0.40590820226024099,
    0.69075917888709448,
    0.59844872816640948,
    0.40071217887401517,
    0.74273651388706152,
    0.76304704918313782,
    0.62634038749139065,
    0.8415198507222702,
    0.66967267156576216,
    0.5808891642416677,
    0.71015160986960257,
    0.65137160765366753,
    0.5166119362044449,
    0.64106938434456939,
    0.62650693826574488,
    0.43316645272082255,
    0.53431647340052557,
    0.58646710734447582,
    0.40928788614569234,
    0.41505184593035693,
    0.55438012479983545,
    0.41220557667312735,
    0.5744086069555473,
    0.5902159769538029,
    0.45008872519682036,
    0.6922703294525423,
    0.63862381356226594,
    0.64548795603030296,
    0.82406119574330217,
    0.59972689373496091,
    0.65635373313931233,
    0.68947663433371897,
    0.64923721443874749,
    0.62201317742452744,
    0.83870239301163885,
    0.69933779474340596,
    0.59374933718163225,
    0.60373950696793921,
    0.66063427518048312,
    0.52682575197233517,
    0.73699079546703949,
    0.58801678449178196,
    0.57315185779133337,
    0.78233349785918793,
    0.70528156369371808,
    0.62405995474355513,
    0.74537125413266614,
    0.60380965247810592,
    0.4970020248657363,
    0.86917060671449842,
    0.65352106325704196,
    0.42007406411441695,
    0.52992210097476178,
    0.60917359099661206,
    0.42429980778598214,
    0.79896157287058611,
    0.67526272980173785,
    0.73685662140979558,
    0.59364298008114702,
    0.59260225011442091,
    0.6862115228199912,
    0.62567701952441157,
    0.71242092064210538,
    0.54522098190139823,
    0.82235627711268378,
    0.64777251030703653,
    0.5586940003934312,
    0.75611432702314785,
    0.60493823358774101,
    0.54369112638834083,
    0.80427934057093309,
    0.6054619372971719,
    0.60547619250966378,
    0.72807482897001252,
    0.65904209615319964,
    0.55011475454992043,
    0.40730247508476503,
    0.69345827743345601,
    0.51765066822174743,
    0.9455108649936359,
    0.62089871937583396,
    0.55362717915029036,
    0.40914654233031822,
    0.63174305912892581,
    0.54984140651169744,
    0.60024883096869397,
    0.64718696705106504,
    0.6392737396330197,
    0.47977549663100738,
    0.65033234442749888,
    0.71328486015966841,
    0.63857315688426208,
    0.62699991616319317,
    0.57225967233967767,
    0.82055874019371045,
    0.61477228068808698,
    0.54185797617041831,
    0.67913454625845626,
    0.5327324114828782,
    0.66993969215541949,
    0.49206143412708364,
    0.53004732658023113,
    0.68218232914187027,
    0.75028232828887931,
    0.53230208750713925,
    0.5272846988211819,
  };
  return &kQuantWeights[0];
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
      double weight = GetQuantWeights()[idx];
      if (weight < 0.4) { weight = 0.4; }
      double mul = GetQuantizeMul()[c];
      table[c * kCoeffsPerBlock + k] = idct_scale / (weight * mul);
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
  const float* const PIK_RESTRICT row0 = blur_x.PlaneRow(c, y_up) + offsetx;
  const float* const PIK_RESTRICT row1 = blur_x.PlaneRow(c, y_cur) + offsetx;
  const float* const PIK_RESTRICT row2 = blur_x.PlaneRow(c, y_down) + offsetx;
  // "Loop" over ix = [0, kBlockEdge), one per lane.
  using namespace PIK_TARGET_NAME;
  using V = V8x32F;
  V sum(0.0f);
  const V val0 = Load<V>(row0);
  const V val1 = Load<V>(row1);
  const V val2 = Load<V>(row2);
  for (int iy = 0; iy < kBlockEdge; ++iy) {
    const V val =
        val0 * V(w_up[iy]) + val1 * V(w_cur[iy]) + val2 * V(w_down[iy]);
    Store(val, out + iy * kBlockEdge);
    sum += val;
  }

  // Horizontal sum
  alignas(32) float sum_lanes[V::N];
  Store(sum, sum_lanes);
  float avg = sum_lanes[0];
  for (size_t i = 1; i < V::N; ++i) {
    avg += sum_lanes[i];
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
static const float kACPredScale = 0.30348289542505313;
static const float kACPred31 = kACPredScale * 0.051028376631910073;

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
  PikImageSizeInfo* ytob_info = pik_info_ ? &pik_info_->ytob_image : nullptr;
  PikImageSizeInfo* quant_info = pik_info_ ? &pik_info_->quant_image : nullptr;
  PikImageSizeInfo* dc_info = pik_info_ ? &pik_info_->dc_image : nullptr;
  PikImageSizeInfo* ac_info = pik_info_ ? &pik_info_->ac_image : nullptr;
  std::string ytob_code =
      std::string(1, ytob_dc_) + EncodePlane(ytob_ac_, 0, 255, ytob_info);
  std::string quant_code = quantizer_.Encode(quant_info);
  std::string dc_code = EncodeImage(PredictDC(dct_coeffs_), 1, dc_info);
  std::string ac_code = EncodeAC(dct_coeffs_, ac_info);
  return PadTo4Bytes(ytob_code + quant_code + dc_code + ac_code);
}

std::string CompressedImage::EncodeFast() const {
  PIK_CHECK(ytob_dc_ >= 0);
  PIK_CHECK(ytob_dc_ < 256);
  PikImageSizeInfo* ytob_info = pik_info_ ? &pik_info_->ytob_image : nullptr;
  PikImageSizeInfo* quant_info = pik_info_ ? &pik_info_->quant_image : nullptr;
  PikImageSizeInfo* dc_info = pik_info_ ? &pik_info_->dc_image : nullptr;
  PikImageSizeInfo* ac_info = pik_info_ ? &pik_info_->ac_image : nullptr;
  std::string ytob_code =
      std::string(1, ytob_dc_) + EncodePlane(ytob_ac_, 0, 255, ytob_info);
  std::string quant_code = quantizer_.Encode(quant_info);
  std::string dc_code = EncodeImage(PredictDC(dct_coeffs_), 1, dc_info);
  std::string ac_code = EncodeACFast(dct_coeffs_, ac_info);
  return PadTo4Bytes(ytob_code + quant_code + dc_code + ac_code);
}

bool CompressedImage::Decode(const uint8_t* data, const size_t data_size,
                             size_t* compressed_size) {
  if (data_size == 0) {
    return PIK_FAILURE("Empty compressed data.");
  }
  BitReader br(data, data_size & ~3);
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
  *compressed_size = br.Position();
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
  const V vYToBAC(kYToBAC);
  for (int k = 0; k < kBlockSize; k += V::N) {
    const V y = Load<V>(block + k + kBlockSize);
    const V b = MulAdd(vYToBAC, y, Load<V>(block + k + kBlockSize2));
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
  using V = V8x32F;
  const V scale_to_16bit(257.0f);

  int k = 0;  // index within 8x8 block (we access 3 consecutive blocks)
  const int yoff = kBlockEdge * block_y;
  const int xoff = kBlockEdge * block_x;
  for (int iy = 0; iy < kBlockEdge; ++iy) {
    uint16_t* PIK_RESTRICT row0 = srgb->PlaneRow(0, iy + yoff);
    uint16_t* PIK_RESTRICT row1 = srgb->PlaneRow(1, iy + yoff);
    uint16_t* PIK_RESTRICT row2 = srgb->PlaneRow(2, iy + yoff);
    for (int ix = 0; ix < kBlockEdge; ix += V::N) {
      const V x = Load<V>(block + k) + V(kXybCenter[0]);
      const V y = Load<V>(block + k + kBlockSize) + V(kXybCenter[1]);
      const V b = Load<V>(block + k + kBlockSize2) + V(kXybCenter[2]);
      k += V::N;
      V out_r, out_g, out_b;
      XybToRgb(x, y, b, &out_r, &out_g, &out_b);

      out_r = LinearToSrgbPoly(out_r) * scale_to_16bit;
      out_g = LinearToSrgbPoly(out_g) * scale_to_16bit;
      out_b = LinearToSrgbPoly(out_b) * scale_to_16bit;

      V8x32I int_r = RoundToInt(out_r);
      V8x32I int_g = RoundToInt(out_g);
      V8x32I int_b = RoundToInt(out_b);

      // Bring upper 128 bits into lower so pack can outputs 8 consecutive u16.
      const V8x32I hi_r(_mm256_permute2x128_si256(int_r, int_r, 0x11));
      const V8x32I hi_g(_mm256_permute2x128_si256(int_g, int_g, 0x11));
      const V8x32I hi_b(_mm256_permute2x128_si256(int_b, int_b, 0x11));

      // We only need the lower 128 bits (8x u16).
      const V8x16U u16_r(
          _mm256_castsi256_si128(_mm256_packus_epi32(int_r, hi_r)));
      const V8x16U u16_g(
          _mm256_castsi256_si128(_mm256_packus_epi32(int_g, hi_g)));
      const V8x16U u16_b(
          _mm256_castsi256_si128(_mm256_packus_epi32(int_b, hi_b)));

      const int px = ix + xoff;
      Store(u16_r, row0 + px);
      Store(u16_g, row1 + px);
      Store(u16_b, row2 + px);
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
