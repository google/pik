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

#include <stdio.h>
#include <algorithm>
#include <sstream>
#include <vector>

#include "arch_specific.h"
#include "cache_aligned.h"
#include "common.h"
#include "compiler_specific.h"
#include "dct.h"
#include "opsin_codec.h"

namespace pik {

static const int kGlobalScaleDenom = 1 << 16;
static const int kQuantMax = 256;
static const int kDefaultQuant = 64;

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
  static double kQuantWeights[192] = {
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
      CacheAligned::Allocate(192 * sizeof(float)));
  for (int c = 0; c < 3; ++c) {
    for (int k_zz = 0; k_zz < 64; ++k_zz) {
      int k = kNaturalCoeffOrder[k_zz];
      int idx = k_zz * 3 + c;
      float idct_scale = kIDCTScales[k % 8] * kIDCTScales[k / 8] / 64.0f;
      double weight = GetQuantWeights()[idx];
      if (weight < 0.4) { weight = 0.4; }
      double mul = GetQuantizeMul()[c];
      table[c * 64 + k] = idct_scale / (weight * mul);
    }
  }
  return table;
}

const float* DequantMatrix() {
  static const float* const kDequantMatrix = NewDequantMatrix();
  return kDequantMatrix;
}

int ClampVal(int val) {
  return std::min(kQuantMax, std::max(1, val));
}

Quantizer::Quantizer(int quant_xsize, int quant_ysize) :
    quant_xsize_(quant_xsize),
    quant_ysize_(quant_ysize),
    global_scale_(kGlobalScaleDenom / kDefaultQuant),
    quant_patch_(kDefaultQuant),
    quant_dc_(kDefaultQuant),
    quant_img_ac_(quant_xsize_, quant_ysize_, kDefaultQuant),
    scale_(quant_xsize_ * 64, quant_ysize_),
    initialized_(false) {
}

bool Quantizer::SetQuantField(const float quant_dc, const ImageF& qf) {
  bool changed = false;
  int new_global_scale = 4096 * quant_dc;
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
    const float* const PIK_RESTRICT kDequantMatrix = DequantMatrix();
    std::vector<float> quant_matrix(192);
    for (int i = 0; i < quant_matrix.size(); ++i) {
      quant_matrix[i] = 1.0f / kDequantMatrix[i];
    }
    const float qdc = scale * quant_dc_;
    for (int y = 0; y < quant_ysize_; ++y) {
      auto row_q = quant_img_ac_.Row(y);
      auto row_scale = scale_.Row(y);
      for (int x = 0; x < quant_xsize_; ++x) {
        const int offset = x * 64;
        const float qac = scale * row_q[x];
        for (int c = 0; c < 3; ++c) {
          const float* const PIK_RESTRICT qm = &quant_matrix[c * 64];
          for (int k = 0; k < 64; ++k) {
            row_scale[c][offset + k] = qac * qm[k];
          }
          row_scale[c][offset] = qdc * qm[0];
        }
      }
    }
    inv_global_scale_ = 1.0f / scale;
    inv_quant_dc_ = 1.0f / qdc;
    inv_quant_patch_ = inv_global_scale_ / quant_patch_;
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

std::string Quantizer::Encode(PikImageSizeInfo* info) const {
  EncodedIntPlane encoded_plane =
      EncodePlane(quant_img_ac_, 1, kQuantMax, kSupertileInBlocks, info);
  std::stringstream ss;
  ss << std::string(1, (global_scale_ - 1) >> 8);
  ss << std::string(1, (global_scale_ - 1) & 0xff);
  ss << std::string(1, quant_patch_ - 1);
  ss << std::string(1, quant_dc_ - 1);
  ss << encoded_plane.preamble;
  for (int y = 0; y < encoded_plane.tiles.size(); y++) {
    for (int x = 0; x < encoded_plane.tiles[0].size(); x++) {
      ss << encoded_plane.tiles[y][x];
    }
  }
  if (info) {
    info->total_size += 4;
  }
  return ss.str();
}

bool Quantizer::Decode(BitReader* br) {
  global_scale_ = br->ReadBits(8) << 8;
  global_scale_ += br->ReadBits(8) + 1;
  quant_patch_ = br->ReadBits(8) + 1;
  quant_dc_ = br->ReadBits(8) + 1;
  IntPlaneDecoder decoder(1, kQuantMax, kSupertileInBlocks);
  if (!decoder.LoadPreamble(br)) {
    return false;
  }
  size_t tile_ysize =
      (quant_img_ac_.ysize() + kSupertileInBlocks - 1) / kSupertileInBlocks;
  size_t tile_xsize =
      (quant_img_ac_.xsize() + kSupertileInBlocks - 1) / kSupertileInBlocks;
  for (int y = 0; y < tile_ysize; y++) {
    for (int x = 0; x < tile_xsize; x++) {
      Image<int> tile =
          Window(&quant_img_ac_, x * kSupertileInBlocks, y * kSupertileInBlocks,
                 std::min<int>(kSupertileInBlocks,
                               quant_img_ac_.xsize() - kSupertileInBlocks * x),
                 std::min<int>(kSupertileInBlocks,
                               quant_img_ac_.ysize() - kSupertileInBlocks * y));
      if (!decoder.DecodeTile(br, &tile)) {
        return false;
      }
    }
  }
  inv_global_scale_ = kGlobalScaleDenom * 1.0 / global_scale_;
  inv_quant_dc_ = inv_global_scale_ / quant_dc_;
  inv_quant_patch_ = inv_global_scale_ / quant_patch_;
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

Image3W QuantizeCoeffs(const Image3F& in, const Quantizer& quantizer) {
  const int block_xsize = in.xsize() / 64;
  const int block_ysize = in.ysize();
  Image3W out(block_xsize * 64, block_ysize);
  for (int block_y = 0; block_y < block_ysize; ++block_y) {
    auto row_in = in.Row(block_y);
    auto row_out = out.Row(block_y);
    for (int block_x = 0; block_x < block_xsize; ++block_x) {
      const int offset = block_x * 64;
      for (int c = 0; c < 3; ++c) {
        quantizer.QuantizeBlock(block_x, block_y, c,
                                &row_in[c][offset], &row_out[c][offset]);
      }
    }
  }
  return out;
}

Image3F DequantizeCoeffs(const Image3W& in, const Quantizer& quantizer) {
  const int block_xsize = in.xsize() / 64;
  const int block_ysize = in.ysize();
  Image3F out(block_xsize * 64, block_ysize);
  const float inv_quant_dc = quantizer.inv_quant_dc();
  const float* PIK_RESTRICT kDequantMatrix = DequantMatrix();
  for (int by = 0; by < block_ysize; ++by) {
    auto row_in = in.Row(by);
    auto row_out = out.Row(by);
    for (int bx = 0; bx < block_xsize; ++bx) {
      const int offset = bx * 64;
      const float inv_quant_ac = quantizer.inv_quant_ac(bx, by);
      for (int c = 0; c < 3; ++c) {
        const int16_t* const PIK_RESTRICT block_in = &row_in[c][offset];
        const float* const PIK_RESTRICT muls = &kDequantMatrix[c * 64];
        float* const PIK_RESTRICT block_out = &row_out[c][offset];
        for (int k = 0; k < 64; ++k) {
          block_out[k] = block_in[k] * (muls[k] * inv_quant_ac);
        }
        block_out[0] = block_in[0] * (muls[0] * inv_quant_dc);
      }
    }
  }
  return out;
}

}  // namespace pik
