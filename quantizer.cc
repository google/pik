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

#define PROFILER_ENABLED 1
#include "arch_specific.h"
#include "cache_aligned.h"
#include "common.h"
#include "compiler_specific.h"
#include "dct.h"
#include "opsin_codec.h"
#include "profiler.h"

namespace pik {

static const int kQuantMax = 256;
static const int kDefaultQuant = 64;

const double *GetQuantizeMul() {
  static double kQuantizeMul[3] = {
    1.4005806299962318,
    0.40933660117814064,
    0.10258725594691469,
  };
  return &kQuantizeMul[0];
}

// kQuantWeights[3 * k_zz + c] is the relative weight of the k_zz coefficient
// (in the zig-zag order) in component c. Higher weights correspond to finer
// quantization intervals and more bits spent in encoding.
const double *GetQuantWeights() {
  static double kQuantWeights[192] = {
    2.9705640194848502,
    1.9202390579895725,
    3.6354976245395934,
    6.3234428150485584,
    1.8801243475008937,
    3.3278798030264283,
    4.516257490140422,
    1.6076869322505747,
    1.2595324090054545,
    1.3005839091766891,
    1.402804976761562,
    0.69705810296671755,
    2.9640522902382695,
    1.3027664574788567,
    2.6902530930556501,
    2.4241905258021594,
    1.2325844590841797,
    0.43880035220542124,
    1.6187352251449436,
    1.0377313026152126,
    0.35884942530210645,
    1.0706202151405182,
    0.92179138448654807,
    0.50629760421856751,
    1.6964041715175964,
    0.93954650719319088,
    0.55078961985035979,
    0.81460344462564382,
    0.67774859284316635,
    0.40589958294743217,
    0.86246185786638474,
    0.66406417481704794,
    0.39929367347116057,
    1.0868953067177509,
    0.73561629321015565,
    0.56859134006672163,
    0.85757966894128235,
    0.81078360425090468,
    0.3497175000335413,
    0.83284380562842331,
    0.6995744353123361,
    0.29242046939636163,
    0.49567471078409497,
    0.77804937433577737,
    0.40781251249535244,
    0.55386451469303266,
    0.69210744055448958,
    0.41141873202834295,
    0.75288462896865138,
    0.68340039007277342,
    0.61495289885705118,
    1.0037567653257879,
    0.74537951626437904,
    0.67255290273528134,
    1.1150561573137656,
    0.72089823366112371,
    0.7777087250238427,
    0.84176489432317803,
    0.67107763397654652,
    0.4940897105456033,
    0.43093722117853456,
    0.58160682128810892,
    0.34004147314011868,
    0.79298748244492678,
    0.43614379530608233,
    0.40586430395555539,
    0.66143985031579156,
    0.54755465239041956,
    0.53732689614708551,
    0.90495447762336134,
    0.74292960937027086,
    0.4743638308097583,
    0.94215500284017584,
    0.69870548352538864,
    0.18108222887874986,
    1.0681139424262458,
    0.66842186893807554,
    0.21118563999573331,
    0.79796223676638511,
    0.60532392222470821,
    0.24812018020168281,
    0.36805089628388965,
    0.44455602993606119,
    0.55883905530146949,
    0.52610433472859164,
    0.59018248646603755,
    0.60233039520352816,
    0.50574598309966279,
    0.49027654943420407,
    0.48716747238516078,
    0.81245099089543915,
    0.6845498074421057,
    0.2001650242276837,
    0.83420144719818179,
    0.58103653914846509,
    0.51235316154284971,
    0.88823649284445261,
    0.60653337132786966,
    0.15294703791328751,
    0.20010517602248992,
    0.58878460203058836,
    0.16019683447834746,
    0.33340239674990435,
    0.38257458823596668,
    0.20017241987883883,
    0.36976533766362812,
    0.38134745973762785,
    0.43070444590454321,
    0.30122543618529551,
    0.3251055768522465,
    0.226454770971834,
    0.47264465293240349,
    0.44177782408395744,
    0.40520909237414054,
    0.30397608064712511,
    0.49678118243028047,
    0.20402862322662335,
    0.20498814831726828,
    0.49486371680512575,
    0.20016325476225283,
    1.0617422929119589,
    0.54076831563621597,
    0.57536777398317229,
    0.45345020776084882,
    0.47571321661267546,
    0.1651918262493626,
    0.76006419454896035,
    0.38199767828112507,
    0.29660585868441502,
    0.55668354104517437,
    0.45622821053589929,
    0.18116705548613102,
    0.39520645984441616,
    0.5486429862240535,
    0.17450755125199355,
    0.41856478615896076,
    0.54379743713824102,
    0.20019173726098408,
    0.38302581605245672,
    0.5599085479028556,
    0.26120916369755121,
    0.98414318246098464,
    0.47490955425323045,
    0.63101633695202564,
    0.46923903538175682,
    0.4128523987645053,
    0.25447152501119352,
    0.44315993036624601,
    0.39127356690248416,
    0.41296205501589534,
    0.41275681477765674,
    0.6083770203616663,
    0.32346544800364269,
    0.61268011098251451,
    0.30091465985392402,
    0.2153085184604463,
    0.629350801717659,
    0.45542629804612705,
    0.50132904539874201,
    0.69025705968126694,
    0.46243178138476393,
    0.22901716192841889,
    0.37063731087030488,
    0.44800166202030933,
    0.4883460358805492,
    0.56449329219083522,
    0.42904910034933913,
    0.56054419569807701,
    0.16863544096144281,
    0.32664337317395886,
    0.2090418605593255,
    0.4836695373708757,
    0.42010547100545675,
    0.52607854741067372,
    0.16515290414067296,
    0.33055029651879408,
    0.41581743946185329,
    0.51365007745885227,
    0.32592062411666761,
    0.38604806626552368,
    0.60039426295653953,
    0.34977268787665594,
    0.58687083947335728,
    0.33767253855803891,
    0.30161924101009951,
    0.51822475503372301,
    0.6029299991619399,
    0.36486662746864756,
    0.39323392344734776,
    0.29357541933261994,
    0.33218174735094863,
    0.46593194758634049,
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
      if (weight < 0.22) { weight = 0.22; }
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

ImageD ComputeBlockDistanceQForm(const double lambda,
                                 const float* const PIK_RESTRICT scales) {
  ImageD A = Identity<double>(64);
  for (int ky = 0, k = 0; ky < 8; ++ky) {
    for (int kx = 0; kx < 8; ++kx, ++k) {
      const float scale =
          kRecipIDCTScales[kx] * kRecipIDCTScales[ky] / scales[k];
      A.Row(k)[k] *= scale * scale;
    }
  }
  for (int i = 0; i < 8; ++i) {
    const double scale_i = kRecipIDCTScales[i];
    for (int k = 0; k < 8; ++k) {
      const double sign_k = (k % 2 == 0 ? 1.0 : -1.0);
      const double scale_k = kRecipIDCTScales[k];
      const double scale_ik = scale_i * scale_k / scales[8 * i + k];
      const double scale_ki = scale_i * scale_k / scales[8 * k + i];
      for (int l = 0; l < 8; ++l) {
        const double sign_l = (l % 2 == 0 ? 1.0 : -1.0);
        const double scale_l = kRecipIDCTScales[l];
        const double scale_il = scale_i * scale_l / scales[8 * i + l];
        const double scale_li = scale_i * scale_l / scales[8 * l + i];
        A.Row(8 * i + k)[8 * i + l] +=
            (1.0 + sign_k * sign_l) * lambda * scale_ik * scale_il;
        A.Row(8 * k + i)[8 * l + i] +=
            (1.0 + sign_k * sign_l) * lambda * scale_ki * scale_li;
      }
    }
  }
  return A;
}

Quantizer::Quantizer(int quant_xsize, int quant_ysize) :
    quant_xsize_(quant_xsize),
    quant_ysize_(quant_ysize),
    global_scale_(kGlobalScaleDenom / kDefaultQuant),
    quant_patch_(kDefaultQuant),
    quant_dc_(kDefaultQuant),
    quant_img_ac_(quant_xsize_, quant_ysize_, kDefaultQuant),
    initialized_(false) {
}

bool Quantizer::SetQuantField(const float quant_dc, const ImageF& qf,
                              const CompressParams& cparams) {
  bool changed = false;
  int new_global_scale = 4096 * quant_dc;
  if (new_global_scale != global_scale_) {
    global_scale_ = new_global_scale;
    changed = true;
  }
  const float scale = Scale();
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
      for (int x = 0; x < quant_xsize_; ++x) {
        const float qac = scale * row_q[x];
        for (int c = 0; c < 3; ++c) {
          const uint64_t key = QuantizerKey(x, y, c);
          if (qmap_.find(key) == qmap_.end()) {
            const float* const PIK_RESTRICT qm = &quant_matrix[c * 64];
            BlockQuantizer bq;
            for (int k = 0; k < 64; ++k) {
              bq.scales[k] = (k == 0 ? qdc : qac) * qm[k];
            }
            if (cparams.quant_border_bias != 0.0) {
              ImageD A = ComputeBlockDistanceQForm(
                  cparams.quant_border_bias, &bq.scales[0]);
              LatticeOptimizer lattice;
              lattice.InitFromQuadraticForm(A);
              lattices_.emplace_back(std::move(lattice));
              bq.lattice_idx = lattices_.size() - 1;
            }
            qmap_.insert(std::make_pair(key, std::move(bq)));
          }
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
  const float scale = Scale();
  *quant_dc = scale * quant_dc_;
  *qf = ImageF(quant_xsize_, quant_ysize_);
  for (int y = 0; y < quant_ysize_; ++y) {
    for (int x = 0; x < quant_xsize_; ++x) {
      qf->Row(y)[x] = scale * quant_img_ac_.Row(y)[x];
    }
  }
}

std::string Quantizer::Encode(PikImageSizeInfo* info) const {
  EncodedIntPlane encoded_plane = EncodePlane(
      quant_img_ac_, 1, kQuantMax, kSupertileInBlocks, 1, nullptr, info);
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
  IntPlaneDecoder decoder(1, kQuantMax, kSupertileInBlocks, 1);
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
      if (!decoder.DecodeTile(br, &tile, nullptr)) {
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

Image3S QuantizeCoeffs(const Image3F& in, const Quantizer& quantizer) {
  const int block_xsize = in.xsize() / 64;
  const int block_ysize = in.ysize();
  Image3S out(block_xsize * 64, block_ysize);
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

Image3F DequantizeCoeffs(const Image3S& in, const Quantizer& quantizer) {
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

ImageF QuantizeRoundtrip(const Quantizer& quantizer, int c, const ImageF& img) {
  const int block_xsize = img.xsize() / 64;
  const int block_ysize = img.ysize();
  const float inv_quant_dc = quantizer.inv_quant_dc();
  const float* const PIK_RESTRICT kDequantMatrix = &DequantMatrix()[c * 64];
  ImageF out(img.xsize(), img.ysize());
  for (int block_y = 0; block_y < block_ysize; ++block_y) {
    const float* const PIK_RESTRICT row_in = img.ConstRow(block_y);
    float* const PIK_RESTRICT row_out = out.Row(block_y);
    for (int block_x = 0; block_x < block_xsize; ++block_x) {
      const float inv_quant_ac = quantizer.inv_quant_ac(block_x, block_y);
      const float* const PIK_RESTRICT block_in = &row_in[block_x * 64];
      float* const PIK_RESTRICT block_out = &row_out[block_x * 64];
      int16_t qblock[64];
      quantizer.QuantizeBlock(block_x, block_y, c, block_in, qblock);
      block_out[0] = qblock[0] * (kDequantMatrix[0] * inv_quant_dc);
      for (int k = 1; k < 64; ++k) {
        block_out[k] = qblock[k] * (kDequantMatrix[k] * inv_quant_ac);
      }
    }
  }
  return out;
}

PIK_INLINE void DequantizeCoeffsT(const ConstImageViewF* in,
                                  const ConstImageViewF& in_quant_ac,
                                  const Quantizer& quantizer,
                                  const OutputRegion& output_region,
                                  const MutableImageViewF* PIK_RESTRICT out) {
  PROFILER_ZONE("|| Dequant");
  const size_t xsize = output_region.xsize;
  PIK_CHECK(xsize % 64 == 0);
  const size_t ysize = output_region.ysize;
  const float inv_quant_dc = quantizer.inv_quant_dc();
  const float inv_global_scale = quantizer.InvGlobalScale();
  const float* PIK_RESTRICT kDequantMatrix = DequantMatrix();

  for (int c = 0; c < 3; ++c) {
    const float* const PIK_RESTRICT muls = &kDequantMatrix[c * 64];

    for (size_t by = 0; by < ysize; ++by) {
      const int16_t* PIK_RESTRICT row_in =
          reinterpret_cast<const int16_t*>(in[c].ConstRow(by));
      const int* PIK_RESTRICT row_quant_ac =
          reinterpret_cast<const int*>(in_quant_ac.ConstRow(by));
      float* PIK_RESTRICT row_out = out[c].Row(by);

      for (size_t bx = 0; bx < xsize; bx += 64) {
        const float inv_quant_ac =
            row_quant_ac[bx / 64] == 0
                ? 1E10
                : inv_global_scale / row_quant_ac[bx / 64];

        // Produce one whole block
        for (int k = 0; k < 64; ++k) {
          row_out[bx + k] = row_in[bx + k] * (muls[k] * inv_quant_ac);
        }
        row_out[bx + 0] = row_in[bx + 0] * (muls[0] * inv_quant_dc);
      }
    }
  }
}

TFNode* AddDequantize(const TFPorts in_xyb, const TFPorts in_quant_ac,
                      const Quantizer& quantizer, TFBuilder* builder) {
  PIK_CHECK(OutType(in_xyb.node) == TFType::kI16);
  PIK_CHECK(OutType(in_quant_ac.node) == TFType::kI32);
  return builder->AddClosure(
      "dequantize", Borders(), Scale(), {in_xyb, in_quant_ac}, 3, TFType::kF32,
      [&quantizer](const ConstImageViewF* in, const OutputRegion& output_region,
                   const MutableImageViewF* PIK_RESTRICT out) {
        DequantizeCoeffsT(in, in[3], quantizer, output_region, out);
      });
}

}  // namespace pik
