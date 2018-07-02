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
#include "entropy_coder.h"
#include "profiler.h"
#include "simd/simd.h"

namespace pik {

static const int kQuantMax = 256;
static const int kDefaultQuant = 64;

// kQuantWeights[3 * k_zz + c] is the relative weight of the k_zz coefficient
// (in the zig-zag order) in component c. Higher weights correspond to finer
// quantization intervals and more bits spent in encoding.
const double *GetQuantWeights() {
  static double kQuantWeights[kNumQuantTables * 192] = {
    // Default weights
    4.1733078095895006,
    0.78593770355968129,
    0.38578847769569313,
    8.853876433840929,
    0.76788312586316276,
    0.14461252929994023,
    6.3232298425000186,
    0.65099838212728112,
    0.1266563243187202,
    1.7862177888519044,
    0.5746008931503862,
    0.074604026163347603,
    4.1555173208263883,
    0.53554285635184773,
    0.15520338192051591,
    3.3458519114778968,
    0.50333857016214811,
    0.070858096506561399,
    2.296893948128746,
    0.42622279050320167,
    0.048674823313911965,
    1.4885780161458244,
    0.3687808436096065,
    0.076300137575658122,
    2.4139184149745567,
    0.39435319695976429,
    0.058025002476468621,
    1.120754297237603,
    0.28333665598728924,
    0.039138950082789496,
    1.2434721004645621,
    0.27472057268984629,
    0.061551670517707983,
    1.5374932355498534,
    0.30308906885952053,
    0.065091208872864834,
    1.2453939161042882,
    0.34080830849246618,
    0.056092514813654779,
    1.1725294693833523,
    0.30038269039534299,
    0.031259908697982924,
    0.75462596142298877,
    0.31850863435368459,
    0.044862822765910852,
    0.77298527895081193,
    0.27748895846884153,
    0.04655026980588621,
    1.0873510280709648,
    0.27827081992196373,
    0.063340526791662438,
    1.4311063444212269,
    0.30536835804819662,
    0.068930612389398307,
    1.5346255308243268,
    0.29465845085681081,
    0.081538270725194692,
    1.1812369490149721,
    0.26817954900446278,
    0.053608569909135893,
    0.62831821500042262,
    0.23729237594874181,
    0.036877370952991241,
    1.1052826204503181,
    0.17214418584113564,
    0.045117485944386991,
    0.85579310239731532,
    0.22549923816315212,
    0.052315718492412205,
    1.3047904539498623,
    0.30941475611054703,
    0.04238559212190364,
    1.3288453520131398,
    0.2860583731917799,
    0.017031196809487469,
    1.496444952650779,
    0.27440223598724678,
    0.019625891067074391,
    1.0470920813213835,
    0.22637311462299828,
    0.026096898367408108,
    0.51376168151717849,
    0.18202449441504173,
    0.058254372275933698,
    0.73859681490493811,
    0.24404843487282943,
    0.056109123335709679,
    0.68920909391753804,
    0.19836413247470186,
    0.060062361211572289,
    1.1153755891168651,
    0.28150710753914721,
    0.018725751574998992,
    1.1261622550719961,
    0.24548186546634082,
    0.055642815789203354,
    1.2753041544134316,
    0.24829973929559968,
    0.015684512650313471,
    0.27104801013678009,
    0.24184587410360447,
    0.01631327440130733,
    0.50050209907053689,
    0.14960357619556947,
    0.02120393922289723,
    0.49041676777706517,
    0.15828091408851544,
    0.043596635056880134,
    0.44179275068438345,
    0.13046340171703644,
    0.028560983864841093,
    0.72748689603392902,
    0.17958975699379678,
    0.038707090744205516,
    0.45256202182289995,
    0.20336945742272233,
    0.019448223663882141,
    0.49171143527691319,
    0.20255480817385629,
    0.020011842131995132,
    1.4979313912649135,
    0.21501229268567693,
    0.061338559036430519,
    0.69213895881381626,
    0.19542968908531771,
    0.015983734877713689,
    1.0775771951849507,
    0.15696697919052124,
    0.032351187298304621,
    0.7828611208588272,
    0.14828436558403618,
    0.01866325423477215,
    0.53413120078476883,
    0.22540549682359542,
    0.017103893921396,
    0.64994757352196075,
    0.21557690374079819,
    0.02028769311311825,
    0.67160098208800201,
    0.22654984681541648,
    0.031979226251359121,
    1.37558530050399797,
    0.19032544668628112,
    0.06550400570919304,
    0.65627644060647694,
    0.18257594324102225,
    0.02736150102783333,
    0.69658722851847443,
    0.16019921330041217,
    0.03928658926666367,
    0.60383586256890809,
    0.25321132092045168,
    0.03265559675305111,
    0.83693327158281239,
    0.14504128077517850,
    0.02099986490870039,
    0.89483155160749195,
    0.19135027758942161,
    0.05451133366516921,
    1.01079312492116169,
    0.18828293699332641,
    0.02114328476201841,
    0.61433852336830252,
    0.18195885412296919,
    0.04756213156467592,
    0.85359113948742638,
    0.17833895130867813,
    0.06022307525778426,
    0.19690151314535168,
    0.14519022078866697,
    0.02011343570768113,
    0.68009239530229870,
    0.17833653288483989,
    0.05546182646799611,
    0.25124318417438091,
    0.13437112957798944,
    0.04817602415230767,
    0.76255544979395051,
    0.13009719708974776,
    0.04122936655509454,
    0.80566896654017395,
    0.13810327055412253,
    0.05703495243404272,
    0.42317017460668760,
    0.12932494829955105,
    0.05788307678117844,
    0.88814332926268424,
    0.14743472727149085,
    0.03950479954925656,
    0.42966332525140621,
    0.14234967223268227,
    0.04663425547477144,

    // Weights for HQ mode
    5.97098671763256839,
    1.69761990929546780,
    0.53853771332792444,
    3.14253990320328125,
    0.78664853390117273,
    0.17190174775266073,
    3.21440032329272363,
    0.79324113975890986,
    0.19073348746478294,
    2.31877876078626644,
    1.20331026659435092,
    0.18068627495819625,
    2.66138348593645846,
    1.27976364845854129,
    0.14314257421962365,
    2.05669558255116414,
    0.67193542643511317,
    0.12976541962082244,
    1.89568467702556931,
    0.87684508464905198,
    0.13840382703261955,
    1.87183877378106178,
    0.87332866827360733,
    0.13886642497406188,
    1.81376345075241852,
    0.90546493291290131,
    0.12640806528419762,
    1.38659706271567829,
    0.67357513728317564,
    0.10046114149393238,
    1.43970070343494916,
    0.64960619501539618,
    0.10434852182815810,
    1.64121019910504962,
    0.67261342673313418,
    0.12842635952387660,
    1.46477538313861277,
    0.70021253533295424,
    0.10903159857162095,
    1.52923458157991998,
    0.69961592005489026,
    0.09710617072173894,
    1.40023384922801197,
    0.68455392518694513,
    0.07908914182002977,
    1.33839402167055588,
    0.69211618315286805,
    0.09308446231558251,
    1.43643407833087378,
    0.69628390114558059,
    0.08386075776976390,
    1.42012129383485530,
    0.65847070005190744,
    0.09890816230765243,
    1.68161355944399049,
    0.71866143043615716,
    0.09940258437481214,
    1.56772586225783828,
    0.67160483088794198,
    0.09466185289417853,
    1.36510921600277535,
    0.62401571586763904,
    0.08236434651134468,
    1.03208101627848747,
    0.59120245407886063,
    0.05912383931246348,
    1.69702527326663510,
    0.68535225222980134,
    0.09623784903161817,
    1.53252814802078419,
    0.72569260047525008,
    0.08106185965616462,
    1.40794482088634898,
    0.68347344254745312,
    0.09180649324754615,
    0.98409717572352373,
    0.66630217832243899,
    0.08752033959765924,
    1.41539698889807042,
    0.51216510866872778,
    0.06712898051010259,
    0.95856947616046384,
    0.50258164653939053,
    0.05876841963372925,
    0.82585586314858783,
    0.60553389435339600,
    0.05851610562550572,
    1.32551190883622838,
    0.52116816325598003,
    0.05776704204017307,
    1.42525227948613864,
    0.66451111078322889,
    0.09029381536978218,
    1.61480964386399450,
    0.58319461600661016,
    0.08374152456688765,
    1.36272444106781343,
    0.56725685656994129,
    0.07447525932711950,
    1.23016114633190843,
    0.54560308162655557,
    0.06244568047577013,
    1.02531080328291346,
    0.51073378680450165,
    0.05900332401163505,
    0.79645147162795993,
    0.48279035076704235,
    0.05942394100369194,
    1.10224441792333505,
    0.51399854683593382,
    0.06488521108420610,
    1.32841168654714181,
    0.55615524649126835,
    0.09305414673745493,
    1.58130787393576955,
    0.52228127315219441,
    0.09462056731598355,
    1.32305080787503493,
    0.56539808782479895,
    0.08967000072418904,
    1.60940316666871319,
    0.60902893903479105,
    0.08559545911151349,
    1.15852808153812559,
    0.57532339125302434,
    0.07594769254966900,
    1.41422670295622654,
    0.51208334668238098,
    0.08262610724104018,
    1.50123574147011585,
    0.61420516049012464,
    0.08996506605454008,
    1.43030813640267751,
    0.52583680560641410,
    0.07164827618952087,
    1.66786924477306031,
    0.56912874262481383,
    0.06055826950374100,
    1.01687835220554090,
    0.53050807292462376,
    0.06116745665900101,
    1.53314369451989063,
    0.58806280311474168,
    0.10622593889251969,
    1.13915364970228650,
    0.51607666905695815,
    0.09892489416863132,
    1.20062442282843995,
    0.62042257791036048,
    0.07859956608053299,
    1.57803627072360175,
    0.56412252798799212,
    0.08054184901244756,
    1.45092323858166239,
    0.52681964760491928,
    0.07837902068062400,
    1.54334806766566768,
    0.52727572293349534,
    0.08728601353049063,
    1.39711767258527320,
    0.57393681796751261,
    0.07930505691716441,
    0.78158104550004404,
    0.60390867209622334,
    0.07462500390508715,
    1.81436012692921311,
    0.54071907903714811,
    0.07981141132875894,
    0.78511966383388032,
    0.55016303699852442,
    0.07926565080862039,
    1.15182975200692361,
    0.56361259875118574,
    0.09215829949648185,
    0.92065100803555544,
    0.56635179840667760,
    0.10282781177064568,
    1.22537108443054898,
    0.54603239891514965,
    0.08249748895287572,
    1.57458694038461045,
    0.53538377686685823,
    0.07811475203273252,
    1.30320516365488825,
    0.46393811087230996,
    0.09657913185935441,
    0.94422674464538836,
    0.46159976390783986,
    0.09834404184403754,
    1.43973209699300408,
    0.46356335670292936,
    0.07601385475613358,
  };
  return &kQuantWeights[0];
}

const float* NewDequantMatrices() {
  float* table = static_cast<float*>(
      CacheAligned::Allocate(kNumQuantTables * 192 * sizeof(float)));
  for (int id = 0; id < kNumQuantTables; ++id) {
    for (int idx = 0; idx < 192; ++idx) {
      int c = idx % 3;
      int k_zz = idx / 3;
      int k = kNaturalCoeffOrder[k_zz];
      float idct_scale = kIDCTScales[k % 8] * kIDCTScales[k / 8] / 64.0f;
      double weight = GetQuantWeights()[id * 192 + idx];
      table[id * 192 + c * 64 + k] = idct_scale / weight;
    }
  }
  return table;
}

// Returns aligned memory.
const float* DequantMatrix(int id) {
  static const float* const kDequantMatrix = NewDequantMatrices();
  return &kDequantMatrix[id * 192];
}

int ClampVal(int val) {
  return std::min(kQuantMax, std::max(1, val));
}

ImageD ComputeBlockDistanceQForm(const double lambda,
                                 const float* PIK_RESTRICT scales) {
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

Quantizer::Quantizer(int template_id, int quant_xsize, int quant_ysize) :
    template_id_(template_id),
    quant_xsize_(quant_xsize),
    quant_ysize_(quant_ysize),
    global_scale_(kGlobalScaleDenom / kDefaultQuant),
    quant_patch_(kDefaultQuant),
    quant_dc_(kDefaultQuant),
    quant_img_ac_(quant_xsize_, quant_ysize_, kDefaultQuant),
    initialized_(false) {
  if (template_id == kQuantHQ) {
    memcpy(zero_bias_, kZeroBiasHQ, sizeof(kZeroBiasHQ));
  } else {
    memcpy(zero_bias_, kZeroBiasDefault, sizeof(kZeroBiasDefault));
  }
}

const float* Quantizer::DequantMatrix() const {
  return pik::DequantMatrix(template_id_);
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
    const float* PIK_RESTRICT kDequantMatrix = DequantMatrix();
    std::vector<float> quant_matrix(192);
    for (int i = 0; i < quant_matrix.size(); ++i) {
      quant_matrix[i] = 1.0f / kDequantMatrix[i];
    }
    const float qdc = scale * quant_dc_;
    for (int y = 0; y < quant_ysize_; ++y) {
      const int32_t* PIK_RESTRICT row_q = quant_img_ac_.Row(y);
      for (int x = 0; x < quant_xsize_; ++x) {
        const float qac = scale * row_q[x];
        const Key key = QuantizerKey(x, y);
        for (int c = 0; c < 3; ++c) {
          if (bq_[c].Find(key) == nullptr) {
            const float* PIK_RESTRICT qm = &quant_matrix[c * 64];
            BlockQuantizer* bq = bq_[c].Add(key);
            bq->scales[0] = qdc * qm[0];
            for (int k = 1; k < 64; ++k) {
              bq->scales[k] = qac * qm[k];
            }
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
  std::stringstream ss;
  ss << std::string(1, (global_scale_ - 1) >> 8);
  ss << std::string(1, (global_scale_ - 1) & 0xff);
  ss << std::string(1, quant_patch_ - 1);
  ss << std::string(1, quant_dc_ - 1);
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
  PROFILER_FUNC;
  const size_t block_xsize = in.xsize() / 64;
  const size_t block_ysize = in.ysize();
  Image3S out(block_xsize * 64, block_ysize);
  for (int c = 0; c < 3; ++c) {
    for (size_t block_y = 0; block_y < block_ysize; ++block_y) {
      const float* PIK_RESTRICT row_in = in.PlaneRow(c, block_y);
      int16_t* PIK_RESTRICT row_out = out.PlaneRow(c, block_y);
      for (size_t block_x = 0; block_x < block_xsize; ++block_x) {
        const float* PIK_RESTRICT block_in = &row_in[block_x * 64];
        int16_t* PIK_RESTRICT block_out = &row_out[block_x * 64];
        quantizer.QuantizeBlock(block_x, block_y, c, block_in, block_out);
      }
    }
  }
  return out;
}

// Returns DC only; "in" is 1x64 blocks.
Image3S QuantizeCoeffsDC(const Image3F& in, const Quantizer& quantizer) {
  const size_t block_xsize = in.xsize() / 64;
  const size_t block_ysize = in.ysize();
  Image3S out(block_xsize, block_ysize);
  for (int c = 0; c < 3; ++c) {
    for (size_t block_y = 0; block_y < block_ysize; ++block_y) {
      const float* PIK_RESTRICT row_in = in.PlaneRow(c, block_y);
      int16_t* PIK_RESTRICT row_out = out.PlaneRow(c, block_y);
      for (size_t block_x = 0; block_x < block_xsize; ++block_x) {
        const float* PIK_RESTRICT block_in = &row_in[block_x * 64];
        row_out[block_x] =
            quantizer.QuantizeBlockDC(block_x, block_y, c, block_in);
      }
    }
  }
  return out;
}

// Superceded by QuantizeRoundtrip and DequantizeCoeffsT.
Image3F DequantizeCoeffs(const Image3S& in, const Quantizer& quantizer) {
  PROFILER_FUNC;
  const int block_xsize = in.xsize() / 64;
  const int block_ysize = in.ysize();
  Image3F out(block_xsize * 64, block_ysize);
  const float inv_quant_dc = quantizer.inv_quant_dc();
  const float* PIK_RESTRICT kDequantMatrix = quantizer.DequantMatrix();
  const int16_t* all_block_in[3];
  float* all_block_out[3];
  for (int by = 0; by < block_ysize; ++by) {
    for (int c = 0; c < 3; ++c) {
      all_block_in[c] = in.PlaneRow(c, by);
      all_block_out[c] = out.PlaneRow(c, by);
    }
    for (int bx = 0; bx < block_xsize; ++bx) {
      const float inv_quant_ac = quantizer.inv_quant_ac(bx, by);
      const int offset = bx * 64;
      for (int c = 0; c < 3; ++c) {
        const int16_t* PIK_RESTRICT block_in = all_block_in[c] + offset;
        const float* PIK_RESTRICT muls = &kDequantMatrix[c * 64];
        float* PIK_RESTRICT block_out = all_block_out[c] + offset;
        for (int k = 0; k < 64; ++k) {
          block_out[k] = block_in[k] * (muls[k] * inv_quant_ac);
        }
        block_out[0] = block_in[0] * (muls[0] * inv_quant_dc);
      }
    }
  }
  return out;
}

ImageF QuantizeRoundtrip(const Quantizer& quantizer, int c,
                         const ImageF& coeffs) {
  const size_t block_xsize = coeffs.xsize() / 64;
  const size_t block_ysize = coeffs.ysize();
  ImageF out(coeffs.xsize(), coeffs.ysize());

  const float inv_quant_dc = quantizer.inv_quant_dc();
  const float* PIK_RESTRICT kDequantMatrix = &quantizer.DequantMatrix()[c * 64];

  for (size_t block_y = 0; block_y < block_ysize; ++block_y) {
    const float* PIK_RESTRICT row_in = coeffs.ConstRow(block_y);
    float* PIK_RESTRICT row_out = out.Row(block_y);
    for (size_t block_x = 0; block_x < block_xsize; ++block_x) {
      const float inv_quant_ac = quantizer.inv_quant_ac(block_x, block_y);
      const float* PIK_RESTRICT block_in = &row_in[block_x * 64];
      float* PIK_RESTRICT block_out = &row_out[block_x * 64];
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

ImageF QuantizeRoundtripExtract189(const Quantizer& quantizer, int c,
                                   const ImageF& coeffs) {
  const size_t block_xsize = coeffs.xsize() / 64;
  const size_t block_ysize = coeffs.ysize();
  ImageF out(4 * block_xsize, block_ysize);

  const float* PIK_RESTRICT kDequantMatrix = &quantizer.DequantMatrix()[c * 64];

  for (size_t block_y = 0; block_y < block_ysize; ++block_y) {
    const float* PIK_RESTRICT row_in = coeffs.ConstRow(block_y);
    float* PIK_RESTRICT row_out = out.Row(block_y);
    for (size_t block_x = 0; block_x < block_xsize; ++block_x) {
      const float inv_quant_ac = quantizer.inv_quant_ac(block_x, block_y);
      const float* PIK_RESTRICT block_in = &row_in[block_x * 64];
      float* PIK_RESTRICT block_out = &row_out[block_x * 4];
      int16_t qblock[64];
      quantizer.QuantizeBlock2x2(block_x, block_y, c, block_in, qblock);
      block_out[1] = qblock[1] * (kDequantMatrix[1] * inv_quant_ac);
      block_out[2] = qblock[8] * (kDequantMatrix[8] * inv_quant_ac);
      block_out[3] = qblock[9] * (kDequantMatrix[9] * inv_quant_ac);
    }
  }
  return out;
}

ImageF QuantizeRoundtripExtractDC(const Quantizer& quantizer, int c,
                                  const ImageF& coeffs) {
  // All coordinates are blocks.
  const int block_xsize = coeffs.xsize() / 64;
  const int block_ysize = coeffs.ysize();
  ImageF out(block_xsize, block_ysize);

  const float mul =
      quantizer.DequantMatrix()[c * 64] * quantizer.inv_quant_dc();
  for (size_t block_y = 0; block_y < block_ysize; ++block_y) {
    const float* PIK_RESTRICT row_in = coeffs.ConstRow(block_y);
    float* PIK_RESTRICT row_out = out.Row(block_y);
    for (size_t block_x = 0; block_x < block_xsize; ++block_x) {
      const float* PIK_RESTRICT block_in = &row_in[block_x * 64];
      row_out[block_x] =
          quantizer.QuantizeBlockDC(block_x, block_y, c, block_in) * mul;
    }
  }
  return out;
}

ImageF QuantizeRoundtripDC(const Quantizer& quantizer, int c,
                           const ImageF& dc) {
  // All coordinates are blocks.
  const int block_xsize = dc.xsize();
  const int block_ysize = dc.ysize();
  ImageF out(block_xsize, block_ysize);

  const float mul =
      quantizer.DequantMatrix()[c * 64] * quantizer.inv_quant_dc();
  for (size_t block_y = 0; block_y < block_ysize; ++block_y) {
    const float* PIK_RESTRICT row_in = dc.ConstRow(block_y);
    float* PIK_RESTRICT row_out = out.Row(block_y);
    for (size_t block_x = 0; block_x < block_xsize; ++block_x) {
      const float* PIK_RESTRICT block_in = row_in + block_x;
      row_out[block_x] =
          quantizer.QuantizeBlockDC(block_x, block_y, c, block_in) * mul;
    }
  }
  return out;
}

Image3F QuantizeRoundtrip(const Quantizer& quantizer, const Image3F& coeffs) {
  return Image3F(QuantizeRoundtrip(quantizer, 0, coeffs.plane(0)),
                 QuantizeRoundtrip(quantizer, 1, coeffs.plane(1)),
                 QuantizeRoundtrip(quantizer, 2, coeffs.plane(2)));
}

Image3F QuantizeRoundtripExtract189(const Quantizer& quantizer,
                                    const Image3F& coeffs) {
  return Image3F(QuantizeRoundtripExtract189(quantizer, 0, coeffs.plane(0)),
                 QuantizeRoundtripExtract189(quantizer, 1, coeffs.plane(1)),
                 QuantizeRoundtripExtract189(quantizer, 2, coeffs.plane(2)));
}

Image3F QuantizeRoundtripExtractDC(const Quantizer& quantizer,
                                   const Image3F& coeffs) {
  return Image3F(QuantizeRoundtripExtractDC(quantizer, 0, coeffs.plane(0)),
                 QuantizeRoundtripExtractDC(quantizer, 1, coeffs.plane(1)),
                 QuantizeRoundtripExtractDC(quantizer, 2, coeffs.plane(2)));
}

Image3F QuantizeRoundtripDC(const Quantizer& quantizer, const Image3F& coeffs) {
  return Image3F(QuantizeRoundtripDC(quantizer, 0, coeffs.plane(0)),
                 QuantizeRoundtripDC(quantizer, 1, coeffs.plane(1)),
                 QuantizeRoundtripDC(quantizer, 2, coeffs.plane(2)));
}

TFNode* AddDequantize(const TFPorts in_xyb, const TFPorts in_quant_ac,
                      const Quantizer& quantizer, TFBuilder* builder) {
  PIK_CHECK(OutType(in_xyb.node) == TFType::kI16);
  PIK_CHECK(OutType(in_quant_ac.node) == TFType::kI32);
  const float inv_global_scale = quantizer.InvGlobalScale();
  const float* PIK_RESTRICT dequant_matrix = quantizer.DequantMatrix();

  return builder->AddClosure(
      "dequantize", Borders(), Scale(), {in_xyb, in_quant_ac}, 3, TFType::kF32,
      [inv_global_scale, dequant_matrix](
          const ConstImageViewF* in, const OutputRegion& output_region,
          const MutableImageViewF* PIK_RESTRICT out) {
        const size_t xsize = output_region.xsize;
        PIK_ASSERT(xsize % 64 == 0);
        const size_t ysize = output_region.ysize;

        for (int c = 0; c < 3; ++c) {
          const float* PIK_RESTRICT muls = &dequant_matrix[c * 64];

          for (size_t by = 0; by < ysize; ++by) {
            const int16_t* PIK_RESTRICT row_in =
                reinterpret_cast<const int16_t*>(in[c].ConstRow(by));
            const int* PIK_RESTRICT row_quant_ac =
                reinterpret_cast<const int*>(in[3].ConstRow(by));
            float* PIK_RESTRICT row_out = out[c].Row(by);

            for (size_t bx = 0; bx < xsize; bx += 64) {
              using namespace SIMD_NAMESPACE;
              using D = Full<float>;
              constexpr D d;
              constexpr Part<int16_t, D::N> d16;
              constexpr Part<int32_t, D::N> d32;

              const auto inv_quant_ac =
                  set1(d, row_quant_ac[bx / 64] == 0
                              ? 1E10f
                              : inv_global_scale / row_quant_ac[bx / 64]);

              // Produce one whole block (k=0 is unnecessary but allows SIMD)
              for (size_t k = 0; k < 64; k += d.N) {
                const auto in16 = load(d16, row_in + bx + k);
                const auto in = convert_to(d, convert_to(d32, in16));
                const auto dequantized = in * load(d, muls + k) * inv_quant_ac;
                store(dequantized, d, row_out + bx + k);
              }
            }
          }
        }
      });
}

TFNode* AddDequantizeDC(const TFPorts in_xyb, const Quantizer& quantizer,
                        TFBuilder* builder) {
  PIK_CHECK(OutType(in_xyb.node) == TFType::kI16);
  const float* PIK_RESTRICT dequant_matrix = quantizer.DequantMatrix();
  float mul_dc[3];
  for (int c = 0; c < 3; ++c) {
    mul_dc[c] = dequant_matrix[c * 64] * quantizer.inv_quant_dc();
  }

  return builder->AddClosure(
      "dequantizeDC", Borders(), Scale(), {in_xyb}, 3, TFType::kF32,
      [mul_dc](const ConstImageViewF* in, const OutputRegion& output_region,
               const MutableImageViewF* PIK_RESTRICT out) {
        for (int c = 0; c < 3; ++c) {
          using namespace SIMD_NAMESPACE;
          using D = Full<float>;
          constexpr D d;
          constexpr Part<int16_t, D::N> d16;
          constexpr Part<int32_t, D::N> d32;
          const auto vmul = set1(d, mul_dc[c]);

          for (size_t y = 0; y < output_region.ysize; ++y) {
            const int16_t* PIK_RESTRICT row_in =
                reinterpret_cast<const int16_t*>(in[c].ConstRow(y));
            float* PIK_RESTRICT row_out = out[c].Row(y);

            for (size_t x = 0; x < output_region.xsize; x += d.N) {
              const auto in16 = load(d16, row_in + x);
              const auto in = convert_to(d, convert_to(d32, in16));
              const auto dequantized = in * vmul;
              store(dequantized, d, row_out + x);
            }
          }
        }
      });
}

}  // namespace pik
