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

static const int kDefaultQuant = 64;

// kQuantWeights[3 * k_zz + c] is the relative weight of the k_zz coefficient
// (in the zig-zag order) in component c. Higher weights correspond to finer
// quantization intervals and more bits spent in encoding.
const double *GetQuantWeights() {
  static double kQuantWeights[kNumQuantTables * 192] = {
    // Default weights
    4.3141203228013438,
    0.83740203419092563,
    0.51137522717087869,
    8.736834574015262,
    0.7776397194427076,
    0.13442931289541543,
    6.3379346742508131,
    0.70692883300525067,
    0.12368899522321172,
    1.7679749651060572,
    0.65347325775644483,
    0.23333737574436145,
    4.1820889419933449,
    0.61826473894479117,
    0.16390510194206945,
    3.321466890319678,
    0.50605143545289055,
    0.1042338362877393,
    2.2121858502448699,
    0.39737454618364199,
    0.039923519117012793,
    1.5048430743835741,
    0.44253191996134272,
    0.081069424968966131,
    2.4362778754556715,
    0.43257013459329918,
    0.074344935656858804,
    1.1783892036481227,
    0.30224131501589768,
    0.035528575800878801,
    1.3283725533453592,
    0.22521680805375197,
    0.081862698246215573,
    1.5580768395244231,
    0.25349576252750305,
    0.11241739107306357,
    1.1384409237139839,
    0.32444867301666058,
    0.069356620702463886,
    1.1693363410240802,
    0.22720605247313719,
    0.072073951417197107,
    0.76754319201101462,
    0.21674064978995528,
    0.049027879973039395,
    0.64727928422122272,
    0.18328656176888211,
    0.063889525390303029,
    1.1045513421611661,
    0.21585103511079337,
    0.074571320833125856,
    1.4832095549023419,
    0.24363059692829409,
    0.046613553523984899,
    1.5262612594710887,
    0.2658115196632741,
    0.095593946920371917,
    1.2583542418439091,
    0.20278784329053368,
    0.10198656903794691,
    0.42008027458003649,
    0.12917105983800514,
    0.10283606990457778,
    1.2115017196865743,
    0.12382691731111502,
    0.1118578524826251,
    1.0284671621456556,
    0.13754624577284763,
    0.065629383999491628,
    1.2151126659170599,
    0.20578654159697155,
    0.067402764565903606,
    1.3360284442899479,
    0.18668594655278073,
    0.056478124053817289,
    1.3941851810988457,
    0.18141839274334995,
    0.069418113786006708,
    1.0857335532257077,
    0.14874461747004106,
    0.071432726113976608,
    0.49362766892933202,
    0.10020854148621977,
    0.027582794969484382,
    0.65702715944968204,
    0.096304882733300276,
    0.091792356939215544,
    0.32829090137286626,
    0.076386710316435236,
    0.012417922327300664,
    0.83052927891064143,
    0.15665165444878068,
    0.043896269389724109,
    1.0434079920710564,
    0.21053548494276011,
    0.046276756365166177,
    1.2094211085965179,
    0.16230067189998562,
    0.066730255542867128,
    0.24506686179103726,
    0.11717160108133351,
    0.071165909610693245,
    0.5961782045984203,
    0.10906846987703636,
    0.1074957435127179,
    0.37538159142202759,
    0.071397660203107588,
    0.050158510029965957,
    0.28792379690718628,
    0.088841068516194235,
    0.032875806394252971,
    0.56685852046049934,
    0.16199207885726091,
    0.070272384862404516,
    0.60238104605220288,
    0.13140304193241348,
    0.036894168984050693,
    0.43756449773263967,
    0.10538224819307006,
    0.061117235994783574,
    1.445481349271859,
    0.11056598177328424,
    0.10550134542793914,
    0.62668250672205428,
    0.1363109543481357,
    0.090595671257557658,
    0.93212201984338217,
    0.091816676840817527,
    0.065804682083453414,
    0.77349681213167532,
    0.094966685329228195,
    0.046235492735370469,
    0.43544066126588366,
    0.14585995730471052,
    0.13016576971130089,
    0.53449533631624657,
    0.16736010700087711,
    0.11194695994990535,
    0.66297724883130826,
    0.13362584923010118,
    0.027664198156865393,
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
      if (weight < 0.02) {
        weight = 0.02;
      }
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

Quantizer::Quantizer(int template_id, int quant_xsize, int quant_ysize)
    : quant_xsize_(quant_xsize),
      quant_ysize_(quant_ysize),
      template_id_(template_id),
      global_scale_(kGlobalScaleDenom / kDefaultQuant),
      quant_patch_(kDefaultQuant),
      quant_dc_(kDefaultQuant),
      quant_img_ac_(quant_xsize_, quant_ysize_),
      initialized_(false) {
  FillImage(kDefaultQuant, &quant_img_ac_);
  if (template_id == kQuantHQ) {
    memcpy(zero_bias_, kZeroBiasHQ, sizeof(kZeroBiasHQ));
  } else {
    memcpy(zero_bias_, kZeroBiasDefault, sizeof(kZeroBiasDefault));
  }
}

const float* Quantizer::DequantMatrix() const {
  return pik::DequantMatrix(template_id_);
}

void Quantizer::GetQuantField(float* quant_dc, ImageF* qf) {
  const float scale = Scale();
  *quant_dc = scale * quant_dc_;
  *qf = ImageF(quant_xsize_, quant_ysize_);
  for (size_t y = 0; y < quant_ysize_; ++y) {
    for (size_t x = 0; x < quant_xsize_; ++x) {
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
  return Image3F(QuantizeRoundtrip(quantizer, 0, coeffs.Plane(0)),
                 QuantizeRoundtrip(quantizer, 1, coeffs.Plane(1)),
                 QuantizeRoundtrip(quantizer, 2, coeffs.Plane(2)));
}

Image3F QuantizeRoundtripExtract189(const Quantizer& quantizer,
                                    const Image3F& coeffs) {
  return Image3F(QuantizeRoundtripExtract189(quantizer, 0, coeffs.Plane(0)),
                 QuantizeRoundtripExtract189(quantizer, 1, coeffs.Plane(1)),
                 QuantizeRoundtripExtract189(quantizer, 2, coeffs.Plane(2)));
}

Image3F QuantizeRoundtripExtractDC(const Quantizer& quantizer,
                                   const Image3F& coeffs) {
  return Image3F(QuantizeRoundtripExtractDC(quantizer, 0, coeffs.Plane(0)),
                 QuantizeRoundtripExtractDC(quantizer, 1, coeffs.Plane(1)),
                 QuantizeRoundtripExtractDC(quantizer, 2, coeffs.Plane(2)));
}

Image3F QuantizeRoundtripDC(const Quantizer& quantizer, const Image3F& coeffs) {
  return Image3F(QuantizeRoundtripDC(quantizer, 0, coeffs.Plane(0)),
                 QuantizeRoundtripDC(quantizer, 1, coeffs.Plane(1)),
                 QuantizeRoundtripDC(quantizer, 2, coeffs.Plane(2)));
}

}  // namespace pik
