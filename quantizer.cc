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

// kQuantWeights[3 * k_zz + c] is the relative weight of the k_zz coefficient
// (in the zig-zag order) in component c. Higher weights correspond to finer
// quantization intervals and more bits spent in encoding.
const double *GetQuantWeights() {
  static double kQuantWeights[kNumQuantTables * 192] = {
    // Default weights
    4.17317809454045818,
    0.78610731495045172,
    0.37882522164694349,
    8.85240954685175474,
    0.77006744022797247,
    0.34592384547663707,
    6.31754060852936572,
    0.65117532248878540,
    0.12799474757657134,
    1.77624432097713192,
    0.57425978648632270,
    0.07506112044912261,
    4.15821627160268292,
    0.53552152065201664,
    0.27926363344859639,
    3.34913959033952580,
    0.50458211659794583,
    0.04361602490215615,
    2.28399714918438157,
    0.42933264930396459,
    0.03755467690066183,
    1.49450037455606899,
    0.37735050844972057,
    0.05956030004378896,
    2.39757305639016716,
    0.38481842768033758,
    0.05831348097451651,
    1.12653252382559543,
    0.27950664559970084,
    0.04382792405506560,
    1.21844604734873840,
    0.27471002210345935,
    0.04666653179163888,
    1.53097794217538463,
    0.30106751892219696,
    0.06307726671068764,
    1.22200655849294804,
    0.33280821723596504,
    0.03941220015763471,
    1.17524697541422518,
    0.28638305663379554,
    0.03208522877711826,
    0.75127670491071263,
    0.31850859296657152,
    0.04486109540684686,
    0.77226565243518897,
    0.27748922010630672,
    0.04666344078529713,
    1.08735121501276844,
    0.27827093784126850,
    0.06336382689520687,
    1.43070752477721830,
    0.30536844765425886,
    0.06902888503033432,
    1.53272631030449369,
    0.29465822125474284,
    0.08152654336613070,
    1.18204966394055289,
    0.26817960183599288,
    0.05485040455888805,
    0.62656733575737089,
    0.23756652529604391,
    0.03756587495986300,
    1.10504462098241163,
    0.17220175263887857,
    0.04556256391383898,
    0.85614468706807267,
    0.22529898347062727,
    0.05236708614758520,
    1.30435943636811436,
    0.30941439681151084,
    0.04238731948096763,
    1.32889262754232940,
    0.28605664583271589,
    0.01713126997120093,
    1.49622670357677845,
    0.27429259374843751,
    0.01962416370801040,
    1.04717555087763192,
    0.22620065718631438,
    0.02537455797619659,
    0.51628278614639511,
    0.18239311116210799,
    0.05836337499537232,
    0.73765048702465486,
    0.24404670751376545,
    0.05632547478810192,
    0.68930392008555674,
    0.19824929843321470,
    0.06011182828535343,
    1.11544802314041691,
    0.28150674187369507,
    0.01872402421593500,
    1.12587494392587240,
    0.24557680817217800,
    0.05628977364980435,
    1.27610588177249551,
    0.24829994562860350,
    0.01568278529124948,
    0.27124628277771612,
    0.24184577922053010,
    0.01641154704224334,
    0.49960619272689105,
    0.14880294705145591,
    0.02120221186383324,
    0.49054627277348750,
    0.15820647192633827,
    0.04441491696411323,
    0.44268509077766055,
    0.13054815824946345,
    0.02853325147404644,
    0.72675696782743338,
    0.17959032230796884,
    0.03827182757786673,
    0.45353712375759447,
    0.20336962941402578,
    0.01944649630481815,
    0.49065874330351711,
    0.20258067046833647,
    0.02001011477293114,
    1.49730803384489475,
    0.21501134116920845,
    0.06141297175035317,
    0.69127686095520724,
    0.19551874107857661,
    0.01598200751864970,
    1.07606674044062833,
    0.15736561084498937,
    0.03185237044694969,
    0.78276072932847207,
    0.18546365337596354,
    0.01866152687570816,
    0.53435941821522404,
    0.22550722418265939,
    0.01710216656233201,
    0.64941563816847947,
    0.22261844590297536,
    0.02028596575405426,
    0.67208745940973336,
    0.22651935654167457,
    0.03213907062303008,
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

const float* DequantMatrix(int id) {
  static const float* const kDequantMatrix = NewDequantMatrices();
  return &kDequantMatrix[id * 192];
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

Quantizer::Quantizer(int template_id, int quant_xsize, int quant_ysize) :
    template_id_(template_id),
    quant_xsize_(quant_xsize),
    quant_ysize_(quant_ysize),
    global_scale_(kGlobalScaleDenom / kDefaultQuant),
    quant_patch_(kDefaultQuant),
    quant_dc_(kDefaultQuant),
    quant_img_ac_(quant_xsize_, quant_ysize_, kDefaultQuant),
    initialized_(false) {
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
        const Key key = QuantizerKey(x, y);
        for (int c = 0; c < 3; ++c) {
          if (bq_[c].Find(key) == nullptr) {
            const float* const PIK_RESTRICT qm = &quant_matrix[c * 64];
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
  PROFILER_FUNC;
  const size_t block_xsize = in.xsize() / 64;
  const size_t block_ysize = in.ysize();
  Image3S out(block_xsize * 64, block_ysize);
  for (int c = 0; c < 3; ++c) {
    for (size_t block_y = 0; block_y < block_ysize; ++block_y) {
      const float* PIK_RESTRICT row_in = in.PlaneRow(c, block_y);
      int16_t* PIK_RESTRICT row_out = out.PlaneRow(c, block_y);
      for (size_t block_x = 0; block_x < block_xsize; ++block_x) {
        const float* const PIK_RESTRICT block_in = &row_in[block_x * 64];
        int16_t* const PIK_RESTRICT block_out = &row_out[block_x * 64];
        quantizer.QuantizeBlock(block_x, block_y, c, block_in, block_out);
      }
    }
  }
  return out;
}

// TODO(janwas): input is 1x64 or DC-only?
Image3S QuantizeCoeffsDC(const Image3F& in, const Quantizer& quantizer) {
  const size_t block_xsize = in.xsize() / 64;
  const size_t block_ysize = in.ysize();
  Image3S out(block_xsize * 64, block_ysize);
  for (int c = 0; c < 3; ++c) {
    for (size_t block_y = 0; block_y < block_ysize; ++block_y) {
      const float* PIK_RESTRICT row_in = in.PlaneRow(c, block_y);
      int16_t* PIK_RESTRICT row_out = out.PlaneRow(c, block_y);
      for (size_t block_x = 0; block_x < block_xsize; ++block_x) {
        const float* const PIK_RESTRICT block_in = &row_in[block_x * 64];
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
  const size_t block_xsize = img.xsize() / 64;
  const size_t block_ysize = img.ysize();
  const float inv_quant_dc = quantizer.inv_quant_dc();
  const float* const PIK_RESTRICT kDequantMatrix =
      &quantizer.DequantMatrix()[c * 64];
  ImageF out(img.xsize(), img.ysize());
  for (size_t block_y = 0; block_y < block_ysize; ++block_y) {
    const float* const PIK_RESTRICT row_in = img.ConstRow(block_y);
    float* const PIK_RESTRICT row_out = out.Row(block_y);
    for (size_t block_x = 0; block_x < block_xsize; ++block_x) {
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

ImageF QuantizeRoundtripDC(const Quantizer& quantizer, int c,
                           const ImageF& img) {
  // All coordinates are blocks.
  const int block_xsize = img.xsize() / 64;
  const int block_ysize = img.ysize();
  ImageF out(block_xsize, block_ysize);

  const float mul =
      quantizer.DequantMatrix()[c * 64] * quantizer.inv_quant_dc();
  for (size_t block_y = 0; block_y < block_ysize; ++block_y) {
    const float* const PIK_RESTRICT row_in = img.ConstRow(block_y);
    float* const PIK_RESTRICT row_out = out.Row(block_y);
    for (size_t block_x = 0; block_x < block_xsize; ++block_x) {
      const float* const PIK_RESTRICT block_in = &row_in[block_x * 64];
      row_out[block_x] =
          quantizer.QuantizeBlockDC(block_x, block_y, c, block_in) * mul;
    }
  }
  return out;
}

Image3F QuantizeRoundtrip(const Quantizer& quantizer, const Image3F& img) {
  return Image3F(QuantizeRoundtrip(quantizer, 0, img.plane(0)),
                 QuantizeRoundtrip(quantizer, 1, img.plane(1)),
                 QuantizeRoundtrip(quantizer, 2, img.plane(2)));
}

Image3F QuantizeRoundtripDC(const Quantizer& quantizer, const Image3F& img) {
  return Image3F(QuantizeRoundtripDC(quantizer, 0, img.plane(0)),
                 QuantizeRoundtripDC(quantizer, 1, img.plane(1)),
                 QuantizeRoundtripDC(quantizer, 2, img.plane(2)));
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
  const float* PIK_RESTRICT kDequantMatrix = quantizer.DequantMatrix();

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

PIK_INLINE void DequantizeCoeffsDC(const ConstImageViewF* in,
                                   const Quantizer& quantizer,
                                   const OutputRegion& output_region,
                                   const MutableImageViewF* PIK_RESTRICT out) {
  PROFILER_ZONE("|| DequantDC");
  const float* PIK_RESTRICT kDequantMatrix = quantizer.DequantMatrix();

  for (int c = 0; c < 3; ++c) {
    const float mul_dc = kDequantMatrix[c * 64] * quantizer.inv_quant_dc();

    for (size_t y = 0; y < output_region.ysize; ++y) {
      const int16_t* PIK_RESTRICT row_in =
          reinterpret_cast<const int16_t*>(in[c].ConstRow(y));
      float* PIK_RESTRICT row_out = out[c].Row(y);

      for (size_t x = 0; x < output_region.xsize; ++x) {
        row_out[x] = row_in[x * 64] * mul_dc;
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

TFNode* AddDequantizeDC(const TFPorts in_xyb, const Quantizer& quantizer,
                        TFBuilder* builder) {
  PIK_CHECK(OutType(in_xyb.node) == TFType::kI16);
  return builder->AddClosure(
      "dequantize", Borders(), Scale(), {in_xyb}, 3, TFType::kF32,
      [&quantizer](const ConstImageViewF* in, const OutputRegion& output_region,
                   const MutableImageViewF* PIK_RESTRICT out) {
        DequantizeCoeffsDC(in, quantizer, output_region, out);
      });
}

}  // namespace pik
