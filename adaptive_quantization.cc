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

#include "adaptive_quantization.h"

#include <string.h>
#include <algorithm>
#include <cmath>
#include <vector>

#ifdef ROI_DETECTOR_OPENCV
#include "roi_detector_opencv.h"
#endif

#undef PROFILER_ENABLED
#define PROFILER_ENABLED 1
#include "approx_cube_root.h"
#include "common.h"
#include "compiler_specific.h"
#include "gauss_blur.h"
#include "profiler.h"
#include "status.h"
#include "dct.h"
#include "entropy_coder.h"

namespace pik {
namespace {

static const float kQuant64[64] = {
  0.0,
  0.7,
  0.7,
  0.7,
  0.7,
  0.588994181958768,
  0.637412731903807,
  0.658265403732354,
  0.664184545211087,
  0.489557940162741,
  0.441809196703547,
  0.466519636930879,
  0.508335971631046,
  0.501624976604618,
  0.447903166596090,
  0.487108568611555,
  0.457498177649116,
  0.452848682971061,
  0.490734792977206,
  0.443057415317534,
  0.392141167751584,
  0.358316869050728,
  0.418207067977596,
  0.467858769794496,
  0.443549876397044,
  0.427476997048851,
  0.356915568577037,
  0.303161982522844,
  0.363256217129902,
  0.352611929358839,
  0.418999909922867,
  0.398691116916629,
  0.377526872102006,
  0.352326109479318,
  0.308206968728254,
  0.280093699425257,
  0.304147527622546,
  0.359202623799680,
  0.339645164723791,
  0.349585237941005,
  0.386203903613304,
  0.353582036654603,
  0.305955548639682,
  0.365259530060446,
  0.333159510814334,
  0.363133568767434,
  0.334161790012618,
  0.389194124900511,
  0.349326306148990,
  0.390310895605386,
  0.408666924454222,
  0.335930464190049,
  0.359313000261458,
  0.381109877480420,
  0.392933763109596,
  0.359529015172913,
  0.347676628893596,
  0.370974565818013,
  0.350361463992334,
  0.338064798002449,
  0.336743523710490,
  0.296631529585931,
  0.304517245589665,
  0.302956514467806,
};

// Increase precision in 8x8 blocks that are complicated in DCT space.
SIMD_ATTR void DctModulation(const ImageF& xyb, ImageF* out) {
  PIK_ASSERT((xyb.xsize() + 7) / 8 == out->xsize());
  PIK_ASSERT((xyb.ysize() + 7) / 8 == out->ysize());
  const int32_t* natural_coeff_order = NaturalCoeffOrder<8>();
  float dct_rescale[64] = { 0 };
  {
    const float* dct_scale = DCTScales<8>();
    for (int i = 0; i < 64; ++i) {
      dct_rescale[i] = dct_scale[i / 8] * dct_scale[i % 8];
    }
  }
  for (int y = 0; y < xyb.ysize(); y += 8) {
    float* const PIK_RESTRICT row_out = out->Row(y / 8);
    for (int x = 0; x < xyb.xsize(); x += 8) {
      SIMD_ALIGN float dct[64] = { 0 };
      for (int dy = 0; dy < 8 && y + dy < xyb.ysize(); ++dy) {
        const float* const PIK_RESTRICT row_in = xyb.Row(y + dy);
        for (int dx = 0; dx < 8 && x + dx < xyb.xsize(); ++dx) {
          dct[dy * 8 + dx] = row_in[x + dx];
        }
      }
      ComputeTransposedScaledDCT<8>(FromBlock<8>(dct), ToBlock<8>(dct));
      double entropyQL2 = 0;
      double entropyQL4 = 0;
      double entropyQL8 = 0;
      for (int k = 1; k < 64; ++k) {
        int i = natural_coeff_order[k];
        const float scale = dct_rescale[i];
        double v = dct[i] * scale;
        v *= v;
        entropyQL2 += kQuant64[k] * v;
        v *= v;
        entropyQL4 += kQuant64[k] * v;
        v *= v;
        entropyQL8 += kQuant64[k] * v;
      }
      entropyQL2 = std::sqrt(entropyQL2);
      entropyQL4 = std::sqrt(std::sqrt(entropyQL4));
      entropyQL8 = std::pow(entropyQL8, 0.125);
      static const double mulQL2 = 5.1517782096782501;
      static const double mulQL4 = -13.190159830831263;
      static const double mulQL8 = 2.128400445818913;
      double v =
          mulQL2 * entropyQL2 + mulQL4 * entropyQL4 + mulQL8 * entropyQL8;
      if (v < 0.7) { v = 0.7; }
      double kMul = -0.86029879634002482;
      row_out[x / 8] *= exp(kMul * v);
    }
  }
}

void ChessboardModulation(ImageF *out) {
  static const double kChessMul[4] = {
    0.93821798758260999,
    0.94169713626559881,
    0.89448558048418603,
    0.90271064414242019,
  };
  for (int y = 0; y < out->ysize(); ++y) {
    float* const PIK_RESTRICT row = out->Row(y);
    for (int x = 0; x < out->xsize(); ++x) {
      int index = (y & 1) * 2 + (x & 1);
      row[x] *= kChessMul[index];
    }
  }
}

void BorderModulation(ImageF *out) {
  static const float kCornerMul = 0.60403817140618821;
  static const float kBorderMul = 1.014011300252267;
  static const int first_column = 0;
  const int last_column = out->xsize() - 1;
  {
    float* const PIK_RESTRICT first_row = out->Row(0);
    float* const PIK_RESTRICT last_row = out->Row(out->ysize() - 1);
    for (int x = 1; x + 1 < out->xsize(); ++x) {
      first_row[x] *= kBorderMul;
      last_row[x] *= kBorderMul;
    }
    first_row[first_column] *= kCornerMul;
    first_row[last_column] *= kCornerMul;
    last_row[first_column] *= kCornerMul;
    last_row[last_column] *= kCornerMul;
  }
  {
    for (int y = 1; y + 1 < out->ysize(); ++y) {
      float* const PIK_RESTRICT scan_row = out->Row(y);
      scan_row[first_column] *= kBorderMul;
      scan_row[last_column] *= kBorderMul;
    }
  }
}

// Increase precision in 8x8 blocks that have high dynamic range.
void RangeModulation(const ImageF& xyb, ImageF *out) {
  PIK_ASSERT((xyb.xsize() + 7) / 8 == out->xsize());
  PIK_ASSERT((xyb.ysize() + 7) / 8 == out->ysize());
  for (int y = 0; y < xyb.ysize(); y += 8) {
    float* const PIK_RESTRICT row_out = out->Row(y / 8);
    for (int x = 0; x < xyb.xsize(); x += 8) {
      float minval = 1e30;
      float maxval = -1e30;
      for (int dy = 0; dy < 8 && y + dy < xyb.ysize(); ++dy) {
        const float* const PIK_RESTRICT row_in = xyb.Row(y + dy);
        for (int dx = 0; dx < 8 && x + dx < xyb.xsize(); ++dx) {
          float v = row_in[x + dx];
          if (minval > v) {
            minval = v;
          }
          if (maxval < v) {
            maxval = v;
          }
        }
      }
      float range = maxval - minval;
      static const double mul = -0.20531261192730726;
      row_out[x / 8] *= exp(mul * range);
    }
  }
}

// Change precision in 8x8 blocks that have high frequency content.
void HfModulation(const ImageF& xyb, ImageF *out) {
  PIK_ASSERT((xyb.xsize() + 7) / 8 == out->xsize());
  PIK_ASSERT((xyb.ysize() + 7) / 8 == out->ysize());
  for (int y = 0; y < xyb.ysize(); y += 8) {
    float* const PIK_RESTRICT row_out = out->Row(y / 8);
    for (int x = 0; x < xyb.xsize(); x += 8) {
      float sum = 0;
      int n = 0;
      for (int dy = 0; dy < 8 && y + dy < xyb.ysize(); ++dy) {
        const float* const PIK_RESTRICT row_in = xyb.Row(y + dy);
        for (int dx = 0; dx < 7 && x + dx + 1 < xyb.xsize(); ++dx) {
          float v = fabs(row_in[x + dx] - row_in[x + dx + 1]);
          sum += v;
          ++n;
        }
      }
      for (int dy = 0; dy < 7 && y + dy + 1 < xyb.ysize(); ++dy) {
        const float* const PIK_RESTRICT row_in = xyb.Row(y + dy);
        const float* const PIK_RESTRICT row_in_next = xyb.Row(y + dy + 1);
        for (int dx = 0; dx < 8 && x + dx < xyb.xsize(); ++dx) {
          float v = fabs(row_in[x + dx] - row_in_next[x + dx]);
          sum += v;
          ++n;
        }
      }
      if (n != 0) {
        sum /= n;
      }
      static const double kMul = -0.98595041351593005;
      sum *= kMul;
      row_out[x / 8] *= exp(sum);
    }
  }
}

namespace {
// Change precision in 8x8 blocks that intersect regions-of-interest.
void ROIModulation(const ImageF& image_y,
                   const CompressParams &cparams,
                   ImageF *out) {
#ifdef ROI_DETECTOR_OPENCV
  if (cparams.roi_factor == 0.0f) {
    return;
  }
  const size_t out_xsize = out->xsize();
  const size_t out_ysize = out->ysize();
  // These are the plane's reported x- and y-size.
  // Since rows are padded, we also need to know bytes-per-row.
  const size_t xsize = image_y.xsize();
  const size_t ysize = image_y.ysize();
  const auto img_bytes = image_y.bytes();
  const size_t bytes_per_row = image_y.bytes_per_row();
  HaarDetector roi_detector(
      {"haarcascade_frontalface_default.xml",
       "haarcascade_profileface.xml",
       "haarcascade_eye.xml",
      });
  const auto& rois = roi_detector.Detect(
      xsize, ysize, bytes_per_row, 2, (void*)(img_bytes));
  // Using the ROIs.
  auto roi_blocks = reinterpret_cast<unsigned char*>(
      // TODO(user): Eliminate calloc() in favor of using ImageB.
      calloc(out_xsize * out_ysize, sizeof(unsigned char)));
  if (roi_blocks == nullptr) {
    fprintf(stderr, "Memory allocation failure, not using ROI-Modulation.");
    // Make sure this gets reported straightaway even if stderr was set
    // to buffer.
    fflush(stderr);
    return;
  }
  int num_roi_blocks = 0;
  for (const auto& roi: rois) {
    const size_t ny_max = std::min(
        out->ysize() - 1,
        DivCeil<size_t>(roi.ypos + roi.height, kBlockDim));
    const size_t nx_max = std::min(
        out->xsize() - 1,
        DivCeil<size_t>(roi.xpos + roi.width, kBlockDim));
    for (int ny = roi.ypos / kBlockDim; ny <= ny_max; ++ny) {
      for (int nx = roi.xpos / kBlockDim; nx <= nx_max; ++nx) {
        roi_blocks[ny * out_xsize + nx] = 1;
        num_roi_blocks += 1;
      }
    }
  }
  for (int ny = 0; ny < out_ysize; ++ny) {
    float* const PIK_RESTRICT row_out = out->Row(ny);
    for (int nx = 0; nx < out_xsize; ++nx) {
      // For regions-of-interest, we actually have to multiply with something
      // quite large.
      double v = cparams.roi_factor * roi_blocks[ny * out_xsize + nx];
      double kMul = -12.507848292533591;  // random guess
      row_out[nx] *= exp(kMul * v);
    }
  }
  printf("Fine-Tuned %d ROI-blocks (%.1f%%).\n",
         num_roi_blocks, (100.0 * num_roi_blocks) / (out_xsize * out_ysize));
  fflush(stdout);
  free(roi_blocks);
#endif
}
}  // namespace

static double SimpleGamma(double v) {
  // A simple HDR compatible gamma function.
  // mul and mul2 represent a scaling difference between pik and butteraugli.
  static const double mul = 105.3039258005453;
  static const double mul2 = 1.0 / (79.565021396458519);

  v *= mul;

  static const double kRetMul = mul2 * 18.6580932135;
  static const double kRetAdd = mul2 * -20.2789020414;
  static const double kVOffset = 7.14672470003;

  if (v < 0) {
    // This should happen rarely, but may lead to a NaN, which is rather
    // undesirable. Since negative photons don't exist we solve the NaNs by
    // clamping here.
    v = 0;
  }
  return kRetMul * log(v + kVOffset) + kRetAdd;
}

static double RatioOfCubicRootToSimpleGamma(double v) {
  // The opsin space in pik is the cubic root of photons, i.e., v * v * v
  // is related to the number of photons.
  //
  // SimpleGamma(v * v * v) is the psychovisual space in butteraugli.
  // This ratio allows quantization to move from pik's opsin space to
  // butteraugli's log-gamma space.
  return v / SimpleGamma(v * v * v);
}

ImageF DiffPrecompute(const ImageF& xyb, float cutoff) {
  PROFILER_ZONE("aq DiffPrecompute");
  PIK_ASSERT(xyb.xsize() > 1);
  PIK_ASSERT(xyb.ysize() > 1);
  ImageF result(xyb.xsize(), xyb.ysize());
  static const double mul0 = 0.021323897159710007;

  // PIK's gamma is 3.0 to be able to decode faster with two muls.
  // Butteraugli's gamma is matching the gamma of human eye, around 2.6.
  // We approximate the gamma difference by adding one cubic root into
  // the adaptive quantization. This gives us a total gamma of 2.6666
  // for quantization uses.
  static const double match_gamma_offset = 0.33206231924251833;
  static const float kOverWeightBorders = 1.3;
  size_t x1, y1;
  size_t x2, y2;
  for (size_t y = 0; y + 1 < xyb.ysize(); ++y) {
    if (y + 1 < xyb.ysize()) {
      y2 = y + 1;
    } else if (y > 0) {
      y2 = y - 1;
    } else {
      y2 = y;
    }
    if (y == 0 && xyb.ysize() >= 2) {
      y1 = y + 1;
    } else if (y > 0) {
      y1 = y - 1;
    } else {
      y1 = y;
    }
    const float* PIK_RESTRICT row_in = xyb.Row(y);
    const float* PIK_RESTRICT row_in1 = xyb.Row(y1);
    const float* PIK_RESTRICT row_in2 = xyb.Row(y2);
    float* const PIK_RESTRICT row_out = result.Row(y);
    for (size_t x = 0; x + 1 < xyb.xsize(); ++x) {
      if (x + 1 < xyb.xsize()) {
        x2 = x + 1;
      } else if (x > 0) {
        x2 = x - 1;
      } else {
        x2 = x;
      }
      if (x == 0 && xyb.xsize() >= 2) {
        x1 = x + 1;
      } else if (x > 0) {
        x1 = x - 1;
      } else {
        x1 = x;
      }
      float diff = mul0 *
          (fabs(row_in[x] - row_in[x2]) +
           fabs(row_in[x] - row_in2[x]) +
           fabs(row_in[x] - row_in[x1]) +
           fabs(row_in[x] - row_in1[x]) +
           3 * (fabs(row_in2[x] - row_in1[x]) +
                fabs(row_in[x1] - row_in[x2])));
      diff *= RatioOfCubicRootToSimpleGamma(row_in[x] + match_gamma_offset);
      row_out[x] = std::min(cutoff, diff);
    }
    // Last pixel of the row.
    {
      const size_t x = xyb.xsize() - 1;
      float diff =
          kOverWeightBorders * 2.0 * mul0 * (fabs(row_in[x] - row_in2[x]));
      diff *= RatioOfCubicRootToSimpleGamma(row_in[x] + match_gamma_offset);
      row_out[x] = std::min(cutoff, diff);
    }
  }
  // Last row.
  {
    const size_t y = xyb.ysize() - 1;
    const float* const PIK_RESTRICT row_in = xyb.Row(y);
    float* const PIK_RESTRICT row_out = result.Row(y);
    for (size_t x = 0; x + 1 < xyb.xsize(); ++x) {
      const size_t x2 = x + 1;
      float diff =
          kOverWeightBorders * 2.0 * mul0 * fabs(row_in[x] - row_in[x2]);
      diff *= RatioOfCubicRootToSimpleGamma(row_in[x] + match_gamma_offset);
      row_out[x] = std::min(cutoff, diff);
    }
    // Last pixel of the last row.
    {
      const size_t x = xyb.xsize() - 1;
      row_out[x] = row_out[x - 1];
    }
  }
  return result;
}

ImageF Expand(const ImageF& img, size_t out_xsize, size_t out_ysize) {
  PIK_ASSERT(img.xsize() > 0);
  PIK_ASSERT(img.ysize() > 0);
  ImageF out(out_xsize, out_ysize);
  for (size_t y = 0; y < out_ysize; ++y) {
    const float* const PIK_RESTRICT row_in =
        y < img.ysize() ? img.Row(y) : out.Row(y - 1);
    float* const PIK_RESTRICT row_out = out.Row(y);
    memcpy(row_out, row_in, img.xsize() * sizeof(row_out[0]));
    const float lastval = row_in[img.xsize() - 1];
    for (size_t x = img.xsize(); x < out_xsize; ++x) {
      row_out[x] = lastval;
    }
  }
  return out;
}

ImageF ComputeMask(const ImageF& diffs) {
  static const float kBase = 0.57325257052582801;
  static const float kMul1 = 0.0079468128384258541;
  static const float kOffset1 = 0.0050013629254370999;
  static const float kMul2 = -0.075232946798815287;
  static const float kOffset2 = 0.22433469484556165;
  ImageF out(diffs.xsize(), diffs.ysize());
  for (int y = 0; y < diffs.ysize(); ++y) {
    const float* const PIK_RESTRICT row_in = diffs.Row(y);
    float * const PIK_RESTRICT row_out = out.Row(y);
    for (int x = 0; x < diffs.xsize(); ++x) {
      const float val = row_in[x];
      // Avoid division by zero.
      double div = std::max<double>(val + kOffset1, 1e-3);
      row_out[x] = kBase +
          kMul1 / div +
          kMul2 / (val * val + kOffset2);
    }
  }
  return out;
}

ImageF SubsampleWithMax(const ImageF& in, int factor) {
  PROFILER_ZONE("aq Subsample");
  PIK_ASSERT(in.xsize() % factor == 0);
  PIK_ASSERT(in.ysize() % factor == 0);
  const size_t out_xsize = in.xsize() / factor;
  const size_t out_ysize = in.ysize() / factor;
  ImageF out(out_xsize, out_ysize);
  for (size_t oy = 0; oy < out_ysize; ++oy) {
    float* PIK_RESTRICT row_out = out.Row(oy);
    for (size_t ox = 0; ox < out_xsize; ++ox) {
      float maxval = 0.0f;
      for (int iy = 0; iy < factor; ++iy) {
        const float* PIK_RESTRICT row_in = in.Row(oy * factor + iy);
        for (int ix = 0; ix < factor; ++ix) {
          const float val = row_in[ox * factor + ix];
          maxval = std::max(maxval, val);
        }
      }
      row_out[ox] = maxval;
    }
  }
  return out;
}

}  // namespace

ImageF AdaptiveQuantizationMap(
    const ImageF& img,
    const ImageF& img_ac,
    const CompressParams &cparams,
    size_t resolution) {
  PROFILER_ZONE("aq AdaptiveQuantMap");
  static const int kSampleRate = 8;
  PIK_ASSERT(resolution % kSampleRate == 0);
  const size_t out_xsize = (img.xsize() + resolution - 1) / resolution;
  const size_t out_ysize = (img.ysize() + resolution - 1) / resolution;
  if (img.xsize() <= 1) {
    ImageF out(1, out_ysize);
    FillImage(1.0f, &out);
    return out;
  }
  if (img.ysize() <= 1) {
    ImageF out(out_xsize, 1);
    FillImage(1.0f, &out);
    return out;
  }
  static const float kSigma0 = 12.180081590321119;
  static const float kWeight0 = 0.85074460070908942;
  static const float kSigma1 = 2.6389153225470978;
  static const float kWeight1 = 1.0 - kWeight0;
  static const int kRadius = static_cast<int>(2 * kSigma0 + 0.5f);
  std::vector<float> kernel0 = GaussianKernel(kRadius, kSigma0);
  std::vector<float> kernel1 = GaussianKernel(kRadius, kSigma1);
  std::vector<float> kernel(kernel0.size());
  for (int i = 0; i < kernel0.size(); ++i) {
    kernel[i] = kWeight0 * kernel0[i] + kWeight1 * kernel1[i];
  }
  static const float kDiffCutoff = 0.10929896077297953;
  ImageF out = DiffPrecompute(img, kDiffCutoff);
  out = Expand(out, resolution * out_xsize, resolution * out_ysize);
  out = ConvolveAndSample(out, kernel, kSampleRate);
  out = ComputeMask(out);
  if (resolution > kSampleRate) {
    out = SubsampleWithMax(out, resolution / kSampleRate);
  }
  BorderModulation(&out);
  ChessboardModulation(&out);
  DctModulation(img_ac, &out);
  RangeModulation(img_ac, &out);
  HfModulation(img_ac, &out);
  // TODO(user): Flag-guard. Only want to do this in slow-mode.
  // Note that this operates on 'img'.
  ROIModulation(img, cparams, &out);
  return out;
}

}  // namespace pik
