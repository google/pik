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

#undef PROFILER_ENABLED
#define PROFILER_ENABLED 1
#include "approx_cube_root.h"
#include "compiler_specific.h"
#include "gauss_blur.h"
#include "profiler.h"
#include "status.h"

namespace pik {
namespace {

// Increase precision in 8x8 blocks that have high dynamic range.
void RangeModulation(const ImageF& xyb, ImageF *out) {
  PIK_ASSERT((xyb.xsize() + 7) / 8 == out->xsize());
  PIK_ASSERT((xyb.ysize() + 7) / 8 == out->ysize());
  for (int y = 0; y < xyb.ysize(); y += 8) {
    for (int x = 0; x < xyb.xsize(); x += 8) {
      float minval = 1e30;
      float maxval = -1e30;
      float* const PIK_RESTRICT row_out = out->Row(y / 8);
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
      static const double mul = 0.040122664180523591;
      static const double one = 0.99758355332643378;
      row_out[x / 8] *= one + mul * range;
    }
  }
}

ImageF DiffPrecompute(const ImageF& xyb, float cutoff) {
  PROFILER_ZONE("aq DiffPrecompute");
  PIK_ASSERT(xyb.xsize() > 1);
  PIK_ASSERT(xyb.ysize() > 1);
  ImageF result(xyb.xsize(), xyb.ysize());
  static const double mul0 = 0.9392459389653951;

  // PIK's gamma is 3.0 to be able to decode faster with two muls.
  // Butteraugli's gamma is matching the gamma of human eye, around 2.6.
  // We approximate the gamma difference by adding one cubic root into
  // the adaptive quantization. This gives us a total gamma of 2.6666
  // for quantization uses.
  static const double match_gamma_offset1 = 0.050226938462922706;
  static const double match_gamma_offset2 = -0.1167664122528226;
  for (size_t y = 0; y + 1 < xyb.ysize(); ++y) {
    const float* const PIK_RESTRICT row_in = xyb.Row(y);
    const float* const PIK_RESTRICT row_in2 = xyb.Row(y + 1);
    float* const PIK_RESTRICT row_out = result.Row(y);
    for (size_t x = 0; x + 1 < xyb.xsize(); ++x) {
      const size_t x2 = x + 1;
      float diff = mul0 * (fabs(row_in[x] - row_in[x2]) +
                           fabs(row_in[x] - row_in2[x]));
      diff *= ApproxCubeRoot(row_in[x] + match_gamma_offset1) +
              match_gamma_offset2;
      row_out[x] = std::min(cutoff, diff);
    }
    // Last pixel of the row.
    {
      const size_t x = xyb.xsize() - 1;
      float diff = 2.0 * mul0 * (fabs(row_in[x] - row_in2[x]));
      diff *= ApproxCubeRoot(row_in[x] + match_gamma_offset1) +
              match_gamma_offset2;
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
      float diff = 2.0 * mul0 * fabs(row_in[x] - row_in[x2]);
      diff *= ApproxCubeRoot(row_in[x] + match_gamma_offset1) +
              match_gamma_offset2;
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
  static const float kBase = 0.38627361795438747;
  static const float kMul1 = 0.0091688688923859517;
  static const float kOffset1 = 0.015345498704630292;
  static const float kMul2 = -0.012913937295721125;
  static const float kOffset2 = 0.065919403536217766;
  ImageF out(diffs.xsize(), diffs.ysize());
  for (int y = 0; y < diffs.ysize(); ++y) {
    const float* const PIK_RESTRICT row_in = diffs.Row(y);
    float * const PIK_RESTRICT row_out = out.Row(y);
    for (int x = 0; x < diffs.xsize(); ++x) {
      const float val = row_in[x];
      row_out[x] = kBase +
          kMul1 / (val + kOffset1) +
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

ImageF AdaptiveQuantizationMap(const ImageF& img, size_t resolution) {
  PROFILER_ZONE("aq AdaptiveQuantMap");
  static const int kSampleRate = 8;
  PIK_ASSERT(resolution % kSampleRate == 0);
  const size_t out_xsize = (img.xsize() + resolution - 1) / resolution;
  const size_t out_ysize = (img.ysize() + resolution - 1) / resolution;
  if (img.xsize() <= 1) {
    return ImageF(1, out_ysize, 1.0f);
  }
  if (img.ysize() <= 1) {
    return ImageF(out_xsize, 1, 1.0f);
  }
  static const float kSigma = 5.9183890169814513;
  static const int kRadius = static_cast<int>(2 * kSigma + 0.5f);
  std::vector<float> kernel = GaussianKernel(kRadius, kSigma);
  static const float kDiffCutoff = 0.17176867501916163;
  ImageF out = DiffPrecompute(img, kDiffCutoff);
  out = Expand(out, resolution * out_xsize, resolution * out_ysize);
  out = ConvolveAndSample(out, kernel, kSampleRate);
  out = ComputeMask(out);
  if (resolution > kSampleRate) {
    out = SubsampleWithMax(out, resolution / kSampleRate);
  }
  RangeModulation(img, &out);
  return out;
}

}  // namespace pik
