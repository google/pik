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

#include "compiler_specific.h"
#include "gauss_blur.h"
#include "status.h"

namespace pik {
namespace {

ImageF DiffPrecompute(const ImageF& xyb, float cutoff) {
  PIK_ASSERT(xyb.xsize() > 1);
  PIK_ASSERT(xyb.ysize() > 1);
  ImageF result(xyb.xsize(), xyb.ysize());
  static const double mul0 = 0.972407512222;
  for (size_t y = 0; y + 1 < xyb.ysize(); ++y) {
    const float* const PIK_RESTRICT row_in = xyb.Row(y);
    const float* const PIK_RESTRICT row_in2 = xyb.Row(y + 1);
    float* const PIK_RESTRICT row_out = result.Row(y);
    for (size_t x = 0; x + 1 < xyb.xsize(); ++x) {
      const size_t x2 = x + 1;
      const float diff = mul0 * (fabs(row_in[x] - row_in[x2]) +
                                 fabs(row_in[x] - row_in2[x]));
      row_out[x] = std::min(cutoff, 0.5f * diff);
    }
    // Last pixel of the row.
    {
      const size_t x = xyb.xsize() - 1;
      row_out[x] = mul0 * (fabs(row_in[x] - row_in2[x]));
    }
  }
  // Last row.
  {
    const size_t y = xyb.ysize() - 1;
    const float* const PIK_RESTRICT row_in = xyb.Row(y);
    float* const PIK_RESTRICT row_out = result.Row(y);
    for (size_t x = 0; x + 1 < xyb.xsize(); ++x) {
      const size_t x2 = x + 1;
      const float diff = mul0 * fabs(row_in[x] - row_in[x2]);
      row_out[x] = std::min(cutoff, 0.5f * diff);
    }
    // Last pixel of the last row.
    {
      const size_t x = xyb.xsize() - 1;
      row_out[x] = 0;
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
  static const float kBase = 0.081994280342603476;
  static const float kMul = 0.024979129332027221;
  static const float kOffset = 0.016770665190778019;
  ImageF out(diffs.xsize(), diffs.ysize());
  for (int y = 0; y < diffs.ysize(); ++y) {
    const float* const PIK_RESTRICT row_in = diffs.Row(y);
    float * const PIK_RESTRICT row_out = out.Row(y);
    for (int x = 0; x < diffs.xsize(); ++x) {
      row_out[x] = kBase + kMul / (row_in[x] + kOffset);
    }
  }
  return out;
}

ImageF SubsampleWithMax(const ImageF& in, int factor) {
  PIK_ASSERT(in.xsize() % factor == 0);
  PIK_ASSERT(in.ysize() % factor == 0);
  const size_t out_xsize = in.xsize() / factor;
  const size_t out_ysize = in.ysize() / factor;
  ImageF out(out_xsize, out_ysize);
  for (int oy = 0; oy < out_ysize; ++oy) {
    for (int ox = 0; ox < out_xsize; ++ox) {
      float maxval = 0.0f;
      for (int iy = 0; iy < factor; ++iy) {
        for (int ix = 0; ix < factor; ++ix) {
          const float val = in.Row(oy * factor + iy)[ox * factor + ix];
          maxval = std::max(maxval, val);
        }
      }
      out.Row(oy)[ox] = maxval;
    }
  }
  return out;
}

}  // namespace

ImageF AdaptiveQuantizationMap(const ImageF& img, size_t resolution) {
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
  static const float kSigma = 5.840439802203778;
  static const int kRadius = static_cast<int>(2 * kSigma + 0.5f);
  std::vector<float> kernel = GaussianKernel(kRadius, kSigma);
  static const float kDiffCutoff = 0.072750703471576167;
  ImageF out = DiffPrecompute(img, kDiffCutoff);
  out = Expand(out, resolution * out_xsize, resolution * out_ysize);
  out = ConvolveAndSample(out, kernel, kSampleRate);
  out = ComputeMask(out);
  if (resolution > kSampleRate) {
    out = SubsampleWithMax(out, resolution / kSampleRate);
  }
  return out;
}

}  // namespace pik
