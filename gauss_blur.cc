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

#include "gauss_blur.h"

#include <math.h>
#include <algorithm>

#undef PROFILER_ENABLED
#define PROFILER_ENABLED 1
#include "compiler_specific.h"
#include "profiler.h"

namespace pik {

inline void ExtrapolateBorders(const float* const PIK_RESTRICT row_in,
                               float* const PIK_RESTRICT row_out,
                               const int xsize, const int radius) {
  const int lastcol = xsize - 1;
  for (int x = 1; x <= radius; ++x) {
    row_out[-x] = row_in[std::min(x, xsize - 1)];
  }
  memcpy(row_out, row_in, xsize * sizeof(row_out[0]));
  for (int x = 1; x <= radius; ++x) {
    row_out[lastcol + x] = row_in[std::max(0, lastcol - x)];
  }
}

ImageF ConvolveXSampleAndTranspose(const ImageF& in,
                                   const std::vector<float>& kernel,
                                   const size_t res) {
  PIK_ASSERT(kernel.size() % 2 == 1);
  PIK_ASSERT(in.xsize() % res == 0);
  const int offset = res / 2;
  const int out_xsize = in.xsize() / res;
  ImageF out(in.ysize(), out_xsize);
  const int r = kernel.size() / 2;
  std::vector<float> row_tmp(in.xsize() + 2 * r);
  float* const PIK_RESTRICT rowp = &row_tmp[r];
  const float* const kernelp = &kernel[r];
  for (int y = 0; y < in.ysize(); ++y) {
    ExtrapolateBorders(in.Row(y), rowp, in.xsize(), r);
    for (int x = offset, ox = 0; x < in.xsize(); x += res, ++ox) {
      float sum = 0.0f;
      for (int i = -r; i <= r; ++i) {
        sum += rowp[x + i] * kernelp[i];
      }
      out.Row(ox)[y] = sum;
    }
  }
  return out;
}

Image3F ConvolveXSampleAndTranspose(const Image3F& in,
                                    const std::vector<float>& kernel,
                                    const size_t res) {
  return Image3F(ConvolveXSampleAndTranspose(in.Plane(0), kernel, res),
                 ConvolveXSampleAndTranspose(in.Plane(1), kernel, res),
                 ConvolveXSampleAndTranspose(in.Plane(2), kernel, res));
}

ImageF ConvolveAndSample(const ImageF& in, const std::vector<float>& kernel_x,
                         const std::vector<float>& kernel_y, const size_t res) {
  ImageF tmp = ConvolveXSampleAndTranspose(in, kernel_x, res);
  return ConvolveXSampleAndTranspose(tmp, kernel_y, res);
}

ImageF Convolve(const ImageF& in, const std::vector<float>& kernel_x,
                const std::vector<float>& kernel_y) {
  return ConvolveAndSample(in, kernel_x, kernel_y, 1);
}

Image3F Convolve(const Image3F& in, const std::vector<float>& kernel_x,
                 const std::vector<float>& kernel_y) {
  return Image3F(Convolve(in.Plane(0), kernel_x, kernel_y),
                 Convolve(in.Plane(1), kernel_x, kernel_y),
                 Convolve(in.Plane(2), kernel_x, kernel_y));
}

ImageF ConvolveAndSample(const ImageF& in, const std::vector<float>& kernel,
                         const size_t res) {
  return ConvolveAndSample(in, kernel, kernel, res);
}

ImageF Convolve(const ImageF& in, const std::vector<float>& kernel) {
  return ConvolveAndSample(in, kernel, 1);
}

Image3F Convolve(const Image3F& in, const std::vector<float>& kernel) {
  return Convolve(in, kernel, kernel);
}

}  // namespace pik
