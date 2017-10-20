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

#ifndef GAUSS_BLUR_H_
#define GAUSS_BLUR_H_

#include <stddef.h>
#include <vector>

#include "image.h"

namespace pik {

template <typename T>
std::vector<T> GaussianKernel(int radius, T sigma) {
  std::vector<T> kernel(2 * radius + 1);
  const T scaler = -1.0 / (2 * sigma * sigma);
  double sum = 0.0;
  for (int i = -radius; i <= radius; ++i) {
    const T val = std::exp(scaler * i * i);
    kernel[i + radius] = val;
    sum += val;
  }
  for (int i = 0; i < kernel.size(); ++i) {
    kernel[i] /= sum;
  }
  return kernel;
}

// All convolution functions below apply mirroring of the input on the borders
// in the following way:
//
//     input: [a0 a1 a2 ...  aN]
//     mirrored input: [aR ... a1 | a0 a1 a2 .... aN | aN-1 ... aN-R]
//
// where R is the radius of the kernel (i.e. kernel size is 2*R+1).

ImageF Convolve(const ImageF& in, const std::vector<float>& kernel);
Image3F Convolve(const Image3F& in, const std::vector<float>& kernel);

ImageF Convolve(const ImageF& in,
                const std::vector<float>& kernel_x,
                const std::vector<float>& kernel_y);

Image3F Convolve(const Image3F& in,
                 const std::vector<float>& kernel_x,
                 const std::vector<float>& kernel_y);

// REQUIRES: in.xsize() and in.ysize() are integer multiples of res.
ImageF ConvolveAndSample(const ImageF& in,
                         const std::vector<float>& kernel,
                         const size_t res);

ImageF ConvolveAndSample(const ImageF& in,
                         const std::vector<float>& kernel_x,
                         const std::vector<float>& kernel_y,
                         const size_t res);

ImageF ConvolveXSampleAndTranspose(const ImageF& in,
                                   const std::vector<float>& kernel,
                                   const size_t res);

Image3F ConvolveXSampleAndTranspose(const Image3F& in,
                                    const std::vector<float>& kernel,
                                    const size_t res);

}  // namespace pik

#endif  // GAUSS_BLUR_H_
