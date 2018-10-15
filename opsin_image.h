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

#ifndef OPSIN_IMAGE_H_
#define OPSIN_IMAGE_H_

#include <stdint.h>
#include <cstdlib>
#include <vector>

#include "codec.h"
#include "compiler_specific.h"
#include "opsin_params.h"

namespace pik {

// r, g, b are linear.
static PIK_INLINE void OpsinAbsorbance(const float r, const float g,
                                       const float b, float out[3]) {
  const float* mix = &kOpsinAbsorbanceMatrix[0];
  const float* bias = &kOpsinAbsorbanceBias[0];
  out[0] = mix[0] * r + mix[1] * g + mix[2] * b + bias[0];
  out[1] = mix[3] * r + mix[4] * g + mix[5] * b + bias[1];
  out[2] = mix[6] * r + mix[7] * g + mix[8] * b + bias[2];
}

void LinearToXyb(const float r, const float g, const float b,
                 float* PIK_RESTRICT valx, float* PIK_RESTRICT valy,
                 float* PIK_RESTRICT valz);

// Returns the opsin XYB. Parallelized.
Image3F OpsinDynamicsImage(const CodecInOut* in);

// DEPRECATED, used by opsin_image_wrapper.
Image3F OpsinDynamicsImage(const Image3B& srgb);

}  // namespace pik

#endif  // OPSIN_IMAGE_H_
