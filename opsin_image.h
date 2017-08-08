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

#include "image.h"
#include "opsin_params.h"

namespace pik {

PIK_INLINE void OpsinAbsorbance(const float in[3], float out[3]) {
  const float* mix = &kOpsinAbsorbanceMatrix[0];
  out[0] = mix[0] * in[0] + mix[1] * in[1] + mix[2] * in[2];
  out[1] = mix[3] * in[0] + mix[4] * in[1] + mix[5] * in[2];
  out[2] = mix[6] * in[0] + mix[7] * in[1] + mix[8] * in[2];
}

// Returns the opsin dynamics image corresponding to the given SRGB input image.
Image3F OpsinDynamicsImage(const Image3B& srgb);

Image3F OpsinDynamicsImage(const Image3F& linear);

void RgbToXyb(uint8_t r, uint8_t g, uint8_t b, float *valx, float *valy,
              float *valz);

}  // namespace pik

#endif  // OPSIN_IMAGE_H_
