// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

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

// Returns the opsin XYB for the part of the image bounded by rect.
Image3F OpsinDynamicsImage(const CodecInOut* in, const Rect& rect);

// DEPRECATED, used by opsin_image_wrapper.
Image3F OpsinDynamicsImage(const Image3B& srgb);

}  // namespace pik

#endif  // OPSIN_IMAGE_H_
