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

#ifndef DC_PREDICTOR_H_
#define DC_PREDICTOR_H_

// DC coefficients serve as an image preview, so they are coded separately.
// Subtracting predicted values leads to a "residual" distribution with lower
// entropy and magnitudes than the original values. These can be coded more
// efficiently, even when context modeling is used.
//
// Our predictors use immediately adjacent causal pixels because more distant
// pixels are only weakly correlated in subsampled DC images. We also utilize
// cross-channel correlation by choosing a predictor based upon its performance
// on a previously decoded channel.
//
// This module decreases final size of DC images by 2-4% vs. the standard
// MED/MAP predictor from JPEG-LS and processes 330 M coefficients per second.
// The average residual is about 1.3% of the maximum DC value.

#include <stdint.h>

#include "compiler_specific.h"
#include "image.h"

namespace pik {

// Prefer to avoid floating-point arithmetic because results may (slightly)
// differ depending on the compiler and platform.
using DC = int16_t;

// Predicts "dc" coefficients from their neighbors and stores the resulting
// "residuals" into a preallocated image. The predictors are optimized for the
// luminance channel.
void ShrinkY(const Image<DC>& dc, Image<DC>* const PIK_RESTRICT residuals);

// Predicts "dc" coefficients from their neighbors and an already expanded
// "dc_y" (luminance). Processes pairs of chrominance coefficients (U, V).
void ShrinkUV(const Image<DC>& dc_y, const Image<DC>& dc,
              Image<DC>* const PIK_RESTRICT residuals);

// Reconstructs "dc" (previously passed to ShrinkY) using "residuals".
void ExpandY(const Image<DC>& residuals, Image<DC>* const PIK_RESTRICT dc);

// Reconstructs "dc" (previously passed to ShrinkUV) using "residuals" and
// "dc_y" (luminance).
void ExpandUV(const Image<DC>& dc_y, const Image<DC>& residuals,
              Image<DC>* const PIK_RESTRICT dc);

}  // namespace pik

#endif  // DC_PREDICTOR_H_
