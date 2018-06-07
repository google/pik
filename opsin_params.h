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

#ifndef OPSIN_PARAMS_H_
#define OPSIN_PARAMS_H_

#include <stdlib.h>

#include "simd/simd.h"  // SIMD_ALIGN

namespace pik {

static constexpr float kScale = 255.0;

// NOTE: inverse of this cannot be constant because we tune these values.
static const float kOpsinAbsorbanceMatrix[9] = {
  static_cast<float>(( 0.3523376466161795 ) / kScale),
  static_cast<float>(( 0.6067972574251409 ) / kScale),
  static_cast<float>(( 0.049151209535235267 ) / kScale),
  static_cast<float>(( 0.26592386506834575 ) / kScale),
  static_cast<float>(( 0.67140860580051065 ) / kScale),
  static_cast<float>(( 0.061225872443361362 ) / kScale),
  static_cast<float>(( 0.22812868140840609 ) / kScale),
  static_cast<float>(( 0.17338855722136276 ) / kScale),
  static_cast<float>(( 0.7418055948399257 ) / kScale),
};

// Returns 3x3 row-major matrix inverse of kOpsinAbsorbanceMatrix.
// opsin_image_test verifies this is actually the inverse.
const float* GetOpsinAbsorbanceInverseMatrix();

static const float kOpsinAbsorbanceBias[3] = {
  static_cast<float>(( 0.070410956432969879 ) / kScale),
  static_cast<float>(( 0.050981882587652023 ) / kScale),
  static_cast<float>(( 0.31617904901529575 ) / kScale),
};
SIMD_ALIGN static const float kNegOpsinAbsorbanceBiasRGB[4] = {
    -kOpsinAbsorbanceBias[0], -kOpsinAbsorbanceBias[1],
    -kOpsinAbsorbanceBias[2], 255.0f};

static const float kScaleR = 1.0017364448897901;
static const float kScaleG = 2.0f - kScaleR;
static const float kInvScaleR = 1.0f / kScaleR;
static const float kInvScaleG = 1.0f / kScaleG;

// kXybCenter[3] is used by opsin_inverse.cc.
static constexpr float kXybCenter[4] = {
    0.0090238451957702637f, 0.53151017427444458f, 0.57673406600952148f, 257.0f};

// This is the radius of the range, not the diameter.
static constexpr float kXybRange[3] = {
    0.023776501417160034f, 0.46970874071121216f, 0.46930274367332458f};
static constexpr float kXybMin[3] = {
    kXybCenter[0] - kXybRange[0], kXybCenter[1] - kXybRange[1],
    kXybCenter[2] - kXybRange[2],
};
static constexpr float kXybMax[3] = {
    kXybCenter[0] + kXybRange[0], kXybCenter[1] + kXybRange[1],
    kXybCenter[2] + kXybRange[2],
};

}  // namespace pik

#endif  // OPSIN_PARAMS_H_
