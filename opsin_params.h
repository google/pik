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
  static_cast<float>(( 0.35234118750506838 ) / kScale),
  static_cast<float>(( 0.60679459831402982 ) / kScale),
  static_cast<float>(( 0.049149439090790822 ) / kScale),
  static_cast<float>(( 0.26592763785090434 ) / kScale),
  static_cast<float>(( 0.67145306491162171 ) / kScale),
  static_cast<float>(( 0.059568235515668762 ) / kScale),
  static_cast<float>(( 0.22812514051951721 ) / kScale),
  static_cast<float>(( 0.17339209811025164 ) / kScale),
  static_cast<float>(( 0.74180913572881457 ) / kScale),
};

// Returns 3x3 row-major matrix inverse of kOpsinAbsorbanceMatrix.
// opsin_image_test verifies this is actually the inverse.
const float* GetOpsinAbsorbanceInverseMatrix();

static const float kOpsinAbsorbanceBias[3] = {
  static_cast<float>(( 0.070170335767859776 ) / kScale),
  static_cast<float>(( 0.050655423476540916 ) / kScale),
  static_cast<float>(( 0.31820877994104918 ) / kScale),
};
SIMD_ALIGN static const float kNegOpsinAbsorbanceBiasRGB[4] = {
    -kOpsinAbsorbanceBias[0], -kOpsinAbsorbanceBias[1],
    -kOpsinAbsorbanceBias[2], 255.0f};

static const float kScaleR = 1.0017389218858195;
static const float kScaleG = 2.0f - kScaleR;
static const float kInvScaleR = 1.0f / kScaleR;
static const float kInvScaleG = 1.0f / kScaleG;

// kXybCenter[3] is used by opsin_inverse.cc.
static constexpr float kXybCenter[4] = {
  0.0091904997825622559f, 0.5313260555267334f, 0.5768505334854126f, 257.0f};

// This is the radius of the range, not the diameter.
static constexpr float kXybRange[3] = {
  0.023611396551132202f, 0.46962425112724304f, 0.46918979287147522f};

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
