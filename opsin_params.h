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
  static_cast<float>(( 0.35234116891864925 ) / kScale),
  static_cast<float>(( 0.60459467598946537 ) / kScale),
  static_cast<float>(( 0.049149438641537209 ) / kScale),
  static_cast<float>(( 0.26769891168703097 ) / kScale),
  static_cast<float>(( 0.67145340133966724 ) / kScale),
  static_cast<float>(( 0.05956824321898456 ) / kScale),
  static_cast<float>(( 0.22807675797825966 ) / kScale),
  static_cast<float>(( 0.17339220265200772 ) / kScale),
  static_cast<float>(( 0.7409094491811542 ) / kScale),
};

// Returns 3x3 row-major matrix inverse of kOpsinAbsorbanceMatrix.
// opsin_image_test verifies this is actually the inverse.
const float* GetOpsinAbsorbanceInverseMatrix();

static const float kOpsinAbsorbanceBias[3] = {
  static_cast<float>(( 0.070609766961235526 ) / kScale),
  static_cast<float>(( 0.05175618760644736 ) / kScale),
  static_cast<float>(( 0.31813395053838894 ) / kScale),
};
SIMD_ALIGN static const float kNegOpsinAbsorbanceBiasRGB[4] = {
    -kOpsinAbsorbanceBias[0], -kOpsinAbsorbanceBias[1],
    -kOpsinAbsorbanceBias[2], 255.0f};

static const float kScaleR = 1.0019387926273635;
static const float kScaleG = 2.0f - kScaleR;
static const float kInvScaleR = 1.0f / kScaleR;
static const float kInvScaleG = 1.0f / kScaleG;

// kXybCenter[3] is used by opsin_inverse.cc.
static constexpr float kXybCenter[4] = {
  0.0087467581033706665f,
  0.53143048286437988f,
  0.576701819896698f,
  257.0f
};

// This is the radius of the range, not the diameter.
static constexpr float kXybRange[3] = {
  0.023478344082832336f,
  0.46945071220397949f,
  0.46904951333999634f
};
static constexpr float kXybMin[3] = {
  kXybCenter[0] - kXybRange[0],
  kXybCenter[1] - kXybRange[1],
  kXybCenter[2] - kXybRange[2],
};
static constexpr float kXybMax[3] = {
  kXybCenter[0] + kXybRange[0],
  kXybCenter[1] + kXybRange[1],
  kXybCenter[2] + kXybRange[2],
};

}  // namespace pik

#endif  // OPSIN_PARAMS_H_
