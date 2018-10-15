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

static constexpr float kScale = 255.0f;

// NOTE: inverse of this cannot be constant because we tune these values.
static const float kOpsinAbsorbanceMatrix[9] = {
  static_cast<float>(( 0.29758629764381878 ) / kScale),
  static_cast<float>(( 0.63479886329551205 ) / kScale),
  static_cast<float>(( 0.088129079251512379 ) / kScale),
  static_cast<float>(( 0.22671198744507498 ) / kScale),
  static_cast<float>(( 0.6936230820580469 ) / kScale),
  static_cast<float>(( 0.098933489737625696 ) / kScale),
  static_cast<float>(( 0.19161912544122028 ) / kScale),
  static_cast<float>(( 0.082898111024512638 ) / kScale),
  static_cast<float>(( 0.53811869403330603 ) / kScale),
};

// Returns 3x3 row-major matrix inverse of kOpsinAbsorbanceMatrix.
// opsin_image_test verifies this is actually the inverse.
const float* GetOpsinAbsorbanceInverseMatrix();

static const float kOpsinAbsorbanceBias[3] = {
  static_cast<float>(( 0.22909219828963151 ) / kScale),
  static_cast<float>(( 0.22754880685562834 ) / kScale),
  static_cast<float>(( 0.1426315625332778 ) / kScale),
};
SIMD_ALIGN static const float kNegOpsinAbsorbanceBiasRGB[4] = {
    -kOpsinAbsorbanceBias[0], -kOpsinAbsorbanceBias[1],
    -kOpsinAbsorbanceBias[2], 255.0f};

static const float kScaleR = 1.0f;
static const float kScaleG = 2.0f - kScaleR;
static const float kInvScaleR = 1.0f / kScaleR;
static const float kInvScaleG = 1.0f / kScaleG;

// kXybCenter[3] is used by opsin_inverse.cc.
static constexpr float kXybCenter[4] = {
  0.007458806037902832f, 0.55163240432739258f, 0.50789362192153931f, 257.0f};

// This is the radius of the range, not the diameter.
static constexpr float kXybRange[3] = {
  0.021414279937744141, 0.45524927973747253f, 0.42550033330917358f};

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
