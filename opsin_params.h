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

namespace pik {

static constexpr float kScale = 255.0;

static constexpr float kOpsinAbsorbanceMatrix[9] = {
    0.355028246972028f / kScale, 0.589422218034148f / kScale,
    0.055549534993826f / kScale, 0.250871605395556f / kScale,
    0.714937756329137f / kScale, 0.034190638275308f / kScale,
    0.091915449087840f / kScale, 0.165250230906774f / kScale,
    0.742834320005384f / kScale,
};
static constexpr float kOpsinAbsorbanceInverseMatrix[9] = {
    6.805644286129f * kScale,  -5.552270790544f * kScale,
    -0.253373707795f * kScale, -2.373074275591f * kScale,
    3.349796660147f * kScale,  0.023277709773f * kScale,
    -0.314192274838f * kScale, -0.058176067042f * kScale,
    1.372368367449f * kScale,
};

static constexpr float kScaleR = 1.001746913108605f;
static constexpr float kScaleG = 2.0f - kScaleR;
static constexpr float kInvScaleR = 1.0f / kScaleR;
static constexpr float kInvScaleG = 1.0f / kScaleG;

static constexpr float kXybCenter[3] = {0.008714601398f, 0.5f, 0.5f};
// This is the radius of the range, not the diameter.
static constexpr float kXybRange[3] = {0.035065606236f, 0.5f, 0.5f};
static constexpr float kXybMin[3] = {
    kXybCenter[0] - kXybRange[0], kXybCenter[1] - kXybRange[1],
    kXybCenter[2] - kXybRange[2],
};
static constexpr float kXybMax[3] = {
    kXybCenter[0] + kXybRange[0], kXybCenter[1] + kXybRange[1],
    kXybCenter[2] + kXybRange[2],
};

// sRGB color space initial cut off, initial slope, offset
// and power values as specified in IEC 61966-2-1:1999.
static constexpr float kGammaInitialCutoff = 10.31475f;
static constexpr float kGammaInitialSlope = 12.92f;
static constexpr float kGammaOffset = 0.055f;
static constexpr float kGammaPower = 2.4f;

}  // namespace pik

#endif  // OPSIN_PARAMS_H_
