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

#ifndef QUANT_WEIGHTS_H_
#define QUANT_WEIGHTS_H_

namespace pik {

// kQuantWeights[N * N * c + N * y + x] is the relative weight of the (x, y)
// coefficient in component c. Higher weights correspond to finer quantization
// intervals and more bits spent in encoding.
const double* GetQuantWeightsDCT4(double* mul01, double* mul11);
const double* GetQuantWeightsDCT8();
const double* GetQuantWeightsDCT16();
const double* GetQuantWeightsDCT32();
const double* GetQuantWeightsIdentity();

}  // namespace pik

#endif  // QUANT_WEIGHTS_H_
