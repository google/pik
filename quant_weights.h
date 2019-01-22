// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef QUANT_WEIGHTS_H_
#define QUANT_WEIGHTS_H_

namespace pik {

// kQuantWeights[N * N * c + N * y + x] is the relative weight of the (x, y)
// coefficient in component c. Higher weights correspond to finer quantization
// intervals and more bits spent in encoding.
const double* GetQuantWeightsDCT2();
const double* GetQuantWeightsDCT4(double* mul01, double* mul11);
const double* GetQuantWeightsDCT8();
const double* GetQuantWeightsDCT16();
const double* GetQuantWeightsDCT32();
const double* GetQuantWeightsIdentity();

}  // namespace pik

#endif  // QUANT_WEIGHTS_H_
