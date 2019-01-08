// Copyright 2018 Google Inc. All Rights Reserved.
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

#ifndef ADAPTIVE_RECONSTRUCTION_H_
#define ADAPTIVE_RECONSTRUCTION_H_

#include "adaptive_reconstruction_fwd.h"
#include "epf.h"
#include "image.h"
#include "multipass_handler.h"
#include "quantizer.h"

namespace pik {

// Edge-preserving smoothing plus clamping the result to the quantized interval
// (which requires `quantizer` and `biases` to reconstruct the values that
// were actually quantized). `in` is the image to filter:  opsin AFTER gaborish.
// `non_smoothed` is BEFORE gaborish.
Image3F AdaptiveReconstruction(Image3F* in, const Image3F& non_smoothed,
                               const Quantizer& quantizer,
                               const ImageI& raw_quant_field,
                               const AcStrategyImage& ac_strategy,
                               const Image3F& biases, const EpfParams& params,
                               AdaptiveReconstructionAux* aux = nullptr);

}  // namespace pik

#endif  // ADAPTIVE_RECONSTRUCTION_H_
