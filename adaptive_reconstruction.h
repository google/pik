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

#include <functional>
#include "compressed_image.h"
#include "data_parallel.h"
#include "image.h"
#include "intra_transform.h"
#include "quantizer.h"

namespace pik {

// Edge-preserving smoothing plus clamping the result to the quantized interval.
// "cmap" is required to predict the actual quantized values.
Image3F AdaptiveReconstruction(const Quantizer& quantizer,
                               const ColorCorrelationMap& cmap,
                               ThreadPool* pool, const Image3F& in,
                               const ImageB& ac_strategy,
                               OpsinIntraTransform* transform);

}  // namespace pik

#endif  // ADAPTIVE_RECONSTRUCTION_H_
