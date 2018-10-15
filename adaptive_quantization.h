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

#ifndef ADAPTIVE_QUANTIZATION_H_
#define ADAPTIVE_QUANTIZATION_H_

#include <stddef.h>

#include "image.h"
#include "pik_params.h"

namespace pik {

// Returns an image subsampled by |resolution| in each direction. If the value
// at pixel (x,y) in the returned image is greater than 1.0, it means that
// more fine-grained quantization should be used in the corresponding block
// of the input image, while a value less than 1.0 indicates that less
// fine-grained quantization should be enough. Typically, |resolution| is
// a linear block size, to match quantizer AC field size.
ImageF AdaptiveQuantizationMap(const ImageF& img,
                               const ImageF& img_ac,
                               const CompressParams& cparams,
                               size_t resolution);

}  // namespace pik

#endif  // ADAPTIVE_QUANTIZATION_H_
