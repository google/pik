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

#ifndef OPSIN_INVERSE_H_
#define OPSIN_INVERSE_H_

#include "data_parallel.h"
#include "image.h"
#include "simd/simd.h"

namespace pik {

// Converts to linear sRGB. "linear" may alias "opsin".
// Prefer to replace this parallelize-across-image version with the one below,
// which is suitable for parallelize-across-group.
SIMD_ATTR void OpsinToLinear(const Image3F& opsin, ThreadPool* pool,
                             Image3F* linear);

// Converts to linear sRGB, writing to linear:rect_out.
SIMD_ATTR void OpsinToLinear(const Image3F& opsin, const Rect& rect_out,
                             Image3F* PIK_RESTRICT linear);

}  // namespace pik

#endif  // OPSIN_INVERSE_H_
