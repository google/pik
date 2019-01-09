// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

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
