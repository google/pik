// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef GABORISH_H_
#define GABORISH_H_

// Linear smoothing (3x3 convolution) for deblocking without too much blur.

#include "data_parallel.h"
#include "image.h"
#include "pik_params.h"

namespace pik {

// Used in encoder to reduce the impact of the decoder's smoothing.
// This is approximate and slow (unoptimized 5x5 convolution).
Image3F GaborishInverse(const Image3F& opsin, double mul);

// Does not accept strength of GaborishStrength::kOff. For those cases it's
// cheaper and simpler to just not do the convolve.
Status ConvolveGaborish(const Image3F& in, GaborishStrength strength,
                        ThreadPool* pool, Image3F* PIK_RESTRICT out);

}  // namespace pik

#endif  // GABORISH_H_
