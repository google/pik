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

namespace pik {

// Additional smoothing helps for medium/low-quality.
enum class GaborishStrength : uint32_t {
  // Serialized, do not change enumerator values.
  kOff = 0,
  k500,
  k750,
  k875,
  k1000

  // Future extensions: [5, 6]
};

// Used in encoder to reduce the impact of the decoder's smoothing.
// This is approximate and slow (unoptimized 5x5 convolution).
Image3F GaborishInverse(const Image3F& opsin, double mul);

// Returns "in" unchanged if strength == kOff (need rvalue to avoid copying).
Image3F ConvolveGaborish(Image3F&& in, GaborishStrength strength,
                         ThreadPool* pool);

}  // namespace pik

#endif  // GABORISH_H_
