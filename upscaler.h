// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef UPSCALER_H_
#define UPSCALER_H_

#include "image.h"

namespace pik {

Image3F Blur(const Image3F& image, float sigma);

}  // namespace pik

#endif  // UPSCALER_H_
