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

#ifndef ALPHA_BLEND_H_
#define ALPHA_BLEND_H_

#include "image.h"

namespace pik {

// img is in linear space, but blending happens in gamma-compressed space using
// (gamma-compressed) grayscale background color, alpha image represents
// weights of the sRGB colors in the [0 .. (1 << bit_depth) - 1] interval,
// output image is in linear space
ImageF AlphaBlend(const ImageF& img, const ImageU& alpha,
                  int bit_depth, uint8_t background);
Image3F AlphaBlend(const MetaImageF& img, uint8_t background);

}  // namespace pik

#endif  // ALPHA_BLEND_H_
