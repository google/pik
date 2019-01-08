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

#ifndef YUV_CONVERT_H_
#define YUV_CONVERT_H_

#include "image.h"

namespace pik {

Image3B RGB8ImageFromYUVRec709(const Image3U& yuv, int bit_depth);
Image3U RGB16ImageFromYUVRec709(const Image3U& yuv, int bit_depth);
Image3F RGBLinearImageFromYUVRec709(const Image3U& yuv, int bit_depth);

Image3U YUVRec709ImageFromRGB8(const Image3B& rgb, int out_bit_depth);
Image3U YUVRec709ImageFromRGB16(const Image3U& rgb, int out_bit_depth);
Image3U YUVRec709ImageFromRGBLinear(const Image3F& rgb, int out_bit_depth);

void SubSampleChroma(const Image3U& yuv, int bit_depth, ImageU* yplane,
                     ImageU* uplane, ImageU* vplane);

Image3U SuperSampleChroma(const ImageU& yplane, const ImageU& uplane,
                          const ImageU& vplane, int bit_depth);

}  // namespace pik

#endif  // YUV_CONVERT_H_
