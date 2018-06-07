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

// Library to decode jpeg coefficients into an RGB image.

#ifndef JPEG_DATA_DECODER_H_
#define JPEG_DATA_DECODER_H_

#include <stdint.h>
#include <vector>

#include "guetzli/jpeg_data.h"

namespace pik {

// Decodes the parsed jpeg coefficients into an RGB image.
// There can be only either 1 or 3 image components, in either case, an RGB
// output image will be generated.
// The size of the rgb array must be at least 3 * jpg.width * jpg.height.
// If 'thumb' is true, decode into a 8x-reduced thumbnail (ignoring AC
// coefficients in 'jpg'). The rgb[] array must be at least of size
// 3 * ((jpg.width + 7) / 8) * ((jpg.height + 7) / 8) bytes.
bool DecodeJpegToRGB(const guetzli::JPEGData& jpg, uint8_t* rgb,
                     bool thumbnail);

// Same as above, but returning a properly allocated vector of rgb samples.
// Vector will be empty if a decoding error occurred.
std::vector<uint8_t> DecodeJpegToRGB(const guetzli::JPEGData& jpg,
                                     bool thumbnail);

}  // namespace pik

#endif  // JPEG_DATA_DECODER_H_
