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

#ifndef PIK_ALPHA_H_
#define PIK_ALPHA_H_

#include <string>

#include "image.h"
#include "padded_bytes.h"
#include "pik_params.h"

namespace pik {

bool AlphaToPik(const CompressParams& params,
                const ImageU& plane, int bit_depth,
                size_t* bytepos, PaddedBytes* compressed);

bool PikToAlpha(const DecompressParams& params,
                size_t bytepos, const PaddedBytes& compressed,
                size_t* bytes_read, int* bit_depth, ImageU* plane);

}  // namespace pik

#endif  // PIK_ALPHA_H_
