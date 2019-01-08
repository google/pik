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

#ifndef ALPHA_H_
#define ALPHA_H_

#include "headers.h"
#include "image.h"
#include "pik_params.h"
#include "status.h"

namespace pik {

Status EncodeAlpha(const CompressParams& params, const ImageU& plane,
                   const Rect& rect, int bit_depth, Alpha* alpha);

// "plane" must be pre-allocated (FileHeader knows the size).
Status DecodeAlpha(const DecompressParams& params, const Alpha& alpha,
                   ImageU* plane, const Rect& rect);

}  // namespace pik

#endif  // ALPHA_H_
