// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

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
