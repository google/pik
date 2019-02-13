// Copyright 2019 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef AR_CONTROL_FIELD_H_
#define AR_CONTROL_FIELD_H_

#include "ac_strategy.h"
#include "image.h"
#include "pik_params.h"

namespace pik {

void FindBestArControlField(float distance, const Image3F& opsin,
                            const AcStrategyImage& ac_strategy,
                            const ImageF& quant_field,
                            GaborishStrength gaborish, ImageB* sigma_lut_ids);

}

#endif  // AR_CONTROL_FIELD_H_
