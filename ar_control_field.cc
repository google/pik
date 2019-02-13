// Copyright 2019 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "ar_control_field.h"
#include "common.h"

namespace pik {

namespace {

// TODO(veluca): Remove. This is for demonstration only.
constexpr bool kTestARParams = false;

// This is temporary, to demonstrate effects of AR's sigma.
// TODO(veluca): Remove.
void FillImageCheckerboard(ImageB* image) {
  for (size_t y = 0; y < image->ysize(); y++) {
    auto* row = image->Row(y);
    for (size_t x = 0; x < image->xsize(); x++) {
      row[x] = (x + y) % 2;
    }
  }
}

}  // namespace

void FindBestArControlField(float distance, const Image3F& opsin,
                            const AcStrategyImage& ac_strategy,
                            const ImageF& quant_field,
                            GaborishStrength gaborish, ImageB* sigma_lut_ids) {
  size_t xsize_blocks = DivCeil(opsin.xsize(), kBlockDim);
  size_t ysize_blocks = DivCeil(opsin.ysize(), kBlockDim);
  *sigma_lut_ids = ImageB(xsize_blocks, ysize_blocks);
  if (kTestARParams) {
    FillImageCheckerboard(sigma_lut_ids);
  } else {
    ZeroFillImage(sigma_lut_ids);
  }
}

}  // namespace pik
