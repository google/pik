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

#ifndef AC_PREDICTIONS_H_
#define AC_PREDICTIONS_H_

#include "color_correlation.h"
#include "compressed_image_fwd.h"
#include "quantizer.h"
#include "simd/simd.h"

namespace pik {

// All the `acs_rect`s here define which area of the ac_strategy image should be
// used to obtain the strategy of the current block from, and are specified in
// block coordinates.

// Common utilities.
SIMD_ATTR void ComputeLlf(const Image3F& dc, const AcStrategyImage& ac_strategy,
                          const Rect& acs_rect, Image3F* PIK_RESTRICT llf);
SIMD_ATTR void PredictLf(const AcStrategyImage& ac_strategy,
                         const Rect& acs_rect, const Image3F& llf,
                         ImageF* tmp2x2, Image3F* lf2x2);

// Encoder API.
SIMD_ATTR void PredictLfForEncoder(bool predict_lf, bool predict_hf,
                                   const Image3F& dc,
                                   const AcStrategyImage& ac_strategy,
                                   const ColorCorrelationMap& cmap,
                                   const Quantizer& quantizer,
                                   Image3F* PIK_RESTRICT ac64, Image3F* dc2x2);
void ComputePredictionResiduals(const Image3F& pred2x2,
                                const AcStrategyImage& ac_strategy,
                                Image3F* PIK_RESTRICT coeffs);

// Decoder API. Encoder-decoder API is currently not symmetric. Ideally both
// should allow tile-wise processing.
SIMD_ATTR void UpdateLfForDecoder(const Rect& tile, bool predict_lf,
                                  bool predict_hf,
                                  const AcStrategyImage& ac_strategy,
                                  const Rect& acs_rect, const ImageF& llf_plane,
                                  ImageF* ac64_plane, ImageF* dc2x2_plane,
                                  ImageF* lf2x2_plane);

void AddPredictions(const Image3F& pred2x2, const AcStrategyImage& ac_strategy,
                    const Rect& acs_rect, Image3F* PIK_RESTRICT dcoeffs);

}  // namespace pik

#endif  // AC_PREDICTIONS_H_
