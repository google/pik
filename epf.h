// Copyright 2018 Google Inc. All Rights Reserved.
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

#ifndef EPF_H_
#define EPF_H_

#include <stdio.h>
#include "ac_strategy.h"
#include "field_encodings.h"
#include "image.h"

namespace pik {

struct EpfParams {
  EpfParams();
  constexpr const char* Name() const { return "EpfParams"; }

  template <class Visitor>
  Status VisitFields(Visitor* PIK_RESTRICT visitor) {
    visitor->Bool(true, &enable_adaptive);
    if (visitor->Conditional(enable_adaptive)) {
      visitor->U32(kU32Direct3Plus4, 0, &lut);
    }
    if (visitor->Conditional(!enable_adaptive)) {
      visitor->U32(0x0A090880, 0, &sigma);
    }
    visitor->Bool(false, &use_sharpened);
    return true;
  }

  // If false, use hardcoded sigma for each block.
  bool enable_adaptive;

  // Only if enable_adaptive:
  uint32_t lut;  // Index of quant->sigma lookup table [0, 3]

  // Only if !enable_adaptive:
  uint32_t sigma;  // ignored if !enable_adaptive, otherwise >= kMinSigma.

  bool use_sharpened;
};

// Unit test. Call via dispatch::ForeachTarget.
struct EdgePreservingFilterTest {
  template <class Target>
  void operator()() const;

  // Returns weight given sigma and SAD.
  template <class Target>
  float operator()(int sigma, int sad) const;
};

// Must be called before EdgePreservingFilter, with the same Target.
struct InitEdgePreservingFilter {
  template <class Target>
  void operator()() const;
};

// Adaptive smoothing based on quantization intervals. "sigma" must be in
// [kMinSigma, kMaxSigma]. Fills each pixel of "smoothed", which must be
// pre-allocated. Call via Dispatch.
struct EdgePreservingFilter {
  // The "sigma" parameter is the SCALED half-width at half-maximum, i.e. the
  // SAD value for which the weight is 0.5, times the scaling factor of
  // 1 << kSigmaShift. Before scaling, sigma is about 1.2 times the standard
  // deviation of a normal distribution. Larger values cause more smoothing.

  // All sigma values are pre-shifted by this value to increase their
  // resolution. This allows adaptive sigma to compute "5.5" (represented as 22)
  // without an additional floating-point multiplication.
  static constexpr int kSigmaShift = 2;

  // This is the smallest value that avoids 16-bit overflow (see kShiftSAD); it
  // corresponds to 1/3 of patch pixels having the minimum integer SAD of 1.
  static constexpr int kMinSigma = 4 << kSigmaShift;
  // Somewhat arbitrary; determines size of a lookup table.
  static constexpr int kMaxSigma = 216 << kSigmaShift;  // 18 per patch pixel

  // For each block, compute adaptive sigma.
  template <class Target>
  void operator()(const Image3F& in_guide, const Image3F& in,
                  const ImageI* ac_quant, float quant_scale,
                  const AcStrategyImage& ac_strategy,
                  const EpfParams& epf_params, Image3F* smoothed,
                  EpfStats* epf_stats) const;

  // Fixed sigma in [kMinSigma, kMaxSigma] for generating training data;
  // sigma == 0 skips filtering and copies "in" to "smoothed".
  // "stretch" is returned for use by AdaptiveReconstructionAux.
  template <class Target>
  void operator()(const Image3F& in_guide, const Image3F& in,
                  const EpfParams& params, float* PIK_RESTRICT stretch,
                  Image3F* smoothed) const;
};

}  // namespace pik

#endif  // EPF_H_
