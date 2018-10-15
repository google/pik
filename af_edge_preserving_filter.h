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

#ifndef AF_EDGE_PRESERVING_FILTER_H_
#define AF_EDGE_PRESERVING_FILTER_H_

#include "image.h"
#include "data_parallel.h"

namespace pik {

// Unit test. Call via dispatch::ForeachTarget.
struct EdgePreservingFilterTest {
  template <class Target>
  void operator()();
};

// Must be called before EdgePreservingFilter, with the same Target.
struct InitEdgePreservingFilter {
  template <class Target>
  void operator()();
};

// Adaptive smoothing based on quantization intervals. "sigma" must be in
// [kMinSigma, kMaxSigma]. Fills each pixel of "smoothed", which must be
// pre-allocated. Call via Dispatch.
struct EdgePreservingFilter {
  // The "sigma" parameter is the half-width at half-maximum, i.e. the SAD value
  // for which the weight is 0.5. It is about 1.2 times the standard deviation
  // of a normal distribution. Larger values cause more smoothing.

  // All sigma values are pre-shifted by this value to increase their
  // resolution. This allows adaptive sigma to compute "5.5" (represented as 22)
  // without an additional floating-point multiplication.
  static constexpr int kSigmaShift = 2;

  // This is the smallest value that avoids 16-bit overflow (see kShiftSAD); it
  // corresponds to 1/3 of patch pixels having the minimum integer SAD of 1.
  static constexpr int kMinSigma = 4 << kSigmaShift;
  // Somewhat arbitrary; determines size of a lookup table.
  static constexpr int kMaxSigma = 288 << kSigmaShift;  // 24 per patch pixel

  // For each block, adaptive sigma :=
  // sigma_mul * max_quantization_interval (i.e. reciprocal of min(ac_quant)).
  template <class Target>
  void operator()(const Image3F& in, const ImageI* ac_quant, float sigma_mul,
                  ThreadPool* pool, Image3F* smoothed);
};

}  // namespace pik

#endif  // AF_EDGE_PRESERVING_FILTER_H_
