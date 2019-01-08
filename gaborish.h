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

#ifndef GABORISH_H_
#define GABORISH_H_

#include "data_parallel.h"
#include "image.h"

namespace pik {

// Additional smoothing helps for medium/low-quality.
enum class GaborishStrength : uint32_t {
  // Serialized, do not change enumerator values.
  kOff = 0,
  k500,
  k750,
  k875,
  k1000

  // Future extensions: [5, 6]
};

// Deprecated, use FastGaborishInverse instead.
Image3F SlowGaborishInverse(const Image3F& opsin, double mul);

// Returns "in" unchanged if strength == kOff (need rvalue to avoid copying).
// Approximate.
Image3F FastGaborishInverse(Image3F&& opsin, GaborishStrength strength,
                            ThreadPool* pool);

// Returns "in" unchanged if strength == kOff (need rvalue to avoid copying).
Image3F ConvolveGaborish(Image3F&& in, GaborishStrength strength,
                         ThreadPool* pool);

}  // namespace pik

#endif  // GABORISH_H_
