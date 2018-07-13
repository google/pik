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

#include "dct.h"
#include <cmath>

#define PROFILER_ENABLED 1
#include "arch_specific.h"
#include "compiler_specific.h"
#include "profiler.h"

namespace pik {

void ComputeBlockDCTFloat(float block[kBlockSize]) {
  ComputeTransposedScaledBlockDCTFloat(block);
  TransposeBlock(block);
  for (size_t y = 0; y < kBlockHeight; ++y) {
    const float scale_y = static_cast<int>(kBlockSize) * kIDCTScales[y];
    for (size_t x = 0; x < kBlockWidth; ++x) {
      block[kBlockWidth * y + x] /= scale_y * kIDCTScales[x];
    }
  }
}

}  // namespace pik
