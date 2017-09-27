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

// Facade/dispatcher for calling all supported SimdTest instantiations.

#include "simd.h"
#include <stdio.h>
#include "dispatch.h"
#include "simd_test_target.h"

namespace simd {
namespace {

// Calls func.operator()<Target>(args) for all instruction sets in "targets".
template <class Func, typename... Args>
SIMD_INLINE void ForeachTarget(const int targets, Func&& func, Args&&... args) {
  if (targets & SIMD_SSE4) {
    std::forward<Func>(func).template operator()<SSE4>(
        std::forward<Args>(args)...);
  }

  if (targets & SIMD_AVX2) {
    std::forward<Func>(func).template operator()<AVX2>(
        std::forward<Args>(args)...);
  }

  std::forward<Func>(func).template operator()<None>(
      std::forward<Args>(args)...);
}

void RunTests() {
  simd::SimdTest tests;
  const int supported = dispatch::SupportedTargets();
  simd::ForeachTarget(supported, tests);

  printf("Supported: %x; ran: %x\n", supported, tests.targets);
}

}  // namespace
}  // namespace simd

int main() {
  setvbuf(stdin, nullptr, _IONBF, 0);
  simd::RunTests();
  return 0;
}
