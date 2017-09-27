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

#ifndef SIMD_SIMD_TEST_TARGET_H_
#define SIMD_SIMD_TEST_TARGET_H_

namespace simd {

// Call via ForeachTarget(SimdTest()). Prints a message to stderr and exits
// immediately after the first test failure. Otherwise, ORs Target::value
// (e.g. SIMD_SSE4) into "targets" to show which instruction set(s) were used.
struct SimdTest {
  template <class Target>
  void operator()();

  int targets = 0;
};

}  // namespace simd

#endif  // SIMD_SIMD_TEST_TARGET_H_
