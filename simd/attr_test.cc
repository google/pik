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

// Demo of using intrinsics without -mavx2 compile flags.

#include <stdio.h>
#include <stdlib.h>  // abort
#include <numeric>   // iota
#include <random>

// Which of the intrinsics referenced by simd.h to use (default: all).
// This can be limited to e.g. SIMD_SSE4 for compilers that don't support AVX2.
#define SIMD_ENABLE ~0u

#define SIMD_USE_ATTR 1

// SIMD_TARGET_ATTR prerequisites on Clang/GCC: __has_attribute(target) does not
// guarantee intrinsics are usable, hence we must check for specific versions.
#ifdef _MSC_VER
#define SIMD_ATTR_TEST_ENABLED
#elif defined(__clang__)
// Apple 8.2 == Clang 3.9
#ifdef __apple_build_version__
#if __clang_major__ > 8 || (__clang_major__ == 8 && __clang_minor__ >= 2)
#define SIMD_ATTR_TEST_ENABLED
#endif
// llvm.org Clang 3.9
#elif __clang_major__ > 3 || (__clang_major__ == 3 && __clang_minor__ >= 9)
#define SIMD_ATTR_TEST_ENABLED
#endif
// GCC 4.9
#elif defined(__GNUC__) && \
    (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 9))
#define SIMD_ATTR_TEST_ENABLED
#endif

// GCC/Clang require certain code generation flags when using intrinsics.
// This file demonstrates the SIMD_ATTR approach for working around this
// requirement. It requires a recent GCC/Clang, or MSVC, which does not need
// extra flags in the first place.
#ifdef SIMD_ATTR_TEST_ENABLED

#include "simd/dispatch.h"
#include "simd/simd.h"
// Must include "normally" so Blaze knows about the dependency.
#include "simd/attr_test_impl.h"

namespace pik {
namespace {

// Demo of targeting specific instruction sets and only calling them if
// supported.

struct AttrTest {
  template <class Target>
  void operator()();  // Implemented in attr_test_impl.h.
};
#define SIMD_ATTR_IMPL "attr_test_impl.h"
#include "foreach_target.h"

void RunTests() {
  dispatch::ForeachTarget(dispatch::SupportedTargets(), AttrTest());
}

}  // namespace
}  // namespace pik

#endif  // SIMD_ATTR_TEST_ENABLED

int main() {
#ifdef SIMD_ATTR_TEST_ENABLED
  pik::RunTests();
  return 0;
#else
  printf("Compiler does not support SIMD_TARGET_ATTR => demo is disabled.\n");
  return 0;
#endif
}
