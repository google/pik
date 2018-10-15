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

// Facade for calling all enabled and supported SimdTest instantiations.

#include "simd/simd.h"
#include <stdio.h>

namespace pik {
namespace {

// Specialized in simd_test.cctest. Called via TargetBitfield. Thread-hostile.
struct SimdTest {
  template <class Target>
  void operator()();
};

// Called by simd_test.cctest functions.
void NotifyFailure(const int target, const int line, const char* vec_name,
                   const int lane, const char* expected, const char* actual) {
  fprintf(stderr,
          "target %x, line %d, %s lane %d mismatch: expected '%s', got '%s'.\n",
          target, line, vec_name, lane, expected, actual);
  SIMD_TRAP();
}

int RunTests() {
  TargetBitfield().Foreach(SimdTest());
  printf("Successfully tested instruction sets: 0x%x.\n", SIMD_ENABLE);
  return 0;
}

}  // namespace
}  // namespace pik

// Must include "normally" so the build system understands the dependency.
#include "simd/simd_test.cctest"

#define SIMD_ATTR_IMPL "simd_test.cctest"
#include "foreach_target.h"

int main() {
  setvbuf(stdin, nullptr, _IONBF, 0);
  return pik::RunTests();
}
