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

#include "simd/simd.h"
#include <stdio.h>
#include "simd/dispatch.h"
#include "simd/simd_test_target.h"

namespace pik {
namespace {

void NotifyFailed(const int target, const int line, const char* vec_name,
                  const int lane, const char* expected, const char* actual) {
  fprintf(stderr,
          "target %x, line %d, %s lane %d mismatch: expected '%s', got '%s'.\n",
          target, line, vec_name, lane, expected, actual);
  SIMD_TRAP();
}

int RunTests() {
  SimdTest tests;
  const int targets = dispatch::SupportedTargets();
  dispatch::ForeachTarget(targets, tests, NotifyFailed);

  if (targets != tests.targets) {
    printf("Did not run all tests. Expected: %x; actual: %x\n", targets,
           tests.targets);
    return 1;
  }
  printf("Successfully tested instruction sets: 0x%x.\n", targets);
  return 0;
}

}  // namespace
}  // namespace pik

int main() {
  setvbuf(stdin, nullptr, _IONBF, 0);
  return pik::RunTests();
}
