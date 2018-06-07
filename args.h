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

#include "pik_params.h"  // Override

#include <stdio.h>
#include <string>

namespace pik {

static inline bool ParseOverride(const int argc, char* argv[], int* i,
                                 Override* out) {
  *i += 1;
  if (*i >= argc) {
    fprintf(stderr, "Expected an override argument.\n");
    return false;
  }

  const std::string arg(argv[*i]);
  if (arg == "1") {
    *out = Override::kOn;
    return true;
  }
  if (arg == "0") {
    *out = Override::kOff;
    return true;
  }
  fprintf(stderr, "Invalid flag, must be 0 or 1\n");
  return false;
}

static inline bool ParseUnsigned(const int argc, char* argv[], int* i,
                                 size_t* out) {
  *i += 1;
  if (*i >= argc) {
    fprintf(stderr, "Expected an unsigned integer argument.\n");
    return false;
  }

  char* end;
  *out = static_cast<size_t>(strtoull(argv[*i], &end, 0));
  if (end[0] != '\0') {
    fprintf(stderr, "Unable to interpret as unsigned integer: %s.\n", argv[*i]);
    return false;
  }
  return true;
}

static inline bool ParseFloat(const int argc, char* argv[], int* i,
                              float* out) {
  *i += 1;
  if (*i >= argc) {
    fprintf(stderr, "Expected a floating-point argument.\n");
    return false;
  }

  char* end;
  *out = static_cast<float>(strtod(argv[*i], &end));
  if (end[0] != '\0') {
    fprintf(stderr, "Unable to interpret as double: %s.\n", argv[*i]);
    return false;
  }
  return true;
}

}  // namespace pik
