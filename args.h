// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik_params.h"  // Override

#include <stdio.h>
#include <string>

#include "status.h"

namespace pik {

static inline bool ParseOverride(const int argc, char* argv[], int* i,
                                 Override* out) {
  *i += 1;
  if (*i >= argc) {
    fprintf(stderr, "Expected an override argument.\n");
    return PIK_FAILURE("Args");
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
  return PIK_FAILURE("Args");
}

static inline bool ParseUnsigned(const int argc, char* argv[], int* i,
                                 size_t* out) {
  *i += 1;
  if (*i >= argc) {
    fprintf(stderr, "Expected an unsigned integer argument.\n");
    return PIK_FAILURE("Args");
  }

  char* end;
  *out = static_cast<size_t>(strtoull(argv[*i], &end, 0));
  if (end[0] != '\0') {
    fprintf(stderr, "Unable to interpret as unsigned integer: %s.\n", argv[*i]);
    return PIK_FAILURE("Args");
  }
  return true;
}

static inline bool ParseFloat(const int argc, char* argv[], int* i,
                              float* out) {
  *i += 1;
  if (*i >= argc) {
    fprintf(stderr, "Expected a floating-point argument.\n");
    return PIK_FAILURE("Args");
  }

  char* end;
  *out = static_cast<float>(strtod(argv[*i], &end));
  if (end[0] != '\0') {
    fprintf(stderr, "Unable to interpret as double: %s.\n", argv[*i]);
    return PIK_FAILURE("Args");
  }
  return true;
}

static inline bool ParseString(const int argc, char* argv[], int* i,
                               std::string* out) {
  *i += 1;
  if (*i >= argc) {
    fprintf(stderr, "Expected a string argument.\n");
    return PIK_FAILURE("Args");
  }

  out->assign(argv[*i]);
  return true;
}

}  // namespace pik
