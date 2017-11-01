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

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include "image.h"
#include "image_io.h"
#include "padded_bytes.h"
#include "pik.h"
#include "pik_info.h"
#include "simd/dispatch.h"

namespace pik {
namespace {

// main() function, within namespace for convenience.
int Compress(const char* pathname_in, const float butteraugli_distance,
             const char* pathname_out, const bool fast_mode) {
#if SIMD_ENABLE_AVX2
  if ((dispatch::SupportedTargets() & SIMD_AVX2) == 0) {
    fprintf(stderr, "Cannot continue because CPU lacks AVX2/FMA support.\n");
    return 1;
  }
#elif SIMD_ENABLE_SSE4
  if ((dispatch::SupportedTargets() & SIMD_SSE4) == 0) {
    fprintf(stderr, "Cannot continue because CPU lacks SSE4 support.\n");
    return 1;
  }
#endif

  MetaImageF in = ReadMetaImageLinear(pathname_in);
  if (in.xsize() == 0 || in.ysize() == 0) {
    fprintf(stderr, "Failed to open image %s.\n", pathname_in);
    return 1;
  }

  if (fast_mode) {
    printf("Compressing with fast mode\n");
  } else {
    printf("Compressing with maximum Butteraugli distance %f\n",
           butteraugli_distance);
  }

  CompressParams params;
  params.butteraugli_distance = butteraugli_distance;
  params.alpha_channel = in.HasAlpha();
  if (fast_mode) {
    params.fast_mode = true;
    params.butteraugli_distance = -1;
  }
  PaddedBytes compressed;
  PikInfo aux_out;
  if (!PixelsToPik(params, in, &compressed, &aux_out)) {
    fprintf(stderr, "Failed to compress.\n");
    return 1;
  }

  printf("Compressed to %zu bytes\n", compressed.size());

  FILE* f = fopen(pathname_out, "wb");
  if (f == nullptr) {
    fprintf(stderr, "Failed to open %s.\n", pathname_out);
    return 1;
  }
  const size_t bytes_written =
      fwrite(compressed.data(), 1, compressed.size(), f);
  if (bytes_written != compressed.size()) {
    fprintf(stderr, "I/O error, only wrote %zu bytes.\n", bytes_written);
    return 1;
  }
  fclose(f);
  return 0;
}

}  // namespace
}  // namespace pik

void PrintArgHelp(int argc, char** argv) {
  fprintf(stderr,
      "Usage: %s in.png out.pik [--distance <maxError>] [--fast]\n"
      " --distance: Maximum butteraugli distance, smaller value means higher"
      " quality.\n"
      "             Good default: 1.0. Supported range: 0.5 .. 3.0.\n"
      " --fast: Use fast encoding, ignores distance.\n"
      " --help: Show this help.\n",
      argv[0]);
}

void ExitWithArgError(int argc, char** argv) {
  PrintArgHelp(argc, argv);
  std::exit(1);
}

int main(int argc, char** argv) {
  bool fast_mode = false;
  const char* arg_maxError = nullptr;
  const char* arg_in = nullptr;
  const char* arg_out = nullptr;
  for (int i = 1; i < argc; i++) {
    if (argv[i][0] == '-') {
      std::string arg = argv[i];
      if (arg == "--fast") {
        fast_mode = true;
      } else if (arg == "--distance") {
        if (i + 1 >= argc) {
          printf("Must give a distance value\n");
          ExitWithArgError(argc, argv);
        }
        arg_maxError = argv[++i];
      } else if (arg == "--help") {
        PrintArgHelp(argc, argv);
        return 0;
      } else {
        ExitWithArgError(argc, argv);
      }
    } else {
      if (arg_in) {
        if (arg_out) ExitWithArgError(argc, argv);
        arg_out = argv[i];
      } else {
        arg_in = argv[i];
      }
    }
  }

  float butteraugli_distance = 1.0;
  if (arg_maxError) {
    butteraugli_distance = strtod(arg_maxError, nullptr);
    if (!(0.5f <= butteraugli_distance && butteraugli_distance <= 3.0f)) {
      fprintf(stderr, "Invalid/out of range distance '%s', try 0.5 to 3.\n",
              arg_maxError);
      return 1;
    }
  }

  if (!arg_in || !arg_out) {
    ExitWithArgError(argc, argv);
  }

  return pik::Compress(arg_in, butteraugli_distance, arg_out, fast_mode);
}
