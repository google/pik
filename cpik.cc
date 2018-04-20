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
#include "profiler.h"
#include "simd/dispatch.h"

namespace pik {
namespace {

int Compress(const char* pathname_in, const char* pathname_out,
             const CompressParams& params) {
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

  if (params.fast_mode) {
    fprintf(stderr, "Compressing with fast mode\n");
  } else if (params.target_size != 0) {
    fprintf(stderr, "Compressing to target size %zd\n", params.target_size);
  } else {
    fprintf(stderr, "Compressing with maximum Butteraugli distance %f\n",
            params.butteraugli_distance);
  }

  PaddedBytes compressed;
  PikInfo aux_out;
  if (!PixelsToPik(params, in, &compressed, &aux_out)) {
    fprintf(stderr, "Failed to compress.\n");
    return 1;
  }

  fprintf(stderr, "Compressed to %zu bytes\n", compressed.size());
  if (params.verbose) {
    aux_out.Print(1);
  }

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
  PROFILER_PRINT_RESULTS();
  return 0;
}

void PrintArgHelp(int argc, char** argv) {
  fprintf(stderr,
          "Usage: %s in.png out.pik [--distance <maxError>] [--fast] "
          "[--denoise <0,1>] [--noise <0,1>]\n"
          " --distance: Maximum butteraugli distance, lower = higher quality.\n"
          "             Good default: 1.0. Supported range: 0.5 .. 3.0.\n"
          " --fast: Use fast encoding, ignores distance.\n"
          " --denoise: force enable/disable edge-preserving smoothing.\n"
          " --noise: force enable/disable noise generation.\n"
          " --help: Show this help.\n",
          argv[0]);
}

PIK_NORETURN void ExitWithArgError(int argc, char** argv) {
  PrintArgHelp(argc, argv);
  std::exit(1);
}

Override GetOverride(const int argc, char** argv, int* i) {
  *i += 1;
  if (*i >= argc) {
    fprintf(stderr, "Expected a 0 or 1 flagnn\n");
    ExitWithArgError(argc, argv);
  }
  const std::string flag = argv[*i];
  if (flag == "1") {
    return Override::kOn;
  } else if (flag == "0") {
    return Override::kOff;
  } else {
    fprintf(stderr, "Invalid flag, must be 0 or 1\n");
    ExitWithArgError(argc, argv);
  }
}

int Run(int argc, char** argv) {
  CompressParams params;
  const char* arg_in = nullptr;
  const char* arg_out = nullptr;
  for (int i = 1; i < argc; i++) {
    if (argv[i][0] == '-') {
      const std::string arg = argv[i];
      if (arg == "--fast") {
        params.fast_mode = true;
      } else if (arg == "--denoise") {
        params.denoise = GetOverride(argc, argv, &i);
      } else if (arg == "--noise") {
        params.apply_noise = GetOverride(argc, argv, &i);
      } else if (arg == "-v") {
        params.verbose = true;
      } else if (arg == "--distance") {
        if (++i >= argc) {
          fprintf(stderr, "Must give a distance value\n");
          ExitWithArgError(argc, argv);
        }
        params.butteraugli_distance = strtod(argv[i], nullptr);
        if (!(0.5f <= params.butteraugli_distance &&
              params.butteraugli_distance <= 3.0f)) {
          fprintf(stderr, "Invalid/out of range distance '%s', try 0.5 to 3.\n",
                  argv[i]);
          return 1;
        }
      } else if (arg == "--target_size") {
        if (++i >= argc) {
          fprintf(stderr, "Must give a size value\n");
          ExitWithArgError(argc, argv);
        }
        params.target_size = strtoul(argv[i], nullptr, 0);
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

  if (params.target_size != 0 && params.butteraugli_distance != -1.0f) {
    fprintf(stderr,
            "Only one of --distance or --target_size can be specified.\n");
    ExitWithArgError(argc, argv);
  }

  if (!arg_in || !arg_out) {
    ExitWithArgError(argc, argv);
  }

  return Compress(arg_in, arg_out, params);
}

}  // namespace
}  // namespace pik

int main(int argc, char** argv) { return pik::Run(argc, argv); }
