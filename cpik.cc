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

#undef PROFILER_ENABLED
#define PROFILER_ENABLED 1
#include "arch_specific.h"
#include "args.h"
#include "image.h"
#include "image_io.h"
#include "os_specific.h"
#include "padded_bytes.h"
#include "pik.h"
#include "pik_info.h"
#include "profiler.h"
#include "simd/dispatch.h"
#include "tsc_timer.h"

namespace pik {
namespace {

bool WriteFile(const PaddedBytes& compressed, const char* pathname) {
  FILE* f = fopen(pathname, "wb");
  if (f == nullptr) {
    fprintf(stderr, "Failed to open %s.\n", pathname);
    return false;
  }
  const size_t bytes_written =
      fwrite(compressed.data(), 1, compressed.size(), f);
  if (bytes_written != compressed.size()) {
    fprintf(stderr, "I/O error, only wrote %zu bytes.\n", bytes_written);
    return false;
  }
  fclose(f);
  return true;
}

struct CompressArgs {
  bool Init(int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
      if (argv[i][0] == '-') {
        const std::string arg = argv[i];
        if (arg == "--fast") {
          params.fast_mode = true;
        } else if (arg == "--denoise") {
          if (!ParseOverride(argc, argv, &i, &params.denoise)) return false;
        } else if (arg == "--noise") {
          if (!ParseOverride(argc, argv, &i, &params.apply_noise)) return false;
        } else if (arg == "--num_threads") {
          if (!ParseUnsigned(argc, argv, &i, &num_threads)) return false;
        } else if (arg == "-v") {
          params.verbose = true;
        } else if (arg == "--print_profile") {
          if (!ParseOverride(argc, argv, &i, &print_profile)) return false;
        } else if (arg == "--distance") {
          if (!ParseFloat(argc, argv, &i, &params.butteraugli_distance)) {
            return false;
          }
          if (!(0.5f <= params.butteraugli_distance &&
                params.butteraugli_distance <= 3.0f)) {
            fprintf(stderr,
                    "Invalid/out of range distance '%s', try 0.5 to 3.\n",
                    argv[i]);
            return false;
          }
        } else if (arg == "--target_size") {
          if (!ParseUnsigned(argc, argv, &i, &params.target_size)) return false;
        } else {
          // Unknown arg or --help: caller will print help string
          return false;
        }
      } else {
        if (file_in == nullptr) {
          file_in = argv[i];
        } else if (file_out == nullptr) {
          file_out = argv[i];
        } else {
          fprintf(stderr, "Extra argument after in/out names: %s.\n", argv[i]);
          return false;
        }
      }
    }

    if (file_in == nullptr) {
      fprintf(stderr, "Missing input filename.\n");
      return false;
    }

    return true;
  }

  static const char* HelpFormatString() {
    return "Usage: %s in.png out.pik [--distance <maxError>] [--fast] "
           "[--denoise <0,1>] [--noise <0,1>] [--num_threads <0..N>\n"
           "[--print_profile <0,1>]\n"
           " --distance: Max. butteraugli distance, lower = higher quality.\n"
           "             Good default: 1.0. Supported range: 0.5 .. 3.0.\n"
           " --fast: Use fast encoding, ignores distance.\n"
           " --denoise: force enable/disable edge-preserving smoothing.\n"
           " --noise: force enable/disable noise generation.\n"
           " --num_threads: number of worker threads (zero = none).\n"
           " --print_profile 1: print timing information before exiting.\n"
           " --help: Show this help.\n";
  }

  const char* file_in = nullptr;
  const char* file_out = nullptr;
  CompressParams params;
  size_t num_threads = 4;
  Override print_profile = Override::kDefault;
};

bool Compress(const CompressArgs& args, ThreadPool* pool,
              PaddedBytes* compressed) {
  MetaImageF in = ReadMetaImageLinear(args.file_in);
  if (in.xsize() == 0 || in.ysize() == 0) {
    fprintf(stderr, "Failed to open image %s.\n", args.file_in);
    return false;
  }

  if (args.params.target_size != 0 &&
      args.params.butteraugli_distance != -1.0f) {
    fprintf(stderr,
            "Only one of --distance or --target_size can be specified.\n");
    return false;
  }

  const size_t xsize = in.xsize();
  const size_t ysize = in.ysize();
  fprintf(stderr, "Compressing %zu x %zu pixels ", xsize, ysize);
  if (args.params.fast_mode) {
    fprintf(stderr, "with fast mode");
  } else if (args.params.target_size != 0) {
    fprintf(stderr, "to target size %zd", args.params.target_size);
  } else {
    fprintf(stderr, "with maximum Butteraugli distance %f",
            args.params.butteraugli_distance);
  }
  printf(", %zu threads.\n", pool->NumThreads());

  PikInfo aux_out;
  const uint64_t t0 = Start<uint64_t>();
  if (!PixelsToPik(args.params, in, pool, compressed, &aux_out)) {
    fprintf(stderr, "Failed to compress.\n");
    return false;
  }
  const uint64_t t1 = Stop<uint64_t>();
  const double elapsed = (t1 - t0) / InvariantTicksPerSecond();
  // TODO(janwas): account for 8 vs 16-bit input
  const size_t bytes = xsize * ysize * (in.HasAlpha() ? 4 : 3);
  fprintf(stderr, "Compressed to %zu bytes (%.2f MB/s).\n", compressed->size(),
          bytes * 1E-6 / elapsed);

  if (args.params.verbose) {
    aux_out.Print(1);
  }

  return true;
}

void InitThreads(ThreadPool* pool) {
  // Warm up profiler on main AND worker threads so its expensive initialization
  // doesn't count towards the timer measurements below for encode throughput.
  PROFILER_ZONE("@InitMainThread");
  pool->RunOnEachThread(
      [](const int task, const int thread) { PROFILER_ZONE("@InitWorkers"); });
}

int CompressAndWrite(int argc, char** argv) {
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

  CompressArgs args;
  if (!args.Init(argc, argv)) {
    fprintf(stderr, CompressArgs::HelpFormatString(), argv[0]);
    return 1;
  }

  ThreadPool pool(static_cast<int>(args.num_threads));
  InitThreads(&pool);

  PaddedBytes compressed;
  if (!Compress(args, &pool, &compressed)) return 1;

  if (!WriteFile(compressed, args.file_out)) return 1;

  if (args.print_profile == Override::kOn) {
    PROFILER_PRINT_RESULTS();
  }
  return 0;
}

}  // namespace
}  // namespace pik

int main(int argc, char** argv) { return pik::CompressAndWrite(argc, argv); }
