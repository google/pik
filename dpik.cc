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
#include <string.h>

#define PROFILER_ENABLED 1
#include "arch_specific.h"
#include "args.h"
#include "gamma_correct.h"
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

struct DecompressArgs {
  bool Init(int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
      if (argv[i][0] == '-') {
        if (strcmp(argv[i], "--16bit") == 0) {
          sixteen_bit = true;
        } else if (strcmp(argv[i], "--denoise") == 0) {
          if (!ParseOverride(argc, argv, &i, &params.denoise)) return false;
        } else if (strcmp(argv[i], "--num_threads") == 0) {
          if (!ParseUnsigned(argc, argv, &i, &num_threads)) return false;
        } else if (strcmp(argv[i], "--num_reps") == 0) {
          if (!ParseUnsigned(argc, argv, &i, &num_reps)) return false;
        } else if (strcmp(argv[i], "--print_profile") == 0) {
          if (!ParseOverride(argc, argv, &i, &print_profile)) return false;
        } else {
          fprintf(stderr, "Unrecognized argument: %s.\n", argv[i]);
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
    return "Usage: %s [--16bit] [--denoise B] [--num_threads N]\n"
           "  [--num_reps N] [--print_profile B] in.pik [out.png]\n"
           "  The output is 16 bit if --16bit is set, otherwise 8-bit sRGB.\n"
           "  B is a boolean (0/1), N an unsigned integer.\n"
           "  --denoise 1: enable deringing/deblocking postprocessor.\n"
           "  --print_profile 1: print timing information before exiting.\n";
  }

  const char* file_in = nullptr;
  const char* file_out = nullptr;
  bool sixteen_bit = false;
  DecompressParams params;
  size_t num_threads = 8;
  size_t num_reps = 1;
  Override print_profile = Override::kDefault;
};

bool LoadFile(const char* pathname, PaddedBytes* compressed) {
  FILE* f = fopen(pathname, "rb");
  if (f == nullptr) {
    fprintf(stderr, "Failed to open %s.\n", pathname);
    return false;
  }

  if (fseek(f, 0, SEEK_END) != 0) {
    fprintf(stderr, "Seek error at end.\n");
    return false;
  }
  compressed->resize(ftell(f));
  if (fseek(f, 0, SEEK_SET) != 0) {
    fprintf(stderr, "Seek error at beginning.\n");
    return false;
  }

  const size_t bytes_read = fread(compressed->data(), 1, compressed->size(), f);
  if (bytes_read != compressed->size()) {
    fprintf(stderr, "I/O error, only read %zu bytes.\n", bytes_read);
    return false;
  }
  fprintf(stderr, "Read %zu compressed bytes\n", compressed->size());
  fclose(f);
  return true;
}

void InitThreads(ThreadPool* pool) {
  // Warm up profiler on main AND worker threads so its expensive initialization
  // doesn't count towards the timer measurements below for decode throughput.
  PROFILER_ZONE("@InitMainThread");
  pool->RunOnEachThread([](const int task, const int thread) {
    PROFILER_ZONE("@InitWorkers");

    // PinThreadToCPU(1 + thread);
    // const int group = thread & 1;
    // const int member = thread >> 1;
    // PinThreadToCPU(member + group * 12);
  });
}

template <typename ComponentType>
bool Decompress(const PaddedBytes& compressed, const DecompressParams& params,
                ThreadPool* pool, MetaImage<ComponentType>* image) {
  PikInfo info;
  const uint64_t t0 = Start<uint64_t>();
  if (!PikToPixels(params, compressed, pool, image, &info)) {
    fprintf(stderr, "Failed to decompress.\n");
    return false;
  }
  const uint64_t t1 = Stop<uint64_t>();
  const double elapsed = (t1 - t0) / InvariantTicksPerSecond();
  const size_t xsize = image->xsize();
  const size_t ysize = image->ysize();
  const size_t bytes =
      xsize * ysize * sizeof(ComponentType) * (image->HasAlpha() ? 4 : 3);
  fprintf(stderr, "Decompressed %zu x %zu pixels (%.2f MB/s, %zu threads).\n",
          xsize, ysize, bytes * 1E-6 / elapsed, pool->NumThreads());

  return true;
}

template <typename ComponentType>
bool DecompressAndWrite(const PaddedBytes& compressed,
                        const DecompressArgs& args, ThreadPool* pool) {
  MetaImage<ComponentType> image;
  for (size_t i = 0; i < args.num_reps; ++i) {
    if (!Decompress(compressed, args.params, pool, &image)) return false;
  }

  // Writing large PNGs is slow, so allow skipping it for benchmarks.
  if (args.file_out != nullptr) {
    if (!WriteImage(ImageFormatPNG(), image, args.file_out)) {
      fprintf(stderr, "Failed to write %s.\n", args.file_out);
      return false;
    }
  }

  return true;
}

int Run(int argc, char* argv[]) {
  DecompressArgs args;
  if (!args.Init(argc, argv)) {
    fprintf(stderr, DecompressArgs::HelpFormatString(), argv[0]);
    return 1;
  }

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

  PaddedBytes compressed;
  if (!LoadFile(args.file_in, &compressed)) {
    return 1;
  }

  ThreadPool pool(static_cast<int>(args.num_threads));
  InitThreads(&pool);

  const auto decompressor = args.sixteen_bit ? &DecompressAndWrite<uint16_t>
                                             : &DecompressAndWrite<uint8_t>;
  if (!decompressor(compressed, args, &pool)) return 1;

  if (args.print_profile == Override::kOn) {
    PROFILER_PRINT_RESULTS();
  }

  return 0;
}

}  // namespace
}  // namespace pik

int main(int argc, char* argv[]) { return pik::Run(argc, argv); }
