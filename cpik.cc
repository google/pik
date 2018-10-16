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
#include "codec.h"
#include "common.h"
#include "file_io.h"
#include "image.h"
#include "os_specific.h"
#include "padded_bytes.h"
#include "pik.h"
#include "pik_info.h"
#include "profiler.h"
#include "simd/targets.h"

namespace pik {
namespace {

struct CompressArgs {
  Status Init(int argc, char** argv) {
    bool got_distance = false;
    bool got_target_size = false;
    for (int i = 1; i < argc; i++) {
      if (argv[i][0] == '-') {
        const std::string arg = argv[i];
        if (arg == "--fast") {
          params.fast_mode = true;
        } else if (arg == "--guetzli") {
          params.guetzli_mode = true;
        } else if (arg == "--noise") {
          PIK_RETURN_IF_ERROR(ParseOverride(argc, argv, &i, &params.noise));
        } else if (arg == "--smooth") {
          PIK_RETURN_IF_ERROR(ParseOverride(argc, argv, &i, &params.smooth));
        } else if (arg == "--gradient") {
          PIK_RETURN_IF_ERROR(ParseOverride(argc, argv, &i, &params.gradient));
        } else if (arg == "--adaptive_reconstruction") {
          PIK_RETURN_IF_ERROR(
              ParseOverride(argc, argv, &i, &params.adaptive_reconstruction));
        } else if (strcmp(argv[i], "--gaborish") == 0) {
          size_t strength;
          PIK_RETURN_IF_ERROR(ParseUnsigned(argc, argv, &i, &strength));
          if (strength > 7) {
            fprintf(stderr, "Invalid gaborish strength, must be 0..7.\n");
            return PIK_FAILURE("Invalid gaborish strength");
          }
          params.gaborish = strength;
        } else if (arg == "--resampleX2") {
          PIK_RETURN_IF_ERROR(
              ParseUnsigned(argc, argv, &i, &params.resampling_factor2));
        } else if (arg == "--num_threads") {
          PIK_RETURN_IF_ERROR(ParseUnsigned(argc, argv, &i, &num_threads));
          got_num_threads = true;
        } else if (arg == "-v") {
          params.verbose = true;
        } else if (arg == "-x") {
          if (i + 2 >= argc) {
            fprintf(stderr, "Expected key and value arguments.\n");
            return PIK_FAILURE("Args");
          }
          const std::string key(argv[i + 1]);
          const std::string value(argv[i + 2]);
          i += 2;
          dec_hints.Add(key, value);
        } else if (arg == "--print_profile") {
          PIK_RETURN_IF_ERROR(ParseOverride(argc, argv, &i, &print_profile));
        } else if (arg == "--distance") {
          PIK_RETURN_IF_ERROR(
              ParseFloat(argc, argv, &i, &params.butteraugli_distance));
          {
            constexpr float butteraugli_min_dist = 0.125f;
            constexpr float butteraugli_max_dist = 15.0f;
            if (!(butteraugli_min_dist <= params.butteraugli_distance &&
                  params.butteraugli_distance <= butteraugli_max_dist)) {
              fprintf(stderr,
                      "Invalid/out of range distance '%s', try %g to %g.\n",
                      argv[i], butteraugli_min_dist, butteraugli_max_dist);
              return false;
            }
          }
          got_distance = true;
        } else if (arg == "--target_size") {
          printf("Warning: target_size does not set all flags/modes.\n");
          PIK_RETURN_IF_ERROR(
              ParseUnsigned(argc, argv, &i, &params.target_size));
          got_target_size = true;
        } else if (arg == "--intensity_target") {
          PIK_RETURN_IF_ERROR(
              ParseFloat(argc, argv, &i, &params.intensity_target));
        } else if (arg == "--roi_factor") {
          PIK_RETURN_IF_ERROR(ParseFloat(argc, argv, &i, &params.roi_factor));
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

    // The default distance is 1, and there is a check downstream that only one
    // of distance and target_size is specified. Thus, if target_size is
    // specified, we reset distance to -1 to avoid the error if it is
    // unwarranted.
    if (got_target_size && !got_distance) {
      params.butteraugli_distance = -1.0f;
    }

    if (got_target_size && got_distance) {
      fprintf(stderr, "Cannot specify both --distance and --target_size.\n");
      return false;
    }

    if (!got_num_threads) {
      num_threads = AvailableCPUs().size();
    }

    if (file_in == nullptr) {
      fprintf(stderr, "Missing input filename.\n");
      return false;
    }

    return true;
  }

  static const char* HelpFormatString() {
    return "Usage: %s in out.pik [--distance <maxError>] [--fast] [-v]\n"
           "[--num_threads <0..N>] [--print_profile <0,1>] [-x key value]\n"
           "[--resampleX2 N]\n"
           "[--noise <0,1>] [--smooth <0,1>] [--gradient <0,1>]\n"
           "[--adaptive_reconstruction <0,1>] [--gaborish <0..7>]\n"
           " in can be PNG, PNM or PFM.\n"
           " --distance: Max. butteraugli distance, lower = higher quality.\n"
           "             Good default: 1.0. Supported range: 0.5 .. 3.0.\n"
           " --resampleX2 is twice the downsampling factor, 3 for 1.5x.\n"
           " --fast: Use fast encoding mode (less dense).\n"
           " --noise: force enable/disable noise generation.\n"
           " --smooth: force enable/disable smooth predictor.\n"
           " --gradient: force enable/disable extra gradient map.\n"
           " --adaptive_reconstruction: force enable/disable decoder filter.\n"
           " --gaborish 0..7 chooses deblocking strength (4=normal).\n"
           " --intensity_target: Intensity target of monitor in nits, higher\n"
           "   results in higher quality image. Supported range: 250..6000,\n"
           "   default is 250.\n"
           " --num_threads: number of worker threads (zero = none).\n"
           " --print_profile 1: print timing information before exiting.\n"
           " -x color_space indicates the ColorEncoding, see Description().\n"
           " -v enable verbose mode with additional output.\n"
           " --help: Show this help.\n";
  }

  const char* file_in = nullptr;
  const char* file_out = nullptr;
  DecoderHints dec_hints;
  CompressParams params;
  size_t num_threads = 0;
  bool got_num_threads = false;
  Override print_profile = Override::kDefault;
};

Status Compress(CodecContext* codec_context,
                const CompressArgs& args, PaddedBytes* compressed) {
  CodecInOut io(codec_context);
  io.dec_hints = args.dec_hints;
  if (!io.SetFromFile(args.file_in)) {
    fprintf(stderr, "Failed to read image %s.\n", args.file_in);
    return false;
  }

  const size_t xsize = io.xsize();
  const size_t ysize = io.ysize();
  char mode[200];
  if (args.params.fast_mode) {
    strcpy(mode, "with fast mode");
  } else if (args.params.target_size != 0) {
    snprintf(mode, sizeof(mode), "with target size %zu",
             args.params.target_size);
  } else {
    snprintf(mode, sizeof(mode), "with maximum Butteraugli distance %f",
             args.params.butteraugli_distance);
  }
  fprintf(stderr, "Read %zu bytes (%zux%zu px); compressing %s, %zu threads.\n",
          io.enc_size, xsize, ysize, mode, codec_context->pool.NumThreads());

  PikInfo aux_out;
  const double t0 = Now();
  if (!PixelsToPik(args.params, &io, compressed, &aux_out)) {
    fprintf(stderr, "Failed to compress.\n");
    return false;
  }
  const double t1 = Now();
  const size_t channels = io.c_current().Channels() + io.HasAlpha();
  const size_t bytes = xsize * ysize * channels * DivCeil(
      io.original_bits_per_sample(), kBitsPerByte);
  fprintf(stderr, "Compressed to %zu bytes (%.2f MB/s).\n", compressed->size(),
          bytes * 1E-6 / (t1 - t0));

  if (args.params.verbose) {
    aux_out.Print(1);
  }

  return true;
}

int CompressAndWrite(int argc, char** argv) {
  const int bits = TargetBitfield().Bits();
  if ((bits & SIMD_ENABLE) != SIMD_ENABLE) {
    fprintf(stderr, "CPU does not support all enabled targets => exiting.\n");
    return 1;
  }

  CompressArgs args;
  if (!args.Init(argc, argv)) {
    fprintf(stderr, CompressArgs::HelpFormatString(), argv[0]);
    return 1;
  }

  CodecContext codec_context(args.num_threads);

  PaddedBytes compressed;
  if (!Compress(&codec_context, args, &compressed)) return 1;

  if (args.file_out != nullptr) {
    if (!WriteFile(compressed, args.file_out)) return 1;
  }

  if (args.print_profile == Override::kOn) {
    PROFILER_PRINT_RESULTS();
  }
  return 0;
}

}  // namespace
}  // namespace pik

int main(int argc, char** argv) { return pik::CompressAndWrite(argc, argv); }
