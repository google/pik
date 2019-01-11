// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include <cstddef>
#include <cstdio>
#include <cstdlib>

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
    bool got_target_bpp = false;
    for (int i = 1; i < argc; i++) {
      if (argv[i][0] == '-') {
        const std::string arg = argv[i];
        if (arg == "--fast") {
          params.fast_mode = true;
        } else if (arg == "--guetzli") {
          params.guetzli_mode = true;
        } else if (arg == "--progressive") {
          params.progressive_mode = true;
        } else if (arg == "--lossless_base") {
          if (i + 1 > argc) {
            fprintf(stderr, "Expected an argument for --lossless_base.\n");
            return PIK_FAILURE("Args");
          }
          params.lossless_base = argv[++i];
        } else if (arg == "--lossless") {
          params.lossless_mode = true;
        } else if (arg == "--keep_tempfiles") {
          params.keep_tempfiles = true;
        } else if (arg == "--noise") {
          PIK_RETURN_IF_ERROR(ParseOverride(argc, argv, &i, &params.noise));
        } else if (arg == "--gradient") {
          PIK_RETURN_IF_ERROR(ParseOverride(argc, argv, &i, &params.gradient));
        } else if (arg == "--adaptive_reconstruction") {
          PIK_RETURN_IF_ERROR(
              ParseOverride(argc, argv, &i, &params.adaptive_reconstruction));
        } else if (strcmp(argv[i], "--gaborish") == 0) {
          size_t strength;
          PIK_RETURN_IF_ERROR(ParseUnsigned(argc, argv, &i, &strength));
          params.gaborish = static_cast<GaborishStrength>(strength);
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
        } else if (arg == "--target_bpp") {
          printf("Warning: target_bpp does not set all flags/modes.\n");
          PIK_RETURN_IF_ERROR(
              ParseFloat(argc, argv, &i, &params.target_bitrate));
          got_target_bpp = true;
        } else if (arg == "--intensity_target") {
          PIK_RETURN_IF_ERROR(
              ParseFloat(argc, argv, &i, &params.intensity_target));
        } else if (arg == "--saliency_extractor") {
          PIK_RETURN_IF_ERROR(ParseString(
              argc, argv, &i, &params.saliency_extractor_for_progressive_mode));
        } else if (arg == "--saliency_threshold") {
          PIK_RETURN_IF_ERROR(
              ParseFloat(argc, argv, &i, &params.saliency_threshold));
        } else if (arg == "--saliency_debug_skip_nonsalient") {
          params.saliency_debug_skip_nonsalient = true;
        } else {
          // Unknown arg or --help: caller will print help string
          return false;
        }
      } else {
        if (params.file_in.empty()) {
          params.file_in = argv[i];
        } else if (params.file_out.empty()) {
          params.file_out = argv[i];
        } else {
          fprintf(stderr, "Extra argument after in/out names: %s.\n", argv[i]);
          return false;
        }
      }
    }

    if (got_target_bpp + got_target_size + got_distance > 1) {
      fprintf(stderr, "You can specify only one of '--distance', "
              "'--target_bpp' and '--target_size'. They are all different ways"
              " to specify the image quality. When in doubt, use --distance."
              " It gives the most visually consistent results.\n");
      return false;
    }

    if (!params.lossless_base.empty() &&
        (params.lossless_mode || params.progressive_mode)) {
      fprintf(stderr,
              "--lossless_base is incompatible with --lossless_mode and "
              "--progressive_mode.\n");
      return false;
    }

    if (!params.saliency_extractor_for_progressive_mode.empty()) {
      if (!params.progressive_mode) {
        printf("Warning: Specifying --saliency_extractor only makes sense "
               "for --progressive mode.\n");
      }
      if (params.file_out.empty()) {
        fprintf(stderr,
                "Need to have output filename to use saliency extractor.\n");
        return PIK_FAILURE("file_out");
      }
    }

    if (!got_num_threads) {
      num_threads = AvailableCPUs().size();
    }

    if (params.file_in.empty()) {
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
           "     Good default: 1.0. Supported range: 0.5 .. 3.0.\n"
           " --target_bpp N. Aim at file size that has N bits per pixel.\n"
           "     Compresses to 1 % of the target BPP in ideal conditions.\n"
           " --target_size N. Aim at file size of N bytes.\n"
           "     Compresses to 1 % of the target size in ideal conditions.\n"
           "     Runs the same algorithm as --target_bpp\n"
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
    // TODO(user): Make some currently still "hidden" new flags officially
    // documented.
  }

  DecoderHints dec_hints;
  CompressParams params;
  size_t num_threads = 0;
  bool got_num_threads = false;
  Override print_profile = Override::kDefault;
};

// Proposes a distance to try for a given bpp target. This could depend
// on the entropy in the image, too, but let's start with something.
static double ApproximateDistanceForBPP(double bpp) {
  return 1.704 * pow(bpp, -0.804);
}

Status Compress(ThreadPool* pool, CompressArgs& args,
                PaddedBytes* compressed) {
  double t0, t1;

  CodecContext codec_context;
  CodecInOut io(&codec_context);
  io.dec_hints = args.dec_hints;
  t0 = Now();
  if (!io.SetFromFile(args.params.file_in)) {
    fprintf(stderr, "Failed to read image %s.\n", args.params.file_in.c_str());
    return false;
  }
  t1 = Now();
  const double decode_mps = io.xsize() * io.ysize() * 1E-6 / (t1 - t0);

  const size_t xsize = io.xsize();
  const size_t ysize = io.ysize();
  if (args.params.target_size > 0 || args.params.target_bitrate > 0) {
    // Search algorithm for target bpp / size.
    CompressArgs s = args;  // Args for search.
    if (s.params.target_size > 0) {
      s.params.target_bitrate = s.params.target_size * 8.0 /
                                (io.xsize() * io.ysize());
      s.params.target_size = 0;
    }
    double dist = ApproximateDistanceForBPP(s.params.target_bitrate);
    s.params.butteraugli_distance = dist;
    double target_size =
        s.params.target_bitrate * (1 / 8.) * io.xsize() * io.ysize();
    s.params.target_bitrate = 0;
    double best_dist = 1.0;
    double best_loss = 1e99;
    for (int i = 0; i < 12; ++i) {
      s.params.butteraugli_distance = dist;
      PaddedBytes candidate;
      bool ok = Compress(pool, s, &candidate);
      if (!ok) {
        printf("Compression error occured during the search for best size."
               " Trying with butteraugli distance %.15g\n", best_dist);
        break;
      }
      printf("Butteraugli distance %g gives to %zu bytes, %g bpp.\n",
             best_dist, candidate.size(),
             candidate.size() * 8.0 / (io.xsize() * io.ysize()));
      const double ratio = static_cast<double>(candidate.size()) /
                           target_size;
      const double loss = std::max(ratio, 1.0 / std::max(ratio, 1e-30));
      if (best_loss > loss) {
        best_dist = dist;
        best_loss = loss;
      }
      dist *= ratio;
      if (dist < 0.01) {
        dist = 0.01;
      }
      if (dist >= 16.0) {
        dist = 16.0;
      }
    }
    printf("Choosing butteraugli distance %.15g\n", best_dist);
    args.params.butteraugli_distance = best_dist;
    args.params.target_bitrate = 0;
    args.params.target_size = 0;
  }
  char mode[200];
  if (args.params.fast_mode) {
    strcpy(mode, "in fast mode ");
  }
  snprintf(mode, sizeof(mode), "with maximum Butteraugli distance %f",
           args.params.butteraugli_distance);
  fprintf(stderr,
          "Read %zu bytes (%zux%zu, %.1f MP/s); compressing %s, %zu threads.\n",
          io.enc_size, xsize, ysize, decode_mps, mode, NumWorkerThreads(pool));

  PikInfo aux_out;
  t0 = Now();
  if (!PixelsToPik(args.params, &io, compressed, &aux_out, pool)) {
    fprintf(stderr, "Failed to compress.\n");
    return false;
  }
  t1 = Now();
  const size_t channels = io.c_current().Channels() + io.HasAlpha();
  const size_t bytes = xsize * ysize * channels *
                       DivCeil(io.original_bits_per_sample(), kBitsPerByte);
  const double bpp = static_cast<double>(compressed->size() * kBitsPerByte) /
                     (xsize * ysize);
  fprintf(stderr, "Compressed to %zu bytes (%.2f bpp, %.2f MB/s).\n",
          compressed->size(), bpp, bytes * 1E-6 / (t1 - t0));

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

  ThreadPool pool(args.num_threads);

  PaddedBytes compressed;
  if (!Compress(&pool, args, &compressed)) return 1;

  if (!args.params.file_out.empty()) {
    if (!WriteFile(compressed, args.params.file_out)) return 1;
  }

  if (args.print_profile == Override::kOn) {
    PROFILER_PRINT_RESULTS();
  }
  return 0;
}

}  // namespace
}  // namespace pik

int main(int argc, char** argv) { return pik::CompressAndWrite(argc, argv); }
