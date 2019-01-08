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
#include "data_parallel.h"

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
#include "robust_statistics.h"
#include "simd/targets.h"

namespace pik {
namespace {

struct DecompressArgs {
  Status Init(int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
      if (argv[i][0] == '-') {
        if (strcmp(argv[i], "--bits_per_sample") == 0) {
          PIK_RETURN_IF_ERROR(ParseUnsigned(argc, argv, &i, &bits_per_sample));
        } else if (strcmp(argv[i], "--num_threads") == 0) {
          PIK_RETURN_IF_ERROR(ParseUnsigned(argc, argv, &i, &num_threads));
          got_num_threads = true;
        } else if (strcmp(argv[i], "--color_space") == 0) {
          i += 1;
          if (i >= argc) {
            fprintf(stderr, "Expected color_space description.\n");
            return PIK_FAILURE("Args");
          }
          color_space = argv[i];
        } else if (strcmp(argv[i], "--num_reps") == 0) {
          PIK_RETURN_IF_ERROR(ParseUnsigned(argc, argv, &i, &num_reps));
        } else if (strcmp(argv[i], "--noise") == 0) {
          PIK_RETURN_IF_ERROR(ParseOverride(argc, argv, &i, &params.noise));
          if (params.noise == Override::kOn) {
            fprintf(stderr, "Noise can only be enabled by the encoder.\n");
            return PIK_FAILURE("Cannot force noise on");
          }
        } else if (strcmp(argv[i], "--gradient") == 0) {
          PIK_RETURN_IF_ERROR(ParseOverride(argc, argv, &i, &params.gradient));
          if (params.gradient == Override::kOn) {
            fprintf(stderr, "Gradient can only be enabled by the encoder.\n");
            return PIK_FAILURE("Cannot force gradient on");
          }
        } else if (strcmp(argv[i], "--adaptive_reconstruction") == 0) {
          PIK_RETURN_IF_ERROR(
              ParseOverride(argc, argv, &i, &params.adaptive_reconstruction));
        } else if (strcmp(argv[i], "--gaborish") == 0) {
          size_t strength;
          PIK_RETURN_IF_ERROR(ParseUnsigned(argc, argv, &i, &strength));
          params.gaborish = static_cast<GaborishStrength>(strength);
        } else if (strcmp(argv[i], "--print_profile") == 0) {
          PIK_RETURN_IF_ERROR(ParseOverride(argc, argv, &i, &print_profile));
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

    if (!got_num_threads) {
      num_threads = AvailableCPUs().size();
    }
    return true;
  }

  static const char* HelpFormatString() {
    return "Usage: %s [--bits_per_sample N] [--num_threads N]\n"
           "[--color_space RGB_D65_SRG_Rel_Lin] [--gaborish N]\n"
           "[--noise 0] [--gradient 0] [--adaptive_reconstruction <0,1>]\n"
           "[--num_reps N] [--print_profile B] in.pik [out]\n"
           "  B is a boolean (0/1), N an unsigned integer.\n"
           "  --bits_per_sample defaults to original (input) bit depth.\n"
           "  --noise 0 disables noise generation.\n"
           "  --gradient 0 disables the extra gradient map.\n"
           "  --adaptive_reconstruction 1/0 enables/disables extra filtering.\n"
           "  --gaborish 0..7 chooses deblocking strength (4=normal).\n"
           "  --color_space defaults to original (input) color space.\n"
           "  --print_profile 1: print timing information before exiting.\n"
           "  out is PNG with ICC, or PPM/PFM.\n";
  }

  const char* file_in = nullptr;
  const char* file_out = nullptr;
  size_t bits_per_sample = 0;
  size_t num_threads = 0;
  bool got_num_threads = false;
  std::string color_space;  // description
  DecompressParams params;
  size_t num_reps = 1;
  Override print_profile = Override::kDefault;
};

class DecompressStats {
 public:
  void NotifyElapsed(double elapsed_seconds) {
    PIK_ASSERT(elapsed_seconds > 0.0);
    elapsed_.push_back(elapsed_seconds);
  }

  Status Print(const CodecInOut& io, ThreadPool* pool) {
    const size_t xsize = io.xsize();
    const size_t ysize = io.ysize();
    const size_t channels = io.c_current().Channels() + io.HasAlpha();
    const size_t bytes = xsize * ysize * channels *
                         DivCeil(io.original_bits_per_sample(), kBitsPerByte);

    double elapsed;
    double variability;
    const char* type;
    PIK_RETURN_IF_ERROR(SummarizeElapsed(&elapsed, &variability, &type));
    if (variability == 0.0) {
      fprintf(stderr,
              "%zu x %zu pixels (%s%.2f MB/s, %zu reps, %zu threads).\n", xsize,
              ysize, type, bytes * 1E-6 / elapsed, elapsed_.size(),
              NumWorkerThreads(pool));
    } else {
      fprintf(
          stderr,
          "%zu x %zu pixels (%s%.2f MB/s (var %.2f), %zu reps, %zu threads).\n",
          xsize, ysize, type, bytes * 1E-6 / elapsed, variability,
          elapsed_.size(), NumWorkerThreads(pool));
    }
    return true;
  }

 private:
  Status SummarizeElapsed(double* PIK_RESTRICT summary,
                          double* PIK_RESTRICT variability, const char** type) {
    // type depends on #reps.
    if (elapsed_.empty()) return PIK_FAILURE("Didn't call NotifyElapsed");

    // Single rep
    if (elapsed_.size() == 1) {
      *summary = elapsed_[0];
      *variability = 0.0;
      *type = "";
      return true;
    }

    // Two: skip first (noisier)
    if (elapsed_.size() == 2) {
      *summary = elapsed_[1];
      *variability = 0.0;
      *type = "second: ";
      return true;
    }

    // Prefer geomean unless numerically unreliable (too many reps)
    if (std::pow(elapsed_[0], elapsed_.size()) < 1E100) {
      double product = 1.0;
      for (size_t i = 1; i < elapsed_.size(); ++i) {
        product *= elapsed_[i];
      }

      *summary = std::pow(product, 1.0 / (elapsed_.size() - 1));
      *variability = 0.0;
      *type = "geomean: ";
      return true;
    }

    // Else: mode
    std::sort(elapsed_.begin(), elapsed_.end());
    *summary = HalfSampleMode()(elapsed_.data(), elapsed_.size());
    *variability = MedianAbsoluteDeviation(elapsed_, *summary);
    *type = "mode: ";
    return true;
  }

  std::vector<double> elapsed_;
};

// Called num_reps times.
Status Decompress(CodecContext* codec_context, const PaddedBytes& compressed,
                  const DecompressParams& params, CodecInOut* PIK_RESTRICT io,
                  DecompressStats* PIK_RESTRICT stats) {
  PikInfo info;
  const double t0 = Now();
  if (!PikToPixels(params, compressed, io, &info)) {
    fprintf(stderr, "Failed to decompress.\n");
    return false;
  }
  const double t1 = Now();
  stats->NotifyElapsed(t1 - t0);
  return true;
}

Status WriteOutput(const DecompressArgs& args, const CodecInOut& io) {
  // Can only write if we decoded and have an output filename.
  // (Writing large PNGs is slow, so allow skipping it for benchmarks.)
  if (args.num_reps == 0 || args.file_out == nullptr) return true;

  // Override original color space with arg if specified.
  ColorEncoding c_out = io.dec_c_original;
  if (!args.color_space.empty()) {
    ProfileParams pp;
    if (!ParseDescription(args.color_space, &pp) ||
        !ColorManagement::SetFromParams(pp, &c_out)) {
      fprintf(stderr, "Failed to apply color_space.\n");
      return false;
    }
  }

  // Override original #bits with arg if specified.
  const size_t bits_per_sample = args.bits_per_sample == 0
                                     ? io.original_bits_per_sample()
                                     : args.bits_per_sample;

  if (!io.EncodeToFile(c_out, bits_per_sample, args.file_out)) {
    fprintf(stderr, "Failed to write decoded image.\n");
    return false;
  }
  fprintf(stderr, "Wrote %zu bytes; done.\n", io.enc_size);
  return true;
}

int Decompress(int argc, char* argv[]) {
  DecompressArgs args;
  if (!args.Init(argc, argv)) {
    fprintf(stderr, DecompressArgs::HelpFormatString(), argv[0]);
    return 1;
  }

  const int bits = TargetBitfield().Bits();
  if ((bits & SIMD_ENABLE) != SIMD_ENABLE) {
    fprintf(stderr, "CPU does not support all enabled targets => exiting.\n");
    return 1;
  }

  PaddedBytes compressed;
  if (!ReadFile(args.file_in, &compressed)) return 1;
  fprintf(stderr, "Read %zu compressed bytes\n", compressed.size());

  CodecContext codec_context;
  ThreadPool pool(args.num_threads);
  DecompressStats stats;

  const std::vector<int> cpus = AvailableCPUs();
  pool.RunOnEachThread([&cpus](const int task, const int thread) {
    // 1.1-1.2x speedup (36 cores) from pinning.
    if (thread < cpus.size()) {
      if (!PinThreadToCPU(cpus[thread])) {
        fprintf(stderr, "WARNING: failed to pin thread %d.\n", thread);
      }
    }
  });

  CodecInOut io(&codec_context);
  for (size_t i = 0; i < args.num_reps; ++i) {
    if (!Decompress(&codec_context, compressed, args.params, &io, &stats)) {
      return 1;
    }
  }

  if (!WriteOutput(args, io)) return 1;

  (void)stats.Print(io, &pool);

  if (args.print_profile == Override::kOn) {
    PROFILER_PRINT_RESULTS();
  }

  return 0;
}

}  // namespace
}  // namespace pik

int main(int argc, char* argv[]) { return pik::Decompress(argc, argv); }
