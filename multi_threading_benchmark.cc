// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

// Verifies to what extent concurrent PIK decoders interfere with each other
// performance-wise.
//
// Usage:
//   ./multi_threading_benchmark --input=in.pik

#include <stddef.h>
#include <string>

#include "gflags/gflags.h"
#include "benchmark/benchmark.h"
#include "codec.h"
#include "color_encoding.h"
#include "common.h"
#include "file_io.h"
#include "padded_bytes.h"
#include "pik.h"
#include "pik_info.h"
#include "pik_params.h"

namespace pik {
namespace {

DEFINE_string(input, "", "Image to decode.");

void BM_PikDecoding(benchmark::State& state) {
  static CodecContext context;
  PaddedBytes compressed;
  PIK_CHECK(ReadFile(FLAGS_input, &compressed));

  CodecInOut io(&context);
  for (auto _ : state) {
    DecompressParams params;
    PikInfo info;
    PIK_CHECK(PikToPixels(params, compressed, &io, &info, /*pool=*/nullptr));
  }
  state.SetBytesProcessed(state.iterations() * io.xsize() * io.ysize() *
                          (io.c_current().Channels() + io.HasAlpha()) *
                          DivCeil(io.original_bits_per_sample(), kBitsPerByte));
}
BENCHMARK(BM_PikDecoding)->Threads(1)->Threads(4)->Threads(8);

}  // namespace
}  // namespace pik

int main(int argc, char** argv) {
  benchmark::Initialize(&argc, argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  benchmark::RunSpecifiedBenchmarks();
}
