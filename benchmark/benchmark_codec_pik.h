// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_BENCHMARK_BENCHMARK_CODEC_PIK_H_
#define PIK_BENCHMARK_BENCHMARK_CODEC_PIK_H_

#include <string>

#include "benchmark/benchmark_xl.h"

namespace pik {
ImageCodec* CreateNewPikCodec(
    const BenchmarkArgs& args, const CodecContext& context);

// Registers the pik-specific command line options.
Status AddCommandLineOptionsPikCodec(BenchmarkArgs* args);
}  // namespace pik

#endif  // PIK_BENCHMARK_BENCHMARK_CODEC_PIK_H_
