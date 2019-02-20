// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.
#include "benchmark/benchmark_codec_png.h"

#include <string>

#include "pik/codec_png.h"

namespace pik {

struct PNGArgs {
  // Empty, no PNG-specific args currently.
};

static PNGArgs* const pngargs = new PNGArgs;

Status AddCommandLineOptionsPNGCodec(BenchmarkArgs* args) {
  return true;
}

// Lossless.
class PNGCodec : public ImageCodec {
 public:
  PNGCodec(const BenchmarkArgs& args, const CodecContext& codec_context) :
      ImageCodec(args, codec_context) {}

  Status ParseParam(const std::string& param) override {
    return true;
  }

  Status Compress(const std::string& filename, const CodecInOut* io,
                  ThreadPool* pool, PaddedBytes* compressed) override {
    int bits = io->HasOriginalBitsPerSample()
             ? io->original_bits_per_sample() : 8;
    return EncodeImagePNG(io, io->c_current(), bits, pool, compressed);
  }

  Status Decompress(const std::string& filename, const PaddedBytes& compressed,
                    ThreadPool* pool, CodecInOut* io) override {
    return DecodeImagePNG(compressed, pool, io);
  }
};


ImageCodec* CreateNewPNGCodec(
    const BenchmarkArgs& args, const CodecContext& context) {
  return new PNGCodec(args, context);
}

}  // namespace pik
