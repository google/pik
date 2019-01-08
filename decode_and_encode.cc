// Copyright 2018 Google Inc. All Rights Reserved.
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

#include "codec.h"

namespace pik {
namespace {

// Reads an input file (typically PNM) with color_space hint and writes to an
// output file (typically PNG) which supports all required metadata.
int Convert(int argc, char** argv) {
  if (argc != 4) {
    fprintf(stderr, "Args: in colorspace_description out\n");
    return 1;
  }
  const std::string& in = argv[1];
  const std::string& desc = argv[2];
  const std::string& out = argv[3];

  CodecContext codec_context;
  CodecInOut io(&codec_context);
  ThreadPool pool(4);
  io.dec_hints.Add("color_space", desc);
  if (!io.SetFromFile(in, &pool)) {
    fprintf(stderr, "Failed to read %s\n", in.c_str());
    return 1;
  }
  if (!io.EncodeToFile(io.dec_c_original, io.original_bits_per_sample(), out,
                       &pool)) {
    fprintf(stderr, "Failed to write %s\n", out.c_str());
    return 1;
  }

  return 0;
}

}  // namespace
}  // namespace pik

int main(int argc, char** argv) { return pik::Convert(argc, argv); }
