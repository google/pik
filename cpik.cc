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

namespace pik {
namespace {

// main() function, within namespace for convenience.
int Compress(const char* pathname_in, const char* distance,
             const char* pathname_out, bool fast_mode) {
  MetaImageF in = ReadMetaImageLinear(pathname_in);
  if (in.xsize() == 0 || in.ysize() == 0) {
    fprintf(stderr, "Failed to open image %s.\n", pathname_in);
    return 1;
  }

  const float butteraugli_distance = strtod(distance, nullptr);
  if (!fast_mode &&
      !(0.5f <= butteraugli_distance && butteraugli_distance <= 3.0f)) {
    fprintf(stderr, "Invalid/out of range distance '%s', try 0.5 to 3.\n",
            distance);
    return 1;
  }
  printf("Compressing with maximum Butteraugli distance %f\n",
         butteraugli_distance);

  CompressParams params;
  params.butteraugli_distance = butteraugli_distance;
  params.alpha_channel = in.HasAlpha();
  if (fast_mode) {
    params.fast_mode = true;
    params.butteraugli_distance = -1;
  }
  PaddedBytes compressed;
  PikInfo aux_out;
  if (!PixelsToPik(params, in, &compressed, &aux_out)) {
    fprintf(stderr, "Failed to compress.\n");
    return 1;
  }

  printf("Compressed to %zu bytes\n", compressed.size());

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
  return 0;
}

}  // namespace
}  // namespace pik

void PrintArgHelp(int argc, char** argv) {
  fprintf(stderr,
          "Usage: %s in_rgb_8bit.png maxError[0.5 .. 3.0] out.pik [--fast]\n"
          "  with --fast, maxError is ignored\n",
          argv[0]);
}

int main(int argc, char** argv) {
  if (argc < 4) {
    PrintArgHelp(argc, argv);
    return 1;
  }

  bool fast_mode = false;
  if (argc > 4) {
    for (int i = 4; i < argc; i++) {
      std::string arg = argv[i];
      if (arg == "--fast") {
        fast_mode = true;
      } else {
        PrintArgHelp(argc, argv);
        return 1;
      }
    }
  }

  return pik::Compress(argv[1], argv[2], argv[3], fast_mode);
}
