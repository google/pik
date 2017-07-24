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

#include "image_io.h"
#include "pik.h"

namespace pik {
namespace {

// main() function, within namespace for convenience.
int Compress(const char* pathname_in, const char* pathname_out) {
  Image3B in;
  if (!ReadImage(ImageFormatPNG(), pathname_in, &in)) {
    fprintf(stderr, "Failed to open %s\n", pathname_in);
    return 1;
  }

  CompressParams params;
  params.butteraugli_distance = 1.0f;
  Bytes compressed;
  PikInfo aux_out;
  if (PixelsToPik(params, in, &compressed, &aux_out) != Status::OK) {
    fprintf(stderr, "Failed to compress.\n");
    return 1;
  }

  printf("Compressed to %zu bytes\n", compressed.size());
  return 0;
}

}  // namespace
}  // namespace pik

int main(int argc, char** argv) {
  if (argc != 3) {
    fprintf(stderr, "Usage: %s rgb_8bit.png out.pik\n", argv[0]);
    return 1;
  }

  return pik::Compress(argv[1], argv[2]);
}
