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

#include "gamma_correct.h"
#include "image.h"
#include "image_io.h"
#include "simd/dispatch.h"
#include "padded_bytes.h"
#include "pik.h"
#include "pik_info.h"

namespace pik {
namespace {

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
  printf("Read %zu compressed bytes\n", compressed->size());
  fclose(f);
  return true;
}

template<typename ComponentType>
int Decompress(const char* pathname_in, const char* pathname_out) {
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
  if (!LoadFile(pathname_in, &compressed)) {
    return 1;
  }

  DecompressParams params;
  MetaImage<ComponentType> image;
  PikInfo info;
  if (!PikToPixels(params, compressed, &image, &info)) {
    fprintf(stderr, "Failed to decompress.\n");
    return 1;
  }
  printf("Decompressed %zu x %zu pixels.\n", image.xsize(), image.ysize());

  if (!WriteImage(ImageFormatPNG(), image, pathname_out)) {
    fprintf(stderr, "Failed to write %s.\n", pathname_out);
    return 1;
  }

  return 0;
}

}  // namespace
}  // namespace pik

int main(int argc, char** argv) {
  const char* file_in = 0;
  const char* file_out = 0;
  bool arg_error = false;
  bool sixteen_bit = false;

  for (int i = 1; i < argc; i++) {
    if (argv[i][0] == '-') {
      if (strcmp(argv[i], "--16bit") == 0) {
        sixteen_bit = true;
      } else {
        arg_error = true;
        break;
      }
    } else {
      if (!file_in) {
        file_in = argv[i];
      } else if (!file_out) {
        file_out = argv[i];
      } else {
        arg_error = true;
        break;
      }
    }
  }

  if (!file_in || !file_out || arg_error) {
    fprintf(
        stderr,
        "Usage: %s [--16bit] [--dump] in.pik out.png\n"
        "    out.png will have 8 bit per color channel by default,\n"
        "    16 bit per channel if --16bit is set. --dump writes intermediate\n"
        "    files to in.pik.dct etc.\n",
        argv[0]);
    return 1;
  }

  if (sixteen_bit) {
    return pik::Decompress<uint16_t>(file_in, file_out);
  } else {
    return pik::Decompress<uint8_t>(file_in, file_out);
  }
}
