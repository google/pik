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

bool LoadFile(const char* pathname, Bytes* compressed) {
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

// main() function, within namespace for convenience.
int Decompress(const char* pathname_in, const char* pathname_out) {
  Bytes compressed;
  bool failed = false;

  if (!LoadFile(pathname_in, &compressed)) {
    return 1;
  }

  DecompressParams params;
  Image3B planes;
  PikInfo info;
  if (!PikToPixels(params, compressed, &planes, &info)) {
    fprintf(stderr, "Failed to decompress.\n");
    return 1;
  }
  printf("Decompressed %zu x %zu pixels.\n", planes.xsize(), planes.ysize());

  if (ImageFormatPNM::IsExtension(pathname_out)) {
	  failed = WriteImage(ImageFormatPNM(), planes, pathname_out);
  } else
  if (ImageFormatPNG::IsExtension(pathname_out)) {
	  failed = WriteImage(ImageFormatPNG(), planes, pathname_out);
  } else
  if (ImageFormatY4M::IsExtension(pathname_out)) {
	  failed = WriteImage(ImageFormatY4M(), planes, pathname_out);
  } else
  /*if (ImageFormatJPG::IsExtension(pathname_out)) {
	  failed = WriteImage(ImageFormatJPG(), planes, pathname_out);
  } else*/
  if (ImageFormatPlanes::IsExtension(pathname_out)) {
	  failed = WriteImage(ImageFormatPlanes(), planes, pathname_out);
  }

  if (!failed) {
    fprintf(stderr, "Failed to write %s.\n", pathname_out);
    return 1;
  }
  return 0;
}

}  // namespace
}  // namespace pik

int main(int argc, char** argv) {
  if (argc != 3) {
    fprintf(stderr, "Usage: %s in.pik out_rgb_8bit.png\n", argv[0]);
    return 1;
  }

  return pik::Decompress(argv[1], argv[2]);
}
