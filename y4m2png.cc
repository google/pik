#include <stdio.h>

#include "image.h"
#include "image_io.h"
#include "yuv_convert.h"

namespace pik {

int Convert(const char* from, const char* to, int bit_depth) {
  Image3B yuv;
  if (!ReadImage(ImageFormatY4M(), from, &yuv)) {
    fprintf(stderr, "Failed to read input file %s\n", from);
    return 1;
  }
  bool ok = false;
  if (bit_depth == 8) {
    ok = WriteImage(ImageFormatPNG(), RGB8ImageFromYUVRec709(yuv), to);
  } else if (bit_depth == 16) {
    ok = WriteImage(ImageFormatPNG(), RGB16ImageFromYUVRec709(yuv), to);
  }
  if (!ok) {
    fprintf(stderr, "Failed to write output file %s\n", to);
    return 1;
  }
  return 0;
}

}  // namespace pik

int PrintArgHelp(int argc, char** argv) {
  fprintf(stderr,
          "Usage: %s <input y4m> <output png> [--bit_depth 8|16]\n",
          argv[0]);
  return 1;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    return PrintArgHelp(argc, argv);
  }
  int bit_depth = 8;
  for (int i = 3; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--bit_depth") {
      if (i + 1 >= argc) {
        return PrintArgHelp(argc, argv);
      }
      std::string param = argv[i + 1];
      if (param == "8") {
        bit_depth = 8;
      } else if (param == "16") {
        bit_depth = 16;
      } else {
        return PrintArgHelp(argc, argv);
      }
      ++i;
    } else {
      return PrintArgHelp(argc, argv);
    }
  }
  return pik::Convert(argv[1], argv[2], bit_depth);
}
