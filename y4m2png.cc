#include <stdio.h>

#include "image.h"
#include "image_io.h"
#include "yuv_convert.h"
#include "yuv_opsin_convert.h"

namespace pik {

int Convert(const char* from, const char* to,
            int out_bit_depth, bool use_opsin) {
  Image3U yuv;
  int in_bit_depth;
  if (!ReadImage(ImageFormatY4M(), from, &yuv, &in_bit_depth)) {
    fprintf(stderr, "Failed to read input file %s\n", from);
    return 1;
  }
  bool ok = false;
  if (out_bit_depth == 8) {
    const Image3B rgb8 = use_opsin ?
        RGB8ImageFromYUVOpsin(yuv, in_bit_depth) :
        RGB8ImageFromYUVRec709(yuv, in_bit_depth);
    ok = WriteImage(ImageFormatPNG(), rgb8, to);
  } else if (out_bit_depth == 16) {
    const Image3U rgb16 = use_opsin ?
        RGB16ImageFromYUVOpsin(yuv, in_bit_depth) :
        RGB16ImageFromYUVRec709(yuv, in_bit_depth);
    ok = WriteImage(ImageFormatPNG(), rgb16, to);
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
  bool use_opsin = false;
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
    } else if (arg == "--opsin") {
      use_opsin = true;
    } else {
      return PrintArgHelp(argc, argv);
    }
  }
  return pik::Convert(argv[1], argv[2], bit_depth, use_opsin);
}
