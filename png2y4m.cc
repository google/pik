#include <stdio.h>

#include "image.h"
#include "image_io.h"
#include "yuv_convert.h"
#include "yuv_opsin_convert.h"

namespace pik {

int Convert(const char* from, const char* to, int bit_depth, bool use_opsin) {
  Image3U rgb16;
  if (!ReadImage(ImageFormatPNG(), from, &rgb16)) {
    fprintf(stderr, "Failed to read input file %s\n", from);
    return 1;
  }
  Image3U yuv = use_opsin ?
      YUVOpsinImageFromRGB16(rgb16, bit_depth) :
      YUVRec709ImageFromRGB16(rgb16, bit_depth);
  if (!WriteImage(ImageFormatY4M(bit_depth), yuv, to)) {
    fprintf(stderr, "Failed to write output file %s\n", to);
    return 1;
  }
  return 0;
}

}  // namespace pik

int PrintArgHelp(int argc, char** argv) {
  fprintf(stderr, "Usage: %s <input png> <output y4m> [--bit_depth 8|10|12]\n",
          argv[0]);
  return 1;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    return PrintArgHelp(argc, argv);
  }
  int bit_depth = 8;
  int use_opsin = false;
  for (int i = 3; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--bit_depth") {
      if (i + 1 >= argc) {
        return PrintArgHelp(argc, argv);
      }
      std::string param = argv[i + 1];
      if (param == "8") {
        bit_depth = 8;
      } else if (param == "10") {
        bit_depth = 10;
      } else if (param == "12") {
        bit_depth = 12;
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
