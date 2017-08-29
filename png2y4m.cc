#include <stdio.h>

#include "image.h"
#include "image_io.h"
#include "yuv_convert.h"

namespace pik {

int Convert(const char* from, const char* to) {
  Image3U rgb16;
  if (!ReadImage(ImageFormatPNG(), from, &rgb16)) {
    fprintf(stderr, "Failed to read input file %s\n", from);
    return 1;
  }
  const Image3B yuv = YUVRec709ImageFromRGB16(rgb16);
  if (!WriteImage(ImageFormatY4M(), yuv, to)) {
    fprintf(stderr, "Failed to write output file %s\n", to);
    return 1;
  }
  return 0;
}

}  // namespace pik

int PrintArgHelp(int argc, char** argv) {
  fprintf(stderr, "Usage: %s <input png> <output y4m>\n", argv[0]);
  return 1;
}

int main(int argc, char** argv) {
  if (argc != 3) {
    return PrintArgHelp(argc, argv);
  }
  return pik::Convert(argv[1], argv[2]);
}
