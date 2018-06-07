#include <stdio.h>

#include "image.h"
#include "image_io.h"
#include "butteraugli_distance.h"

static const float kHfAsymmetry = 3.14;

int PrintArgHelp(int argc, char** argv) {
  fprintf(stderr, "Usage: %s <image a> <image b>\n", argv[0]);
  return 1;
}

int main(int argc, char** argv) {
  if (argc != 3) {
    return PrintArgHelp(argc, argv);
  }

  pik::MetaImageF a = pik::ReadMetaImageLinear(argv[1]);
  if (a.xsize() == 0) {
    fprintf(stderr, "Failed to read image from %s\n", argv[1]);
    return 1;
  }

  pik::MetaImageF b = pik::ReadMetaImageLinear(argv[2]);
  if (b.xsize() == 0) {
    fprintf(stderr, "Failed to read image from %s\n", argv[2]);
    return 1;
  }

  if (a.xsize() != b.xsize()) {
    fprintf(stderr, "%s and %s have different widths\n", argv[1], argv[2]);
    return 1;
  }
  if (a.ysize() != b.ysize()) {
    fprintf(stderr, "%s and %s have different heights\n", argv[1], argv[2]);
    return 1;
  }

  float distance = pik::ButteraugliDistance(a, b, kHfAsymmetry, nullptr);
  printf("%.10f\n", distance);

  return 0;
}
