#include <stdio.h>

#include "butteraugli_distance.h"
#include "codec.h"
#include "image.h"
#include "status.h"

namespace pik {
namespace {

Status RunButteraugli(const char* pathname1, const char* pathname2) {
  CodecContext codec_context;
  CodecInOut io1(&codec_context);
  ThreadPool pool(4);
  if (!io1.SetFromFile(pathname1, &pool)) {
    fprintf(stderr, "Failed to read image from %s\n", pathname1);
    return false;
  }

  CodecInOut io2(&codec_context);
  if (!io2.SetFromFile(pathname2, &pool)) {
    fprintf(stderr, "Failed to read image from %s\n", pathname2);
    return false;
  }

  if (io1.xsize() != io2.xsize()) {
    fprintf(stderr, "Width mismatch: %zu %zu\n", io1.xsize(), io2.xsize());
    return false;
  }
  if (io1.ysize() != io2.ysize()) {
    fprintf(stderr, "Height mismatch: %zu %zu\n", io1.ysize(), io2.ysize());
    return false;
  }

  const float kHfAsymmetry = 0.8;
  const float distance =
      ButteraugliDistance(&io1, &io2, kHfAsymmetry, /*distmap=*/nullptr, &pool);
  printf("%.10f\n", distance);
  return true;
}

}  // namespace
}  // namespace pik

int main(int argc, char** argv) {
  if (argc != 3) {
    fprintf(stderr, "Usage: %s <reference> <distorted>\n", argv[0]);
    return 1;
  }
  return pik::RunButteraugli(argv[1], argv[2]) ? 0 : 1;
}
