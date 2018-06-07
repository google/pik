#ifndef BRUNSLI_V2_COMMON_H_
#define BRUNSLI_V2_COMMON_H_

#include <stdint.h>

namespace pik {

static const uint8_t kBrunsliHeaderMarker = 0x12;
static const uint8_t kBrunsliQuantDataMarker = 0x2a;
static const uint8_t kBrunsliHistogramDataMarker = 0x32;
static const uint8_t kBrunsliDCDataMarker = 0x3a;
static const uint8_t kBrunsliACDataMarker = 0x42;

}  // namespace pik

#endif  // BRUNSLI_V2_COMMON_H_
