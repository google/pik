#ifndef BRUNSLI_V2_DECODE_H_
#define BRUNSLI_V2_DECODE_H_

#include <stdint.h>
#include <cstddef>

#include "guetzli/jpeg_data.h"
#include "status.h"

namespace pik {

// Parses the brunsli v2 byte stream contained in data[0 ... len) and fills in
// *jpg with the parsed information.
// The *jpg object is valid only as long as the input data is valid.
// Returns true, unless the data is not valid brunsli v2 byte stream, or is
// truncated.
Status BrunsliV2DecodeJpegData(const uint8_t* data, const size_t len,
                               guetzli::JPEGData* jpg);

}  // namespace pik

#endif  // BRUNSLI_V2_DECODE_H_
