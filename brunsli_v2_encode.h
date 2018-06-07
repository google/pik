#ifndef BRUNSLI_V2_ENCODE_H_
#define BRUNSLI_V2_ENCODE_H_

#include <stddef.h>
#include <stdint.h>
#include <vector>

#include "guetzli/jpeg_data.h"
#include "padded_bytes.h"

namespace pik {

// Used by callers of BrunsliV2EncodeJpegData.
size_t BrunsliV2MaximumEncodedSize(const guetzli::JPEGData& jpg);

// Encodes the given jpg to *out in brunsli v2 format starting at byte
// offset "header_size". Returns false on invalid jpg data.
// "out" must be pre-allocated to MaxHeaderSize + BrunsliV2MaximumEncodedSize.
bool BrunsliV2EncodeJpegData(const guetzli::JPEGData& jpg,
                             const size_t header_size, PaddedBytes* out);

}  // namespace pik

#endif  // BRUNSLI_V2_ENCODE_H_
