// @author Alexander Rhatushnyak

#ifndef LOSSLESS16_H_
#define LOSSLESS16_H_

#include "image.h"
#include "padded_bytes.h"

namespace pik {

bool Grayscale16bit_compress(const ImageU& img, PaddedBytes* bytes);
bool Grayscale16bit_decompress(const PaddedBytes& bytes, size_t* pos,
                               ImageU* result);

}  // namespace pik

#endif  // LOSSLESS16_H_
