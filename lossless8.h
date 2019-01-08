// @author Alexander Rhatushnyak

#ifndef LOSSLESS8_H_
#define LOSSLESS8_H_

#include "image.h"
#include "padded_bytes.h"

namespace pik {

bool Grayscale8bit_compress(const ImageB& img, PaddedBytes* bytes);
bool Grayscale8bit_decompress(const PaddedBytes& bytes, size_t* pos,
                              ImageB* result);

}  // namespace pik

#endif  // LOSSLESS8_H_
