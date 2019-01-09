// Copyright 2018 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef BROTLI_H_
#define BROTLI_H_

// Convenience functions for Brotli compression/decompression.

#include <stddef.h>
#include <stdint.h>
#include "compiler_specific.h"
#include "padded_bytes.h"
#include "status.h"

namespace pik {

// Appends to out.
Status BrotliCompress(int quality, const PaddedBytes& in,
                      PaddedBytes* PIK_RESTRICT out);

// Appends to out and ADDS to "bytes_read", which must be pre-initialized.
Status BrotliDecompress(const uint8_t* in, size_t max_input_size,
                        size_t max_output_size, size_t* PIK_RESTRICT bytes_read,
                        PaddedBytes* PIK_RESTRICT out);

// Appends to out and ADDS to "bytes_read", which must be pre-initialized.
Status BrotliDecompress(const PaddedBytes& in, size_t max_output_size,
                        size_t* PIK_RESTRICT bytes_read,
                        PaddedBytes* PIK_RESTRICT out);

}  // namespace pik

#endif  // BROTLI_H_
