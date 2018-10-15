// Copyright 2018 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
