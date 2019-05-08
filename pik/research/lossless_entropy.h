// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_LOSSLESS_ENTROPY_H_
#define PIK_LOSSLESS_ENTROPY_H_

#include <vector>

#include "pik/base/padded_bytes.h"

namespace pik {

bool EncodeVarInt(uint64_t value, size_t output_size, size_t* output_pos,
                  uint8_t* output);
void EncodeVarInt(uint64_t value, PaddedBytes* data);
size_t EncodeVarInt(uint64_t value, uint8_t* output);

uint64_t DecodeVarInt(const uint8_t* input, size_t inputSize, size_t* pos);

// TODO(janwas): output to PaddedBytes for compatibility with brotli.h.

// Compresses the source data using MaybeEntropyEncode and handles the RLE
// or uncompressible cases as well.
// Reads starting from &src[0] and writes starting at &dst[*pos], and updates
// pos to point to the end of the written data in dst.
// The src and dst buffers are allowed to overlap
// Returns an error if the dst_capacity was not large enough to hold the
// compressed data. The dst_capacity must at least be slightly larger than the
// src_size to account for the case of data that doesn't compress well.
bool CompressWithEntropyCode(size_t* pos, size_t src_size, const uint8_t* src,
                             size_t dst_capacity, uint8_t* dst);

// Decompresses the data compressed with compressWithEntropyCode
// ds = decompressed size output
// pos = position in the compressed src buffer
bool DecompressWithEntropyCode(uint8_t* dst, size_t dst_capacity,
                               const uint8_t* src, size_t src_capacity,
                               size_t* ds, size_t* pos);

// Compresses multiple independent streams with the entropy code.
bool CompressWithEntropyCode(size_t* pos, const size_t* src_size,
                             const uint8_t* const* src, size_t num_src,
                             size_t dst_capacity, uint8_t* dst);

// Compresses multiple independent streams with the entropy code.
bool DecompressWithEntropyCode(size_t src_capacity, const uint8_t* src,
                               size_t max_decompressed_size, size_t num_dst,
                               std::vector<uint8_t>* dst, size_t* pos);
}  // namespace pik

#endif  // PIK_LOSSLESS_ENTROPY_H_
