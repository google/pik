// Copyright 2019 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef COMPRESSED_DC_H_
#define COMPRESSED_DC_H_

#include "color_correlation.h"
#include "compressed_image_fwd.h"
#include "data_parallel.h"
#include "headers.h"
#include "padded_bytes.h"
#include "pik_info.h"
#include "quantizer.h"

// DC handling functions: encoding and decoding of DC to and from bitstream, and
// related function to initialize the per-group decoder cache.

namespace pik {

// Encodes the DC-related information from pass_enc_cache: quantized dc itself
// and gradient map.
PaddedBytes EncodeDC(const Quantizer& quantizer,
                     const PassEncCache& pass_enc_cache, ThreadPool* pool,
                     PikImageSizeInfo* dc_info);

// Decodes and dequantizes DC, and optionally decodes and applies the
// gradient map if requested.
Status DecodeDC(BitReader* reader, const PaddedBytes& compressed,
                const PassHeader& pass_header, size_t xsize_blocks,
                size_t ysize_blocks, const Quantizer& quantizer,
                const ColorCorrelationMap& cmap, ThreadPool* pool,
                PassDecCache* pass_dec_cache);

// Clamps the input coordinate `candidate` to the [0, size) interval, using 1 px
// of border (extended by cloning, not mirroring).
PIK_INLINE size_t SourceCoord(size_t candidate, size_t size) {
  return candidate == 0 ? 0
                        : (candidate == size + 1 ? size - 1 : candidate - 1);
}

// Initializes the dec_cache for decoding the `rect` part of the image (in pixel
// units) from the pass decoder cache.
void InitializeDecCache(const PassDecCache& pass_dec_cache, const Rect& rect,
                        DecCache* dec_cache);

}  // namespace pik

#endif  // COMPRESSED_DC_H_
