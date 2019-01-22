// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_PASS_H_
#define PIK_PASS_H_

#include "codec.h"
#include "compressed_image.h"
#include "data_parallel.h"
#include "headers.h"
#include "multipass_handler.h"
#include "padded_bytes.h"
#include "pik_info.h"
#include "pik_params.h"
#include "quantizer.h"
#include "status.h"

// Encode and decode a single pass of an image. A pass can be either a
// decomposition of an image (eg. DC-only pass), or a frame in an animation.
// The behaviour of the (en/de)coder is defined by the given multipass_manager.

namespace pik {

struct PassParams {
  bool is_last;
  FrameInfo frame_info;
};

// These process each group in parallel.

// Encodes an input image `io` in a byte stream, without adding a container.
// `pos` represents the bit position in the output data that we should
// start writing to.
Status PixelsToPikPass(CompressParams params, const PassParams& pass_params,
                       const CodecInOut* io, ThreadPool* pool,
                       PaddedBytes* compressed, size_t& pos, PikInfo* aux_out,
                       MultipassManager* multipass_manager);

// Decodes an input image from a byte stream, using the provided container
// information. See PikToPixels for explanation of `io` color space.
Status PikPassToPixels(const DecompressParams& params,
                       const PaddedBytes& compressed,
                       const FileHeader& container, ThreadPool* pool,
                       BitReader* reader, CodecInOut* io, PikInfo* aux_out,
                       MultipassManager* multipass_manager);

}  // namespace pik

#endif  // PIK_PASS_H_
