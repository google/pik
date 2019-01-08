// Copyright 2017 Google Inc. All Rights Reserved.
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

#ifndef PIK_PASS_H_
#define PIK_PASS_H_

#include "codec.h"
#include "compressed_image.h"
#include "data_parallel.h"
#include "headers.h"
#include "multipass_handler.h"
#include "noise.h"
#include "padded_bytes.h"
#include "pik_info.h"
#include "pik_params.h"
#include "quantizer.h"
#include "status.h"

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
