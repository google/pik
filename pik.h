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

#ifndef PIK_H_
#define PIK_H_

#include "codec.h"
#include "compressed_image.h"
#include "data_parallel.h"
#include "headers.h"
#include "multipass_handler.h"
#include "noise.h"
#include "padded_bytes.h"
#include "pik_info.h"
#include "pik_params.h"
#include "pik_pass.h"
#include "quantizer.h"
#include "status.h"

namespace pik {

// Compresses pixels from `io` (given in any ColorEncoding).
// `io` must have original_bits_per_sample and dec_c_original fields set.
Status PixelsToPik(const CompressParams& params, const CodecInOut* io,
                   PaddedBytes* compressed, PikInfo* aux_out = nullptr,
                   ThreadPool* pool = nullptr);

// Implementation detail: currently decodes to linear sRGB. The contract is:
// `io` appears 'identical' (modulo compression artifacts) to the encoder input
// in a color-aware viewer. Note that `io`->dec_c_original identifies the color
// space that was passed to the encoder; clients that need that encoding must
// call `io`->TransformTo afterwards.
Status PikToPixels(const DecompressParams& params,
                   const PaddedBytes& compressed, CodecInOut* io,
                   PikInfo* aux_out = nullptr, ThreadPool* pool = nullptr);

}  // namespace pik

#endif  // PIK_H_
