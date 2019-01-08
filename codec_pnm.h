// Copyright 2018 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Disclaimer: This is not an official Google product.

#ifndef CODEC_PNM_H_
#define CODEC_PNM_H_

// Encodes/decodes PGM/PPM/PFM pixels in memory.

#include "codec.h"
#include "color_management.h"
#include "data_parallel.h"
#include "padded_bytes.h"

namespace pik {

// Decodes "bytes" and transforms to io->c_current color space. io->dec_hints
// may specify "color_space" and "range" (defaults are sRGB and full-range).
Status DecodeImagePNM(const PaddedBytes& bytes, ThreadPool* pool,
                      CodecInOut* io);

// Transforms from io->c_current to io->c_external and encodes into "bytes".
Status EncodeImagePNM(const CodecInOut* io, const ColorEncoding& c_desired,
                      size_t bits_per_sample, ThreadPool* pool,
                      PaddedBytes* bytes);

}  // namespace pik

#endif  // CODEC_PNM_H_
