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

#include <functional>
#include "codec.h"
#include "compressed_image.h"
#include "guetzli/jpeg_data.h"
#include "header.h"
#include "intra_transform.h"
#include "noise.h"
#include "padded_bytes.h"
#include "pik_info.h"
#include "pik_params.h"
#include "quantizer.h"
#include "status.h"

namespace pik {

// Computes a quantizer for the given opsin_orig image according to the passed
// compression parameters. (x/y)size_blocks represent the block size of the
// given image. opsin should be the same image as opsin_orig, but padded to a
// multiple of the block size. pool is the thread pool the computations will be
// executed on. rescale is a coefficient that computed quantization values
// should be multipled by. Other parameters have the same meaning as the
// variables in PixelsToPikFrame.
std::shared_ptr<Quantizer> GetCompressQuantizer(
    const CompressParams& params, size_t xsize_blocks, size_t ysize_blocks,
    const Image3F& opsin_orig, const Image3F& opsin,
    const NoiseParams& noise_params, const Header& header,
    const ColorCorrelationMap& cmap, ThreadPool* pool, PikInfo* aux_out,
    OpsinIntraTransform* transform, double rescale = 1);

// Type for a function that either initializes the quantizer passed as a
// parameter or returns a pointer to an existing quantizer.
using GetQuantizer = std::function<std::shared_ptr<Quantizer>(
    const CompressParams& params, size_t xsize_blocks, size_t ysize_blocks,
    const Image3F& opsin_orig, const Image3F& opsin,
    const NoiseParams& noise_params, const Header& header,
    const ColorCorrelationMap& cmap, ThreadPool* pool, PikInfo* aux_out,
    OpsinIntraTransform* transform)>;

// Encodes an input image (io) in a byte stream, without adding a container.
// get_quantizer is the function that will be called to compute a quantizer for
// the current frame. pos represent the bit position in the output data that we
// should start writing to. pass_info can be used to retrieve the header, the
// header sections, and the start position of the encoded image for the caller's
// use. transform_opsin defines a transformation to be applied to the image
// after it has been converted to Xyb color space, and untrasform_opsin gives
// the reverse transform. Other parameters have the same meaning as in
// PikToPixels.
Status PixelsToPikFrame(
    CompressParams params, const CodecInOut* io,
    const GetQuantizer& get_quantizer, PaddedBytes* compressed, size_t& pos,
    PikInfo* aux_out, OpsinIntraTransform* transform,
    const std::function<void(const Header& header, Sections&& sections,
                             size_t image_start)>& pass_info = nullptr);

// Decodes an input image from a byte stream, using the provided container
// information.
Status PikFrameToPixels(const DecompressParams& params,
                        const PaddedBytes& compressed,
                        const Container& container,
                        const ContainerSections& container_sections,
                        const Header& header, const Sections& sections,
                        BitReader* reader, CodecInOut* io, PikInfo* aux_out,
                        OpsinIntraTransform* transform);

Status PixelsToPik(const CompressParams& params, const CodecInOut* io,
                   PaddedBytes* compressed, PikInfo* aux_out = nullptr);
Status PikToPixels(const DecompressParams& params,
                   const PaddedBytes& compressed, CodecInOut* io,
                   PikInfo* aux_out = nullptr);

// The input image is a (partially decoded) JPEG image.
Status JpegToPik(CodecContext* codec_context, const CompressParams& params,
                 const guetzli::JPEGData& jpeg, PaddedBytes* compressed,
                 PikInfo* aux_out = nullptr);

}  // namespace pik

#endif  // PIK_H_
