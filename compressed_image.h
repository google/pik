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

#ifndef COMPRESSED_IMAGE_H_
#define COMPRESSED_IMAGE_H_

#include <stddef.h>
#include <stdint.h>

#include "adaptive_reconstruction.h"
#include "bit_reader.h"
#include "color_correlation.h"
#include "common.h"
#include "compressed_image_fwd.h"
#include "data_parallel.h"
#include "headers.h"
#include "image.h"
#include "multipass_handler.h"
#include "noise.h"
#include "padded_bytes.h"
#include "pik_info.h"
#include "pik_params.h"
#include "quantizer.h"

namespace pik {

struct GradientMap;

// Reference to |opsin| might be stored -> image should not be modified or
// discarded until the last invocation of ComputeCoefficients.
void ComputeInitialCoefficients(const PassHeader& pass_header,
                                const GroupHeader& group_header,
                                const Image3F& opsin, EncCache* cache);

SIMD_ATTR void ComputeCoefficients(const Quantizer& quantizer,
                                   const ColorCorrelationMap& cmap,
                                   ThreadPool* pool, EncCache* enc_cache,
                                   MultipassManager* manager,
                                   const PikInfo* aux_out = nullptr);

// The gradient may be image with dimension 0 if disabled.
PaddedBytes EncodeToBitstream(const EncCache& cache, const Rect& rect,
                              const Quantizer& quantizer,
                              const NoiseParams& noise_params,
                              const ColorCorrelationMap& cmap, bool fast_mode,
                              MultipassHandler* handler,
                              PikInfo* info = nullptr);

// "compressed" is the same range from which reader was constructed, and allows
// seeking to tiles and constructing per-thread BitReader.
// Writes to (cache->eager_dequant ? cache->dc/ac : cache->quantized_dc/ac).
bool DecodeFromBitstream(const PassHeader& pass_header,
                         const GroupHeader& header,
                         const PaddedBytes& compressed, BitReader* reader,
                         const Rect& group_rect, MultipassHandler* handler,
                         const size_t xsize_blocks, const size_t ysize_blocks,
                         ColorCorrelationMap* cmap, NoiseParams* noise_params,
                         Quantizer* quantizer, DecCache* cache,
                         PassDecCache* pass_dec_cache);

// Dequantizes AC and DC coefficients.
void DequantImage(const Quantizer& quantizer, const ColorCorrelationMap& cmap,
                  ThreadPool* pool, DecCache* cache,
                  PassDecCache* pass_dec_cache, const Rect& group_rect);

// Optionally does DC preconditioning, performs IDCT, and
// optionally applies image post-processing.
Image3F ReconOpsinImage(const PassHeader& pass_header,
                        const GroupHeader& header, const Quantizer& quantizer,
                        const Rect& block_group_rect, DecCache* cache,
                        PassDecCache* pass_dec_cache,
                        PikInfo* pik_info = nullptr);

Image3F FinalizePassDecoding(Image3F&& idct, const PassHeader& pass_header,
                             const Quantizer& quantizer,
                             PassDecCache* pass_dec_cache,
                             PikInfo* pik_info = nullptr);

ImageF IntensityAcEstimate(const ImageF& image, float multiplier,
                           ThreadPool* pool);

}  // namespace pik

#endif  // COMPRESSED_IMAGE_H_
