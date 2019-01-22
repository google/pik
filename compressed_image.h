// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

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

// Methods to encode (decode) an image into (from) the bit stream:
// initialization of per-pass information and per-group information, actual
// computation of quantized coefficients, and encoding, plus corresponding
// methods for the decoder.

namespace pik {

struct GradientMap;

// Initialize per-pass information.
SIMD_ATTR void InitializePassEncCache(const PassHeader& pass_header,
                                      const Image3F& opsin_full,
                                      const AcStrategyImage& ac_strategy,
                                      const Quantizer& quantizer,
                                      const ColorCorrelationMap& cmap,
                                      ThreadPool* pool,
                                      PassEncCache* pass_enc_cache);

// Initializes the encoder cache, setting parameters from the headers,
// setting up the `coeffs` and `dc_init` images in enc_cache.
SIMD_ATTR void InitializeEncCache(const PassHeader& pass_header,
                                  const GroupHeader& group_header,
                                  const PassEncCache& pass_enc_cache,
                                  const Rect& group_rect, EncCache* enc_cache);

// Computes quantized coefficients from the non-quantized ones already present
// in enc_cache.
SIMD_ATTR void ComputeCoefficients(const Quantizer& quantizer,
                                   const ColorCorrelationMap& cmap,
                                   ThreadPool* pool, EncCache* enc_cache,
                                   MultipassManager* manager,
                                   const PikInfo* aux_out = nullptr);

// Encodes AC quantized coefficients from the given encoder cache.
PaddedBytes EncodeToBitstream(const EncCache& cache, const Rect& rect,
                              const Quantizer& quantizer,
                              const NoiseParams& noise_params,
                              const ColorCorrelationMap& cmap, bool fast_mode,
                              MultipassHandler* handler,
                              PikInfo* info = nullptr);

// Decodes AC coefficients from the bit stream, populating the AC
// fields of the decoder cache, and the corresponding rectangles in the global
// information (quant_field and ac_strategy) in the per-pass decoder cache.
bool DecodeFromBitstream(const PassHeader& pass_header,
                         const GroupHeader& header,
                         const PaddedBytes& compressed, BitReader* reader,
                         const Rect& group_rect, MultipassHandler* handler,
                         const size_t xsize_blocks, const size_t ysize_blocks,
                         const ColorCorrelationMap& cmap,
                         NoiseParams* noise_params, const Quantizer& quantizer,
                         DecCache* cache, PassDecCache* pass_dec_cache);

// Dequantizes the provided quantized_ac image into the decoder cache. Used in
// the encoder loop in adaptive_quantization.cc
void DequantImageAC(const Quantizer& quantizer, const ColorCorrelationMap& cmap,
                    const Image3S& quantized_ac, ThreadPool* pool,
                    DecCache* cache, PassDecCache* pass_dec_cache,
                    const Rect& group_rect);

// Applies predictions to de-quantized AC coefficients, copies DC coefficients
// into AC, and does IDCT. Writes opsin IDCT values into `idct:idct_rect`.
void ReconOpsinImage(const PassHeader& pass_header, const GroupHeader& header,
                     const Quantizer& quantizer, const Rect& block_group_rect,
                     DecCache* cache, PassDecCache* pass_dec_cache,
                     Image3F* PIK_RESTRICT idct, const Rect& idct_rect,
                     PikInfo* pik_info = nullptr);

// Finalizes the decoding of a pass by running per-pass post processing:
// smoothing and adaptive reconstruction. Writes linear sRGB to `linear`.
// TODO(janwas): move NoiseParams into PassHeader.
void FinalizePassDecoding(Image3F&& idct, const PassHeader& pass_header,
                          const NoiseParams& noise_params,
                          const Quantizer& quantizer, ThreadPool* pool,
                          PassDecCache* pass_dec_cache,
                          Image3F* PIK_RESTRICT linear,
                          PikInfo* pik_info = nullptr);

}  // namespace pik

#endif  // COMPRESSED_IMAGE_H_
