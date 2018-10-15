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

#ifndef ENTROPY_CODER_H_
#define ENTROPY_CODER_H_

#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <sys/types.h>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ans_decode.h"
#include "ans_encode.h"
#include "bit_reader.h"
#include "cluster.h"
#include "common.h"
#include "compiler_specific.h"
#include "context_map_encode.h"
#include "fast_log.h"
#include "image.h"
#include "lehmer_code.h"
#include "pik_info.h"
#include "status.h"

namespace pik {

/* Python snippet to generate the zig-zag sequence:
N = 8
out, lut = [0] * (N * N), [0] * (N * N)
x, y, d = 0, 0, 1
for i in range(N * N / 2):
  out[i], out[N * N - 1 - i] = x + y * N, N * N - 1 - x - y * N
  x, y = x + d, y - d
  if y < 0: y, d = 0, -d
  if x < 0: x, d = 0, -d
for i in range(N * N): lut[out[i]] = i
print("Order: " + str(out) + "\nLut: " + str(lut))
*/
// "Natural order" means the order of increasing of "anisotropic" frequency of
// continuous version of DCT basis.
// Surprisingly, frequency along the (i + j == const) diagonals is roughly the
// same. For historical reasons, consequent diagonals are traversed
// in alternating directions - so called "zig-zag" (or "snake") order.
// Round-trip:
//  X = kNaturalCoeffOrderN[kNaturalCoeffOrderLutN[X]]
//  X = kNaturalCoeffOrderLutN[kNaturalCoeffOrderN[X]]
constexpr int32_t kNaturalCoeffOrder8[8 * 8] = {
    0,  1,  8,  16, 9,  2,  3,  10, 17, 24, 32, 25, 18, 11, 4,  5,
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6,  7,  14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63};

constexpr int32_t kNaturalCoeffOrderLut8[8 * 8] = {
    0,  1,  5,  6,  14, 15, 27, 28, 2,  4,  7,  13, 16, 26, 29, 42,
    3,  8,  12, 17, 25, 30, 41, 43, 9,  11, 18, 24, 31, 40, 44, 53,
    10, 19, 23, 32, 39, 45, 52, 54, 20, 22, 33, 38, 46, 51, 55, 60,
    21, 34, 37, 47, 50, 56, 59, 61, 35, 36, 48, 49, 57, 58, 62, 63};

constexpr int32_t kNaturalCoeffOrder16[16 * 16] = {
    0,   1,   16,  32,  17,  2,   3,   18,  33,  48,  64,  49,  34,  19,  4,
    5,   20,  35,  50,  65,  80,  96,  81,  66,  51,  36,  21,  6,   7,   22,
    37,  52,  67,  82,  97,  112, 128, 113, 98,  83,  68,  53,  38,  23,  8,
    9,   24,  39,  54,  69,  84,  99,  114, 129, 144, 160, 145, 130, 115, 100,
    85,  70,  55,  40,  25,  10,  11,  26,  41,  56,  71,  86,  101, 116, 131,
    146, 161, 176, 192, 177, 162, 147, 132, 117, 102, 87,  72,  57,  42,  27,
    12,  13,  28,  43,  58,  73,  88,  103, 118, 133, 148, 163, 178, 193, 208,
    224, 209, 194, 179, 164, 149, 134, 119, 104, 89,  74,  59,  44,  29,  14,
    15,  30,  45,  60,  75,  90,  105, 120, 135, 150, 165, 180, 195, 210, 225,
    240, 241, 226, 211, 196, 181, 166, 151, 136, 121, 106, 91,  76,  61,  46,
    31,  47,  62,  77,  92,  107, 122, 137, 152, 167, 182, 197, 212, 227, 242,
    243, 228, 213, 198, 183, 168, 153, 138, 123, 108, 93,  78,  63,  79,  94,
    109, 124, 139, 154, 169, 184, 199, 214, 229, 244, 245, 230, 215, 200, 185,
    170, 155, 140, 125, 110, 95,  111, 126, 141, 156, 171, 186, 201, 216, 231,
    246, 247, 232, 217, 202, 187, 172, 157, 142, 127, 143, 158, 173, 188, 203,
    218, 233, 248, 249, 234, 219, 204, 189, 174, 159, 175, 190, 205, 220, 235,
    250, 251, 236, 221, 206, 191, 207, 222, 237, 252, 253, 238, 223, 239, 254,
    255};

constexpr int32_t kNaturalCoeffOrderLut16[16 * 16] = {
    0,   1,   5,   6,   14,  15,  27,  28,  44,  45,  65,  66,  90,  91,  119,
    120, 2,   4,   7,   13,  16,  26,  29,  43,  46,  64,  67,  89,  92,  118,
    121, 150, 3,   8,   12,  17,  25,  30,  42,  47,  63,  68,  88,  93,  117,
    122, 149, 151, 9,   11,  18,  24,  31,  41,  48,  62,  69,  87,  94,  116,
    123, 148, 152, 177, 10,  19,  23,  32,  40,  49,  61,  70,  86,  95,  115,
    124, 147, 153, 176, 178, 20,  22,  33,  39,  50,  60,  71,  85,  96,  114,
    125, 146, 154, 175, 179, 200, 21,  34,  38,  51,  59,  72,  84,  97,  113,
    126, 145, 155, 174, 180, 199, 201, 35,  37,  52,  58,  73,  83,  98,  112,
    127, 144, 156, 173, 181, 198, 202, 219, 36,  53,  57,  74,  82,  99,  111,
    128, 143, 157, 172, 182, 197, 203, 218, 220, 54,  56,  75,  81,  100, 110,
    129, 142, 158, 171, 183, 196, 204, 217, 221, 234, 55,  76,  80,  101, 109,
    130, 141, 159, 170, 184, 195, 205, 216, 222, 233, 235, 77,  79,  102, 108,
    131, 140, 160, 169, 185, 194, 206, 215, 223, 232, 236, 245, 78,  103, 107,
    132, 139, 161, 168, 186, 193, 207, 214, 224, 231, 237, 244, 246, 104, 106,
    133, 138, 162, 167, 187, 192, 208, 213, 225, 230, 238, 243, 247, 252, 105,
    134, 137, 163, 166, 188, 191, 209, 212, 226, 229, 239, 242, 248, 251, 253,
    135, 136, 164, 165, 189, 190, 210, 211, 227, 228, 240, 241, 249, 250, 254,
    255};

template <int N>
constexpr const int32_t* NaturalCoeffOrder() {
  return (N == 8) ? kNaturalCoeffOrder8 : kNaturalCoeffOrder16;
}

template <int N>
constexpr const int32_t* NaturalCoeffOrderLut() {
  return (N == 8) ? kNaturalCoeffOrderLut8 : kNaturalCoeffOrderLut16;
}

// Block context used for scanning order, number of non-zeros, AC coefficients.
// 0..2: flat, = channel; 3..5: directional (ignore channel), 6..8: IDENTITY,
// 9..20: DCT16
constexpr uint32_t kFlatOrderContextStart = 0;
constexpr uint32_t kDirectionalOrderContextStart = 3;
constexpr uint32_t kIdentityOrderContextStart = 6;
constexpr uint32_t kDct16OrderContextStart = 9;
constexpr uint32_t kOrderContexts = 21;

// Quantizer values are in range [1..256]. To reduce the total number of
// contexts, the values are shifted and combined in pairs,
// i.e. 1..256 -> 0..127.
constexpr uint32_t kQuantFieldContexts = 128;

// For DCT 8x8 there could be up to 63 non-zero AC coefficients (and one DC
// coefficient). To reduce the total number of contexts,
// the values are combined in pairs, i.e. 0..63 -> 0..31.
constexpr uint32_t kNonZeroBuckets = 32;

// Borrowed from context.h
// TODO(user): find better clustering for PIK use case.
static const uint8_t kCoeffFreqContext[64] = {
    0,  1,  2,  3,  4,  4,  5,  5,  6,  6,  7,  7,  8,  8,  8,  8,
    9,  9,  9,  9,  10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12,
    13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14,
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
};

// Borrowed from context.h
// TODO(user): find better clustering for PIK use case.
static const uint16_t kCoeffNumNonzeroContext[65] = {
    0xBAD, 0,  0,  16, 16, 16, 32, 32, 32, 32, 48, 48, 48, 48, 48, 48, 64,
    64,    64, 64, 64, 64, 64, 64, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79,
    79,    79, 79, 79, 79, 79, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93,
    93,    93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93};

// Borrowed from context.h
// Supremum of ZeroDensityContext(x, y) + 1.
constexpr int kZeroDensityContextCount = 105;

// Borrowed from context.h
// TODO(user): investigate, why disabling pre-clustering makes entropy code
// less dense. Perhaps we would need to add HQ clustering algorithm that would
// be able to squeeze better by spending more CPU cycles.
inline int ZeroDensityContext(int nonzeros_left, int k) {
  PIK_ASSERT(nonzeros_left > 0);
  PIK_ASSERT(nonzeros_left + k < 65);
  return kCoeffNumNonzeroContext[nonzeros_left] + kCoeffFreqContext[k];
}

// Context map consists of 4 blocks:
//  |kQuantFieldContexts|      : context for quantization levels,
//                               computed from top and left values
//  |kOrderContexts x          : context for number of non-zeros in the block
//   kNonZeroBuckets|            computed from block context and predicted value
//                               (based top and left values)
//  |kOrderContexts x          : context for AC coefficient symbols,
//   kZeroDensityContextCount|   computed from block context,
//                               number of non-zeros left and
//                               index in scan order
constexpr uint32_t kNumContexts = kQuantFieldContexts +
                                  (kOrderContexts * kNonZeroBuckets) +
                                  (kOrderContexts * kZeroDensityContextCount);

constexpr uint32_t QuantContext(uint32_t quant) { return (quant - 1) >> 1; }

// Non-zero context is based on number of non-zeros and block context.
// For better clustering, contexts with same number of non-zeros are grouped.
template <int N>
constexpr uint32_t NonZeroContext(uint32_t non_zeros, uint32_t block_ctx);

template <>
constexpr uint32_t NonZeroContext<8>(uint32_t non_zeros, uint32_t block_ctx) {
  return kQuantFieldContexts + kOrderContexts * (non_zeros >> 1) + block_ctx;
}

// Currently we don't have data on 16x16 DCT histograms; reuse existing mapping,
// with input values scaled to match requirements, so that output range is same.
template <>
constexpr uint32_t NonZeroContext<16>(uint32_t non_zeros, uint32_t block_ctx) {
  return NonZeroContext<8>(non_zeros >> 1, block_ctx);
}

// Non-zero context is based on number of non-zeros and block context.
// For better clustering, contexts with same number of non-zeros are grouped.
template <int N>
constexpr uint32_t ZeroDensityContextsOffset(uint32_t block_ctx) {
  // Currently, NonZeroContext<16> returns the same values as NonZeroContext<8>;
  // consequently, offset does not depend on N.
  return kQuantFieldContexts + kOrderContexts * kNonZeroBuckets +
         kZeroDensityContextCount * block_ctx;
}

class AcStrategyType {
 public:
  enum {
    // Strategy types
    // Do not modify AC values, but encode them in the stream - used for
    // neighbours of DCT16 blocks.
    NONE = 0,
    // Regular block size DCT.
    DCT = 1,
    // Leave blocks as they are, encode pixels with no space transforms.
    IDENTITY = 2,
    // Use 16-by-16 DCT
    DCT16X16 = 3,
    // Mask for the strategy types
    MASK = 3,
    // Optional strategy-specific parameters or variations
    // TODO(user): move tile-level strategies somewhere else.
    // Only set for (0, 0)-block of tile. 3-bit value. 0 means no x-resampling.
    RESAMPLE_X = 4,  // 8, 16
    // Only set for (0, 0)-block of tile. 3-bit value. 0 means no y-resampling.
    RESAMPLE_Y = 32,  // 64, 128
  };
  static bool IsNone(uint8_t strategy) { return (strategy & MASK) == NONE; }
  static bool IsDct(uint8_t strategy) { return (strategy & MASK) == DCT; }
  static bool IsIdentity(uint8_t strategy) {
    return (strategy & MASK) == IDENTITY;
  }
  static bool IsDct16x16(uint8_t strategy) {
    return (strategy & MASK) == DCT16X16;
  }
};

// Predicts |rect_dc| (typically a "group" of DC values, or less on the borders)
// within |dc| and stores residuals in |tmp_residuals| starting at 0,0.
void ShrinkDC(const Rect& rect_dc, const Image3S& dc,
              Image3S* PIK_RESTRICT tmp_residuals);

// Reconstructs |rect_dc| within |dc|: replaces these prediction residuals
// (generated by ShrinkDC) with reconstructed DC values. All images are at least
// rect.xsize * ysize (2*xsize for xz); tmp_* start at 0,0. Must be called with
// (one of) the same rect arguments passed to ShrinkDC.
void ExpandDC(const Rect& rect_dc, Image3S* PIK_RESTRICT dc,
              ImageS* PIK_RESTRICT tmp_y, ImageS* PIK_RESTRICT tmp_xz_residuals,
              ImageS* PIK_RESTRICT tmp_xz_expanded);

// Just use zig-zag order for all the contexts.
// This order will have the shortest entropy code.
template <int N>
void ComputeCoeffOrderFast(int32_t* PIK_RESTRICT order);

// Modify zig-zag order, so that DCT bands with more zeros go later.
// Order of DCT bands with same number of zeros is untouched, so
// permutation will be cheaper to encode.
template <int N>
void ComputeCoeffOrder(const Image3S& ac, const Image3B& block_ctx,
                       int32_t* PIK_RESTRICT order);

template <int N>
std::string EncodeCoeffOrders(const int32_t* PIK_RESTRICT order,
                              PikInfo* PIK_RESTRICT pik_info);

// Encodes the "rect" subset of "img".
// Typically used for DC.
// See also DecodeImageData.
std::string EncodeImageData(const Rect& rect, const Image3S& img,
                            PikImageSizeInfo* info);

struct Token {
  Token(uint32_t c, uint32_t s, uint32_t nb, uint32_t b)
      : context(c), bits(b), nbits(nb), symbol(s) {}
  uint32_t context;
  uint16_t bits;
  uint8_t nbits;
  uint8_t symbol;
};

// Generate quantization field tokens.
// Only the subset "rect" [in units of blocks] within all images.
// Appends one token per pixel to output.
// TODO(user): quant field seems to be useful for all the AC strategies.
// perhaps, we could just have different quant_ctx based on the block type.
// See also DecodeQuantField.
void TokenizeQuantField(const Rect& rect, const ImageI& quant_field,
                        const ImageB& ac_strategy,
                        std::vector<Token>* PIK_RESTRICT output);

// Generate DCT NxN quantized AC values tokens.
// Only the subset "rect" [in units of blocks] within all images.
// Warning: uses the DC coefficients in "coeffs"!
// See also DecodeCoefficients.
template <int N>
void TokenizeCoefficients(const int32_t* orders, const Rect& rect,
                          const Image3S& coeffs, const Image3B& block_ctx,
                          const ImageB& ac_strategy,
                          std::vector<Token>* PIK_RESTRICT output);

std::string BuildAndEncodeHistograms(
    size_t num_contexts, const std::vector<std::vector<Token> >& tokens,
    std::vector<ANSEncodingData>* codes, std::vector<uint8_t>* context_map,
    PikImageSizeInfo* info);

template <int N>
std::string BuildAndEncodeHistogramsFast(
    const std::vector<std::vector<Token> >& tokens,
    std::vector<ANSEncodingData>* codes, std::vector<uint8_t>* context_map,
    PikImageSizeInfo* info);

std::string WriteTokens(const std::vector<Token>& tokens,
                        const std::vector<ANSEncodingData>& codes,
                        const std::vector<uint8_t>& context_map,
                        PikImageSizeInfo* pik_info);

template <int N>
bool DecodeCoeffOrder(int32_t* order, bool decode_first, BitReader* br);

bool DecodeHistograms(BitReader* br, const size_t num_contexts,
                      const size_t max_alphabet_size, ANSCode* code,
                      std::vector<uint8_t>* context_map);

// Decodes into "rect" within "img".
bool DecodeImage(BitReader* PIK_RESTRICT br, const Rect& rect,
                 Image3S* PIK_RESTRICT img);

// Decode quantization field.
// See also TokenizeQuantField.
bool DecodeQuantField(BitReader* PIK_RESTRICT br,
                      ANSSymbolReader* PIK_RESTRICT decoder,
                      const std::vector<uint8_t>& context_map,
                      const Rect& rect_qf, const ImageB& PIK_RESTRICT block_map,
                      uint8_t block_map_mask, ImageI* PIK_RESTRICT quant_field);

// Decode DCT NxN quantized AC values.
// DC component in ac's DCT blocks is invalid.
// See also TokenizeCoefficients.
template <int N>
bool DecodeCoefficients(
    BitReader* PIK_RESTRICT br, ANSSymbolReader* PIK_RESTRICT decoder,
    const Image3B& tmp_block_ctx, const std::vector<uint8_t>& context_map,
    const int32_t* PIK_RESTRICT coeff_order, const Rect& rect_ac,
    Image3S* PIK_RESTRICT ac, const Rect& rect_bm,
    const ImageB& PIK_RESTRICT block_map, uint8_t block_map_mask,
    Image3I* PIK_RESTRICT tmp_num_nzeroes);

// Validate parameters and invoke DecodeQuantField and DecodeCoefficients.
// "rect_ac/qf" are in blocks.
template <int N>
bool DecodeAC(const Image3B& tmp_block_ctx, const ANSCode& code,
              const std::vector<uint8_t>& context_map,
              const int32_t* PIK_RESTRICT coeff_order,
              BitReader* PIK_RESTRICT br, const Rect& rect_ac,
              Image3S* PIK_RESTRICT ac, const Rect& rect_qf,
              ImageI* PIK_RESTRICT quant_field,
              const ImageB& PIK_RESTRICT ac_strategy,
              Image3I* PIK_RESTRICT tmp_num_nzeroes);

// Encodes non-negative (X) into (2 * X), negative (-X) into (2 * X - 1)
constexpr uint32_t PackSigned(int32_t value) {
  return ((uint32_t)value << 1) ^ (((uint32_t)(~value) >> 31) - 1);
}

// Reverse to PackSigned, i.e. UnpackSigned(PackSigned(X)) == X.
constexpr int32_t UnpackSigned(uint32_t value) {
  return (int32_t)((value >> 1) ^ (((~value) & 1) - 1));
}

// Encode non-negative integer as a pair (N, bits), where len(bits) == N.
// 0 is encoded as (0, ''); X from range [2**N - 1, 2 * (2**N - 1)]
// is encoded as (N, X + 1 - 2**N).
static PIK_INLINE void EncodeVarLenUint(uint32_t value, int* PIK_RESTRICT nbits,
                                        int* PIK_RESTRICT bits) {
  if (value == 0) {
    *nbits = 0;
    *bits = 0;
  } else {
    int len = Log2FloorNonZero(value + 1);
    *nbits = len;
    *bits = (value + 1) & ((1 << len) - 1);
  }
}

// Decode variable length non-negative value. Reverse to EncodeVarLenUint.
constexpr uint32_t DecodeVarLenUint(int nbits, int bits) {
  return (1u << nbits) + bits - 1;
}

// Decode value and unpack signed integer.
constexpr int32_t DecodeVarLenInt(int nbits, int bits) {
  return UnpackSigned(DecodeVarLenUint(nbits, bits));
}

// Pack signed integer and encode value.
static PIK_INLINE void EncodeVarLenInt(int32_t value, int* PIK_RESTRICT nbits,
                                       int* PIK_RESTRICT bits) {
  EncodeVarLenUint(PackSigned(value), nbits, bits);
}

}  // namespace pik

#endif  // ENTROPY_CODER_H_
