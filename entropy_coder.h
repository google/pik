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

#include "ac_strategy.h"
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
#include "multipass_handler.h"
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

constexpr const int32_t* NaturalCoeffOrder() { return kNaturalCoeffOrder8; }

constexpr const int32_t* NaturalCoeffOrderLut() {
  return kNaturalCoeffOrderLut8;
}

// Block context used for scanning order, number of non-zeros, AC coefficients.
// Equal to the channel.
constexpr uint32_t kDCTOrderContextStart = 0;
constexpr uint32_t kOrderContexts = 3;

// Quantizer values are in range [1..256]. To reduce the total number of
// contexts, the values are shifted and combined in pairs,
// i.e. 1..256 -> 0..127.
constexpr uint32_t kQuantFieldContexts = 1;

// AC strategy contexts.
constexpr uint32_t kAcStrategyContexts = 1;

// Total number of order-free contextes.
constexpr uint32_t kNumOrderFreeContexts =
    kQuantFieldContexts + kAcStrategyContexts;

// For DCT 8x8 there could be up to 63 non-zero AC coefficients (and one DC
// coefficient). To reduce the total number of contexts,
// the values are combined in pairs, i.e. 0..63 -> 0..31.
constexpr uint32_t kNonZeroBuckets = 32;

// TODO(user): find better clustering for PIK use case.
static const uint8_t kCoeffFreqContext[64] = {
    0,  1,  2,  3,  4,  4,  5,  5,  6,  6,  7,  7,  8,  8,  8,  8,
    9,  9,  9,  9,  10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12,
    13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14,
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
};

// TODO(user): find better clustering for PIK use case.
static const uint16_t kCoeffNumNonzeroContext[65] = {
    0xBAD, 0,  0,  16, 16, 16, 32, 32, 32, 32, 48, 48, 48, 48, 48, 48, 64,
    64,    64, 64, 64, 64, 64, 64, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79,
    79,    79, 79, 79, 79, 79, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93,
    93,    93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93};

// Supremum of ZeroDensityContext(x, y) + 1.
constexpr int kZeroDensityContextCount = 105;

/* This function is used for entropy-sources pre-clustering.
 *
 * Ideally, each combination of |nonzeros_left| and |k| should go to its own
 * bucket; but it implies (64 * 63 / 2) == 2016 buckets. If there is other
 * dimension (e.g. block context), then number of primary clusters becomes too
 * big.
 *
 * To solve this problem, |nonzeros_left| and |k| values are clustered. It is
 * known that their sum is at most 64, consequently, the total number buckets
 * is at most A(64) * B(64).
 *
 * |bits| controls the granularity of pre-clustering. When |bits| is 0, all |k|
 * values are put together. When |bits| is 6, then all 64 |k| values go to
 * different buckets.
 *
 * Also see the test code, where more compact presentation is expanded into
 * those lookup tables.
 */
// TODO(user): investigate, why disabling pre-clustering makes entropy code
// less dense. Perhaps we would need to add HQ clustering algorithm that would
// be able to squeeze better by spending more CPU cycles.
inline int ZeroDensityContext(int nonzeros_left, int k) {
  PIK_ASSERT(nonzeros_left > 0);
  PIK_ASSERT(nonzeros_left + k < 65);
  return kCoeffNumNonzeroContext[nonzeros_left] + kCoeffFreqContext[k];
}

// Context map consists of 3 blocks:
//  |kNumOrderFreeContexts|    : context for residuals of quantization levels,
//                               and AC strategy.
//  |kOrderContexts x          : context for number of non-zeros in the block
//   kNonZeroBuckets|            computed from block context and predicted value
//                               (based top and left values)
//  |kOrderContexts x          : context for AC coefficient symbols,
//   kZeroDensityContextCount|   computed from block context,
//                               number of non-zeros left and
//                               index in scan order
constexpr uint32_t kNumContexts = kNumOrderFreeContexts +
                                  (kOrderContexts * kNonZeroBuckets) +
                                  (kOrderContexts * kZeroDensityContextCount);

constexpr uint32_t AcStrategyContext() { return 0; }

constexpr uint32_t QuantContext() { return kAcStrategyContexts; }

// Non-zero context is based on number of non-zeros and block context.
// For better clustering, contexts with same number of non-zeros are grouped.
constexpr uint32_t NonZeroContext(uint32_t non_zeros, uint32_t block_ctx) {
  return kNumOrderFreeContexts + kOrderContexts * (non_zeros >> 1) + block_ctx;
}

// Non-zero context is based on number of non-zeros and block context.
// For better clustering, contexts with same number of non-zeros are grouped.
constexpr uint32_t ZeroDensityContextsOffset(uint32_t block_ctx) {
  return kNumOrderFreeContexts + kOrderContexts * kNonZeroBuckets +
         kZeroDensityContextCount * block_ctx;
}

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

// Modify zig-zag order, so that DCT bands with more zeros go later.
// Order of DCT bands with same number of zeros is untouched, so
// permutation will be cheaper to encode.
void ComputeCoeffOrder(const Image3S& ac, const Rect& rect,
                       int32_t* PIK_RESTRICT order);

std::string EncodeCoeffOrders(const int32_t* PIK_RESTRICT order,
                              PikInfo* PIK_RESTRICT pik_info);

// Encodes the "rect" subset of "img".
// Typically used for DC.
// See also DecodeImageData.
std::string EncodeImageData(const Rect& rect, const Image3S& img,
                            PikImageSizeInfo* info);

struct Token {
  Token(uint32_t c, uint32_t s, uint32_t nb, uint32_t b)
      : bits(b), context(c), nbits(nb), symbol(s) {
#ifdef ADDRESS_SANITIZER
    PIK_ASSERT(c < (1UL << 16));
#endif
    static_assert(sizeof(Token) == 8, "Token must be a 8 byte struct!");
  }
  uint32_t bits;
  uint16_t context;
  uint8_t nbits;
  uint8_t symbol;
};

// Generate AC strategy tokens.
// Only the subset "rect" [in units of blocks] within all images.
// Appends one token per pixel to output.
// See also DecodeAcStrategy.
void TokenizeAcStrategy(const Rect& rect, const AcStrategyImage& ac_strategy,
                        const AcStrategyImage* hint,
                        std::vector<Token>* PIK_RESTRICT output);

// Generate quantization field tokens.
// Only the subset "rect" [in units of blocks] within all images.
// Appends one token per pixel to output.
// TODO(user): quant field seems to be useful for all the AC strategies.
// perhaps, we could just have different quant_ctx based on the block type.
// See also DecodeQuantField.
void TokenizeQuantField(const Rect& rect, const ImageI& quant_field,
                        const ImageI* hint, const AcStrategyImage& ac_strategy,
                        std::vector<Token>* PIK_RESTRICT output);

// Generate DCT NxN quantized AC values tokens.
// Only the subset "rect" [in units of blocks] within all images.
// Warning: uses the DC coefficients in "coeffs"!
// See also DecodeCoefficients.
void TokenizeCoefficients(const int32_t* orders, const Rect& rect,
                          const Image3S& coeffs,
                          std::vector<Token>* PIK_RESTRICT output);

// Decode AC strategy. The `rect` argument does *not* apply to the hint!
// See also TokenizeAcStrategy.
bool DecodeAcStrategy(BitReader* PIK_RESTRICT br,
                      ANSSymbolReader* PIK_RESTRICT decoder,
                      const std::vector<uint8_t>& context_map, const Rect& rect,
                      AcStrategyImage* PIK_RESTRICT ac_strategy,
                      const AcStrategyImage* PIK_RESTRICT hint);

std::string BuildAndEncodeHistograms(
    size_t num_contexts, const std::vector<std::vector<Token> >& tokens,
    std::vector<ANSEncodingData>* codes, std::vector<uint8_t>* context_map,
    PikImageSizeInfo* info);

std::string BuildAndEncodeHistogramsFast(
    const std::vector<std::vector<Token> >& tokens,
    std::vector<ANSEncodingData>* codes, std::vector<uint8_t>* context_map,
    PikImageSizeInfo* info);

std::string WriteTokens(const std::vector<Token>& tokens,
                        const std::vector<ANSEncodingData>& codes,
                        const std::vector<uint8_t>& context_map,
                        PikImageSizeInfo* pik_info);

bool DecodeCoeffOrder(int32_t* order, BitReader* br);

bool DecodeHistograms(BitReader* br, const size_t num_contexts,
                      const size_t max_alphabet_size, ANSCode* code,
                      std::vector<uint8_t>* context_map);

// Decodes into "rect" within "img".
bool DecodeImage(BitReader* PIK_RESTRICT br, const Rect& rect,
                 Image3S* PIK_RESTRICT img);

bool DecodeQuantField(BitReader* PIK_RESTRICT br,
                      ANSSymbolReader* PIK_RESTRICT decoder,
                      const std::vector<uint8_t>& context_map,
                      const Rect& rect_qf,
                      const AcStrategyImage& PIK_RESTRICT ac_strategy,
                      ImageI* PIK_RESTRICT quant_field,
                      const ImageI* PIK_RESTRICT hint);

// Decode DCT NxN quantized AC values.
// DC component in ac's DCT blocks is invalid.
// Decodes to ac; `rect` is used only for size information.
bool DecodeAC(const std::vector<uint8_t>& context_map,
              const int32_t* PIK_RESTRICT coeff_order,
              BitReader* PIK_RESTRICT br, ANSSymbolReader* decoder,
              Image3S* PIK_RESTRICT ac, const Rect& rect,
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
