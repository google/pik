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

#include "opsin_codec.h"

#include <stdint.h>
#include <string.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include "ans_params.h"
#include "bit_reader.h"
#include "common.h"
#include "compiler_specific.h"
#include "context_map_decode.h"
#include "dc_predictor.h"
#include "dc_predictor_slow.h"
#include "fast_log.h"
#include "huffman_encode.h"
#include "status.h"
#include "write_bits.h"

namespace pik {

static inline int SymbolFromSignedInt(int diff) {
  return diff >= 0 ? 2 * diff : -2 * diff - 1;
}

static inline int SignedIntFromSymbol(int symbol) {
  return symbol % 2 == 0 ? symbol / 2 : (-symbol - 1) / 2;
}

void PredictDCBlock(size_t x, size_t y, size_t xsize, size_t row_stride,
                    std::array<const int16_t* PIK_RESTRICT, 3> row_in,
                    std::array<int16_t* PIK_RESTRICT, 3> row_out) {
  const int x_in = x * 64;
  int16_t pred = MinCostPredict(&row_in[1][x_in], x, y, xsize,
                                -64, -row_stride, false, 0);
  row_out[1][x] = row_in[1][x_in] - pred;
  int uv_predictor = GetUVPredictor(&row_in[1][x_in], x, y, xsize,
                                    -64, -row_stride);
  for (int c = 0; c < 3; c += 2) {
    int16_t pred = MinCostPredict(&row_in[c][x_in], x, y, xsize,
                                  -64, -row_stride, true, uv_predictor);
    row_out[c][x] = row_in[c][x_in] - pred;
  }
}

void PredictDCTile(const Image3S& coeffs, Image3S* out) {
  PIK_ASSERT(coeffs.ysize() == out->ysize());
  PIK_ASSERT(coeffs.xsize() == 64 * out->xsize());
  const size_t row_stride = coeffs.plane(0).bytes_per_row() / sizeof(int16_t);
  for (int y = 0; y < out->ysize(); y++) {
    auto row_in = coeffs.Row(y);
    auto row_out = out->Row(y);
    for (int x = 0; x < out->xsize(); x++) {
      PredictDCBlock(x, y, out->xsize(), row_stride, row_in, row_out);
    }
  }
}

void UnpredictDCTile(Image3S* coeffs) {
  PIK_ASSERT(coeffs->xsize() % 64 == 0);
  ImageS dc_y(coeffs->xsize() / 64, coeffs->ysize());
  ImageS dc_xz(coeffs->xsize() / 64 * 2, coeffs->ysize());

  for (int y = 0; y < coeffs->ysize(); y++) {
    auto row = coeffs->Row(y);
    auto row_y = dc_y.Row(y);
    auto row_xz = dc_xz.Row(y);
    for (int x = 0, block_x = 0; x < coeffs->xsize(); x += 64, block_x++) {
      row_y[block_x] = row[1][x];

      row_xz[2 * block_x] = row[0][x];
      row_xz[2 * block_x + 1] = row[2][x];
    }
  }

  ImageS dc_y_out(coeffs->xsize() / 64, coeffs->ysize());
  ImageS dc_xz_out(coeffs->xsize() / 64 * 2, coeffs->ysize());

  ExpandY(dc_y, &dc_y_out);
  ExpandUV(dc_y_out, dc_xz, &dc_xz_out);

  for (int y = 0; y < coeffs->ysize(); y++) {
    auto row_y = dc_y_out.Row(y);
    auto row_xz = dc_xz_out.Row(y);
    auto row_out = coeffs->Row(y);
    for (int x = 0, block_x = 0; x < coeffs->xsize(); x += 64, block_x++) {
      row_out[1][x] = row_y[block_x];

      row_out[0][x] = row_xz[2 * block_x];
      row_out[2][x] = row_xz[2 * block_x + 1];
    }
  }
}

PIK_INLINE size_t RoundToBytes(size_t num_bits, int lg2_byte_alignment) {
  const size_t bias = (8 << lg2_byte_alignment) - 1;
  return ((1 << lg2_byte_alignment) *
          ((num_bits + bias) >> (lg2_byte_alignment + 3)));
}

size_t HistogramBuilder::EncodedSize(
    int lg2_histo_align, int lg2_data_align) const {
  size_t total_histogram_bits = 0;
  size_t total_data_bits = num_extra_bits();
  for (int c = 0; c < histograms_.size(); ++c) {
    size_t histogram_bits;
    size_t data_bits;
    BuildHuffmanTreeAndCountBits(histograms_[c].data_.data(),
                                 histograms_[c].data_.size(),
                                 &histogram_bits, &data_bits);
    total_histogram_bits += histogram_bits;
    total_data_bits += data_bits;
  }
  if (lg2_histo_align >= 0) {
    return (RoundToBytes(total_histogram_bits, lg2_histo_align) +
            RoundToBytes(total_data_bits, lg2_data_align));
  } else {
    return RoundToBytes(total_histogram_bits + total_data_bits,
                        lg2_data_align);
  }
}

void ComputeCoeffOrder(const Image3S& img, const Image3B& block_ctx,
                       int* order) {
  for (int ctx = 0; ctx < kOrderContexts; ++ctx) {
    int num_zeros[64] = { 0 };
    for (int y = 0; y < img.ysize(); ++y) {
      auto row = img.Row(y);
      for (int x = 0; x < img.xsize(); x += 64) {
        for (int c = 0; c < 3; ++c) {
          if (block_ctx.Row(y)[c][x >> 6] != ctx) continue;
          for (int k = 1; k < 64; ++k) {
            if (row[c][x + k] == 0) ++num_zeros[k];
          }
        }
      }
    }
    std::vector<std::pair<int, int> > pos_and_val(64);
    for (int i = 0; i < 64; ++i) {
      pos_and_val[i].first = i;
      pos_and_val[i].second = num_zeros[kNaturalCoeffOrder[i]];
    }
    std::stable_sort(
        pos_and_val.begin(), pos_and_val.end(),
        [](const std::pair<int, int>& a, const std::pair<int, int>& b) -> bool {
          return a.second < b.second; });
    for (int i = 0; i < 64; ++i) {
      order[ctx * 64 + i] = kNaturalCoeffOrder[pos_and_val[i].first];
    }
  }
}

void EncodeCoeffOrder(const int* order, size_t* storage_ix, uint8_t* storage) {
  const int kJPEGZigZagOrder[64] = {
    0,   1,  5,  6, 14, 15, 27, 28,
    2,   4,  7, 13, 16, 26, 29, 42,
    3,   8, 12, 17, 25, 30, 41, 43,
    9,  11, 18, 24, 31, 40, 44, 53,
    10, 19, 23, 32, 39, 45, 52, 54,
    20, 22, 33, 38, 46, 51, 55, 60,
    21, 34, 37, 47, 50, 56, 59, 61,
    35, 36, 48, 49, 57, 58, 62, 63
  };
  int order_zigzag[64];
  for (int i = 0; i < 64; ++i) {
    order_zigzag[i] = kJPEGZigZagOrder[order[i]];
  }
  int lehmer[64];
  ComputeLehmerCode(order_zigzag, 64, lehmer);
  int end = 63;
  while (end >= 1 && lehmer[end] == 0) {
    --end;
  }
  for (int i = 1; i <= end; ++i) {
    ++lehmer[i];
  }
  static const int kSpan = 16;
  for (int i = 0; i < 64; i += kSpan) {
    const int start = (i > 0) ? i : 1;
    const int end = i + kSpan;
    int has_non_zero = 0;
    for (int j = start; j < end; ++j) has_non_zero |= lehmer[j];
    if (!has_non_zero) {   // all zero in the span -> escape
      WriteBits(1, 0, storage_ix, storage);
      continue;
    } else {
      WriteBits(1, 1, storage_ix, storage);
    }
    for (int j = start; j < end; ++j) {
      int v;
      PIK_ASSERT(lehmer[j] <= 64);
      for (v = lehmer[j]; v >= 7; v -= 7) {
        WriteBits(3, 7, storage_ix, storage);
      }
      WriteBits(3, v, storage_ix, storage);
    }
  }
}

std::string EncodeImage(const Image3S& img, int stride,
                        PikImageSizeInfo* info) {
  CoeffProcessor processor(stride);
  std::vector<ANSEncodingData> codes;
  std::vector<uint8_t> context_map;
  const std::string enc_hist =
      BuildAndEncodeHistograms(img, &processor, &codes, &context_map, info);
  const std::string enc_img =
      EncodeImageData(img, codes, context_map, &processor, info);
  return enc_hist + enc_img;
}

// Returns an image with a per-block count of zeroes, given an image of
// coefficients.
ImageI ExtractNumNZeroes(const ImageS& coeffs) {
  PIK_CHECK(coeffs.xsize() % 64 == 0);
  const size_t xsize = coeffs.xsize() / 64;
  const size_t ysize = coeffs.ysize();
  ImageI output(xsize, ysize);
  for (size_t y = 0; y < ysize; y++) {
    const int16_t* PIK_RESTRICT coeffs_row = coeffs.Row(y);
    int* PIK_RESTRICT output_row = output.Row(y);
    for (size_t x = 0; x < xsize; x++) {
      int num_nzeroes = 0;
      for (int i = 1; i < 64; i++) {
        if (coeffs_row[x * 64 + i] != 0) num_nzeroes++;
      }
      output_row[x] = num_nzeroes;
    }
  }
  return output;
}

Image3I ExtractNumNZeroes(const Image3S& coeffs) {
  return Image3I(ExtractNumNZeroes(coeffs.plane(0)),
                 ExtractNumNZeroes(coeffs.plane(1)),
                 ExtractNumNZeroes(coeffs.plane(2)));
}

namespace {

int PredictFromTopAndLeft(const int* const PIK_RESTRICT row_top,
                          const int* const PIK_RESTRICT row,
                          int x, int default_val) {
  if (x % kTileSize == 0) {
    return row_top == nullptr ? default_val : row_top[x];
  }
  if (row_top == nullptr) {
    return row[x - 1];
  }
  return (row_top[x] + row[x - 1] + 1) / 2;
}

int ContextFromTopAndLeft(const int* const PIK_RESTRICT row_top,
                          const int* const PIK_RESTRICT row,
                          const uint8_t* const PIK_RESTRICT ctx_row,
                          int x) {
  int pred = PredictFromTopAndLeft(row_top, row, x, 32);
  return kOrderContexts * (pred >> 1) + ctx_row[x];
}

}  // namespace

std::string BuildAndStoreNumNonzeroHistograms(
    const Image3I& img, const Image3B& block_ctx,
    std::vector<ANSEncodingData>* codes,
    std::vector<uint8_t>* context_map,
    PikImageSizeInfo* info) {
  HistogramBuilder builder(kOrderContexts * 32);
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < img.ysize(); ++y) {
      auto row_top = y % kTileSize == 0 ? nullptr : img.ConstPlaneRow(c, y - 1);
      auto row = img.ConstPlaneRow(c, y);
      auto row_ctx = block_ctx.ConstPlaneRow(c, y);
      for (int x = 0; x < img.xsize(); ++x) {
        int ctx = ContextFromTopAndLeft(row_top, row, row_ctx, x);
        builder.VisitSymbol(row[x], ctx);
      }
    }
  }
  const size_t max_out_size = 1024;
  std::string output(max_out_size, 0);
  size_t storage_ix = 0;
  uint8_t* storage = reinterpret_cast<uint8_t*>(&output[0]);
  storage[0] = 0;
  builder.BuildAndStoreEntropyCodes(codes, context_map, &storage_ix,
                                    storage, nullptr, info);
  const int out_size = (storage_ix + 7) >> 3;
  PIK_CHECK(out_size <= max_out_size);
  PIK_CHECK(out_size >= 0);
  output.resize(out_size);
  if (info) {
    info->num_clustered_histograms += codes->size();
    info->histogram_size += out_size;
    info->total_size += out_size;
  }
  return output;
}

static std::string EncodeNumNonzeroes(const Image3I& num_nzeroes,
                                      const Image3B& block_ctx,
                                      const std::vector<ANSEncodingData>& codes,
                                      const std::vector<uint8_t>& context_map,
                                      PikImageSizeInfo* info) {
  std::string encoded_num_nzeroes;
  for (int c = 0; c < 3; c++) {
    const size_t max_out_size =
        2 * num_nzeroes.xsize() * num_nzeroes.ysize() + 1024;
    std::string output(max_out_size, 0);
    size_t storage_ix = 0;
    uint8_t* storage = reinterpret_cast<uint8_t*>(&output[0]);
    storage[0] = 0;
    ANSSymbolWriter symbol_writer(codes, context_map, &storage_ix, storage);
    for (int y = 0; y < num_nzeroes.ysize(); ++y) {
      PIK_CHECK(y < kTileSize);
      auto row_top = y == 0 ? nullptr : num_nzeroes.ConstPlaneRow(c, y - 1);
      auto row = num_nzeroes.ConstPlaneRow(c, y);
      auto row_ctx = block_ctx.ConstPlaneRow(c, y);
      for (int x = 0; x < num_nzeroes.xsize(); ++x) {
        PIK_CHECK(x < kTileSize);
        int ctx = ContextFromTopAndLeft(row_top, row, row_ctx, x);
        symbol_writer.VisitSymbol(row[x], ctx);
      }
    }
    symbol_writer.FlushToBitStream();
    const int out_size = (storage_ix + 7) >> 3;
    PIK_CHECK(out_size <= max_out_size);
    output.resize(out_size);
    encoded_num_nzeroes += output;
    if (info) {
      info->entropy_coded_bits += storage_ix - symbol_writer.num_extra_bits();
      info->extra_bits += symbol_writer.num_extra_bits();
      info->total_size += out_size;
    }
  }
  return encoded_num_nzeroes;
}

std::string EncodeCoeffOrders(const int* order, PikInfo* pik_info) {
  std::string encoded_coeff_order(kOrderContexts * 1024, 0);
  uint8_t* storage = reinterpret_cast<uint8_t*>(&encoded_coeff_order[0]);
  size_t storage_ix = 0;
  for (int c = 0; c < kOrderContexts; c++) {
    EncodeCoeffOrder(&order[c * 64], &storage_ix, storage);
  }
  PIK_CHECK(storage_ix < encoded_coeff_order.size() * 8);
  encoded_coeff_order.resize((storage_ix + 7) / 8);
  if (pik_info) {
    pik_info->layers[kLayerOrder].total_size += encoded_coeff_order.size();
  }
  return encoded_coeff_order;
}

std::string EncodeNaturalCoeffOrders(PikInfo* pik_info) {
  std::string encoded_coeff_order(kOrderContexts * 1024, 0);
  uint8_t* storage = reinterpret_cast<uint8_t*>(&encoded_coeff_order[0]);
  size_t storage_ix = 0;
  for (int c = 0; c < kOrderContexts; c++) {
    EncodeCoeffOrder(kNaturalCoeffOrder, &storage_ix, storage);
  }
  PIK_CHECK(storage_ix < encoded_coeff_order.size() * 8);
  encoded_coeff_order.resize((storage_ix + 7) / 8);
  if (pik_info) {
    pik_info->layers[kLayerOrder].total_size += encoded_coeff_order.size();
  }
  return encoded_coeff_order;
}

std::string EncodeAC(const Image3S& coeffs,
                     const Image3B& block_ctx,
                     const std::vector<ANSEncodingData>& nzero_codes,
                     const std::vector<uint8_t>& nzero_context_map,
                     const std::vector<ANSEncodingData>& codes,
                     const std::vector<uint8_t>& context_map,
                     const int* order,
                     PikInfo* pik_info) {
  PikImageSizeInfo* ac_info = pik_info ? &pik_info->layers[kLayerAC] : nullptr;
  Image3I num_nzeroes = ExtractNumNZeroes(coeffs);
  std::string num_nzeroes_data = EncodeNumNonzeroes(
      num_nzeroes, block_ctx, nzero_codes, nzero_context_map, ac_info);
  ACBlockProcessor processor(&block_ctx, coeffs.xsize());
  processor.SetCoeffOrder(order);
  std::string coeffs_code =
      EncodeImageData(coeffs, codes, context_map, &processor, ac_info);
  return num_nzeroes_data + coeffs_code;
}

PIK_INLINE uint32_t MakeToken(const uint32_t context, const uint32_t symbol,
                              const uint32_t nbits, const uint32_t bits) {
  return (context << 26) | (symbol << 18) | (nbits << 14) | bits;
}

static const int kNumStaticZdensContexts = 7;
static const int kNumStaticContexts = 4 * kNumStaticZdensContexts;

std::vector<uint8_t> StaticContextMap() {
  static const int kNumContexts = kOrderContexts * 120;
  static const int kStaticZdensContextMap[120] = {
    0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
    0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
    4, 4, 4, 4, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
    4, 4, 4, 4, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
    5, 5, 5, 5, 5, 5, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
    6, 6, 6, 6, 5, 5, 2, 2, 2, 2, 2, 3, 3, 3, 3,
    6, 6, 6, 6, 5, 5, 2, 2, 2, 2, 2, 3, 3, 3,
    6, 6, 6, 6, 5, 5, 2, 2, 2, 2, 2,
  };
  PIK_ASSERT(kNumStaticContexts <= 64);
  std::vector<uint8_t> context_map(kNumContexts);
  for (int c = 0; c < kOrderContexts; ++c) {
    const int ctx = std::min(c, 3);
    for (int i = 0; i < 120; ++i) {
      context_map[c * 120 + i] =
          ctx * kNumStaticZdensContexts + kStaticZdensContextMap[i];
    }
  }
  return context_map;
}

std::vector<uint32_t> TokenizeCoefficients(
    const Image3S& coeffs,
    const Image3B& block_ctx,
    const std::vector<uint8_t>& context_map) {
  std::vector<uint32_t> tokens;
  tokens.reserve(3 * coeffs.xsize() * coeffs.ysize());
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < coeffs.ysize(); ++y) {
      const int16_t* const PIK_RESTRICT row = coeffs.ConstPlaneRow(c, y);
      const uint8_t* const PIK_RESTRICT ctx_row = block_ctx.ConstPlaneRow(c, y);
      for (int x = 0, bx = 0; x < coeffs.xsize(); x += 64, ++bx) {
        const int bctx = ctx_row[bx];
        const int16_t* coeffs = &row[x];
        int num_nzeros = 0;
        for (int k = 1; k < 64; ++k) {
          if (coeffs[k] != 0) ++num_nzeros;
        }
        if (num_nzeros == 0) continue;
        int r = 0;
        const int histo_offset = bctx * 120;
        int histo_idx = histo_offset + ZeroDensityContext(num_nzeros - 1, 0, 4);
        histo_idx = context_map[histo_idx];
        for (int k = 1; k < 64; ++k) {
          int16_t coeff = coeffs[kNaturalCoeffOrder[k]];
          if (coeff == 0) {
            r++;
            continue;
          }
          while (r > 15) {
            tokens.push_back(MakeToken(histo_idx, kIndexLut[0xf0], 0, 0));
            r -= 16;
          }
          int nbits, bits;
          EncodeCoeff(coeff, &nbits, &bits);
          PIK_ASSERT(nbits <= 14);
          int symbol = kIndexLut[(r << 4) + nbits];
          tokens.push_back(MakeToken(histo_idx, symbol, nbits, bits));
          r = 0;
          histo_idx = histo_offset + ZeroDensityContext(num_nzeros - 1, k, 4);
          histo_idx = context_map[histo_idx];
          --num_nzeros;
        }
      }
    }
  }
  return tokens;
}

std::string BuildAndEncodeHistogramsFast(
    const std::vector<uint8_t>& context_map,
    const std::vector<uint32_t>& tokens,
    std::vector<ANSEncodingData>* codes,
    PikInfo* pik_info) {
  // Build histograms from tokens.
  std::vector<uint32_t> histograms(kNumStaticContexts << 8);
  for (int i = 0; i < tokens.size(); ++i) {
    ++histograms[tokens[i] >> 18];
  }
  PikImageSizeInfo* ac_info = pik_info ? &pik_info->layers[kLayerAC] : nullptr;
  if (ac_info) {
    for (int c = 0; c < kNumStaticContexts; ++c) {
      ac_info->clustered_entropy += ShannonEntropy(&histograms[c << 8], 256);
    }
  }
  const size_t max_out_size = kNumStaticContexts * 1024;
  std::string output(max_out_size, 0);
  size_t storage_ix = 0;
  uint8_t* storage = reinterpret_cast<uint8_t*>(&output[0]);
  storage[0] = 0;
  // Encode the histograms.
  EncodeContextMap(context_map, kNumStaticContexts, &storage_ix, storage);
  for (int c = 0; c < kNumStaticContexts; ++c) {
    ANSEncodingData code;
    code.BuildAndStore(&histograms[c << 8], 256, &storage_ix, storage);
    codes->emplace_back(std::move(code));
  }
  // Close the histogram bit stream.
  WriteZeroesToByteBoundary(&storage_ix, storage);
  const size_t histo_bytes = (storage_ix >> 3);
  PIK_CHECK(histo_bytes <= max_out_size);
  output.resize(histo_bytes);
  if (ac_info) {
    ac_info->num_clustered_histograms += codes->size();
    ac_info->histogram_size += histo_bytes;
  }
  return output;
}

std::string EncodeACFast(const Image3S& coeffs,
                         const Image3B& block_ctx,
                         const std::vector<ANSEncodingData>& nzero_codes,
                         const std::vector<uint8_t>& nzero_context_map,
                         const std::vector<ANSEncodingData>& codes,
                         const std::vector<uint8_t>& context_map,
                         PikInfo* pik_info) {
  PikImageSizeInfo* ac_info = pik_info ? &pik_info->layers[kLayerAC] : nullptr;
  Image3I num_nzeroes = ExtractNumNZeroes(coeffs);
  std::vector<uint32_t> tokens = TokenizeCoefficients(
      coeffs, block_ctx, context_map);
  std::string encoded_num_nzeroes =
      EncodeNumNonzeroes(
          num_nzeroes, block_ctx, nzero_codes, nzero_context_map, ac_info);
  const size_t max_out_size = 3 * coeffs.xsize() * coeffs.ysize() + 4096;
  std::string output(max_out_size, 0);
  size_t storage_ix = 0;
  uint8_t* storage = reinterpret_cast<uint8_t*>(&output[0]);
  storage[0] = 0;
  memcpy(storage, encoded_num_nzeroes.data(), encoded_num_nzeroes.size());
  storage_ix += encoded_num_nzeroes.size() << 3;
  const size_t nonzeroes_bytes = encoded_num_nzeroes.size();
  size_t num_extra_bits = 0;
  PIK_ASSERT(kANSBufferSize <= (1 << 16));
  for (int start = 0; start < tokens.size(); start += kANSBufferSize) {
    std::vector<uint32_t> out;
    out.reserve(kANSBufferSize);
    const int end = std::min<int>(start + kANSBufferSize, tokens.size());
    ANSCoder ans;
    for (int i = end - 1; i >= start; --i) {
      const uint32_t token = tokens[i];
      const uint32_t context = token >> 26;
      const uint32_t symbol = (token >> 18) & 0xff;
      const ANSEncSymbolInfo info = codes[context].ans_table[symbol];
      uint8_t nbits = 0;
      uint32_t bits = ans.PutSymbol(info, &nbits);
      if (nbits == 16) {
        out.push_back(((i - start) << 16) | bits);
      }
    }
    const uint32_t state = ans.GetState();
    WriteBits(16, (state >> 16) & 0xffff, &storage_ix, storage);
    WriteBits(16, state & 0xffff, &storage_ix, storage);
    int tokenidx = start;
    for (int i = out.size(); i >= 0; --i) {
      int nextidx = i > 0 ? start + (out[i - 1] >> 16) : end;
      for (; tokenidx < nextidx; ++tokenidx) {
        const uint32_t token = tokens[tokenidx];
        const uint32_t nbits = (token >> 14) & 0xf;
        const uint32_t bits = token & 0x3fff;
        WriteBits(nbits, bits, &storage_ix, storage);
        num_extra_bits += nbits;
      }
      if (i > 0) {
        WriteBits(16, out[i - 1] & 0xffff, &storage_ix, storage);
      }
    }
  }
  const size_t data_bits = storage_ix - 8 * nonzeroes_bytes;
  const size_t data_bytes = (data_bits + 7) >> 3;
  const int out_size = nonzeroes_bytes + data_bytes;
  PIK_CHECK(out_size <= max_out_size);
  output.resize(out_size);
  if (ac_info) {
    ac_info->entropy_coded_bits += data_bits - num_extra_bits;
    ac_info->extra_bits += num_extra_bits;
    ac_info->total_size += out_size;
  }
  return output;
}

class ANSBitCounter {
 public:
  ANSBitCounter(const std::vector<ANSEncodingData>& codes,
                const std::vector<uint8_t>& context_map)
      : codes_(codes), context_map_(context_map), nbits_(0.0f) {}

  void VisitBits(size_t nbits, uint64_t bits, int c) {
    nbits_ += nbits;
  }

  void VisitSymbol(int symbol, int ctx) {
    const uint8_t histo_idx = context_map_[ctx];
    const ANSEncSymbolInfo info = codes_[histo_idx].ans_table[symbol];
    nbits_ += ANS_LOG_TAB_SIZE - std::log2(info.freq_);
  }

  float nbits() const { return nbits_; }

 private:
  const std::vector<ANSEncodingData>& codes_;
  const std::vector<uint8_t>& context_map_;
  float nbits_;
};

bool DecodeHistograms(BitReader* br, const size_t num_contexts,
                      const size_t max_alphabet_size, const uint8_t* symbol_lut,
                      size_t symbol_lut_size, ANSCode* code,
                      std::vector<uint8_t>* context_map) {
  size_t num_histograms = 1;
  context_map->resize(num_contexts);
  if (num_contexts > 1) {
    if (!DecodeContextMap(context_map, &num_histograms, br)) return false;
  }
  if (!DecodeANSCodes(num_histograms, max_alphabet_size, symbol_lut,
                      symbol_lut_size, br, code)) {
    return false;
  }
  br->JumpToByteBoundary();
  return true;
}

bool DecodeImageData(BitReader* const PIK_RESTRICT br,
                     const std::vector<uint8_t>& context_map,
                     const int stride,
                     ANSSymbolReader* const PIK_RESTRICT decoder,
                     Image3S* const PIK_RESTRICT img) {
  for (int c = 0; c < 3; ++c) {
    const int histo_idx = context_map[c];
    for (int y = 0; y < img->ysize(); ++y) {
      int16_t* const PIK_RESTRICT row = img->PlaneRow(c, y);
      for (int x = 0; x < img->xsize(); x += stride) {
        br->FillBitBuffer();
        int s = decoder->ReadSymbol(histo_idx, br);
        if (s > 0) {
          int bits = br->PeekBits(s);
          br->Advance(s);
          s = bits < (1 << (s - 1)) ? bits + ((~0U) << s ) + 1 : bits;
        }
        row[x] = s;
      }
    }
  }
  br->JumpToByteBoundary();
  return true;
}

bool DecodeCoeffOrder(int* order, BitReader* br) {
  int lehmer[64] = { 0 };
  static const int kSpan = 16;
  for (int i = 0; i < 64; i += kSpan) {
    br->FillBitBuffer();
    const int has_non_zero = br->ReadBits(1);
    if (!has_non_zero) continue;
    const int start = (i > 0) ? i : 1;
    const int end = i + kSpan;
    for (int j = start; j < end; ++j) {
      int v = 0;
      while (v <= 64) {
        br->FillBitBuffer();
        const int bits = br->ReadBits(3);
        v += bits;
        if (bits < 7) break;
      }
      if (v > 64) v = 64;
      lehmer[j] = v;
    }
  }
  int end = 63;
  while (end > 0 && lehmer[end] == 0) {
    --end;
  }
  for (int i = 1; i <= end; ++i) {
    --lehmer[i];
  }
  DecodeLehmerCode(lehmer, 64, order);
  for (int k = 0; k < 64; ++k) {
    order[k] = kNaturalCoeffOrder[order[k]];
  }
  return true;
}

bool DecodeACData(BitReader* const PIK_RESTRICT br,
                  const std::vector<uint8_t>& context_map,
                  const Image3B& block_ctx,
                  ANSSymbolReader* const PIK_RESTRICT decoder,
                  const int* const PIK_RESTRICT coeff_order,
                  const Image3I& num_nzeroes,
                  Image3S* const PIK_RESTRICT coeffs) {
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < coeffs->ysize(); ++y) {
      int16_t* const PIK_RESTRICT row = coeffs->PlaneRow(c, y);
      const uint8_t* const PIK_RESTRICT row_bctx =
          block_ctx.ConstPlaneRow(c, y);
      const int* const PIK_RESTRICT row_nzeros =
          num_nzeroes.ConstPlaneRow(c, y);
      for (int x = 0, bx = 0; x < coeffs->xsize(); x += 64, ++bx) {
        memset(&row[x + 1], 0, 63 * sizeof(row[0]));
        const int block_ctx = row_bctx[bx];
        int num_nzeros = row_nzeros[bx];
        if (num_nzeros > 63) {
          return PIK_FAILURE("Invalid AC data.");
        }
        if (num_nzeros == 0) continue;
        const int histo_offset = block_ctx * 120;
        const int context2 = ZeroDensityContext(num_nzeros - 1, 0, 4);
        int histo_idx = context_map[histo_offset + context2];
        const int order_offset = block_ctx * 64;
        const int* PIK_RESTRICT const block_order = &coeff_order[order_offset];
        for (int k = 1; k < 64 && num_nzeros > 0; ++k) {
          br->FillBitBuffer();
          int s = decoder->ReadSymbol(histo_idx, br);
          k += (s >> 4);
          if (k + num_nzeros > 64) {
            return PIK_FAILURE("Invalid AC data.");
          }
          s &= 15;
          if (s > 0) {
            int bits = br->PeekBits(s);
            br->Advance(s);
            s = bits < (1 << (s - 1)) ? bits + ((~0U) << s ) + 1 : bits;
            const int context = ZeroDensityContext(num_nzeros - 1, k, 4);
            histo_idx = context_map[histo_offset + context];
            --num_nzeros;
          }
          row[x + block_order[k]] = s;
        }
        if (num_nzeros != 0) {
          return PIK_FAILURE("Invalid AC data.");
        }
      }
    }
  }
  br->JumpToByteBoundary();
  return true;
}

bool DecodeImage(BitReader* br, int stride,  Image3S* coeffs) {
  std::vector<uint8_t> context_map;
  ANSCode code;
  if (!DecodeHistograms(br, CoeffProcessor::num_contexts(), 16, nullptr, 0,
                        &code, &context_map)) {
    return false;
  }
  ANSSymbolReader decoder(&code);
  if (!DecodeImageData(br, context_map, stride, &decoder, coeffs)) {
    return false;
  }
  if (!decoder.CheckANSFinalState()) {
    return PIK_FAILURE("ANS checksum failure.");
  }
  return true;
}

bool DecodeNumNZeroes(BitReader* br, Image3I* num_nzeroes,
                      const Image3B& block_ctx,
                      const std::vector<uint8_t>& context_map,
                      const ANSCode& ans_code) {
  // TODO(user): Consider storing the planes interleaved with AC.
  for (int c = 0; c < 3; c++) {
    ANSSymbolReader decoder(&ans_code);
    for (int y = 0; y < num_nzeroes->ysize(); ++y) {
      PIK_CHECK(y < kTileSize);
      auto row_top = y == 0 ? nullptr : num_nzeroes->PlaneRow(c, y - 1);
      auto row = num_nzeroes->PlaneRow(c, y);
      auto row_ctx = block_ctx.ConstPlaneRow(c, y);
      for (int x = 0; x < num_nzeroes->xsize(); ++x) {
        PIK_CHECK(x < kTileSize);
        int ctx = ContextFromTopAndLeft(row_top, row, row_ctx, x);
        br->FillBitBuffer();
        row[x] = decoder.ReadSymbol(context_map[ctx], br);
        if (row[x] > 63 || row[x] < 0) {
          return PIK_FAILURE("Out of range value in num nonzeros plane.");
        }
      }
    }
    if (!decoder.CheckANSFinalState()) {
      return PIK_FAILURE("ANS checksum failure.");
    }
    br->JumpToByteBoundary();
  }
  return true;
}

bool DecodeAC(const Image3B& block_ctx,
              const ANSCode& nzero_code,
              const std::vector<uint8_t>& nzero_context_map,
              const ANSCode& code,
              const std::vector<uint8_t>& context_map, const int* coeff_order,
              BitReader* br, Image3S* coeffs) {
  PIK_ASSERT(coeffs->xsize() % 64 == 0);
  Image3I num_nzeroes(coeffs->xsize() / 64, coeffs->ysize());
  if (!DecodeNumNZeroes(br, &num_nzeroes, block_ctx, nzero_context_map,
                        nzero_code)) {
    return false;
  }
  ANSSymbolReader decoder(&code);
  if (!DecodeACData(br, context_map, block_ctx, &decoder, coeff_order,
                    num_nzeroes, coeffs)) {
    return false;
  }
  if (!decoder.CheckANSFinalState()) {
    return PIK_FAILURE("ANS checksum failure.");
  }
  return true;
}

namespace {

class ContextProcessor {
 public:
  ContextProcessor(int minval, int maxval, int x_tilesize, int y_tilesize,
                   int external_context_count)
      : minval_(minval),
        maxval_(maxval),
        x_tilesize_(x_tilesize),
        y_tilesize_(y_tilesize),
        external_context_source_(nullptr),
        external_num_contexts_(external_context_count) {
    Reset();
  }

  void set_external_context_source(const ImageB* external_context_source) {
    external_context_source_ = external_context_source;
  }

  void Reset() { row_ = std::vector<int>(x_tilesize_); }

  int block_size() const { return 1; }
  int num_contexts() {
    return (1 + ((maxval_ - minval_) >> kContextShift)) *
           external_num_contexts_;
  }

  int PredictVal(int x, int y, int c) {
    PIK_ASSERT(0 <= x && x < x_tilesize_);
    PIK_ASSERT(0 <= y && y < y_tilesize_);
    if (x == 0) {
      return y == 0 ? (minval_ + maxval_ + 1) / 2 : row_[x];
    }
    if (y == 0) {
      return row_[x - 1];
    }
    return (row_[x] + row_[x - 1] + 1) / 2;
  }

  int Context(int x, int y, int c) {
    int context = external_num_contexts_ *
                  ((PredictVal(x, y, c) - minval_) >> kContextShift);
    PIK_ASSERT(external_context_source_->Row(y)[x] >= 0);
    PIK_ASSERT(external_context_source_->Row(y)[x] < external_num_contexts_);
    context += external_context_source_->Row(y)[x];
    return context;
  }

  void SetVal(int x, int y, int c, int val) { row_[x] = val; }

  template <class Visitor>
  void ProcessBlock(const int* val, int x, int y, int c, Visitor* visitor) {
    PIK_ASSERT(0 <= x && x < x_tilesize_);
    PIK_ASSERT(0 <= y && y < y_tilesize_);
    PIK_ASSERT(*val >= minval_ && *val <= maxval_);
    int ctx = Context(x, y, c);
    visitor->VisitSymbol(*val - minval_, ctx);
    SetVal(x, y, c, *val);
  }

 private:
  static const int kContextShift = 1;
  const int minval_;
  const int maxval_;
  const int x_tilesize_;
  const int y_tilesize_;
  std::vector<int> row_;
  const ImageB* external_context_source_;  // Not owned. Must be alive whenever
                                           // any method of Processor is called.
  const int external_num_contexts_;
};

}  // namespace

EncodedIntPlane EncodePlane(const ImageI& img, int minval, int maxval,
                            int tile_size, int num_external_contexts,
                            const ImageB* external_context,
                            PikImageSizeInfo* info) {
  EncodedIntPlane result;
  ContextProcessor processor(minval, maxval, tile_size, tile_size,
                             num_external_contexts);
  HistogramBuilder builder(processor.num_contexts());
  const size_t tile_xsize = (img.xsize() + tile_size - 1) / tile_size;
  const size_t tile_ysize = (img.ysize() + tile_size - 1) / tile_size;
  ImageB dummy_context(tile_size, tile_size, 0);
  for (int y = 0; y < tile_ysize; y++) {
    for (int x = 0; x < tile_xsize; x++) {
      ConstWrapper<ImageI> tile =
          ConstWindow(img, x * tile_size, y * tile_size, tile_size, tile_size);
      ConstWrapper<ImageB> context_tile =
          external_context == nullptr
              ? ConstWindow(dummy_context, 0, 0, tile_size, tile_size)
              : ConstWindow(*external_context, x * tile_size, y * tile_size,
                            tile_size, tile_size);
      processor.set_external_context_source(&context_tile.get());
      ProcessImage(tile.get(), &processor, &builder);
    }
  }

  std::vector<ANSEncodingData> codes;
  std::vector<uint8_t> context_map;
  {
    const size_t max_out_size = 1024;
    result.preamble = std::string(max_out_size, 0);
    size_t storage_ix = 0;
    uint8_t* storage = reinterpret_cast<uint8_t*>(&result.preamble[0]);
    storage[0] = 0;
    builder.BuildAndStoreEntropyCodes(&codes, &context_map, &storage_ix,
                                      storage, nullptr, info);
    const int out_size = (storage_ix + 7) >> 3;
    PIK_CHECK(out_size <= max_out_size);
    PIK_CHECK(out_size >= 0);
    result.preamble.resize(out_size);
    if (info) {
      info->num_clustered_histograms += codes.size();
      info->histogram_size += out_size;
      info->total_size += out_size;
    }
  }
  result.tiles.resize(tile_ysize);
  for (int y = 0; y < tile_ysize; y++) {
    result.tiles[y].resize(tile_xsize);
    for (int x = 0; x < tile_xsize; x++) {
      const size_t max_out_size = 2 * tile_size * tile_size;
      result.tiles[y][x] = std::string(max_out_size, 0);
      size_t storage_ix = 0;
      uint8_t* storage = reinterpret_cast<uint8_t*>(&result.tiles[y][x][0]);
      storage[0] = 0;
      ANSSymbolWriter symbol_writer(codes, context_map, &storage_ix, storage);
      ConstWrapper<ImageI> tile =
          ConstWindow(img, x * tile_size, y * tile_size, tile_size, tile_size);
      ConstWrapper<ImageB> context_tile =
          external_context == nullptr
              ? ConstWindow(dummy_context, 0, 0, tile_size, tile_size)
              : ConstWindow(*external_context, x * tile_size, y * tile_size,
                            tile_size, tile_size);
      processor.set_external_context_source(&context_tile.get());
      ProcessImage(tile.get(), &processor, &symbol_writer);
      symbol_writer.FlushToBitStream();
      const int out_size = (storage_ix + 7) >> 3;
      PIK_CHECK(out_size <= max_out_size);
      result.tiles[y][x].resize(out_size);
      if (info) {
        info->entropy_coded_bits += storage_ix - symbol_writer.num_extra_bits();
        info->extra_bits += symbol_writer.num_extra_bits();
        info->total_size += out_size;
      }
    }
  }
  return result;
}

bool IntPlaneDecoder::LoadPreamble(BitReader* br) {
  PIK_CHECK(!ready_);
  ContextProcessor processor(minval_, maxval_, tile_size_, tile_size_,
                             num_external_contexts_);
  const size_t max_alphabet_size = maxval_ - minval_ + 1;
  if (!DecodeHistograms(br, processor.num_contexts(), max_alphabet_size,
                        nullptr, 0, &ans_code_, &context_map_)) {
    return false;
  }
  br->JumpToByteBoundary();
  ready_ = true;
  return true;
}

bool IntPlaneDecoder::DecodeTile(BitReader* br, ImageI* img,
                                 const ImageB* external_context) {
  ImageB dummy_context;
  ContextProcessor processor(minval_, maxval_, tile_size_, tile_size_,
                             num_external_contexts_);
  if (external_context == nullptr) {
    dummy_context = ImageB(tile_size_, tile_size_, 0);
    external_context = &dummy_context;
  }
  processor.set_external_context_source(external_context);
  ANSSymbolReader decoder(&ans_code_);
  for (int y = 0; y < img->ysize(); ++y) {
    auto row = img->Row(y);
    for (int x = 0; x < img->xsize(); ++x) {
      int ctx = processor.Context(x, y, 0);
      br->FillBitBuffer();
      row[x] = decoder.ReadSymbol(context_map_[ctx], br) + minval_;
      if (row[x] > maxval_ || row[x] < minval_) {
        return PIK_FAILURE("Out of range value in quantization plane.");
      }
      processor.SetVal(x, y, 0, row[x]);
    }
  }
  if (!decoder.CheckANSFinalState()) {
    return PIK_FAILURE("ANS checksum failure.");
  }
  br->JumpToByteBoundary();
  return true;
}

}  // namespace pik
