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

#include "entropy_coder.h"

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
#include "status.h"
#include "write_bits.h"

namespace pik {

void PredictDCTile(const Image3S& dc, Image3S* out) {
  // One DC per block.
  const size_t xsize = dc.xsize();
  const size_t ysize = dc.ysize();
  PIK_CHECK(SameSize(dc, *out));

  ShrinkY(dc.plane(1), out->MutablePlane(1));

  ImageS dc_xz(xsize * 2, ysize);

  for (size_t y = 0; y < ysize; y++) {
    const int16_t* PIK_RESTRICT row0 = dc.ConstPlaneRow(0, y);
    const int16_t* PIK_RESTRICT row2 = dc.ConstPlaneRow(2, y);
    int16_t* PIK_RESTRICT row_xz = dc_xz.Row(y);
    for (size_t x = 0; x < xsize; ++x) {
      row_xz[2 * x + 0] = row0[x];
      row_xz[2 * x + 1] = row2[x];
    }
  }

  ImageS dc_xz_out(xsize * 2, ysize);
  ShrinkUV(dc.plane(1), dc_xz, &dc_xz_out);

  for (size_t y = 0; y < ysize; y++) {
    const int16_t* PIK_RESTRICT row_xz = dc_xz_out.ConstRow(y);
    int16_t* PIK_RESTRICT row_out0 = out->PlaneRow(0, y);
    int16_t* PIK_RESTRICT row_out2 = out->PlaneRow(2, y);
    for (size_t x = 0; x < xsize; ++x) {
      row_out0[x] = row_xz[2 * x + 0];
      row_out2[x] = row_xz[2 * x + 1];
    }
  }
}

void UnpredictDCTile(Image3S* dc) {
  // One DC per block.
  const size_t xsize = dc->xsize();
  const size_t ysize = dc->ysize();

  ImageS dc_y_out(xsize, ysize);
  ExpandY(dc->plane(1), &dc_y_out);

  ImageS dc_xz(xsize * 2, ysize);

  for (size_t y = 0; y < ysize; y++) {
    const int16_t* PIK_RESTRICT row0 = dc->PlaneRow(0, y);
    const int16_t* PIK_RESTRICT row2 = dc->PlaneRow(2, y);
    int16_t* PIK_RESTRICT row_xz = dc_xz.Row(y);
    for (size_t x = 0; x < xsize; ++x) {
      row_xz[2 * x + 0] = row0[x];
      row_xz[2 * x + 1] = row2[x];
    }
  }

  ImageS dc_xz_out(xsize * 2, ysize);

  ExpandUV(dc_y_out, dc_xz, &dc_xz_out);

  for (size_t y = 0; y < ysize; y++) {
    const int16_t* PIK_RESTRICT row_y = dc_y_out.Row(y);
    const int16_t* PIK_RESTRICT row_xz = dc_xz_out.Row(y);
    int16_t* PIK_RESTRICT row_out0 = dc->PlaneRow(0, y);
    int16_t* PIK_RESTRICT row_out1 = dc->PlaneRow(1, y);
    int16_t* PIK_RESTRICT row_out2 = dc->PlaneRow(2, y);
    for (size_t x = 0; x < xsize; ++x) {
      row_out0[x] = row_xz[2 * x + 0];
      row_out1[x] = row_y[x];
      row_out2[x] = row_xz[2 * x + 1];
    }
  }
}

void ComputeCoeffOrder(const Image3S& ac, const Image3B& block_ctx,
                       int* PIK_RESTRICT order) {
  for (int ctx = 0; ctx < kOrderContexts; ++ctx) {
    int num_zeros[64] = { 0 };
    for (int c = 0; c < 3; ++c) {
      for (size_t y = 0; y < ac.ysize(); ++y) {
        const int16_t* PIK_RESTRICT row = ac.PlaneRow(c, y);
        const uint8_t* PIK_RESTRICT row_ctx = block_ctx.PlaneRow(c, y);
        for (size_t x = 0; x < ac.xsize(); x += kDCTBlockSize) {
          if (row_ctx[x >> 6] != ctx) continue;
          for (size_t k = 1; k < kDCTBlockSize; ++k) {
            if (row[x + k] == 0) ++num_zeros[k];
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

void EncodeCoeffOrder(const int* PIK_RESTRICT order,
                      size_t* PIK_RESTRICT storage_ix, uint8_t* storage) {
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
  for (size_t i = 0; i < 64; ++i) {
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

static const int kNumStaticZdensContexts = 7;
static const int kNumStaticContexts = 12 + 4 * kNumStaticZdensContexts;

// Layout of the context map:
//    0 -  127 : context for quantization levels,
//               computed from top and left values
//  128 -  319 : context for number of non-zeros in the block
//               computed from block context (0..5) and top and left
//               values (0..31)
//  320 - 1039 : context for AC coefficient symbols, computed from
//               block context, number of non-zeros left and index in scan order
std::vector<uint8_t> StaticContextMap() {
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
  static const uint8_t kStaticQuantContextMap[128] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 3,
    4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
  };
  PIK_ASSERT(kNumStaticContexts <= 64);
  std::vector<uint8_t> context_map(kNumContexts);
  memcpy(&context_map[0], kStaticQuantContextMap,
         sizeof(kStaticQuantContextMap));
  for (int c = 0; c < kOrderContexts; ++c) {
    const int ctx = std::min(c, 3);
    for (int i = 0; i < 32; ++i) {
      context_map[128 + c * 32 + i] = 8 + ctx;
    }
    for (int i = 0; i < 120; ++i) {
      context_map[128 + kOrderContexts * 32 + c * 120 + i] =
          12 + ctx * kNumStaticZdensContexts + kStaticZdensContextMap[i];
    }
  }
  return context_map;
}

PIK_INLINE int PredictFromTopAndLeft(const int* const PIK_RESTRICT row_top,
                                     const int* const PIK_RESTRICT row,
                                     size_t x, int default_val) {
  if (x % kTileWidth == 0) {
    return row_top == nullptr ? default_val : row_top[x];
  }
  if (row_top == nullptr) {
    return row[x - 1];
  }
  return (row_top[x] + row[x - 1] + 1) / 2;
}

PIK_INLINE int ContextFromTopAndLeft(const int* const PIK_RESTRICT row_top,
                                     const int* const PIK_RESTRICT row,
                                     int block_ctx, int x) {
  int pred = PredictFromTopAndLeft(row_top, row, x, 32);
  return kOrderContexts * (pred >> 1) + block_ctx;
}

PIK_INLINE void EncodeCoeff(int coeff, int* nbits, int* bits) {
  int coeff_bits = coeff;
  if (coeff < 0) {
    coeff = -coeff;
    coeff_bits--;
  }
  *nbits = Log2Floor(coeff) + 1;
  *bits = coeff_bits & ((1 << *nbits) - 1);
}

std::vector<Token> TokenizeCoefficients(
    const int* orders,
    const ImageI& quant_ac,
    const Image3S& coeffs,
    const Image3B& block_ctx) {
  std::vector<Token> tokens;
  tokens.reserve(3 * coeffs.xsize() * coeffs.ysize());
  for (size_t y = 0; y < quant_ac.ysize(); ++y) {
    const int* PIK_RESTRICT row_quant = quant_ac.ConstRow(y);
    const int* PIK_RESTRICT row_quant_top =
        y % kTileHeight == 0 ? nullptr : quant_ac.ConstRow(y - 1);
    for (size_t bx = 0; bx < quant_ac.xsize(); ++bx) {
      int quant_pred =
          PredictFromTopAndLeft(row_quant_top, row_quant, bx, 32);
      int quant_ctx = (quant_pred - 1) >> 1;
      tokens.emplace_back(Token(quant_ctx, row_quant[bx] - 1, 0, 0));
    }
  }
  for (int c = 0; c < 3; ++c) {
    ImageI num_nzeros = ExtractNumNZeroes(coeffs.plane(c));
    for (size_t y = 0; y < coeffs.ysize(); ++y) {
      const int16_t* PIK_RESTRICT row = coeffs.ConstPlaneRow(c, y);
      const uint8_t* PIK_RESTRICT ctx_row = block_ctx.ConstPlaneRow(c, y);
      const int* PIK_RESTRICT row_nzeros = num_nzeros.ConstRow(y);
      const int* PIK_RESTRICT row_nzeros_top =
          y % kTileHeight == 0 ? nullptr : num_nzeros.ConstRow(y - 1);
      for (size_t x = 0, bx = 0; x < coeffs.xsize(); x += 64, ++bx) {
        const int bctx = ctx_row[bx];
        const int* order = &orders[bctx * 64];
        const int16_t* PIK_RESTRICT coeffs = &row[x];
        int num_nzeros = 0;
        for (int k = 1; k < 64; ++k) {
          if (coeffs[k] != 0) ++num_nzeros;
        }
        int nzero_ctx = 128 + ContextFromTopAndLeft(row_nzeros_top, row_nzeros,
                                                    bctx, bx);
        tokens.emplace_back(Token(nzero_ctx, num_nzeros, 0, 0));
        PIK_CHECK(num_nzeros == row_nzeros[bx]);
        if (num_nzeros == 0) continue;
        int r = 0;
        const int histo_offset = 128 + kOrderContexts * 32 + bctx * 120;
        int histo_idx = histo_offset + ZeroDensityContext(num_nzeros - 1, 0, 4);
        for (int k = 1; k < 64; ++k) {
          int16_t coeff = coeffs[order[k]];
          if (coeff == 0) {
            r++;
            continue;
          }
          while (r > 15) {
            tokens.emplace_back(Token(histo_idx, kIndexLut[0xf0], 0, 0));
            r -= 16;
          }
          int nbits, bits;
          EncodeCoeff(coeff, &nbits, &bits);
          PIK_ASSERT(nbits <= 14);
          int symbol = kIndexLut[(r << 4) + nbits];
          tokens.emplace_back(Token(histo_idx, symbol, nbits, bits));
          r = 0;
          histo_idx = histo_offset + ZeroDensityContext(num_nzeros - 1, k, 4);
          --num_nzeros;
        }
      }
    }
  }
  return tokens;
}

namespace {

inline double CrossEntropy(const uint32_t* counts, const size_t counts_len,
                           const uint32_t* codes,  const size_t codes_len) {
  double sum = 0.0f;
  uint32_t total_count = 0;
  uint32_t total_codes = 0;
  for (int i = 0; i < codes_len; ++i) {
    if (codes[i] > 0) {
      if (i < counts_len && counts[i] > 0) {
        sum -= counts[i] * std::log2(codes[i]);
        total_count += counts[i];
      }
      total_codes += codes[i];
    }
  }
  if (total_codes > 0) {
    sum += total_count * std::log2(total_codes);
  }
  return sum;
}

inline double ShannonEntropy(const uint32_t* data, const size_t data_size) {
  return CrossEntropy(data, data_size, data, data_size);
}

class HistogramBuilder {
 public:
  explicit HistogramBuilder(const size_t num_contexts)
      : histograms_(num_contexts) {}

  void VisitSymbol(int symbol, int histo_idx) {
    PIK_ASSERT(histo_idx < histograms_.size());
    histograms_[histo_idx].Add(symbol);
  }

  template <class EntropyEncodingData>
  void BuildAndStoreEntropyCodes(std::vector<EntropyEncodingData>* codes,
                                 std::vector<uint8_t>* context_map,
                                 size_t* storage_ix, uint8_t* storage,
                                 PikImageSizeInfo* info) const {
    std::vector<Histogram> clustered_histograms(histograms_);
    context_map->resize(histograms_.size());
    if (histograms_.size() > 1) {
      std::vector<uint32_t> histogram_symbols;
      ClusterHistograms(histograms_, histograms_.size(), 1, std::vector<int>(),
                        64, &clustered_histograms, &histogram_symbols);
      for (int c = 0; c < histograms_.size(); ++c) {
        (*context_map)[c] = static_cast<uint8_t>(histogram_symbols[c]);
      }
      if (storage_ix != nullptr && storage != nullptr) {
        EncodeContextMap(*context_map, clustered_histograms.size(),
                         storage_ix, storage);
      }
    }
    if (info) {
      for (int i = 0; i < clustered_histograms.size(); ++i) {
        info->clustered_entropy += clustered_histograms[i].ShannonEntropy();
      }
    }
    for (int c = 0; c < clustered_histograms.size(); ++c) {
      EntropyEncodingData code;
      code.BuildAndStore(clustered_histograms[c].data_.data(),
                         clustered_histograms[c].data_.size(), storage_ix,
                         storage);
      codes->emplace_back(std::move(code));
    }
  }

 private:
  struct Histogram {
    Histogram() {
      data_.reserve(256);
      total_count_ = 0;
    }
    void Clear() {
      memset(data_.data(), 0, data_.size() * sizeof(data_[0]));
      total_count_ = 0;
    }
    void Add(int symbol) {
      if (symbol >= data_.size()) {
        data_.resize(symbol + 1);
      }
      ++data_[symbol];
      ++total_count_;
    }
    void AddHistogram(const Histogram& other) {
      if (other.data_.size() > data_.size()) {
        data_.resize(other.data_.size());
      }
      for (int i = 0; i < other.data_.size(); ++i) {
        data_[i] += other.data_[i];
      }
      total_count_ += other.total_count_;
    }
    float PopulationCost() const {
      std::vector<int> counts(data_.size());
      for (int i = 0; i < data_.size(); ++i) {
        counts[i] = data_[i];
      }
      return ANSPopulationCost(counts.data(), counts.size(), total_count_);
    }
    double ShannonEntropy() const {
      return pik::ShannonEntropy(data_.data(), data_.size());
    }

    std::vector<uint32_t> data_;
    uint32_t total_count_;
  };
  std::vector<Histogram> histograms_;
};

}  // namespace

std::string BuildAndEncodeHistograms(
    int num_contexts,
    const std::vector<std::vector<Token> >& tokens,
    std::vector<ANSEncodingData>* codes,
    std::vector<uint8_t>* context_map,
    PikImageSizeInfo* info) {
  // Build histograms.
  HistogramBuilder builder(num_contexts);
  for (int i = 0; i < tokens.size(); ++i) {
    for (int j = 0; j < tokens[i].size(); ++j) {
      const Token token = tokens[i][j];
      builder.VisitSymbol(token.symbol, token.context);
    }
  }
  // Encode histograms.
  const size_t max_out_size = 1024 * (num_contexts + 4);
  std::string output(max_out_size, 0);
  size_t storage_ix = 0;
  uint8_t* storage = reinterpret_cast<uint8_t*>(&output[0]);
  storage[0] = 0;
  builder.BuildAndStoreEntropyCodes(codes, context_map, &storage_ix, storage,
                                    info);
  // Close the histogram bit stream.
  size_t jump_bits = ((storage_ix + 7) & ~7) - storage_ix;
  WriteBits(jump_bits, 0, &storage_ix, storage);
  PIK_ASSERT(storage_ix % 8 == 0);
  const size_t histo_bytes = storage_ix >> 3;
  PIK_CHECK(histo_bytes <= max_out_size);
  output.resize(histo_bytes);
  if (info) {
    info->num_clustered_histograms += codes->size();
    info->histogram_size += histo_bytes;
    info->total_size += histo_bytes;
  }
  return output;
}

std::string BuildAndEncodeHistogramsFast(
    const std::vector<std::vector<Token> >& tokens,
    std::vector<ANSEncodingData>* codes,
    std::vector<uint8_t>* context_map,
    PikImageSizeInfo* info) {
  *context_map = StaticContextMap();
  // Build histograms from tokens.
  std::vector<uint32_t> histograms(kNumStaticContexts << 8);
  for (int i = 0; i < tokens.size(); ++i) {
    for (int j = 0; j < tokens[i].size(); ++j) {
      const Token token = tokens[i][j];
      const uint32_t histo_idx = (*context_map)[token.context];
      ++histograms[(histo_idx << 8) + token.symbol];
    }
  }
  if (info) {
    for (int c = 0; c < kNumStaticContexts; ++c) {
      info->clustered_entropy += ShannonEntropy(&histograms[c << 8], 256);
    }
  }
  const size_t max_out_size = kNumStaticContexts * 1024;
  std::string output(max_out_size, 0);
  size_t storage_ix = 0;
  uint8_t* storage = reinterpret_cast<uint8_t*>(&output[0]);
  storage[0] = 0;
  // Encode the histograms.
  EncodeContextMap(*context_map, kNumStaticContexts, &storage_ix, storage);
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
  if (info) {
    info->num_clustered_histograms += codes->size();
    info->histogram_size += histo_bytes;
  }
  return output;
}

std::string WriteTokens(const std::vector<Token>& tokens,
                        const std::vector<ANSEncodingData>& codes,
                        const std::vector<uint8_t>& context_map,
                        PikImageSizeInfo* pik_info) {
  const size_t max_out_size = 4 * tokens.size() + 4096;
  std::string output(max_out_size, 0);
  size_t storage_ix = 0;
  uint8_t* storage = reinterpret_cast<uint8_t*>(&output[0]);
  storage[0] = 0;
  size_t num_extra_bits = 0;
  PIK_ASSERT(kANSBufferSize <= (1 << 16));
  for (int start = 0; start < tokens.size(); start += kANSBufferSize) {
    std::vector<uint32_t> out;
    out.reserve(kANSBufferSize);
    const int end = std::min<int>(start + kANSBufferSize, tokens.size());
    ANSCoder ans;
    for (int i = end - 1; i >= start; --i) {
      const Token token = tokens[i];
      const uint8_t histo_idx = context_map[token.context];
      const ANSEncSymbolInfo info = codes[histo_idx].ans_table[token.symbol];
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
        const Token token = tokens[tokenidx];
        WriteBits(token.nbits, token.bits, &storage_ix, storage);
        num_extra_bits += token.nbits;
      }
      if (i > 0) {
        WriteBits(16, out[i - 1] & 0xffff, &storage_ix, storage);
      }
    }
  }
  const size_t out_size = (storage_ix + 7) >> 3;
  PIK_CHECK(out_size <= max_out_size);
  output.resize(out_size);
  if (pik_info) {
    pik_info->entropy_coded_bits += storage_ix - num_extra_bits;
    pik_info->extra_bits += num_extra_bits;
    pik_info->total_size += out_size;
  }
  return output;
}

std::string EncodeImage(const Image3S& img, int stride,
                        PikImageSizeInfo* info) {
  std::vector<std::vector<Token> > tokens(1);
  tokens[0].reserve(3 * img.ysize() * img.xsize() / stride);
  for (int c = 0; c < 3; ++c) {
    for (size_t y = 0; y < img.ysize(); ++y) {
      const int16_t* const PIK_RESTRICT row = img.ConstPlaneRow(c, y);
      for (size_t x = 0; x < img.xsize(); x += stride) {
        int nbits, bits;
        EncodeCoeff(row[x], &nbits, &bits);
        tokens[0].emplace_back(Token(c, nbits, nbits, bits));
      }
    }
  }
  std::vector<ANSEncodingData> codes;
  std::vector<uint8_t> context_map;
  const std::string enc_hist =
      BuildAndEncodeHistograms(3, tokens, &codes, &context_map, info);
  const std::string enc_img = WriteTokens(tokens[0], codes, context_map, info);
  return enc_hist + enc_img;
}

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
    return PIK_FAILURE("Histo DecodeANSCodes");
  }
  br->JumpToByteBoundary();
  return true;
}

bool DecodeImageData(BitReader* PIK_RESTRICT br,
                     const std::vector<uint8_t>& context_map,
                     ANSSymbolReader* PIK_RESTRICT decoder,
                     Image3S* PIK_RESTRICT img) {
  for (int c = 0; c < 3; ++c) {
    const int histo_idx = context_map[c];
    for (size_t y = 0; y < img->ysize(); ++y) {
      int16_t* PIK_RESTRICT row = img->PlaneRow(c, y);
      for (size_t x = 0; x < img->xsize(); ++x) {
        br->FillBitBuffer();
        int s = decoder->ReadSymbol(histo_idx, br);
        if (s > 0) {
          int bits = br->PeekBits(s);
          br->Advance(s);
          s = bits < (1U << (s - 1)) ? bits + ((~0U) << s) + 1 : bits;
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

bool DecodeImage(BitReader* PIK_RESTRICT br, Image3S* PIK_RESTRICT coeffs) {
  std::vector<uint8_t> context_map;
  ANSCode code;
  if (!DecodeHistograms(br, 3, 16, nullptr, 0, &code, &context_map)) {
    return false;
  }
  ANSSymbolReader decoder(&code);
  if (!DecodeImageData(br, context_map, &decoder, coeffs)) {
    return false;
  }
  if (!decoder.CheckANSFinalState()) {
    return PIK_FAILURE("ANS checksum failure.");
  }
  return true;
}

bool DecodeAC(const Image3B& block_ctx, const ANSCode& code,
              const std::vector<uint8_t>& context_map,
              const int* PIK_RESTRICT coeff_order, BitReader* PIK_RESTRICT br,
              Image3S* PIK_RESTRICT ac, ImageI* PIK_RESTRICT quant_ac) {
  PIK_ASSERT(ac->xsize() % kBlockSize == 0);
  Image3I num_nzeroes(ac->xsize() / kBlockSize, ac->ysize());
  ANSSymbolReader decoder(&code);
  for (size_t y = 0; y < quant_ac->ysize(); ++y) {
    int* PIK_RESTRICT row_quant = quant_ac->Row(y);
    const int* PIK_RESTRICT row_quant_top =
        y % kTileHeight == 0 ? nullptr : quant_ac->ConstRow(y - 1);
    for (size_t bx = 0; bx < quant_ac->xsize(); ++bx) {
      br->FillBitBuffer();
      int quant_pred =
          PredictFromTopAndLeft(row_quant_top, row_quant, bx, 32);
      int quant_ctx = (quant_pred - 1) >> 1;
      row_quant[bx] =
          kIndexLut[decoder.ReadSymbol(context_map[quant_ctx], br)] + 1;
    }
  }
  for (int c = 0; c < 3; ++c) {
    for (size_t y = 0; y < ac->ysize(); ++y) {
      int16_t* PIK_RESTRICT row = ac->PlaneRow(c, y);
      const uint8_t* PIK_RESTRICT row_bctx = block_ctx.ConstPlaneRow(c, y);
      int* PIK_RESTRICT row_nzeros = num_nzeroes.PlaneRow(c, y);
      const int* PIK_RESTRICT row_nzeros_top =
          y % kTileHeight == 0 ? nullptr : num_nzeroes.ConstPlaneRow(c, y - 1);
      for (size_t x = 0, bx = 0; x < ac->xsize(); x += kBlockSize, ++bx) {
        memset(&row[x], 0, kBlockSize * sizeof(row[0]));
        const int block_ctx = row_bctx[bx];
        const int nzero_ctx = 128 + ContextFromTopAndLeft(
            row_nzeros_top, row_nzeros, block_ctx, bx);
        br->FillBitBuffer();
        row_nzeros[bx] =
            kIndexLut[decoder.ReadSymbol(context_map[nzero_ctx], br)];
        int num_nzeros = row_nzeros[bx];
        if (num_nzeros > 63) {
          return PIK_FAILURE("Invalid AC data.");
        }
        if (num_nzeros == 0) continue;
        const int histo_offset = 128 + kOrderContexts * 32 + block_ctx * 120;
        const int context2 = ZeroDensityContext(num_nzeros - 1, 0, 4);
        int histo_idx = context_map[histo_offset + context2];
        const int order_offset = block_ctx * 64;
        const int* PIK_RESTRICT block_order = &coeff_order[order_offset];
        for (int k = 1; k < 64 && num_nzeros > 0; ++k) {
          br->FillBitBuffer();
          int s = decoder.ReadSymbol(histo_idx, br);
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
          // block_order[k] != 0, only writes to AC coefficients.
          row[x + block_order[k]] = s;
        }
        if (num_nzeros != 0) {
          return PIK_FAILURE("Invalid AC data.");
        }
      }
    }
  }
  br->JumpToByteBoundary();
  if (!decoder.CheckANSFinalState()) {
    return PIK_FAILURE("ANS checksum failure.");
  }
  return true;
}

}  // namespace pik
