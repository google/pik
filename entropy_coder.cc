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

void ShrinkDC(const Rect& rect_dc, const Image3S& dc,
              Image3S* PIK_RESTRICT tmp_residuals) {
  const size_t xsize = rect_dc.xsize();
  const size_t ysize = rect_dc.ysize();
  PIK_ASSERT(tmp_residuals->xsize() >= xsize);
  PIK_ASSERT(tmp_residuals->ysize() >= ysize);
  const Rect tmp_rect(0, 0, xsize, ysize);

  ShrinkY(rect_dc, dc.Plane(1), tmp_rect, tmp_residuals->MutablePlane(1));

  ImageS tmp_xz(xsize * 2, ysize);

  // Interleave X and Z into XZ for ShrinkXB.
  for (size_t y = 0; y < ysize; ++y) {
    const int16_t* PIK_RESTRICT row_x = rect_dc.ConstRow(dc.Plane(0), y);
    const int16_t* PIK_RESTRICT row_z = rect_dc.ConstRow(dc.Plane(2), y);
    int16_t* PIK_RESTRICT row_xz = tmp_xz.Row(y);

    for (size_t x = 0; x < xsize; ++x) {
      row_xz[2 * x + 0] = row_x[x];
      row_xz[2 * x + 1] = row_z[x];
    }
  }

  ImageS tmp_xz_residuals(xsize * 2, ysize);
  ShrinkXB(rect_dc, dc.Plane(1), tmp_xz, &tmp_xz_residuals);

  // Deinterleave XZ into residuals X and Z.
  for (size_t y = 0; y < ysize; ++y) {
    const int16_t* PIK_RESTRICT row_xz = tmp_xz_residuals.ConstRow(y);
    int16_t* PIK_RESTRICT row_out_x = tmp_residuals->PlaneRow(0, y);
    int16_t* PIK_RESTRICT row_out_z = tmp_residuals->PlaneRow(2, y);

    for (size_t x = 0; x < xsize; ++x) {
      row_out_x[x] = row_xz[2 * x + 0];
      row_out_z[x] = row_xz[2 * x + 1];
    }
  }
}

void ExpandDC(const Rect& rect_dc, Image3S* PIK_RESTRICT dc,
              ImageS* PIK_RESTRICT tmp_y, ImageS* PIK_RESTRICT tmp_xz_residuals,
              ImageS* PIK_RESTRICT tmp_xz_expanded) {
  const size_t xsize = rect_dc.xsize();
  const size_t ysize = rect_dc.ysize();
  PIK_ASSERT(xsize <= tmp_y->xsize() && ysize <= tmp_y->ysize());
  PIK_ASSERT(SameSize(*tmp_xz_residuals, *tmp_xz_expanded));

  ExpandY(rect_dc, dc->Plane(1), tmp_y);

  // The predictor expects a single image with interleaved X and Z.
  for (size_t y = 0; y < ysize; ++y) {
    const int16_t* PIK_RESTRICT row0 = rect_dc.ConstRow(dc->Plane(0), y);
    const int16_t* PIK_RESTRICT row2 = rect_dc.ConstRow(dc->Plane(2), y);
    int16_t* PIK_RESTRICT row_xz = tmp_xz_residuals->Row(y);

    for (size_t x = 0; x < xsize; ++x) {
      row_xz[2 * x + 0] = row0[x];
      row_xz[2 * x + 1] = row2[x];
    }
  }

  ExpandXB(xsize, ysize, *tmp_y, *tmp_xz_residuals, tmp_xz_expanded);

  if (rect_dc.x0() == 0 && rect_dc.y0() == 0 && SameSize(*dc, *tmp_y)) {
    // Avoid copying Y; tmp_y remains valid for next call.
    dc->MutablePlane(1)->Swap(*tmp_y);
  } else {
    for (size_t y = 0; y < ysize; ++y) {
      const int16_t* PIK_RESTRICT row_from = tmp_y->ConstRow(y);
      int16_t* PIK_RESTRICT row_to = rect_dc.Row(dc->MutablePlane(1), y);
      memcpy(row_to, row_from, xsize * sizeof(row_to[0]));
    }
  }

  // Deinterleave |tmp_xz_expanded| and copy into |dc|.
  for (size_t y = 0; y < ysize; ++y) {
    const int16_t* PIK_RESTRICT row_xz = tmp_xz_expanded->ConstRow(y);
    int16_t* PIK_RESTRICT row_out0 = rect_dc.Row(dc->MutablePlane(0), y);
    int16_t* PIK_RESTRICT row_out2 = rect_dc.Row(dc->MutablePlane(2), y);

    for (size_t x = 0; x < xsize; ++x) {
      row_out0[x] = row_xz[2 * x + 0];
      row_out2[x] = row_xz[2 * x + 1];
    }
  }
}

void ComputeCoeffOrder(const Image3S& ac, const Image3B& block_ctx,
                       int32_t* PIK_RESTRICT order) {
  for (uint8_t ctx = 0; ctx < kOrderContexts; ++ctx) {
    int32_t num_zeros[kBlockSize] = {0};
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
    struct PosAndCount {
      uint32_t pos;
      uint32_t count;
    };
    PosAndCount pos_and_val[kBlockSize];
    for (size_t i = 0; i < kBlockSize; ++i) {
      pos_and_val[i].pos = i;
      pos_and_val[i].count = num_zeros[kNaturalCoeffOrder[i]];
    }
    std::stable_sort(pos_and_val, pos_and_val + kBlockSize,
                     [](const PosAndCount& a, const PosAndCount& b) -> bool {
                       return a.count < b.count;
                     });
    for (size_t i = 0; i < kBlockSize; ++i) {
      order[ctx * kBlockSize + i] = kNaturalCoeffOrder[pos_and_val[i].pos];
    }
  }
}

void EncodeCoeffOrder(const int32_t* PIK_RESTRICT order,
                      size_t* PIK_RESTRICT storage_ix, uint8_t* storage) {
  const int32_t kJPEGZigZagOrder[kBlockSize] = {
      0,  1,  5,  6,  14, 15, 27, 28, 2,  4,  7,  13, 16, 26, 29, 42,
      3,  8,  12, 17, 25, 30, 41, 43, 9,  11, 18, 24, 31, 40, 44, 53,
      10, 19, 23, 32, 39, 45, 52, 54, 20, 22, 33, 38, 46, 51, 55, 60,
      21, 34, 37, 47, 50, 56, 59, 61, 35, 36, 48, 49, 57, 58, 62, 63};
  int32_t order_zigzag[kBlockSize];
  for (size_t i = 0; i < kBlockSize; ++i) {
    order_zigzag[i] = kJPEGZigZagOrder[order[i]];
  }
  int32_t lehmer[kBlockSize];
  ComputeLehmerCode(order_zigzag, kBlockSize, lehmer);
  int32_t end = kBlockSize - 1;
  while (end >= 1 && lehmer[end] == 0) {
    --end;
  }
  for (int32_t i = 1; i <= end; ++i) {
    ++lehmer[i];
  }
  static const int32_t kSpan = 16;
  for (int32_t i = 0; i < kBlockSize; i += kSpan) {
    const int32_t start = (i > 0) ? i : 1;
    const int32_t end = i + kSpan;
    int32_t has_non_zero = 0;
    for (int32_t j = start; j < end; ++j) has_non_zero |= lehmer[j];
    if (!has_non_zero) {   // all zero in the span -> escape
      WriteBits(1, 0, storage_ix, storage);
      continue;
    } else {
      WriteBits(1, 1, storage_ix, storage);
    }
    for (int32_t j = start; j < end; ++j) {
      int32_t v;
      PIK_ASSERT(lehmer[j] <= kBlockSize);
      for (v = lehmer[j]; v >= 7; v -= 7) {
        WriteBits(3, 7, storage_ix, storage);
      }
      WriteBits(3, v, storage_ix, storage);
    }
  }
}

// Fills "tmp_num_nzeros" with per-block count of non-zero coefficients in
// "coeffs" within "rect".
void ExtractNumNZeroes(const Rect& rect, const ImageS& coeffs,
                       ImageI* PIK_RESTRICT tmp_num_nzeros) {
  const size_t xsize = rect.xsize();
  const size_t ysize = rect.ysize();
  PIK_CHECK(coeffs.xsize() % kBlockSize == 0);
  for (size_t y = 0; y < ysize; ++y) {
    const int16_t* PIK_RESTRICT coeffs_row =
        coeffs.ConstRow(rect.y0() + y) + rect.x0() * kBlockSize;
    int32_t* PIK_RESTRICT output_row = tmp_num_nzeros->Row(y);
    for (size_t x = 0; x < xsize; ++x) {
      size_t num_nzeros = 0;
      for (size_t i = 1; i < kBlockSize; ++i) {
        num_nzeros += coeffs_row[x * kBlockSize + i] != 0;
      }
      output_row[x] = static_cast<int32_t>(num_nzeros);
    }
  }
}

std::string EncodeCoeffOrders(const int32_t* order, PikInfo* pik_info) {
  std::string encoded_coeff_order(kOrderContexts * 1024, 0);
  uint8_t* storage = reinterpret_cast<uint8_t*>(&encoded_coeff_order[0]);
  size_t storage_ix = 0;
  for (size_t c = 0; c < kOrderContexts; c++) {
    EncodeCoeffOrder(&order[c * kBlockSize], &storage_ix, storage);
  }
  PIK_CHECK(storage_ix < encoded_coeff_order.size() * kBitsPerByte);
  encoded_coeff_order.resize((storage_ix + 7) / kBitsPerByte);
  if (pik_info) {
    pik_info->layers[kLayerOrder].total_size += encoded_coeff_order.size();
  }
  return encoded_coeff_order;
}

static const size_t kNumStaticZdensContexts = 7;
static const size_t kNumStaticContexts = 12 + 4 * kNumStaticZdensContexts;

// Layout of the context map:
//    0 -  127 : context for quantization levels,
//               computed from top and left values
//  128 -  319 : context for number of non-zeros in the block
//               computed from block context (0..5) and top and left
//               values (0..31)
//  320 - 1039 : context for AC coefficient symbols, computed from
//               block context, number of non-zeros left and index in scan order
std::vector<uint8_t> StaticContextMap() {
  static const int32_t kStaticZdensContextMap[120] = {
      0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 0, 0, 1, 1, 1, 1, 2, 2,
      2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
      4, 4, 4, 4, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 5, 2, 2,
      2, 2, 2, 3, 3, 3, 3, 3, 6, 6, 6, 6, 5, 5, 2, 2, 2, 2, 2, 3, 3, 3, 3, 6,
      6, 6, 6, 5, 5, 2, 2, 2, 2, 2, 3, 3, 3, 6, 6, 6, 6, 5, 5, 2, 2, 2, 2, 2,
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
  PIK_ASSERT(kNumStaticContexts <= kBlockSize);
  std::vector<uint8_t> context_map(kNumContexts);
  memcpy(&context_map[0], kStaticQuantContextMap,
         sizeof(kStaticQuantContextMap));
  for (size_t c = 0; c < kOrderContexts; ++c) {
    const size_t ctx = std::min<size_t>(c, 3);
    for (size_t i = 0; i < 32; ++i) {
      context_map[128 + c * 32 + i] = 8 + ctx;
    }
    for (size_t i = 0; i < 120; ++i) {
      context_map[128 + kOrderContexts * 32 + c * 120 + i] =
          12 + ctx * kNumStaticZdensContexts + kStaticZdensContextMap[i];
    }
  }
  return context_map;
}

PIK_INLINE int32_t PredictFromTopAndLeft(
    const int32_t* const PIK_RESTRICT row_top,
    const int32_t* const PIK_RESTRICT row, size_t x, int32_t default_val) {
  if (x % kTileWidth == 0) {
    return row_top == nullptr ? default_val : row_top[x];
  }
  if (row_top == nullptr) {
    return row[x - 1];
  }
  return (row_top[x] + row[x - 1] + 1) / 2;
}

PIK_INLINE int32_t ContextFromTopAndLeft(
    const int32_t* const PIK_RESTRICT row_top,
    const int32_t* const PIK_RESTRICT row, int32_t block_ctx, int32_t x) {
  int32_t pred = PredictFromTopAndLeft(row_top, row, x, 32);
  return kOrderContexts * (pred >> 1) + block_ctx;
}

PIK_INLINE void EncodeCoeff(int32_t coeff, int* nbits, int* bits) {
  int coeff_bits = coeff;
  if (coeff < 0) {
    coeff = -coeff;
    coeff_bits--;
  }
  *nbits = Log2Floor(coeff) + 1;
  *bits = coeff_bits & ((1 << *nbits) - 1);
}

std::vector<Token> TokenizeCoefficients(const int32_t* orders, const Rect& rect,
                                        const ImageI& quant_field,
                                        const Image3S& coeffs,
                                        const Image3B& block_ctx) {
  const size_t xsize = rect.xsize();
  const size_t ysize = rect.ysize();
  PIK_ASSERT(SameSize(quant_field, block_ctx));

  std::vector<Token> tokens;
  tokens.reserve(3 * coeffs.xsize() * coeffs.ysize());

  // Compute actual quant values from prediction residuals.
  for (size_t y = 0; y < ysize; ++y) {
    const int32_t* PIK_RESTRICT row_quant = rect.ConstRow(quant_field, y);
    const int32_t* PIK_RESTRICT row_quant_top =
        y % kTileHeight == 0 ? nullptr : rect.ConstRow(quant_field, y - 1);
    for (size_t bx = 0; bx < xsize; ++bx) {
      int32_t quant_pred =
          PredictFromTopAndLeft(row_quant_top, row_quant, bx, 32);
      int32_t quant_ctx = (quant_pred - 1) >> 1;
      tokens.emplace_back(Token(quant_ctx, row_quant[bx] - 1, 0, 0));
    }
  }

  ImageI tmp_num_nzeros(rect.xsize(), rect.ysize());
  for (int c = 0; c < 3; ++c) {
    ExtractNumNZeroes(rect, coeffs.Plane(c), &tmp_num_nzeros);
    for (size_t y = 0; y < ysize; ++y) {
      const int16_t* PIK_RESTRICT row =
          coeffs.ConstPlaneRow(c, rect.y0() + y) + rect.x0() * kBlockSize;
      const uint8_t* PIK_RESTRICT ctx_row =
          rect.ConstRow(block_ctx.Plane(c), y);
      const int32_t* PIK_RESTRICT row_nzeros = tmp_num_nzeros.ConstRow(y);
      const int32_t* PIK_RESTRICT row_nzeros_top =
          y % kTileHeight == 0 ? nullptr : tmp_num_nzeros.ConstRow(y - 1);
      for (size_t bx = 0; bx < xsize; ++bx) {
        const int bctx = ctx_row[bx];
        const int32_t* order = &orders[bctx * kBlockSize];
        const int16_t* PIK_RESTRICT block = row + bx * kBlockSize;
        size_t num_nzeros = row_nzeros[bx];
        int32_t nzero_ctx =
            128 + ContextFromTopAndLeft(row_nzeros_top, row_nzeros, bctx, bx);
        tokens.emplace_back(Token(nzero_ctx, num_nzeros, 0, 0));
        if (num_nzeros == 0) continue;
        int r = 0;
        const int histo_offset = 128 + kOrderContexts * 32 + bctx * 120;
        int histo_idx = histo_offset + ZeroDensityContext(num_nzeros - 1, 0, 4);
        for (size_t k = 1; k < kBlockSize; ++k) {
          int16_t coeff = block[order[k]];
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
  for (size_t i = 0; i < codes_len; ++i) {
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
                        kBlockSize, &clustered_histograms, &histogram_symbols);
      for (size_t c = 0; c < histograms_.size(); ++c) {
        (*context_map)[c] = static_cast<uint8_t>(histogram_symbols[c]);
      }
      if (storage_ix != nullptr && storage != nullptr) {
        EncodeContextMap(*context_map, clustered_histograms.size(),
                         storage_ix, storage);
      }
    }
    if (info) {
      for (size_t i = 0; i < clustered_histograms.size(); ++i) {
        info->clustered_entropy += clustered_histograms[i].ShannonEntropy();
      }
    }
    for (size_t c = 0; c < clustered_histograms.size(); ++c) {
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
      for (size_t i = 0; i < other.data_.size(); ++i) {
        data_[i] += other.data_[i];
      }
      total_count_ += other.total_count_;
    }
    float PopulationCost() const {
      std::vector<int> counts(data_.size());
      for (size_t i = 0; i < data_.size(); ++i) {
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
    size_t num_contexts, const std::vector<std::vector<Token> >& tokens,
    std::vector<ANSEncodingData>* codes, std::vector<uint8_t>* context_map,
    PikImageSizeInfo* info) {
  // Build histograms.
  HistogramBuilder builder(num_contexts);
  for (size_t i = 0; i < tokens.size(); ++i) {
    for (size_t j = 0; j < tokens[i].size(); ++j) {
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
  PIK_ASSERT(storage_ix % kBitsPerByte == 0);
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
  for (size_t i = 0; i < tokens.size(); ++i) {
    for (size_t j = 0; j < tokens[i].size(); ++j) {
      const Token token = tokens[i][j];
      const uint32_t histo_idx = (*context_map)[token.context];
      ++histograms[(histo_idx << 8) + token.symbol];
    }
  }
  if (info) {
    for (size_t c = 0; c < kNumStaticContexts; ++c) {
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
  for (size_t c = 0; c < kNumStaticContexts; ++c) {
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

std::string EncodeImage(const Rect& rect, const Image3S& img,
                        PikImageSizeInfo* info) {
  const size_t xsize = rect.xsize();
  const size_t ysize = rect.ysize();

  std::vector<std::vector<Token> > tokens(1);
  tokens[0].reserve(3 * ysize * xsize);
  for (int c = 0; c < 3; ++c) {
    for (size_t y = 0; y < ysize; ++y) {
      const int16_t* const PIK_RESTRICT row = rect.ConstRow(img.Plane(c), y);
      for (size_t x = 0; x < xsize; ++x) {
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
                     ANSSymbolReader* PIK_RESTRICT decoder, const Rect& rect,
                     Image3S* PIK_RESTRICT img) {
  const size_t xsize = rect.xsize();
  const size_t ysize = rect.ysize();
  PIK_ASSERT(xsize <= img->xsize() && ysize <= img->ysize());
  for (int c = 0; c < 3; ++c) {
    const int histo_idx = context_map[c];

    for (size_t y = 0; y < ysize; ++y) {
      int16_t* PIK_RESTRICT row = rect.Row(img->MutablePlane(c), y);

      for (size_t x = 0; x < xsize; ++x) {
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

bool DecodeCoeffOrder(int32_t* order, BitReader* br) {
  int32_t lehmer[kBlockSize] = {0};
  static const int32_t kSpan = 16;
  for (int32_t i = 0; i < kBlockSize; i += kSpan) {
    br->FillBitBuffer();
    const int32_t has_non_zero = br->ReadBits(1);
    if (!has_non_zero) continue;
    const int32_t start = (i > 0) ? i : 1;
    const int32_t end = i + kSpan;
    for (int32_t j = start; j < end; ++j) {
      int32_t v = 0;
      while (v <= kBlockSize) {
        br->FillBitBuffer();
        const int32_t bits = br->ReadBits(3);
        v += bits;
        if (bits < 7) break;
      }
      if (v > kBlockSize) v = kBlockSize;
      lehmer[j] = v;
    }
  }
  int32_t end = kBlockSize - 1;
  while (end > 0 && lehmer[end] == 0) {
    --end;
  }
  for (int32_t i = 1; i <= end; ++i) {
    --lehmer[i];
  }
  DecodeLehmerCode(lehmer, kBlockSize, order);
  for (size_t k = 0; k < kBlockSize; ++k) {
    order[k] = kNaturalCoeffOrder[order[k]];
  }
  return true;
}

bool DecodeImage(BitReader* PIK_RESTRICT br, const Rect& rect,
                 Image3S* PIK_RESTRICT img) {
  std::vector<uint8_t> context_map;
  ANSCode code;
  if (!DecodeHistograms(br, 3, 16, nullptr, 0, &code, &context_map)) {
    return false;
  }
  ANSSymbolReader decoder(&code);
  if (!DecodeImageData(br, context_map, &decoder, rect, img)) {
    return false;
  }
  if (!decoder.CheckANSFinalState()) {
    return PIK_FAILURE("ANS checksum failure.");
  }
  return true;
}

bool DecodeAC(const Image3B& tmp_block_ctx, const ANSCode& code,
              const std::vector<uint8_t>& context_map,
              const int32_t* PIK_RESTRICT coeff_order,
              BitReader* PIK_RESTRICT br, const Rect& rect_ac,
              Image3S* PIK_RESTRICT ac, const Rect& rect_qf,
              ImageI* PIK_RESTRICT quant_field,
              Image3I* PIK_RESTRICT tmp_num_nzeroes) {
  const size_t xsize = rect_ac.xsize();
  const size_t ysize = rect_ac.ysize();
  PIK_ASSERT(SameSize(rect_ac, rect_qf));
  PIK_ASSERT(ac->xsize() % kBlockSize == 0);
  PIK_ASSERT(xsize <= ac->xsize() / kBlockSize && ysize <= ac->ysize());
  PIK_ASSERT(xsize <= quant_field->xsize() && ysize <= quant_field->ysize());
  PIK_ASSERT(SameSize(tmp_block_ctx, *tmp_num_nzeroes));

  ANSSymbolReader decoder(&code);
  for (size_t y = 0; y < ysize; ++y) {
    int32_t* PIK_RESTRICT row_quant = rect_qf.Row(quant_field, y);
    const int32_t* PIK_RESTRICT row_quant_top =
        y % kTileHeight == 0 ? nullptr : rect_qf.ConstRow(*quant_field, y - 1);

    for (size_t bx = 0; bx < xsize; ++bx) {
      br->FillBitBuffer();
      int32_t quant_pred =
          PredictFromTopAndLeft(row_quant_top, row_quant, bx, 32);
      int32_t quant_ctx = (quant_pred - 1) >> 1;
      row_quant[bx] =
          kIndexLut[decoder.ReadSymbol(context_map[quant_ctx], br)] + 1;
    }
  }

  for (int c = 0; c < 3; ++c) {
    for (size_t y = 0; y < ysize; ++y) {
      const uint8_t* PIK_RESTRICT row_bctx = tmp_block_ctx.ConstPlaneRow(c, y);
      int16_t* PIK_RESTRICT row_ac =
          ac->PlaneRow(c, rect_ac.y0() + y) + rect_ac.x0() * kBlockSize;
      int32_t* PIK_RESTRICT row_nzeros = tmp_num_nzeroes->PlaneRow(c, y);
      const int32_t* PIK_RESTRICT row_nzeros_top =
          y % kTileHeight == 0 ? nullptr
                               : tmp_num_nzeroes->ConstPlaneRow(c, y - 1);

      for (size_t bx = 0; bx < xsize; ++bx) {
        int16_t* PIK_RESTRICT block_ac = row_ac + bx * kBlockSize;
        memset(block_ac, 0, kBlockSize * sizeof(row_ac[0]));
        const size_t block_ctx = row_bctx[bx];
        PIK_ASSERT(block_ctx < kOrderContexts);
        const size_t nzero_ctx =
            128 +
            ContextFromTopAndLeft(row_nzeros_top, row_nzeros, block_ctx, bx);
        br->FillBitBuffer();
        row_nzeros[bx] =
            kIndexLut[decoder.ReadSymbol(context_map[nzero_ctx], br)];
        size_t num_nzeros = row_nzeros[bx];
        if (num_nzeros > kBlockSize - 1) {
          return PIK_FAILURE("Invalid AC: nzeros too large");
        }
        if (num_nzeros == 0) continue;
        const int histo_offset = 128 + kOrderContexts * 32 + block_ctx * 120;
        const int context2 = ZeroDensityContext(num_nzeros - 1, 0, 4);
        int histo_idx = context_map[histo_offset + context2];
        const size_t order_offset = block_ctx * kBlockSize;
        const int* PIK_RESTRICT block_order = &coeff_order[order_offset];
        for (size_t k = 1; k < kBlockSize && num_nzeros > 0; ++k) {
          br->FillBitBuffer();
          int s = decoder.ReadSymbol(histo_idx, br);
          k += (s >> 4);
          if (k + num_nzeros > kBlockSize) {
            return PIK_FAILURE("Invalid AC data.");
          }
          s &= 15;
          if (s > 0) {
            int32_t bits = br->PeekBits(s);
            br->Advance(s);
            s = bits < (1 << (s - 1)) ? bits + ((~0U) << s ) + 1 : bits;
            const int context = ZeroDensityContext(num_nzeros - 1, k, 4);
            histo_idx = context_map[histo_offset + context];
            --num_nzeros;
          }
          // block_order[k] != 0, only writes to AC coefficients.
          block_ac[block_order[k]] = s;
        }
        if (num_nzeros != 0) {
          return PIK_FAILURE("Invalid AC: nzeros not 0.");
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
