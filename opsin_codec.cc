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

void PredictDCTile(const Image3W& coeffs, Image3W* out) {
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

void UnpredictDCTile(Image3W* coeffs) {
  PIK_ASSERT(coeffs->xsize() % 64 == 0);
  ImageW dc_y(coeffs->xsize() / 64, coeffs->ysize());
  ImageW dc_xz(coeffs->xsize() / 64 * 2, coeffs->ysize());

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

  ImageW dc_y_out(coeffs->xsize() / 64, coeffs->ysize());
  ImageW dc_xz_out(coeffs->xsize() / 64 * 2, coeffs->ysize());

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

void ComputeCoeffOrder(const Image3W& img, const Image3B& block_ctx,
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

template <class EntropyEncodingData, class SymbolWriter>
struct EncodeImageInternal {
  template <class Processor>
  std::string operator()(const Image3W& img, Processor* processor,
                         PikImageSizeInfo* info) {
    // Build histograms.
    HistogramBuilder builder(Processor::num_contexts());
    ProcessImage3(img, processor, &builder);
    // Encode histograms.
    const size_t max_out_size = 2 * builder.EncodedSize(1, 2) + 1024;
    std::string output(max_out_size, 0);
    size_t storage_ix = 0;
    uint8_t* storage = reinterpret_cast<uint8_t*>(&output[0]);
    storage[0] = 0;
    std::vector<EntropyEncodingData> codes;
    std::vector<uint8_t> context_map;
    std::vector<double> entropy_per_context;
    builder.BuildAndStoreEntropyCodes(
        &codes, &context_map, &storage_ix, storage,
        info ? &entropy_per_context : nullptr, info);
    // Close the histogram bit stream.
    size_t jump_bits = ((storage_ix + 7) & ~7) - storage_ix;
    WriteBits(jump_bits, 0, &storage_ix, storage);
    PIK_ASSERT(storage_ix % 8 == 0);
    const size_t histo_bytes = storage_ix >> 3;
    // Entropy encode data.
    SymbolWriter symbol_writer(codes, context_map, &storage_ix, storage);
    ProcessImage3(img, processor, &symbol_writer);
    symbol_writer.FlushToBitStream();
    const size_t data_bits = storage_ix - 8 * histo_bytes;
    const size_t data_bytes = (data_bits + 7) >> 3;
    const int out_size = histo_bytes + data_bytes;
    PIK_CHECK(out_size <= max_out_size);
    output.resize(out_size);
    if (info) {
      info->num_clustered_histograms += codes.size();
      info->histogram_size += histo_bytes;
      info->entropy_coded_bits += data_bits - builder.num_extra_bits();
      info->extra_bits += builder.num_extra_bits();
      info->total_size += out_size;
      for (int c = 0; c < 3; ++c) {
        if (context_map.size() == 3) {
          info->entropy_per_channel[c] += entropy_per_context[c];
        } else if (context_map.size() == ACBlockProcessor::num_contexts()) {
          for (int i = 0; i < 32; ++i) {
            info->entropy_per_channel[c] += entropy_per_context[c * 32 + i];
          }
          for (int i = 0; i < 120; ++i) {
            info->entropy_per_channel[c] +=
                entropy_per_context[96 + c * 120 + i];
          }
        }
        info->entropy_per_channel[c] += builder.num_extra_bits(c);
      }
    }
    return output;
  }
};

template <class Processor>
size_t EncodedImageSizeInternal(const Image3W& img,
                                Processor* processor) {
  HistogramBuilder builder(Processor::num_contexts());
  ProcessImage3(img, processor, &builder);
  return builder.EncodedSize(1, 2);
}

std::string EncodeImage(const Image3W& img, int stride,
                        PikImageSizeInfo* info) {
  CoeffProcessor processor(stride);
  return EncodeImageInternal<ANSEncodingData, ANSSymbolWriter>()(
      img, &processor, info);
}

std::string EncodeAC(const Image3W& coeffs,
                     const Image3B& block_ctx,
                     PikInfo* pik_info) {
  ACBlockProcessor processor(&block_ctx, coeffs.xsize());
  int order[kOrderContexts * 64];
  ComputeCoeffOrder(coeffs, block_ctx, order);
  // TODO(user): Check this upper bound on size of encoded order.
  std::string encoded_coeff_order(kOrderContexts * 1024, 0);
  {
    uint8_t* storage = reinterpret_cast<uint8_t*>(&encoded_coeff_order[0]);
    size_t storage_ix = 0;
    for(int c = 0; c < kOrderContexts; c++) {
      EncodeCoeffOrder(&order[c * 64], &storage_ix, storage);
    }
    PIK_CHECK(storage_ix < encoded_coeff_order.size() * 8);
    encoded_coeff_order.resize((storage_ix + 7) / 8);
  }
  if (pik_info) {
    pik_info->layers[kLayerOrder].total_size += encoded_coeff_order.size();
  }
  PikImageSizeInfo* ac_info = pik_info ? &pik_info->layers[kLayerAC] : nullptr;
  processor.SetCoeffOrder(order);
  std::string output = EncodeImageInternal<ANSEncodingData, ANSSymbolWriter>()(
      coeffs, &processor, ac_info);
  return encoded_coeff_order + output;
}

PIK_INLINE uint32_t MakeToken(const uint32_t context, const uint32_t symbol,
                              const uint32_t nbits, const uint32_t bits) {
  return (context << 26) | (symbol << 18) | (nbits << 14) | bits;
}

std::string EncodeACFast(const Image3W& coeffs,
                         const Image3B& block_ctx,
                         PikInfo* pik_info) {
  // Build static context map.
  static const int kNumContexts = kOrderContexts * (32 + 120);
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
  static const int kNumStaticZdensContexts = 7;
  static const int kNumStaticContexts =
      4 * (1 + kNumStaticZdensContexts);
  PIK_ASSERT(kNumStaticContexts <= 64);
  std::vector<uint8_t> context_map(kNumContexts);
  for (int c = 0; c < kOrderContexts; ++c) {
    const int ctx = std::min(c, 3);
    for (int i = 0; i < 32; ++i) {
      context_map[c * 32 + i] = ctx;
    }
    for (int i = 0; i < 120; ++i) {
      context_map[kOrderContexts * 32 + c * 120 + i] =
          4 + ctx * kNumStaticZdensContexts +
          kStaticZdensContextMap[i];
    }
  }
  // Tokenize the coefficient stream.
  std::vector<uint32_t> tokens;
  tokens.reserve(3 * coeffs.xsize() * coeffs.ysize());
  size_t num_extra_bits = 0;
  for (int y = 0; y < coeffs.ysize(); ++y) {
    auto row = coeffs.Row(y);
    for (int x = 0; x < coeffs.xsize(); x += 64) {
      for (int c = 0; c < 3; ++c) {
        const int bctx = block_ctx.Row(y)[c][x / 64];
        const int16_t* coeffs = &row[c][x];
        int num_nzeros = 0;
        for (int k = 1; k < 64; ++k) {
          if (coeffs[k] != 0) ++num_nzeros;
        }
        tokens.push_back(MakeToken(std::min(bctx, 3), num_nzeros, 0, 0));
        if (num_nzeros == 0) continue;
        int r = 0;
        const int histo_offset = kOrderContexts * 32 + bctx * 120;
        int histo_idx = histo_offset + ZeroDensityContext(num_nzeros - 1, 0, 4);
        histo_idx = context_map[histo_idx];
        for (int k = 1; k < 64; ++k) {
          int16_t coeff = coeffs[kNaturalCoeffOrder[k]];
          if (coeff == 0) {
            r++;
            continue;
          }
          PIK_ASSERT(histo_idx < kNumContexts);
          while (r > 15) {
            tokens.push_back(MakeToken(histo_idx, kIndexLut[0xf0], 0, 0));
            r -= 16;
          }
          int nbits, bits;
          EncodeCoeff(coeff, &nbits, &bits);
          PIK_ASSERT(nbits <= 14);
          int symbol = kIndexLut[(r << 4) + nbits];
          tokens.push_back(MakeToken(histo_idx, symbol, nbits, bits));
          num_extra_bits += nbits;
          r = 0;
          histo_idx = histo_offset + ZeroDensityContext(num_nzeros - 1, k, 4);
          histo_idx = context_map[histo_idx];
          --num_nzeros;
        }
      }
    }
  }
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
  // Estimate total output size.
  size_t num_bits = num_extra_bits;
  for (int c = 0; c < kNumStaticContexts; ++c) {
    size_t histogram_bits;
    size_t data_bits;
    BuildHuffmanTreeAndCountBits(&histograms[c << 8], 256,
                                 &histogram_bits, &data_bits);
    num_bits += histogram_bits + data_bits;
  }
  size_t num_bytes = (num_bits + 7) >> 3;
  const size_t max_out_size = 2 * num_bytes + 1024;
  // Allocate output string.
  std::string output(max_out_size, 0);
  size_t storage_ix = 0;
  uint8_t* storage = reinterpret_cast<uint8_t*>(&output[0]);
  storage[0] = 0;
  for (int c = 0; c < kOrderContexts; c++)
    EncodeCoeffOrder(kNaturalCoeffOrder, &storage_ix, storage);
  WriteZeroesToByteBoundary(&storage_ix, storage);
  const size_t order_bytes = storage_ix >> 3;
  if (pik_info) {
    pik_info->layers[kLayerOrder].total_size += order_bytes;
  }
  // Encode the histograms.
  std::vector<ANSEncodingData> codes;
  EncodeContextMap(context_map, kNumStaticContexts, &storage_ix, storage);
  for (int c = 0; c < kNumStaticContexts; ++c) {
    ANSEncodingData code;
    code.BuildAndStore(&histograms[c << 8], 256, &storage_ix, storage);
    codes.emplace_back(std::move(code));
  }
  // Close the histogram bit stream.
  WriteZeroesToByteBoundary(&storage_ix, storage);
  const size_t histo_bytes = (storage_ix >> 3) - order_bytes;
  // Entropy encode data.
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
      }
      if (i > 0) {
        WriteBits(16, out[i - 1] & 0xffff, &storage_ix, storage);
      }
    }
  }
  const size_t data_bits = storage_ix - 8 * histo_bytes - 8 * order_bytes;
  const size_t data_bytes = (data_bits + 7) >> 3;
  const int out_size = order_bytes + histo_bytes + data_bytes;
  PIK_CHECK(out_size <= max_out_size);
  output.resize(out_size);
  if (ac_info) {
    ac_info->num_clustered_histograms += codes.size();
    ac_info->histogram_size += histo_bytes;
    ac_info->entropy_coded_bits += data_bits - num_extra_bits;
    ac_info->extra_bits += num_extra_bits;
    ac_info->total_size += out_size;
  }
  return output;
}

size_t EncodedImageSize(const Image3W& img, int stride) {
  CoeffProcessor processor(stride);
  return EncodedImageSizeInternal(img, &processor);
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

Image3F LocalACInformationDensity(const Image3W& coeffs) {
  Image3B block_ctx(coeffs.xsize() / 64, coeffs.ysize(), 0);
  ACBlockProcessor processor(&block_ctx, coeffs.xsize());
  int order[kOrderContexts * 64];
  ComputeCoeffOrder(coeffs, block_ctx, order);
  processor.SetCoeffOrder(order);
  HistogramBuilder builder(ACBlockProcessor::num_contexts());
  ProcessImage3(coeffs, &processor, &builder);
  std::vector<ANSEncodingData> codes;
  std::vector<uint8_t> context_map;
  builder.BuildAndStoreEntropyCodes(&codes, &context_map,
                                    nullptr, nullptr, nullptr, nullptr);
  ANSBitCounter counter(codes, context_map);
  Image3F out(coeffs.xsize() / processor.block_size(), coeffs.ysize());
  processor.Reset();
  for (int y = 0; y < coeffs.ysize(); ++y) {
    auto row_in = coeffs.Row(y);
    auto row_out = out.Row(y);
    for (int bx = 0; bx < out.xsize(); ++bx) {
      const int x = bx * processor.block_size();
      for (int c = 0; c < 3; ++c) {
        float nbits_start = counter.nbits();
        processor.ProcessBlock(&row_in[c][x], x, y, c, &counter);
        row_out[c][bx] = counter.nbits() - nbits_start;
      }
    }
  }
  return out;
}

static const int kArithCoderPrecision = 24;


static const uint8_t kSymbolLut[256] = {
  0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x21, 0x12,
  0x31, 0x41, 0x05, 0x51, 0x13, 0x61, 0x22, 0x71,
  0x81, 0x06, 0x91, 0x32, 0xa1, 0xf0, 0x14, 0xb1,
  0xc1, 0xd1, 0x23, 0x42, 0x52, 0xe1, 0xf1, 0x15,
  0x07, 0x62, 0x33, 0x72, 0x24, 0x82, 0x92, 0x43,
  0xa2, 0x53, 0xb2, 0x16, 0x34, 0xc2, 0x63, 0x73,
  0x25, 0xd2, 0x93, 0x83, 0x44, 0x54, 0xa3, 0xb3,
  0xc3, 0x35, 0xd3, 0x94, 0x17, 0x45, 0x84, 0x64,
  0x55, 0x74, 0x26, 0xe2, 0x08, 0x75, 0xc4, 0xd4,
  0x65, 0xf2, 0x85, 0x95, 0xa4, 0xb4, 0x36, 0x46,
  0x56, 0xe3, 0xa5, 0x09, 0x0a, 0x0b, 0x0c, 0x0d,
  0x0e, 0x0f, 0x10, 0x18, 0x19, 0x1a, 0x1b, 0x1c,
  0x1d, 0x1e, 0x1f, 0x20, 0x27, 0x28, 0x29, 0x2a,
  0x2b, 0x2c, 0x2d, 0x2e, 0x2f, 0x30, 0x37, 0x38,
  0x39, 0x3a, 0x3b, 0x3c, 0x3d, 0x3e, 0x3f, 0x40,
  0x47, 0x48, 0x49, 0x4a, 0x4b, 0x4c, 0x4d, 0x4e,
  0x4f, 0x50, 0x57, 0x58, 0x59, 0x5a, 0x5b, 0x5c,
  0x5d, 0x5e, 0x5f, 0x60, 0x66, 0x67, 0x68, 0x69,
  0x6a, 0x6b, 0x6c, 0x6d, 0x6e, 0x6f, 0x70, 0x76,
  0x77, 0x78, 0x79, 0x7a, 0x7b, 0x7c, 0x7d, 0x7e,
  0x7f, 0x80, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x8b,
  0x8c, 0x8d, 0x8e, 0x8f, 0x90, 0x96, 0x97, 0x98,
  0x99, 0x9a, 0x9b, 0x9c, 0x9d, 0x9e, 0x9f, 0xa0,
  0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xab, 0xac, 0xad,
  0xae, 0xaf, 0xb0, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9,
  0xba, 0xbb, 0xbc, 0xbd, 0xbe, 0xbf, 0xc0, 0xc5,
  0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xcb, 0xcc, 0xcd,
  0xce, 0xcf, 0xd0, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9,
  0xda, 0xdb, 0xdc, 0xdd, 0xde, 0xdf, 0xe0, 0xe4,
  0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xeb, 0xec,
  0xed, 0xee, 0xef, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7,
  0xf8, 0xf9, 0xfa, 0xfb, 0xfc, 0xfd, 0xfe, 0xff,
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
                     Image3W* const PIK_RESTRICT img) {
  for (int y = 0; y < img->ysize(); ++y) {
    auto row = img->Row(y);
    for (int x = 0; x < img->xsize(); x += stride) {
      for (int c = 0; c < 3; ++c) {
        br->FillBitBuffer();
        int histo_idx = context_map[c];
        int s = decoder->ReadSymbol(histo_idx, br);
        if (s > 0) {
          int bits = br->PeekBits(s);
          br->Advance(s);
          s = bits < (1 << (s - 1)) ? bits + ((~0U) << s ) + 1 : bits;
        }
        row[c][x] = s;
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
                  Image3W* const PIK_RESTRICT coeffs) {
  std::vector<uint8_t> prev_num_nzeros(3 * (coeffs->xsize() >> 6));
  for (int y = 0; y < coeffs->ysize(); ++y) {
    auto row = coeffs->Row(y);
    auto row_bctx = block_ctx.Row(y);
    for (int x = 0, ix = 0, bx = 0; x < coeffs->xsize(); x += 64, ++bx) {
      for (int c = 0; c < 3; ++c, ++ix) {
        memset(&row[c][x + 1], 0, 63 * sizeof(row[0][0]));
        br->FillBitBuffer();
        const int block_ctx = row_bctx[c][bx];
        const int context1 = block_ctx * 32 +
            NumNonZerosContext(x, y, &prev_num_nzeros[ix]);
        int num_nzeros =
            kIndexLut[decoder->ReadSymbol(context_map[context1], br)];
        if (num_nzeros > 63) {
          return PIK_FAILURE("Invalid AC data.");
        }
        prev_num_nzeros[ix] = num_nzeros;
        if (num_nzeros == 0) continue;
        const int histo_offset = kOrderContexts * 32 + block_ctx * 120;
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
          row[c][x + block_order[k]] = s;
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

bool DecodeImage(BitReader* br, int stride,  Image3W* coeffs) {
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

bool DecodeAC(const Image3B& block_ctx, BitReader* br, Image3W* coeffs) {
  std::vector<uint8_t> context_map;
  int coeff_order[kOrderContexts * 64];
  for (int c = 0; c < kOrderContexts; ++c) {
    DecodeCoeffOrder(&coeff_order[c * 64], br);
  }
  br->JumpToByteBoundary();
  ANSCode code;
  if (!DecodeHistograms(br, ACBlockProcessor::num_contexts(), 256, kSymbolLut,
                        sizeof(kSymbolLut), &code, &context_map)) {
    return false;
  }
  ANSSymbolReader decoder(&code);
  if (!DecodeACData(br, context_map, block_ctx,
                    &decoder, coeff_order, coeffs)) {
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
  ContextProcessor(int minval, int maxval, int x_tilesize, int y_tilesize)
      : minval_(minval),
        maxval_(maxval),
        x_tilesize_(x_tilesize),
        y_tilesize_(y_tilesize) {
    Reset();
  }

  void Reset() { row_ = std::vector<int>(x_tilesize_); }

  int block_size() const { return 1; }
  int num_contexts() { return 1 + ((maxval_ - minval_) >> kContextShift); }

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
    return (PredictVal(x, y, c) - minval_) >> kContextShift;
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
};

}  // namespace

EncodedIntPlane EncodePlane(const Image<int>& img, int minval, int maxval,
                            int tile_size, PikImageSizeInfo* info) {
  EncodedIntPlane result;
  ContextProcessor processor(minval, maxval, tile_size, tile_size);
  HistogramBuilder builder(processor.num_contexts());
  const size_t tile_xsize = (img.xsize() + tile_size - 1) / tile_size;
  const size_t tile_ysize = (img.ysize() + tile_size - 1) / tile_size;
  for (int y = 0; y < tile_ysize; y++) {
    for (int x = 0; x < tile_xsize; x++) {
      ConstWrapper<Image<int>> tile =
          ConstWindow(img, x * tile_size, y * tile_size, tile_size, tile_size);
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
      ConstWrapper<Image<int>> tile =
          ConstWindow(img, x * tile_size, y * tile_size, tile_size, tile_size);
      ProcessImage(tile.get(), &processor, &symbol_writer);
      symbol_writer.FlushToBitStream();
      const int out_size = (storage_ix + 7) >> 3;
      PIK_CHECK(out_size <= max_out_size);
      result.tiles[y][x].resize(out_size);
      if (info) {
        info->entropy_coded_bits += storage_ix - builder.num_extra_bits();
        info->extra_bits += builder.num_extra_bits();
        info->total_size += out_size;
      }
    }
  }
  return result;
}

bool IntPlaneDecoder::LoadPreamble(BitReader* br) {
  PIK_CHECK(!ready_);
  ContextProcessor processor(minval_, maxval_, tile_size_, tile_size_);
  const size_t max_alphabet_size = maxval_ - minval_ + 1;
  if (!DecodeHistograms(br, processor.num_contexts(), max_alphabet_size,
                        nullptr, 0, &ans_code_, &context_map_)) {
    return false;
  }
  br->JumpToByteBoundary();
  ready_ = true;
  return true;
}

bool IntPlaneDecoder::DecodeTile(BitReader* br, Image<int>* img) {
  ContextProcessor processor(minval_, maxval_, tile_size_, tile_size_);
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
