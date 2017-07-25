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

#include <string.h>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "bit_reader.h"
#include "compiler_specific.h"
#include "context_map_decode.h"
#include "dc_predictor.h"
#include "dc_predictor_slow.h"
#include "fast_log.h"
#include "histogram_decode.h"
#include "huffman_decode.h"
#include "huffman_encode.h"
#include "status.h"
#include "write_bits.h"

namespace pik {

static const int kANSBufferSize = 1 << 16;

static inline int SymbolFromSignedInt(int diff) {
  return diff >= 0 ? 2 * diff : -2 * diff - 1;
}

static inline int SignedIntFromSymbol(int symbol) {
  return symbol % 2 == 0 ? symbol / 2 : (-symbol - 1) / 2;
}

void PredictDCBlock(size_t x, size_t y, size_t xsize, int row_stride,
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

Image3W PredictDC(const Image3W& coeffs) {
  Image3W out(coeffs.xsize() / 64, coeffs.ysize());
  const size_t row_stride = coeffs.plane(0).bytes_per_row() / sizeof(int16_t);
  for (int y = 0; y < out.ysize(); y++) {
    auto row_in = coeffs.Row(y);
    auto row_out = out.Row(y);
    for (int x = 0; x < out.xsize(); x++) {
      PredictDCBlock(x, y, out.xsize(), row_stride, row_in, row_out);
    }
  }
  return out;
}

void UpdateDCPrediction(const Image3W& coeffs,
                        const int block_x, const int block_y,
                        Image3W* dc_residuals,
                        HistogramBuilder* builder) {
  const size_t row_stride = coeffs.plane(0).bytes_per_row() / sizeof(int16_t);
  const int block_x0 = std::max(0, block_x - 1);
  const int block_y0 = block_y;
  const int block_x1 = std::min<int>(dc_residuals->xsize(), block_x + 3);
  const int block_y1 = std::min<int>(dc_residuals->ysize(), block_y + 3);
  for (int y = block_y0; y < block_y1; ++y) {
    auto row_in = coeffs.Row(y);
    auto row_out = dc_residuals->Row(y);
    for (int x = block_x0; x < block_x1; ++x) {
      builder->set_weight(-1);
      for (int c = 0; c < 3; ++c) {
        VisitCoefficient(row_out[c][x], c, builder);
      }
      PredictDCBlock(x, y, dc_residuals->xsize(), row_stride, row_in, row_out);
      builder->set_weight(1);
      for (int c = 0; c < 3; ++c) {
        VisitCoefficient(row_out[c][x], c, builder);
      }
    }
  }
}

void UnpredictDC(Image3W* coeffs) {
  Image<int32_t> dc_y(coeffs->xsize() / 64, coeffs->ysize());
  Image<int32_t> dc_xz(coeffs->xsize() / 64 * 2, coeffs->ysize());

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

  Image<int32_t> dc_y_out(coeffs->xsize() / 64, coeffs->ysize());
  Image<int32_t> dc_xz_out(coeffs->xsize() / 64 * 2, coeffs->ysize());

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
  size_t total_data_bits = num_extra_bits_;
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

struct HuffmanEncodingData {
  void BuildAndStore(const uint32_t* histogram, size_t histo_size,
                     size_t* storage_ix, uint8_t* storage) {
    depths.resize(histo_size);
    bits.resize(histo_size);
    BuildAndStoreHuffmanTree(histogram, histo_size,
                             depths.data(), bits.data(),
                             storage_ix, storage);
  }

  std::vector<uint8_t> depths;
  std::vector<uint16_t> bits;
};

// Symbol visitor that emits the entropy coded symbols to the bit stream.
class HuffmanSymbolWriter : public BitWriter {
 public:
  // "storage" points to a pre-allocated memory area that must have enough size
  // hold all of the encoded symbols. It is the responsibility of the caller
  // to compute the required size or an upper bound thereof.
  HuffmanSymbolWriter(const std::vector<HuffmanEncodingData>& codes,
                      const std::vector<uint8_t>& context_map,
                      size_t* storage_ix, uint8_t* storage)
      : BitWriter(storage_ix, storage), codes_(codes),
        context_map_(context_map) {}

  void VisitSymbol(int symbol, int ctx) {
    const int histo_idx = context_map_[ctx];
    WriteBits(codes_[histo_idx].depths[symbol], codes_[histo_idx].bits[symbol],
              storage_ix_, storage_);
  }

  void FlushToBitStream() {}

 private:
  const std::vector<HuffmanEncodingData>& codes_;
  const std::vector<uint8_t>& context_map_;
};

struct ANSEncodingData {
  void BuildAndStore(const uint32_t* histogram, size_t histo_size,
                     size_t* storage_ix, uint8_t* storage) {
    std::vector<int> counts(histo_size);
    for (int i = 0; i < histo_size; ++i) {
      counts[i] = histogram[i];
    }
    ans_table.resize(histo_size);
    BuildAndStoreANSEncodingData(counts.data(), counts.size(), ans_table.data(),
                                 storage_ix, storage);
  }

  std::vector<ANSEncSymbolInfo> ans_table;
};

// Symbol visitor that collects symbols and raw bits to be encoded.
class ANSSymbolWriter {
 public:
  ANSSymbolWriter(const std::vector<ANSEncodingData>& codes,
                  const std::vector<uint8_t>& context_map,
                  size_t* storage_ix, uint8_t* storage)
      : idx_(0), symbol_idx_(0), code_words_(2 * kANSBufferSize),
        symbols_(kANSBufferSize), codes_(codes), context_map_(context_map),
        storage_ix_(storage_ix), storage_(storage) {}

  void VisitBits(size_t nbits, uint64_t bits) {
    PIK_ASSERT(nbits <= 16);
    PIK_ASSERT(idx_ < code_words_.size());
    if (nbits > 0) {
      code_words_[idx_++] = (bits << 16) + nbits;
    }
  }

  void VisitSymbol(int symbol, int ctx) {
    PIK_ASSERT(idx_ < code_words_.size());
    code_words_[idx_++] = 0xffff;  // Placeholder, to be encoded later.
    symbols_[symbol_idx_++] = (ctx << 16) + symbol;
    if (symbol_idx_ == kANSBufferSize) {
      FlushToBitStream();
    }
  }

  void FlushToBitStream() {
    const int num_codewords = idx_;
    ANSCoder ans;
    int first_symbol = num_codewords;
    // Replace placeholder code words with actual bits by feeding symbols to the
    // ANS encoder in a reverse order.
    for (int i = num_codewords - 1; i >= 0; --i) {
      const uint32_t cw = code_words_[i];
      if ((cw & 0xffff) == 0xffff) {
        const uint32_t sym = symbols_[--symbol_idx_];
        const uint32_t context = sym >> 16;
        const uint8_t histo_idx = context_map_[context];
        const uint32_t symbol = sym & 0xffff;
        const ANSEncSymbolInfo info = codes_[histo_idx].ans_table[symbol];
        uint8_t nbits = 0;
        uint32_t bits = ans.PutSymbol(info, &nbits);
        code_words_[i] = (bits << 16) + nbits;
        first_symbol = i;
      }
    }
    for (int i = 0; i < num_codewords; ++i) {
      if (i == first_symbol) {
        const uint32_t state = ans.GetState();
        WriteBits(16, (state >> 16) & 0xffff, storage_ix_, storage_);
        WriteBits(16, state & 0xffff, storage_ix_, storage_);
      }
      const uint32_t cw = code_words_[i];
      const uint32_t nbits = cw & 0xffff;
      const uint32_t bits = cw >> 16;
      WriteBits(nbits, bits, storage_ix_, storage_);
    }
    idx_ = 0;
    PIK_ASSERT(symbol_idx_ == 0);
  }

 private:
  int idx_;
  int symbol_idx_;
  // Vector of (bits, nbits) pairs to be encoded.
  std::vector<uint32_t> code_words_;
  // Vector of (context, symbol) pairs to be encoded.
  std::vector<uint32_t> symbols_;
  const std::vector<ANSEncodingData>& codes_;
  const std::vector<uint8_t>& context_map_;
  size_t* storage_ix_;
  uint8_t* storage_;
};

void ComputeCoeffOrder(const Image3W& img, int* order) {
  for (int c = 0; c < 3; ++c) {
    int num_zeros[64] = { 0 };
    for (int y = 0; y < img.ysize(); ++y) {
      auto row = img.Row(y);
      for (int x = 0; x < img.xsize(); x += 64) {
        for (int k = 1; k < 64; ++k) {
          if (row[c][x + k] == 0) ++num_zeros[k];
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
      order[c * 64 + i] = kNaturalCoeffOrder[pos_and_val[i].first];
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
    builder.BuildAndStoreEntropyCodes(
        &codes, &context_map, &storage_ix, storage);
    // Close the histogram bit stream.
    size_t jump_bits = ((storage_ix + 31) & ~31) - storage_ix;
    WriteBits(jump_bits, 0, &storage_ix, storage);
    PIK_ASSERT(storage_ix % 32 == 0);
    const size_t histo_bytes = storage_ix >> 3;
    // Entropy encode data.
    SymbolWriter symbol_writer(codes, context_map, &storage_ix, storage);
    ProcessImage3(img, processor, &symbol_writer);
    symbol_writer.FlushToBitStream();
    const size_t data_bits = storage_ix - 8 * histo_bytes;
    const size_t data_bytes = 4 * ((data_bits + 31) >> 5);
    const int out_size = histo_bytes + data_bytes;
    PIK_CHECK(out_size <= max_out_size);
    output.resize(out_size);
    if (info) {
      info->num_clustered_histograms += codes.size();
      info->histogram_size += histo_bytes;
      info->entropy_coded_bits += data_bits - builder.num_extra_bits();
      info->extra_bits += builder.num_extra_bits();
      info->total_size += out_size;
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

std::string EncodeAC(const Image3W& coeffs, PikImageSizeInfo* info) {
  ACBlockProcessor processor;
  int order[192];
  ComputeCoeffOrder(coeffs, order);
  processor.SetCoeffOrder(order);
  return EncodeImageInternal<ANSEncodingData, ANSSymbolWriter>()(
      coeffs, &processor, info);
}

PIK_INLINE uint32_t MakeToken(const uint32_t context, const uint32_t symbol,
                              const uint32_t nbits, const uint32_t bits) {
  return (context << 26) | (symbol << 18) | (nbits << 14) | bits;
}

std::string EncodeACFast(const Image3W& coeffs, PikImageSizeInfo* info) {
  // Build static context map.
  static const int kNumContexts = 408;
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
  static const int kNumStaticContexts = 3 * (1 + kNumStaticZdensContexts);
  PIK_ASSERT(kNumStaticContexts <= 64);
  std::vector<uint8_t> context_map(kNumContexts);
  for (int c = 0; c < 3; ++c) {
    for (int i = 0; i < 16; ++i) {
      context_map[c * 16 + i] = c;
    }
    for (int i = 0; i < 120; ++i) {
      context_map[48 + c * 120 + i] =
          3 + c * kNumStaticZdensContexts + kStaticZdensContextMap[i];
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
        const int16_t* coeffs = &row[c][x];
        int num_nzeros = 0;
        for (int k = 1; k < 64; ++k) {
          if (coeffs[k] != 0) ++num_nzeros;
        }
        tokens.push_back(MakeToken(c, num_nzeros, 0, 0));
        if (num_nzeros == 0) continue;
        int r = 0;
        const int histo_offset = 48 + c * 120;
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
  // Encode the histograms.
  std::vector<ANSEncodingData> codes;
  EncodeContextMap(context_map, kNumStaticContexts, &storage_ix, storage);
  for (int c = 0; c < kNumStaticContexts; ++c) {
    ANSEncodingData code;
    code.BuildAndStore(&histograms[c << 8], 256, &storage_ix, storage);
    codes.emplace_back(std::move(code));
  }
  // Close the histogram bit stream.
  size_t jump_bits = ((storage_ix + 31) & ~31) - storage_ix;
  WriteBits(jump_bits, 0, &storage_ix, storage);
  PIK_ASSERT(storage_ix % 32 == 0);
  const size_t histo_bytes = storage_ix >> 3;
  // Entropy encode data.
  WriteBits(12, 0, &storage_ix, storage);  // zig-zag coefficient order
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
  const size_t data_bits = storage_ix - 8 * histo_bytes;
  const size_t data_bytes = 4 * ((data_bits + 31) >> 5);
  const int out_size = histo_bytes + data_bytes;
  PIK_CHECK(out_size <= max_out_size);
  output.resize(out_size);
  if (info) {
    info->num_clustered_histograms += codes.size();
    info->histogram_size += histo_bytes;
    info->entropy_coded_bits += data_bits - num_extra_bits;
    info->extra_bits += num_extra_bits;
    info->total_size += out_size;
  }
  return output;
}

size_t EncodedImageSize(const Image3W& img, int stride) {
  CoeffProcessor processor(stride);
  return EncodedImageSizeInternal(img, &processor);
}

size_t EncodedACSize(const Image3W& coeffs) {
  ACBlockProcessor processor;
  return EncodedImageSizeInternal(coeffs, &processor);
}

class ANSBitCounter {
 public:
  ANSBitCounter(const std::vector<ANSEncodingData>& codes,
                const std::vector<uint8_t>& context_map)
      : codes_(codes), context_map_(context_map), nbits_(0.0f) {}

  void VisitBits(size_t nbits, uint64_t bits) {
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
  ACBlockProcessor processor;
  int order[192];
  ComputeCoeffOrder(coeffs, order);
  processor.SetCoeffOrder(order);
  HistogramBuilder builder(ACBlockProcessor::num_contexts());
  ProcessImage3(coeffs, &processor, &builder);
  std::vector<ANSEncodingData> codes;
  std::vector<uint8_t> context_map;
  builder.BuildAndStoreEntropyCodes(&codes, &context_map, nullptr, nullptr);
  ANSBitCounter counter(codes, context_map);
  Image3F out(coeffs.xsize() / processor.block_size(), coeffs.ysize());
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

class HuffmanSymbolReader {
 public:
  int ReadSymbol(const HuffmanDecodingData& code,
                 BitReader* const PIK_RESTRICT br) {
    const HuffmanCode* const PIK_RESTRICT table = &code.table_[0];
    int offset = br->PeekFixedBits<kHuffmanTableBits>();
    const int nbits = table[offset].bits - kHuffmanTableBits;
    if (nbits > 0) {
      br->Advance(kHuffmanTableBits);
      offset += table[offset].value + br->PeekBits(nbits);
    }
    br->Advance(table[offset].bits);
    return table[offset].value;
  }
};

class ANSSymbolReader {
 public:
  void DecodeHistograms(const size_t num_histograms,
                        const uint8_t* symbol_lut, size_t symbol_lut_size,
                        BitReader* in) {
    map_.resize(num_histograms << ANS_LOG_TAB_SIZE);
    info_.resize(num_histograms << 8);
    for (int c = 0; c < num_histograms; ++c) {
      std::vector<int> counts;
      ReadHistogram(ANS_LOG_TAB_SIZE, &counts, in);
      PIK_CHECK(counts.size() <= 256);
      int offset = 0;
      for (int i = 0, pos = 0; i < counts.size(); ++i) {
        int symbol = i;
        if (symbol_lut != nullptr && symbol < symbol_lut_size) {
          symbol = symbol_lut[symbol];
        }
        info_[(c << 8) + symbol].offset_ = offset;
        info_[(c << 8) + symbol].freq_ = counts[i];
        offset += counts[i];
        for (int j = 0; j < counts[i]; ++j, ++pos) {
          map_[(c << ANS_LOG_TAB_SIZE) + pos] = symbol;
        }
      }
    }
  }

  int ReadSymbol(const int histo_idx, BitReader* const PIK_RESTRICT br) {
    if (symbols_left_ == 0) {
      state_ = br->ReadBits(16);
      state_ = (state_ << 16) | br->ReadBits(16);
      br->FillBitBuffer();
      symbols_left_ = kANSBufferSize;
    }
    const uint32_t res = state_ & (ANS_TAB_SIZE - 1);
    const uint8_t symbol = map_[(histo_idx << ANS_LOG_TAB_SIZE) + res];
    const ANSSymbolInfo s = info_[(histo_idx << 8) + symbol];
    state_ = s.freq_ * (state_ >> ANS_LOG_TAB_SIZE) + res - s.offset_;
    --symbols_left_;
    if (state_ < (1u << 16)) {
      state_ = (state_ << 16) | br->PeekFixedBits<16>();
      br->Advance(16);
    }
    return symbol;
  }

  bool CheckANSFinalState() { return state_ == (ANS_SIGNATURE << 16); }

 private:
  struct ANSSymbolInfo {
    uint16_t offset_;
    uint16_t freq_;
  };
  size_t symbols_left_ = 0;
  uint32_t state_ = 0;
  std::vector<uint8_t> map_;
  std::vector<ANSSymbolInfo> info_;
};

size_t DecodeHistograms(const uint8_t* data, size_t len,
                        const size_t num_contexts,
                        const uint8_t* symbol_lut, size_t symbol_lut_size,
                        ANSSymbolReader* decoder,
                        std::vector<uint8_t>* context_map) {
  BitReader in(data, len);
  size_t num_histograms = 1;
  context_map->resize(num_contexts);
  if (num_contexts > 1) {
    DecodeContextMap(context_map, &num_histograms, &in);
  }
  decoder->DecodeHistograms(num_histograms, symbol_lut, symbol_lut_size, &in);
  return in.Position();
}

size_t DecodeImageData(const uint8_t* const PIK_RESTRICT data, const size_t len,
                       const std::vector<uint8_t>& context_map,
                       const int stride,
                       ANSSymbolReader* const PIK_RESTRICT decoder,
                       Image3W* const PIK_RESTRICT img) {
  uint8_t dummy_buffer[4] = { 0 };
  PIK_CHECK(len >= 4 || len == 0);
  BitReader br(len == 0 ? dummy_buffer : data, len);
  for (int y = 0; y < img->ysize(); ++y) {
    auto row = img->Row(y);
    for (int x = 0; x < img->xsize(); x += stride) {
      for (int c = 0; c < 3; ++c) {
        br.FillBitBuffer();
        int histo_idx = context_map[c];
        int s = decoder->ReadSymbol(histo_idx, &br);
        if (s > 0) {
          int bits = br.PeekBits(s);
          br.Advance(s);
          s = bits < (1 << (s - 1)) ? bits + ((~0U) << s ) + 1 : bits;
        }
        row[c][x] = s;
      }
    }
  }
  return br.Position();
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

size_t DecodeACData(const uint8_t* const PIK_RESTRICT data, const size_t len,
                    const std::vector<uint8_t>& context_map,
                    ANSSymbolReader* const PIK_RESTRICT decoder,
                    Image3W* const PIK_RESTRICT coeffs) {
  uint8_t dummy_buffer[4] = { 0 };
  PIK_CHECK(len >= 4 || len == 0);
  BitReader br(len == 0 ? dummy_buffer : data, len);
  int coeff_order[192];
  for (int c = 0; c < 3; ++c) {
    DecodeCoeffOrder(&coeff_order[c * 64], &br);
  }
  for (int y = 0; y < coeffs->ysize(); ++y) {
    auto row = coeffs->Row(y);
    int prev_num_nzeros[3] = { 0 };
    for (int x = 0; x < coeffs->xsize(); x += 64) {
      for (int c = 0; c < 3; ++c) {
        memset(&row[c][x + 1], 0, 63 * sizeof(row[0][0]));
        br.FillBitBuffer();
        const int context1 = c * 16 + (prev_num_nzeros[c] >> 2);
        int num_nzeros =
            kIndexLut[decoder->ReadSymbol(context_map[context1], &br)];
        prev_num_nzeros[c] = num_nzeros;
        if (num_nzeros == 0) continue;
        const int histo_offset = 48 + c * 120;
        const int context2 = ZeroDensityContext(num_nzeros - 1, 0, 4);
        int histo_idx = context_map[histo_offset + context2];
        for (int k = 1; k < 64 && num_nzeros > 0; ++k) {
          br.FillBitBuffer();
          int s = decoder->ReadSymbol(histo_idx, &br);
          k += (s >> 4);
          s &= 15;
          if (s > 0) {
            int bits = br.PeekBits(s);
            br.Advance(s);
            s = bits < (1 << (s - 1)) ? bits + ((~0U) << s ) + 1 : bits;
            const int context = ZeroDensityContext(num_nzeros - 1, k, 4);
            histo_idx = context_map[histo_offset + context];
            --num_nzeros;
          }
          row[c][x + coeff_order[c * 64 + k]] = s;
        }
        PIK_CHECK(num_nzeros == 0);
      }
    }
  }
  return br.Position();
}

size_t DecodeNonZeroValsData(
    const uint8_t* const PIK_RESTRICT data, const size_t len,
    const std::vector<uint8_t>& context_map,
    ANSSymbolReader* const PIK_RESTRICT decoder,
    std::vector<Image3W>* const PIK_RESTRICT absvals,
    std::vector<Image3W>* const PIK_RESTRICT phases) {
  uint8_t dummy_buffer[4] = { 0 };
  PIK_CHECK(len >= 4 || len == 0);
  BitReader br(len == 0 ? dummy_buffer : data, len);
  for (int i = 0; i < absvals->size(); ++i) {
    for (int y = 0; y < (*absvals)[i].ysize(); ++y) {
      auto row_absvals = (*absvals)[i].Row(y);
      auto row_phases = (*phases)[i].Row(y);
      for (int x = 0; x < (*absvals)[i].xsize(); ++x) {
        for (int c = 0; c < 3; ++c) {
          if (row_absvals[c][x] != 0) {
            br.FillBitBuffer();
            int histo_idx1 = context_map[c * absvals->size() + i];
            row_absvals[c][x] = decoder->ReadSymbol(histo_idx1, &br) + 1;
            int histo_idx2 = context_map[(c + 3) * absvals->size() + i];
            row_phases[c][x] = SignedIntFromSymbol(
                decoder->ReadSymbol(histo_idx2, &br));
          }
        }
      }
    }
  }
  return br.Position();
}

size_t DecodeImage(const uint8_t* data, size_t len, int stride,
                   Image3W* coeffs) {
  size_t pos = 0;
  std::vector<uint8_t> context_map;
  ANSSymbolReader decoder;
  pos += DecodeHistograms(data, len, CoeffProcessor::num_contexts(), nullptr, 0,
                          &decoder, &context_map);
  pos += DecodeImageData(data + pos, len - pos, context_map, stride,
                         &decoder, coeffs);
  PIK_CHECK(decoder.CheckANSFinalState());
  return pos;
}

size_t DecodeAC(const uint8_t* data, size_t len, Image3W* coeffs) {
  size_t pos = 0;
  std::vector<uint8_t> context_map;
  ANSSymbolReader decoder;
  pos += DecodeHistograms(data, len, ACBlockProcessor::num_contexts(),
                          kSymbolLut, sizeof(kSymbolLut),
                          &decoder, &context_map);
  pos += DecodeACData(data + pos, len - pos, context_map, &decoder, coeffs);
  return pos;
}

size_t DecodeNonZeroVals(const  uint8_t* data, size_t len,
                         const size_t num_nonzeros,
                         std::vector<Image3W>* absvals,
                         std::vector<Image3W>* phases) {
  if (num_nonzeros == 0) return 0;
  size_t pos = 0;
  std::vector<uint8_t> context_map;
  ANSSymbolReader decoder;
  pos += DecodeHistograms(data, len, 6 * absvals->size(), nullptr, 0,
                          &decoder, &context_map);
  pos += DecodeNonZeroValsData(data + pos, len - pos, context_map,
                               &decoder, absvals, phases);
  PIK_CHECK(decoder.CheckANSFinalState());
  return pos;
}

class DeltaCodingProcessor {
 public:
  DeltaCodingProcessor(int minval, int maxval, int xsize)
      : minval_(minval), maxval_(maxval), xsize_(xsize) {
    Reset();
  }

  void Reset() {
    row_ = std::vector<int>(xsize_);
  }

  int block_size() const { return 1; }
  static int num_contexts() { return 1; }

  int PredictVal(int x, int y, int c) {
    if (x == 0) {
      return y == 0 ? (minval_ + maxval_ + 1) / 2 : row_[x];
    }
    if (y == 0) {
      return row_[x - 1];
    }
    return (row_[x] + row_[x - 1] + 1) / 2;
  }

  void SetVal(int x, int y, int c, int val) {
    row_[x] = val;
  }

  template <class Visitor>
  void ProcessBlock(const int* val, int x, int y, int c, Visitor* visitor) {
    PIK_ASSERT(*val >= minval_ && *val <= maxval_);
    int diff = *val - PredictVal(x, y, c);
    visitor->VisitSymbol(SymbolFromSignedInt(diff), c);
    SetVal(x, y, c, *val);
  }

 private:
  const int minval_;
  const int maxval_;
  const int xsize_;
  std::vector<int> row_;
};

std::string EncodePlane(const Image<int>& img, int minval, int maxval) {
  DeltaCodingProcessor processor(minval, maxval, img.xsize());
  HistogramBuilder builder(processor.num_contexts());
  ProcessImage(img, &processor, &builder);
  const size_t max_out_size = 2 * img.xsize() * img.ysize() + 1024;
  std::string output(max_out_size, 0);
  size_t storage_ix = 0;
  uint8_t* storage = reinterpret_cast<uint8_t*>(&output[0]);
  storage[0] = 0;
  std::vector<HuffmanEncodingData> codes;
  std::vector<uint8_t> context_map;
  builder.BuildAndStoreEntropyCodes(&codes, &context_map, &storage_ix, storage);
  HuffmanSymbolWriter symbol_writer(codes, context_map, &storage_ix, storage);
  ProcessImage(img, &processor, &symbol_writer);
  symbol_writer.FlushToBitStream();
  const int out_size = 4 * ((storage_ix + 31) >> 5);
  PIK_CHECK(out_size <= max_out_size);
  output.resize(out_size);
  return output;
}

size_t EncodedPlaneSize(const Image<int>& img, int minval, int maxval) {
  DeltaCodingProcessor processor(minval, maxval, img.xsize());
  HistogramBuilder builder(processor.num_contexts());
  ProcessImage(img, &processor, &builder);
  return builder.EncodedSize(-1, 2);
}

size_t DecodePlane(const uint8_t* data, size_t len, int minval, int maxval,
                   Image<int>* img) {
  BitReader in(data, len);
  HuffmanDecodingData huff;
  huff.ReadFromBitStream(&in);
  HuffmanDecoder decoder;
  DeltaCodingProcessor processor(minval, maxval, img->xsize());
  for (int y = 0; y < img->ysize(); ++y) {
    auto row = img->Row(y);
    for (int x = 0; x < img->xsize(); ++x) {
      int symbol = decoder.ReadSymbol(huff, &in);
      int diff = SignedIntFromSymbol(symbol);
      row[x] = diff + processor.PredictVal(x, y, 0);
      processor.SetVal(x, y, 0, row[x]);
    }
  }
  return in.Position();
}

}  // namespace pik
