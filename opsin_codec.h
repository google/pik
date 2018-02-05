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

#ifndef OPSIN_CODEC_H_
#define OPSIN_CODEC_H_

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
#include "compiler_specific.h"
#include "context.h"
#include "context_map_encode.h"
#include "fast_log.h"
#include "histogram_encode.h"
#include "huffman_decode.h"
#include "image.h"
#include "lehmer_code.h"
#include "pik_info.h"
#include "status.h"

namespace pik {

const int kDCAlphabetSize = 16;
const int kACAlphabetSize = 256;

const int kNaturalCoeffOrder[80] = {
  0,   1,  8, 16,  9,  2,  3, 10,
  17, 24, 32, 25, 18, 11,  4,  5,
  12, 19, 26, 33, 40, 48, 41, 34,
  27, 20, 13,  6,  7, 14, 21, 28,
  35, 42, 49, 56, 57, 50, 43, 36,
  29, 22, 15, 23, 30, 37, 44, 51,
  58, 59, 52, 45, 38, 31, 39, 46,
  53, 60, 61, 54, 47, 55, 62, 63,
  // extra entries for safety in decoder
  63, 63, 63, 63, 63, 63, 63, 63,
  63, 63, 63, 63, 63, 63, 63, 63
};

PIK_INLINE std::string PadTo4Bytes(const std::string& s) {
  size_t rem = s.size() % 4;
  return rem == 0 ? s : s + std::string(4 - rem, 0);
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

template <class Visitor>
void VisitCoefficient(int coeff, int histo_idx, Visitor* visitor) {
  int nbits, bits;
  EncodeCoeff(coeff, &nbits, &bits);
  visitor->VisitSymbol(nbits, histo_idx);
  visitor->VisitBits(nbits, bits, histo_idx);
}

class CoeffProcessor {
 public:
  CoeffProcessor(int stride) : stride_(stride) {}

  void Reset() {}
  int block_size() const { return stride_; }
  static int num_contexts() { return 3; }

  template <class Visitor>
  void ProcessBlock(const int16_t* coeffs, int x, int y, int c,
                    Visitor* visitor) {
    VisitCoefficient(*coeffs, c, visitor);
  }

 private:
  const int stride_;
};

// Reorder the symbols by decreasing population-count (keeping the first
// end-of-block symbol in place).
static const uint8_t kIndexLut[256] = {
  0,   1,   2,   3,   5,  10,  17,  32,
  68,  83,  84,  85,  86,  87,  88,  89,
  90,   4,   7,  12,  22,  31,  43,  60,
  91,  92,  93,  94,  95,  96,  97,  98,
  99,   6,  14,  26,  36,  48,  66, 100,
  101, 102, 103, 104, 105, 106, 107, 108,
  109,   8,  19,  34,  44,  57,  78, 110,
  111, 112, 113, 114, 115, 116, 117, 118,
  119,   9,  27,  39,  52,  61,  79, 120,
  121, 122, 123, 124, 125, 126, 127, 128,
  129,  11,  28,  41,  53,  64,  80, 130,
  131, 132, 133, 134, 135, 136, 137, 138,
  139,  13,  33,  46,  63,  72, 140, 141,
  142, 143, 144, 145, 146, 147, 148, 149,
  150,  15,  35,  47,  65,  69, 151, 152,
  153, 154, 155, 156, 157, 158, 159, 160,
  161,  16,  37,  51,  62,  74, 162, 163,
  164, 165, 166, 167, 168, 169, 170, 171,
  172,  18,  38,  50,  59,  75, 173, 174,
  175, 176, 177, 178, 179, 180, 181, 182,
  183,  20,  40,  54,  76,  82, 184, 185,
  186, 187, 188, 189, 190, 191, 192, 193,
  194,  23,  42,  55,  77, 195, 196, 197,
  198, 199, 200, 201, 202, 203, 204, 205,
  206,  24,  45,  56,  70, 207, 208, 209,
  210, 211, 212, 213, 214, 215, 216, 217,
  218,  25,  49,  58,  71, 219, 220, 221,
  222, 223, 224, 225, 226, 227, 228, 229,
  230,  29,  67,  81, 231, 232, 233, 234,
  235, 236, 237, 238, 239, 240, 241, 242,
  21,  30,  73, 243, 244, 245, 246, 247,
  248, 249, 250, 251, 252, 253, 254, 255,
};

template <class Visitor>
void VisitACSymbol(int runlen, int nbits, int histo_idx, Visitor* visitor) {
  int symbol = (runlen << 4) + nbits;
  symbol = kIndexLut[symbol];
  visitor->VisitSymbol(symbol, histo_idx);
}

inline int NumNonZerosContext(int x, int y, uint8_t* prev) {
  // *prev is updated after context is computed, so prev[0] is for top-block,
  // prev[-3] is for previous block.
  return x > 0 ? (prev[0] + prev[-3]) >> 2 : prev[0] >> 1;
}

static const int kOrderContexts = 6;

class ACBlockProcessor {
 public:
  ACBlockProcessor(const Image3B* block_ctx, int xsize)
      : block_ctx_(*block_ctx), xsize_(xsize) {
    Reset();
    for (int c = 0; c < kOrderContexts; ++c) {
      memcpy(&order_[c * 64], kNaturalCoeffOrder, 64 * sizeof(order_[0]));
    }
  }
  void Reset() {
    prev_num_nzeros_ = std::vector<uint8_t>(3 * (xsize_ >> 6));
  }
  int block_size() const { return 64; }
  static int num_contexts() { return kOrderContexts * (32 + 120); }

  void SetCoeffOrder(int order[kOrderContexts * 64]) {
    memcpy(order_, order, sizeof(order_));
  }

  template <class Visitor>
  void ProcessBlock(const int16_t* coeffs, int x, int y, int c,
                    Visitor* visitor) {
    int num_nzeros = 0;
    for (int k = 1; k < block_size(); ++k) {
      if (coeffs[k] != 0) ++num_nzeros;
    }
    const int ix = 3 * (x >> 6) + c;
    const int block_ctx = block_ctx_.Row(y)[c][x >> 6];
    int context = block_ctx * 32 +
        NumNonZerosContext(x, y, &prev_num_nzeros_[ix]);
    PIK_ASSERT(context < kOrderContexts * 32);
    visitor->VisitSymbol(num_nzeros, context);
    prev_num_nzeros_[ix] = num_nzeros;
    if (num_nzeros == 0) return;
    // Run length of zero coefficients preceding the current non-zero symbol.
    int r = 0;
    const int histo_offset = kOrderContexts * 32 + block_ctx * 120;
    int histo_idx = histo_offset + ZeroDensityContext(num_nzeros - 1, 0, 4);
    const int order_offset = block_ctx * 64;
    for (int k = 1; k < block_size(); ++k) {
      int16_t coeff = coeffs[order_[order_offset + k]];
      if (coeff == 0) {
        r++;
        continue;
      }
      PIK_ASSERT(histo_idx < num_contexts());
      while (r > 15) {
        VisitACSymbol(15, 0, histo_idx, visitor);
        r -= 16;
      }
      int nbits, bits;
      EncodeCoeff(coeff, &nbits, &bits);
      VisitACSymbol(r, nbits, histo_idx, visitor);
      visitor->VisitBits(nbits, bits, c);
      r = 0;
      histo_idx = histo_offset + ZeroDensityContext(num_nzeros - 1, k, 4);
      --num_nzeros;
    }
    PIK_CHECK(num_nzeros == 0);
  }
 private:
  const Image3B& block_ctx_;
  const int xsize_;
  int order_[kOrderContexts * 64];
  std::vector<uint8_t> prev_num_nzeros_;
};

template <typename T, class Processor, class Visitor>
void ProcessImage3(const Image3<T>& img,
                   Processor* processor,
                   Visitor* visitor) {
  processor->Reset();
  for (int y = 0; y < img.ysize(); ++y) {
    auto row = img.Row(y);
    for (int x = 0; x < img.xsize(); x += processor->block_size()) {
      for (int c = 0; c < 3; ++c) {
        processor->ProcessBlock(&row[c][x], x, y, c, visitor);
      }
    }
  }
}

template <typename T, class Processor, class Visitor>
void ProcessImage(const Image<T>& img,
                  Processor* processor,
                  Visitor* visitor) {
  processor->Reset();
  for (int y = 0; y < img.ysize(); ++y) {
    auto row = img.Row(y);
    for (int x = 0; x < img.xsize(); x += processor->block_size()) {
      processor->ProcessBlock(&row[x], x, y, 0, visitor);
    }
  }
}

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
      : weight_(1), histograms_(num_contexts) {
    num_extra_bits_[0] = num_extra_bits_[1] = num_extra_bits_[2] = 0;
  }

  void set_weight(int weight) { weight_ = weight; }

  void VisitSymbol(int symbol, int histo_idx) {
    histograms_[histo_idx].Add(symbol, weight_);
  }

  void VisitBits(size_t nbits, uint64_t bits, int c) {
    num_extra_bits_[c] += weight_ * nbits;
  }

  template <class EntropyEncodingData>
  void BuildAndStoreEntropyCodes(std::vector<EntropyEncodingData>* codes,
                                 std::vector<uint8_t>* context_map,
                                 size_t* storage_ix, uint8_t* storage,
                                 std::vector<double>* entropy_per_context,
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
    if (entropy_per_context) {
      entropy_per_context->resize(histograms_.size());
      for (int i = 0; i < histograms_.size(); ++i) {
        (*entropy_per_context)[i] = histograms_[i].CrossEntropy(
            clustered_histograms[(*context_map)[i]]);
      }
    }
    for (int c = 0; c < clustered_histograms.size(); ++c) {
      EntropyEncodingData code;
      code.BuildAndStore(&clustered_histograms[c].data_[0],
                         clustered_histograms[c].data_.size(),
                         storage_ix, storage);
      codes->emplace_back(std::move(code));
    }
  }

  size_t EncodedSize(int lg2_histo_align, int lg2_data_align) const;
  size_t num_extra_bits() const {
    return num_extra_bits_[0] + num_extra_bits_[1] + num_extra_bits_[2];
  }

  size_t num_extra_bits(int c) const { return num_extra_bits_[c]; }

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
    void Add(int symbol, int weight) {
      if (symbol >= data_.size()) {
        data_.resize(symbol + 1);
      }
      data_[symbol] += weight;
      total_count_ += weight;
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
      return pik::PopulationCost(counts.data(), counts.size(), total_count_);
    }
    double ShannonEntropy() const {
      return pik::ShannonEntropy(data_.data(), data_.size());
    }
    double CrossEntropy(const Histogram& coding_histo) const {
      return pik::CrossEntropy(data_.data(), data_.size(),
                               coding_histo.data_.data(),
                               coding_histo.data_.size());
    }

    std::vector<uint32_t> data_;
    uint32_t total_count_;
  };
  int weight_;
  size_t num_extra_bits_[3];
  std::vector<Histogram> histograms_;
};

void PredictDCTile(const Image3W& coeffs, Image3W* out);
void UnpredictDCTile(Image3W* coeffs);

std::string EncodeImage(const Image3W& img, int stride,
                        PikImageSizeInfo* info);

std::string EncodeAC(const Image3W& coeffs, const Image3B& block_ctx,
                     PikInfo* pik_info);
std::string EncodeACFast(const Image3W& coeffs, const Image3B& block_ctx,
                         PikInfo* pik_info);

size_t EncodedImageSize(const Image3W& img, int stride);

size_t EncodedACSize(const Image3W& coeffs);

Image3F LocalACInformationDensity(const Image3W& coeffs);

std::string EncodeNonZeroLocations(const std::vector<Image3W>& vals);

std::string EncodeNonZeroVals(const std::vector<Image3W>& absvals,
                         const std::vector<Image3W>& phases);

bool DecodeImage(BitReader* br, int stride, Image3W* coeffs);

bool DecodeAC(const Image3B& block_ctx, BitReader* br,Image3W* coeffs);

struct EncodedIntPlane {
  std::string preamble;
  std::vector<std::vector<std::string>> tiles;
};

EncodedIntPlane EncodePlane(const Image<int>& img, int minval, int maxval,
                            int tile_size, PikImageSizeInfo* info);

class IntPlaneDecoder {
 public:
  IntPlaneDecoder(int minval, int maxval, int tile_size)
      : minval_(minval), maxval_(maxval), tile_size_(tile_size) {}
  bool LoadPreamble(BitReader* br);
  bool DecodeTile(BitReader* br, Image<int>* img);

 private:
  bool ready_ = false;
  int minval_;
  int maxval_;
  int tile_size_;
  std::vector<uint8_t> context_map_;
  ANSCode ans_code_;
};

}  // namespace pik

#endif  // OPSIN_CODEC_H_
