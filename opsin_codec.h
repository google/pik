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
#include <cstdlib>
#include <string>
#include <vector>

#include "bit_reader.h"
#include "ans_encode.h"
#include "context.h"
#include "cluster.h"
#include "context_map_encode.h"
#include "histogram_encode.h"
#include "image.h"
#include "fast_log.h"
#include "lehmer_code.h"
#include "pik_info.h"

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
  visitor->VisitBits(nbits, bits);
}

class CoeffProcessor {
 public:
  CoeffProcessor(int stride) : stride_(stride) {}

  void Reset() {}
  int block_size() const { return stride_; }
  static int num_contexts() { return 3; }

  template <class Visitor>
  void ProcessHeader(Visitor* visitor) {}

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

class ACBlockProcessor {
 public:
  ACBlockProcessor() {
    Reset();
    for (int c = 0; c < 3; ++c) {
      memcpy(&order_[c * 64], kNaturalCoeffOrder, 64 * sizeof(order_[0]));
    }
  }
  void Reset() {
    prev_num_nzeros_[0] = prev_num_nzeros_[1] = prev_num_nzeros_[2] = 0;
  }
  int block_size() const { return 64; }
  static int num_contexts() { return 408; }

  void SetCoeffOrder(int order[192]) {
    memcpy(order_, order, sizeof(order_));
  }

  template <class Visitor>
  void ProcessHeader(Visitor* visitor) {
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
    for (int c = 0; c < 3; ++c) {
      int order_zigzag[64];
      for (int i = 0; i < 64; ++i) {
        order_zigzag[i] = kJPEGZigZagOrder[order_[c * 64 + i]];
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
          visitor->VisitBits(1, 0);
          continue;
        } else {
          visitor->VisitBits(1, 1);
        }
        for (int j = start; j < end; ++j) {
          int v;
          PIK_ASSERT(lehmer[j] <= 64);
          for (v = lehmer[j]; v >= 7; v -= 7) {
            visitor->VisitBits(3, 7);
          }
          visitor->VisitBits(3, v);
        }
      }
    }
  }

  template <class Visitor>
  void ProcessBlock(const int16_t* coeffs, int x, int y, int c,
                    Visitor* visitor) {
    int num_nzeros = 0;
    for (int k = 1; k < block_size(); ++k) {
      if (coeffs[k] != 0) ++num_nzeros;
    }
    if (x == 0) {
      prev_num_nzeros_[c] = 0;
    }
    int context = c * 16 + (prev_num_nzeros_[c] >> 2);
    visitor->VisitSymbol(num_nzeros, context);
    prev_num_nzeros_[c] = num_nzeros;
    if (num_nzeros == 0) return;
    // Run length of zero coefficients preceding the current non-zero symbol.
    int r = 0;
    const int histo_offset = 48 + c * 120;
    int histo_idx = histo_offset + ZeroDensityContext(num_nzeros - 1, 0, 4);
    for (int k = 1; k < block_size(); ++k) {
      int16_t coeff = coeffs[order_[c * 64 + k]];
      if (coeff == 0) {
        r++;
        continue;
      }
      while (r > 15) {
        VisitACSymbol(15, 0, histo_idx, visitor);
        r -= 16;
      }
      int nbits, bits;
      EncodeCoeff(coeff, &nbits, &bits);
      VisitACSymbol(r, nbits, histo_idx, visitor);
      visitor->VisitBits(nbits, bits);
      r = 0;
      histo_idx = histo_offset + ZeroDensityContext(num_nzeros - 1, k, 4);
      --num_nzeros;
    }
    PIK_CHECK(num_nzeros == 0);
  }
 private:
  int order_[192];
  int prev_num_nzeros_[3];
};

template <typename T, class Processor, class Visitor>
void ProcessImage3(const Image3<T>& img,
                   Processor* processor,
                   Visitor* visitor) {
  processor->Reset();
  processor->ProcessHeader(visitor);
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

class HistogramBuilder {
 public:
  explicit HistogramBuilder(const size_t num_contexts)
      : weight_(1), num_extra_bits_(0), histograms_(num_contexts) {
  }

  void set_weight(int weight) { weight_ = weight; }

  void VisitSymbol(int symbol, int histo_idx) {
    histograms_[histo_idx].Add(symbol, weight_);
  }

  void VisitBits(size_t nbits, uint64_t bits) {
    num_extra_bits_ += weight_ * nbits;
  }

  template <class EntropyEncodingData>
  void BuildAndStoreEntropyCodes(std::vector<EntropyEncodingData>* codes,
                                 std::vector<uint8_t>* context_map,
                                 size_t* storage_ix, uint8_t* storage) {
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
    for (int c = 0; c < clustered_histograms.size(); ++c) {
      EntropyEncodingData code;
      code.BuildAndStore(&clustered_histograms[c].data_[0],
                         clustered_histograms[c].data_.size(),
                         storage_ix, storage);
      codes->emplace_back(std::move(code));
    }
  }

  size_t EncodedSize(int lg2_histo_align, int lg2_data_align) const;
  size_t num_extra_bits() const { return num_extra_bits_; }

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
    std::vector<uint32_t> data_;
    uint32_t total_count_;
  };
  int weight_;
  size_t num_extra_bits_;
  std::vector<Histogram> histograms_;
};

Image3W PredictDC(const Image3W& coeffs);
void UnpredictDC(Image3W* coeffs);

std::string EncodeImage(const Image3W& img, int stride,
                        PikImageSizeInfo* info);

std::string EncodeAC(const Image3W& coeffs, PikImageSizeInfo* info);
std::string EncodeACFast(const Image3W& coeffs, PikImageSizeInfo* info);

size_t EncodedImageSize(const Image3W& img, int stride);

size_t EncodedACSize(const Image3W& coeffs);

Image3F LocalACInformationDensity(const Image3W& coeffs);

std::string EncodeNonZeroLocations(const std::vector<Image3W>& vals);

std::string EncodeNonZeroVals(const std::vector<Image3W>& absvals,
                         const std::vector<Image3W>& phases);

bool DecodeNonZeroLocations(BitReader* br,
                            size_t* num_nzeros,
                            std::vector<Image3W>* vals);

bool DecodeNonZeroVals(BitReader* br,
                       const size_t num_nonzeros,
                       std::vector<Image3W>* absvals,
                       std::vector<Image3W>* phases);

bool DecodeImage(BitReader* br, int stride, Image3W* coeffs);

bool DecodeAC(BitReader* br,Image3W* coeffs);

std::string EncodePlane(const Image<int>& img, int minval, int maxval);

size_t EncodedPlaneSize(const Image<int>& img, int minval, int maxval);

bool DecodePlane(BitReader* br, int minval, int maxval, Image<int>* img);

}  // namespace pik

#endif  // OPSIN_CODEC_H_
