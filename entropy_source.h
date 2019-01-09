// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef ENTROPY_SOURCE_H_
#define ENTROPY_SOURCE_H_

#include <stddef.h>
#include <stdint.h>
#include <cstdint>
#include <memory>
#include <vector>

#include "ans_encode.h"
#include "cluster.h"
#include "context_map_encode.h"
#include "write_bits.h"

namespace pik {

// Manages building, clustering and encoding of the histograms of an entropy
// source.
class EntropySource {
 public:
  explicit EntropySource(int num_contexts)
      : num_bands_(0), num_contexts_(num_contexts) {}

  void Resize(int num_bands) {
    num_bands_ = num_bands;
    histograms_.resize(num_bands * num_contexts_);
  }

  void AddCode(int code, int histo_ix) { histograms_[histo_ix].Add(code); }

  void ClusterHistograms(const std::vector<int>& offsets) {
    std::vector<uint32_t> context_map32;
    pik::ClusterHistograms(histograms_, num_contexts_, num_bands_, offsets,
                           kMaxNumberOfHistograms, &clustered_, &context_map32);
    context_map_.resize(context_map32.size());
    for (size_t i = 0; i < context_map_.size(); ++i) {
      context_map_[i] = static_cast<uint8_t>(context_map32[i]);
    }
  }

  void EncodeContextMap(size_t* storage_ix, uint8_t* storage) const {
    pik::EncodeContextMap(context_map_, clustered_.size(), storage_ix, storage);
  }

  void BuildAndStoreEntropyCodes(size_t* storage_ix, uint8_t* storage) {
    ans_tables_.resize(clustered_.size() * kAlphabetSize);
    for (int i = 0; i < clustered_.size(); ++i) {
      BuildAndStoreANSEncodingData(&clustered_[i].data_[0], kAlphabetSize,
                                   &ans_tables_[i * kAlphabetSize], storage_ix,
                                   storage);
    }
  }

  const ANSEncSymbolInfo* GetANSTable(int context) const {
    const int entropy_ix = context_map_[context];
    return &ans_tables_[entropy_ix * kAlphabetSize];
  }

 private:
  template <int kSize>
  struct Histogram {
    Histogram() { Clear(); }

    void Clear() {
      memset(data_, 0, sizeof(data_));
      total_count_ = 0;
    }

    void AddHistogram(const Histogram& other) {
      for (int i = 0; i < kSize; ++i) {
        data_[i] += other.data_[i];
      }
      total_count_ += other.total_count_;
    }

    void Add(int val) {
      PIK_ASSERT(val < kSize);
      ++data_[val];
      ++total_count_;
    }

    float PopulationCost() const {
      return ANSPopulationCost(&data_[0], kSize, total_count_);
    }

    int data_[kSize];
    int total_count_;
  };

  static const int kAlphabetSize = 18;
  static const int kMaxNumberOfHistograms = 256;
  int num_bands_;
  const int num_contexts_;
  std::vector<Histogram<kAlphabetSize> > histograms_;
  std::vector<Histogram<kAlphabetSize> > clustered_;
  std::vector<uint8_t> context_map_;
  std::vector<ANSEncSymbolInfo> ans_tables_;
};

}  // namespace pik

#endif  // ENTROPY_SOURCE_H_
