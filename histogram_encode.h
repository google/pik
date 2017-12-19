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

// Library to store a histogram to the bit-stream.

#ifndef HISTOGRAM_ENCODE_H_
#define HISTOGRAM_ENCODE_H_

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "ans_params.h"
#include "status.h"

namespace pik {

// Returns an estimate of the number of bits required to encode the given
// histogram (header bits plus data bits).
float PopulationCost(const int* data, int alphabet_size, int total_count);

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
    return pik::PopulationCost(&data_[0], kSize, total_count_);
  }

  int data_[kSize];
  int total_count_;
};

static const int kMaxNumSymbolsForSmallCode = 4;

// Normalizes the population counts in counts[0 .. length) so that the sum of
// all counts will be 1 << precision_bits.
// Sets *num_symbols to the number of symbols in the range [0 .. length) with
// non-zero population counts.
// Fills in symbols[0 .. kMaxNumSymbolsForSmallCode) with the first few symbols
// with non-zero population counts.
// Each count will all be rounded to multiples of
// 1 << GetPopulationCountPrecision(count), except possibly for one. The index
// of that count will be stored in *omit_pos.
bool NormalizeCounts(int* counts,
                     int* omit_pos,
                     const int length,
                     const int precision_bits,
                     int* num_symbols,
                     int* symbols);

// Stores a histogram in counts[0 .. alphabet_size) to the bit-stream where
// the sum of all population counts is ANS_TAB_SIZE and the number of symbols
// with non-zero counts is num_symbols.
// symbols[0 .. kMaxNumSymbolsForSmallCode) contains the first few symbols
// with non-zero population counts.
// Each count must be rounded to a multiple of
// 1 << GetPopulationCountPrecision(count), except possibly counts[omit_pos].
void EncodeCounts(const int* counts,
                  const int alphabet_size,
                  const int omit_pos,
                  const int num_symbols,
                  const int* symbols,
                  size_t* storage_ix,
                  uint8_t* storage);

// Stores the flat histogram created by CreateFlatHistogram() to the bit-stream.
void EncodeFlatHistogram(const int alphabet_size,
                         size_t* storage_ix,
                         uint8_t* storage);

}  // namespace pik

#endif  // HISTOGRAM_ENCODE_H_
