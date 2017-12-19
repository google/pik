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

#include "ans_encode.h"

#include <stdint.h>
#include <vector>

#include "fast_log.h"
#include "histogram.h"
#include "histogram_encode.h"

namespace pik {

namespace {

void ANSBuildInfoTable(const int* counts, int alphabet_size,
                       ANSEncSymbolInfo* info) {
  int total = 0;
  for (int s = 0; s < alphabet_size; ++s) {
    const uint32_t freq = counts[s];
    info[s].freq_ = counts[s];
    info[s].start_ = total;
    total += freq;
#ifdef USE_MULT_BY_RECIPROCAL
    if (freq != 0) {
      info[s].ifreq_ =
          ((1ull << RECIPROCAL_PRECISION) + info[s].freq_ - 1) / info[s].freq_;
    } else {
      info[s].ifreq_ = 1;  // shouldn't matter (symbol shoudln't occur), but...
    }
#endif
  }
}

int EstimateDataBits(const int* histogram, const int* counts, size_t len) {
  float sum = 0.0f;
  int total_histogram = 0;
  int total_counts = 0;
  for (int i = 0; i < len; ++i) {
    total_histogram += histogram[i];
    total_counts += counts[i];
    if (histogram[i] > 0) {
      PIK_ASSERT(counts[i] > 0);
      sum -= histogram[i] * FastLog2(counts[i]);
    }
  }
  if (total_histogram > 0) {
    PIK_ASSERT(total_counts == ANS_TAB_SIZE);
    const int log2_total_counts = ANS_LOG_TAB_SIZE;
    sum += total_histogram * log2_total_counts;
  }
  return static_cast<int>(sum + 1.0f);
}

int EstimateDataBitsFlat(const int* histogram, size_t len) {
  const float flat_bits = FastLog2(len);
  int total_histogram = 0;
  for (int i = 0; i < len; ++i) {
    total_histogram += histogram[i];
  }
  return static_cast<int>(total_histogram * flat_bits + 1.0);
}

}  // namespace

void BuildAndStoreANSEncodingData(const int* histogram,
                                  int alphabet_size,
                                  ANSEncSymbolInfo* info,
                                  size_t* storage_ix, uint8_t* storage) {
  PIK_ASSERT(alphabet_size <= ANS_TAB_SIZE);
  int num_symbols;
  int symbols[kMaxNumSymbolsForSmallCode] = { 0 };
  std::vector<int> counts(histogram, histogram + alphabet_size);
  int omit_pos;
  PIK_CHECK(NormalizeCounts(counts.data(), &omit_pos, alphabet_size,
                            ANS_LOG_TAB_SIZE, &num_symbols, symbols));
  ANSBuildInfoTable(counts.data(), alphabet_size, info);
  if (storage_ix != nullptr && storage != nullptr) {
    const int storage_ix0 = *storage_ix;
    EncodeCounts(counts.data(), alphabet_size,
                 omit_pos, num_symbols, symbols, storage_ix, storage);
    if (alphabet_size <= kMaxNumSymbolsForSmallCode) {
      return;
    }
    // Let's see if we can do better in terms of histogram size + data size.
    const int histo_bits = *storage_ix - storage_ix0;
    const int data_bits = EstimateDataBits(histogram, counts.data(),
                                           alphabet_size);
    const int histo_bits_flat = ANS_LOG_TAB_SIZE + 2;
    const int data_bits_flat = EstimateDataBitsFlat(histogram, alphabet_size);
    if (histo_bits_flat + data_bits_flat < histo_bits + data_bits) {
      counts = CreateFlatHistogram(alphabet_size, ANS_TAB_SIZE);
      ANSBuildInfoTable(counts.data(), alphabet_size, info);
      RewindStorage(storage_ix0, storage_ix, storage);
      EncodeFlatHistogram(alphabet_size, storage_ix, storage);
    }
  }
}

}  // namespace pik
