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

// Library to encode the ANS population counts to the bit-stream and encode
// symbols based on the respective distributions.

#ifndef ANS_ENCODE_H_
#define ANS_ENCODE_H_

#include <stddef.h>
#include <stdint.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "ans_params.h"

namespace pik {

#define USE_MULT_BY_RECIPROCAL

// precision must be equal to:  #bits(state_) + #bits(freq)
#define RECIPROCAL_PRECISION 42

// Data structure representing one element of the encoding table built
// from a distribution.
struct ANSEncSymbolInfo {
  uint16_t freq_;
  uint16_t start_;
#ifdef USE_MULT_BY_RECIPROCAL
  uint64_t ifreq_;
#endif
};

class ANSCoder {
 public:
  ANSCoder() : state_(ANS_SIGNATURE << 16) {}

  uint32_t PutSymbol(const ANSEncSymbolInfo t, uint8_t* nbits) {
    uint32_t bits = 0;
    *nbits = 0;
    if ((state_ >> (32 - ANS_LOG_TAB_SIZE)) >= t.freq_) {
      bits = state_ & 0xffff;
      state_ >>= 16;
      *nbits = 16;
    }
#ifdef USE_MULT_BY_RECIPROCAL
    // We use mult-by-reciprocal trick, but that requires 64b calc.
    const uint32_t v = (state_ * t.ifreq_) >> RECIPROCAL_PRECISION;
    const uint32_t offset = state_ - v * t.freq_ + t.start_;
    state_ = (v << ANS_LOG_TAB_SIZE) + offset;
#else
    state_ = ((state_ / t.freq_) << ANS_LOG_TAB_SIZE)
           + (state_ % t.freq_) + t.start_;
#endif
    return bits;
  }

  uint32_t GetState() const { return state_; }

 private:
  uint32_t state_;
};

void BuildAndStoreANSEncodingData(const int* histogram,
                                  int alphabet_size,
                                  ANSEncSymbolInfo* info,
                                  size_t* storage_ix, uint8_t* storage);

}  // namespace pik

#endif  // ANS_ENCODE_H_
