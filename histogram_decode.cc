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

#include "histogram_decode.h"

#include "ans_params.h"
#include "fast_log.h"
#include "histogram.h"
#include "huffman_decode.h"
#include "status.h"

namespace pik {

namespace {

// Decodes a number in the range [0..65535], by reading 1 - 20 bits.
inline int DecodeVarLenUint16(BitReader* input) {
  if (input->ReadBits(1)) {
    int nbits = static_cast<int>(input->ReadBits(4));
    if (nbits == 0) {
      return 1;
    } else {
      return static_cast<int>(input->ReadBits(nbits)) + (1 << nbits);
    }
  }
  return 0;
}

}  //namespace

bool ReadHistogram(int precision_bits, std::vector<int>* counts,
                   BitReader* input) {
  int simple_code = input->ReadBits(1);
  if (simple_code == 1) {
    int i;
    int symbols[2] = { 0 };
    int max_symbol = 0;
    const int num_symbols = input->ReadBits(1) + 1;
    for (i = 0; i < num_symbols; ++i) {
      symbols[i] = DecodeVarLenUint16(input);
      if (symbols[i] > max_symbol) max_symbol = symbols[i];
    }
    counts->resize(max_symbol + 1);
    if (num_symbols == 1) {
      (*counts)[symbols[0]] = 1 << precision_bits;
    } else {
      if (symbols[0] == symbols[1]) {  // corrupt data
        return false;
      }
      (*counts)[symbols[0]] = input->ReadBits(precision_bits);
      (*counts)[symbols[1]] = (1 << precision_bits) - (*counts)[symbols[0]];
    }
  } else {
    int length = DecodeVarLenUint16(input) + 3;
    counts->resize(length);
    int total_count = 0;
    static const HuffmanCode huff[64] = {
      {2, 6}, {3, 7}, {3, 4}, {4, 1}, {2, 6}, {3, 8}, {3, 5}, {4, 3},
      {2, 6}, {3, 7}, {3, 4}, {4, 2}, {2, 6}, {3, 8}, {3, 5}, {5, 0},
      {2, 6}, {3, 7}, {3, 4}, {4, 1}, {2, 6}, {3, 8}, {3, 5}, {4, 3},
      {2, 6}, {3, 7}, {3, 4}, {4, 2}, {2, 6}, {3, 8}, {3, 5}, {6, 9},
      {2, 6}, {3, 7}, {3, 4}, {4, 1}, {2, 6}, {3, 8}, {3, 5}, {4, 3},
      {2, 6}, {3, 7}, {3, 4}, {4, 2}, {2, 6}, {3, 8}, {3, 5}, {5, 0},
      {2, 6}, {3, 7}, {3, 4}, {4, 1}, {2, 6}, {3, 8}, {3, 5}, {4, 3},
      {2, 6}, {3, 7}, {3, 4}, {4, 2}, {2, 6}, {3, 8}, {3, 5}, {6, 10},
    };
    std::vector<int> logcounts(counts->size());
    int omit_log = -1;
    int omit_pos = -1;
    for (int i = 0; i < logcounts.size(); ++i) {
      const HuffmanCode* p = huff;
      input->FillBitBuffer();
      p += input->PeekFixedBits<6>();
      input->Advance(p->bits);
      logcounts[i] = p->value;
      if (logcounts[i] > omit_log) {
        omit_log = logcounts[i];
        omit_pos = i;
      }
    }
    for (int i = 0; i < logcounts.size(); ++i) {
      int code = logcounts[i];
      if (i == omit_pos) {
        continue;
      } else if (code == 0) {
        continue;
      } else if (code == 1) {
        (*counts)[i] = 1;
      } else {
        int bitcount = GetPopulationCountPrecision(code - 1);
        (*counts)[i] = (1 << (code - 1)) +
            (input->ReadBits(bitcount) << (code - 1 - bitcount));
      }
      total_count += (*counts)[i];
    }
    PIK_ASSERT(omit_pos >= 0);
    (*counts)[omit_pos] = (1 << precision_bits) - total_count;
    if ((*counts)[omit_pos] <= 0) {
      // The histogram we've read sums to more than total_count (including at
      // least 1 for the omitted value).
      return false;
    }
  }
  return true;
}

}  // namespace pik
