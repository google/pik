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

#include "context_map_decode.h"

#include <cstring>
#include <vector>

#include "huffman_decode.h"

namespace pik {

namespace {

void MoveToFront(uint8_t* v, uint8_t index) {
  uint8_t value = v[index];
  uint8_t i = index;
  for (; i; --i) v[i] = v[i - 1];
  v[0] = value;
}

void InverseMoveToFrontTransform(uint8_t* v, int v_len) {
  uint8_t mtf[256];
  int i;
  for (i = 0; i < 256; ++i) {
    mtf[i] = static_cast<uint8_t>(i);
  }
  for (i = 0; i < v_len; ++i) {
    uint8_t index = v[i];
    v[i] = mtf[index];
    if (index) MoveToFront(mtf, index);
  }
}

// Decodes a number in the range [0..255], by reading 1 - 11 bits.
inline int DecodeVarLenUint8(BitReader* input) {
  if (input->ReadBits(1)) {
    int nbits = static_cast<int>(input->ReadBits(3));
    if (nbits == 0) {
      return 1;
    } else {
      return static_cast<int>(input->ReadBits(nbits)) + (1 << nbits);
    }
  }
  return 0;
}

}  // namespace

bool DecodeContextMap(std::vector<uint8_t>* context_map,
                      size_t* num_htrees,
                      BitReader* input) {
  *num_htrees = DecodeVarLenUint8(input) + 1;

  if (*num_htrees <= 1) {
    memset(&(*context_map)[0], 0, context_map->size());
    return true;
  }

  int max_run_length_prefix = 0;
  int use_rle_for_zeros = input->ReadBits(1);
  if (use_rle_for_zeros) {
    max_run_length_prefix = input->ReadBits(4) + 1;
  }
  HuffmanDecodingData entropy;
  if (!entropy.ReadFromBitStream(input)) {
    return PIK_FAILURE("Invalid histogram data.");
  }
  HuffmanDecoder decoder;
  int i;
  for (i = 0; i < context_map->size();) {
    int code;
    code = decoder.ReadSymbol(entropy, input);
    if (code == 0) {
      (*context_map)[i] = 0;
      ++i;
    } else if (code <= max_run_length_prefix) {
      int reps = 1 + (1 << code) + input->ReadBits(code);
      while (--reps) {
        if (i >= context_map->size()) {
          return PIK_FAILURE("Invalid context map data.");
        }
        (*context_map)[i] = 0;
        ++i;
      }
    } else {
      (*context_map)[i] = static_cast<uint8_t>(code - max_run_length_prefix);
      ++i;
    }
  }
  if (input->ReadBits(1)) {
    InverseMoveToFrontTransform(&(*context_map)[0], context_map->size());
  }
  return true;
}

}  // namespace pik
