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

// Library to decode the Huffman code lengths from the bit-stream and build a
// decoding table from them.

#ifndef HUFFMAN_DECODE_H_
#define HUFFMAN_DECODE_H_

#include <stddef.h>
#include <stdint.h>
#include <memory>
#include <vector>

#include "bit_reader.h"

namespace pik {

static const int kHuffmanMaxLength = 15;
static const int kHuffmanTableMask = 0xff;
static const int kHuffmanTableBits = 8;

typedef struct {
  uint8_t bits;   /* number of bits used for this symbol */
  uint16_t value; /* symbol value or table offset */
} HuffmanCode;

struct HuffmanDecodingData {
  HuffmanDecodingData() { table_.reserve(2048); }

  // Decodes the Huffman code lengths from the bit-stream and fills in the
  // pre-allocated table with the corresponding 2-level Huffman decoding table.
  // Returns false if the Huffman code lengths can not de decoded.
  bool ReadFromBitStream(BitReader* input);

  std::vector<HuffmanCode> table_;
};

struct HuffmanDecoder {
  // Decodes the next Huffman coded symbol from the bit-stream.
  int ReadSymbol(const HuffmanDecodingData& code, BitReader* input) {
    int nbits;
    const HuffmanCode* table = &code.table_[0];
    input->FillBitBuffer();
    table += input->PeekFixedBits<kHuffmanTableBits>();
    nbits = table->bits - kHuffmanTableBits;
    if (nbits > 0) {
      input->Advance(kHuffmanTableBits);
      table += table->value;
      table += input->PeekBits(nbits);
    }
    input->Advance(table->bits);
    return table->value;
  }
};

}  // namespace pik

#endif  // HUFFMAN_DECODE_H_
