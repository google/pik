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

#ifndef HUFFMAN_ENCODE_H_
#define HUFFMAN_ENCODE_H_

#include <stdint.h>
#include <cstddef>

namespace pik {

void BuildAndStoreHuffmanTree(const uint32_t* histogram, const size_t length,
                              uint8_t* depth, uint16_t* bits,
                              size_t* storage_ix, uint8_t* storage);

void BuildHuffmanTreeAndCountBits(const uint32_t* histogram,
                                  const size_t length, size_t* histogram_bits,
                                  size_t* data_bits);
}  // namespace pik

#endif  // HUFFMAN_ENCODE_H_
