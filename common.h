// Copyright 2018 Google Inc. All Rights Reserved.
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

#ifndef COMMON_H_
#define COMMON_H_

#include <stddef.h>

namespace pik {

constexpr size_t kBitsPerByte = 8;  // more clear than CHAR_BIT

template <typename T>
inline T DivCeil(T a, T b) {
  return (a + b - 1) / b;
}

template <typename T>
constexpr T Pi(T multiplier) {
  return static_cast<T>(multiplier * 3.1415926535897932);
}

// Block is the square grid of pixels to which an "energy compaction"
// transformation (e.g. DCT) is applied. Each block has its own AC quantizer.
constexpr size_t kBlockDim = 8;

// Group is the rectangular grid of blocks that can be decoded in parallel.
constexpr size_t kGroupWidth = 512;
constexpr size_t kGroupHeight = 512;
static_assert(kGroupWidth % kBlockDim == 0,
              "Group width should be divisible by block dim");
static_assert(kGroupHeight % kBlockDim == 0,
              "Group height should be divisible by block dim");
constexpr size_t kGroupWidthInBlocks = kGroupWidth / kBlockDim;
constexpr size_t kGroupHeightInBlocks = kGroupHeight / kBlockDim;

}  // namespace pik

#endif  // COMMON_H_
