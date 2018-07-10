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

#include "image.h"

namespace pik {

constexpr size_t kBitsPerByte = 8;  // more clear than CHAR_BIT

template <typename T>
inline T DivCeil(T a, T b) {
  return (a + b - 1) / b;
}

// The suffix "InBlocks" indicates the measuring unit. If blank, it is pixels.

// Block is the rectangular grid of pixels to which an "energy compaction"
// transformation (e.g. DCT) is applied. Each block has its own AC quantizer.
constexpr size_t kBlockWidth = 8;
constexpr size_t kBlockHeight = 8;
// "size" is the number of coefficients required to describe transformed block.
// It matches the number of pixels because the transform is not overcomplete.
constexpr size_t kBlockSize = kBlockWidth * kBlockHeight;

// Tile is the rectangular grid of blocks that share a "color transform".
constexpr size_t kTileWidthInBlocks = 8;
constexpr size_t kTileHeightInBlocks = 8;
constexpr size_t kTileWidth = kTileWidthInBlocks * kBlockWidth;
constexpr size_t kTileHeight = kTileHeightInBlocks * kBlockHeight;

// Group is the rectangular grid of tiles that can be decoded in parallel.
constexpr size_t kGroupWidthInTiles = 8;
constexpr size_t kGroupHeightInTiles = 8;
constexpr size_t kGroupWidthInBlocks = kGroupWidthInTiles * kTileWidthInBlocks;
constexpr size_t kGroupHeightInBlocks =
    kGroupHeightInTiles * kTileHeightInBlocks;

}  // namespace pik

#endif  // COMMON_H_
