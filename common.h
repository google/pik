// Copyright 2018 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef COMMON_H_
#define COMMON_H_

// Shared constants and helper functions.

#include <stddef.h>
#include <memory>  // unique_ptr

namespace pik {

constexpr size_t kBitsPerByte = 8;  // more clear than CHAR_BIT

template <typename T>
constexpr inline T DivCeil(T a, T b) {
  return (a + b - 1) / b;
}

template <typename T>
constexpr T Pi(T multiplier) {
  return static_cast<T>(multiplier * 3.1415926535897932);
}

// Block is the square grid of pixels to which an "energy compaction"
// transformation (e.g. DCT) is applied. Each block has its own AC quantizer.
constexpr size_t kBlockDim = 8;

constexpr size_t kDCTBlockSize = kBlockDim * kBlockDim;

// Group is the rectangular grid of blocks that can be decoded in parallel. This
// is different for DC.
constexpr size_t kDcGroupDimInBlocks = 256;
constexpr size_t kGroupWidth = 512;
constexpr size_t kGroupHeight = 512;
static_assert(kGroupWidth % kBlockDim == 0,
              "Group width should be divisible by block dim");
static_assert(kGroupHeight % kBlockDim == 0,
              "Group height should be divisible by block dim");
constexpr size_t kGroupWidthInBlocks = kGroupWidth / kBlockDim;
constexpr size_t kGroupHeightInBlocks = kGroupHeight / kBlockDim;

// We split groups into tiles to increase locality and cache hits.
const constexpr size_t kTileDim = 64;

static_assert(kTileDim % kBlockDim == 0,
              "Tile dim should be divisible by block dim");
constexpr size_t kTileDimInBlocks = kTileDim / kBlockDim;

static_assert(kGroupWidthInBlocks % kTileDimInBlocks == 0,
              "Group dim should be divisible by tile dim");

// Can't rely on C++14 yet.
template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

}  // namespace pik

#endif  // COMMON_H_
