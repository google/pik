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

#ifndef BLOCK_H_
#define BLOCK_H_

#include "common.h"
#include "compiler_specific.h"
#include "simd/simd.h"

namespace pik {

// Adapters for source/destination.
//
// Block: (x, y) <-> (N * y + x)
// Lines: (x, y) <-> (stride * y + x)
//
// I.e. Block is a specialization of Lines with fixed stride.
//
// FromXXX should implement Read and Load (Read vector).
// ToXXX should implement Write and Store (Write vector).

// The AVX2 8x8 column DCT cannot process more than 8 independent lanes.
using BlockDesc = SIMD_PART(float, SIMD_MIN(8, SIMD_FULL(float)::N));

template <size_t N>
class FromBlock {
 public:
  explicit FromBlock(const float* block) : block_(block) {}

  FromBlock View(size_t dx, size_t dy) const {
    return FromBlock<N>(Address(dx, dy));
  }

  SIMD_ATTR PIK_INLINE BlockDesc::V Load(const size_t row, size_t i) const {
    return load(BlockDesc(), block_ + row * N + i);
  }

  SIMD_ATTR PIK_INLINE float Read(const size_t row, const size_t i) const {
    return *Address(row, i);
  }

  constexpr PIK_INLINE const float* Address(const size_t row,
                                            const size_t i) const {
    return block_ + row * N + i;
  }

 private:
  const float* block_;
};

template <size_t N>
class ToBlock {
 public:
  explicit ToBlock(float* block) : block_(block) {}

  ToBlock View(size_t dx, size_t dy) const {
    return ToBlock<N>(Address(dx, dy));
  }

  SIMD_ATTR PIK_INLINE void Store(const BlockDesc::V& v, const size_t row,
                                  const size_t i) const {
    store(v, BlockDesc(), Address(row, i));
  }

  SIMD_ATTR PIK_INLINE void Write(float v, const size_t row,
                                  const size_t i) const {
    *Address(row, i) = v;
  }

  constexpr PIK_INLINE float* Address(const size_t row, const size_t i) const {
    return block_ + row * N + i;
  }

 private:
  float* block_;
};

// Same as ToBlock, but multiplies result by (N * N)
// TODO(user): perhaps we should get rid of this one.
template <size_t N>
class ScaleToBlock {
 public:
  explicit SIMD_ATTR ScaleToBlock(float* block)
      : block_(block), mul_(set1(BlockDesc(), 1.0f / (N * N))) {}

  SIMD_ATTR PIK_INLINE void Store(const BlockDesc::V& v, const size_t row,
                                  const size_t i) const {
    store(v * mul_, BlockDesc(), Address(row, i));
  }

  SIMD_ATTR PIK_INLINE void Write(float v, const size_t row,
                                  const size_t i) const {
    *Address(row, i) = v;
  }

  constexpr PIK_INLINE float* Address(const size_t row, const size_t i) const {
    return block_ + row * N + i;
  }

 private:
  float* block_;
  BlockDesc::V mul_;
};

class FromLines {
 public:
  FromLines(const float* top_left, size_t stride)
      : top_left_(top_left), stride_(stride) {}

  FromLines View(size_t dx, size_t dy) const {
    return FromLines(Address(dx, dy), stride_);
  }

  SIMD_ATTR PIK_INLINE BlockDesc::V Load(const size_t row,
                                         const size_t i) const {
    return load(BlockDesc(), Address(row, i));
  }

  SIMD_ATTR PIK_INLINE float Read(const size_t row, const size_t i) const {
    return *Address(row, i);
  }

  PIK_INLINE const float* SIMD_RESTRICT Address(const size_t row,
                                                const size_t i) const {
    return top_left_ + row * stride_ + i;
  }

 private:
  const float* SIMD_RESTRICT top_left_;
  size_t stride_;  // move to next line by adding this to pointer
};

// Pointers are restrict-qualified: assumes we don't use both FromLines and
// ToLines in the same DCT. NOTE: Transpose uses From/ToBlock, not *Lines.
class ToLines {
 public:
  ToLines(float* top_left, size_t stride)
      : top_left_(top_left), stride_(stride) {}

  ToLines View(const ToLines& other, size_t dx, size_t dy) const {
    return ToLines(Address(dx, dy), stride_);
  }

  SIMD_ATTR PIK_INLINE void Store(const BlockDesc::V& v, const size_t row,
                                  const size_t i) const {
    store(v, BlockDesc(), Address(row, i));
  }

  SIMD_ATTR PIK_INLINE void Write(float v, const size_t row,
                                  const size_t i) const {
    *Address(row, i) = v;
  }

  PIK_INLINE float* SIMD_RESTRICT Address(const size_t row,
                                          const size_t i) const {
    return top_left_ + row * stride_ + i;
  }

 private:
  float* SIMD_RESTRICT top_left_;
  size_t stride_;  // move to next line by adding this to pointer
};

}  // namespace pik

#endif  // BLOCK_H_
