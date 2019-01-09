// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef DCT_UTIL_H_
#define DCT_UTIL_H_

#include "common.h"
#include "data_parallel.h"
#include "image.h"

namespace pik {

// Scatter/gather a SX*SY block into (SX/kBlockDim)*(SY/kBlockDim)
// kBlockDim*kBlockDim blocks. `block` should be contiguous, SX*SY-sized. Output
// Each kBlockDim*kBlockDim block should be contiguous, and the same "block row"
// should be too, but different block rows are at a distance of `stride` pixels.
// This only copies the top KX times KY pixels, and assumes block to be a KX*KY
// block.
template <size_t SX, size_t SY, size_t KX = SX, size_t KY = SY>
void ScatterBlock(const float* PIK_RESTRICT block, float* PIK_RESTRICT row,
                  size_t stride) {
  constexpr size_t xblocks = SX / kBlockDim;
  constexpr size_t yblocks = SY / kBlockDim;
  for (size_t y = 0; y < KY; y++) {
    float* PIK_RESTRICT current_row =
        row + (y & (yblocks - 1)) * stride + (y / yblocks) * kBlockDim;
    for (size_t x = 0; x < KX; x++) {
      current_row[(x & (xblocks - 1)) * kDCTBlockSize + (x / xblocks)] =
          block[y * KX + x];
    }
  }
}

template <size_t SX, size_t SY, size_t KX = SX, size_t KY = SY>
void GatherBlock(const float* PIK_RESTRICT row, size_t stride,
                 float* PIK_RESTRICT block) {
  constexpr size_t xblocks = SX / kBlockDim;
  constexpr size_t yblocks = SY / kBlockDim;
  for (size_t y = 0; y < KY; y++) {
    const float* PIK_RESTRICT current_row =
        row + (y & (yblocks - 1)) * stride + (y / yblocks) * kBlockDim;
    for (size_t x = 0; x < KX; x++) {
      block[y * KX + x] =
          current_row[(x & (xblocks - 1)) * kDCTBlockSize + (x / xblocks)];
    }
  }
}

// Returns a (N*N)*W x H image where each (N*N)x1 block is produced with
// ComputeTransposedScaledDCT() from the corresponding NxN block of
// the image. Note that the whole coefficient image is scaled by 1 / (N*N)
// afterwards, so that ComputeTransposedScaledIDCT() applied to each block will
// return exactly the input image block.
// REQUIRES: img.xsize() == N*W, img.ysize() == N*H
SIMD_ATTR void TransposedScaledDCT(const Image3F& img,
                                   Image3F* PIK_RESTRICT dct);

}  // namespace pik

#endif  // DCT_UTIL_H_
