#ifndef DCT_UTIL_H_
#define DCT_UTIL_H_

#include "common.h"
#include "data_parallel.h"
#include "image.h"

namespace pik {

// Returns an N x M image by taking the DC coefficient from each 64x1 block.
// REQUIRES: coeffs.xsize() == 64*N, coeffs.ysize() == M
template <typename T>
Image<T> DCImage(const Image<T>& coeffs) {
  constexpr int N = kBlockDim;
  constexpr size_t block_size = N * N;
  PIK_ASSERT(coeffs.xsize() % block_size == 0);
  Image<T> out(coeffs.xsize() / block_size, coeffs.ysize());
  for (size_t y = 0; y < out.ysize(); ++y) {
    const T* PIK_RESTRICT row_in = coeffs.ConstRow(y);
    T* PIK_RESTRICT row_out = out.Row(y);
    for (size_t x = 0; x < out.xsize(); ++x) {
      row_out[x] = row_in[x * block_size];
    }
  }
  return out;
}

template <typename T>
Image3<T> DCImage(const Image3<T>& coeffs) {
  return Image3<T>(DCImage(coeffs.Plane(0)), DCImage(coeffs.Plane(1)),
                   DCImage(coeffs.Plane(2)));
}

template <typename T>
void FillDC(const Image<T>& dc, Image<T>* coeffs) {
  constexpr int N = kBlockDim;
  constexpr size_t block_size = N * N;
  const size_t xsize = dc.xsize();
  const size_t ysize = dc.ysize();

  for (size_t y = 0; y < ysize; y++) {
    const T* PIK_RESTRICT row_dc = dc.ConstRow(y);
    T* PIK_RESTRICT row_out = coeffs->Row(y);
    for (size_t x = 0; x < xsize; ++x) {
      row_out[block_size * x] = row_dc[x];
    }
  }
}

// Scatters dc into "coeffs" at offset 0 within 1x64 blocks.
template <typename T>
void FillDC(const Image3<T>& dc, Image3<T>* coeffs) {
  for (int c = 0; c < 3; c++) {
    FillDC(dc.Plane(c), coeffs->MutablePlane(c));
  }
}

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

// Returns an image that is defined by the following transformations:
//  1) Upsample image 8x8 with nearest-neighbor
//  2) Blur with a Gaussian kernel of radius 8 and given sigma
//  3) perform TransposedScaledDCT()
//  4) Zero out the DC coefficients
Image3F UpSample8x8BlurDCT(const Image3F& img, const float sigma);

// Returns an image that is defined by the following transformations:
//  1) Upsample image 8x8 with nearest-neighbor
//  2) Blur with a Gaussian kernel of radius 8 and given sigma
Image3F UpSample8x8Blur(const Image3F& img, const float sigma);

// Returns a (N*N)*W x H image where each (N*N)x1 block is produced with
// ComputeTransposedScaledDCT() from the corresponding NxN block of
// the image. Note that the whole coefficient image is scaled by 1 / (N*N)
// afterwards, so that ComputeTransposedScaledIDCT() applied to each block will
// return exactly the input image block.
// REQUIRES: img.xsize() == N*W, img.ysize() == N*H
SIMD_ATTR void TransposedScaledDCT(const Image3F& img,
                                   Image3F* PIK_RESTRICT dct);

// Same as above, but only DC coefficients are computed. Not "transposed",
// because output consists of single coefficient per block.
ImageF ScaledDC(const ImageF& image, ThreadPool* pool);

Image3F ScaledDC(const Image3F& image, ThreadPool* pool);

}  // namespace pik

#endif  // DCT_UTIL_H_
