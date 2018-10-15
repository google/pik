#ifndef DCT_UTIL_H_
#define DCT_UTIL_H_

#include "data_parallel.h"
#include "image.h"

namespace pik {

// Returns an N x M image by taking the DC coefficient from each 64x1 block.
// REQUIRES: coeffs.xsize() == 64*N, coeffs.ysize() == M
template <size_t N, typename T>
Image<T> DCImage(const Image<T>& coeffs) {
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

template <size_t N, typename T>
Image3<T> DCImage(const Image3<T>& coeffs) {
  return Image3<T>(DCImage<N>(coeffs.Plane(0)),
                   DCImage<N>(coeffs.Plane(1)),
                   DCImage<N>(coeffs.Plane(2)));
}

template <int N, typename T>
void FillDC(const Image<T>& dc, Image<T>* coeffs) {
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
template <int N, typename T>
void FillDC(const Image3<T>& dc, Image3<T>* coeffs) {
  for (int c = 0; c < 3; c++) {
    FillDC<N>(dc.Plane(c), coeffs->MutablePlane(c));
  }
}

// Returns an image that is defined by the following transformations:
//  1) Upsample image 8x8 with nearest-neighbor
//  2) Blur with a Gaussian kernel of radius 8 and given sigma
//  3) perform TransposedScaledDCT()
//  4) Zero out the DC coefficients
Image3F UpSample8x8BlurDCT(const Image3F& img, const float sigma);

// Adds to "add_to" (DCT) an image defined by the following transformations:
//  1) Upsample image 4x4 with nearest-neighbor
//  2) Blur with a Gaussian kernel of radius 4 and given sigma
//  3) perform TransposedScaledDCT()
//  4) Zero out the top 2x2 corner of each DCT block
//  5) XOR with "sign" (decoder adds predictions, encoder would subtract)
void UpSample4x4BlurDCT(const Image3F& img, const float sigma, const float sign,
                        ThreadPool* pool, Image3F* add_to);

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
template <size_t N>
Image3F TransposedScaledDCT(const Image3F& img, ThreadPool* pool);

// Same as above, but only DC coefficients are computed. Not "transposed",
// because output consists of single coefficient per block.
template <size_t N>
ImageF ScaledDC(const ImageF& image, ThreadPool* pool);

template <size_t N>
Image3F ScaledDC(const Image3F& image, ThreadPool* pool);

}  // namespace pik

#endif  // DCT_UTIL_H_
