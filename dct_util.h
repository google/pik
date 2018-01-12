#ifndef DCT_UTIL_H_
#define DCT_UTIL_H_

#include "image.h"

namespace pik {

// Returns same size image, such that ComputeBlockIDCTFloat could be called
// for each block (interpretation/layout: see TransposedScaledIDCT).
// REQUIRES: xsize() == 64*N
Image3F UndoTransposeAndScale(const Image3F& transposed_scaled);

// Returns a 8*N x 8*M image where each 8x8 block is produced with
// ComputeBlockIDCTFloat() from the corresponding 64x1 block of
// the coefficient image, which is obtained from UndoTransposeAndScale.
// REQUIRES: coeffs.xsize() == 64*N, coeffs.ysize() == M.
// Intended for testing deblocking algorithms.
Image3F SlowIDCT(const Image3F& coeffs);

// Returns a 8*N x 8*M image where each 8x8 block is produced with
// ComputeTransposedScaledBlockIDCTFloat() from the corresponding 64x1 block of
// the coefficient image.
// REQUIRES: coeffs.xsize() == 64*N, coeffs.ysize() == M
Image3F TransposedScaledIDCT(const Image3F& coeffs);

// Returns a 64*N x M image where each 64x1 block is produced with
// ComputeTransposedScaledBlockDCTFloat() from the corresponding 8x8 block of
// the image. Note that the whole coefficient image is scaled by 1/64
// afterwards, so that this is exactly the inverse of TransposedScaledIDCT().
// REQUIRES: coeffs.xsize() == 8*N, coeffs.ysize() == 8*M
Image3F TransposedScaledDCT(const Image3F& img);

// Returns an N x M image by taking the DC coefficient from each 64x1 block.
// REQUIRES: coeffs.xsize() == 64*N, coeffs.ysize() == M
Image3F DCImage(const Image3F& coeffs);

// Zeroes out the top-left 2x2 corner of each DCT block.
// REQUIRES: coeffs.xsize() == 64*N, coeffs.ysize() == M
void ZeroOut2x2(Image3F* coeffs);
Image3F KeepOnly2x2Corners(const Image3F& coeffs);

// Returns a 2*N x 2*M image which is defined by the following 3 transforms:
//  1) zero out every coefficient that is outside the top 2x2 corner
//  2) perform TransposedScaledIDCT()
//  3) subsample the result 4x4 by taking simple averages
// REQUIRES: coeffs.xsize() == 64*N, coeffs.ysize() == M
Image3F GetPixelSpaceImageFrom2x2Corners(const Image3F& coeffs);

// Puts back the top 2x2 corner of each 8x8 block of *coeffs from the
// transformed pixel space image img.
// REQUIRES: coeffs->xsize() == 64*N, coeffs->ysize() == M
void Add2x2CornersFromPixelSpaceImage(const Image3F& img,
                                      Image3F* coeffs);

// Returns an image that is defined by the following transformations:
//  1) Upsample image 8x8 with nearest-neighbor
//  2) Blur with a Gaussian kernel of radius 8 and given sigma
//  3) perform TransposedScaledDCT()
//  4) Zero out the DC coefficients
Image3F UpSample8x8BlurDCT(const Image3F& img, const float sigma);

// Returns an image that is defined by the following transformations:
//  1) Upsample image 4x4 with nearest-neighbor
//  2) Blur with a Gaussian kernel of radius 4 and given sigma
//  3) perform TransposedScaledDCT()
//  4) Zero out the top 2x2 corner of each DCT block
Image3F UpSample4x4BlurDCT(const Image3F& img, const float sigma);

// Returns an image that is defined by the following transformations:
//  1) Upsample image 8x8 with nearest-neighbor
//  2) Blur with a Gaussian kernel of radius 8 and given sigma
Image3F UpSample8x8Blur(const Image3F& img, const float sigma);

// Returns an image that is defined by the following transformations:
//  1) Upsample image 4x4 with nearest-neighbor
//  2) Blur with a Gaussian kernel of radius 4 and given sigma
Image3F UpSample4x4Blur(const Image3F& img, const float sigma);

// Returns an N x M image where each pixel is the average of the corresponding
// f x f block in the original.
// REQUIRES: image.xsize() == f*N, image.ysize() == f *M
ImageF Subsample(const ImageF& image, int f);
Image3F Subsample(const Image3F& image, int f);

// Returns an f*N x f*M upsampled image where each fxf block has the same value
// as the corresponding pixel in the original image.
ImageF Upsample(const ImageF& image, int f);
Image3F Upsample(const Image3F& image, int f);

// Takes the maximum of the 3x3 block around each pixel.
ImageF Dilate(const ImageF& in);
Image3F Dilate(const Image3F& in);

// Takes the minimum of the 3x3 block around each pixel.
ImageF Erode(const ImageF& in);
Image3F Erode(const Image3F& in);

// Takes the pixel-by-pixel min/max of the two inputs.
ImageF Min(const ImageF& a, const ImageF& b);
ImageF Max(const ImageF& a, const ImageF& b);
Image3F Min(const Image3F& a, const Image3F& b);
Image3F Max(const Image3F& a, const Image3F& b);

}  // namespace pik

#endif  // DCT_UTIL_H_
