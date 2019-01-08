#include "dct_util.h"

#undef PROFILER_ENABLED
#define PROFILER_ENABLED 1
#include "bits.h"
#include "common.h"
#include "dct.h"
#include "gauss_blur.h"
#include "profiler.h"
#include "simd/simd.h"
#include "status.h"

namespace pik {

static SIMD_ATTR PIK_INLINE void ScaledDcRow(const ImageF& image, size_t by,
                                             ImageF* out) {
  const constexpr size_t N = kBlockDim;
  const size_t xsize_blocks = image.xsize() / N;
  const size_t stride = image.PixelsPerRow();
  const float* PIK_RESTRICT row_in = image.Row(by * N);
  constexpr float mul = 1.0f / (N * N);
  float* PIK_RESTRICT row_out = out->Row(by);
  for (size_t bx = 0; bx < xsize_blocks; ++bx) {
    row_out[bx] =
        ComputeScaledDC<N>(FromLines<N>(row_in + bx * N, stride)) * mul;
  }
}

SIMD_ATTR ImageF ScaledDC(const ImageF& image, ThreadPool* pool) {
  PROFILER_FUNC;

  const constexpr size_t N = kBlockDim;
  PIK_CHECK(image.xsize() % N == 0);
  PIK_CHECK(image.ysize() % N == 0);
  const size_t xsize_blocks = image.xsize() / N;
  const size_t ysize_blocks = image.ysize() / N;
  ImageF out(xsize_blocks, ysize_blocks);

  // TODO(user): perhaps, scheduling other than "by-line" would be better.
  RunOnPool(
      pool, 0, ysize_blocks,
      [&image, &out](const int task, const int thread)
          SIMD_ATTR { ScaledDcRow(image, task, &out); },
      "dct scale");
  return out;
}

SIMD_ATTR Image3F ScaledDC(const Image3F& image, ThreadPool* pool) {
  PROFILER_FUNC;

  const constexpr size_t N = kBlockDim;
  PIK_CHECK(image.xsize() % N == 0);
  PIK_CHECK(image.ysize() % N == 0);
  const size_t xsize_blocks = image.xsize() / N;
  const size_t ysize_blocks = image.ysize() / N;
  Image3F out(xsize_blocks, ysize_blocks);

  // TODO(user): perhaps, scheduling other than "by-line" would be better.
  RunOnPool(
      pool, 0, ysize_blocks,
      [&image, &out](const int task, const int thread) SIMD_ATTR {
        for (int c = 0; c < 3; ++c) {
          ScaledDcRow(image.Plane(c), task, out.MutablePlane(c));
        }
      },
      "dct scale3");
  return out;
}

SIMD_ATTR void TransposedScaledDCT(const Image3F& img,
                                   Image3F* PIK_RESTRICT dct) {
  const constexpr size_t N = kBlockDim;
  constexpr int block_size = N * N;
  PIK_ASSERT(img.xsize() % N == 0);
  PIK_ASSERT(img.ysize() % N == 0);
  const size_t xsize_blocks = img.xsize() / N;
  const size_t ysize_blocks = img.ysize() / N;
  PIK_ASSERT(dct->xsize() == xsize_blocks * N * N);
  PIK_ASSERT(dct->ysize() == ysize_blocks);

  {
    PROFILER_ZONE("dct TransposedScaled2");
    for (size_t by = 0; by < ysize_blocks; ++by) {
      const size_t stride = img.PixelsPerRow();
      for (int c = 0; c < 3; ++c) {
        const float* PIK_RESTRICT row_in = img.PlaneRow(c, by * N);
        float* PIK_RESTRICT row_out = dct->PlaneRow(c, by);

        for (size_t bx = 0; bx < xsize_blocks; ++bx) {
          ComputeTransposedScaledDCT<N>()(
              FromLines<N>(row_in + bx * N, stride),
              ScaleToBlock<N>(row_out + bx * block_size));
        }
      }
    }
  }
}

}  // namespace pik
