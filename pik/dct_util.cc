// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/dct_util.h"

#undef PROFILER_ENABLED
#define PROFILER_ENABLED 1
#include "pik/bits.h"
#include "pik/common.h"
#include "pik/dct.h"
#include "pik/gauss_blur.h"
#include "pik/profiler.h"
#include "pik/simd/simd.h"
#include "pik/status.h"

namespace pik {

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
