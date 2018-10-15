#include "dct_util.h"

#undef PROFILER_ENABLED
#define PROFILER_ENABLED 1
#include "bits.h"
#include "dct.h"
#include "gauss_blur.h"
#include "profiler.h"
#include "simd/simd.h"
#include "status.h"

namespace pik {

namespace {

// "Adds" (if sign == +0, otherwise subtracts if sign == -0) block to "add_to",
// except elements 0,H,V,D. May overwrite parts of "block".
SIMD_ATTR void AddBlockExcept0HVDTo(float* PIK_RESTRICT block, const float sign,
                                    float* PIK_RESTRICT add_to) {
  constexpr int N = kBlockDim;

  const SIMD_PART(float, SIMD_MIN(SIMD_FULL(float)::N, 8)) d;

#if SIMD_TARGET_VALUE == SIMD_NONE
  // Fallback because SIMD version assumes at least two lanes.
  block[0] = 0.0f;
  block[1] = 0.0f;
  block[N] = 0.0f;
  block[N + 1] = 0.0f;
  if (ext::movemask(set1(d, sign))) {
    for (size_t i = 0; i < N * N; ++i) {
      add_to[i] -= block[i];
    }
  } else {
    for (size_t i = 0; i < N * N; ++i) {
      add_to[i] += block[i];
    }
  }
#else
  // Negated to enable default zero-initialization of upper lanes.
  SIMD_ALIGN uint32_t mask2[d.N] = {~0u, ~0u};
  const auto only_01 = load(d, reinterpret_cast<float*>(mask2));
  const auto vsign = set1(d, sign);

  // First block row: don't add block[0, 1].
  auto prev = load(d, add_to + 0);
  auto coefs = load(d, block + 0);
  auto sum = prev + (andnot(only_01, coefs) ^ vsign);
  store(sum, d, add_to + 0);
  // Handle remnants of DCT row (for 128-bit SIMD, or N > 8)
  for (size_t ix = d.N; ix < N; ix += d.N) {
    prev = load(d, add_to + ix);
    coefs = load(d, block + ix);
    sum = prev + (coefs ^ vsign);
    store(sum, d, add_to + ix);
  }

  // Second block row: don't add block[V, D].
  prev = load(d, add_to + N);
  coefs = load(d, block + N);
  sum = prev + (andnot(only_01, coefs) ^ vsign);
  store(sum, d, add_to + N);
  // Handle remnants of DCT row (for 128-bit SIMD, or N > 8)
  for (size_t ix = d.N; ix < N; ix += d.N) {
    prev = load(d, add_to + N + ix);
    coefs = load(d, block + N + ix);
    sum = prev + (coefs ^ vsign);
    store(sum, d, add_to + N + ix);
  }

  for (size_t i = 2 * N; i < N * N; i += d.N) {
    prev = load(d, add_to + i);
    coefs = load(d, block + i);
    sum = prev + (coefs ^ vsign);
    store(sum, d, add_to + i);
  }
#endif
}

}  // namespace

SIMD_ATTR void UpSample4x4BlurDCT(const Image3F& img, const float sigma,
                                  const float sign, ThreadPool* pool,
                                  Image3F* add_to) {
  // TODO(user): There's no good reason to compute the full DCT here. It's
  // fine if the output is in pixel space, we just need to zero out top 2x2 DCT
  // coefficients. We can do that by computing a "partial DCT" and subtracting
  // (we can have two outputs: a positive pixel-space output and a negative
  // DCT-space output).

  // TODO(user): Failing that, merge the blur and DCT into a single linear
  // operation, if feasible.
  const int xs = img.xsize();
  const int ys = img.ysize();
  const int bxs = xs / 2;
  const int bys = ys / 2;
  PIK_CHECK(add_to->xsize() == 64 * bxs && add_to->ysize() == bys);

  float w0[4] = {0.0f};
  float w1[4] = {0.0f};
  float w2[4] = {0.0f};
  std::vector<float> kernel = GaussianKernel(4, sigma);
  for (int k = 0; k < 4; ++k) {
    const int split0 = 4 - k;
    const int split1 = 8 - k;
    for (int j = 0; j < split0; ++j) {
      w0[k] += kernel[j];
    }
    for (int j = split0; j < split1; ++j) {
      w1[k] += kernel[j];
    }
    for (int j = split1; j < kernel.size(); ++j) {
      w2[k] += kernel[j];
    }
    w0[k] *= 0.125f;
    w1[k] *= 0.125f;
    w2[k] *= 0.125f;
  }

  using D = SIMD_PART(float, SIMD_MIN(SIMD_FULL(float)::N, 8));
  using V = D::V;
  const D d;
  V vw0[4] = {set1(d, w0[0]), set1(d, w0[1]), set1(d, w0[2]), set1(d, w0[3])};
  V vw1[4] = {set1(d, w1[0]), set1(d, w1[1]), set1(d, w1[2]), set1(d, w1[3])};
  V vw2[4] = {set1(d, w2[0]), set1(d, w2[1]), set1(d, w2[2]), set1(d, w2[3])};

  Image3F blur_x(xs * 4, ys);
  std::vector<ImageF> padded;
  for (size_t i = 0; i < std::max<size_t>(pool->NumThreads(), 1); ++i) {
    padded.emplace_back(xs + 2, 1);
  }
  pool->Run(0, ys, [&](const int task, const int thread) {
    const size_t y = task;
    float* PIK_RESTRICT row_tmp = padded[thread].Row(0);
    for (int c = 0; c < 3; ++c) {
      const float* PIK_RESTRICT row = img.PlaneRow(c, y);
      memcpy(&row_tmp[1], row, xs * sizeof(row[0]));
      row_tmp[0] = row_tmp[1 + std::min(1, xs - 1)];
      row_tmp[xs + 1] = row_tmp[1 + std::max(0, xs - 2)];
      float* const PIK_RESTRICT row_out = blur_x.PlaneRow(c, y);
      for (int x = 0; x < xs; ++x) {
        const float v0 = row_tmp[x];
        const float v1 = row_tmp[x + 1];
        const float v2 = row_tmp[x + 2];
        for (int ix = 0; ix < 4; ++ix) {
          row_out[4 * x + ix] = v0 * w0[ix] + v1 * w1[ix] + v2 * w2[ix];
        }
      }
    }
  });

  pool->Run(0, bys,
            [bxs, bys, &vw0, &vw1, &vw2, &blur_x, sign, add_to](
                const int by, const int thread) {
              const D d;
              SIMD_ALIGN float block[64];

              for (int c = 0; c < 3; ++c) {
                float* PIK_RESTRICT row_out = add_to->PlaneRow(c, by);
                const int by0 = by == 0 ? 1 : 2 * by - 1;
                const int by1 = 2 * by;
                const int by2 = 2 * by + 1;
                const int by3 = by + 1 < bys ? 2 * by + 2 : 2 * by;
                const float* PIK_RESTRICT row0 = blur_x.ConstPlaneRow(c, by0);
                const float* PIK_RESTRICT row1 = blur_x.ConstPlaneRow(c, by1);
                const float* PIK_RESTRICT row2 = blur_x.ConstPlaneRow(c, by2);
                const float* PIK_RESTRICT row3 = blur_x.ConstPlaneRow(c, by3);
                for (int bx = 0; bx < bxs; ++bx) {
                  for (int ix = 0; ix < 8; ix += d.N) {
                    const auto val0 = load(d, &row0[bx * 8 + ix]);
                    const auto val1 = load(d, &row1[bx * 8 + ix]);
                    const auto val2 = load(d, &row2[bx * 8 + ix]);
                    const auto val3 = load(d, &row3[bx * 8 + ix]);
                    for (int iy = 0; iy < 4; ++iy) {
                      // A mul_add pair is faster but causes 1E-5 difference.
                      const auto vala =
                          val0 * vw0[iy] + val1 * vw1[iy] + val2 * vw2[iy];
                      const auto valb =
                          val1 * vw0[iy] + val2 * vw1[iy] + val3 * vw2[iy];
                      store(vala, d, &block[iy * 8 + ix]);
                      store(valb, d, &block[iy * 8 + 32 + ix]);
                    }
                  }
                  ComputeTransposedScaledDCT<8>(FromBlock<8>(block),
                                                ToBlock<8>(block));
                  AddBlockExcept0HVDTo(block, sign, row_out + 64 * bx);
                }
              }
            },
            "dct upsample");
}

template <size_t N>
static SIMD_ATTR PIK_INLINE void ScaledDcRow(const ImageF& image, size_t by,
                                             ImageF* out) {
  const size_t xsize_blocks = image.xsize() / N;
  const size_t stride = image.PixelsPerRow();
  const float* PIK_RESTRICT row_in = image.Row(by * N);
  constexpr float mul = 1.0f / (N * N);
  float* PIK_RESTRICT row_out = out->Row(by);
  for (size_t bx = 0; bx < xsize_blocks; ++bx) {
    row_out[bx] = ComputeScaledDC<N>(FromLines(row_in + bx * N, stride)) * mul;
  }
}

template <size_t N>
SIMD_ATTR ImageF ScaledDC(const ImageF& image, ThreadPool* pool) {
  PROFILER_FUNC;

  PIK_CHECK(image.xsize() % N == 0);
  PIK_CHECK(image.ysize() % N == 0);
  const size_t xsize_blocks = image.xsize() / N;
  const size_t ysize_blocks = image.ysize() / N;
  ImageF out(xsize_blocks, ysize_blocks);

  // TODO(user): perhaps, scheduling other than "by-line" would be better.
  pool->Run(0, ysize_blocks,
            [&image, &out](const int task, const int thread) {
              ScaledDcRow<N>(image, task, &out);
            },
            "dct scale");
  return out;
}
template ImageF ScaledDC<8>(const ImageF& image, ThreadPool* pool);

template <size_t N>
SIMD_ATTR Image3F ScaledDC(const Image3F& image, ThreadPool* pool) {
  PROFILER_FUNC;

  PIK_CHECK(image.xsize() % N == 0);
  PIK_CHECK(image.ysize() % N == 0);
  const size_t xsize_blocks = image.xsize() / N;
  const size_t ysize_blocks = image.ysize() / N;
  Image3F out(xsize_blocks, ysize_blocks);

  // TODO(user): perhaps, scheduling other than "by-line" would be better.
  pool->Run(0, ysize_blocks,
            [&image, &out](const int task, const int thread) {
              for (int c = 0; c < 3; ++c) {
                ScaledDcRow<N>(image.Plane(c), task, out.MutablePlane(c));
              }
            },
            "dct scale3");
  return out;
}
template Image3F ScaledDC<8>(const Image3F& image, ThreadPool* pool);

template <size_t N>
SIMD_ATTR Image3F TransposedScaledDCT(const Image3F& img, ThreadPool* pool) {
  constexpr int block_size = N * N;
  PIK_ASSERT(img.xsize() % N == 0);
  PIK_ASSERT(img.ysize() % N == 0);
  const size_t xsize_blocks = img.xsize() / N;
  const size_t ysize_blocks = img.ysize() / N;
  Image3F coeffs(xsize_blocks * block_size, ysize_blocks);

  pool->Run(0, ysize_blocks,
            [&img, &coeffs, &xsize_blocks](const int task, const int thread) {
              const size_t by = task;
              const size_t stride = img.PixelsPerRow();
              for (int c = 0; c < 3; ++c) {
                const float* PIK_RESTRICT row_in = img.PlaneRow(c, by * N);
                float* PIK_RESTRICT row_out = coeffs.PlaneRow(c, by);

                for (size_t bx = 0; bx < xsize_blocks; ++bx) {
                  ComputeTransposedScaledDCT<N>(
                      FromLines(row_in + bx * N, stride),
                      ScaleToBlock<N>(row_out + bx * block_size));
                }
              }
            },
            "dct TransposedScaled");
  return coeffs;
}
template Image3F TransposedScaledDCT<8>(const Image3F&, ThreadPool*);

}  // namespace pik
