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

Image3F UndoTransposeAndScale(const Image3F& transposed_scaled) {
  PIK_ASSERT(transposed_scaled.xsize() % 64 == 0);
  Image3F out(transposed_scaled.xsize(), transposed_scaled.ysize());
  SIMD_ALIGN float block[64];
      for (int c = 0; c < 3; ++c) {
        for (size_t y = 0; y < transposed_scaled.ysize(); ++y) {
          const float* PIK_RESTRICT row_in = transposed_scaled.PlaneRow(c, y);
          float* PIK_RESTRICT row_out = out.PlaneRow(c, y);
          for (size_t x = 0; x < transposed_scaled.xsize(); x += 64) {
            memcpy(block, row_in + x, sizeof(block));
            TransposeBlock(block);

            for (size_t iy = 0; iy < 8; ++iy) {
              const float rcp_sy = kRecipIDCTScales[iy];
              for (size_t ix = 0; ix < 8; ++ix) {
                block[iy * 8 + ix] *= rcp_sy * kRecipIDCTScales[ix];
              }
            }
            memcpy(row_out + x, block, sizeof(block));
          }
        }
  }
  return out;
}

void ZeroOut2x2(Image3F* coeffs) {
  PIK_ASSERT(coeffs->xsize() % 64 == 0);
      for (int c = 0; c < 3; ++c) {
        for (size_t y = 0; y < coeffs->ysize(); ++y) {
          float* PIK_RESTRICT row = coeffs->PlaneRow(c, y);
          for (size_t x = 0; x < coeffs->xsize(); x += 64) {
            row[x] = row[x + 1] = row[x + 8] = row[x + 9] = 0.0f;
          }
        }
  }
}

Image3F KeepOnly2x2Corners(const Image3F& coeffs) {
  Image3F copy = CopyImage(coeffs);
  for (int c = 0; c < 3; ++c) {
    for (size_t y = 0; y < coeffs.ysize(); ++y) {
      float* PIK_RESTRICT row = copy.PlaneRow(c, y);
      for (size_t x = 0; x < coeffs.xsize(); x += 64) {
        for (int k = 0; k < 64; ++k) {
          if (k >= 16 || (k % 8) >= 2) row[x + k] = 0.0f;
        }
      }
    }
  }
  return copy;
}

Image3F GetPixelSpaceImageFrom0189_64(const Image3F& coeffs) {
  PIK_ASSERT(coeffs.xsize() % 64 == 0);
  const size_t block_xsize = coeffs.xsize() / 64;
  const size_t block_ysize = coeffs.ysize();
  Image3F out(block_xsize * 2, block_ysize * 2);
  const float kScale01 = 0.113265930794111f / (kIDCTScales[0] * kIDCTScales[1]);
  const float kScale11 = 0.102633368629251f / (kIDCTScales[1] * kIDCTScales[1]);
  for (int c = 0; c < 3; ++c) {
    for (size_t by = 0; by < block_ysize; ++by) {
      const float* PIK_RESTRICT row_coeffs = coeffs.PlaneRow(c, by);
      float* PIK_RESTRICT row_out0 = out.PlaneRow(c, 2 * by + 0);
      float* PIK_RESTRICT row_out1 = out.PlaneRow(c, 2 * by + 1);
      for (size_t bx = 0; bx < block_xsize; ++bx) {
        const float* block = row_coeffs + bx * 64;
        const float a00 = block[0];
        const float a01 = block[8] * kScale01;
        const float a10 = block[1] * kScale01;
        const float a11 = block[9] * kScale11;
        row_out0[2 * bx + 0] = a00 + a01 + a10 + a11;
        row_out0[2 * bx + 1] = a00 - a01 + a10 - a11;
        row_out1[2 * bx + 0] = a00 + a01 - a10 - a11;
        row_out1[2 * bx + 1] = a00 - a01 - a10 + a11;
      }
    }
  }
  return out;
}

void Add2x2CornersFromPixelSpaceImage(const Image3F& img,
                                      Image3F* coeffs) {
  PIK_ASSERT(coeffs->xsize() % 64 == 0);
  PIK_ASSERT(coeffs->xsize() / 32 <= img.xsize());
  PIK_ASSERT(coeffs->ysize() * 2 <= img.ysize());
  const size_t block_xsize = coeffs->xsize() / 64;
  const size_t block_ysize = coeffs->ysize();
  const float kScale01 = 0.113265930794111f / (kIDCTScales[0] * kIDCTScales[1]);
  const float kScale11 = 0.102633368629251f / (kIDCTScales[1] * kIDCTScales[1]);
  for (int c = 0; c < 3; ++c) {
    for (size_t by = 0; by < block_ysize; ++by) {
      const float* PIK_RESTRICT row0 = img.PlaneRow(c, 2 * by + 0);
      const float* PIK_RESTRICT row1 = img.PlaneRow(c, 2 * by + 1);
      float* row_out = coeffs->PlaneRow(c, by);
      for (size_t bx = 0; bx < block_xsize; ++bx) {
        const float b00 = row0[2 * bx + 0];
        const float b01 = row0[2 * bx + 1];
        const float b10 = row1[2 * bx + 0];
        const float b11 = row1[2 * bx + 1];
        const float a00 = 0.25f * (b00 + b01 + b10 + b11);
        const float a01 = 0.25f * (b00 - b01 + b10 - b11);
        const float a10 = 0.25f * (b00 + b01 - b10 - b11);
        const float a11 = 0.25f * (b00 - b01 - b10 + b11);
        float* PIK_RESTRICT block = &row_out[bx * 64];
        block[0] = a00;
        block[1] = a10 / kScale01;
        block[8] = a01 / kScale01;
        block[9] = a11 / kScale11;
      }
    }
  }
}

namespace {

// "Adds" (if sign == +0, otherwise subtracts if sign == -0) block to "add_to",
// except elements 0,1,8,9. May overwrite parts of "block".
void AddBlockExcept0189To(float* PIK_RESTRICT block, const float sign,
                          float* PIK_RESTRICT add_to) {
  using namespace SIMD_NAMESPACE;

  const Part<float, SIMD_MIN(Full<float>::N, 8)> d;

#if SIMD_TARGET_VALUE == SIMD_NONE
  // Fallback because SIMD version assumes at least two lanes.
    block[0] = 0.0f;
    block[1] = 0.0f;
    block[8] = 0.0f;
    block[9] = 0.0f;
  if (ext::movemask(set1(d, sign))) {
    for (size_t i = 0; i < 64; ++i) {
      add_to[i] -= block[i];
    }
  } else {
    for (size_t i = 0; i < 64; ++i) {
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
  // Handle remnants of DCT row (for 128-bit SIMD)
  for (size_t ix = d.N; ix < 8; ix += d.N) {
    prev = load(d, add_to + ix);
    coefs = load(d, block + ix);
    sum = prev + (coefs ^ vsign);
    store(sum, d, add_to + ix);
  }

  // Second block row: don't add block[8, 9].
  prev = load(d, add_to + 8);
  coefs = load(d, block + 8);
  sum = prev + (andnot(only_01, coefs) ^ vsign);
  store(sum, d, add_to + 8);
  // Handle remnants of DCT row (for 128-bit SIMD)
  for (size_t ix = d.N; ix < 8; ix += d.N) {
    prev = load(d, add_to + 8 + ix);
    coefs = load(d, block + 8 + ix);
    sum = prev + (coefs ^ vsign);
    store(sum, d, add_to + 8 + ix);
  }

  for (size_t i = 16; i < 64; i += d.N) {
    prev = load(d, add_to + i);
    coefs = load(d, block + i);
    sum = prev + (coefs ^ vsign);
    store(sum, d, add_to + i);
  }
#endif
}

}  // namespace

void UpSample4x4BlurDCT(const Image3F& img, const float sigma, const float sign,
                        ThreadPool* pool, Image3F* add_to) {
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

  using namespace SIMD_NAMESPACE;
  using D = Part<float, SIMD_MIN(Full<float>::N, 8)>;
  using V = D::V;
  const D d;
  V vw0[4] = {set1(d, w0[0]), set1(d, w0[1]), set1(d, w0[2]), set1(d, w0[3])};
  V vw1[4] = {set1(d, w1[0]), set1(d, w1[1]), set1(d, w1[2]), set1(d, w1[3])};
  V vw2[4] = {set1(d, w2[0]), set1(d, w2[1]), set1(d, w2[2]), set1(d, w2[3])};

  Image3F blur_x(xs * 4, ys);
  std::vector<float> row_tmp(xs + 2);
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < ys; ++y) {
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
  }

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
                  ComputeTransposedScaledBlockDCTFloat(block);
                  AddBlockExcept0189To(block, sign, row_out + 64 * bx);
                }
              }
            });
}

template <int N>
Image3F UpSampleBlur(const Image3F& img, const float sigma) {
  const int xs = img.xsize();  // -1 => signed
  const int ys = img.ysize();
  float w0[N] = { 0.0f };
  float w1[N] = { 0.0f };
  float w2[N] = { 0.0f };
  std::vector<float> kernel = GaussianKernel(N, sigma);
  float weight = 0.0f;
  for (int i = 0; i < kernel.size(); ++i) {
    weight += kernel[i];
  }
  float scale = 1.0f / weight;
  for (int k = 0; k < N; ++k) {
    const int split0 = N - k;
    const int split1 = 2 * N - k;
    for (int j = 0; j < split0; ++j) {
      w0[k] += kernel[j];
    }
    for (int j = split0; j < split1; ++j) {
      w1[k] += kernel[j];
    }
    for (int j = split1; j < kernel.size(); ++j) {
      w2[k] += kernel[j];
    }
    w0[k] *= scale;
    w1[k] *= scale;
    w2[k] *= scale;
  }
  Image3F blur_x(xs * N, ys);
  std::vector<float> row_tmp(xs + 2);
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < ys; ++y) {
      const float* PIK_RESTRICT row = img.ConstPlaneRow(c, y);
      memcpy(&row_tmp[1], row, xs * sizeof(row[0]));
      row_tmp[0] = row_tmp[1 + std::min(1, xs - 1)];
      row_tmp[xs + 1] = row_tmp[1 + std::max(0, xs - 2)];
      float* PIK_RESTRICT row_out = blur_x.PlaneRow(c, y);
      for (int x = 0; x < xs; ++x) {
        const float v0 = row_tmp[x];
        const float v1 = row_tmp[x + 1];
        const float v2 = row_tmp[x + 2];
        const int offset = x * N;
        for (int ix = 0; ix < N; ++ix) {
          row_out[offset + ix] = v0 * w0[ix] + v1 * w1[ix] + v2 * w2[ix];
        }
      }
    }
  }
  Image3F out(xs * N, ys * N);
  for (int c = 0; c < 3; ++c) {
    for (int by = 0; by < ys; ++by) {
      int by_u = ys == 1 ? 0 : by == 0 ? 1 : by - 1;
      int by_d = ys == 1 ? 0 : by + 1 < ys ? by + 1 : by - 1;
      const float* PIK_RESTRICT row0 = blur_x.ConstPlaneRow(c, by_u);
      const float* PIK_RESTRICT row1 = blur_x.ConstPlaneRow(c, by);
      const float* PIK_RESTRICT row2 = blur_x.ConstPlaneRow(c, by_d);
      for (int bx = 0; bx < xs; ++bx) {
        using namespace SIMD_NAMESPACE;
        constexpr int kLanes =
            SIMD_MIN(N, SIMD_TARGET::template NumLanes<float>());
        const Part<float, kLanes> d;
        for (int ix = 0; ix < N; ix += d.N) {
          const auto val0 = load(d, &row0[bx * N + ix]);
          const auto val1 = load(d, &row1[bx * N + ix]);
          const auto val2 = load(d, &row2[bx * N + ix]);
          for (int iy = 0; iy < N; ++iy) {
            const auto val = (val0 * set1(d, w0[iy]) + val1 * set1(d, w1[iy]) +
                              val2 * set1(d, w2[iy]));
            store(val, d, &out.PlaneRow(c, by * N + iy)[bx * N + ix]);
          }
        }
      }
    }
  }
  return out;
}

Image3F UpSample4x4Blur(const Image3F& img, const float sigma) {
  return UpSampleBlur<4>(img, sigma);
}

ImageF Subsample(const ImageF& image, int f) {
  PROFILER_FUNC;
  PIK_CHECK(image.xsize() % f == 0);
  PIK_CHECK(image.ysize() % f == 0);
  const int shift = CeilLog2Nonzero(static_cast<uint32_t>(f));
  PIK_CHECK(f == (1 << shift));
  const size_t nxs = image.xsize() >> shift;
  const size_t nys = image.ysize() >> shift;
  ImageF retval(nxs, nys);
  FillImage(0.0f, &retval);
  const float mul = 1.0f / (f * f);
  for (size_t y = 0; y < image.ysize(); ++y) {
    const float* PIK_RESTRICT row_in = image.Row(y);
    const size_t ny = y >> shift;
    float* PIK_RESTRICT row_out = retval.Row(ny);
    for (size_t x = 0; x < image.xsize(); ++x) {
      size_t nx = x >> shift;
      row_out[nx] += mul * row_in[x];
    }
  }
  return retval;
}

// Averages 8x8 blocks to match DCT behavior.
ImageF Subsample8(const ImageF& image) {
  PROFILER_FUNC;

  PIK_CHECK(image.xsize() % 8 == 0);
  PIK_CHECK(image.ysize() % 8 == 0);
  const size_t block_xsize = image.xsize() / 8;
  const size_t block_ysize = image.ysize() / 8;

  ImageF out(block_xsize, block_ysize);

  using namespace SIMD_NAMESPACE;
  using D = Part<float, SIMD_MIN(Full<float>::N, 8)>;
  const D d;
  const auto mul = set1(d, 1.0f / 64);

  for (size_t y = 0; y < block_ysize; ++y) {
    float* PIK_RESTRICT row_out = out.Row(y);
    for (size_t x = 0; x < block_xsize; ++x) {
      // Produce a single output pixel by averaging an 8x8 input block.
      auto sum = setzero(d);
      for (size_t iy = 0; iy < 8; ++iy) {
        for (size_t ix = 0; ix < 8; ix += d.N) {
          sum += load(d, image.Row(8 * y + iy) + 8 * x + ix);
        }
      }
      sum = ext::sum_of_lanes(sum);
      row_out[x] = get_part(Part<float, 1>(), sum * mul);
    }
  }
  return out;
}

Image3F Subsample(const Image3F& in, int f) {
  return Image3F(Subsample(in.Plane(0), f),
                 Subsample(in.Plane(1), f),
                 Subsample(in.Plane(2), f));
}

ImageF Upsample(const ImageF& image, int f) {
  int nxs = image.xsize() * f;
  int nys = image.ysize() * f;
  ImageF retval(nxs, nys);
  for (int ny = 0; ny < nys; ++ny) {
    int y = ny / f;
    for (int nx = 0; nx < nxs; ++nx) {
      int x = nx / f;
      retval.Row(ny)[nx] = image.Row(y)[x];
    }
  }
  return retval;
}

Image3F Upsample(const Image3F& in, int f) {
  return Image3F(Upsample(in.Plane(0), f),
                 Upsample(in.Plane(1), f),
                 Upsample(in.Plane(2), f));
}

ImageF Dilate(const ImageF& in) {
  ImageF out(in.xsize(), in.ysize());
  for (int y = 0; y < in.ysize(); ++y) {
    const int ymin = std::max(y - 1, 0);
    const int ymax = std::min<int>(y + 1, in.ysize() - 1);
    for (int x = 0; x < in.xsize(); ++x) {
      const int xmin = std::max(x - 1, 0);
      const int xmax = std::min<int>(x + 1, in.xsize() - 1);
      float maxval = 0.0f;
      for (int yy = ymin; yy <= ymax; ++yy) {
        for (int xx = xmin; xx <= xmax; ++xx) {
          maxval = std::max(maxval, in.Row(yy)[xx]);
        }
      }
      out.Row(y)[x] = maxval;
    }
  }
  return out;
}

Image3F Dilate(const Image3F& in) {
  return Image3F(Dilate(in.Plane(0)), Dilate(in.Plane(1)), Dilate(in.Plane(2)));
}

ImageF Erode(const ImageF& in) {
  ImageF out(in.xsize(), in.ysize());
  for (int y = 0; y < in.ysize(); ++y) {
    const int ymin = std::max(y - 1, 0);
    const int ymax = std::min<int>(y + 1, in.ysize() - 1);
    for (int x = 0; x < in.xsize(); ++x) {
      const int xmin = std::max(x - 1, 0);
      const int xmax = std::min<int>(x + 1, in.xsize() - 1);
      float minval = in.Row(y)[x];
      for (int yy = ymin; yy <= ymax; ++yy) {
        for (int xx = xmin; xx <= xmax; ++xx) {
          minval = std::min(minval, in.Row(yy)[xx]);
        }
      }
      out.Row(y)[x] = minval;
    }
  }
  return out;
}

Image3F Erode(const Image3F& in) {
  return Image3F(Erode(in.Plane(0)), Erode(in.Plane(1)), Erode(in.Plane(2)));
}

ImageF Min(const ImageF& a, const ImageF& b) {
  ImageF out(a.xsize(), a.ysize());
  for (int y = 0; y < a.ysize(); ++y) {
    for (int x = 0; x < a.xsize(); ++x) {
      out.Row(y)[x] = std::min(a.Row(y)[x], b.Row(y)[x]);
    }
  }
  return out;
}

ImageF Max(const ImageF& a, const ImageF& b) {
  ImageF out(a.xsize(), a.ysize());
  for (int y = 0; y < a.ysize(); ++y) {
    for (int x = 0; x < a.xsize(); ++x) {
      out.Row(y)[x] = std::max(a.Row(y)[x], b.Row(y)[x]);
    }
  }
  return out;
}

Image3F Min(const Image3F& a, const Image3F& b) {
  return Image3F(Min(a.Plane(0), b.Plane(0)),
                 Min(a.Plane(1), b.Plane(1)),
                 Min(a.Plane(2), b.Plane(2)));
}

Image3F Max(const Image3F& a, const Image3F& b) {
  return Image3F(Max(a.Plane(0), b.Plane(0)),
                 Max(a.Plane(1), b.Plane(1)),
                 Max(a.Plane(2), b.Plane(2)));
}

}  // namespace pik
