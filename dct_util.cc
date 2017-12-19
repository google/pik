#include "dct_util.h"

#include "dct.h"
#include "gauss_blur.h"
#include "simd/simd.h"
#include "status.h"

namespace pik {

Image3F TransposedScaledIDCT(const Image3F& coeffs) {
  PIK_ASSERT(coeffs.xsize() % 64 == 0);
  Image3F img(coeffs.xsize() / 8, coeffs.ysize() * 8);
  alignas(32) float block[64];
  for (int y = 0; y < coeffs.ysize(); ++y) {
    const int yoff = y * 8;
    auto row_in = coeffs.Row(y);
    for (int x = 0; x < coeffs.xsize(); x += 64) {
      const int xoff = x / 8;
      for (int c = 0; c < 3; ++c) {
        memcpy(block, &row_in[c][x], sizeof(block));
        ComputeTransposedScaledBlockIDCTFloat(block);
        for (int iy = 0; iy < 8; ++iy) {
          float* const PIK_RESTRICT row_out = &img.Row(yoff + iy)[c][xoff];
          memcpy(row_out, &block[iy * 8], 8 * sizeof(block[0]));
        }
      }
    }
  }
  return img;
}

Image3F TransposedScaledDCT(const Image3F& img) {
  PIK_ASSERT(img.xsize() % 8 == 0);
  PIK_ASSERT(img.ysize() % 8 == 0);
  Image3F coeffs(img.xsize() * 8, img.ysize() / 8);
  alignas(32) float block[64];
  for (int y = 0; y < coeffs.ysize(); ++y) {
    const int yoff = y * 8;
    auto row_out = coeffs.Row(y);
    for (int x = 0; x < coeffs.xsize(); x += 64) {
      const int xoff = x / 8;
      for (int c = 0; c < 3; ++c) {
        for (int iy = 0; iy < 8; ++iy) {
          const float* const PIK_RESTRICT row_in = &img.Row(yoff + iy)[c][xoff];
          memcpy(&block[iy * 8], row_in, 8 * sizeof(block[0]));
        }
        ComputeTransposedScaledBlockDCTFloat(block);
        const float kScale = 1.0f / 64.0f;
        for (int k = 0; k < 64; ++k) block[k] *= kScale;
        memcpy(&row_out[c][x], block, sizeof(block));
      }
    }
  }
  return coeffs;
}

Image3F DCImage(const Image3F& coeffs) {
  PIK_ASSERT(coeffs.xsize() % 64 == 0);
  Image3F out(coeffs.xsize() / 64, coeffs.ysize());
  for (int y = 0; y < out.ysize(); ++y) {
    for (int x = 0; x < out.xsize(); ++x) {
      for (int c = 0; c < 3; ++c) {
        out.Row(y)[c][x] = coeffs.Row(y)[c][x * 64];
      }
    }
  }
  return out;
}

void ZeroOut2x2(Image3F* coeffs) {
  PIK_ASSERT(coeffs->xsize() % 64 == 0);
  for (int y = 0; y < coeffs->ysize(); ++y) {
    auto row = coeffs->Row(y);
    for (int x = 0; x < coeffs->xsize(); x += 64) {
      for (int c = 0; c < 3; ++c) {
        row[c][x] = row[c][x + 1] = row[c][x + 8] = row[c][x + 9] = 0.0f;
      }
    }
  }
}

Image3F KeepOnly2x2Corners(const Image3F& coeffs) {
  Image3F copy = CopyImage3(coeffs);
  for (int y = 0; y < coeffs.ysize(); ++y) {
    for (int x = 0; x < coeffs.xsize(); x += 64) {
      for (int c = 0; c < 3; ++c) {
        for (int k = 0; k < 64; ++k) {
          if (k >= 16 || (k % 8) >= 2) copy.Row(y)[c][x + k] = 0.0f;
        }
      }
    }
  }
  return copy;
}

Image3F GetPixelSpaceImageFrom2x2Corners(const Image3F& coeffs) {
  PIK_ASSERT(coeffs.xsize() % 64 == 0);
  const int block_xsize = coeffs.xsize() / 64;
  const int block_ysize = coeffs.ysize();
  Image3F out(block_xsize * 2, block_ysize * 2);
  const float kScale01 = 0.113265930794111f / (kIDCTScales[0] * kIDCTScales[1]);
  const float kScale11 = 0.102633368629251f / (kIDCTScales[1] * kIDCTScales[1]);
  for (int by = 0; by < block_ysize; ++by) {
    for (int bx = 0; bx < block_xsize; ++bx) {
      for (int c = 0; c < 3; ++c) {
        const float* block = &coeffs.Row(by)[c][bx * 64];
        const float a00 = block[0];
        const float a01 = block[8] * kScale01;
        const float a10 = block[1] * kScale01;
        const float a11 = block[9] * kScale11;
        out.Row(2 * by + 0)[c][2 * bx + 0] = a00 + a01 + a10 + a11;
        out.Row(2 * by + 0)[c][2 * bx + 1] = a00 - a01 + a10 - a11;
        out.Row(2 * by + 1)[c][2 * bx + 0] = a00 + a01 - a10 - a11;
        out.Row(2 * by + 1)[c][2 * bx + 1] = a00 - a01 - a10 + a11;
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
  const int block_xsize = coeffs->xsize() / 64;
  const int block_ysize = coeffs->ysize();
  const float kScale01 = 0.113265930794111f / (kIDCTScales[0] * kIDCTScales[1]);
  const float kScale11 = 0.102633368629251f / (kIDCTScales[1] * kIDCTScales[1]);
  for (int by = 0; by < block_ysize; ++by) {
    for (int bx = 0; bx < block_xsize; ++bx) {
      for (int c = 0; c < 3; ++c) {
        const float b00 = img.Row(2 * by + 0)[c][2 * bx + 0];
        const float b01 = img.Row(2 * by + 0)[c][2 * bx + 1];
        const float b10 = img.Row(2 * by + 1)[c][2 * bx + 0];
        const float b11 = img.Row(2 * by + 1)[c][2 * bx + 1];
        const float a00 = 0.25f * (b00 + b01 + b10 + b11);
        const float a01 = 0.25f * (b00 - b01 + b10 - b11);
        const float a10 = 0.25f * (b00 + b01 - b10 - b11);
        const float a11 = 0.25f * (b00 - b01 - b10 + b11);
        float* block = &coeffs->Row(by)[c][bx * 64];
        block[0] = a00;
        block[1] = a10 / kScale01;
        block[8] = a01 / kScale01;
        block[9] = a11 / kScale11;
      }
    }
  }
}

Image3F UpSample8x8BlurDCT(const Image3F& img, const float sigma) {
  const int xs = img.xsize();
  const int ys = img.ysize();
  float w0[8] = { 0.0f };
  float w1[8] = { 0.0f };
  float w2[8] = { 0.0f };
  std::vector<float> kernel = GaussianKernel(8, sigma);
  float weight = 0.0f;
  for (int i = 0; i < kernel.size(); ++i) {
    weight += kernel[i];
  }
  float scale = 1.0f / (8.0f * weight);
  for (int k = 0; k < 8; ++k) {
    const int split0 = 8 - k;
    const int split1 = 16 - k;
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
  Image3F blur_x(xs * 8, ys);
  for (int y = 0; y < ys; ++y) {
    auto row = img.Row(y);
    for (int c = 0; c < 3; ++c) {
      std::vector<float> row_tmp(xs + 2);
      memcpy(&row_tmp[1], row[c], xs * sizeof(row[c][0]));
      row_tmp[0] = row_tmp[1 + std::min(1, xs - 1)];
      row_tmp[xs + 1] = row_tmp[1 + std::max(0, xs - 2)];
      float* const PIK_RESTRICT row_out = blur_x.Row(y)[c];
      for (int x = 0; x < xs; ++x) {
        const float v0 = row_tmp[x];
        const float v1 = row_tmp[x + 1];
        const float v2 = row_tmp[x + 2];
        const int offset = x * 8;
        for (int ix = 0; ix < 8; ++ix) {
          row_out[offset + ix] = v0 * w0[ix] + v1 * w1[ix] + v2 * w2[ix];
        }
      }
    }
  }
  Image3F out(xs * 64, ys);
  for (int by = 0; by < ys; ++by) {
    int by_u = ys == 1 ? 0 : by == 0 ? 1 : by - 1;
    int by_d = ys == 1 ? 0 : by + 1 < ys ? by + 1 : by - 1;
    auto row = out.Row(by);
    auto row0 = blur_x.ConstRow(by_u);
    auto row1 = blur_x.ConstRow(by);
    auto row2 = blur_x.ConstRow(by_d);
    for (int bx = 0; bx < xs; ++bx) {
      for (int c = 0; c < 3; ++c) {
        float* const PIK_RESTRICT block = &row[c][bx * 64];
        using namespace SIMD_NAMESPACE;
        const Full<float> d;
        for (int ix = 0; ix < 8; ix += d.N) {
          const auto val0 = load(d, &row0[c][bx * 8 + ix]);
          const auto val1 = load(d, &row1[c][bx * 8 + ix]);
          const auto val2 = load(d, &row2[c][bx * 8 + ix]);
          for (int iy = 0; iy < 8; ++iy) {
            const auto val = (val0 * set1(d, w0[iy]) + val1 * set1(d, w1[iy]) +
                              val2 * set1(d, w2[iy]));
            store(val, d, &block[iy * 8 + ix]);
          }
        }
        ComputeTransposedScaledBlockDCTFloat(block);
        block[0] = 0.0f;
      }
    }
  }
  return out;
}

Image3F UpSample4x4BlurDCT(const Image3F& img, const float sigma) {
  const int xs = img.xsize();
  const int ys = img.ysize();
  const int bxs = xs / 2;
  const int bys = ys / 2;
  float w0[4] = { 0.0f };
  float w1[4] = { 0.0f };
  float w2[4] = { 0.0f };
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
  Image3F blur_x(xs * 4, ys);
  for (int y = 0; y < ys; ++y) {
    auto row = img.Row(y);
    for (int c = 0; c < 3; ++c) {
      std::vector<float> row_tmp(xs + 2);
      memcpy(&row_tmp[1], row[c], xs * sizeof(row[c][0]));
      row_tmp[0] = row_tmp[1 + std::min(1, xs - 1)];
      row_tmp[xs + 1] = row_tmp[1 + std::max(0, xs - 2)];
      float* const PIK_RESTRICT row_out = blur_x.Row(y)[c];
      for (int x = 0; x < xs; ++x) {
        const float v0 = row_tmp[x];
        const float v1 = row_tmp[x + 1];
        const float v2 = row_tmp[x + 2];
        const int offset = x * 4;
        for (int ix = 0; ix < 4; ++ix) {
          row_out[offset + ix] = v0 * w0[ix] + v1 * w1[ix] + v2 * w2[ix];
        }
      }
    }
  }
  Image3F out(bxs * 64, bys);
  for (int by = 0; by < bys; ++by) {
    auto row = out.Row(by);
    const int by0 = by == 0 ? 1 : 2 * by - 1;
    const int by1 = 2 * by;
    const int by2 = 2 * by + 1;
    const int by3 = by + 1 < bys ? 2 * by + 2 : 2 * by;
    auto row0 = blur_x.ConstRow(by0);
    auto row1 = blur_x.ConstRow(by1);
    auto row2 = blur_x.ConstRow(by2);
    auto row3 = blur_x.ConstRow(by3);
    for (int bx = 0; bx < bxs; ++bx) {
      for (int c = 0; c < 3; ++c) {
        float* const PIK_RESTRICT block = &row[c][bx * 64];
        using namespace SIMD_NAMESPACE;
        const Full<float> d;
        for (int ix = 0; ix < 8; ix += d.N) {
          const auto val0 = load(d, &row0[c][bx * 8 + ix]);
          const auto val1 = load(d, &row1[c][bx * 8 + ix]);
          const auto val2 = load(d, &row2[c][bx * 8 + ix]);
          const auto val3 = load(d, &row3[c][bx * 8 + ix]);
          for (int iy = 0; iy < 4; ++iy) {
            const auto vala = (val0 * set1(d, w0[iy]) + val1 * set1(d, w1[iy]) +
                               val2 * set1(d, w2[iy]));
            const auto valb = (val1 * set1(d, w0[iy]) + val2 * set1(d, w1[iy]) +
                               val3 * set1(d, w2[iy]));
            store(vala, d, &block[iy * 8 + ix]);
            store(valb, d, &block[iy * 8 + 32 + ix]);
          }
        }
        ComputeTransposedScaledBlockDCTFloat(block);
        block[0] = 0.0f;
        block[1] = 0.0f;
        block[8] = 0.0f;
        block[9] = 0.0f;
      }
    }
  }
  return out;
}

template <int N>
Image3F UpSampleBlur(const Image3F& img, const float sigma) {
  const int xs = img.xsize();
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
  for (int y = 0; y < ys; ++y) {
    auto row = img.Row(y);
    for (int c = 0; c < 3; ++c) {
      std::vector<float> row_tmp(xs + 2);
      memcpy(&row_tmp[1], row[c], xs * sizeof(row[c][0]));
      row_tmp[0] = row_tmp[1 + std::min(1, xs - 1)];
      row_tmp[xs + 1] = row_tmp[1 + std::max(0, xs - 2)];
      float* const PIK_RESTRICT row_out = blur_x.Row(y)[c];
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
  for (int by = 0; by < ys; ++by) {
    int by_u = ys == 1 ? 0 : by == 0 ? 1 : by - 1;
    int by_d = ys == 1 ? 0 : by + 1 < ys ? by + 1 : by - 1;
    auto row0 = blur_x.ConstRow(by_u);
    auto row1 = blur_x.ConstRow(by);
    auto row2 = blur_x.ConstRow(by_d);
    for (int bx = 0; bx < xs; ++bx) {
      for (int c = 0; c < 3; ++c) {
        using namespace SIMD_NAMESPACE;
        constexpr int kLanes =
            SIMD_MIN(N, SIMD_TARGET::template NumLanes<float>());
        const Part<float, kLanes> d;
        for (int ix = 0; ix < N; ix += d.N) {
          const auto val0 = load(d, &row0[c][bx * N + ix]);
          const auto val1 = load(d, &row1[c][bx * N + ix]);
          const auto val2 = load(d, &row2[c][bx * N + ix]);
          for (int iy = 0; iy < N; ++iy) {
            const auto val = (val0 * set1(d, w0[iy]) + val1 * set1(d, w1[iy]) +
                              val2 * set1(d, w2[iy]));
            store(val, d, &out.Row(by * N + iy)[c][bx * N + ix]);
          }
        }
      }
    }
  }
  return out;
}

Image3F UpSample8x8Blur(const Image3F& img, const float sigma) {
  return UpSampleBlur<8>(img, sigma);
}

Image3F UpSample4x4Blur(const Image3F& img, const float sigma) {
  return UpSampleBlur<4>(img, sigma);
}


ImageF Subsample(const ImageF& image, int f) {
  PIK_ASSERT(image.xsize() % f == 0);
  PIK_ASSERT(image.ysize() % f == 0);
  const int nxs = image.xsize() / f;
  const int nys = image.ysize() / f;
  ImageF retval(nxs, nys, 0.0f);
  float mul = 1.0f / (f * f);
  for (int y = 0; y < image.ysize(); ++y) {
    for (int x = 0; x < image.xsize(); ++x) {
      int ny = y / f;
      int nx = x / f;
      retval.Row(ny)[nx] += mul * image.Row(y)[x];
    }
  }
  return retval;
}

Image3F Subsample(const Image3F& in, int f) {
  return Image3F(Subsample(in.plane(0), f),
                 Subsample(in.plane(1), f),
                 Subsample(in.plane(2), f));
}

ImageF Upsample(const ImageF& image, int f) {
  int nxs = image.xsize() * f;
  int nys = image.ysize() * f;
  ImageF retval(nxs, nys);
  for (int ny = 0; ny < nys; ++ny) {
    int y = ny  / f;
    for (int nx = 0; nx < nxs; ++nx) {
      int x = nx / f;
      for (int c = 0; c < 3; ++c) {
        retval.Row(ny)[nx] = image.Row(y)[x];
      }
    }
  }
  return retval;
}

Image3F Upsample(const Image3F& in, int f) {
  return Image3F(Upsample(in.plane(0), f),
                 Upsample(in.plane(1), f),
                 Upsample(in.plane(2), f));
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
  return Image3F(Dilate(in.plane(0)), Dilate(in.plane(1)), Dilate(in.plane(2)));
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
  return Image3F(Erode(in.plane(0)), Erode(in.plane(1)), Erode(in.plane(2)));
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
  return Image3F(Min(a.plane(0), b.plane(0)),
                 Min(a.plane(1), b.plane(1)),
                 Min(a.plane(2), b.plane(2)));
}

Image3F Max(const Image3F& a, const Image3F& b) {
  return Image3F(Max(a.plane(0), b.plane(0)),
                 Max(a.plane(1), b.plane(1)),
                 Max(a.plane(2), b.plane(2)));
}

}  // namespace pik
