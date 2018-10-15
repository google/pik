#include "upscaler.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cmath>
#include <vector>
#include "compiler_specific.h"
#include "image.h"
#include "resample.h"

namespace pik {

namespace {

std::vector<float> ComputeKernel(float sigma) {
  // Filtering becomes slower, but more Gaussian when m is increased.
  // More Gaussian doesn't mean necessarily better results altogether.
  const float m = 2.5;
  const float scaler = -1.0 / (2 * sigma * sigma);
  const int diff = std::max<int>(1, m * fabs(sigma));
  std::vector<float> kernel(2 * diff + 1);
  for (int i = -diff; i <= diff; ++i) {
    kernel[i + diff] = exp(scaler * i * i);
  }
  return kernel;
}

void ConvolveBorderColumn(const ImageF& in, const std::vector<float>& kernel,
                          const float weight_no_border,
                          const float border_ratio, const size_t x,
                          float* const PIK_RESTRICT row_out) {
  const int offset = kernel.size() / 2;
  int minx = x < offset ? 0 : x - offset;
  int maxx = std::min<int>(in.xsize() - 1, x + offset);
  float weight = 0.0f;
  for (int j = minx; j <= maxx; ++j) {
    weight += kernel[j - x + offset];
  }
  // Interpolate linearly between the no-border scaling and border scaling.
  weight = (1.0f - border_ratio) * weight + border_ratio * weight_no_border;
  float scale = 1.0f / weight;
  for (size_t y = 0; y < in.ysize(); ++y) {
    const float* const PIK_RESTRICT row_in = in.Row(y);
    float sum = 0.0f;
    for (int j = minx; j <= maxx; ++j) {
      sum += row_in[j] * kernel[j - x + offset];
    }
    row_out[y] = sum * scale;
  }
}

// Computes a horizontal convolution and transposes the result.
ImageF Convolution(const ImageF& in, const std::vector<float>& kernel,
                   const float border_ratio) {
  ImageF out(in.ysize(), in.xsize());
  const int len = kernel.size();
  const int offset = kernel.size() / 2;
  float weight_no_border = 0.0f;
  for (int j = 0; j < len; ++j) {
    weight_no_border += kernel[j];
  }
  float scale_no_border = 1.0f / weight_no_border;
  const int border1 = in.xsize() <= offset ? in.xsize() : offset;
  const int border2 = in.xsize() - offset;
  int x = 0;
  // left border
  for (; x < border1; ++x) {
    ConvolveBorderColumn(in, kernel, weight_no_border, border_ratio, x,
                         out.Row(x));
  }
  // middle
  for (; x < border2; ++x) {
    float* const PIK_RESTRICT row_out = out.Row(x);
    for (size_t y = 0; y < in.ysize(); ++y) {
      const float* const PIK_RESTRICT row_in = &in.Row(y)[x - offset];
      float sum = 0.0f;
      for (int j = 0; j < len; ++j) {
        sum += row_in[j] * kernel[j];
      }
      row_out[y] = sum * scale_no_border;
    }
  }
  // right border
  for (; x < in.xsize(); ++x) {
    ConvolveBorderColumn(in, kernel, weight_no_border, border_ratio, x,
                         out.Row(x));
  }
  return out;
}

// A blur somewhat similar to a 2D Gaussian blur.
// See: https://en.wikipedia.org/wiki/Gaussian_blur
ImageF Blur(const ImageF& in, float sigma, float border_ratio) {
  std::vector<float> kernel = ComputeKernel(sigma);
  return Convolution(Convolution(in, kernel, border_ratio), kernel,
                     border_ratio);
}

Image3F SuperSample4x4(const Image3F& image) {
  size_t nxs = image.xsize() << 2;
  size_t nys = image.ysize() << 2;
  Image3F retval(nxs, nys);
  for (int c = 0; c < 3; ++c) {
    for (size_t ny = 0; ny < nys; ++ny) {
      const float* PIK_RESTRICT row_in = image.PlaneRow(c, ny >> 2);
      float* PIK_RESTRICT row_out = retval.PlaneRow(c, ny);

      for (size_t nx = 0; nx < nxs; ++nx) {
        row_out[nx] = row_in[nx >> 2];
      }
    }
  }
  return retval;
}

void Smooth4x4Corners(Image3F& ima) {
  static const float overshoot = 3.5;
  static const float m = 1.0 / (4.0 - overshoot);
  for (int y = 3; y + 3 < ima.ysize(); y += 4) {
    for (int x = 3; x + 3 < ima.xsize(); x += 4) {
      float ave[3] = {0};
      for (int c = 0; c < 3; ++c) {
        ave[c] += ima.PlaneRow(c, y)[x];
        ave[c] += ima.PlaneRow(c, y)[x + 1];
        ave[c] += ima.PlaneRow(c, y + 1)[x];
        ave[c] += ima.PlaneRow(c, y + 1)[x + 1];
      }
      const int off = 2;
      for (int c = 0; c < 3; ++c) {
        float others = (ave[c] - overshoot * ima.PlaneRow(c, y)[x]) * m;
        ima.PlaneRow(c, y - off)[x - off] -= (others - ima.PlaneRow(c, y)[x]);
        ima.PlaneRow(c, y)[x] = others;
      }
      for (int c = 0; c < 3; ++c) {
        float others = (ave[c] - overshoot * ima.PlaneRow(c, y)[x + 1]) * m;
        ima.PlaneRow(c, y - off)[x + off + 1] -=
            (others - ima.PlaneRow(c, y)[x + 1]);
        ima.PlaneRow(c, y)[x + 1] = others;
      }
      for (int c = 0; c < 3; ++c) {
        float others = (ave[c] - overshoot * ima.PlaneRow(c, y + 1)[x]) * m;
        ima.PlaneRow(c, y + off + 1)[x - off] -=
            (others - ima.PlaneRow(c, y + 1)[x]);
        ima.PlaneRow(c, y + 1)[x] = others;
      }
      for (int c = 0; c < 3; ++c) {
        float others = (ave[c] - overshoot * ima.PlaneRow(c, y + 1)[x + 1]) * m;
        ima.PlaneRow(c, y + off + 1)[x + off + 1] -=
            (others - ima.PlaneRow(c, y + 1)[x + 1]);
        ima.PlaneRow(c, y + 1)[x + 1] = others;
      }
    }
  }
}

}  // namespace

SIMD_ATTR Image3F UpscalerReconstruct(const Image3F& in) {
  Image3F out1 = SuperSample4x4(in);
  Smooth4x4Corners(out1);
  out1 = Blur(out1, 2.5);
  Image3F out(out1.xsize() * 2, out1.ysize() * 2);
  Upsample<slow::Upsampler>(ExecutorLoop(), out1, kernel::CatmullRom(), &out);
  return out;
}

Image3F Blur(const Image3F& image, float sigma) {
  float border = 0.0;
  return Image3F(Blur(image.Plane(0), sigma, border),
                 Blur(image.Plane(1), sigma, border),
                 Blur(image.Plane(2), sigma, border));
}

}  // namespace pik
