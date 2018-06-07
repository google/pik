// This file implements a ppm to (dc-ppm, ac-ppm) mapping that allows
// us to experiment in different ways to compose the image into
// (4x4) pseudo-dc and respective ac components.
#include "upscaler.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cmath>
#include <vector>
#include "butteraugli_distance.h"
#include "gamma_correct.h"
#include "image.h"
#include "image_io.h"
#include "resample.h"

#define BUTTERAUGLI_RESTRICT __restrict__

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
                          float* const BUTTERAUGLI_RESTRICT row_out) {
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
    const float* const BUTTERAUGLI_RESTRICT row_in = in.Row(y);
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
    float* const BUTTERAUGLI_RESTRICT row_out = out.Row(x);
    for (size_t y = 0; y < in.ysize(); ++y) {
      const float* const BUTTERAUGLI_RESTRICT row_in = &in.Row(y)[x - offset];
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

Image3F Blur(const Image3F& image, float sigma) {
  float border = 0.0;
  return Image3F(Blur(image.plane(0), sigma, border),
                 Blur(image.plane(1), sigma, border),
                 Blur(image.plane(2), sigma, border));
}

// DoGBlur is an approximate of difference of Gaussians. We use it to
// approximate LoG (Laplacian of Gaussians).
// See: https://en.wikipedia.org/wiki/Difference_of_Gaussians
// For motivation see:
// https://en.wikipedia.org/wiki/Pyramid_(image_processing)#Laplacian_pyramid
ImageF DoGBlur(const ImageF& in, float sigma, float border_ratio) {
  ImageF blur1 = Blur(in, sigma, border_ratio);
  ImageF blur2 = Blur(in, sigma * 2.0f, border_ratio);
  static const float mix = 0.25;
  ImageF out(in.xsize(), in.ysize());
  for (size_t y = 0; y < in.ysize(); ++y) {
    const float* const BUTTERAUGLI_RESTRICT row1 = blur1.Row(y);
    const float* const BUTTERAUGLI_RESTRICT row2 = blur2.Row(y);
    float* const BUTTERAUGLI_RESTRICT row_out = out.Row(y);
    for (size_t x = 0; x < in.xsize(); ++x) {
      row_out[x] = (1.0f + mix) * row1[x] - mix * row2[x];
    }
  }
  return out;
}

Image3F DoGBlur(const Image3F& image, float sigma) {
  float border = 0.0;
  return Image3F(DoGBlur(image.plane(0), sigma, border),
                 DoGBlur(image.plane(1), sigma, border),
                 DoGBlur(image.plane(2), sigma, border));
}

void SelectiveBlur(Image3F& image, float sigma, float select) {
  Image3F copy = Blur(image, sigma);
  float select2 = select * 2;
  float ramp = 0.8f;
  float onePerSelect = ramp / select;
  float onePerSelect2 = ramp / select2;
  for (int y = 0; y < image.ysize(); ++y) {
    for (int x = 0; x < image.xsize(); ++x) {
      for (int c = 0; c < 3; ++c) {
        float dist = fabs(copy.Row(y)[c][x] - image.Row(y)[c][x]);
        float w = 0.0f;
        if ((x & 7) == 0 || (x & 7) == 7 || (y & 7) == 0 || (y & 7) == 7) {
          if (dist < select2) {
            w = ramp - dist * onePerSelect2;
            if (w > 1.0f) w = 1.0f;
          }
        } else if (dist < select) {
          w = ramp - dist * onePerSelect;
          if (w > 1.0f) w = 1.0f;
        }
        image.Row(y)[c][x] =
            w * copy.Row(y)[c][x] + (1.0 - w) * image.Row(y)[c][x];
      }
    }
  }
}

void SelectiveBlur8x8(Image3F& image, Image3F& ac, float sigma,
                      float select_mod) {
  Image3F copy = Blur(image, sigma);
  float ramp = 1.0f;
  for (int wy = 0; wy < image.ysize(); wy += 8) {
    for (int wx = 0; wx < image.xsize(); wx += 8) {
      for (int c = 0; c < 3; ++c) {
        // Find maxdiff
        double max = 0;
        for (int dy = 0; dy < 6; ++dy) {
          for (int dx = 0; dx < 6; ++dx) {
            int y = wy + dy;
            int x = wx + dx;
            if (y >= image.ysize() || x >= image.xsize()) {
              break;
            }
            // Look at the criss-cross of diffs between two pixels.
            // Scale the smoothing within the block of the amplitude
            // of such local change.
            float dist = fabs(ac.Row(y)[c][x] - ac.Row(y)[c][x + 2]);
            if (max < dist) max = dist;
            dist = fabs(ac.Row(y)[c][x] - ac.Row(y + 2)[c][x]);
            if (max < dist) max = dist;
            dist = fabs(ac.Row(y)[c][x] - ac.Row(y + 2)[c][x + 2]);
            if (max < dist) max = dist;
            dist = fabs(ac.Row(y)[c][x + 2] - ac.Row(y + 2)[c][x]);
            if (max < dist) max = dist;
          }
        }

        float select = select_mod * max;
        float select2 = 2.0 * select;
        float onePerSelect = ramp / select;
        float onePerSelect2 = ramp / select2;

        for (int dy = 0; dy < 8; ++dy) {
          for (int dx = 0; dx < 8; ++dx) {
            int y = wy + dy;
            int x = wx + dx;
            if (y >= image.ysize() || x >= image.xsize()) {
              break;
            }
            float dist = fabs(copy.Row(y)[c][x] - image.Row(y)[c][x]);
            float w = 0.0f;
            if ((x & 7) == 0 || (x & 7) == 7 || (y & 7) == 0 || (y & 7) == 7) {
              if (dist < select2) {
                w = ramp - dist * onePerSelect2;
                if (w > 1.0f) w = 1.0f;
              }
            } else if (dist < select) {
              w = ramp - dist * onePerSelect;
              if (w > 1.0f) w = 1.0f;
            }
            image.Row(y)[c][x] =
                w * copy.Row(y)[c][x] + (1.0 - w) * image.Row(y)[c][x];
          }
        }
      }
    }
  }
}
Image3F SubSampleSimple8x8(const Image3F& image) {
  const int nxs = (image.xsize() + 7) >> 3;
  const int nys = (image.ysize() + 7) >> 3;
  Image3F retval(nxs, nys, 0.0f);
  float mul = 1 / 64.0;
  for (int y = 0; y < image.ysize(); ++y) {
    for (int x = 0; x < image.xsize(); ++x) {
      for (int c = 0; c < 3; ++c) {
        int ny = y >> 3;
        int nx = x >> 3;
        retval.Row(ny)[c][nx] += mul * image.Row(y)[c][x];
      }
    }
  }
  if ((image.xsize() & 7) != 0) {
    const float last_column_mul = 8.0 / (image.xsize() & 7);
    for (int y = 0; y < nys; ++y) {
      for (int c = 0; c < 3; ++c) {
        retval.Row(y)[c][nxs - 1] *= last_column_mul;
      }
    }
  }
  if ((image.ysize() & 7) != 0) {
    const float last_row_mul = 8.0 / (image.ysize() & 7);
    for (int x = 0; x < nxs; ++x) {
      for (int c = 0; c < 3; ++c) {
        retval.Row(nys - 1)[c][x] *= last_row_mul;
      }
    }
  }
  return retval;
}

Image3F SubSampleSimple4x4(const Image3F& image) {
  const int nxs = (image.xsize() + 3) >> 2;
  const int nys = (image.ysize() + 3) >> 2;
  Image3F retval(nxs, nys, 0.0f);
  float mul = 1 / 16.0;
  for (int y = 0; y < image.ysize(); ++y) {
    for (int x = 0; x < image.xsize(); ++x) {
      for (int c = 0; c < 3; ++c) {
        int ny = y >> 2;
        int nx = x >> 2;
        retval.Row(ny)[c][nx] += mul * image.Row(y)[c][x];
      }
    }
  }
  if ((image.xsize() & 3) != 0) {
    const float last_column_mul = 4.0 / (image.xsize() & 3);
    for (int y = 0; y < nys; ++y) {
      for (int c = 0; c < 3; ++c) {
        retval.Row(y)[c][nxs - 1] *= last_column_mul;
      }
    }
  }
  if ((image.ysize() & 3) != 0) {
    const float last_row_mul = 4.0 / (image.ysize() & 3);
    for (int x = 0; x < nxs; ++x) {
      for (int c = 0; c < 3; ++c) {
        retval.Row(nys - 1)[c][x] *= last_row_mul;
      }
    }
  }
  return retval;
}

Image3F SuperSample2x2(const Image3F& image) {
  int nxs = image.xsize() << 1;
  int nys = image.ysize() << 1;
  Image3F retval(nxs, nys);
  for (int ny = 0; ny < nys; ++ny) {
    int y = ny >> 1;
    for (int nx = 0; nx < nxs; ++nx) {
      int x = nx >> 1;
      for (int c = 0; c < 3; ++c) {
        retval.Row(ny)[c][nx] = image.Row(y)[c][x];
      }
    }
  }
  return retval;
}

Image3F SuperSample4x4(const Image3F& image) {
  int nxs = image.xsize() << 2;
  int nys = image.ysize() << 2;
  Image3F retval(nxs, nys);
  for (int ny = 0; ny < nys; ++ny) {
    int y = ny >> 2;
    for (int nx = 0; nx < nxs; ++nx) {
      int x = nx >> 2;
      for (int c = 0; c < 3; ++c) {
        retval.Row(ny)[c][nx] = image.Row(y)[c][x];
      }
    }
  }
  return retval;
}
Image3F SuperSample8x8(const Image3F& image) {
  int nxs = image.xsize() << 3;
  int nys = image.ysize() << 3;
  Image3F retval(nxs, nys);
  for (int ny = 0; ny < nys; ++ny) {
    int y = ny >> 3;
    for (int nx = 0; nx < nxs; ++nx) {
      int x = nx >> 3;
      for (int c = 0; c < 3; ++c) {
        retval.Row(ny)[c][nx] = image.Row(y)[c][x];
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
        ave[c] += ima.Row(y)[c][x];
        ave[c] += ima.Row(y)[c][x + 1];
        ave[c] += ima.Row(y + 1)[c][x];
        ave[c] += ima.Row(y + 1)[c][x + 1];
      }
      const int off = 2;
      for (int c = 0; c < 3; ++c) {
        float others = (ave[c] - overshoot * ima.Row(y)[c][x]) * m;
        ima.Row(y - off)[c][x - off] -= (others - ima.Row(y)[c][x]);
        ima.Row(y)[c][x] = others;
      }
      for (int c = 0; c < 3; ++c) {
        float others = (ave[c] - overshoot * ima.Row(y)[c][x + 1]) * m;
        ima.Row(y - off)[c][x + off + 1] -= (others - ima.Row(y)[c][x + 1]);
        ima.Row(y)[c][x + 1] = others;
      }
      for (int c = 0; c < 3; ++c) {
        float others = (ave[c] - overshoot * ima.Row(y + 1)[c][x]) * m;
        ima.Row(y + off + 1)[c][x - off] -= (others - ima.Row(y + 1)[c][x]);
        ima.Row(y + 1)[c][x] = others;
      }
      for (int c = 0; c < 3; ++c) {
        float others = (ave[c] - overshoot * ima.Row(y + 1)[c][x + 1]) * m;
        ima.Row(y + off + 1)[c][x + off + 1] -=
            (others - ima.Row(y + 1)[c][x + 1]);
        ima.Row(y + 1)[c][x + 1] = others;
      }
    }
  }
}

void Subtract(Image3F& a, const Image3F& b) {
  for (int y = 0; y < a.ysize(); ++y) {
    for (int x = 0; x < a.xsize(); ++x) {
      for (int c = 0; c < 3; ++c) {
        a.Row(y)[c][x] -= b.Row(y)[c][x];
      }
    }
  }
}

void Add(Image3F& a, const Image3F& b) {
  for (int y = 0; y < a.ysize(); ++y) {
    for (int x = 0; x < a.xsize(); ++x) {
      for (int c = 0; c < 3; ++c) {
        a.Row(y)[c][x] += b.Row(y)[c][x];
      }
    }
  }
}

Image3F Crop(const Image3F& image, int newxsize, int newysize) {
  Image3F retval(newxsize, newysize);
  for (int y = 0; y < newysize; ++y) {
    for (int x = 0; x < newxsize; ++x) {
      for (int c = 0; c < 3; ++c) {
        double v = image.Row(y)[c][x];
        if (v < 0) {
          v = 0;
        }
        if (v > 255) {
          v = 255;
        }
        retval.Row(y)[c][x] = v;
      }
    }
  }
  return retval;
}

Image3F ToLinear(const Image3F& image) {
  Image3F out(image.xsize(), image.ysize());
  for (int y = 0; y < image.ysize(); ++y) {
    for (int x = 0; x < image.xsize(); ++x) {
      for (int c = 0; c < 3; ++c) {
        float val = image.Row(y)[c][x];
        out.Row(y)[c][x] = Srgb8ToLinearDirect(val);
      }
    }
  }
  return out;
}

Image3F EncodePseudoDC(const Image3F& in) {
  Image3F goal = CopyImage(in);
  Image3F image8x8sub;
  static const int kIters = 2;
  for (int ii = 0; ii < kIters; ++ii) {
    if (ii != 0) {
      Image3F normal = UpscalerReconstruct(image8x8sub);
      // adjust the image by diff of normal and image.
      for (int y = 0; y < in.ysize(); ++y) {
        for (int x = 0; x < in.xsize(); ++x) {
          for (int c = 0; c < 3; ++c) {
            goal.Row(y)[c][x] -= 0.1 * (normal.Row(y)[c][x] - in.Row(y)[c][x]);
          }
        }
      }
    }
    image8x8sub = SubSampleSimple8x8(goal);
  }

  // Encode pseudo dc.
  return image8x8sub;
}

}  // namespace

Image3F UpscalerReconstruct(const Image3F& in) {
  Image3F out1 = SuperSample4x4(in);
  Smooth4x4Corners(out1);
  out1 = Blur(out1, 2.5);
  Image3F out(out1.xsize() * 2, out1.ysize() * 2);
  Upsample<slow::Upsampler>(ExecutorLoop(), out1, kernel::CatmullRom(), &out);
  return out;
}

}  // namespace pik
