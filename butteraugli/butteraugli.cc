// Copyright 2017 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "butteraugli/butteraugli.h"

// Author: Jyrki Alakuijala (jyrki.alakuijala@gmail.com)
//
// The physical architecture of butteraugli is based on the following naming
// convention:
//   * Opsin - dynamics of the photosensitive chemicals in the retina
//             with their immediate electrical processing
//   * Xyb - hybrid opponent/trichromatic color space
//     x is roughly red-subtract-green.
//     y is yellow.
//     b is blue.
//     Xyb values are computed from Opsin mixing, not directly from rgb.
//   * Mask - for visual masking
//   * Hf - color modeling for spatially high-frequency features
//   * Lf - color modeling for spatially low-frequency features
//   * Diffmap - to cluster and build an image of error between the images
//   * Blur - to hold the smoothing code

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <atomic>

#include <algorithm>
#include <array>

#ifndef PROFILER_ENABLED
#define PROFILER_ENABLED 0
#endif
#if PROFILER_ENABLED
#else
#define PROFILER_FUNC
#define PROFILER_ZONE(name)
#endif

namespace pik {
namespace butteraugli {

void *CacheAligned::Allocate(const size_t bytes) {
  char *const allocated = static_cast<char *>(malloc(bytes + kCacheLineSize));
  if (allocated == nullptr) {
    return nullptr;
  }
  const uintptr_t misalignment =
      reinterpret_cast<uintptr_t>(allocated) & (kCacheLineSize - 1);
  // malloc is at least kPointerSize aligned, so we can store the "allocated"
  // pointer immediately before the aligned memory.
  assert(misalignment % kPointerSize == 0);
  char *const aligned = allocated + kCacheLineSize - misalignment;
  memcpy(aligned - kPointerSize, &allocated, kPointerSize);
  return BUTTERAUGLI_ASSUME_ALIGNED(aligned, 64);
}

void CacheAligned::Free(void *aligned_pointer) {
  if (aligned_pointer == nullptr) {
    return;
  }
  char *const aligned = static_cast<char *>(aligned_pointer);
  assert(reinterpret_cast<uintptr_t>(aligned) % kCacheLineSize == 0);
  char *allocated;
  memcpy(&allocated, aligned - kPointerSize, kPointerSize);
  assert(allocated <= aligned - kPointerSize);
  assert(allocated >= aligned - kCacheLineSize);
  free(allocated);
}

static inline bool IsNan(const float x) {
  uint32_t bits;
  memcpy(&bits, &x, sizeof(bits));
  const uint32_t bitmask_exp = 0x7F800000;
  return (bits & bitmask_exp) == bitmask_exp && (bits & 0x7FFFFF);
}

static inline bool IsNan(const double x) {
  uint64_t bits;
  memcpy(&bits, &x, sizeof(bits));
  return (0x7ff0000000000001ULL <= bits && bits <= 0x7fffffffffffffffULL) ||
         (0xfff0000000000001ULL <= bits && bits <= 0xffffffffffffffffULL);
}

static inline void CheckImage(const ImageF &image, const char *name) {
  for (size_t y = 0; y < image.ysize(); ++y) {
    const float * const BUTTERAUGLI_RESTRICT row = image.Row(y);
    for (size_t x = 0; x < image.xsize(); ++x) {
      if (IsNan(row[x])) {
        printf("Image %s @ %lu,%lu (of %lu,%lu)\n", name, x, y, image.xsize(),
               image.ysize());
        exit(1);
      }
    }
  }
}

#if BUTTERAUGLI_ENABLE_CHECKS

#define CHECK_NAN(x, str)                \
  do {                                   \
    if (IsNan(x)) {                      \
      printf("%d: %s\n", __LINE__, str); \
      abort();                           \
    }                                    \
  } while (0)

#define CHECK_IMAGE(image, name) CheckImage(image, name)

#else

#define CHECK_NAN(x, str)
#define CHECK_IMAGE(image, name)

#endif


// Purpose of kInternalGoodQualityThreshold:
// Normalize 'ok' image degradation to 1.0 across different versions of
// butteraugli.
static const double kInternalGoodQualityThreshold = 5.374425491762814;
static const double kGlobalScale = 1.0 / kInternalGoodQualityThreshold;

inline float DotProduct(const float u[3], const float v[3]) {
  return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
}

std::vector<float> ComputeKernel(float sigma) {
  const float m = 2.25;  // Accuracy increases when m is increased.
  const float scaler = -1.0 / (2 * sigma * sigma);
  const int diff = std::max<int>(1, m * fabs(sigma));
  std::vector<float> kernel(2 * diff + 1);
  for (int i = -diff; i <= diff; ++i) {
    kernel[i + diff] = exp(scaler * i * i);
  }
  return kernel;
}

void ConvolveBorderColumn(
    const ImageF& in,
    const std::vector<float>& kernel,
    const float weight_no_border,
    const float border_ratio,
    const size_t x,
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
ImageF Convolution(const ImageF& in,
                   const std::vector<float>& kernel,
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
  return Convolution(Convolution(in, kernel, border_ratio),
                     kernel, border_ratio);
}

// DoGBlur is an approximate of difference of Gaussians. We use it to
// approximate LoG (Laplacian of Gaussians).
// See: https://en.wikipedia.org/wiki/Difference_of_Gaussians
// For motivation see:
// https://en.wikipedia.org/wiki/Pyramid_(image_processing)#Laplacian_pyramid
ImageF DoGBlur(const ImageF& in, float sigma, float border_ratio) {
  ImageF blur1 = Blur(in, sigma, border_ratio);
  ImageF blur2 = Blur(in, sigma * 2.0f, border_ratio);
  static const float mix = 0.5;
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

// Clamping linear interpolator.
inline double InterpolateClampNegative(const double *array,
                                       int size, double ix) {
  if (ix < 0) {
    ix = 0;
  }
  int baseix = static_cast<int>(ix);
  double res;
  if (baseix >= size - 1) {
    res = array[size - 1];
  } else {
    double mix = ix - baseix;
    int nextix = baseix + 1;
    res = array[baseix] + mix * (array[nextix] - array[baseix]);
  }
  return res;
}

double GammaMinArg() {
  double out0, out1, out2;
  OpsinAbsorbance(0.0, 0.0, 0.0, &out0, &out1, &out2);
  return std::min(out0, std::min(out1, out2));
}

double GammaMaxArg() {
  double out0, out1, out2;
  OpsinAbsorbance(255.0, 255.0, 255.0, &out0, &out1, &out2);
  return std::max(out0, std::max(out1, out2));
}

// The input images c0 and c1 include the high frequency component only.
// The output scalar images b0 and b1 include the correlation of Y and
// B component at a Gaussian locality around the respective pixel.
ImageF BlurredBlueCorrelation(const std::vector<ImageF>& uhf,
                              const std::vector<ImageF>& hf) {
  const size_t xsize = uhf[0].xsize();
  const size_t ysize = uhf[0].ysize();
  ImageF yb(xsize, ysize);
  ImageF yy(xsize, ysize);
  for (size_t y = 0; y < ysize; ++y) {
    const float* const BUTTERAUGLI_RESTRICT row_uhf_y = uhf[1].Row(y);
    const float* const BUTTERAUGLI_RESTRICT row_uhf_b = uhf[2].Row(y);
    const float* const BUTTERAUGLI_RESTRICT row_hf_y = hf[1].Row(y);
    const float* const BUTTERAUGLI_RESTRICT row_hf_b = hf[2].Row(y);
    float* const BUTTERAUGLI_RESTRICT row_yb = yb.Row(y);
    float* const BUTTERAUGLI_RESTRICT row_yy = yy.Row(y);
    for (size_t x = 0; x < xsize; ++x) {
      const float yval = row_hf_y[x] + row_uhf_y[x];
      const float bval = row_hf_b[x] + row_uhf_b[x];
      row_yb[x] = yval * bval;
      row_yy[x] = yval * yval;
    }
  }
  const double kSigma = 7.58381422616;
  ImageF yy_blurred = Blur(yy, kSigma, 0.0);
  ImageF yb_blurred = Blur(yb, kSigma, 0.0);
  for (size_t y = 0; y < ysize; ++y) {
    const float* const BUTTERAUGLI_RESTRICT row_uhf_y = uhf[1].Row(y);
    const float* const BUTTERAUGLI_RESTRICT row_hf_y = hf[1].Row(y);
    const float* const BUTTERAUGLI_RESTRICT row_yy = yy_blurred.Row(y);
    float* const BUTTERAUGLI_RESTRICT row_yb = yb_blurred.Row(y);
    for (size_t x = 0; x < xsize; ++x) {
      static const float epsilon = 20;
      const float yval = row_hf_y[x] + row_uhf_y[x];
      row_yb[x] *= yval / (row_yy[x] + epsilon);
    }
  }
  return yb_blurred;
}

double SimpleGamma(double v) {
  static const double kGamma = 0.38666411018;
  static const double limit = 43.0100364492;
  double bright = v - limit;
  if (bright >= 0) {
    static const double mul = 0.0859052834393;
    v -= bright * mul;
  }
  static const double limit2 = 93.4201379994;
  double bright2 = v - limit2;
  if (bright2 >= 0) {
    static const double mul = 0.293598007156;
    v -= bright2 * mul;
  }
  static const double offset = 0.194471449584;
  static const double scale = 8.84156416367;
  double retval = scale * (offset + pow(v, kGamma));
  return retval;
}

static inline double Gamma(double v) {
  //return SimpleGamma(v);
  return GammaPolynomial(v);
}

std::vector<ImageF> OpsinDynamicsImage(const std::vector<ImageF>& rgb) {
  PROFILER_FUNC;
#if 0
  PrintStatistics("rgb", rgb);
#endif
  std::vector<ImageF> xyb(3);
  std::vector<ImageF> blurred(3);
  const double kSigma = 1.25895091427;
  for (int i = 0; i < 3; ++i) {
    xyb[i] = ImageF(rgb[i].xsize(), rgb[i].ysize());
    blurred[i] = Blur(rgb[i], kSigma, 0.0f);
  }
  for (size_t y = 0; y < rgb[0].ysize(); ++y) {
    const float* const BUTTERAUGLI_RESTRICT row_r = rgb[0].Row(y);
    const float* const BUTTERAUGLI_RESTRICT row_g = rgb[1].Row(y);
    const float* const BUTTERAUGLI_RESTRICT row_b = rgb[2].Row(y);
    const float* const BUTTERAUGLI_RESTRICT row_blurred_r = blurred[0].Row(y);
    const float* const BUTTERAUGLI_RESTRICT row_blurred_g = blurred[1].Row(y);
    const float* const BUTTERAUGLI_RESTRICT row_blurred_b = blurred[2].Row(y);
    float* const BUTTERAUGLI_RESTRICT row_out_x = xyb[0].Row(y);
    float* const BUTTERAUGLI_RESTRICT row_out_y = xyb[1].Row(y);
    float* const BUTTERAUGLI_RESTRICT row_out_b = xyb[2].Row(y);
    for (size_t x = 0; x < rgb[0].xsize(); ++x) {
      float sensitivity[3];
      {
        // Calculate sensitivity based on the smoothed image gamma derivative.
        float pre_mixed0, pre_mixed1, pre_mixed2;
        OpsinAbsorbance(row_blurred_r[x], row_blurred_g[x], row_blurred_b[x],
                        &pre_mixed0, &pre_mixed1, &pre_mixed2);
        // TODO(janwas): use new polynomial to compute Gamma(x)/x derivative.
        sensitivity[0] = Gamma(pre_mixed0) / pre_mixed0;
        sensitivity[1] = Gamma(pre_mixed1) / pre_mixed1;
        sensitivity[2] = Gamma(pre_mixed2) / pre_mixed2;
      }
      float cur_mixed0, cur_mixed1, cur_mixed2;
      OpsinAbsorbance(row_r[x], row_g[x], row_b[x],
                      &cur_mixed0, &cur_mixed1, &cur_mixed2);
      cur_mixed0 *= sensitivity[0];
      cur_mixed1 *= sensitivity[1];
      cur_mixed2 *= sensitivity[2];
      RgbToXyb(cur_mixed0, cur_mixed1, cur_mixed2,
               &row_out_x[x], &row_out_y[x], &row_out_b[x]);
    }
  }
#if 0
  PrintStatistics("xyb", xyb);
#endif
  return xyb;
}

// Make area around zero less important (remove it).
static BUTTERAUGLI_INLINE float RemoveRangeAroundZero(float w, float x) {
  return x > w ? x - w : x < -w ? x + w : 0.0f;
}

// Make area around zero more important (2x it until the limit).
static BUTTERAUGLI_INLINE float AmplifyRangeAroundZero(float w, float x) {
  return x > w ? x + w : x < -w ? x - w : 2.0f * x;
}

std::vector<ImageF> ModifyRangeAroundZero(const double warray[2],
                                          const std::vector<ImageF>& in) {
  std::vector<ImageF> out;
  for (int k = 0; k < 3; ++k) {
    ImageF plane(in[k].xsize(), in[k].ysize());
    for (int y = 0; y < plane.ysize(); ++y) {
      auto row_in = in[k].Row(y);
      auto row_out = plane.Row(y);
      if (k == 2) {
        memcpy(row_out, row_in, plane.xsize() * sizeof(row_out[0]));
      } else if (warray[k] >= 0) {
        const double w = warray[k];
        for (int x = 0; x < plane.xsize(); ++x) {
          row_out[x] = RemoveRangeAroundZero(w, row_in[x]);
        }
      } else {
        const double w = -warray[k];
        for (int x = 0; x < plane.xsize(); ++x) {
          row_out[x] = AmplifyRangeAroundZero(w, row_in[x]);
        }
      }
    }
    out.emplace_back(std::move(plane));
  }
  return out;
}

// XybLowFreqToVals converts from low-frequency XYB space to the 'vals' space.
// Vals space can be converted to L2-norm space (Euclidean and normalized)
// through visual masking.
template <class V>
BUTTERAUGLI_INLINE void XybLowFreqToVals(const V &x, const V &y, const V &b_arg,
                                         V *BUTTERAUGLI_RESTRICT valx,
                                         V *BUTTERAUGLI_RESTRICT valy,
                                         V *BUTTERAUGLI_RESTRICT valb) {
  static const double xmuli = 6.41026595943;
  static const double ymuli = 7.03852940672;
  static const double bmuli = 7.43333021578;
  static const double y_to_b_muli = -0.748101111304;

  const V xmul(xmuli);
  const V ymul(ymuli);
  const V bmul(bmuli);
  const V y_to_b_mul(y_to_b_muli);
  const V b = b_arg + y_to_b_mul * y;
  *valb = b * bmul;
  *valx = x * xmul;
  *valy = y * ymul;
}

static void SeparateFrequencies(size_t xsize, size_t ysize,
                                const ImageF& plane,
                                ImageF* const BUTTERAUGLI_RESTRICT lf,
                                ImageF* const BUTTERAUGLI_RESTRICT mf,
                                ImageF* const BUTTERAUGLI_RESTRICT hf,
                                ImageF* const BUTTERAUGLI_RESTRICT uhf) {
  // Extract lf ...
  static const double kSigmaLf = 6.36682082798;
  *lf = DoGBlur(plane, kSigmaLf, 0.0f);
  // ... and keep everything else in mf.
  *mf = ImageF(xsize, ysize);
  for (size_t y = 0; y < ysize; ++y) {
    for (size_t x = 0; x < xsize; ++x) {
      mf->Row(y)[x] = plane.Row(y)[x] - lf->Row(y)[x];
    }
  }
  // Divide mf into mf and hf.
  static const double kSigmaHf = 0.5 * kSigmaLf;
  *hf = ImageF(xsize, ysize);
  for (size_t y = 0; y < ysize; ++y) {
    for (size_t x = 0; x < xsize; ++x) {
      hf->Row(y)[x] = mf->Row(y)[x];
    }
  }
  *mf = DoGBlur(*mf, kSigmaHf, 0.0f);
  for (size_t y = 0; y < ysize; ++y) {
    for (size_t x = 0; x < xsize; ++x) {
      hf->Row(y)[x] -= mf->Row(y)[x];
    }
  }
  // Divide hf into hf and uhf.
  static const double kSigmaUhf = 0.5 * kSigmaHf;
  *uhf  = ImageF(xsize, ysize);
  for (size_t y = 0; y < ysize; ++y) {
    for (size_t x = 0; x < xsize; ++x) {
      uhf->Row(y)[x] = hf->Row(y)[x];
    }
  }
  *hf = DoGBlur(*hf, kSigmaUhf, 0.0f);
  for (size_t y = 0; y < ysize; ++y) {
    for (size_t x = 0; x < xsize; ++x) {
      uhf->Row(y)[x] -= hf->Row(y)[x];
    }
  }
}

static void SeparateFrequencies(
    size_t xsize, size_t ysize,
    const std::vector<ImageF>& xyb,
    PsychoImage &ps) {
  PROFILER_FUNC;
  ps.lf.resize(3);
  ps.mf.resize(3);
  ps.hf.resize(3);
  ps.uhf.resize(3);
  for (int i = 0; i < 3; ++i) {
    SeparateFrequencies(xsize, ysize, xyb[i],
                        &ps.lf[i], &ps.mf[i], &ps.hf[i], &ps.uhf[i]);
  }
  // Modify range around zero code only concerns the high frequency
  // planes and only the X and Y channels.
  static const double uhf_xy_modification[2] = {
    -0.34424105878,
    -2.26396645429,
  };
  static const double hf_xy_modification[2] = {
    0.00406506728122,
    -0.0646629947971,
  };
  ps.uhf = ModifyRangeAroundZero(uhf_xy_modification, ps.uhf);
  ps.hf = ModifyRangeAroundZero(hf_xy_modification, ps.hf);
  // Convert low freq xyb to vals space so that we can do a simple squared sum
  // diff on the low frequencies later.
  for (size_t y = 0; y < ysize; ++y) {
    float* BUTTERAUGLI_RESTRICT const row_x = ps.lf[0].Row(y);
    float* BUTTERAUGLI_RESTRICT const row_y = ps.lf[1].Row(y);
    float* BUTTERAUGLI_RESTRICT const row_b = ps.lf[2].Row(y);
    for (size_t x = 0; x < xsize; ++x) {
      float valx, valy, valb;
      XybLowFreqToVals(row_x[x], row_y[x], row_b[x], &valx, &valy, &valb);
      row_x[x] = valx;
      row_y[x] = valy;
      row_b[x] = valb;
    }
  }
}

static void SameNoiseLevels(const ImageF& i0, const ImageF& i1,
                            const double kSigma,
                            const double w,
                            ImageF* BUTTERAUGLI_RESTRICT diffmap) {
  ImageF blurred0 = CopyPixels(i0);
  ImageF blurred1 = CopyPixels(i1);
  for (size_t y = 0; y < i0.ysize(); ++y) {
    float* BUTTERAUGLI_RESTRICT const row0 = blurred0.Row(y);
    float* BUTTERAUGLI_RESTRICT const row1 = blurred1.Row(y);
    for (size_t x = 0; x < i0.xsize(); ++x) {
      row0[x] = fabs(row0[x]);
      row1[x] = fabs(row1[x]);
    }
  }
  blurred0 = Blur(blurred0, kSigma, 0.0);
  blurred1 = Blur(blurred1, kSigma, 0.0);
  for (size_t y = 0; y < i0.ysize(); ++y) {
    const float* BUTTERAUGLI_RESTRICT const row0 = blurred0.Row(y);
    const float* BUTTERAUGLI_RESTRICT const row1 = blurred1.Row(y);
    float* BUTTERAUGLI_RESTRICT const row_diff = diffmap->Row(y);
    for (size_t x = 0; x < i0.xsize(); ++x) {
      double diff = row0[x] - row1[x];
      row_diff[x] += w * diff * diff;
    }
  }
}


static void L2Diff(const ImageF& i0, const ImageF& i1, const double w,
                   ImageF* BUTTERAUGLI_RESTRICT diffmap) {
  for (size_t y = 0; y < i0.ysize(); ++y) {
    const float* BUTTERAUGLI_RESTRICT const row0 = i0.Row(y);
    const float* BUTTERAUGLI_RESTRICT const row1 = i1.Row(y);
    float* BUTTERAUGLI_RESTRICT const row_diff = diffmap->Row(y);
    for (size_t x = 0; x < i0.xsize(); ++x) {
      double diff = row0[x] - row1[x];
      row_diff[x] += w * diff * diff;
    }
  }
}

// Making a cluster of local errors to be more impactful than
// just a single error.
ImageF CalculateDiffmap(const ImageF& diffmap_in) {
  PROFILER_FUNC;
  // Take square root.
  ImageF diffmap(diffmap_in.xsize(), diffmap_in.ysize());
  for (size_t y = 0; y < diffmap.ysize(); ++y) {
    const float* const BUTTERAUGLI_RESTRICT row_in = diffmap_in.Row(y);
    float* const BUTTERAUGLI_RESTRICT row_out = diffmap.Row(y);
    for (size_t x = 0; x < diffmap.xsize(); ++x) {
      const float orig_val = row_in[x];
      constexpr float kInitialSlope = 100.0f;
      // TODO(b/29974893): Until that is fixed do not call sqrt on very small
      // numbers.
      row_out[x] = (orig_val < (1.0f / (kInitialSlope * kInitialSlope))
                    ? kInitialSlope * orig_val
                    : std::sqrt(orig_val));
    }
  }
  {
    static const double kSigma = 8.67159747549;
    static const double mul1 = 21.5955218386;
    static const float scale = 1.0f / (1.0f + mul1);
    static const double border_ratio = 0.14172952214;
    ImageF blurred = Blur(diffmap, kSigma, border_ratio);
    for (int y = 0; y < diffmap.ysize(); ++y) {
      const float* const BUTTERAUGLI_RESTRICT row_blurred = blurred.Row(y);
      float* const BUTTERAUGLI_RESTRICT row = diffmap.Row(y);
      for (int x = 0; x < diffmap.xsize(); ++x) {
        row[x] += mul1 * row_blurred[x];
        row[x] *= scale;
      }
    }
  }
  static const double kSigma2 = 0.210786763509;
  return Blur(diffmap, kSigma2, 0.0f);
}

void MaskPsychoImage(const PsychoImage& pi0, const PsychoImage& pi1,
                     const size_t xsize, const size_t ysize,
                     std::vector<ImageF>* BUTTERAUGLI_RESTRICT mask,
                     std::vector<ImageF>* BUTTERAUGLI_RESTRICT mask_dc) {
  std::vector<ImageF> mask_xyb0 = CreatePlanes<float>(xsize, ysize, 3);
  std::vector<ImageF> mask_xyb1 = CreatePlanes<float>(xsize, ysize, 3);
  static const double muls[4] = {
    0.0,
    1.46057244384,
    0.923109683343,
    2.23386060381,
  };
  for (int i = 0; i < 2; ++i) {
    double a = muls[2 * i];
    double b = muls[2 * i + 1];
    for (size_t y = 0; y < ysize; ++y) {
      const float* const BUTTERAUGLI_RESTRICT row_hf0 = pi0.hf[i].Row(y);
      const float* const BUTTERAUGLI_RESTRICT row_hf1 = pi1.hf[i].Row(y);
      const float* const BUTTERAUGLI_RESTRICT row_uhf0 = pi0.uhf[i].Row(y);
      const float* const BUTTERAUGLI_RESTRICT row_uhf1 = pi1.uhf[i].Row(y);
      float* const BUTTERAUGLI_RESTRICT row0 = mask_xyb0[i].Row(y);
      float* const BUTTERAUGLI_RESTRICT row1 = mask_xyb1[i].Row(y);
      for (size_t x = 0; x < xsize; ++x) {
        row0[x] = a * row_uhf0[x] + b * row_hf0[x];
        row1[x] = a * row_uhf1[x] + b * row_hf1[x];
      }
    }
  }
  Mask(mask_xyb0, mask_xyb1, mask, mask_dc);
}

ButteraugliComparator::ButteraugliComparator(const std::vector<ImageF>& rgb0)
    : xsize_(rgb0[0].xsize()),
      ysize_(rgb0[0].ysize()),
      num_pixels_(xsize_ * ysize_) {
  if (xsize_ < 8 || ysize_ < 8) return;
  std::vector<ImageF> xyb0 = OpsinDynamicsImage(rgb0);
#if 0
  DumpPpm("/tmp/sep0_orig.ppm", xyb0);
#endif
  SeparateFrequencies(xsize_, ysize_, xyb0, pi0_);
}

void ButteraugliComparator::Mask(
    std::vector<ImageF>* BUTTERAUGLI_RESTRICT mask,
    std::vector<ImageF>* BUTTERAUGLI_RESTRICT mask_dc) const {
  MaskPsychoImage(pi0_, pi0_, xsize_, ysize_, mask, mask_dc);
}

void ButteraugliComparator::Diffmap(const std::vector<ImageF>& rgb1,
                                    ImageF &result) const {
  PROFILER_FUNC;
  if (xsize_ < 8 || ysize_ < 8) return;
  DiffmapOpsinDynamicsImage(OpsinDynamicsImage(rgb1), result);
}

void ButteraugliComparator::DiffmapOpsinDynamicsImage(
    const std::vector<ImageF>& xyb1,
    ImageF &result) const {
  PROFILER_FUNC;
  if (xsize_ < 8 || ysize_ < 8) return;
  PsychoImage pi1;
  SeparateFrequencies(xsize_, ysize_, xyb1, pi1);
#if 0
  DumpPpm("/tmp/sep1_orig.ppm", xyb1);
#endif
  result = ImageF(xsize_, ysize_);
  DiffmapPsychoImage(pi1, result);
}

void ButteraugliComparator::DiffmapPsychoImage(const PsychoImage& pi1,
                                               ImageF& result) const {
  PROFILER_FUNC;
  if (xsize_ < 8 || ysize_ < 8) {
    return;
  }
#if 0
  PrintStatistics("hf0", pi0_.hf);
  PrintStatistics("hf1", pi1.hf);
  PrintStatistics("mf0", pi0_.mf);
  PrintStatistics("mf1", pi1.mf);
  PrintStatistics("lf0", pi0_.lf);
  PrintStatistics("lf1", pi1.lf);
#endif

  std::vector<ImageF> block_diff_dc(3);
  std::vector<ImageF> block_diff_ac(3);
  for (int c = 0; c < 3; ++c) {
    block_diff_dc[c] = ImageF(xsize_, ysize_, 0.0);
    block_diff_ac[c] = ImageF(xsize_, ysize_, 0.0);
  }

  static const double wUhfMalta = 0.101284636565;
  MaltaDiffMap(pi0_.uhf[1], pi1.uhf[1], wUhfMalta, &block_diff_ac[1]);

  static const double wHfMalta = 0.383664443656;
  MaltaDiffMap(pi0_.hf[1], pi1.hf[1], wHfMalta, &block_diff_ac[1]);

  static const double wmul[11] = {
    17.7349834613,
    3.31830119078,
    0,
    0.5,
    0.5,
    2.66390198411,
    0.463471206635,
    0.81710151933,
    0.715902543607,
    0.0,
    5.76225859514,
  };

  static const double kSigmaHf = 2.12433021934;
  SameNoiseLevels(pi0_.hf[1], pi1.hf[1], kSigmaHf, wmul[10],
                  &block_diff_ac[1]);

  for (int c = 0; c < 3; ++c) {
    if (wmul[c] != 0) {
      L2Diff(pi0_.hf[c], pi1.hf[c], wmul[c], &block_diff_ac[c]);
    }
    L2Diff(pi0_.mf[c], pi1.mf[c], wmul[3 + c], &block_diff_ac[c]);
    L2Diff(pi0_.lf[c], pi1.lf[c], wmul[6 + c], &block_diff_dc[c]);
  }

  static const double wBlueCorr = 0.0259677529058;
  ImageF blurred_b_y_correlation0 = BlurredBlueCorrelation(pi0_.uhf, pi0_.hf);
  ImageF blurred_b_y_correlation1 = BlurredBlueCorrelation(pi1.uhf, pi1.hf);
  L2Diff(blurred_b_y_correlation0, blurred_b_y_correlation1, wBlueCorr,
         &block_diff_ac[2]);

  std::vector<ImageF> mask_xyb;
  std::vector<ImageF> mask_xyb_dc;
  MaskPsychoImage(pi0_, pi1, xsize_, ysize_, &mask_xyb, &mask_xyb_dc);

#if 0
  DumpPpm("/tmp/mask.ppm", mask_xyb, 777);
#endif

  result = CalculateDiffmap(
      CombineChannels(mask_xyb, mask_xyb_dc, block_diff_dc, block_diff_ac));
#if 0
  PrintStatistics("diffmap", result);
#endif
}

void ButteraugliComparator::MaltaDiffMap(
    const ImageF& y0, const ImageF& y1,
    const double weight,
    ImageF* BUTTERAUGLI_RESTRICT block_diff_ac) const {
  PROFILER_FUNC;
  const double len = 3.25;
  const double w = sqrt(weight) / (len * 2 + 1);
  const double norm1 = 338.598521738;
  const double norm2 = w * norm1;
  std::vector<float> diffs(ysize_ * xsize_);
  // Only process the Y channel for now.
  for (size_t y = 0, ix = 0; y < ysize_; ++y) {
    const float* BUTTERAUGLI_RESTRICT const row0 = y0.Row(y);
    const float* BUTTERAUGLI_RESTRICT const row1 = y1.Row(y);
    for (size_t x = 0; x < xsize_; ++x, ++ix) {
      double absval = 0.5 * (std::abs(row0[x]) + std::abs(row1[x]));
      double diff = row0[x] - row1[x];
      double scaler = norm2 / (norm1 + absval);
      diffs[ix] = scaler * diff;
    }
  }
  for (size_t y0 = 0; y0 < ysize_; ++y0) {
    float* const BUTTERAUGLI_RESTRICT row_diff = block_diff_ac->Row(y0);
    for (size_t x0 = 0; x0 < xsize_; ++x0) {
      int delta[8][2] = { { 256, 0 }, {0, 256},
                          { 256, 256 }, { 256, -256 },
                          { 256, 128 }, { 256, -128 },
                          { 128, 256}, { 128, -256 } };
      int len[8] = { 3, 3, 2, 2, 3, 3, 3, 3 };
      for (int dir = 0; dir < 8; ++dir) {
        double sum = 0;
        for (int scan = -len[dir]; scan <= len[dir]; ++scan) {
          int dx = delta[dir][0] * scan;
          int dy = delta[dir][1] * scan;
          int fracdx = dx & 0xff;
          int fracdy = dy & 0xff;
          int mfracdx = 255 - fracdx;
          int mfracdy = 255 - fracdy;
          int x = x0 + (dx >> 8);
          int y = y0 + (dy >> 8);
          if (x < 0 || y < 0 || x >= xsize_ - 1 || y >= ysize_ - 1) {
            continue;
          }
          int ix = y * xsize_ + x;
          const double val = (1.0 / (255 * 255)) *
              (mfracdx * mfracdy * diffs[ix] +
               fracdx * mfracdy * diffs[ix + 1] +
               mfracdx * fracdy * diffs[ix + xsize_] +
               fracdx * fracdy * diffs[ix + xsize_ + 1]);
          sum += val;
        }
        row_diff[x0] += sum * sum;
      }
    }
  }
}

ImageF ButteraugliComparator::CombineChannels(
    const std::vector<ImageF>& mask_xyb,
    const std::vector<ImageF>& mask_xyb_dc,
    const std::vector<ImageF>& block_diff_dc,
    const std::vector<ImageF>& block_diff_ac) const {
  PROFILER_FUNC;
  ImageF result(xsize_, ysize_);
  for (size_t y = 0; y < ysize_; ++y) {
    float* const BUTTERAUGLI_RESTRICT row_out = result.Row(y);
    for (size_t x = 0; x < xsize_; ++x) {
      float mask[3];
      float dc_mask[3];
      float diff_dc[3];
      float diff_ac[3];
      for (int i = 0; i < 3; ++i) {
        mask[i] = mask_xyb[i].Row(y)[x];
        dc_mask[i] = mask_xyb_dc[i].Row(y)[x];
        diff_dc[i] = block_diff_dc[i].Row(y)[x];
        diff_ac[i] = block_diff_ac[i].Row(y)[x];
      }
      row_out[x] = (DotProduct(diff_dc, dc_mask) + DotProduct(diff_ac, mask));
    }
  }
  return result;
}

double ButteraugliScoreFromDiffmap(const ImageF& diffmap) {
  PROFILER_FUNC;
  float retval = 0.0f;
  for (size_t y = 0; y < diffmap.ysize(); ++y) {
    const float * const BUTTERAUGLI_RESTRICT row = diffmap.Row(y);
    for (size_t x = 0; x < diffmap.xsize(); ++x) {
      retval = std::max(retval, row[x]);
    }
  }
  return retval;
}

// ===== Functions used by Mask only =====
static std::array<double, 512> MakeMask(
    double extmul, double extoff,
    double mul, double offset,
    double scaler) {
  std::array<double, 512> lut;
  for (int i = 0; i < lut.size(); ++i) {
    const double c = mul / ((0.01 * scaler * i) + offset);
    lut[i] = kGlobalScale * (1.0 + extmul * (c + extoff));
    //assert(lut[i] >= 0.0);
    lut[i] *= lut[i];
  }
  return lut;
}

double MaskX(double delta) {
  PROFILER_FUNC;
  static const double extmul = 3.64283931704;
  static const double extoff = 2.98710342481;
  static const double offset = 0.351975910434;
  static const double scaler = 39.1249494195;
  static const double mul = 5.62518603179;
  static const std::array<double, 512> lut =
                MakeMask(extmul, extoff, mul, offset, scaler);
  return InterpolateClampNegative(lut.data(), lut.size(), delta);
}

double MaskY(double delta) {
  PROFILER_FUNC;
  static const double extmul = 1.32169579961;
  static const double extoff = -0.443630344604;
  static const double offset = 1.49231185359;
  static const double scaler = 1.1091847392;
  static const double mul = 8.6494152807;
  static const std::array<double, 512> lut =
      MakeMask(extmul, extoff, mul, offset, scaler);
  return InterpolateClampNegative(lut.data(), lut.size(), delta);
}

double MaskDcX(double delta) {
  PROFILER_FUNC;
  static const double extmul = 10.7425569904;
  static const double extoff = -0.202005949852;
  static const double offset = 1.61221792276;
  static const double scaler = 20.0343306931;
  static const double mul = 4.97988707281;
  static const std::array<double, 512> lut =
      MakeMask(extmul, extoff, mul, offset, scaler);
  return InterpolateClampNegative(lut.data(), lut.size(), delta);
}

double MaskDcY(double delta) {
  PROFILER_FUNC;
  static const double extmul = 0.00101370711517;
  static const double extoff = 117.250352326;
  static const double offset = 0.0353091313192;
  static const double scaler = 0.434400168812;
  static const double mul = 71.8677642068;
  static const std::array<double, 512> lut =
      MakeMask(extmul, extoff, mul, offset, scaler);
  return InterpolateClampNegative(lut.data(), lut.size(), delta);
}

ImageF DiffPrecompute(const ImageF& xyb0, const ImageF& xyb1) {
  PROFILER_FUNC;
  const size_t xsize = xyb0.xsize();
  const size_t ysize = xyb0.ysize();
  ImageF result(xsize, ysize);
  size_t x2, y2;
  for (size_t y = 0; y < ysize; ++y) {
    if (y + 1 < ysize) {
      y2 = y + 1;
    } else if (y > 0) {
      y2 = y - 1;
    } else {
      y2 = y;
    }
    const float* const BUTTERAUGLI_RESTRICT row0_in = xyb0.Row(y);
    const float* const BUTTERAUGLI_RESTRICT row1_in = xyb1.Row(y);
    const float* const BUTTERAUGLI_RESTRICT row0_in2 = xyb0.Row(y2);
    const float* const BUTTERAUGLI_RESTRICT row1_in2 = xyb1.Row(y2);
    float* const BUTTERAUGLI_RESTRICT row_out = result.Row(y);
    for (size_t x = 0; x < xsize; ++x) {
      if (x + 1 < xsize) {
        x2 = x + 1;
      } else if (x > 0) {
        x2 = x - 1;
      } else {
        x2 = x;
      }
      float minDir = fabsf(row0_in[x] - row0_in[x2]);
      minDir = std::min(minDir, fabsf(row0_in[x] - row0_in2[x]));
      minDir = std::min(minDir, fabsf(row0_in[x] - row0_in2[x2]));
      minDir = std::min(minDir, fabsf(row0_in[x2] - row0_in2[x]));
      minDir = std::min(minDir, fabsf(row1_in[x] - row1_in[x2]));
      minDir = std::min(minDir, fabsf(row1_in[x] - row1_in2[x]));
      minDir = std::min(minDir, fabsf(row1_in[x] - row1_in2[x2]));
      minDir = std::min(minDir, fabsf(row1_in[x2] - row1_in2[x]));
      double sup0 =
          (fabsf(row0_in[x] - row0_in[x2]) + fabsf(row0_in[x] - row0_in2[x]));
      double sup1 =
          (fabsf(row1_in[x] - row1_in[x2]) + fabsf(row1_in[x] - row1_in2[x]));
      static const double mul0 = 0.758790907906;
      static const double mul1 = 1.51115538575;
      row_out[x] = mul0 * std::min(sup0, sup1) + mul1 * minDir;
      //      static const double cutoff = 170.633555669
      static const double cutoff = 110;
      if (row_out[x] >= cutoff) {
        row_out[x] = cutoff;
      }
    }
  }
  return result;
}

void Mask(const std::vector<ImageF>& xyb0,
          const std::vector<ImageF>& xyb1,
          std::vector<ImageF>* BUTTERAUGLI_RESTRICT mask,
          std::vector<ImageF>* BUTTERAUGLI_RESTRICT mask_dc) {
  PROFILER_FUNC;
  const size_t xsize = xyb0[0].xsize();
  const size_t ysize = xyb0[0].ysize();
  mask->resize(3);
  *mask_dc = CreatePlanes<float>(xsize, ysize, 3);
  for (int i = 0; i < 2; ++i) {
    static const double sigma[3] = {
      8.33097939385,
      6.58474755576,
    };
    (*mask)[i] = Blur(DiffPrecompute(xyb0[i], xyb1[i]), sigma[i], 0.0f);
  }
  (*mask)[2] = ImageF(xsize, ysize);
  static const double mul[2] = {
    9.28145708505,
    2.80365829948,
  };
  static const double w00 = 9.0922759209;
  static const double w11 = 2.73439127133;
  static const double w_ytob_hf = 4.15536498944;
  static const double w_ytob_lf = 6.24696277952;
  static const double p1_to_p0 = 0.0153416058747;

  for (size_t y = 0; y < ysize; ++y) {
    for (size_t x = 0; x < xsize; ++x) {
      const double s0 = (*mask)[0].Row(y)[x];
      const double s1 = (*mask)[1].Row(y)[x];
      const double p1 = mul[1] * w11 * s1;
      const double p0 = mul[0] * w00 * s0 + p1_to_p0 * p1;

      (*mask)[0].Row(y)[x] = MaskX(p0);
      (*mask)[1].Row(y)[x] = MaskY(p1);
      (*mask)[2].Row(y)[x] = w_ytob_hf * MaskY(p1);
      (*mask_dc)[0].Row(y)[x] = MaskDcX(p0);
      (*mask_dc)[1].Row(y)[x] = MaskDcY(p1);
      (*mask_dc)[2].Row(y)[x] = w_ytob_lf * MaskDcY(p1);
    }
  }
#if 0
  PrintStatistics("mask", *mask);
  PrintStatistics("mask_dc", *mask_dc);
#endif
}

void ButteraugliDiffmap(const std::vector<ImageF> &rgb0_image,
                        const std::vector<ImageF> &rgb1_image,
                        ImageF &result_image) {
  const size_t xsize = rgb0_image[0].xsize();
  const size_t ysize = rgb0_image[0].ysize();
  static const int kMax = 8;
  if (xsize < kMax || ysize < kMax) {
    // Butteraugli values for small (where xsize or ysize is smaller
    // than 8 pixels) images are non-sensical, but most likely it is
    // less disruptive to try to compute something than just give up.
    // Temporarily extend the borders of the image to fit 8 x 8 size.
    int xborder = xsize < kMax ? (kMax - xsize) / 2 : 0;
    int yborder = ysize < kMax ? (kMax - ysize) / 2 : 0;
    size_t xscaled = std::max<size_t>(kMax, xsize);
    size_t yscaled = std::max<size_t>(kMax, ysize);
    std::vector<ImageF> scaled0 = CreatePlanes<float>(xscaled, yscaled, 3);
    std::vector<ImageF> scaled1 = CreatePlanes<float>(xscaled, yscaled, 3);
    for (int i = 0; i < 3; ++i) {
      for (int y = 0; y < yscaled; ++y) {
        for (int x = 0; x < xscaled; ++x) {
          size_t x2 = std::min<size_t>(xsize - 1, std::max(0, x - xborder));
          size_t y2 = std::min<size_t>(ysize - 1, std::max(0, y - yborder));
          scaled0[i].Row(y)[x] = rgb0_image[i].Row(y2)[x2];
          scaled1[i].Row(y)[x] = rgb1_image[i].Row(y2)[x2];
        }
      }
    }
    ImageF diffmap_scaled;
    ButteraugliDiffmap(scaled0, scaled1, diffmap_scaled);
    result_image = ImageF(xsize, ysize);
    for (int y = 0; y < ysize; ++y) {
      for (int x = 0; x < xsize; ++x) {
        result_image.Row(y)[x] = diffmap_scaled.Row(y + yborder)[x + xborder];
      }
    }
    return;
  }
  ButteraugliComparator butteraugli(rgb0_image);
  butteraugli.Diffmap(rgb1_image, result_image);
}

bool ButteraugliInterface(const std::vector<ImageF> &rgb0,
                          const std::vector<ImageF> &rgb1,
                          ImageF &diffmap,
                          double &diffvalue) {
  const size_t xsize = rgb0[0].xsize();
  const size_t ysize = rgb0[0].ysize();
  if (xsize < 1 || ysize < 1) {
    return false;  // No image.
  }
  for (int i = 1; i < 3; i++) {
    if (rgb0[i].xsize() != xsize || rgb0[i].ysize() != ysize ||
        rgb1[i].xsize() != xsize || rgb1[i].ysize() != ysize) {
      return false;  // Image planes must have same dimensions.
    }
  }
  ButteraugliDiffmap(rgb0, rgb1, diffmap);
  diffvalue = ButteraugliScoreFromDiffmap(diffmap);
  return true;
}

bool ButteraugliAdaptiveQuantization(size_t xsize, size_t ysize,
    const std::vector<std::vector<float> > &rgb, std::vector<float> &quant) {
  if (xsize < 16 || ysize < 16) {
    return false;  // Butteraugli is undefined for small images.
  }
  size_t size = xsize * ysize;

  std::vector<ImageF> rgb_planes = PlanesFromPacked(xsize, ysize, rgb);
  std::vector<ImageF> scale_xyb;
  std::vector<ImageF> scale_xyb_dc;
  Mask(rgb_planes, rgb_planes, &scale_xyb, &scale_xyb_dc);
  quant.reserve(size);

  // Mask gives us values in 3 color channels, but for now we take only
  // the intensity channel.
  for (size_t y = 0; y < ysize; ++y) {
    for (size_t x = 0; x < xsize; ++x) {
      quant.push_back(scale_xyb[1].Row(y)[x]);
    }
  }
  return true;
}

double ButteraugliFuzzyClass(double score) {
  static const double fuzzy_width_up = 13.7735765392;
  static const double fuzzy_width_down = 4.91801889737;
  static const double m0 = 2.0;
  static const double scaler = 0.654697646654;
  double val;
  if (score < 1.0) {
    // val in [scaler .. 2.0]
    val = m0 / (1.0 + exp((score - 1.0) * fuzzy_width_down));
    val -= 1.0;  // from [1 .. 2] to [0 .. 1]
    val *= 2.0 - scaler;  // from [0 .. 1] to [0 .. 2.0 - scaler]
    val += scaler;  // from [0 .. 2.0 - scaler] to [scaler .. 2.0]
  } else {
    // val in [0 .. scaler]
    val = m0 / (1.0 + exp((score - 1.0) * fuzzy_width_up));
    val *= scaler;
  }
  return val;
}

double ButteraugliFuzzyInverse(double seek) {
  double pos = 0;
  for (double range = 1.0; range >= 1e-10; range *= 0.5) {
    double cur = ButteraugliFuzzyClass(pos);
    if (cur < seek) {
      pos -= range;
    } else {
      pos += range;
    }
  }
  return pos;
}

namespace {

void ScoreToRgb(double score, double good_threshold, double bad_threshold,
                uint8_t rgb[3]) {
  double heatmap[12][3] = {
      {0, 0, 0},
      {0, 0, 1},
      {0, 1, 1},
      {0, 1, 0},  // Good level
      {1, 1, 0},
      {1, 0, 0},  // Bad level
      {1, 0, 1},
      {0.5, 0.5, 1.0},
      {1.0, 0.5, 0.5},  // Pastel colors for the very bad quality range.
      {1.0, 1.0, 0.5},
      {
          1, 1, 1,
      },
      {
          1, 1, 1,
      },
  };
  if (score < good_threshold) {
    score = (score / good_threshold) * 0.3;
  } else if (score < bad_threshold) {
    score = 0.3 +
            (score - good_threshold) / (bad_threshold - good_threshold) * 0.15;
  } else {
    score = 0.45 + (score - bad_threshold) / (bad_threshold * 12) * 0.5;
  }
  static const int kTableSize = sizeof(heatmap) / sizeof(heatmap[0]);
  score = std::min<double>(std::max<double>(score * (kTableSize - 1), 0.0),
                           kTableSize - 2);
  int ix = static_cast<int>(score);
  double mix = score - ix;
  for (int i = 0; i < 3; ++i) {
    double v = mix * heatmap[ix + 1][i] + (1 - mix) * heatmap[ix][i];
    rgb[i] = static_cast<uint8_t>(255 * pow(v, 0.5) + 0.5);
  }
}

}  // namespace

void CreateHeatMapImage(const std::vector<float>& distmap,
                        double good_threshold, double bad_threshold,
                        size_t xsize, size_t ysize,
                        std::vector<uint8_t>* heatmap) {
  heatmap->resize(3 * xsize * ysize);
  for (size_t y = 0; y < ysize; ++y) {
    for (size_t x = 0; x < xsize; ++x) {
      int px = xsize * y + x;
      double d = distmap[px];
      uint8_t* rgb = &(*heatmap)[3 * px];
      ScoreToRgb(d, good_threshold, bad_threshold, rgb);
    }
  }
}

}  // namespace butteraugli
}  // namespace pik
