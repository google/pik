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

#include "compressed_image.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <array>
#include <atomic>

#define PROFILER_ENABLED 1
#include "common.h"
#include "compiler_specific.h"
#include "convolve.h"
#include "dc_predictor.h"
#include "dct_util.h"
#include "deconvolve.h"
#include "entropy_coder.h"
#include "fields.h"
#include "gauss_blur.h"
#include "huffman_decode.h"
#include "huffman_encode.h"
#include "opsin_inverse.h"
#include "opsin_params.h"
#include "profiler.h"
#include "resample.h"
#include "simd/simd.h"
#include "status.h"
#include "upscaler.h"

namespace pik {

namespace {

/* DCT coefficients (kBlockSize real numbers) are unrolled in x-direction
   to improve locality. */
constexpr ImageSize kDctTileSize{
    kTileWidthInBlocks * kBlockSize, kTileHeightInBlocks};

const double kDCBlurSigma = 5.5;

}  // namespace

Image3F AlignImage(const Image3F& in, const size_t N) {
  PROFILER_FUNC;
  const size_t xsize_blocks = DivCeil(in.xsize(), N);
  const size_t ysize_blocks = DivCeil(in.ysize(), N);
  const size_t xsize = N * xsize_blocks;
  const size_t ysize = N * ysize_blocks;
  Image3F out(xsize, ysize);
  for (int c = 0; c < 3; ++c) {
    int y = 0;
    for (; y < in.ysize(); ++y) {
      const float* PIK_RESTRICT row_in = in.ConstPlaneRow(c, y);
      float* PIK_RESTRICT row_out = out.PlaneRow(c, y);
      memcpy(row_out, row_in, in.xsize() * sizeof(row_in[0]));
      const int lastcol = in.xsize() - 1;
      const float lastval = row_out[lastcol];
      for (int x = in.xsize(); x < xsize; ++x) {
        row_out[x] = lastval;
      }
    }

    // TODO(janwas): no need to copy if we can 'extend' image: if rows are
    // pointers to any memory? Or allocate larger image before IO?
    const int lastrow = in.ysize() - 1;
    for (; y < ysize; ++y) {
      const float* PIK_RESTRICT row_in = out.ConstPlaneRow(c, lastrow);
      float* PIK_RESTRICT row_out = out.PlaneRow(c, y);
      memcpy(row_out, row_in, xsize * sizeof(row_out[0]));
    }
  }
  return out;
}

void CenterOpsinValues(Image3F* PIK_RESTRICT img) {
  PROFILER_FUNC;
  const size_t xsize = img->xsize();
  const size_t ysize = img->ysize();
  for (int c = 0; c < 3; ++c) {
    for (size_t y = 0; y < ysize; ++y) {
      float* PIK_RESTRICT row = img->PlaneRow(c, y);
      for (size_t x = 0; x < xsize; ++x) {
        row[x] -= kXybCenter[c];
      }
    }
  }
}

// "y_plane" may refer to plane#1 of "coeffs"; it is also organized in the
// block layout (64 consecutive block coefficient `pixels').
void ApplyColorTransform(const ColorTransform& ctan, const float factor,
                         const ImageF& y_plane, Image3F* coeffs) {
  const float kYToBScale = 1.0f / 128.0f;
  const float kYToXScale = 1.0f / 256.0f;
  const float factor_b = factor * kYToBScale;
  const float factor_x = factor * kYToXScale;

  const int bxsize = coeffs->xsize() / kBlockSize;
  const int bysize = coeffs->ysize();
  for (int y = 0; y < bysize; ++y) {
    const float* PIK_RESTRICT row_y = y_plane.Row(y);
    float* PIK_RESTRICT row_x = coeffs->PlaneRow(0, y);
    float* PIK_RESTRICT row_b = coeffs->PlaneRow(2, y);
    const int* PIK_RESTRICT row_ytob =
        ctan.ytob_map.Row(y / kTileHeightInBlocks);
    const int* PIK_RESTRICT row_ytox =
        ctan.ytox_map.Row(y / kTileHeightInBlocks);

    for (int x = 0; x < bxsize; ++x) {
      const int xoff = x * kBlockSize;
      const float ytob_ac = factor_b * row_ytob[x / kTileWidthInBlocks];
      const float ytox_ac = factor_x * (row_ytox[x / kTileWidthInBlocks] - 128);
      row_b[xoff] += factor_b * ctan.ytob_dc * row_y[xoff];
      row_x[xoff] += factor_x * (ctan.ytox_dc - 128) * row_y[xoff];
      for (int k = 1; k < kBlockSize; ++k) {
        row_b[xoff + k] += ytob_ac * row_y[xoff + k];
        row_x[xoff + k] += ytox_ac * row_y[xoff + k];
      }
    }
  }
}

constexpr float kColorFactorX = 1.0f / 256.0f;
constexpr float kColorFactorB = 1.0f / 128.0f;

// DC component is invalid (use separate DC image instead).
TFNode* AddColorTransform(const ColorTransform& ctan, TFNode* dq_xyb,
                          TFNode* ytox_ac, TFNode* ytob_ac,
                          TFBuilder* builder) {
  PIK_CHECK(OutType(dq_xyb) == TFType::kF32);
  PIK_CHECK(OutType(ytox_ac) == TFType::kI32);
  PIK_CHECK(OutType(ytob_ac) == TFType::kI32);
  return builder->AddClosure(
      "ctan", Borders(), Scale(), {dq_xyb, ytox_ac, ytob_ac}, 3, TFType::kF32,
      [](const ConstImageViewF* PIK_RESTRICT in,
         const OutputRegion& output_region,
         const MutableImageViewF* PIK_RESTRICT out) {
        using namespace SIMD_NAMESPACE;
        const Full<float> d;

        // yto* maps are downsampled 8x (kTile(Width|Height)InBlocks),
        // so we have only one yto* value for the entire tile.
        const int* PIK_RESTRICT row_ytox_ac =
            reinterpret_cast<const int*>(in[3].ConstRow(0));
        const int* PIK_RESTRICT row_ytob_ac =
            reinterpret_cast<const int*>(in[4].ConstRow(0));
        const auto ytox = set1(d, kColorFactorX * (row_ytox_ac[0] - 128));
        const auto ytob = set1(d, kColorFactorB * row_ytob_ac[0]);

        for (uint32_t y = 0; y < output_region.ysize; ++y) {
          const float* PIK_RESTRICT row_in_x = in[0].ConstRow(y);
          const float* PIK_RESTRICT row_in_y = in[1].ConstRow(y);
          const float* PIK_RESTRICT row_in_b = in[2].ConstRow(y);
          float* PIK_RESTRICT row_out_x = out[0].Row(y);
          float* PIK_RESTRICT row_out_y = out[1].Row(y);
          float* PIK_RESTRICT row_out_b = out[2].Row(y);

          for (uint32_t x = 0; x < output_region.xsize; x += kBlockSize) {
            // Single loop is faster than separate x/b loops (inner or outer).
            for (size_t k = 0; k < kBlockSize; k += d.N) {
              const auto in_x = load(d, row_in_x + x + k);
              const auto in_y = load(d, row_in_y + x + k);
              const auto in_b = load(d, row_in_b + x + k);
              const auto out_x = mul_add(ytox, in_y, in_x);
              const auto out_b = mul_add(ytob, in_y, in_b);
              store(out_x, d, row_out_x + x + k);
              store(in_y, d, row_out_y + x + k);
              store(out_b, d, row_out_b + x + k);
            }
          }
        }
      });
}

TFNode* AddColorTransformDC(const ColorTransform& ctan, TFNode* dq_xyb,
                            TFBuilder* builder) {
  PIK_CHECK(OutType(dq_xyb) == TFType::kF32);
  const float ytox_dc = (ctan.ytox_dc - 128) * kColorFactorX;
  const float ytob_dc = ctan.ytob_dc * kColorFactorB;
  return builder->AddClosure(
      "ctan_dc", Borders(), Scale(), {dq_xyb}, 3, TFType::kF32,
      // ytox_dc already has 128 subtracted; both are pre-multiplied by
      // kColorFactor*.
      [ytox_dc, ytob_dc](const ConstImageViewF* PIK_RESTRICT in,
                         const OutputRegion& output_region,
                         const MutableImageViewF* PIK_RESTRICT out) {
        using namespace SIMD_NAMESPACE;
        const Full<float> d;

        const auto ytox = set1(d, ytox_dc);
        const auto ytob = set1(d, ytob_dc);

        for (uint32_t y = 0; y < output_region.ysize; ++y) {
          const float* PIK_RESTRICT row_in_x = in[0].ConstRow(y);
          const float* PIK_RESTRICT row_in_y = in[1].ConstRow(y);
          const float* PIK_RESTRICT row_in_b = in[2].ConstRow(y);
          float* PIK_RESTRICT row_out_x = out[0].Row(y);
          float* PIK_RESTRICT row_out_y = out[1].Row(y);
          float* PIK_RESTRICT row_out_b = out[2].Row(y);

          for (uint32_t x = 0; x < output_region.xsize; x += d.N) {
            const auto in_x = load(d, row_in_x + x);
            const auto in_y = load(d, row_in_y + x);
            const auto in_b = load(d, row_in_b + x);
            const auto out_x = mul_add(ytox, in_y, in_x);
            const auto out_b = mul_add(ytob, in_y, in_b);
            store(out_x, d, row_out_x + x);
            store(in_y, d, row_out_y + x);
            store(out_b, d, row_out_b + x);
          }
        }
      });
}

namespace kernel {

struct Gaborish3 {
  PIK_INLINE const Weights3x3& Weights() const {
    // Unnormalized.
    constexpr float wu0 = 1.0f;
    const float wu1 = static_cast<float>(0.11001538179658321);
    const float wu2 = static_cast<float>(0.089979079587015454);
    const float mul = 1.0 / (wu0 + 4 * (wu1 + wu2));
    const float w0 = wu0 * mul;
    const float w1 = wu1 * mul;
    const float w2 = wu2 * mul;
    static const Weights3x3 weights = {
        {PIK_REP4(w2)}, {PIK_REP4(w1)}, {PIK_REP4(w2)},
        {PIK_REP4(w1)}, {PIK_REP4(w0)}, {PIK_REP4(w1)},
        {PIK_REP4(w2)}, {PIK_REP4(w1)}, {PIK_REP4(w2)}};
    return weights;
  }
};

}  // namespace kernel

void GaborishInverse(Image3F &opsin) {
  PROFILER_FUNC;
  static const double kGaborish[5] = {
    -0.095346974121859995,
    -0.049147719433952346,
    0.00917293007962648,
    0.0031177055481802014,
    0.0063673837572018844,
  };
  const float smooth_weights5[9] = {
    1.0f,
    static_cast<float>(kGaborish[0]),
    static_cast<float>(kGaborish[2]),

    static_cast<float>(kGaborish[0]),
    static_cast<float>(kGaborish[1]),
    static_cast<float>(kGaborish[3]),

    static_cast<float>(kGaborish[2]),
    static_cast<float>(kGaborish[3]),
    static_cast<float>(kGaborish[4]),
  };
  ImageF res[3] = {ImageF(opsin.xsize(), opsin.ysize()),
                   ImageF(opsin.xsize(), opsin.ysize()),
                   ImageF(opsin.xsize(), opsin.ysize())};
  for (int i = 0; i < 3; ++i) {
    slow::SymmetricConvolution<2, WrapClamp>::Run(
        opsin.plane(i), opsin.xsize(), opsin.ysize(), smooth_weights5,
        &res[i]);
  }
  Image3F smooth(std::move(res[0]), std::move(res[1]), std::move(res[2]));
  smooth.Swap(opsin);
}

Image3F ConvolveGaborish(const Image3F& in, ThreadPool* pool) {
  PROFILER_ZONE("|| gaborish");
  Image3F out(in.xsize(), in.ysize());
  for (int c = 0; c < 3; ++c) {
    using namespace SIMD_NAMESPACE;
    ConvolveT<strategy::Symmetric3>::Run(BorderNeverUsed(), ExecutorPool(pool),
                                         in.plane(c), kernel::Gaborish3(),
                                         out.MutablePlane(c));
    out.CheckSizesSame();
  }
  return out;
}

Image3F ConvolveGaborishTF(const Image3F& in, ThreadPool* pool) {
  PROFILER_ZONE("|| gaborishTF");

  TFBuilder builder;
  TFNode* src = builder.AddSource("src", 3, TFType::kF32, TFWrap::kMirror);
  builder.SetSource(src, &in);

  TFNode* smoothed = builder.Add(
      "gaborish", Borders(1), Scale(), {src}, 3, TFType::kF32,
      [](const void*, const ConstImageViewF* in,
         const OutputRegion& output_region, const MutableImageViewF* out) {
        kernel::Gaborish3 kernel;
        for (int c = 0; c < 3; ++c) {
          using namespace SIMD_NAMESPACE;
          ConvolveT<strategy::Symmetric3>::RunView(
              in + c, kernel, BorderAlreadyValid(), output_region, out + c);
        }
      });

  Image3F out(in.xsize(), in.ysize());
  builder.SetSink(smoothed, &out);

  auto graph = builder.Finalize(ImageSize::Make(in.xsize(), in.ysize()),
                                ImageSize{512, 32}, pool);
  graph->Run();
  return out;
}

// Avoid including <functional> for such tiny functions.
struct Plus {
  constexpr float operator()(const float first, const float second) const {
    return first + second;
  }
};
struct Minus {
  constexpr float operator()(const float first, const float second) const {
    return first - second;
  }
};

// Adds/subtracts all AC coefficients to "out"; leaves DC unchanged.
template <class Operator>  // Plus/Minus
void InjectExceptDC(const Image3F& what, Image3F* out) {
  const size_t xsize = what.xsize();
  const size_t ysize = what.ysize();
  PIK_CHECK(xsize % kBlockSize == 0);
  const Operator op;

  for (int c = 0; c < 3; ++c) {
    for (size_t y = 0; y < ysize; ++y) {
      const float* PIK_RESTRICT row_what = what.ConstPlaneRow(c, y);
      float* PIK_RESTRICT row_out = out->PlaneRow(c, y);
      for (size_t bx = 0; bx < xsize; bx += kBlockSize) {
        for (size_t x = 1; x < kBlockSize; ++x) {
          row_out[bx + x] = op(row_out[bx + x], row_what[bx + x]);
        }
      }
    }
  }
}

template <typename T>
void ZeroDC(Image3<T>* img) {
  for (int c = 0; c < 3; c++) {
    for (size_t y = 0; y < img->ysize(); y++) {
      T* PIK_RESTRICT row = img->PlaneRow(c, y);
      for (size_t x = 0; x < img->xsize(); x += kBlockSize) {
        row[x] = T(0);
      }
    }
  }
}

std::vector<float> DCfiedGaussianKernel(float sigma) {
  std::vector<float> result(3, 0.0);
  std::vector<float> hires = GaussianKernel<float>(8, sigma);
  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < hires.size(); j++) {
      result[(i + j) / 8] += hires[j] / 8.0;
    }
  }
  return result;
}

// Called from local static ctor.
kernel::Custom<3> MakeUpsampleKernel() {
  Image3F impulse_dc(20, 20, 0.0);
  impulse_dc.PlaneRow(0, 10)[10] = 1.0;
  Image3F probe_expected = UpscalerReconstruct(impulse_dc);
  // We are trying to extract a kernel with a smaller radius. This kernel will
  // be unnormalized. However, we don't mind that: the encoder will compensate
  // when encoding.
  auto kernel6 = kernel::Custom<3>::FromResult(probe_expected.plane(0));

  ImageF probe_test(probe_expected.xsize(), probe_expected.ysize());
  Upsample<GeneralUpsampler8_6x6>(ExecutorLoop(), impulse_dc.plane(0), kernel6,
                                  &probe_test);
  VerifyRelativeError(probe_expected.plane(0), probe_test, 5e-2, 5e-2);

  return kernel6;
}

template <class Image>  // ImageF or Image3F
Image BlurUpsampleDC(const Image& original_dc, ThreadPool* pool) {
  const ExecutorPool executor(pool);
  Image out(original_dc.xsize() * 8, original_dc.ysize() * 8);
  // TODO(user): In the encoder we want only the DC of the result. That could
  // be done more quickly.
  static auto kernel6 = MakeUpsampleKernel();
  Upsample<GeneralUpsampler8_6x6>(executor, original_dc, kernel6, &out);
  return out;
}

// Returns DCT(blur) - original_dc
Image3F BlurUpsampleDCAndDCT(const Image3F& original_dc, ThreadPool* pool) {
  Image3F blurred = BlurUpsampleDC(original_dc, pool);
  Image3F dct = TransposedScaledDCT(blurred, pool);
  for (int c = 0; c < 3; c++) {
    for (size_t y = 0; y < dct.ysize(); y++) {
      float* PIK_RESTRICT dct_row = dct.PlaneRow(c, y);
      const float* PIK_RESTRICT original_dc_row =
          original_dc.ConstPlaneRow(c, y);
      for (size_t dct_x = 0, x = 0; dct_x < dct.xsize(); dct_x += kBlockSize) {
        dct_row[dct_x] -= original_dc_row[x++];
      }
    }
  }
  return dct;
}

// Called by local static ctor.
std::vector<float> MakeSharpenKernel() {
  // TODO(user): With the new blur, this makes no sense.
  std::vector<float> blur_kernel = DCfiedGaussianKernel(kDCBlurSigma);
  constexpr int kSharpenKernelSize = 3;
  std::vector<float> sharpen_kernel(kSharpenKernelSize);
  InvertConvolution(&blur_kernel[0], blur_kernel.size(), &sharpen_kernel[0],
                    sharpen_kernel.size());
  return sharpen_kernel;
}

// Returns true if L1(residual) < max_error.
bool AddResidualAndCompare(const ImageF& dc, const ImageF& blurred_dc,
                           const float max_error,
                           ImageF* PIK_RESTRICT dc_to_encode) {
  PIK_CHECK(SameSize(dc, blurred_dc) && SameSize(dc, *dc_to_encode));
  const size_t xsize = dc.xsize();
  const size_t ysize = dc.ysize();
  bool all_less = true;
  for (size_t y = 0; y < ysize; ++y) {
    const float* PIK_RESTRICT row_dc = dc.ConstRow(y);
    const float* PIK_RESTRICT row_blurred = blurred_dc.ConstRow(y);
    float* PIK_RESTRICT row_out = dc_to_encode->Row(y);
    for (size_t x = 0; x < xsize; ++x) {
      const float diff = row_dc[x] - row_blurred[x];
      all_less &= fabsf(diff) < max_error;
      row_out[x] += diff;
    }
  }
  return all_less;
}

Image3F SharpenDC(const Image3F& original_dc, ThreadPool* pool) {
  PROFILER_FUNC;
  constexpr int kMaxIters = 200;
  constexpr float kAcceptableError[3] = {1e-4, 1e-4, 1e-4};

  static const std::vector<float> sharpen_kernel = MakeSharpenKernel();
  Image3F dc_to_encode = Convolve(original_dc, sharpen_kernel);

  // Individually per channel, until error is acceptable:
  for (int c = 0; c < 3; ++c) {
    for (int iter = 0; iter < kMaxIters; iter++) {
      const ImageF up = BlurUpsampleDC(dc_to_encode.plane(c), pool);
      const ImageF blurred = Subsample8(up);
      // Change pixels of dc_to_encode but not its size.
      if (AddResidualAndCompare(original_dc.plane(c), blurred,
                                kAcceptableError[c],
                                dc_to_encode.MutablePlane(c))) {
        break;  // next channel
      }
    }
    dc_to_encode.CheckSizesSame();
  }

  return dc_to_encode;
}

void ComputePredictionResiduals_Smooth(const Quantizer& quantizer,
                                       ThreadPool* pool, EncCache* cache) {
  PROFILER_FUNC;
  if (!cache->have_pred) {
    const Image3F dc_orig = DCImage(cache->coeffs_init);
    cache->dc_sharp = SharpenDC(dc_orig, pool);
    const Image3F dc_dec = QuantizeRoundtripDC(quantizer, cache->dc_sharp);
    cache->pred_smooth = BlurUpsampleDCAndDCT(dc_dec, pool);
    cache->have_pred = true;
  }

  cache->coeffs = CopyImage(cache->coeffs_init);
  FillDC(cache->dc_sharp, &cache->coeffs);

  // We have already tried to take into account the effect on DC. We will
  // assume here that we've done that correctly.
  InjectExceptDC<Minus>(cache->pred_smooth, &cache->coeffs);
}

namespace kernel {

constexpr float kWeight0 = 0.027630534023046f;
constexpr float kWeight1 = 0.133676439523697f;
constexpr float kWeight2 = 0.035697385668755f;

struct AC11 {
  PIK_INLINE const Weights3x3& Weights() const {
    static constexpr Weights3x3 weights = {
        {PIK_REP4(kWeight2)},  {PIK_REP4(0.)}, {PIK_REP4(-kWeight2)},
        {PIK_REP4(0.)},        {PIK_REP4(0.)}, {PIK_REP4(0.)},
        {PIK_REP4(-kWeight2)}, {PIK_REP4(0.)}, {PIK_REP4(kWeight2)}};
    return weights;
  }
};

struct AC01 {
  PIK_INLINE const Weights3x3& Weights() const {
    static constexpr Weights3x3 weights = {
        {PIK_REP4(kWeight0)},  {PIK_REP4(kWeight1)},  {PIK_REP4(kWeight0)},
        {PIK_REP4(0.)},        {PIK_REP4(0.)},        {PIK_REP4(0.)},
        {PIK_REP4(-kWeight0)}, {PIK_REP4(-kWeight1)}, {PIK_REP4(-kWeight0)}};
    return weights;
  }
};

struct AC10 {
  PIK_INLINE const Weights3x3& Weights() const {
    static constexpr Weights3x3 weights = {
        {PIK_REP4(kWeight0)}, {PIK_REP4(0.)}, {PIK_REP4(-kWeight0)},
        {PIK_REP4(kWeight1)}, {PIK_REP4(0.)}, {PIK_REP4(-kWeight1)},
        {PIK_REP4(kWeight0)}, {PIK_REP4(0.)}, {PIK_REP4(-kWeight0)}};
    return weights;
  }
};

}  // namespace kernel

template <class Operator>  // Plus/Minus
void Adjust189_64FromDC(const Image3F& dc, Image3F* coeffs) {
  const size_t xsize = dc.xsize();
  const size_t ysize = dc.ysize();

  const Operator op;

  Image3F ac01(xsize, ysize);
  Image3F ac10(xsize, ysize);
  Image3F ac11(xsize, ysize);
  // Must avoid ConvolveT for tiny images (PIK_CHECK fails)
  if (xsize < kConvolveMinWidth) {
    using Convolution = slow::General3x3Convolution<1, WrapMirror>;
    Convolution::Run(dc, xsize, ysize, kernel::AC01(), &ac01);
    Convolution::Run(dc, xsize, ysize, kernel::AC10(), &ac10);
    Convolution::Run(dc, xsize, ysize, kernel::AC11(), &ac11);
  } else {
    ConvolveT<strategy::GradY3>::Run(dc, kernel::AC01(), &ac01);
    ConvolveT<strategy::GradX3>::Run(dc, kernel::AC10(), &ac10);
    ConvolveT<strategy::Corner3>::Run(dc, kernel::AC11(), &ac11);
  }
  for (int c = 0; c < 3; ++c) {
    for (size_t by = 0; by < ysize; ++by) {
      const float* PIK_RESTRICT row01 = ac01.ConstPlaneRow(c, by);
      const float* PIK_RESTRICT row10 = ac10.ConstPlaneRow(c, by);
      const float* PIK_RESTRICT row11 = ac11.ConstPlaneRow(c, by);
      float* SIMD_RESTRICT row_out = coeffs->PlaneRow(c, by);
      for (size_t bx = 0; bx < xsize; ++bx) {
        float* PIK_RESTRICT block = row_out + kBlockSize * bx;
        block[1] = op(block[1], row01[bx]);
        block[8] = op(block[8], row10[bx]);
        block[9] = op(block[9], row11[bx]);
      }
    }
  }
}

// Returns pixel-space prediction using same adjustment as above followed by
// GetPixelSpaceImageFrom0189. "ac" is [--, 1, 8, 9].
Image3F PredictSpatial2x2_AC4(const Image3F& dc, const Image3F& ac) {
  const size_t xsize = dc.xsize();
  const size_t ysize = dc.ysize();

  Image3F ac01(xsize, ysize);
  Image3F ac10(xsize, ysize);
  Image3F ac11(xsize, ysize);
  // Must avoid ConvolveT for tiny images (PIK_CHECK fails)
  if (xsize < kConvolveMinWidth) {
    using Convolution = slow::General3x3Convolution<1, WrapMirror>;
    Convolution::Run(dc, xsize, ysize, kernel::AC01(), &ac01);
    Convolution::Run(dc, xsize, ysize, kernel::AC10(), &ac10);
    Convolution::Run(dc, xsize, ysize, kernel::AC11(), &ac11);
  } else {
    ConvolveT<strategy::GradY3>::Run(dc, kernel::AC01(), &ac01);
    ConvolveT<strategy::GradX3>::Run(dc, kernel::AC10(), &ac10);
    ConvolveT<strategy::Corner3>::Run(dc, kernel::AC11(), &ac11);
  }

  const float kScale01 = 0.113265930794111f / (kIDCTScales[0] * kIDCTScales[1]);
  const float kScale11 = 0.102633368629251f / (kIDCTScales[1] * kIDCTScales[1]);
  Image3F out2x2(xsize * 2, ysize * 2);
  for (int c = 0; c < 3; ++c) {
    for (size_t by = 0; by < ysize; ++by) {
      const float* PIK_RESTRICT row_dc = dc.ConstPlaneRow(c, by);
      const float* PIK_RESTRICT row_ac = ac.ConstPlaneRow(c, by);
      const float* PIK_RESTRICT row01 = ac01.ConstPlaneRow(c, by);
      const float* PIK_RESTRICT row10 = ac10.ConstPlaneRow(c, by);
      const float* PIK_RESTRICT row11 = ac11.ConstPlaneRow(c, by);
      float* SIMD_RESTRICT row_out0 = out2x2.PlaneRow(c, by * 2 + 0);
      float* SIMD_RESTRICT row_out1 = out2x2.PlaneRow(c, by * 2 + 1);
      for (size_t bx = 0; bx < xsize; ++bx) {
        const float* PIK_RESTRICT block_ac = row_ac + bx * 4;
        const float a00 = row_dc[bx];
        const float a01 = (block_ac[2] + row10[bx]) * kScale01;
        const float a10 = (block_ac[1] + row01[bx]) * kScale01;
        const float a11 = (block_ac[3] + row11[bx]) * kScale11;
        row_out0[2 * bx + 0] = a00 + a01 + a10 + a11;
        row_out0[2 * bx + 1] = a00 - a01 + a10 - a11;
        row_out1[2 * bx + 0] = a00 + a01 - a10 - a11;
        row_out1[2 * bx + 1] = a00 - a01 - a10 + a11;
      }
    }
  }

  return out2x2;
}

// Returns pixel-space prediction using same adjustment as above followed by
// GetPixelSpaceImageFrom0189. Modifies "ac64" 0189.
Image3F PredictSpatial2x2_AC64(const Image3F& dc, Image3F* ac64) {
  const size_t xsize = dc.xsize();
  const size_t ysize = dc.ysize();

  Image3F ac01(xsize, ysize);
  Image3F ac10(xsize, ysize);
  Image3F ac11(xsize, ysize);
  // Must avoid ConvolveT for tiny images (PIK_CHECK fails)
  if (xsize < kConvolveMinWidth) {
    using Convolution = slow::General3x3Convolution<1, WrapMirror>;
    Convolution::Run(dc, xsize, ysize, kernel::AC01(), &ac01);
    Convolution::Run(dc, xsize, ysize, kernel::AC10(), &ac10);
    Convolution::Run(dc, xsize, ysize, kernel::AC11(), &ac11);
  } else {
    ConvolveT<strategy::GradY3>::Run(dc, kernel::AC01(), &ac01);
    ConvolveT<strategy::GradX3>::Run(dc, kernel::AC10(), &ac10);
    ConvolveT<strategy::Corner3>::Run(dc, kernel::AC11(), &ac11);
  }

  const float kScale01 = 0.113265930794111f / (kIDCTScales[0] * kIDCTScales[1]);
  const float kScale11 = 0.102633368629251f / (kIDCTScales[1] * kIDCTScales[1]);
  Image3F out2x2(xsize * 2, ysize * 2);
  for (int c = 0; c < 3; ++c) {
    for (size_t by = 0; by < ysize; ++by) {
      const float* PIK_RESTRICT row_dc = dc.ConstPlaneRow(c, by);
      float* PIK_RESTRICT row_ac64 = ac64->PlaneRow(c, by);
      const float* PIK_RESTRICT row01 = ac01.ConstPlaneRow(c, by);
      const float* PIK_RESTRICT row10 = ac10.ConstPlaneRow(c, by);
      const float* PIK_RESTRICT row11 = ac11.ConstPlaneRow(c, by);
      float* SIMD_RESTRICT row_out0 = out2x2.PlaneRow(c, by * 2 + 0);
      float* SIMD_RESTRICT row_out1 = out2x2.PlaneRow(c, by * 2 + 1);
      for (size_t bx = 0; bx < xsize; ++bx) {
        float* PIK_RESTRICT block_ac = row_ac64 + bx * 64;
        block_ac[0] = row_dc[bx];
        block_ac[1] += row01[bx];
        block_ac[8] += row10[bx];
        block_ac[9] += row11[bx];

        const float a00 = block_ac[0];
        const float a01 = block_ac[8] * kScale01;
        const float a10 = block_ac[1] * kScale01;
        const float a11 = block_ac[9] * kScale11;
        row_out0[2 * bx + 0] = a00 + a01 + a10 + a11;
        row_out0[2 * bx + 1] = a00 - a01 + a10 - a11;
        row_out1[2 * bx + 0] = a00 + a01 - a10 - a11;
        row_out1[2 * bx + 1] = a00 - a01 - a10 + a11;
      }
    }
  }

  return out2x2;
}

void ComputePredictionResiduals(const Quantizer& quantizer, ThreadPool* pool,
                                EncCache* cache) {
  if (!cache->have_pred) {
    cache->dc_dec = QuantizeRoundtripExtractDC(quantizer, cache->coeffs_init);
    Adjust189_64FromDC<Minus>(cache->dc_dec, &cache->coeffs_init);
    cache->have_pred = true;
  }
  cache->coeffs = CopyImage(cache->coeffs_init);
  const Image3F ac189_rounded =
      QuantizeRoundtripExtract189(quantizer, cache->coeffs);
  Image3F pred2x2 = PredictSpatial2x2_AC4(cache->dc_dec, ac189_rounded);
  UpSample4x4BlurDCT(pred2x2, 1.5f, -0.0f, pool, &cache->coeffs);
}

QuantizedCoeffs ComputeCoefficients(const CompressParams& params,
                                    const Header& header, const Image3F& opsin,
                                    const Quantizer& quantizer,
                                    const ColorTransform& ctan,
                                    ThreadPool* pool, EncCache* cache,
                                    const PikInfo* aux_out) {
  if (!cache->have_coeffs_init) {
    cache->coeffs_init = TransposedScaledDCT(opsin, pool);
    cache->have_coeffs_init = true;
  }

  if (header.flags & Header::kSmoothDCPred) {
    ComputePredictionResiduals_Smooth(quantizer, pool, cache);
  } else {
    ComputePredictionResiduals(quantizer, pool, cache);
  }

  PROFILER_ZONE("enc ctan+quant");
  ApplyColorTransform(ctan, -1.0f,
                      QuantizeRoundtrip(quantizer, 1, cache->coeffs.plane(1)),
                      &cache->coeffs);

  QuantizedCoeffs qcoeffs;
  qcoeffs.dc = QuantizeCoeffsDC(cache->coeffs, quantizer);
  qcoeffs.ac = QuantizeCoeffs(cache->coeffs, quantizer);
  return qcoeffs;
}

void ComputeBlockContextFromDC(const Image3S& dc, const Quantizer& quantizer,
                               Image3B* out) {
  PROFILER_FUNC;
  // One DC per block.
  const int bxsize = dc.xsize();
  const int bysize = dc.ysize();
  const float iquant_base = quantizer.inv_quant_dc();
  for (int c = 0; c < 3; ++c) {
    const float iquant =
        iquant_base * quantizer.DequantMatrix()[c * kBlockSize];
    const float range = kXybRange[c] / iquant;
    const int64_t kR2Thresh = std::min(10.24f * range * range + 1.0f, 1E18f);
    memset(out->PlaneRow(c, 0), c, bxsize);
    memset(out->PlaneRow(c, bysize - 1), c, bxsize);
    for (int y = 1; y + 1 < bysize; ++y) {
      const int16_t* PIK_RESTRICT row_t = dc.ConstPlaneRow(c, y - 1);
      const int16_t* PIK_RESTRICT row_m = dc.ConstPlaneRow(c, y);
      const int16_t* PIK_RESTRICT row_b = dc.ConstPlaneRow(c, y + 1);
      uint8_t* PIK_RESTRICT row_out = out->PlaneRow(c, y);
      row_out[0] = row_out[bxsize - 1] = c;
      for (int bx = 1; bx + 1 < bxsize; ++bx) {
        const int16_t val_tl = row_t[bx - 1];
        const int16_t val_tm = row_t[bx];
        const int16_t val_tr = row_t[bx + 1];
        const int16_t val_ml = row_m[bx - 1];
        const int16_t val_mr = row_m[bx + 1];
        const int16_t val_bl = row_b[bx - 1];
        const int16_t val_bm = row_b[bx];
        const int16_t val_br = row_b[bx + 1];
        const int64_t dx =
            (3 * (val_tr - val_tl + val_br - val_bl) + 10 * (val_mr - val_ml));
        const int64_t dy =
            (3 * (val_bl - val_tl + val_br - val_tr) + 10 * (val_bm - val_tm));
        const int64_t dx2 = dx * dx;
        const int64_t dy2 = dy * dy;
        const int64_t dxdy = std::abs(2 * dx * dy);
        const int64_t r2 = dx2 + dy2;
        const int64_t d2 = dy2 - dx2;
        if (r2 < kR2Thresh) {
          row_out[bx] = c;
        } else if (d2 < -dxdy) {
          row_out[bx] = 3;
        } else if (d2 > dxdy) {
          row_out[bx] = 5;
        } else {
          row_out[bx] = 4;
        }
      }
    }
  }
}

std::string EncodeColorMap(const Image<int>& ac_map, const int dc_val,
                           PikImageSizeInfo* info) {
  const size_t max_out_size = ac_map.xsize() * ac_map.ysize() + 1024;
  std::string output(max_out_size, 0);
  size_t storage_ix = 0;
  uint8_t* storage = reinterpret_cast<uint8_t*>(&output[0]);
  storage[0] = 0;
  std::vector<uint32_t> histogram(256);
  ++histogram[dc_val];
  for (int y = 0; y < ac_map.ysize(); ++y) {
    for (int x = 0; x < ac_map.xsize(); ++x) {
      ++histogram[ac_map.Row(y)[x]];
    }
  }
  std::vector<uint8_t> bit_depths(256);
  std::vector<uint16_t> bit_codes(256);
  BuildAndStoreHuffmanTree(histogram.data(), histogram.size(),
                           bit_depths.data(), bit_codes.data(), &storage_ix,
                           storage);
  const size_t histo_bits = storage_ix;
  WriteBits(bit_depths[dc_val], bit_codes[dc_val], &storage_ix, storage);
  for (int y = 0; y < ac_map.ysize(); ++y) {
    const int* PIK_RESTRICT row = ac_map.Row(y);
    for (int x = 0; x < ac_map.xsize(); ++x) {
      WriteBits(bit_depths[row[x]], bit_codes[row[x]], &storage_ix, storage);
    }
  }
  WriteZeroesToByteBoundary(&storage_ix, storage);
  PIK_ASSERT((storage_ix >> 3) <= output.size());
  output.resize(storage_ix >> 3);
  if (info) {
    info->histogram_size += histo_bits >> 3;
    info->entropy_coded_bits += storage_ix - histo_bits;
    info->total_size += output.size();
  }
  return output;
}

namespace {

template <typename ImageT>
ImageT DcTile(ImageT* from, size_t tile_x, size_t tile_y) {
  return Window(from,
      tile_x * kGroupWidthInBlocks, tile_y * kGroupHeightInBlocks,
      kGroupWidthInBlocks, kGroupHeightInBlocks);
}

template <typename ImageT>
ImageT Tile(ImageT* from, size_t tile_x, size_t tile_y) {
  return Window(from, tile_x * kGroupWidthInBlocks * kBlockSize,
                tile_y * kGroupHeightInBlocks,
                kGroupWidthInBlocks * kBlockSize, kGroupHeightInBlocks);
}

template <typename ImageT>
ConstWrapper<ImageT> ConstDcTile(const ImageT& from,
                                 size_t tile_x, size_t tile_y) {
  return ConstWindow(from, tile_x * kGroupWidthInBlocks,
                     tile_y * kGroupHeightInBlocks,
                     kGroupWidthInBlocks, kGroupHeightInBlocks);
}

template <typename ImageT>
ConstWrapper<ImageT > ConstTile(const ImageT& from,
                                size_t tile_x, size_t tile_y) {
  return ConstWindow(from, tile_x * kGroupWidthInBlocks * kBlockSize,
                     tile_y * kGroupHeightInBlocks,
                     kGroupWidthInBlocks * kBlockSize, kGroupHeightInBlocks);
}

template <uint32_t kDistribution>
class SizeCoderT {
 public:
  static size_t MaxSize(const size_t num_sizes) {
    // 8 extra bytes for WriteBits.
    return DivCeil(U32Coder::MaxEncodedBits(kDistribution) * num_sizes, 8) + 8;
  }

  static void Encode(const size_t size, size_t* PIK_RESTRICT pos,
                     uint8_t* storage) {
    PIK_CHECK(U32Coder::Store(kDistribution, size, pos, storage));
  }

  static size_t Decode(BitReader* reader) {
    return U32Coder::Load(kDistribution, reader);
  }
};

// Quartiles of distribution observed from benchmark.
using DcTileSizeCoder = SizeCoderT<0x0E0C0A09>;  // max observed: 8K
using AcTileSizeCoder = SizeCoderT<0x120F0E0C>;  // max observed: 142K

static inline void Append(const std::string& s, PaddedBytes* out,
                          size_t* PIK_RESTRICT byte_pos) {
  memcpy(out->data() + *byte_pos, s.data(), s.length());
  *byte_pos += s.length();
  PIK_CHECK(*byte_pos <= out->size());
}

}  // namespace

PaddedBytes EncodeToBitstream(const QuantizedCoeffs& qcoeffs,
                              const Quantizer& quantizer,
                              const NoiseParams& noise_params,
                              const ColorTransform& ctan, bool fast_mode,
                              PikInfo* info) {
  PROFILER_FUNC;
  const size_t xsize_blocks = qcoeffs.dc.xsize();
  const size_t ysize_blocks = qcoeffs.dc.ysize();
  const size_t xsize_group = DivCeil(xsize_blocks, kGroupWidthInBlocks);
  const size_t ysize_group = DivCeil(ysize_blocks, kGroupHeightInBlocks);
  const size_t num_group = xsize_group * ysize_group;
  PikImageSizeInfo* ctan_info = info ? &info->layers[kLayerCtan] : nullptr;
  std::string ctan_code =
      EncodeColorMap(ctan.ytob_map, ctan.ytob_dc, ctan_info) +
      EncodeColorMap(ctan.ytox_map, ctan.ytox_dc, ctan_info);
  PikImageSizeInfo* quant_info = info ? &info->layers[kLayerQuant] : nullptr;
  PikImageSizeInfo* dc_info = info ? &info->layers[kLayerDC] : nullptr;
  PikImageSizeInfo* ac_info = info ? &info->layers[kLayerAC] : nullptr;
  std::string noise_code = EncodeNoise(noise_params);
  std::string quant_code = quantizer.Encode(quant_info);

  std::string dc_code = "";
  std::string dc_toc(DcTileSizeCoder::MaxSize(num_group), '\0');
  size_t dc_toc_pos = 0;
  uint8_t* dc_toc_storage =
      reinterpret_cast<uint8_t*>(const_cast<char*>(dc_toc.data()));

  Image3S predicted_dc(xsize_blocks, ysize_blocks);
  Image3B block_ctx(xsize_blocks, ysize_blocks);
  for (size_t y = 0; y < ysize_group; y++) {
    for (size_t x = 0; x < xsize_group; x++) {
      Image3S predicted_dc_tile = DcTile(&predicted_dc, x, y);
      Image3B block_ctx_tile = DcTile(&block_ctx, x, y);
      ConstWrapper<Image3S> dc_tile = ConstDcTile(qcoeffs.dc, x, y);
      PredictDCTile(dc_tile.get(), &predicted_dc_tile);
      ComputeBlockContextFromDC(dc_tile.get(), quantizer, &block_ctx_tile);

      const std::string& tile_code = EncodeImage(predicted_dc_tile, 1, dc_info);
      DcTileSizeCoder::Encode(tile_code.size(), &dc_toc_pos, dc_toc_storage);
      dc_code += tile_code;
    }
  }
  WriteZeroesToByteBoundary(&dc_toc_pos, dc_toc_storage);
  dc_toc.resize(dc_toc_pos / 8);

  std::string ac_code = "";
  std::string order_code = "";
  std::string histo_code = "";
  int order[kOrderContexts * kBlockSize];
  std::vector<ANSEncodingData> codes;
  std::vector<uint8_t> context_map;
  std::vector<std::vector<Token> > all_tokens;
  if (fast_mode) {
    for (int i = 0; i < kOrderContexts; ++i) {
      memcpy(&order[i * kBlockSize], kNaturalCoeffOrder,
          kBlockSize * sizeof(order[0]));
    }
  } else {
    ComputeCoeffOrder(qcoeffs.ac, block_ctx, order);
  }

  order_code = EncodeCoeffOrders(order, info);
  const ImageI& quant_ac = quantizer.RawQuantField();
  for (size_t y = 0; y < ysize_group; y++) {
    for (size_t x = 0; x < xsize_group; x++) {
      ConstWrapper<ImageI> q_tile = ConstDcTile(quant_ac, x, y);
      ConstWrapper<Image3S> qac_tile = ConstTile(qcoeffs.ac, x, y);
      ConstWrapper<Image3B> block_ctx_tile = ConstDcTile(block_ctx, x, y);
      // WARNING: TokenizeCoefficients uses the DC values in qcoeffs.ac!
      all_tokens.emplace_back(TokenizeCoefficients(
          order, q_tile.get(), qac_tile.get(), block_ctx_tile.get()));
    }
  }
  if (fast_mode) {
    histo_code = BuildAndEncodeHistogramsFast(all_tokens,
                                              &codes, &context_map,
                                              ac_info);
  } else {
    histo_code = BuildAndEncodeHistograms(kNumContexts, all_tokens,
                                          &codes, &context_map,
                                          ac_info);
  }

  std::string ac_toc(AcTileSizeCoder::MaxSize(all_tokens.size()), '\0');
  size_t ac_toc_pos = 0;
  uint8_t* ac_toc_storage =
      reinterpret_cast<uint8_t*>(const_cast<char*>(ac_toc.data()));

  for (int i = 0; i < all_tokens.size(); ++i) {
    std::string ac_tile_code =
        WriteTokens(all_tokens[i], codes, context_map, ac_info);
    AcTileSizeCoder::Encode(ac_tile_code.size(), &ac_toc_pos, ac_toc_storage);
    ac_code += ac_tile_code;
  }
  WriteZeroesToByteBoundary(&ac_toc_pos, ac_toc_storage);
  ac_toc.resize(ac_toc_pos / 8);

  if (info) {
    info->layers[kLayerHeader].total_size +=
        noise_code.size() + dc_toc.size() + ac_toc.size();
  }

  PaddedBytes out(ctan_code.size() + noise_code.size() + quant_code.size() +
                  dc_toc.size() + dc_code.size() + order_code.size() +
                  histo_code.size() + ac_toc.size() + ac_code.size());
  size_t byte_pos = 0;
  Append(ctan_code, &out, &byte_pos);
  Append(noise_code, &out, &byte_pos);
  Append(quant_code, &out, &byte_pos);
  Append(dc_toc, &out, &byte_pos);
  Append(dc_code, &out, &byte_pos);
  Append(order_code, &out, &byte_pos);
  Append(histo_code, &out, &byte_pos);
  Append(ac_toc, &out, &byte_pos);
  Append(ac_code, &out, &byte_pos);
  return out;
}

bool DecodeColorMap(BitReader* PIK_RESTRICT br, ImageI* PIK_RESTRICT ac_map,
                    int* PIK_RESTRICT dc_val) {
  HuffmanDecodingData entropy;
  if (!entropy.ReadFromBitStream(br)) {
    return PIK_FAILURE("Invalid histogram data.");
  }
  HuffmanDecoder decoder;
  br->FillBitBuffer();
  *dc_val = decoder.ReadSymbol(entropy, br);
  for (size_t y = 0; y < ac_map->ysize(); ++y) {
    int* PIK_RESTRICT row = ac_map->Row(y);
    for (size_t x = 0; x < ac_map->xsize(); ++x) {
      br->FillBitBuffer();
      row[x] = decoder.ReadSymbol(entropy, br);
    }
  }
  br->JumpToByteBoundary();
  return true;
}

template <class Coder>
std::vector<uint64_t> OffsetsFromSizes(const size_t num_group,
                                       BitReader* PIK_RESTRICT reader) {
  // = prefix sum of sizes.
  std::vector<uint64_t> offsets;
  offsets.reserve(1 + num_group);
  offsets.push_back(0);
  for (size_t i = 0; i < num_group; ++i) {
    const uint32_t size = Coder::Decode(reader);
    offsets.push_back(offsets.back() + size);
  }
  reader->JumpToByteBoundary();
  return offsets;
}

// Returns whether the entire tile is accessible - we need bounds checking
// because offsets are derived from untrusted sizes.
bool IsSizeWithinBounds(const uint8_t* tiles_begin, const uint8_t* data_end,
                        const uint64_t offset_begin, const uint64_t offset_end,
                        size_t* PIK_RESTRICT size) {
  if (tiles_begin + offset_end > data_end) {
    return PIK_FAILURE("Tile size exceeds [truncated?] stream length");
  }
  *size = offset_end - offset_begin;
  return true;
}

void InitEagerDequantDC(ColorTransform* PIK_RESTRICT ctan,
                        Quantizer* PIK_RESTRICT quantizer,
                        DecCache* PIK_RESTRICT cache) {
  const float* PIK_RESTRICT dequant_matrix = quantizer->DequantMatrix();
  for (int c = 0; c < 3; ++c) {
    cache->mul_dc[c] =
        dequant_matrix[c * kBlockSize] * quantizer->inv_quant_dc();
  }

  cache->ytox = (ctan->ytox_dc - 128) * kColorFactorX;
  cache->ytob = ctan->ytob_dc * kColorFactorB;
}

void EagerDequantDC(const Image3S& dc_tile, DecCache* PIK_RESTRICT cache,
                    Image3F* PIK_RESTRICT dc_out) {
  using namespace SIMD_NAMESPACE;
  using D = Full<float>;
  constexpr D d;
  constexpr Part<int16_t, D::N> d16;
  constexpr Part<int32_t, D::N> d32;

  const auto dequant_y = set1(d, cache->mul_dc[1]);
  for (size_t y = 0; y < dc_tile.ysize(); ++y) {
    const int16_t* PIK_RESTRICT row_q1 = dc_tile.ConstPlaneRow(1, y);
    float* PIK_RESTRICT row_out1 = dc_out->PlaneRow(1, y);
    for (size_t x = 0; x < dc_tile.xsize(); x += d.N) {
      const auto y16 = load(d16, row_q1 + x);
      const auto yf = convert_to(d, convert_to(d32, y16));

      const auto out_y = yf * dequant_y;
      store(out_y, d, row_out1 + x);
    }
  }

  for (int c = 0; c < 3; c += 2) {  // === for c in {0, 2}
    const auto y_mul = set1(d, (c == 0) ? cache->ytox : cache->ytob);
    const auto xb_mul = set1(d, cache->mul_dc[c]);
    for (size_t y = 0; y < dc_tile.ysize(); ++y) {
      const int16_t* PIK_RESTRICT row_q = dc_tile.ConstPlaneRow(c, y);
      const float* PIK_RESTRICT row_out1 = dc_out->ConstPlaneRow(1, y);
      float* PIK_RESTRICT row_out = dc_out->PlaneRow(c, y);
      for (size_t x = 0; x < dc_tile.xsize(); x += d.N) {
        const auto xb16 = load(d16, row_q + x);
        const auto xb = convert_to(d, convert_to(d32, xb16));
        const auto y_dq = load(d, row_out1 + x);

        const auto out_xb = mul_add(y_mul, y_dq, xb * xb_mul);
        store(out_xb, d, row_out + x);
      }
    }
  }
}

bool DecodeAC(const size_t xsize_blocks, const size_t ysize_blocks,
              const size_t xsize_group, const size_t ysize_group,
              const PaddedBytes& compressed, BitReader* reader,
              ColorTransform* ctan, ThreadPool* pool, DecCache* cache,
              Quantizer* quantizer) {
  PROFILER_FUNC;

  const size_t num_group = xsize_group * ysize_group;

  const uint8_t* const data_end = compressed.data() + compressed.size();

  const std::vector<uint64_t>& dc_tile_offsets =
      OffsetsFromSizes<DcTileSizeCoder>(num_group, reader);
  const uint8_t* dc_tiles_begin = compressed.data() + reader->Position();
  // Skip past what the independent BitReaders will consume.
  reader->SkipBits(dc_tile_offsets[num_group] * 8);

  int coeff_order[kOrderContexts * kBlockSize];
  for (int c = 0; c < kOrderContexts; ++c) {
    DecodeCoeffOrder(&coeff_order[c * kBlockSize], reader);
  }
  reader->JumpToByteBoundary();

  ANSCode code;
  std::vector<uint8_t> context_map;
  // Histogram data size is small and does not require parallelization.
  if (!DecodeHistograms(reader, kNumContexts, 256, kSymbolLut,
                        sizeof(kSymbolLut), &code, &context_map)) {
    return false;
  }
  reader->JumpToByteBoundary();

  const std::vector<uint64_t>& ac_tile_offsets =
      OffsetsFromSizes<AcTileSizeCoder>(num_group, reader);

  const uint8_t* ac_tiles_begin = compressed.data() + reader->Position();
  // Skip past what the independent BitReaders will consume.
  reader->SkipBits(ac_tile_offsets[num_group] * 8);

  Image3B block_ctx(xsize_blocks, ysize_blocks);
  ImageI quant_ac(xsize_blocks, ysize_blocks);

  cache->dc_quant = Image3S(xsize_blocks, ysize_blocks);
  cache->ac_quant = Image3S(xsize_blocks * kBlockSize, ysize_blocks);
  if (cache->eager_dc_dequant) {
    cache->dc = Image3F(xsize_blocks, ysize_blocks);
    InitEagerDequantDC(ctan, quantizer, cache);
  }

  std::atomic<int> num_errors{0};
  pool->Run(0, num_group, [&](const int task, const int thread) {
    const size_t x = task % xsize_group;
    const size_t y = task / xsize_group;
    Image3S dc_tile = DcTile(&cache->dc_quant, x, y);

    size_t dc_size;
    if (!IsSizeWithinBounds(dc_tiles_begin, data_end, dc_tile_offsets[task],
                            dc_tile_offsets[task + 1], &dc_size)) {
      num_errors.fetch_add(1);
      return;
    }
    BitReader dc_reader(dc_tiles_begin + dc_tile_offsets[task], dc_size);
    if (!DecodeImage(&dc_reader, &dc_tile)) {
      num_errors.fetch_add(1);
      return;
    }

    UnpredictDCTile(&dc_tile);

    if (cache->eager_dc_dequant) {
      Image3F dc_out = DcTile(&cache->dc, x, y);
      EagerDequantDC(dc_tile, cache, &dc_out);
    }

    Image3B block_ctx_tile = DcTile(&block_ctx, x, y);
    ComputeBlockContextFromDC(dc_tile, *quantizer, &block_ctx_tile);

    Image3S ac_tile = Tile(&cache->ac_quant, x, y);
    ImageI q_tile = DcTile(&quant_ac, x, y);

    size_t ac_size;
    if (!IsSizeWithinBounds(ac_tiles_begin, data_end, ac_tile_offsets[task],
                            ac_tile_offsets[task + 1], &ac_size)) {
      num_errors.fetch_add(1);
      return;
    }
    BitReader ac_reader(ac_tiles_begin + ac_tile_offsets[task], ac_size);
    if (!DecodeAC(block_ctx_tile, code, context_map, coeff_order, &ac_reader,
                  &ac_tile, &q_tile)) {
      num_errors.fetch_add(1);
    }
  });

  quantizer->SetRawQuantField(std::move(quant_ac));
  return num_errors.load(std::memory_order_relaxed) == 0;
}

bool DecodeFromBitstream(const PaddedBytes& compressed, BitReader* reader,
                         const size_t xsize, const size_t ysize,
                         ThreadPool* pool, ColorTransform* ctan,
                         NoiseParams* noise_params, Quantizer* quantizer,
                         DecCache* cache) {
  const size_t xsize_blocks = DivCeil(xsize, kBlockWidth);
  const size_t ysize_blocks = DivCeil(ysize, kBlockHeight);
  const size_t xsize_group = DivCeil(xsize_blocks, kGroupWidthInBlocks);
  const size_t ysize_group = DivCeil(ysize_blocks, kGroupHeightInBlocks);

  DecodeColorMap(reader, &ctan->ytob_map, &ctan->ytob_dc);
  DecodeColorMap(reader, &ctan->ytox_map, &ctan->ytox_dc);

  if (!DecodeNoise(reader, noise_params)) return false;
  if (!quantizer->Decode(reader)) return false;

  return DecodeAC(xsize_blocks, ysize_blocks, xsize_group, ysize_group,
                  compressed, reader, ctan, pool, cache, quantizer);
}

TFGraphPtr MakeTileFlow(const Image3S& ac_quant, const Quantizer& quantizer,
                        const ColorTransform& ctan, Image3F* dcoeffs,
                        ThreadPool* pool) {
  PROFILER_FUNC;
  PIK_CHECK(ac_quant.xsize() % kBlockSize == 0);
  PIK_CHECK(dcoeffs->xsize() != 0);  // output must already be allocated.

  TFBuilder builder;
  TFNode* ac = builder.AddSource("src_ac", 3, TFType::kI16);
  builder.SetSource(ac, &ac_quant);

  TFNode* quant_ac =
      builder.AddSource("src_ac", 1, TFType::kI32, TFWrap::kZero, Scale(-6, 0));
  builder.SetSource(quant_ac, &quantizer.RawQuantField());

  TFNode* ytox =
      builder.AddSource("ytox", 1, TFType::kI32, TFWrap::kZero, Scale(-9, -3));
  builder.SetSource(ytox, &ctan.ytox_map);

  TFNode* ytob =
      builder.AddSource("ytob", 1, TFType::kI32, TFWrap::kZero, Scale(-9, -3));
  builder.SetSource(ytob, &ctan.ytob_map);

  TFNode* dq_xyb = AddDequantize(ac, quant_ac, quantizer, &builder);

  TFNode* ctan_xyb = AddColorTransform(ctan, dq_xyb, ytox, ytob, &builder);
  builder.SetSink(ctan_xyb, dcoeffs);

  return builder.Finalize(ImageSize::Make(ac_quant.xsize(), ac_quant.ysize()),
                          kDctTileSize, pool);
}

TFGraphPtr MakeTileFlowDC(const Image3S& dc_quant, const Quantizer& quantizer,
                          const ColorTransform& ctan, Image3F* dcoeffs,
                          ThreadPool* pool) {
  PROFILER_FUNC;
  PIK_CHECK(dcoeffs->xsize() != 0);  // output must already be allocated.

  TFBuilder builder;
  TFNode* dc = builder.AddSource("src_dct_dc", 3, TFType::kI16);
  builder.SetSource(dc, &dc_quant);

  TFNode* dq_xyb = AddDequantizeDC(dc, quantizer, &builder);

  TFNode* ctan_xyb = AddColorTransformDC(ctan, dq_xyb, &builder);
  builder.SetSink(ctan_xyb, dcoeffs);

  return builder.Finalize(ImageSize::Make(dc_quant.xsize(), dc_quant.ysize()),
                          ImageSize{64, 64}, pool);
}

void AddPredictions_Smooth(const Image3F& dc, ThreadPool* pool,
                           const Image3F* PIK_RESTRICT dcoeffs,
                           Image3F* PIK_RESTRICT idct) {
  PROFILER_FUNC;

  const Image3F upsampled_dc = BlurUpsampleDC(dc, pool);

  // Treats dcoeffs.DC as 0, then adds upsampled_dc after IDCT.
  *idct = TransposedScaledIDCTAndAdd<DC_Zero>(*dcoeffs, upsampled_dc, pool);
}

void AddPredictions(const Image3F& dc, ThreadPool* pool,
                    Image3F* PIK_RESTRICT dcoeffs, Image3F* PIK_RESTRICT idct) {
  PROFILER_FUNC;

  // Sets dcoeffs.0 from DC and updates 189.
  const Image3F pred2x2 = PredictSpatial2x2_AC64(dc, dcoeffs);
  // Updates dcoeffs _except_ 0189.
  UpSample4x4BlurDCT(pred2x2, 1.5f, 0.0f, pool, dcoeffs);

  *idct = TransposedScaledIDCT(*dcoeffs, pool);
}

Image3F ReconOpsinImage(const Header& header, const Quantizer& quantizer,
                        const ColorTransform& ctan, ThreadPool* pool,
                        DecCache* cache, PikInfo* pik_info) {
  PROFILER_ZONE("recon");
  const size_t dct_xsize = cache->ac_quant.xsize();
  const size_t dct_ysize = cache->ac_quant.ysize();

  if (!cache->eager_dc_dequant) {
    cache->dc = Image3F(dct_xsize / kBlockSize, dct_ysize);
    auto tile_flow_dc =
        MakeTileFlowDC(cache->dc_quant, quantizer, ctan, &cache->dc, pool);
    tile_flow_dc->Run();
  }

  Image3F dcoeffs(dct_xsize, dct_ysize);
  auto tile_flow =
      MakeTileFlow(cache->ac_quant, quantizer, ctan, &dcoeffs, pool);
  tile_flow->Run();

  // AddPredictions* do not use the (invalid) DC component of dcoeffs.

  const size_t out_xsize = (dct_xsize / kBlockSize) * kBlockWidth;
  const size_t out_ysize = dct_ysize * kBlockHeight;
  Image3F idct(out_xsize, out_ysize);

  // Does IDCT already internally to save work.
  if (header.flags & Header::kSmoothDCPred) {
    AddPredictions_Smooth(cache->dc, pool, &dcoeffs, &idct);
  } else {
    AddPredictions(cache->dc, pool, &dcoeffs, &idct);
  }

  if (header.flags & Header::kGaborishTransform) {
    idct = ConvolveGaborish(idct, pool);
  }
  return idct;
}

}  // namespace pik
