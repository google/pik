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

#define PROFILER_ENABLED 1
#include "bit_reader.h"
#include "common.h"
#include "compiler_specific.h"
#include "convolve.h"
#include "dc_predictor.h"
#include "dct_util.h"
#include "deconvolve.h"
#include "gauss_blur.h"
#include "huffman_encode.h"
#include "opsin_codec.h"
#include "opsin_inverse.h"
#include "opsin_params.h"
#include "profiler.h"
#include "resample.h"
#include "simd/simd.h"
#include "status.h"
#include "upscaler.h"

namespace pik {

namespace {

static const int kBlockEdge = 8;
static const int kBlockSize = kBlockEdge * kBlockEdge;

constexpr ImageSize kDctTileSize{512, 8};

const double kDCBlurSigma = 5.5;

}  // namespace

Image3F AlignImage(const Image3F& in, const size_t N) {
  PROFILER_FUNC;
  const size_t block_xsize = DivCeil(in.xsize(), N);
  const size_t block_ysize = DivCeil(in.ysize(), N);
  const size_t xsize = N * block_xsize;
  const size_t ysize = N * block_ysize;
  Image3F out(xsize, ysize);
  for (int c = 0; c < 3; ++c) {
    int y = 0;
    for (; y < in.ysize(); ++y) {
      const float* const PIK_RESTRICT row_in = &in.Row(y)[c][0];
      float* const PIK_RESTRICT row_out = &out.Row(y)[c][0];
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
      const float* const PIK_RESTRICT row_in = out.ConstPlaneRow(c, lastrow);
      float* const PIK_RESTRICT row_out = out.PlaneRow(c, y);
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
    const float* const PIK_RESTRICT row_y = y_plane.Row(y);
    float* const PIK_RESTRICT row_x = coeffs->PlaneRow(0, y);
    float* const PIK_RESTRICT row_b = coeffs->PlaneRow(2, y);
    const int* const PIK_RESTRICT row_ytob =
        ctan.ytob_map.Row(y / kTileInBlocks);
    const int* const PIK_RESTRICT row_ytox =
        ctan.ytox_map.Row(y / kTileInBlocks);

    for (int x = 0; x < bxsize; ++x) {
      const int xoff = x * kBlockSize;
      const float ytob_ac = factor_b * row_ytob[x / kTileInBlocks];
      const float ytox_ac = factor_x * (row_ytox[x / kTileInBlocks] - 128);
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

// ytox_dc already has 128 subtracted; both are pre-multiplied by kColorFactor*.
PIK_INLINE void ApplyColorTransformT(const ConstImageViewF* in_xyb,
                                     const ConstImageViewF& in_ytox_ac,
                                     const ConstImageViewF& in_ytob_ac,
                                     const float ytox_dc, const float ytob_dc,
                                     const OutputRegion& output_region,
                                     const MutableImageViewF* out_xyb) {
  PROFILER_ZONE("|| colorTransform");

  using namespace SIMD_NAMESPACE;
  const Full<float> d;

  // TileFlow tiles are 512 x 8 and blocks are 64 x 1 and yto* maps are
  // downsampled 8x, so we have only one yto* value for the entire tile.
  static_assert(kDctTileSize.xsize == 512, "Use row_ytox[x/512]");
  static_assert(kDctTileSize.ysize == 8, "Use ytox.ConstRow(y/8)");
  const int* const PIK_RESTRICT row_ytox_ac =
      reinterpret_cast<const int*>(in_ytox_ac.ConstRow(0));
  const int* const PIK_RESTRICT row_ytob_ac =
      reinterpret_cast<const int*>(in_ytob_ac.ConstRow(0));
  const auto ytox_ac = set1(d, kColorFactorX * (row_ytox_ac[0] - 128));
  const auto ytob_ac = set1(d, kColorFactorB * row_ytob_ac[0]);
  // Negated to enable default zero-initialization of upper lanes.
  SIMD_ALIGN uint32_t mask1[d.N] = {~0u};
  const auto only_0 = load(d, reinterpret_cast<float*>(mask1));

  const auto ytox = select(ytox_ac, set1(d, ytox_dc), only_0);
  const auto ytob = select(ytob_ac, set1(d, ytob_dc), only_0);

  for (uint32_t y = 0; y < output_region.ysize; ++y) {
    const float* const PIK_RESTRICT row_in_x = in_xyb[0].ConstRow(y);
    const float* const PIK_RESTRICT row_in_y = in_xyb[1].ConstRow(y);
    const float* const PIK_RESTRICT row_in_b = in_xyb[2].ConstRow(y);
    float* const PIK_RESTRICT row_out_x = out_xyb[0].Row(y);
    float* const PIK_RESTRICT row_out_y = out_xyb[1].Row(y);
    float* const PIK_RESTRICT row_out_b = out_xyb[2].Row(y);

    for (uint32_t x = 0; x < output_region.xsize; x += kBlockSize) {
      // First vector: dc in first lane, otherwise ac.
      const auto in_x = load(d, row_in_x + x);
      const auto in_y = load(d, row_in_y + x);
      const auto in_b = load(d, row_in_b + x);
      const auto out_x = mul_add(ytox, in_y, in_x);
      const auto out_b = mul_add(ytob, in_y, in_b);
      store(out_x, d, row_out_x + x);
      store(in_y, d, row_out_y + x);
      store(out_b, d, row_out_b + x);

      // Subsequent vectors: ac-only. Single loop is faster than separate x/b
      // loops (inner or outer).
      // TODO(janwas): avoid copying y via mixed sink node or inplace buffer.
      for (size_t k = d.N; k < kBlockSize; k += d.N) {
        const auto in_x = load(d, row_in_x + x + k);
        const auto in_y = load(d, row_in_y + x + k);
        const auto in_b = load(d, row_in_b + x + k);
        const auto out_x = mul_add(ytox_ac, in_y, in_x);
        const auto out_b = mul_add(ytob_ac, in_y, in_b);
        store(out_x, d, row_out_x + x + k);
        store(in_y, d, row_out_y + x + k);
        store(out_b, d, row_out_b + x + k);
      }
    }
  }
}

// ytox_dc already has 128 subtracted; both are pre-multiplied by kColorFactor*.
PIK_INLINE void ApplyColorTransformDC(const ConstImageViewF* in_xyb,
                                      const float ytox_dc, const float ytob_dc,
                                      const OutputRegion& output_region,
                                      const MutableImageViewF* out_xyb) {
  PROFILER_ZONE("|| colorTransformDC");

  using namespace SIMD_NAMESPACE;
  const Full<float> d;

  const auto ytox = set1(d, ytox_dc);
  const auto ytob = set1(d, ytob_dc);

  for (uint32_t y = 0; y < output_region.ysize; ++y) {
    const float* const PIK_RESTRICT row_in_x = in_xyb[0].ConstRow(y);
    const float* const PIK_RESTRICT row_in_y = in_xyb[1].ConstRow(y);
    const float* const PIK_RESTRICT row_in_b = in_xyb[2].ConstRow(y);
    float* const PIK_RESTRICT row_out_x = out_xyb[0].Row(y);
    float* const PIK_RESTRICT row_out_y = out_xyb[1].Row(y);
    float* const PIK_RESTRICT row_out_b = out_xyb[2].Row(y);

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
}

TFNode* AddColorTransform(const ColorTransform& ctan, TFNode* dq_xyb,
                          TFNode* ytox_ac, TFNode* ytob_ac,
                          TFBuilder* builder) {
  PIK_CHECK(OutType(dq_xyb) == TFType::kF32);
  PIK_CHECK(OutType(ytox_ac) == TFType::kI32);
  PIK_CHECK(OutType(ytob_ac) == TFType::kI32);
  const float ytox_dc = (ctan.ytox_dc - 128) * kColorFactorX;
  const float ytob_dc = ctan.ytob_dc * kColorFactorB;
  return builder->AddClosure(
      "ctan", Borders(), Scale(), {dq_xyb, ytox_ac, ytob_ac}, 3, TFType::kF32,
      [ytox_dc, ytob_dc](const ConstImageViewF* in,
                         const OutputRegion& output_region,
                         const MutableImageViewF* out) {
        ApplyColorTransformT(in + 0, in[3], in[4], ytox_dc, ytob_dc,
                             output_region, out + 0);
      });
}

TFNode* AddColorTransformDC(const ColorTransform& ctan, TFNode* dq_xyb,
                            TFBuilder* builder) {
  PIK_CHECK(OutType(dq_xyb) == TFType::kF32);
  const float ytox_dc = (ctan.ytox_dc - 128) * kColorFactorX;
  const float ytob_dc = ctan.ytob_dc * kColorFactorB;
  return builder->AddClosure(
      "ctan_dc", Borders(), Scale(), {dq_xyb}, 3, TFType::kF32,
      [ytox_dc, ytob_dc](const ConstImageViewF* in,
                         const OutputRegion& output_region,
                         const MutableImageViewF* out) {
        ApplyColorTransformDC(in + 0, ytox_dc, ytob_dc, output_region, out + 0);
      });
}

namespace kernel {

struct Gaborish3 {
  PIK_INLINE const Weights3x3& Weights() const {
    // Unnormalized.
    constexpr float wu0 = 1.0f;
    constexpr float wu1 = static_cast<float>(0.11000238179658321);
    constexpr float wu2 = static_cast<float>(0.089979079587015454);
    constexpr float mul = 1.0 / (wu0 + 4 * (wu1 + wu2));
    constexpr float w0 = wu0 * mul;
    constexpr float w1 = wu1 * mul;
    constexpr float w2 = wu2 * mul;
    static constexpr Weights3x3 weights = {
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
    -0.095336974121859999,
    -0.049147719433952346,
    0.00917293007962648,
    0.0031287055481802016,
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

  using namespace SIMD_NAMESPACE;
  auto out = Image3F(in.xsize(), in.ysize()).Deconstruct();
  for (int i = 0; i < 3; ++i) {
    ConvolveT<strategy::Symmetric3>::Run(BorderNeverUsed(), ExecutorPool(pool),
                                         in.plane(i), kernel::Gaborish3(),
                                         &out[i]);
  }
  return Image3F(out);
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
  PIK_CHECK(xsize % 64 == 0);
  const Operator op;

  for (int c = 0; c < 3; ++c) {
    for (size_t y = 0; y < ysize; ++y) {
      const float* PIK_RESTRICT row_what = what.ConstPlaneRow(c, y);
      float* PIK_RESTRICT row_out = out->PlaneRow(c, y);
      for (size_t bx = 0; bx < xsize; bx += 64) {
        for (size_t x = 1; x < 64; ++x) {
          row_out[bx + x] = op(row_out[bx + x], row_what[bx + x]);
        }
      }
    }
  }
}

static double UpsampleFactor() {
  static double f = 1.5395200630338877;
  return f;
}

template <typename T>
void ZeroDC(Image3<T>* img) {
  for (int c = 0; c < 3; c++) {
    for (size_t y = 0; y < img->ysize(); y++) {
      T* PIK_RESTRICT row = img->PlaneRow(c, y);
      for (size_t x = 0; x < img->xsize(); x += 64) {
        row[x] = T(0);
      }
    }
  }
}

// Scatters dc into "coeffs" at offset 0 within 1x64 blocks.
void FillDC(const Image3F& dc, Image3F* coeffs) {
  const size_t xsize = dc.xsize();
  const size_t ysize = dc.ysize();

  for (int c = 0; c < 3; c++) {
    for (size_t y = 0; y < ysize; y++) {
      const float* PIK_RESTRICT row_dc = dc.PlaneRow(c, y);
      float* PIK_RESTRICT row_out = coeffs->PlaneRow(c, y);
      for (size_t x = 0; x < xsize; ++x) {
        row_out[64 * x] = row_dc[x];
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
  Image3F dct = TransposedScaledDCT(blurred);
  for (int c = 0; c < 3; c++) {
    for (size_t y = 0; y < dct.ysize(); y++) {
      float* PIK_RESTRICT dct_row = dct.PlaneRow(c, y);
      const float* PIK_RESTRICT original_dc_row =
          original_dc.ConstPlaneRow(c, y);
      for (size_t x = 0; x < dct.xsize() / 64; x++) {
        dct_row[64 * x] -= original_dc_row[x];
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
  auto dc_to_encode = Convolve(original_dc, sharpen_kernel).Deconstruct();

  // Individually per channel, until error is acceptable:
  for (int c = 0; c < 3; ++c) {
    for (int iter = 0; iter < kMaxIters; iter++) {
      const ImageF up = BlurUpsampleDC(dc_to_encode[c], pool);
      const ImageF blurred = Subsample8(up);
      if (AddResidualAndCompare(original_dc.plane(c), blurred,
                                kAcceptableError[c], &dc_to_encode[c])) {
        break;  // next channel
      }
    }
  }

  return Image3F(dc_to_encode);
}

void ComputePredictionResiduals_Smooth(const Quantizer& quantizer,
                                       ThreadPool* pool, Image3F* coeffs) {
  PROFILER_FUNC;
  const Image3F original_dc = DCImage(*coeffs);
  const Image3F dc_to_encode = SharpenDC(original_dc, pool);
  FillDC(dc_to_encode, coeffs);

  const Image3F dcoeffs_dc = QuantizeRoundtripDC(quantizer, *coeffs);
  const Image3F prediction_from_dc = BlurUpsampleDCAndDCT(dcoeffs_dc, pool);
  // We have already tried to take into account the effect on DC. We will
  // assume here that we've done that correctly.
  InjectExceptDC<Minus>(prediction_from_dc, coeffs);
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

void Adjust2x2ACFromDC(const Image3F& dc, const int direction,
                       Image3F* coeffs) {
  const size_t xsize = dc.xsize();
  const size_t ysize = dc.ysize();
  Image3F ac01(xsize, ysize);
  Image3F ac10(xsize, ysize);
  Image3F ac11(xsize, ysize);
  // Must avoid ConvolveT for tiny images (PIK_CHECK fails)
  if (xsize <= 8) {
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
    for (int by = 0; by < ysize; ++by) {
      const float* PIK_RESTRICT const row01 = ac01.ConstPlaneRow(c, by);
      const float* PIK_RESTRICT const row10 = ac10.ConstPlaneRow(c, by);
      const float* PIK_RESTRICT const row11 = ac11.ConstPlaneRow(c, by);
      float* SIMD_RESTRICT row_out = coeffs->PlaneRow(c, by);
      for (int bx = 0; bx < xsize; ++bx) {
        row_out[kBlockSize * bx + 1] += direction * row01[bx];
        row_out[kBlockSize * bx + 8] += direction * row10[bx];
        row_out[kBlockSize * bx + 9] += direction * row11[bx];
      }
    }
  }
}

void ComputePredictionResiduals(const Quantizer& quantizer,
                                ThreadPool* pool,
                                Image3F* coeffs) {
  Image3F dc_image = QuantizeRoundtripDC(quantizer, *coeffs);
  Adjust2x2ACFromDC(dc_image, -1, coeffs);
  Image3F dcoeffs = QuantizeRoundtrip(quantizer, *coeffs);
  Adjust2x2ACFromDC(dc_image, 1, &dcoeffs);
  Image3F pred = GetPixelSpaceImageFrom2x2Corners(dcoeffs);
  UpSample4x4BlurDCT(pred, 1.5f, -0.0f, false, pool, coeffs);
}

QuantizedCoeffs ComputeCoefficients(const CompressParams& params,
                                    const Header& header,
                                    const Image3F& opsin,
                                    const Quantizer& quantizer,
                                    const ColorTransform& ctan,
                                    ThreadPool* pool, const PikInfo* aux_out) {
  Image3F coeffs = TransposedScaledDCT(opsin);

  if (header.flags & Header::kSmoothDCPred) {
    ComputePredictionResiduals_Smooth(quantizer, pool, &coeffs);
  } else {
    ComputePredictionResiduals(quantizer, pool, &coeffs);
  }

  PROFILER_ZONE("enc ctan+quant");
  ApplyColorTransform(
      ctan, -1.0f, QuantizeRoundtrip(quantizer, 1, coeffs.plane(1)), &coeffs);
  QuantizedCoeffs qcoeffs;
  qcoeffs.dct = QuantizeCoeffs(coeffs, quantizer);

  if (WantDebugOutput(aux_out)) {
    Image3S dct_copy = CopyImage(qcoeffs.dct);
    ZeroDC(&dct_copy);
    aux_out->DumpCoeffImage("coeff_residuals", dct_copy);
  }
  return qcoeffs;
}

void ComputeBlockContextFromDC(const Image3S& coeffs,
                               const Quantizer& quantizer, Image3B* out) {
  const int bxsize = coeffs.xsize() / 64;
  const int bysize = coeffs.ysize();
  const float iquant_base = quantizer.inv_quant_dc();
  for (int c = 0; c < 3; ++c) {
    const float iquant = iquant_base * quantizer.DequantMatrix()[c * 64];
    const float range = kXybRange[c] / iquant;
    const int kR2Thresh = 10.24f * range * range + 1.0f;
    for (int x = 0; x < bxsize; ++x) {
      out->Row(0)[c][x] = c;
      out->Row(bysize - 1)[c][x] = c;
    }
    for (int y = 1; y + 1 < bysize; ++y) {
      const int16_t* const PIK_RESTRICT row_t = coeffs.ConstPlaneRow(c, y - 1);
      const int16_t* const PIK_RESTRICT row_m = coeffs.ConstPlaneRow(c, y);
      const int16_t* const PIK_RESTRICT row_b = coeffs.ConstPlaneRow(c, y + 1);
      uint8_t* const PIK_RESTRICT row_out = out->PlaneRow(c, y);
      row_out[0] = row_out[bxsize - 1] = c;
      // TODO(janwas): SIMD once we have a separate DC image
      for (int bx = 1; bx + 1 < bxsize; ++bx) {
        const int x = bx * 64;
        const int16_t val_tl = row_t[x - 64];
        const int16_t val_tm = row_t[x];
        const int16_t val_tr = row_t[x + 64];
        const int16_t val_ml = row_m[x - 64];
        const int16_t val_mr = row_m[x + 64];
        const int16_t val_bl = row_b[x - 64];
        const int16_t val_bm = row_b[x];
        const int16_t val_br = row_b[x + 64];
        const int dx =
            (3 * (val_tr - val_tl + val_br - val_bl) + 10 * (val_mr - val_ml));
        const int dy =
            (3 * (val_bl - val_tl + val_br - val_tr) + 10 * (val_bm - val_tm));
        const int dx2 = dx * dx;
        const int dy2 = dy * dy;
        const int dxdy = std::abs(2 * dx * dy);
        const long long r2 = (long long)(dx2) + dy2;
        const int d2 = dy2 - dx2;
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
    const int* const PIK_RESTRICT row = ac_map.Row(y);
    for (int x = 0; x < ac_map.xsize(); ++x) {
      WriteBits(bit_depths[row[x]], bit_codes[row[x]], &storage_ix, storage);
    }
  }
  WriteZeroesToByteBoundary(&storage_ix, storage);
  PIK_ASSERT((storage_ix >> 3) <= output.size());
  output.resize(storage_ix >> 3);
  if (info) {
    info->histogram_size = histo_bits >> 3;
    info->entropy_coded_bits = storage_ix - histo_bits;
    info->total_size += output.size();
  }
  return output;
}

std::string EncodeToBitstream(const QuantizedCoeffs& qcoeffs,
                              const Quantizer& quantizer,
                              const NoiseParams& noise_params,
                              const ColorTransform& ctan, bool fast_mode,
                              PikInfo* info) {
  PROFILER_FUNC;
  const Image3S& qdct = qcoeffs.dct;
  const size_t x_tilesize =
      DivCeil(qdct.xsize(), kSupertileInBlocks * kBlockSize);
  const size_t y_tilesize = DivCeil(qdct.ysize(), kSupertileInBlocks);
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
  Image3S predicted_dc(qdct.xsize() / kBlockSize, qdct.ysize());
  Image3B block_ctx(qdct.xsize() / kBlockSize, qdct.ysize());
  for (int y = 0; y < y_tilesize; y++) {
    for (int x = 0; x < x_tilesize; x++) {
      Image3S predicted_dc_tile =
          Window(&predicted_dc, x * kSupertileInBlocks, y * kSupertileInBlocks,
                 kSupertileInBlocks, kSupertileInBlocks);
      ConstWrapper<Image3S> qdct_tile = ConstWindow(
          qdct, x * kSupertileInBlocks * kBlockSize, y * kSupertileInBlocks,
          kSupertileInBlocks * kBlockSize, kSupertileInBlocks);
      PredictDCTile(qdct_tile.get(), &predicted_dc_tile);
      dc_code += EncodeImage(predicted_dc_tile, 1, dc_info);
      Image3B block_ctx_tile =
          Window(&block_ctx, x * kSupertileInBlocks, y * kSupertileInBlocks,
                 kSupertileInBlocks, kSupertileInBlocks);
      ComputeBlockContextFromDC(qdct_tile.get(), quantizer, &block_ctx_tile);
    }
  }
  std::string ac_code = "";
  std::string order_code = "";
  std::string histo_code = "";
  int order[kOrderContexts * 64];
  std::vector<uint32_t> tokens;
  std::vector<ANSEncodingData> codes;
  std::vector<uint8_t> context_map;
  if (fast_mode) {
    order_code = EncodeNaturalCoeffOrders(info);
    context_map = StaticContextMap();
    tokens = TokenizeCoefficients(qdct, block_ctx, context_map);
    histo_code = BuildAndEncodeHistogramsFast(context_map, tokens, &codes,
                                              info);
  } else {
    ComputeCoeffOrder(qdct, block_ctx, order);
    order_code = EncodeCoeffOrders(order, info);
    ACBlockProcessor processor(&block_ctx, qdct.xsize());
    processor.SetCoeffOrder(order);
    histo_code = BuildAndEncodeHistograms(qdct, &processor, &codes,
                                          &context_map, ac_info);
  }
  for (int y = 0; y < y_tilesize; y++) {
    for (int x = 0; x < x_tilesize; x++) {
      ConstWrapper<Image3S> qdct_tile = ConstWindow(
          qdct, x * kSupertileInBlocks * kBlockSize, y * kSupertileInBlocks,
          kSupertileInBlocks * kBlockSize, kSupertileInBlocks);
      ConstWrapper<Image3B> block_ctx_tile =
          ConstWindow(block_ctx, x * kSupertileInBlocks, y * kSupertileInBlocks,
                      kSupertileInBlocks, kSupertileInBlocks);
      ac_code += fast_mode ?
                 EncodeACFast(qdct_tile.get(), block_ctx_tile.get(), codes,
                              context_map, info) :
                 EncodeAC(qdct_tile.get(), block_ctx_tile.get(), codes,
                          context_map, order, info);
    }
  }
  std::string out = ctan_code + noise_code + quant_code + dc_code + order_code +
                    histo_code + ac_code;
  if (info) {
    info->layers[kLayerHeader].total_size += noise_code.size();
  }
  return out;
}

bool DecodeColorMap(BitReader* br, Image<int>* ac_map, int* dc_val) {
  HuffmanDecodingData entropy;
  if (!entropy.ReadFromBitStream(br)) {
    return PIK_FAILURE("Invalid histogram data.");
  }
  HuffmanDecoder decoder;
  br->FillBitBuffer();
  *dc_val = decoder.ReadSymbol(entropy, br);
  for (int y = 0; y < ac_map->ysize(); ++y) {
    int* const PIK_RESTRICT row = ac_map->Row(y);
    for (int x = 0; x < ac_map->xsize(); ++x) {
      br->FillBitBuffer();
      row[x] = decoder.ReadSymbol(entropy, br);
    }
  }
  br->JumpToByteBoundary();
  return true;
}

bool DecodeFromBitstream(const uint8_t* data, const size_t data_size,
                         const size_t xsize, const size_t ysize,
                         ColorTransform* ctan, NoiseParams* noise_params,
                         Quantizer* quantizer, QuantizedCoeffs* qcoeffs,
                         size_t* compressed_size) {
  const size_t x_blocksize = DivCeil(xsize, kBlockEdge);
  const size_t y_blocksize = DivCeil(ysize, kBlockEdge);
  const size_t x_stilesize = DivCeil(x_blocksize, kSupertileInBlocks);
  const size_t y_stilesize = DivCeil(y_blocksize, kSupertileInBlocks);
  if (data_size == 0) {
    return PIK_FAILURE("Empty compressed data.");
  }
  qcoeffs->dct = Image3S(x_blocksize * kBlockSize, y_blocksize);
  BitReader br(data, data_size);
  DecodeColorMap(&br, &ctan->ytob_map, &ctan->ytob_dc);
  DecodeColorMap(&br, &ctan->ytox_map, &ctan->ytox_dc);
  if (!DecodeNoise(&br, noise_params)) {
    return PIK_FAILURE("noise decoding failed.");
  }
  if (!quantizer->Decode(&br)) {
    return PIK_FAILURE("quantizer Decode failed.");
  }
  Image3B block_ctx(x_blocksize, y_blocksize);
  {
    PROFILER_ZONE("BR decodeImage+unpredict");
    for (int y = 0; y < y_stilesize; y++) {
      for (int x = 0; x < x_stilesize; x++) {
        Image3S qdct_tile =
            Window(&qcoeffs->dct, x * kSupertileInBlocks * kBlockSize,
                   y * kSupertileInBlocks, kSupertileInBlocks * kBlockSize,
                   kSupertileInBlocks);
        if (!DecodeImage(&br, kBlockSize, &qdct_tile)) {
          return PIK_FAILURE("DecodeImage failed.");
        }
        UnpredictDCTile(&qdct_tile);
        Image3B block_ctx_tile =
            Window(&block_ctx, x * kSupertileInBlocks, y * kSupertileInBlocks,
                   kSupertileInBlocks, kSupertileInBlocks);
        ComputeBlockContextFromDC(qdct_tile, *quantizer, &block_ctx_tile);
      }
    }
  }
  {
    PROFILER_ZONE("BR decodeAC");
    int coeff_order[kOrderContexts * 64];
    for (int c = 0; c < kOrderContexts; ++c) {
      DecodeCoeffOrder(&coeff_order[c * 64], &br);
    }
    br.JumpToByteBoundary();
    ANSCode code;
    std::vector<uint8_t> context_map;
    if (!DecodeHistograms(&br, ACBlockProcessor::num_contexts(), 256,
                          kSymbolLut, sizeof(kSymbolLut), &code,
                          &context_map)) {
      return PIK_FAILURE("DecodeHistograms failed");
    }
    for (int y = 0; y < y_stilesize; y++) {
      for (int x = 0; x < x_stilesize; x++) {
        ConstWrapper<Image3B> block_ctx_tile = ConstWindow(
            block_ctx, x * kSupertileInBlocks, y * kSupertileInBlocks,
            kSupertileInBlocks, kSupertileInBlocks);
        Image3S qdct_tile =
            Window(&qcoeffs->dct, x * kSupertileInBlocks * kBlockSize,
                   y * kSupertileInBlocks, kSupertileInBlocks * kBlockSize,
                   kSupertileInBlocks);
        if (!DecodeAC(block_ctx_tile.get(), code, context_map, coeff_order, &br,
                      &qdct_tile)) {
          return PIK_FAILURE("DecodeAC failed.");
        }
      }
    }
  }
  *compressed_size = br.Position();
  return true;
}

TFGraphPtr MakeTileFlow(const QuantizedCoeffs& qcoeffs,
                        const Quantizer& quantizer, const ColorTransform& ctan,
                        Image3F* dcoeffs, ThreadPool* pool) {
  PROFILER_FUNC;
  PIK_CHECK(qcoeffs.dct.xsize() % 64 == 0);
  PIK_CHECK(dcoeffs->xsize() != 0);  // output must already be allocated.

  TFBuilder builder;
  TFNode* dct = builder.AddSource("src_dct", 3, TFType::kI16);
  builder.SetSource(dct, &qcoeffs.dct);

  TFNode* quant_ac =
      builder.AddSource("src_ac", 1, TFType::kI32, TFWrap::kZero, Scale(-6, 0));
  builder.SetSource(quant_ac, &quantizer.RawQuantField());

  TFNode* ytox =
      builder.AddSource("ytox", 1, TFType::kI32, TFWrap::kZero, Scale(-9, -3));
  builder.SetSource(ytox, &ctan.ytox_map);

  TFNode* ytob =
      builder.AddSource("ytob", 1, TFType::kI32, TFWrap::kZero, Scale(-9, -3));
  builder.SetSource(ytob, &ctan.ytob_map);

  TFNode* dq_xyb = AddDequantize(dct, quant_ac, quantizer, &builder);

  TFNode* ctan_xyb = AddColorTransform(ctan, dq_xyb, ytox, ytob, &builder);
  builder.SetSink(ctan_xyb, dcoeffs);

  return builder.Finalize(
      ImageSize::Make(qcoeffs.dct.xsize(), qcoeffs.dct.ysize()), kDctTileSize,
      pool);
}

TFGraphPtr MakeTileFlowDC(const QuantizedCoeffs& qcoeffs,
                          const Quantizer& quantizer,
                          const ColorTransform& ctan, Image3F* dcoeffs,
                          ThreadPool* pool) {
  PROFILER_FUNC;
  PIK_CHECK(dcoeffs->xsize() != 0);  // output must already be allocated.

  TFBuilder builder;
  TFNode* dct = builder.AddSource("src_dct_dc", 3, TFType::kI16, TFWrap::kZero,
                                  Scale(6, 0));
  builder.SetSource(dct, &qcoeffs.dct);

  TFNode* dq_xyb = AddDequantizeDC(dct, quantizer, &builder);

  TFNode* ctan_xyb = AddColorTransformDC(ctan, dq_xyb, &builder);
  builder.SetSink(ctan_xyb, dcoeffs);

  return builder.Finalize(
      ImageSize::Make(qcoeffs.dct.xsize() / 64, qcoeffs.dct.ysize()),
      ImageSize{64, 64}, pool);
}

TFGraphPtr MakeTileFlowIDCT(const Image3F& dcoeffs, bool dc_zero,
                            const size_t xsize, const size_t ysize,
                            Image3F* out, ThreadPool* pool) {
  PROFILER_FUNC;
  PIK_CHECK(dcoeffs.xsize() % 64 == 0);
  PIK_CHECK(out->xsize() != 0);  // output must already be allocated.

  TFBuilder builder;
  TFNode* dct = builder.AddSource("src_dct", 3, TFType::kF32, TFWrap::kZero,
                                  Scale(3, -3));
  builder.SetSource(dct, &dcoeffs);

  TFNode* opsin = AddTransposedScaledIDCT(dct, dc_zero, &builder);

  // Note: integrating Gaborish here is difficult - IDCT cannot easily produce
  // more border data (only in 8x8 units) and not crossing tile boundaries
  // leads to unacceptable errors.
  builder.SetSink(opsin, out);

  return builder.Finalize(ImageSize::Make(xsize, ysize), ImageSize{64, 64},
                          pool);
}

void AddPredictions_Smooth(const Image3F& dcoeffs_dc,
                           ThreadPool* pool,
                           Image3F* PIK_RESTRICT dcoeffs,
                           Image3F* PIK_RESTRICT idct,
                           PikInfo* pik_info) {
  PROFILER_FUNC;

  const Image3F pixel_delta = BlurUpsampleDC(dcoeffs_dc, pool);

  if (WantDebugOutput(pik_info)) {
    Image3B dc_pred_srgb(pixel_delta.xsize(), pixel_delta.ysize());
    const bool dither = true;
    CenteredOpsinToSrgb(pixel_delta, dither, pool, &dc_pred_srgb);
    pik_info->DumpImage("dc_pred", dc_pred_srgb);
  }

  const bool dc_zero = true;
  TFGraphPtr tile_flow_idct = MakeTileFlowIDCT(*dcoeffs, dc_zero, idct->xsize(),
                                               idct->ysize(), idct, pool);
  tile_flow_idct->Run();

  AddTo(pixel_delta, idct);
}

void AddPredictions(const Image3F& dcoeffs_dc,
                    ThreadPool* pool,
                    Image3F* PIK_RESTRICT dcoeffs,
                    Image3F* PIK_RESTRICT idct, PikInfo* pik_info) {
  Adjust2x2ACFromDC(dcoeffs_dc, 1, dcoeffs);
  FillDC(dcoeffs_dc, dcoeffs);
  Image3F pred = GetPixelSpaceImageFrom2x2Corners(*dcoeffs);
  UpSample4x4BlurDCT(pred, 1.5f, 0.0f, false, pool, dcoeffs);
  *idct = TransposedScaledIDCT(*dcoeffs);
}

Image3F ReconOpsinImage(const Header& header,
                        const QuantizedCoeffs& qcoeffs,
                        const Quantizer& quantizer, const ColorTransform& ctan,
                        ThreadPool* pool, PikInfo* pik_info) {
  PROFILER_ZONE("recon");
  const size_t dct_xsize = qcoeffs.dct.xsize();
  const size_t dct_ysize = qcoeffs.dct.ysize();

  Image3F dcoeffs_dc(dct_xsize / 64, dct_ysize);
  auto tile_flow_dc =
      MakeTileFlowDC(qcoeffs, quantizer, ctan, &dcoeffs_dc, pool);
  tile_flow_dc->Run();

  Image3F dcoeffs(dct_xsize, dct_ysize);
  auto tile_flow = MakeTileFlow(qcoeffs, quantizer, ctan, &dcoeffs, pool);
  tile_flow->Run();

  const size_t out_xsize = dct_xsize / 8;
  const size_t out_ysize = dct_ysize * 8;
  Image3F idct(out_xsize, out_ysize);

  // Does IDCT already internally to save work.
  if (header.flags & Header::kSmoothDCPred) {
    AddPredictions_Smooth(dcoeffs_dc, pool, &dcoeffs, &idct, pik_info);
  } else {
    AddPredictions(dcoeffs_dc, pool, &dcoeffs, &idct, pik_info);
  }

  if (header.flags & Header::kGaborishTransform) {
    idct = ConvolveGaborish(idct, pool);
  }
  return idct;
}

}  // namespace pik
