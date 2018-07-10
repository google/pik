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
#include <string>

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

// Predict image with gradients in 64x64 blocks (8x8 DC components)
// Gradient map improves the banding of highly quantized sky. Most effective
// for lower qualities.
struct GradientMap {
  // xsize and ysize must match the one of the opsin image to use. This may be
  // a multiple of 8 instead of the original image size so beware.
  GradientMap(size_t xsize, size_t ysize) : xsize_(xsize), ysize_(ysize) {
    xsizeb_ = DivCeil(xsize_, kNumBlocks_);
    ysizeb_ = DivCeil(ysize_, kNumBlocks_);
    xsizec_ = xsizeb_ + 1;
    ysizec_ = ysizeb_ + 1;

    gradient_ = Image3F(xsize_, ysize_);
  }

  static double SmoothInterpolate(double v00, double v01,
      double v10, double v11, double x, double y) {
    // smoothstep: prevents visible transitions and improves butteraugli score
    x = x * x * (3 - 2 * x);
    y = y * y * (3 - 2 * y);
    return v00 * (1 - x) * (1 - y) + v10 * x * (1 - y) +
           v01 * (1 - x) * y + v11 * x * y;
  }

  // opsin is centered
  void ComputeFromSource(const Image3F& opsin) {

    for (int c = 0; c < 3; c++) {
      corners_[c].resize(xsizec_ * ysizec_);
      for (size_t y2 = 0; y2 < ysizec_; y2++) {
        size_t y = std::min<size_t>(ysize_ - 1, y2 * kNumBlocks_);
        for (size_t x2 = 0; x2 < xsizec_; x2++) {
          size_t x = std::min<size_t>(xsize_ - 1, x2 * kNumBlocks_);
          corners_[c][y2 * xsizec_ + x2] = opsin.PlaneRow(c, y)[x];
        }
      }
    }

    AccountForSerialization();

    std::vector<uint8_t> ok[3];
    for (int c = 0; c < 3; c++) {
      ok[c].resize(xsizeb_ * ysizeb_);
      for (size_t y2 = 0; y2 < ysizeb_; y2++) {
        for (size_t x2 = 0; x2 < xsizeb_; x2++) {
          ok[c][y2 * xsizeb_ + x2] = AcceptBlock(x2, y2, c, opsin);
        }
      }
    }
    // make all channels not-ok if any is not-ok
    for (size_t y2 = 0; y2 < ysizeb_; y2++) {
      for (size_t x2 = 0; x2 < xsizeb_; x2++) {
        bool all_ok = ok[0][y2 * xsizeb_ + x2] &&
            ok[1][y2 * xsizeb_ + x2] && ok[2][y2 * xsizeb_ + x2];
        ok[0][y2 * xsizeb_ + x2] &= all_ok;
        ok[1][y2 * xsizeb_ + x2] &= all_ok;
        ok[2][y2 * xsizeb_ + x2] &= all_ok;
      }
    }

    // set not-ok tiles to zero in all corners so they have no effect
    for (int c = 0; c < 3; c++) {
      for (size_t y2 = 0; y2 < ysizeb_; y2++) {
        for (size_t x2 = 0; x2 < xsizeb_; x2++) {
          if (!ok[c][y2 * xsizeb_ + x2]) {
            corners_[c][y2 * xsizec_ + x2] = 0;
            corners_[c][(y2 + 1) * xsizec_ + x2] = 0;
            corners_[c][y2 * xsizec_ + x2 + 1] = 0;
            corners_[c][(y2 + 1) * xsizec_ + x2 + 1] = 0;
          }
        }
      }
    }

    AccountForSerialization();
  }

  bool AcceptBlock(size_t x2, size_t y2, int c, const Image3F& opsin) const {
    if (x2 == 0 || y2 == 0 || x2 + 1 == xsizeb_ || y2 + 1 == ysizeb_) {
      return false;
    }
    if (!AcceptRoughness(x2, y2, c, opsin)) return false;
    if (!XybInRange(x2, y2, c, opsin)) return false;
    return true;
  }

  // a metric of largest difference between two neighboring pixels
  // TODO(user): tweak this, allow a few outliers
  bool AcceptRoughness(size_t x2, size_t y2, int c,
      const Image3F& opsin) const {
    size_t bx0 = x2 * kNumBlocks_;
    size_t by0 = y2 * kNumBlocks_;
    size_t bx1 = std::min<size_t>(bx0 + kNumBlocks_, xsize_);
    size_t by1 = std::min<size_t>(by0 + kNumBlocks_, ysize_);

    static const float accept[3] = {
      kXybRange[0] * 0.1f,
      kXybRange[1] * 0.1f,
      kXybRange[2] * 0.1f,
    };

    size_t numhigh = 0;
    size_t num = 0;
    for (size_t y = by0 + 1; y < by1; y++) {
      for (size_t x = bx0 + 1; x < bx1; x++) {
        float up = opsin.PlaneRow(c, y - 1)[x];
        float left = opsin.PlaneRow(c, y)[x - 1];
        float mid = opsin.PlaneRow(c, y)[x];
        float d = std::max(std::abs(mid - up), std::abs(mid - left));
        if (d > accept[c]) {
          numhigh++;
        }
        num++;
      }
    }

    static const float numallow[3] = { 0.05f, 0.05f, 0.05f };

    return numhigh  < numallow[c] * num;
  }

  bool XybInRange(size_t x2, size_t y2, int c, const Image3F& opsin) const {
    size_t bx0 = x2 * kNumBlocks_;
    size_t by0 = y2 * kNumBlocks_;
    size_t bx1 = std::min<size_t>(bx0 + kNumBlocks_, xsize_);
    size_t by1 = std::min<size_t>(by0 + kNumBlocks_, ysize_);

    static const float accept[3] = {
      kXybRange[0] * 0.8f,
      kXybRange[1] * 0.8f,
      kXybRange[2] * 0.8f,
    };

    for (size_t y = by0; y < by1; y++) {
      for (size_t x = bx0; x < bx1; x++) {
        float value =  opsin.PlaneRow(c, y)[x] - gradient_.PlaneRow(c, y)[x];
        if (value < -accept[c]) return false;
        if (value > accept[c]) return false;
      }
    }
    return true;
  }

  void ComputeGradientImage() {
    for (int c = 0; c < 3; c++) {
      for (size_t by = 0; by < ysize_; by += kNumBlocks_) {
        size_t by0 = by;
        size_t by1 = std::min<size_t>(by + kNumBlocks_, ysize_);
        for (size_t bx = 0; bx < xsize_; bx += kNumBlocks_) {
          size_t bx0 = bx;
          size_t bx1 = std::min<size_t>(bx + kNumBlocks_, xsize_);
          size_t x2 = bx / kNumBlocks_;
          size_t y2 = by / kNumBlocks_;
          float v00 = corners_[c][y2 * xsizec_ + x2];
          float v01 = corners_[c][(y2 + 1) * xsizec_ + x2];
          float v10 = corners_[c][(by / kNumBlocks_) * xsizec_ + x2 + 1];
          float v11 = corners_[c][(y2 + 1) * xsizec_ + x2 + 1];
          for (size_t y = by0; y < by1; y++) {
            auto* row_out = gradient_.PlaneRow(c, y);
            for (size_t x = bx0; x < bx1; x++) {
              row_out[x] = SmoothInterpolate(v00, v01, v10, v11,
                  (x - bx0) / (float)(bx1 - bx0),
                  (y - by0) / (float)(by1 - by0));
            }
          }
        }
      }
    }
  }

  // For encoder
  void Apply(Image3F* opsin) const {
    for (int c = 0; c < 3; c++) {
      for (size_t y = 0; y < ysize_; y++) {
        float* PIK_RESTRICT row_out = opsin->PlaneRow(c, y);
        const float* PIK_RESTRICT row_in = gradient_.ConstPlaneRow(c, y);
        for (size_t x = 0; x < xsize_; x++) {
          row_out[x] -= row_in[x];
        }
      }
    }
  }

  // For decoder
  void Unapply(Image3F* opsin) const {
    for (int c = 0; c < 3; c++) {
      for (size_t y = 0; y < ysize_; y++) {
        float* PIK_RESTRICT row_out = opsin->PlaneRow(c, y);
        const float* PIK_RESTRICT row_in = gradient_.ConstPlaneRow(c, y);
        for (size_t x = 0; x < xsize_; x++) {
          row_out[x] += row_in[x];
        }
      }
    }
  }

  static std::vector<uint8_t> RleEncode(const std::vector<uint8_t>& v) {
    std::vector<uint8_t> result;
    size_t count = 1;
    uint8_t prev = 0;
    for (size_t i = 0; i < v.size(); i++) {
      uint8_t c = v[i];
      if (i > 0 && c == prev && count < 258) {
        count++;
        if (count <= 3) result.push_back(c);
      } else {
        if (count >= 3) {
          result.push_back(count - 3);
        }
        result.push_back(c);
        count = 1;
        prev = c;
      }
    }
    if (count >= 3) result.push_back(count - 3);
    return result;
  }

  static std::vector<uint8_t> RleDecode(
      const uint8_t* rle, size_t max_size, size_t* pos, size_t result_size) {
    if (*pos >= max_size) return {};
    uint8_t prev = rle[(*pos)++];
    std::vector<uint8_t> result = {prev};
    size_t count = 1;
    for (;;) {
      if (*pos >= max_size) return {};
      uint8_t c = rle[(*pos)++];
      if (c == prev) {
        count++;
        result.push_back(c);
        if (count == 3) {
          if (*pos >= max_size) return {};
          int more = rle[(*pos)++];
          for (int j = 0; j < more; j++) {
            result.push_back(prev);
          }
          count = 0;
        }
      } else {
        result.push_back(c);
        count = 1;
        prev = c;
      }
      if (result.size() > result_size) return {};
      if (result.size() == result_size) return result;
    }
    return {};
  }


  void Serialize(PaddedBytes* compressed) const {
    std::vector<uint8_t> encoded(corners_[0].size() * 3);
    size_t pos = 0;
    for (int c = 0; c < 3; c++) {
      double center = kXybRange[c];
      double range = kXybRange[c] * 2;
      double mul = 255 / range;
      for (size_t i = 0; i < corners_[c].size(); i++) {
        int value = std::round((corners_[c][i] + center) * mul);
        value = std::min<int>(std::max<int>(0, value), 255);
        encoded[pos++] = value;
      }
    }
    encoded = RleEncode(encoded);
    pos = compressed->size();
    compressed->resize(compressed->size() + encoded.size());
    for (size_t i = 0; i < encoded.size(); i++) {
      compressed->data()[pos++] = encoded[i];
    }
  }

  bool Deserialize(const PaddedBytes& compressed, size_t* byte_pos) {
    size_t encoded_size = xsizec_ * ysizec_ * 3;
    std::vector<uint8_t> encoded =
        RleDecode(compressed.data(), compressed.size(), byte_pos, encoded_size);
    if (encoded.size() != encoded_size) {
      return PIK_FAILURE("failed to decode gradient map");
    }
    size_t pos = 0;
    for (int c = 0; c < 3; c++) {
      float center = kXybRange[c];
      float range = kXybRange[c] * 2;
      float mul = range / 255;
      int zerolevel = std::round((0 + center) / mul);

      corners_[c].resize(xsizec_ * ysizec_);
      for (size_t i = 0; i < corners_[c].size(); i++) {
        int value = encoded[pos++];
        float v = value * mul - center;
        if (value == zerolevel) v = 0;
        corners_[c][i] = v;
      }
    }

    return true;  // success
  }

  // Serializes and deserializes the gradient image so it has the values the
  // decoder will see.
  void AccountForSerialization() {
    PaddedBytes compressed;
    Serialize(&compressed);
    size_t pos = 0;
    Deserialize(compressed, &pos);
    ComputeGradientImage();
  }

  Image3F gradient_;
  std::vector<float> corners_[3];  // in centered opsin
  // Size of the superblock, in amount of DCT blocks. So we operate on
  // blocks of kNumBlocks_ * kNumBlocks_ DC components, or 8x8 times as much
  // original image pixels.
  static const size_t kNumBlocks_ = 8;
  size_t xsize_;  // image size
  size_t ysize_;
  size_t xsizec_;  // corners map size
  size_t ysizec_;
  size_t xsizeb_;  // num large blocks
  size_t ysizeb_;
};

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

  const size_t xsize_blocks = coeffs->xsize() / kBlockSize;
  const size_t ysize_blocks = coeffs->ysize();
  for (int y = 0; y < ysize_blocks; ++y) {
    const float* PIK_RESTRICT row_y = y_plane.Row(y);
    float* PIK_RESTRICT row_x = coeffs->PlaneRow(0, y);
    float* PIK_RESTRICT row_b = coeffs->PlaneRow(2, y);
    const int* PIK_RESTRICT row_ytob =
        ctan.ytob_map.Row(y / kTileHeightInBlocks);
    const int* PIK_RESTRICT row_ytox =
        ctan.ytox_map.Row(y / kTileHeightInBlocks);

    for (size_t x = 0; x < xsize_blocks; ++x) {
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
        opsin.Plane(i), opsin.xsize(), opsin.ysize(), smooth_weights5,
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
                                         in.Plane(c), kernel::Gaborish3(),
                                         out.MutablePlane(c));
    out.CheckSizesSame();
  }
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
  Image3F impulse_dc(20, 20);
  FillImage(0.0f, &impulse_dc);
  impulse_dc.PlaneRow(0, 10)[10] = 1.0;
  Image3F probe_expected = UpscalerReconstruct(impulse_dc);
  // We are trying to extract a kernel with a smaller radius. This kernel will
  // be unnormalized. However, we don't mind that: the encoder will compensate
  // when encoding.
  auto kernel6 = kernel::Custom<3>::FromResult(probe_expected.Plane(0));

  ImageF probe_test(probe_expected.xsize(), probe_expected.ysize());
  Upsample<GeneralUpsampler8_6x6>(ExecutorLoop(), impulse_dc.Plane(0), kernel6,
                                  &probe_test);
  VerifyRelativeError(probe_expected.Plane(0), probe_test, 5e-2, 5e-2);

  return kernel6;
}

template <class Image>  // ImageF or Image3F
Image BlurUpsampleDC(const Image& original_dc, ThreadPool* pool) {
  const ExecutorPool executor(pool);
  Image out(original_dc.xsize() * kBlockWidth,
            original_dc.ysize() * kBlockHeight);
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
  std::vector<float> blur_kernel = DCfiedGaussianKernel(5.5);
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
      const ImageF up = BlurUpsampleDC(dc_to_encode.Plane(c), pool);
      const ImageF blurred = Subsample8(up);
      // Change pixels of dc_to_encode but not its size.
      if (AddResidualAndCompare(original_dc.Plane(c), blurred,
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
                                       const GradientMap* gradient_map,
                                       ThreadPool* pool, EncCache* cache) {
  PROFILER_FUNC;
  if (!cache->have_pred) {
    const Image3F dc_orig = DCImage(cache->coeffs_init);
    cache->dc_sharp = SharpenDC(dc_orig, pool);
    Image3F dc_dec = QuantizeRoundtripDC(quantizer, cache->dc_sharp);
    if (gradient_map) {
      gradient_map->Unapply(&dc_dec);
    }
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
void Adjust189_64FromDC(const Image3F& dc, Image3F* PIK_RESTRICT coeffs) {
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
        static_assert(kBlockWidth == 8, "Update block offsets");
        block[1] = op(block[1], row01[bx]);
        block[8] = op(block[8], row10[bx]);
        block[9] = op(block[9], row11[bx]);
      }
    }
  }
}

// Returns pixel-space prediction using same adjustment as above followed by
// GetPixelSpaceImageFrom0189. "ac4" is [--, 1, 8, 9].
// Parallelizing doesn't help, even for XX MP images (DC is still small).
Image3F PredictSpatial2x2_AC4(const Image3F& dc, const Image3F& ac4) {
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
      const float* PIK_RESTRICT row_ac = ac4.ConstPlaneRow(c, by);
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
        float* PIK_RESTRICT block_ac = row_ac64 + bx * kBlockSize;
        static_assert(kBlockWidth == 8, "Update block offsets");
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

  QuantizedCoeffs qcoeffs;

  std::unique_ptr<GradientMap> gradient_map;
  if (header.flags & Header::kGradientMap) {
    auto dc = DCImage(cache->coeffs_init);
    gradient_map.reset(new GradientMap(dc.xsize(), dc.ysize()));
    gradient_map->ComputeFromSource(dc);
    gradient_map->Apply(&dc);
    gradient_map->Serialize(&cache->gradient_map);
    cache->gradient[0] = gradient_map->corners_[0];
    cache->gradient[1] = gradient_map->corners_[1];
    cache->gradient[2] = gradient_map->corners_[2];
    FillDC(dc, &cache->coeffs_init);
  }

  if (header.flags & Header::kSmoothDCPred) {
    ComputePredictionResiduals_Smooth(
        quantizer, gradient_map.get(), pool, cache);
  } else {
    ComputePredictionResiduals(quantizer, pool, cache);
  }

  PROFILER_ZONE("enc ctan+quant");
  ApplyColorTransform(ctan, -1.0f,
                      QuantizeRoundtrip(quantizer, 1, cache->coeffs.Plane(1)),
                      &cache->coeffs);

  qcoeffs.dc = QuantizeCoeffsDC(cache->coeffs, quantizer);
  qcoeffs.ac = QuantizeCoeffs(cache->coeffs, quantizer);
  return qcoeffs;
}

// Computes contexts in [0, kOrderContexts) from "rect_dc" within "dc" and
// writes to "rect_ctx" within "ctx".
void ComputeBlockContextFromDC(const Rect& rect_dc, const Image3S& dc,
                               const Quantizer& quantizer, const Rect& rect_ctx,
                               Image3B* PIK_RESTRICT ctx) {
  PROFILER_FUNC;
  PIK_ASSERT(SameSize(rect_dc, rect_ctx));
  const size_t xsize = rect_dc.xsize();
  const size_t ysize = rect_dc.ysize();

  const float iquant_base = quantizer.inv_quant_dc();
  const float* PIK_RESTRICT dequant_matrix = quantizer.DequantMatrix();
  for (int c = 0; c < 3; ++c) {
    const ImageS& plane_dc = dc.Plane(c);
    ImageB* PIK_RESTRICT plane_ctx = ctx->MutablePlane(c);
    memset(rect_ctx.Row(plane_ctx, 0), c, xsize);
    memset(rect_ctx.Row(plane_ctx, ysize - 1), c, xsize);

    const float iquant = iquant_base * dequant_matrix[c * kBlockSize];
    const float range = kXybRange[c] / iquant;
    const int64_t kR2Thresh = std::min(10.24f * range * range + 1.0f, 1E18f);

    for (size_t y = 1; y + 1 < ysize; ++y) {
      const int16_t* PIK_RESTRICT row_t = rect_dc.ConstRow(plane_dc, y - 1);
      const int16_t* PIK_RESTRICT row_m = rect_dc.ConstRow(plane_dc, y);
      const int16_t* PIK_RESTRICT row_b = rect_dc.ConstRow(plane_dc, y + 1);
      uint8_t* PIK_RESTRICT row_out = rect_ctx.Row(plane_ctx, y);
      row_out[0] = row_out[xsize - 1] = c;
      for (size_t bx = 1; bx + 1 < xsize; ++bx) {
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


template <uint32_t kDistribution>
class SizeCoderT {
 public:
  static size_t MaxSize(const size_t num_sizes) {
    const size_t bits = U32Coder::MaxEncodedBits(kDistribution) * num_sizes;
    return DivCeil(bits, kBitsPerByte) + 8;  // 8 extra bytes for WriteBits.
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
using DcGroupSizeCoder = SizeCoderT<0x110C0A09>;  // max observed: 8K
using AcGroupSizeCoder = SizeCoderT<0x150F0E0C>;  // max observed: 142K

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
  const size_t xsize_groups = DivCeil(xsize_blocks, kGroupWidthInBlocks);
  const size_t ysize_groups = DivCeil(ysize_blocks, kGroupHeightInBlocks);
  const size_t num_groups = xsize_groups * ysize_groups;
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
  std::string dc_toc(DcGroupSizeCoder::MaxSize(num_groups), '\0');
  size_t dc_toc_pos = 0;
  uint8_t* dc_toc_storage =
      reinterpret_cast<uint8_t*>(const_cast<char*>(dc_toc.data()));

  // TODO(janwas): per-thread; also pass other tmp args to ShrinkDC
  Image3S tmp_dc_residuals(kGroupWidthInBlocks, kGroupHeightInBlocks);

  // TODO(janwas): per-group once ComputeCoeffOrder is incremental
  Image3B block_ctx(xsize_blocks, ysize_blocks);
  for (size_t y = 0; y < ysize_groups; y++) {
    for (size_t x = 0; x < xsize_groups; x++) {
      const Rect rect(x * kGroupWidthInBlocks, y * kGroupHeightInBlocks,
                      kGroupWidthInBlocks, kGroupHeightInBlocks, xsize_blocks,
                      ysize_blocks);
      const Rect tmp_rect(0, 0, rect.xsize(), rect.ysize());

      ShrinkDC(rect, qcoeffs.dc, &tmp_dc_residuals);
      ComputeBlockContextFromDC(rect, qcoeffs.dc, quantizer, rect, &block_ctx);

      // (Need rect to indicate size because border groups may be smaller)
      const std::string& dc_group_code =
          EncodeImage(tmp_rect, tmp_dc_residuals, dc_info);
      DcGroupSizeCoder::Encode(
          dc_group_code.size(), &dc_toc_pos, dc_toc_storage);
      dc_code += dc_group_code;
    }
  }
  WriteZeroesToByteBoundary(&dc_toc_pos, dc_toc_storage);
  dc_toc.resize(dc_toc_pos / kBitsPerByte);

  std::string ac_code = "";
  std::string order_code = "";
  std::string histo_code = "";
  int32_t order[kOrderContexts * kBlockSize];
  std::vector<ANSEncodingData> codes;
  std::vector<uint8_t> context_map;
  std::vector<std::vector<Token> > all_tokens;
  if (fast_mode) {
    for (size_t i = 0; i < kOrderContexts; ++i) {
      memcpy(&order[i * kBlockSize], kNaturalCoeffOrder,
          kBlockSize * sizeof(order[0]));
    }
  } else {
    ComputeCoeffOrder(qcoeffs.ac, block_ctx, order);
  }

  order_code = EncodeCoeffOrders(order, info);
  const ImageI& quant_field = quantizer.RawQuantField();
  for (size_t y = 0; y < ysize_groups; y++) {
    for (size_t x = 0; x < xsize_groups; x++) {
      const Rect rect(x * kGroupWidthInBlocks, y * kGroupHeightInBlocks,
                      kGroupWidthInBlocks, kGroupHeightInBlocks, xsize_blocks,
                      ysize_blocks);
      // WARNING: TokenizeCoefficients also uses the DC values in qcoeffs.ac!
      all_tokens.emplace_back(TokenizeCoefficients(order, rect, quant_field,
                                                   qcoeffs.ac, block_ctx));
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

  std::string ac_toc(AcGroupSizeCoder::MaxSize(all_tokens.size()), '\0');
  size_t ac_toc_pos = 0;
  uint8_t* ac_toc_storage =
      reinterpret_cast<uint8_t*>(const_cast<char*>(ac_toc.data()));

  for (int i = 0; i < all_tokens.size(); ++i) {
    std::string ac_group_code =
        WriteTokens(all_tokens[i], codes, context_map, ac_info);
    AcGroupSizeCoder::Encode(ac_group_code.size(), &ac_toc_pos, ac_toc_storage);
    ac_code += ac_group_code;
  }
  WriteZeroesToByteBoundary(&ac_toc_pos, ac_toc_storage);
  ac_toc.resize(ac_toc_pos / kBitsPerByte);

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
std::vector<uint64_t> OffsetsFromSizes(const size_t num_groups,
                                       BitReader* PIK_RESTRICT reader) {
  // = prefix sum of sizes.
  std::vector<uint64_t> offsets;
  offsets.reserve(1 + num_groups);
  offsets.push_back(0);
  for (size_t i = 0; i < num_groups; ++i) {
    const uint32_t size = Coder::Decode(reader);
    offsets.push_back(offsets.back() + size);
  }
  reader->JumpToByteBoundary();
  return offsets;
}

// Returns whether the entire group is accessible - we need bounds checking
// because offsets are derived from untrusted sizes.
bool IsSizeWithinBounds(const uint8_t* groups_begin, const uint8_t* data_end,
                        const uint64_t offset_begin, const uint64_t offset_end,
                        size_t* PIK_RESTRICT size) {
  if (groups_begin + offset_end > data_end) {
    return PIK_FAILURE("Group size exceeds [truncated?] stream length");
  }
  *size = offset_end - offset_begin;
  return true;
}

class Dequant {
 public:
  void Init(const ColorTransform& ctan, const Quantizer& quantizer) {
    dequant_matrix_ = quantizer.DequantMatrix();

    // Precompute DC dequantization multipliers.
    for (int c = 0; c < 3; ++c) {
      mul_dc_[c] = dequant_matrix_[c * kBlockSize] * quantizer.inv_quant_dc();
    }

    // Precompute DC inverse color transform.
    ytox_dc_ = (ctan.ytox_dc - 128) * kColorFactorX;
    ytob_dc_ = ctan.ytob_dc * kColorFactorB;

    inv_global_scale_ = quantizer.InvGlobalScale();
  }

  // Dequantizes and inverse color-transforms one group's worth of DC, i.e. the
  // window "rect_dc" within the entire output image "cache->dc".
  void DoDC(const Rect& rect_dc16, const Image3S& img_dc16, const Rect& rect_dc,
            DecCache* PIK_RESTRICT cache) const {
    PIK_ASSERT(SameSize(rect_dc16, rect_dc));
    const size_t xsize = rect_dc16.xsize();
    const size_t ysize = rect_dc16.ysize();

    using namespace SIMD_NAMESPACE;
    using D = Full<float>;
    constexpr D d;
    constexpr Part<int16_t, D::N> d16;
    constexpr Part<int32_t, D::N> d32;

    const auto dequant_y = set1(d, mul_dc_[1]);

    for (size_t by = 0; by < ysize; ++by) {
      const int16_t* PIK_RESTRICT row_y16 =
          rect_dc16.ConstRow(img_dc16.Plane(1), by);
      float* PIK_RESTRICT row_y = rect_dc.Row(cache->dc.MutablePlane(1), by);

      for (size_t bx = 0; bx < xsize; bx += d.N) {
        const auto quantized_y16 = load(d16, row_y16 + bx);
        const auto quantized_y = convert_to(d, convert_to(d32, quantized_y16));
        store(quantized_y * dequant_y, d, row_y + bx);
      }
    }

    for (int c = 0; c < 3; c += 2) {  // === for c in {0, 2}
      const auto y_mul = set1(d, (c == 0) ? ytox_dc_ : ytob_dc_);
      const auto xb_mul = set1(d, mul_dc_[c]);
      for (size_t by = 0; by < ysize; ++by) {
        const int16_t* PIK_RESTRICT row_xb16 =
            rect_dc16.ConstRow(img_dc16.Plane(c), by);
        const float* PIK_RESTRICT row_y =
            rect_dc.ConstRow(cache->dc.Plane(1), by);
        float* PIK_RESTRICT row_xb = rect_dc.Row(cache->dc.MutablePlane(c), by);

        for (size_t bx = 0; bx < xsize; bx += d.N) {
          const auto quantized_xb16 = load(d16, row_xb16 + bx);
          const auto quantized_xb =
              convert_to(d, convert_to(d32, quantized_xb16));
          const auto out_y = load(d, row_y + bx);

          const auto out_xb = mul_add(y_mul, out_y, quantized_xb * xb_mul);
          store(out_xb, d, row_xb + bx);
        }
      }
    }
  }

  // Dequantizes and inverse color-transforms one group, i.e. the window "rect"
  // (in block units) within the entire output image "cache->ac".
  void DoAC(const Rect& rect_ac16, const Image3S& img_ac16, const Rect& rect,
            const ImageI& img_quant_field, const ImageI& img_ytox,
            const ImageI& img_ytob, DecCache* PIK_RESTRICT cache) const {
    const size_t xsize = rect_ac16.xsize();  // [blocks]
    const size_t ysize = rect_ac16.ysize();
    PIK_ASSERT(img_ac16.xsize() % kBlockSize == 0);
    PIK_ASSERT(xsize <= img_ac16.xsize() / kBlockSize);
    PIK_ASSERT(ysize <= img_ac16.ysize());
    PIK_ASSERT(SameSize(rect_ac16, rect));
    PIK_ASSERT(SameSize(img_ytox, img_ytob));

    using namespace SIMD_NAMESPACE;
    using D = Full<float>;
    constexpr D d;
    constexpr Part<int16_t, D::N> d16;
    constexpr Part<int32_t, D::N> d32;

    const size_t x0_ctan = rect.x0() / kTileWidthInBlocks;
    const size_t y0_ctan = rect.y0() / kTileHeightInBlocks;
    const size_t x0_dct = rect.x0() * kBlockSize;
    const size_t x0_dct16 = rect_ac16.x0() * kBlockSize;

    for (size_t by = 0; by < ysize; ++by) {
      const int16_t* PIK_RESTRICT row_y16 =
          img_ac16.PlaneRow(1, rect_ac16.y0() + by) + x0_dct16;
      const int* PIK_RESTRICT row_quant_field =
          rect.ConstRow(img_quant_field, by);
      float* PIK_RESTRICT row_y =
          cache->ac.PlaneRow(1, rect.y0() + by) + x0_dct;

      for (size_t bx = 0; bx < xsize; ++bx) {
        for (size_t k = 0; k < kBlockSize; k += d.N) {
          const size_t x = bx * kBlockSize + k;

          const auto dequant = load(d, dequant_matrix_ + kBlockSize + k);
          const auto y_mul = dequant * set1(d, SafeDiv(inv_global_scale_,
                                                       row_quant_field[bx]));

          const auto quantized_y16 = load(d16, row_y16 + x);
          const auto quantized_y =
              convert_to(d, convert_to(d32, quantized_y16));
          store(quantized_y * y_mul, d, row_y + x);
        }
      }
    }

    for (int c = 0; c < 3; c += 2) {  // === for c in {0, 2}
      const ImageI& img_ctan = (c == 0) ? img_ytox : img_ytob;
      for (size_t by = 0; by < ysize; ++by) {
        const int16_t* PIK_RESTRICT row_xb16 =
            img_ac16.PlaneRow(c, rect_ac16.y0() + by) + x0_dct16;
        const int* PIK_RESTRICT row_quant_field =
            rect.ConstRow(img_quant_field, by);
        const int* PIK_RESTRICT row_ctan =
            img_ctan.ConstRow(y0_ctan + by / kTileHeightInBlocks) + x0_ctan;
        const float* PIK_RESTRICT row_y =
            cache->ac.ConstPlaneRow(1, rect.y0() + by) + x0_dct;
        float* PIK_RESTRICT row_xb =
            cache->ac.PlaneRow(c, rect.y0() + by) + x0_dct;

        for (size_t bx = 0; bx < xsize; ++bx) {
          const int32_t ctan = row_ctan[bx / kTileWidthInBlocks];
          const auto y_mul = (c == 0) ? set1(d, kColorFactorX * (ctan - 128))
                                      : set1(d, kColorFactorB * ctan);

          for (size_t k = 0; k < kBlockSize; k += d.N) {
            const size_t x = bx * kBlockSize + k;

            const auto dequant = load(d, dequant_matrix_ + c * kBlockSize + k);
            const auto xb_mul = dequant * set1(d, SafeDiv(inv_global_scale_,
                                                          row_quant_field[bx]));

            const auto quantized_xb16 = load(d16, row_xb16 + x);
            const auto quantized_xb =
                convert_to(d, convert_to(d32, quantized_xb16));

            const auto out_y = load(d, row_y + x);

            store(mul_add(y_mul, out_y, quantized_xb * xb_mul), d, row_xb + x);
          }
        }
      }
    }
  }

 private:
  static PIK_INLINE float SafeDiv(float num, int32_t div) {
    return div == 0 ? 1E10f : num / div;
  }

  // Precomputed DC dequant/color transform
  float mul_dc_[3];
  float ytox_dc_;
  float ytob_dc_;

  // AC dequant
  const float* PIK_RESTRICT dequant_matrix_;
  float inv_global_scale_;
};

// Temporary storage; one per thread, for one group.
struct DecoderBuffers {
  void InitOnce(const bool eager_dequant) {
    // This thread already allocated its buffers.
    if (num_nzeroes.xsize() != 0) return;

    // Allocate enough for a whole group - partial groups on the right/bottom
    // border just use a subset. The valid size is passed via Rect.
    const size_t xsize_blocks = kGroupWidthInBlocks;
    const size_t ysize_blocks = kGroupHeightInBlocks;

    block_ctx = Image3B(xsize_blocks, ysize_blocks);

    if (eager_dequant) {
      quantized_dc = Image3S(xsize_blocks, ysize_blocks);
      quantized_ac = Image3S(xsize_blocks * kBlockSize, ysize_blocks);
    }  // else: Decode uses DecCache->quantized_dc/ac.

    dc_y = ImageS(xsize_blocks, ysize_blocks);
    dc_xz_residuals = ImageS(xsize_blocks * 2, ysize_blocks);
    dc_xz_expanded = ImageS(xsize_blocks * 2, ysize_blocks);

    num_nzeroes = Image3I(xsize_blocks, ysize_blocks);
  }

  Image3B block_ctx;

  // Decode (only if eager_dequant)
  Image3S quantized_dc;
  Image3S quantized_ac;

  // ExpandDC
  ImageS dc_y;
  ImageS dc_xz_residuals;
  ImageS dc_xz_expanded;

  // DequantAC
  Image3I num_nzeroes;
};

bool DecodeCoefficientsAndDequantize(
    const size_t xsize_blocks, const size_t ysize_blocks,
    const PaddedBytes& compressed, BitReader* reader, ColorTransform* ctan,
    ThreadPool* pool, DecCache* cache, Quantizer* quantizer) {
  PROFILER_FUNC;

  const size_t xsize_groups = DivCeil(xsize_blocks, kGroupWidthInBlocks);
  const size_t ysize_groups = DivCeil(ysize_blocks, kGroupHeightInBlocks);
  const size_t num_groups = xsize_groups * ysize_groups;

  const uint8_t* const data_end = compressed.data() + compressed.size();

  const std::vector<uint64_t>& dc_group_offsets =
      OffsetsFromSizes<DcGroupSizeCoder>(num_groups, reader);
  const uint8_t* dc_groups_begin = compressed.data() + reader->Position();
  // Skip past what the independent BitReaders will consume.
  reader->SkipBits(dc_group_offsets[num_groups] * kBitsPerByte);

  int coeff_order[kOrderContexts * kBlockSize];
  for (size_t c = 0; c < kOrderContexts; ++c) {
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

  const std::vector<uint64_t>& ac_group_offsets =
      OffsetsFromSizes<AcGroupSizeCoder>(num_groups, reader);

  const uint8_t* ac_groups_begin = compressed.data() + reader->Position();
  // Skip past what the independent BitReaders will consume.
  reader->SkipBits(ac_group_offsets[num_groups] * kBitsPerByte);

  // Will be moved into quantizer.
  ImageI ac_quant_field(xsize_blocks, ysize_blocks);

  Dequant dequant;
  if (cache->eager_dequant) {
    dequant.Init(*ctan, *quantizer);
    cache->dc = Image3F(xsize_blocks, ysize_blocks);
    cache->ac = Image3F(xsize_blocks * kBlockSize, ysize_blocks);
  } else {
    cache->quantized_dc = Image3S(xsize_blocks, ysize_blocks);
    cache->quantized_ac = Image3S(xsize_blocks * kBlockSize, ysize_blocks);
  }

  std::vector<DecoderBuffers> decoder_buf(
      std::max<size_t>(1, pool->NumThreads()));

  // For each group: independent/parallel decode
  std::atomic<int> num_errors{0};
  pool->Run(0, num_groups, [&](const int task, const int thread) {
    const size_t group_x = task % xsize_groups;
    const size_t group_y = task / xsize_groups;
    const Rect rect(group_x * kGroupWidthInBlocks,
                    group_y * kGroupHeightInBlocks, kGroupWidthInBlocks,
                    kGroupHeightInBlocks, xsize_blocks, ysize_blocks);
    const Rect tmp_rect(0, 0, rect.xsize(), rect.ysize());
    DecoderBuffers& tmp = decoder_buf[thread];
    tmp.InitOnce(cache->eager_dequant);

    size_t dc_size;
    if (!IsSizeWithinBounds(dc_groups_begin, data_end, dc_group_offsets[task],
                            dc_group_offsets[task + 1], &dc_size)) {
      num_errors.fetch_add(1);
      return;
    }
    BitReader dc_reader(dc_groups_begin + dc_group_offsets[task], dc_size);

    Image3S* quantized_dc =
        cache->eager_dequant ? &tmp.quantized_dc : &cache->quantized_dc;
    const Rect& rect16 = cache->eager_dequant ? tmp_rect : rect;

    if (!DecodeImage(&dc_reader, rect16, quantized_dc)) {
      num_errors.fetch_add(1);
      return;
    }

    ExpandDC(rect16, quantized_dc, &tmp.dc_y, &tmp.dc_xz_residuals,
             &tmp.dc_xz_expanded);

    if (cache->eager_dequant) {
      dequant.DoDC(rect16, *quantized_dc, rect, cache);
    }

    ComputeBlockContextFromDC(rect16, *quantized_dc, *quantizer, tmp_rect,
                              &tmp.block_ctx);

    size_t ac_size;
    if (!IsSizeWithinBounds(ac_groups_begin, data_end, ac_group_offsets[task],
                            ac_group_offsets[task + 1], &ac_size)) {
      num_errors.fetch_add(1);
      return;
    }
    BitReader ac_reader(ac_groups_begin + ac_group_offsets[task], ac_size);
    Image3S* quantized_ac =
        cache->eager_dequant ? &tmp.quantized_ac : &cache->quantized_ac;
    if (!DecodeAC(tmp.block_ctx, code, context_map, coeff_order, &ac_reader,
                  rect16, quantized_ac, rect, &ac_quant_field,
                  &tmp.num_nzeroes)) {
      num_errors.fetch_add(1);
    }

    if (cache->eager_dequant) {
      dequant.DoAC(rect16, *quantized_ac, rect, ac_quant_field, ctan->ytox_map,
                   ctan->ytob_map, cache);
    }
  });

  quantizer->SetRawQuantField(std::move(ac_quant_field));
  return num_errors.load(std::memory_order_relaxed) == 0;
}

bool DecodeFromBitstream(const Header& header, const PaddedBytes& compressed,
                         BitReader* reader,
                         const size_t xsize_blocks, const size_t ysize_blocks,
                         ThreadPool* pool, ColorTransform* ctan,
                         NoiseParams* noise_params, Quantizer* quantizer,
                         DecCache* cache) {
  if (header.flags & Header::kGradientMap) {
    GradientMap gradient_map(xsize_blocks, ysize_blocks);
    size_t byte_pos = reader->Position();
    gradient_map.Deserialize(compressed, &byte_pos);
    reader->SkipBits((byte_pos - reader->Position()) * 8);
    cache->gradient[0] = gradient_map.corners_[0];
    cache->gradient[1] = gradient_map.corners_[1];
    cache->gradient[2] = gradient_map.corners_[2];
  }

  DecodeColorMap(reader, &ctan->ytob_map, &ctan->ytob_dc);
  DecodeColorMap(reader, &ctan->ytox_map, &ctan->ytox_dc);

  if (!DecodeNoise(reader, noise_params)) return false;
  if (!quantizer->Decode(reader)) return false;

  return DecodeCoefficientsAndDequantize(xsize_blocks, ysize_blocks, compressed,
                                         reader, ctan, pool, cache, quantizer);
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
  const size_t xsize_blocks = quantizer.RawQuantField().xsize();
  const size_t ysize_blocks = quantizer.RawQuantField().ysize();
  const size_t xsize_groups = DivCeil(xsize_blocks, kGroupWidthInBlocks);
  const size_t ysize_groups = DivCeil(ysize_blocks, kGroupHeightInBlocks);

  // If not already done (when called after DecodeFromBitstream)
  if (!cache->eager_dequant) {
    // Caller must have allocated/filled quantized_dc/ac.
    PIK_CHECK(SameSize(cache->quantized_dc, quantizer.RawQuantField()));

    Dequant dequant;
    dequant.Init(ctan, quantizer);
    cache->dc = Image3F(xsize_blocks, ysize_blocks);
    cache->ac = Image3F(xsize_blocks * kBlockSize, ysize_blocks);

    std::vector<DecoderBuffers> decoder_buf(
        std::max<size_t>(1, pool->NumThreads()));

    const size_t num_groups = xsize_groups * ysize_groups;
    pool->Run(0, num_groups, [&](const int task, const int thread) {
      DecoderBuffers& tmp = decoder_buf[thread];
      tmp.InitOnce(cache->eager_dequant);

      const size_t group_x = task % xsize_groups;
      const size_t group_y = task / xsize_groups;
      const Rect rect(group_x * kGroupWidthInBlocks,
                      group_y * kGroupHeightInBlocks, kGroupWidthInBlocks,
                      kGroupHeightInBlocks, xsize_blocks, ysize_blocks);

      dequant.DoDC(rect, cache->quantized_dc, rect, cache);
      dequant.DoAC(rect, cache->quantized_ac, rect, quantizer.RawQuantField(),
                   ctan.ytox_map, ctan.ytob_map, cache);
    });
  }

  if (header.flags & Header::kGradientMap) {
    GradientMap map(xsize_blocks, ysize_blocks);
    map.corners_[0] = cache->gradient[0];
    map.corners_[1] = cache->gradient[1];
    map.corners_[2] = cache->gradient[2];
    map.ComputeGradientImage();
    map.Unapply(&cache->dc);
  }

  Image3F idct(xsize_blocks * kBlockWidth, ysize_blocks * kBlockHeight);

  // AddPredictions* do not use the (invalid) DC component of cache->ac.
  // Does IDCT already internally to save work.
  if (header.flags & Header::kSmoothDCPred) {
    AddPredictions_Smooth(cache->dc, pool, &cache->ac, &idct);
  } else {
    AddPredictions(cache->dc, pool, &cache->ac, &idct);
  }

  if (header.flags & Header::kGaborishTransform) {
    idct = ConvolveGaborish(idct, pool);
  }

  return idct;
}

}  // namespace pik
