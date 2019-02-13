// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_QUANTIZER_H_
#define PIK_QUANTIZER_H_

#include <stddef.h>
#include <stdint.h>
#include <cmath>
#include <string>
#include <vector>

#include "pik/ac_strategy.h"
#include "pik/bit_reader.h"
#include "pik/bits.h"
#include "pik/common.h"
#include "pik/compiler_specific.h"
#include "pik/image.h"
#include "pik/linalg.h"
#include "pik/pik_info.h"
#include "pik/quant_weights.h"
#include "pik/robust_statistics.h"
#include "pik/simd/simd.h"

// Quantizes DC and AC coefficients, with separate quantization tables according
// to the quant_kind (which is currently computed from the AC strategy and the
// block index inside that strategy).

namespace pik {

static const int kGlobalScaleDenom = 1 << 16;
static const int kGlobalScaleNumerator = 4096;

// zero-biases for quantizing channels X, Y, B
static constexpr float kZeroBiasDefault[3] = {0.65f, 0.6f, 0.7f};

// Quantization biases.
// The residuals of AC coefficients that we quantize are not uniformly
// distributed. Numerical experiments show that they have a distribution with
// the "shape" of 1/(1+x^2) [up to some coefficients]. This means that the
// expected value of a coefficient that gets quantized to x will not be x
// itself, but (at least with reasonable approximation):
// - 0 if x is 0
// - x * (1 - kOneBias[c]) if x is 1 or -1
// - x - kBiasNumerator/x otherwise
// This follows from computing the distribution of the quantization bias, which
// can be approximated fairly well by <constant>/x when |x| is at least two. If
// |x| is 1, kZeroBias creates a different bias for each channel, thus we look
// it up in the kOneBias LUT.
static constexpr float kOneBias[3] = {
    0.05465007330715401f, 0.07005449891748593f, 0.049935103337343655f};
static constexpr float kBiasNumerator = 0.145f;

// Returns adjusted version of a quantized integer, such that its value is
// closer to the expected value of the original (see comment above).
template <int c>
PIK_INLINE float AdjustQuantBias(int16_t quant) {
  if (quant == 0) return 0;
  if (quant == 1) return 1 - kOneBias[c];
  if (quant == -1) return kOneBias[c] - 1;
  return quant - kBiasNumerator / quant;
}

// SIMD version of the method above.
template <int c>
SIMD_ATTR PIK_INLINE SIMD_FULL(float)::V
    AdjustQuantBias(const SIMD_FULL(float)::V quant) {
  SIMD_FULL(float) df;
  SIMD_FULL(uint32_t) du;
  const auto quant_sign = quant & cast_to(df, set1(du, 0x80000000u));
  const auto quant_abs = andnot(quant_sign, quant);
  const auto quant_one = select(quant, set1(df, 1 - kOneBias[c]) ^ quant_sign,
                                quant_abs >= set1(df, 0.5f));
  const auto quant_minus_inv = quant - set1(df, kBiasNumerator) / quant;
  return select(quant_one, quant_minus_inv, quant_abs >= set1(df, 1.5f));
}

// Accessor for retrieving a single constant without initializing an image.
class QuantConst {
 public:
  explicit QuantConst(const float quant) : quant_(quant) {}
  const float* PIK_RESTRICT Row(size_t y) const { return nullptr; }
  float Get(const float* PIK_RESTRICT row, size_t x) const { return quant_; }

 private:
  const float quant_;
};

class QuantField {
 public:
  explicit QuantField(const ImageF& quant) : quant_(quant) {}
  const float* PIK_RESTRICT Row(size_t y) const { return quant_.Row(y); }
  float Get(const float* PIK_RESTRICT row, size_t x) const { return row[x]; }

 private:
  const ImageF& quant_;
};

class Quantizer {
 public:
  Quantizer(const DequantMatrices* dequant, int quant_xsize, int quant_ysize);
  Quantizer(const DequantMatrices* dequant, int quant_xsize, int quant_ysize,
            int quant_dc, int global_scale);

  Quantizer Copy(const Rect& rect) const {
    Quantizer copy(dequant_, rect.xsize(), rect.ysize(), quant_dc_,
                   global_scale_);
    copy.inv_quant_dc_ = inv_quant_dc_;
    copy.SetRawQuantField(CopyImage(rect, RawQuantField()));
    return copy;
  }

  static PIK_INLINE int ClampVal(float val) {
    static const int kQuantMax = 256;
    return std::min<float>(kQuantMax, std::max<float>(1, val));
  }

  // Recomputes other derived fields after global_scale_ has changed.
  void RecomputeFromGlobalScale() {
    global_scale_float_ = global_scale_ * (1.0 / kGlobalScaleDenom);
    inv_global_scale_ = 1.0 * kGlobalScaleDenom / global_scale_;
  }
  // Returns scaling factor such that Scale() * (RawDC() or RawQuantField())
  // pixels yields the same float values returned by GetQuantField.
  PIK_INLINE float Scale() const { return global_scale_float_; }
  // Reciprocal of Scale().
  PIK_INLINE float InvGlobalScale() const { return inv_global_scale_; }

  template <class QuantInput>  // Quant[Const/Map]
  bool SetQuantField(const float quant_dc, const QuantInput& qf) {
    bool changed = false;
    std::vector<float> data(quant_ysize_ * quant_xsize_);
    for (size_t y = 0; y < quant_ysize_; ++y) {
      const float* PIK_RESTRICT row_qf = qf.Row(y);
      for (size_t x = 0; x < quant_xsize_; ++x) {
        float quant = qf.Get(row_qf, x);
        data[quant_xsize_ * y + x] = quant;
      }
    }
    const float quant_median = Median(&data);
    const float quant_median_absd = MedianAbsoluteDeviation(data, quant_median);
    // Target value for the median value in the quant field.
    const float kQuantFieldTarget = 3.80987740592518214386;
    // We reduce the median of the quant field by the median absolute deviation:
    // higher resolution on highly varying quant fields.
    int new_global_scale = kGlobalScaleDenom *
                           (quant_median - quant_median_absd) /
                           kQuantFieldTarget;
    // Ensure that quant_dc_ will always be at least
    // kGlobalScaleDenom/kGlobalScaleNumerator.
    if (new_global_scale > quant_dc * kGlobalScaleNumerator) {
      new_global_scale = quant_dc * kGlobalScaleNumerator;
    }
    // Ensure that new_global_scale is positive and no more than 1<<15.
    if (new_global_scale <= 0) new_global_scale = 1;
    if (new_global_scale > (1 << 15)) new_global_scale = 1 << 15;
    if (new_global_scale != global_scale_) {
      global_scale_ = new_global_scale;
      RecomputeFromGlobalScale();
      changed = true;
    }
    int val = ClampVal(quant_dc * inv_global_scale_ + 0.5f);
    if (val != quant_dc_) {
      quant_dc_ = val;
      changed = true;
    }
    for (size_t y = 0; y < quant_ysize_; ++y) {
      const float* PIK_RESTRICT row_qf = qf.Row(y);
      int32_t* PIK_RESTRICT row_qi = quant_img_ac_.Row(y);
      for (size_t x = 0; x < quant_xsize_; ++x) {
        int val = ClampVal(qf.Get(row_qf, x) * inv_global_scale_ + 0.5f);
        if (val != row_qi[x]) {
          row_qi[x] = val;
          changed = true;
        }
      }
    }

    if (changed) {
      const float qdc = global_scale_float_ * quant_dc_;
      inv_quant_dc_ = 1.0f / qdc;
    }
    return changed;
  }

  const DequantMatrices& GetDequantMatrices() const { return *dequant_; }

  // Accessors used for adaptive edge-preserving filter:
  // Returns integer AC quantization field.
  const ImageI& RawQuantField() const { return quant_img_ac_; }

  void SetRawQuantField(ImageI&& qf) { quant_img_ac_ = std::move(qf); }

  // Returns "key" that could be used to check if DC quantization is changed.
  // Normally key value is less than (1 << 24), so (~0u) would never occur.
  uint32_t QuantDcKey() const { return (global_scale_ << 16) + quant_dc_; }

  void SetQuant(float quant) { SetQuantField(quant, QuantConst(quant)); }

  // Returns the DC quantization base value, which is currently global (not
  // adaptive). The actual scale factor used to dequantize pixels in channel c
  // is: inv_quant_dc() * DequantMatrix(c, kQuantKindDCT8)[0].
  float inv_quant_dc() const { return inv_quant_dc_; }

  // Dequantize by multiplying with this times dequant_matrix.
  float inv_quant_ac(int32_t quant) const { return inv_global_scale_ / quant; }

  void QuantizeBlockAC(int32_t quant, size_t quant_kind, int c, size_t xsize,
                       size_t ysize, const float* PIK_RESTRICT block_in,
                       size_t in_stride, int16_t* PIK_RESTRICT block_out,
                       size_t out_stride) const {
    constexpr size_t kBlockSize = kBlockDim * kBlockDim;
    const float* qm = dequant_->InvMatrix(quant_kind, c);
    const float qac = Scale() * quant;
    // Not SIMD-fied for now.
    const float thres = zero_bias_[c];
    size_t block_shift =
        NumZeroBitsBelowLSBNonzero(kBlockDim * kBlockDim * xsize);
    // Done in a somewhat weird way to preserve the previous behaviour of
    // dithering.
    // TODO(jyrki): properly dither DCT blocks larger than 8.
    for (size_t iy = 0; iy < ysize; iy++) {
      for (size_t ix = 0; ix < xsize; ix++) {
        double previous_row_err[8] = {0};
        double err = 0;
        for (size_t k = 0; k < kBlockSize; ++k) {
          if ((k & 7) == 0) {
            err = previous_row_err[0];
          } else {
            err += previous_row_err[k & 7];
          }
          size_t x = xsize * (k % kBlockDim) + ix;
          size_t y = ysize * (k / kBlockDim) + iy;
          size_t pos = y * kBlockDim * xsize + x;
          size_t block_off = pos >> block_shift;
          size_t block_idx = pos & (xsize * kBlockDim * kBlockDim - 1);
          float val =
              block_in[block_off * in_stride + block_idx] * (qm[pos] * qac);
          if (err > 0) {
            if (val > 0) {
              val += 0.5 * err;
            } else {
              val -= 0.5 * err;
            }
            err = 0;
          }
          if (err > 0.6) {
            err = 0.6;
          }
          if (fabs(val) > 1) {
            err = 0;
          }
          double v = (std::abs(val) < thres) ? 0 : std::round(val);
          if (fabs(v) < fabs(val)) {
            err += fabs(v - val);
          } else {
            err -= fabs(v - val);
          }
          if (k == 0) {
            err = 0;
          }
          if ((k & 7) == 7) {
            previous_row_err[k & 7] = 1.0 * err;
            err = 0;
          } else {
            err *= 0.5;
            previous_row_err[k & 7] = err;
          }
          if (v > 32767) v = 32767;
          if (v < -32767) v = -32767;
          block_out[block_off * out_stride + block_idx] = v;
        }
      }
    }
  }

  template <int c>
  SIMD_ATTR PIK_INLINE void QuantizeRoundtripBlockAC(
      int32_t quant, size_t quant_kind, size_t xsize, size_t ysize,
      const float* in, size_t in_stride, float* out, size_t out_stride) const {
    constexpr size_t N = kBlockDim;
    constexpr size_t kBlockSize = N * N;
    int16_t quantized[AcStrategy::kMaxCoeffArea];
    float inv_qac = inv_quant_ac(quant);
    QuantizeBlockAC(quant, quant_kind, c, xsize, ysize, in, in_stride,
                    quantized, xsize * kBlockSize);
    const float* PIK_RESTRICT dequant_matrix = DequantMatrix(quant_kind, c);
    for (size_t y = 0; y < ysize; y++) {
      for (size_t k = 0; k < kBlockSize * xsize; k++) {
        float quantized_coeff = quantized[y * kBlockSize * xsize + k];
        out[y * out_stride + k] = AdjustQuantBias<c>(quantized_coeff) *
                                  dequant_matrix[y * kBlockSize * xsize + k] *
                                  inv_qac;
      }
    }
  }

  // Quantizes the specified values in the given block, given as a bitmask in
  // coefficients (as coefficient indices go up to 64, a bitmask is a convenient
  // way to encode them), and then dequantizes them. Note that this requires the
  // specified coefficients to be valid for the given quantization table (i.e.
  // no DC for dct8 or dct16 blocks). `block_in` (and out) may not necessarily
  // be contiguous, but it can be composed of `ysize` slices of size
  // `xsize`*kBlockDim*kBlockDim that are `block_stride` apart.
  template <int c>
  void QuantizeRoundtripBlockCoefficients(int32_t quant, size_t quant_kind,
                                          size_t xsize, size_t ysize,
                                          const float* block_in,
                                          size_t in_stride, float* block_out,
                                          size_t out_stride,
                                          uint64_t coefficients) const {
    constexpr size_t N = kBlockDim;
    int16_t quantized[AcStrategy::kMaxCoeffArea];
    float inv_qac = inv_quant_ac(quant);
    QuantizeBlockAC(quant, quant_kind, c, xsize, ysize, block_in, in_stride,
                    quantized, xsize * N * N);
    const float* PIK_RESTRICT dequant_matrix = DequantMatrix(quant_kind, c);
    size_t block_shift =
        NumZeroBitsBelowLSBNonzero(kBlockDim * kBlockDim * xsize);
    for (uint64_t bits = coefficients; bits != 0; bits &= bits - 1) {
      size_t k = NumZeroBitsBelowLSBNonzero(bits);
      for (size_t iy = 0; iy < ysize; iy++) {
        for (size_t ix = 0; ix < xsize; ix++) {
          size_t x = k % kBlockDim;
          size_t y = k / kBlockDim;
          size_t pos = (y * ysize + iy) * xsize * kBlockDim + x * xsize + ix;
          size_t block_off = pos >> block_shift;
          size_t block_idx = pos & (kBlockDim * kBlockDim * xsize - 1);
          float quantized_coeff = quantized[pos];
          block_out[block_off * out_stride + block_idx] =
              AdjustQuantBias<c>(quantized_coeff) * dequant_matrix[pos] *
              inv_qac;
        }
      }
    }
  }

  PIK_INLINE int16_t QuantizeDC(int c, float dc) const {
    return std::round(dc * (dequant_->InvMatrix(kQuantKindDCT8, c)[0] *
                            (global_scale_float_ * quant_dc_)));
  }

  std::string Encode(PikImageSizeInfo* info) const;

  bool Decode(BitReader* br);

  void DumpQuantizationMap() const;

  PIK_INLINE const float* DequantMatrix(size_t quant_kind, int c) const {
    return dequant_->Matrix(quant_kind, c);
  }

  PIK_INLINE const size_t DequantMatrixOffset(size_t quant_kind, int c) const {
    return dequant_->MatrixOffset(quant_kind, c);
  }

 private:
  size_t quant_xsize_;
  size_t quant_ysize_;

  // These are serialized:
  int global_scale_;
  int quant_dc_;
  ImageI quant_img_ac_;

  // These are derived from global_scale_:
  float inv_global_scale_;
  float global_scale_float_;  // reciprocal of inv_global_scale_
  float inv_quant_dc_;

  float zero_bias_[3];
  const DequantMatrices* dequant_;
};

Image3S QuantizeCoeffsDC(const Image3F& in, const Quantizer& quantizer);

// Input is already 1 DC per block!
ImageF QuantizeRoundtripDC(const Quantizer& quantizer, int c, const ImageF& dc);

}  // namespace pik

#endif  // PIK_QUANTIZER_H_
