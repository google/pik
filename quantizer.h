// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef QUANTIZER_H_
#define QUANTIZER_H_

#include <stddef.h>
#include <stdint.h>
#include <cmath>
#include <string>
#include <vector>

#include "bit_reader.h"
#include "bits.h"
#include "common.h"
#include "compiler_specific.h"
#include "image.h"
#include "linalg.h"
#include "pik_info.h"
#include "robust_statistics.h"
#include "simd/simd.h"

namespace pik {

static const int kGlobalScaleDenom = 1 << 16;
static const int kGlobalScaleNumerator = 4096;

static const int kNumQuantTables = 1;
static const int kQuantDefault = 0;

// zero-biases for quantizing channels X, Y, B
static constexpr float kZeroBiasDefault[3] = {0.65f, 0.6f, 0.7f};

// Quantization biases.
// The residuals of AC coefficients that we quantize are not uniformly
// distributed. Numerical experiments show that they have a distribution with
// the "shape" of 1/(1+x^2) [up to some coefficients]. This means that the
// expected value of a coefficient that gets quantized to a will not be a
// itself, but (at least with reasonable approximation):
// - 0 if a is 0
// - a * (1 - kOneBias[c]) if a is 1 or -1
// - a - kBiasNumerator/a otherwise
// This follows from computing the distribution of the quantization bias, which
// can be approximated fairly well by <constant>/a when |a| is at least two. If
// |a| is 1, kZeroBias creates a different bias for each channel, thus we look
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

// ac_strategy.h GetQuantKind static_asserts these values remain unchanged.
enum QuantKinds {
  kQuantKindDCT8 = 0,
  kQuantKindID,
  kQuantKindDCT4,
  kQuantKindDCT16Start,
  kQuantKindDCT16_1,
  kQuantKindDCT16_2,
  kQuantKindDCT16_3,
  kQuantKindDCT32Start,
  kQuantKindDCT32_1,
  kQuantKindDCT32_2,
  kQuantKindDCT32_3,
  kQuantKindDCT32_4,
  kQuantKindDCT32_5,
  kQuantKindDCT32_6,
  kQuantKindDCT32_7,
  kQuantKindDCT32_8,
  kQuantKindDCT32_9,
  kQuantKindDCT32_10,
  kQuantKindDCT32_11,
  kQuantKindDCT32_12,
  kQuantKindDCT32_13,
  kQuantKindDCT32_14,
  kQuantKindDCT32_15,
  kNumQuantKinds
};

// Offset of a specific quant matrix in the array of all quant matrices, in
// units of blocks.
PIK_INLINE constexpr size_t DequantMatrixOffset(int id, size_t quant_kind,
                                                int c) {
  return (id * 3 + c) * kNumQuantKinds + quant_kind;
}

// Quantizer::BqCache::map_ marker value.
constexpr const uint32_t kBlockQuantizerCacheUnpopulated = ~0u;

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
  // template_id = kQuant*, < kNumQuantTables
  Quantizer(size_t block_dim, int template_id, int quant_xsize,
            int quant_ysize);
  Quantizer(size_t block_dim, int template_id, int quant_xsize, int quant_ysize,
            int quant_dc, int global_scale);

  Quantizer Copy(const Rect& rect) const {
    Quantizer copy(block_dim_, template_id_, rect.xsize(), rect.ysize(),
                   quant_dc_, global_scale_);
    memcpy(copy.dc_quant_, dc_quant_, sizeof(dc_quant_));
    copy.quant_matrix_ = quant_matrix_;
    copy.initialized_ = true;
    copy.inv_quant_dc_ = inv_quant_dc_;
    copy.SetRawQuantField(CopyImage(rect, RawQuantField()));
    return copy;
  }

  size_t block_dim() const { return block_dim_; }

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
    const size_t n = block_dim_;
    const size_t block_size = n * n;
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
    if (!initialized_) {
      changed = true;
    }
    if (changed) {
      const float* PIK_RESTRICT dequant_matrices =
          DequantMatrix(0, kQuantKindDCT8);
      for (int i = 0; i < quant_matrix_.size(); ++i) {
        quant_matrix_[i] = 1.0f / dequant_matrices[i];
      }
      const float qdc = global_scale_float_ * quant_dc_;
      // Use kQuantKindDCT8 for DC
      for (size_t c = 0; c < 3; ++c) {
        dc_quant_[c] =
            qdc * quant_matrix_[DequantMatrixOffset(0, kQuantKindDCT8, c) *
                                block_size];
      }
      SelectBqLibrary();
      // Precompute regular BlockQuantizers.
      for (size_t y = 0; y < quant_ysize_; ++y) {
        const int32_t* PIK_RESTRICT row_q = quant_img_ac_.ConstRow(y);
        for (size_t x = 0; x < quant_xsize_; ++x) {
          for (size_t quant_kind = 0; quant_kind < kNumQuantKinds;
               quant_kind++) {
            GetBlockQuantizer(row_q[x], quant_kind);
          }
        }
      }
      inv_quant_dc_ = 1.0f / qdc;
      initialized_ = true;
    }
    return changed;
  }

  // Accessors used for adaptive edge-preserving filter:
  // Returns integer AC quantization field.
  const ImageI& RawQuantField() const { return quant_img_ac_; }

  void SetRawQuantField(ImageI&& qf) {
    quant_img_ac_ = std::move(qf);
    SelectBqLibrary();
    for (size_t y = 0; y < quant_ysize_; ++y) {
      const int32_t* PIK_RESTRICT row_q = quant_img_ac_.ConstRow(y);
      for (size_t x = 0; x < quant_xsize_; ++x) {
        for (size_t quant_kind = 0; quant_kind < kNumQuantKinds; quant_kind++) {
          GetBlockQuantizer(row_q[x], quant_kind);
        }
      }
    }
  }

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

  const float* DequantMatrix(int c, size_t quant_kind) const;

  void QuantizeBlockAC(int32_t quant, size_t quant_kind, int c,
                       const float* PIK_RESTRICT block_in,
                       int16_t* PIK_RESTRICT block_out) const {
    const size_t n = block_dim_;
    const size_t block_size = n * n;
    const float* PIK_RESTRICT bq = GetBlockQuantizer(quant, quant_kind);
    const float* PIK_RESTRICT scales = &bq[c * block_size];
#if SIMD_TARGET_VALUE == SIMD_NONE
    const float thres = zero_bias_[c];
    for (size_t k = 0; k < block_size; ++k) {
      const float val = block_in[k] * scales[k];
      block_out[k] = (std::abs(val) < thres) ? 0 : std::round(val);
    }
#else  // SIMD_TARGET_VALUE == SIMD_NONE
    const SIMD_PART(uint32_t, 4) du;
    const SIMD_PART(float, 4) df;
    const SIMD_PART(int32_t, 4) di32;
    const SIMD_PART(int16_t, 8) di16;
    const auto clear_sign = cast_to(df, set1(du, 0x7FFFFFFFu));
    const auto thres = set1(df, zero_bias_[c]);
    for (size_t k = 0; k < block_size; k += 2 * df.N) {
      const auto val_a = load(df, block_in + k);
      const auto val_b = load(df, block_in + k + df.N);
      const auto scale_a = load(df, scales + k);
      const auto scale_b = load(df, scales + k + df.N);
      const auto scaled_a = val_a * scale_a;
      const auto scaled_b = val_b * scale_b;
      const auto abs_scaled_a = scaled_a & clear_sign;
      const auto abs_scaled_b = scaled_b & clear_sign;
      const auto quantized_a = round(scaled_a) & (abs_scaled_a >= thres);
      const auto quantized_b = round(scaled_b) & (abs_scaled_b >= thres);

      const auto s10203040 = cast_to(di16, convert_to(di32, quantized_a));
      const auto s50607080 = cast_to(di16, convert_to(di32, quantized_b));

      const auto s15002600 = interleave_lo(s10203040, s50607080);
      const auto s37004800 = interleave_hi(s10203040, s50607080);

      const auto s13570000 = interleave_lo(s15002600, s37004800);
      const auto s24680000 = interleave_hi(s15002600, s37004800);

      const auto s12345678 = interleave_lo(s13570000, s24680000);
      store(s12345678, di16, block_out + k);
    }
#endif
  }

  template <int c>
  SIMD_ATTR PIK_INLINE void QuantizeRoundtripBlockAC(
      int32_t quant, size_t quant_kind, const float* PIK_RESTRICT in,
      float* PIK_RESTRICT out) const {
    const size_t n = block_dim_;
    const size_t block_size = n * n;
    const SIMD_FULL(uint32_t) du;
    const SIMD_FULL(float) df;
    const auto clear_sign = cast_to(df, set1(du, 0x7FFFFFFFu));
    const auto inv_qac = set1(df, inv_quant_ac(quant));
    const float* PIK_RESTRICT dequant_matrix = DequantMatrix(c, quant_kind);
    const float* PIK_RESTRICT bq = GetBlockQuantizer(quant, quant_kind);
    const float* PIK_RESTRICT scales = &bq[c * block_size];
    const auto thres = set1(df, zero_bias_[c]);
    for (size_t k = 0; k < block_size; k += df.N) {
      const auto val = load(df, in + k);
      const auto scale = load(df, scales + k);
      const auto inv_scale = load(df, dequant_matrix + k);
      const auto scaled = val * scale;
      const auto abs_scaled = scaled & clear_sign;
      const auto quantized = round(scaled) & (abs_scaled >= thres);
      store(AdjustQuantBias<c>(quantized) * inv_scale * inv_qac, df, out + k);
    }
  }

  // Quantizes the specified values in the given block, given as a bitmask in
  // coefficients (as coefficient indices go up to 64, a bitmask is a convenient
  // way to encode them). Note that this requires the specified coefficients to
  // be valid for the given quantization table (i.e. no DC for dct8 or dct16
  // blocks).
  void QuantizeBlockCoefficients(int32_t quant, size_t quant_kind, int c,
                                 const float* PIK_RESTRICT block_in,
                                 int16_t* PIK_RESTRICT block_out,
                                 uint64_t coefficients) const {
    const size_t n = block_dim_;
    const size_t block_size = n * n;
    const float* bq = GetBlockQuantizer(quant, quant_kind);
    const float* scales = &bq[c * block_size];
    const float thres = zero_bias_[c];
    for (uint64_t bits = coefficients; bits != 0; bits &= bits - 1) {
      size_t idx = NumZeroBitsBelowLSBNonzero(bits);
      const float val = block_in[idx] * scales[idx];
      block_out[idx] = std::abs(val) < thres ? 0 : std::round(val);
    }
  }

  PIK_INLINE int16_t QuantizeDC(int c, float dc) const {
    return std::round(dc * dc_quant_[c]);
  }

  std::string Encode(PikImageSizeInfo* info) const;

  bool Decode(BitReader* br);

  void DumpQuantizationMap() const;

 private:
  struct BqCache {
    uint32_t key_;
    std::vector<float> storage_;

    // Map stores offset of BlockQuantizer data in storage.
    // kBlockQuantizerCacheUnpopulated value denotes unpopulated slots.
    std::vector<uint32_t> map_;

    BqCache(uint32_t key) : key_(key) {}
  };

  void SelectBqLibrary() {
    PIK_ASSERT(global_scale_ <= 65535);
    uint32_t key = static_cast<uint32_t>(global_scale_);
    for (size_t i = 0; i < bq_library_.size(); ++i) {
      if (bq_library_[i].key_ == key) {
        bq_cache_ = &bq_library_[i];
        return;
      }
    }
    bq_library_.emplace_back(key);
    bq_cache_ = &bq_library_.back();
  }

  const float* GetBlockQuantizer(int quant, size_t quant_kind) const {
    int map_position = kNumQuantKinds * quant + quant_kind;
    if (bq_cache_->map_.size() <= map_position) {
      constexpr int kBatch = 16;
      bq_cache_->map_.resize(kBatch * DivCeil(map_position + 1, kBatch),
                             kBlockQuantizerCacheUnpopulated);
    }
    size_t offset = bq_cache_->map_[map_position];
    if (offset == kBlockQuantizerCacheUnpopulated) {
      const size_t n = block_dim_;
      const size_t block_size = n * n;
      offset = bq_cache_->storage_.size();
      bq_cache_->map_[map_position] = offset;
      bq_cache_->storage_.resize(offset + 3 * block_size);
      const float qac = Scale() * quant;
      float* bq = &bq_cache_->storage_[offset];
      for (int c = 0; c < 3; ++c) {
        const float* PIK_RESTRICT qm =
            &quant_matrix_[DequantMatrixOffset(0, quant_kind, c) * block_size];
        float* scales = &bq[c * block_size];
        // Block quantizer should not be used for DC quantization of DCT8
        // blocks. To make it clear that something is wrong, we set the scale to
        // be very invalid.
        if (quant_kind == kQuantKindDCT8) scales[0] = -1e5;
        for (size_t k = 0; k < block_size; ++k) {
          scales[k] = qac * qm[k];
        }
      }
    }
    return &bq_cache_->storage_[offset];
  }

  size_t block_dim_;
  size_t quant_xsize_;
  size_t quant_ysize_;

  // These are serialized:
  int template_id_;
  int global_scale_;
  int quant_dc_;
  ImageI quant_img_ac_;

  // These are derived from global_scale_:
  float inv_global_scale_;
  float global_scale_float_;  // reciprocal of inv_global_scale_
  float dc_quant_[3];
  float inv_quant_dc_;

  float zero_bias_[3];
  bool initialized_ = false;
  BqCache* bq_cache_;
  std::vector<float> quant_matrix_;
  std::vector<BqCache> bq_library_;
};

const float* DequantMatrix(int id, size_t quant_kind, int c);

Image3S QuantizeCoeffsDC(const Image3F& in, const Quantizer& quantizer);

// Returns NxN coefficients per block.
ImageF QuantizeRoundtrip(const Quantizer& quantizer, int c,
                         const ImageF& coeffs);

// Input is already 1 DC per block!
ImageF QuantizeRoundtripDC(const Quantizer& quantizer, int c, const ImageF& dc);

}  // namespace pik

#endif  // QUANTIZER_H_
