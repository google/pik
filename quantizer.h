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

#ifndef QUANTIZER_H_
#define QUANTIZER_H_

#include <stddef.h>
#include <stdint.h>
#include <cmath>
#include <string>
#include <vector>

#include "bit_reader.h"
#include "compiler_specific.h"
#include "dct.h"
#include "image.h"
#include "linalg.h"
#include "pik_info.h"
#include "pik_params.h"

namespace pik {

static const int kGlobalScaleDenom = 1 << 16;
static const int kNumQuantTables = 2;
static const int kQuantDefault = 0;
static const int kQuantHQ = 1;
// zero-biases for quantizing channels X, Y, B
static const float kZeroBiasHQ[3] = { 0.52f, 0.63f, 0.72f };
static const float kZeroBiasDefault[3] = { 0.65f, 0.6f, 0.7f };

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
  Quantizer(int template_id, int quant_xsize, int quant_ysize);

  static PIK_INLINE int ClampVal(int val) {
    static const int kQuantMax = 256;
    return std::min(kQuantMax, std::max(1, val));
  }

  template <class QuantInput>  // Quant[Const/Map]
  bool SetQuantField(const float quant_dc, const QuantInput& qf,
                     const CompressParams& cparams) {
    bool changed = false;
    int new_global_scale = 4096 * quant_dc;
    if (new_global_scale != global_scale_) {
      global_scale_ = new_global_scale;
      changed = true;
    }
    const float scale = Scale();
    const float inv_scale = 1.0f / scale;
    int val = ClampVal(quant_dc * inv_scale + 0.5f);
    if (val != quant_dc_) {
      quant_dc_ = val;
      changed = true;
    }
    for (size_t y = 0; y < quant_ysize_; ++y) {
      const float* PIK_RESTRICT row_qf = qf.Row(y);
      int32_t* PIK_RESTRICT row_qi = quant_img_ac_.Row(y);
      for (size_t x = 0; x < quant_xsize_; ++x) {
        int val = ClampVal(qf.Get(row_qf, x) * inv_scale + 0.5f);
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
      const float* PIK_RESTRICT kDequantMatrix = DequantMatrix();
      std::vector<float> quant_matrix(192);
      for (int i = 0; i < quant_matrix.size(); ++i) {
        quant_matrix[i] = 1.0f / kDequantMatrix[i];
      }
      const float qdc = scale * quant_dc_;
      for (size_t y = 0; y < quant_ysize_; ++y) {
        const int32_t* PIK_RESTRICT row_q = quant_img_ac_.Row(y);
        for (size_t x = 0; x < quant_xsize_; ++x) {
          const float qac = scale * row_q[x];
          const Key key = QuantizerKey(x, y);
          for (int c = 0; c < 3; ++c) {
            if (bq_[c].Find(key) == nullptr) {
              const float* PIK_RESTRICT qm = &quant_matrix[c * 64];
              BlockQuantizer* bq = bq_[c].Add(key);
              bq->scales[0] = qdc * qm[0];
              for (int k = 1; k < 64; ++k) {
                bq->scales[k] = qac * qm[k];
              }
            }
          }
        }
      }
      inv_global_scale_ = 1.0f / scale;
      inv_quant_dc_ = 1.0f / qdc;
      inv_quant_patch_ = inv_global_scale_ / quant_patch_;
      initialized_ = true;
    }
    return changed;
  }

  // Accessors used for adaptive edge-preserving filter:

  // Returns integer AC quantization field.
  const ImageI& RawQuantField() const { return quant_img_ac_; }
  void SetRawQuantField(ImageI&& qf) { quant_img_ac_ = std::move(qf); }
  float RawDC() const { return quant_dc_; }
  // Returns scaling factor such that Scale() * RawDC() or RawQuantField()
  // pixels yields the same float values returned by GetQuantField.
  float Scale() const { return global_scale_ * (1.0f / kGlobalScaleDenom); }

  // Reciprocal of Scale().
  float InvGlobalScale() const { return inv_global_scale_; }

  void GetQuantField(float* quant_dc, ImageF* qf);

  void SetQuant(float quant, const CompressParams& cparams) {
    SetQuantField(quant, QuantConst(quant), cparams);
  }
  void SetQuant(float quant) {
    SetQuant(quant, CompressParams());
  }

  float inv_quant_patch() const { return inv_quant_patch_; }
  float inv_quant_dc() const { return inv_quant_dc_; }
  float inv_quant_ac(int quant_x, int quant_y) const {
    return inv_global_scale_ / quant_img_ac_.Row(quant_y)[quant_x];
  }

  const float* DequantMatrix() const;

  void QuantizeBlock(size_t quant_x, size_t quant_y, int c,
                     const float* PIK_RESTRICT block_in,
                     int16_t* PIK_RESTRICT block_out) const {
    const BlockQuantizer& bq = bq_[c].Get(QuantizerKey(quant_x, quant_y));
    const float thres = zero_bias_[c];
    for (int k = 0; k < 64; ++k) {
      const float val = block_in[k] * bq.scales[k];
      block_out[k] = (k > 0 && std::abs(val) < thres) ? 0 : std::round(val);
    }
  }

  void QuantizeBlock2x2(size_t quant_x, size_t quant_y, int c,
                        const float* PIK_RESTRICT block_in,
                        int16_t* PIK_RESTRICT block_out) const {
    const BlockQuantizer& bq = bq_[c].Get(QuantizerKey(quant_x, quant_y));
    const float thres = zero_bias_[c];
    const float val1 = block_in[1] * bq.scales[1];
    const float val8 = block_in[8] * bq.scales[8];
    const float val9 = block_in[9] * bq.scales[9];
    block_out[1] = std::abs(val1) < thres ? 0 : std::round(val1);
    block_out[8] = std::abs(val8) < thres ? 0 : std::round(val8);
    block_out[9] = std::abs(val9) < thres ? 0 : std::round(val9);
  }

  // Returns only DC.
  int16_t QuantizeBlockDC(size_t quant_x, size_t quant_y, int c,
                          const float* PIK_RESTRICT block_in) const {
    const BlockQuantizer& bq = bq_[c].Get(QuantizerKey(quant_x, quant_y));
    const float val = block_in[0] * bq.scales[0];
    return std::round(val);
  }

  // Returns the sum of squared errors in pixel space (i.e. after applying
  // ComputeTransposedScaledBlockIDCTFloat() on error) that would result from
  // quantizing and dequantizing the given block.
  float PixelErrorSquared(size_t qx, size_t qy, int c,
                          const float* PIK_RESTRICT block_in) const {
    const BlockQuantizer& bq = bq_[c].Get(QuantizerKey(qx, qy));
    int16_t coeffs[64];
    QuantizeBlock(qx, qy, c, block_in, coeffs);
    float sumsq = 0.0f;
    for (int ky = 0, k = 0; ky < 8; ++ky) {
      for (int kx = 0; kx < 8; ++kx, ++k) {
        const float idct_scale = 1.0f / (kIDCTScales[kx] * kIDCTScales[ky]);
        const float e = (block_in[k] - coeffs[k] / bq.scales[k]) * idct_scale;
        sumsq += e * e;
      }
    }
    return sumsq;
  }

  std::string Encode(PikImageSizeInfo* info) const;

  bool Decode(BitReader* br);

  void DumpQuantizationMap() const;

 private:
  using Key = uint64_t;

  Key QuantizerKey(size_t qx, size_t qy) const {
    return (static_cast<uint64_t>(global_scale_) << 32) +
           (static_cast<uint64_t>(quant_dc_) << 16) +
           static_cast<uint64_t>(quant_img_ac_.Row(qy)[qx]);
  }

  struct BlockQuantizer {  // POD
    float scales[64];
  };

  class BQCache {
   public:
    // Returns pointer or nullptr.
    const BlockQuantizer* Find(const Key key) const {
      for (size_t i = 0; i < keys_.size(); ++i) {
        if (keys_[i] == key) return &qmap_[i];
      }
      return nullptr;
    }

    // WARNING: reference is potentially invalidated by Add (SetQuantField).
    const BlockQuantizer& Get(const Key key) const {
      const BlockQuantizer* bq = Find(key);
      PIK_ASSERT(bq != nullptr);
      return *bq;
    }

    // Returns pointer to newly added BQ. Precondition: !Find(key).
    BlockQuantizer* Add(const Key key) {
      keys_.push_back(key);
      qmap_.resize(qmap_.size() + 1);
      return &qmap_.back();
    }

   private:
    std::vector<Key> keys_;
    std::vector<BlockQuantizer> qmap_;
  };

  const size_t quant_xsize_;
  const size_t quant_ysize_;
  const int template_id_;
  int global_scale_;
  int quant_patch_;
  int quant_dc_;
  ImageI quant_img_ac_;
  float inv_global_scale_;
  float inv_quant_patch_;
  float inv_quant_dc_;
  float zero_bias_[3];
  bool initialized_ = false;
  BQCache bq_[3];  // one per channel
};

const float* DequantMatrix(int id);

Image3S QuantizeCoeffs(const Image3F& in, const Quantizer& quantizer);
Image3S QuantizeCoeffsDC(const Image3F& in, const Quantizer& quantizer);
Image3F DequantizeCoeffs(const Image3S& in, const Quantizer& quantizer);

// Returns 64 coefficients per block.
ImageF QuantizeRoundtrip(const Quantizer& quantizer, int c,
                         const ImageF& coeffs);
Image3F QuantizeRoundtrip(const Quantizer& quantizer, const Image3F& coeffs);

// Returns 1x4 [--, 1, 8, 9] per block.
Image3F QuantizeRoundtripExtract189(const Quantizer& quantizer,
                                    const Image3F& coeffs);

// Returns 1 DC per block.
Image3F QuantizeRoundtripExtractDC(const Quantizer& quantizer,
                                   const Image3F& coeffs);

// Input is already 1 DC per block!
Image3F QuantizeRoundtripDC(const Quantizer& quantizer, const Image3F& dc);

// Returns the matrix of the quadratic function
//
//   Q(x) = BlockDistance(lambda, scales, x)
//
// See quantizer_test.cc for the definition of BlockDistance().
ImageD ComputeBlockDistanceQForm(const double lambda,
                                 const float* const PIK_RESTRICT scales);

}  // namespace pik

#endif  // QUANTIZER_H_
