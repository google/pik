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
#include "tile_flow.h"

namespace pik {

static const int kGlobalScaleDenom = 1 << 16;
static const int kNumQuantTables = 2;
static const int kQuantDefault = 0;
static const int kQuantHQ = 1;

class Quantizer {
 public:
  Quantizer(int template_id, int quant_xsize, int quant_ysize);

  bool SetQuantField(const float quant_dc, const ImageF& qf,
                     const CompressParams& cparams);

  // Accessors used for adaptive edge-preserving filter:

  // Returns integer AC quantization field.
  const ImageI& RawQuantField() const { return quant_img_ac_; }
  float RawDC() const { return quant_dc_; }
  // Returns scaling factor such that Scale() * RawDC() or RawQuantField()
  // pixels yields the same float values returned by GetQuantField.
  float Scale() const { return global_scale_ * (1.0f / kGlobalScaleDenom); }

  // Reciprocal of Scale().
  float InvGlobalScale() const { return inv_global_scale_; }

  void GetQuantField(float* quant_dc, ImageF* qf);

  void SetQuant(float quant, const CompressParams& cparams) {
    SetQuantField(quant, ImageF(quant_xsize_, quant_ysize_, quant), cparams);
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
    for (int k = 0; k < 64; ++k) {
      const float val = block_in[k] * bq.scales[k];
      static const float kZeroBias[3] = { 0.65f, 0.6f, 0.7f };
      const float thres = kZeroBias[c];
      block_out[k] = (k > 0 && std::abs(val) < thres) ? 0 : std::round(val);
    }
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

  const int template_id_;
  const int quant_xsize_;
  const int quant_ysize_;
  int global_scale_;
  int quant_patch_;
  int quant_dc_;
  ImageI quant_img_ac_;
  float inv_global_scale_;
  float inv_quant_patch_;
  float inv_quant_dc_;
  bool initialized_ = false;
  BQCache bq_[3];  // one per channel
};

const float* DequantMatrix(int id);

Image3S QuantizeCoeffs(const Image3F& in, const Quantizer& quantizer);
Image3F DequantizeCoeffs(const Image3S& in, const Quantizer& quantizer);

ImageF QuantizeRoundtrip(const Quantizer& quantizer, int c, const ImageF& img);
Image3F QuantizeRoundtrip(const Quantizer& quantizer, const Image3F& img);

Image3F QuantizeRoundtripDC(const Quantizer& quantizer, const Image3F& img);

// Returns the matrix of the quadratic function
//
//   Q(x) = BlockDistance(lambda, scales, x)
//
// See quantizer_test.cc for the definition of BlockDistance().
ImageD ComputeBlockDistanceQForm(const double lambda,
                                 const float* const PIK_RESTRICT scales);

TFNode* AddDequantize(const TFPorts in_xyb, const TFPorts in_quant_ac,
                      const Quantizer& quantizer, TFBuilder* builder);

TFNode* AddDequantizeDC(const TFPorts in_xyb, const Quantizer& quantizer,
                        TFBuilder* builder);

}  // namespace pik

#endif  // QUANTIZER_H_
