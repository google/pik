// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef QUANT_WEIGHTS_H_
#define QUANT_WEIGHTS_H_

#include <vector>
#include "bit_reader.h"
#include "common.h"
#include "image.h"
#include "pik_info.h"
#include "status.h"
namespace pik {

static constexpr size_t kMaxQuantTableSize = kBlockDim * kBlockDim * 16;

// ac_strategy.h GetQuantKind static_asserts these values remain unchanged.
enum QuantKind {
  kQuantKindDCT8 = 0,
  kQuantKindID,
  kQuantKindDCT2,
  kQuantKindDCT4,
  kQuantKindDCT16,
  kQuantKindDCT32,
  kQuantKindLines,
  kNumQuantKinds
};

struct DctQuantWeightParams {
  static constexpr size_t kMaxRadialBands = 8;
  static constexpr size_t kMaxDistanceBands = 16;
  size_t num_distance_bands;
  float distance_bands[3][kMaxDistanceBands];
  size_t num_eccentricity_bands;
  float eccentricity_bands[3][kMaxRadialBands];

  DctQuantWeightParams() : num_distance_bands(0), num_eccentricity_bands(0) {}
  template <size_t num_dist_bands, size_t num_ecc_bands>
  DctQuantWeightParams(const float dist_bands[3][num_dist_bands],
                       const float ecc_bands[3][num_ecc_bands]) {
    num_distance_bands = num_dist_bands;
    num_eccentricity_bands = num_ecc_bands;
    for (size_t c = 0; c < 3; c++) {
      memcpy(distance_bands[c], dist_bands[c], sizeof(float) * num_dist_bands);
    }
    for (size_t c = 0; c < 3; c++) {
      memcpy(eccentricity_bands[c], ecc_bands[c],
             sizeof(float) * num_ecc_bands);
    }
  }
};

struct QuantEncoding {
  static QuantEncoding Default() {
    return QuantEncoding{/*mode=*/kQuantModeDefault};
  }

  static QuantEncoding Identity(const float* xweights, const float* yweights,
                                const float* bweights) {
    QuantEncoding encoding;
    encoding.mode = kQuantModeID;
    memcpy(encoding.idweights[0], xweights, sizeof(float) * 3);
    memcpy(encoding.idweights[1], yweights, sizeof(float) * 3);
    memcpy(encoding.idweights[2], bweights, sizeof(float) * 3);
    return encoding;
  }

  static QuantEncoding DCT2(const float* xweights, const float* yweights,
                            const float* bweights) {
    QuantEncoding encoding;
    encoding.mode = kQuantModeDCT2;
    memcpy(encoding.dct2weights[0], xweights, sizeof(float) * 6);
    memcpy(encoding.dct2weights[1], yweights, sizeof(float) * 6);
    memcpy(encoding.dct2weights[2], bweights, sizeof(float) * 6);
    return encoding;
  }

  static QuantEncoding DCT4(const DctQuantWeightParams& params,
                            const float* xmul, const float* ymul,
                            const float* bmul) {
    QuantEncoding encoding;
    encoding.mode = kQuantModeDCT4;
    encoding.dct_params = params;
    memcpy(encoding.dct4multipliers[0], xmul, sizeof(float) * 2);
    memcpy(encoding.dct4multipliers[1], ymul, sizeof(float) * 2);
    memcpy(encoding.dct4multipliers[2], bmul, sizeof(float) * 2);
    PIK_ASSERT(params.num_distance_bands <=
               DctQuantWeightParams::kMaxDistanceBands);
    PIK_ASSERT(params.num_eccentricity_bands <=
               DctQuantWeightParams::kMaxRadialBands);
    return encoding;
  }

  static QuantEncoding DCT(const DctQuantWeightParams& params) {
    QuantEncoding encoding;
    encoding.mode = kQuantModeDCT;
    encoding.dct_params = params;
    PIK_ASSERT(params.num_distance_bands <=
               DctQuantWeightParams::kMaxDistanceBands);
    PIK_ASSERT(params.num_eccentricity_bands <=
               DctQuantWeightParams::kMaxRadialBands);
    return encoding;
  }

  static QuantEncoding Raw(size_t block_dim, const float* xweights,
                           const float* yweights, const float* bweights) {
    QuantEncoding encoding;
    encoding.mode = kQuantModeRaw;
    encoding.block_dim = block_dim;
    PIK_ASSERT(block_dim == 1 || block_dim == 2 || block_dim == 4);
    memcpy(encoding.weights[0], xweights,
           block_dim * block_dim * kBlockDim * kBlockDim * sizeof(float));
    memcpy(encoding.weights[1], yweights,
           block_dim * block_dim * kBlockDim * kBlockDim * sizeof(float));
    memcpy(encoding.weights[2], bweights,
           block_dim * block_dim * kBlockDim * kBlockDim * sizeof(float));
    // Override LLF values in the quantization table with invalid values.
    for (size_t c = 0; c < 3; c++) {
      for (size_t y = 0; y < encoding.block_dim; y++) {
        for (size_t x = 0; x < encoding.block_dim; x++) {
          encoding.weights[c][y * encoding.block_dim * kBlockDim + x] = 0xBAD;
        }
      }
    }
    return encoding;
  }

  static QuantEncoding RawScaled(size_t block_dim, const float* base_weights,
                                 const float* scales) {
    QuantEncoding encoding;
    encoding.mode = kQuantModeRawScaled;
    encoding.block_dim = block_dim;
    PIK_ASSERT(block_dim == 1 || block_dim == 2 || block_dim == 4);
    memcpy(encoding.weights[0], base_weights,
           block_dim * block_dim * kBlockDim * kBlockDim * sizeof(float));
    memcpy(encoding.scales, scales, sizeof(encoding.scales));
    // Override LLF values in the quantization table with invalid values.
    for (size_t y = 0; y < encoding.block_dim; y++) {
      for (size_t x = 0; x < encoding.block_dim; x++) {
        encoding.weights[0][y * encoding.block_dim * kBlockDim + x] = 0xBAD;
      }
    }
    return encoding;
  }

  enum Mode {
    kQuantModeDefault,
    kQuantModeID,
    kQuantModeDCT2,
    kQuantModeDCT4,
    kQuantModeDCT,
    kQuantModeRaw,
    kQuantModeRawScaled,
  };
  Mode mode;

  // Only used for raw and raw scaled.
  uint32_t block_dim;

  // Raw weights. Uses only the first channel for raw scaled, all
  // three for Raw, unused otherwise. `scales` is only used for raw scaled.
  float weights[3][kMaxQuantTableSize];
  float scales[3];

  // Weights for identity.
  float idweights[3][3];

  // Weights for DCT2.
  float dct2weights[3][6];

  // Extra multipliers for coefficients 01/10 and 11 for DCT4.
  float dct4multipliers[3][2];

  // Weights for DCT4+ tables.
  DctQuantWeightParams dct_params;
};

class DequantMatrices {
 public:
  DequantMatrices(bool need_inv_matrices)
      : need_inv_matrices_(need_inv_matrices),
        encodings_({kNumQuantKinds, QuantEncoding::Default()}) {
    // Default quantization tables need to be valid.
    PIK_CHECK(Compute());
  }

  PIK_INLINE size_t MatrixOffset(size_t quant_kind, int c) const {
    return table_offsets_[quant_kind * 3 + c];
  }

  // Returns aligned memory.
  PIK_INLINE const float* Matrix(size_t quant_kind, int c) const {
    PIK_ASSERT(quant_kind < kNumQuantKinds);
    return &table_[MatrixOffset(quant_kind, c)];
  }

  PIK_INLINE const float* InvMatrix(size_t quant_kind, int c) const {
    PIK_ASSERT(quant_kind < kNumQuantKinds);
    return &inv_table_[MatrixOffset(quant_kind, c)];
  }

  size_t Size() const { return size_; }

  void SetCustom(const std::vector<QuantEncoding>& encodings) {
    // For now, we require a constant number of quantization tables.
    PIK_ASSERT(encodings.size() == kNumQuantKinds);
    encodings_ = encodings;
    // Called only in the encoder: should fail only for programmer errors.
    PIK_CHECK(Compute());
  }

  std::string Encode(PikImageSizeInfo* info) const;

  Status Decode(BitReader* br);

 private:
  Status Compute();
  static constexpr size_t kTableSize = kNumQuantKinds * 3 * kMaxQuantTableSize;
  alignas(64) float table_[kTableSize];
  alignas(64) float inv_table_[kTableSize];
  size_t table_offsets_[3 * kNumQuantKinds];
  bool need_inv_matrices_;
  size_t size_;
  std::vector<QuantEncoding> encodings_;
};

DequantMatrices FindBestDequantMatrices(float butteraugli_target,
                                        const Image3F& opsin);

}  // namespace pik

#endif  // QUANT_WEIGHTS_H_
