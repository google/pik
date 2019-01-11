// Copyright 2018 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef AC_STRATEGY_H_
#define AC_STRATEGY_H_

#include <stdint.h>
#include "common.h"
#include "data_parallel.h"
#include "dct.h"
#include "image.h"
#include "pik_info.h"
#include "quantizer.h"

// Defines the different kinds of transforms, and heuristics to choose between
// them.
// `AcStrategy` represents what transform should be used, and which sub-block of
// that transform we are currently in. Note that DCT4x4 is applied on all four
// 4x4 sub-blocks of an 8x8 block.
// `AcStrategyImage` defines which strategy should be used for each 8x8 block
// of the image. The highest 4 bits represent the strategy to be used, the
// lowest 4 represent the index of the block inside that strategy. Blocks should
// be aligned, i.e. 32x32 blocks should only start in positions that are
// multiples of 32.
// `FindBestAcStrategy` uses heuristics to choose which AC strategy should be
// used in each block.

namespace pik {

class AcStrategy {
 public:
  // Extremal values for the number of blocks/coefficients of a single strategy.
  static constexpr size_t kMaxCoeffBlocks = 4;
  static constexpr size_t kMaxBlockDim = kBlockDim * kMaxCoeffBlocks;
  static constexpr size_t kMaxCoeffArea = kMaxBlockDim * kMaxBlockDim;

  // Raw strategy types.
  enum class Type : uint32_t {
    // Regular block size DCT (value matches kQuantKind)
    DCT = 0,
    // Encode pixels without transforming (value matches kQuantKind)
    IDENTITY = 1,
    // Use 4-by-4 DCT (value matches kQuantKind)
    DCT4X4 = 2,
    // Use 16-by-16 DCT
    DCT16X16 = 3,
    // Use 32-by-32 DCT
    DCT32X32 = 4,
  };

  // Returns true if this block is the first 8x8 block (i.e. top-left) of a
  // possibly multi-block strategy.
  bool IsFirstBlock() const { return block_ == 0; }

  // Returns the raw strategy value. Should only be used for tokenization.
  uint8_t RawStrategy() const { return static_cast<uint8_t>(strategy_); }

  Type Strategy() const { return strategy_; }

  // Inverse check
  static bool IsRawStrategyValid(uint8_t raw_strategy) {
    return raw_strategy <= 4;
  }
  static AcStrategy FromRawStrategy(uint8_t raw_strategy) {
    return AcStrategy((Type)raw_strategy, 0);
  }

  // Get the quant kind for this type of strategy. A non-zero block parameter
  // overrides block_.
  PIK_INLINE size_t GetQuantKind(size_t block = 0) const {
#ifdef ADDRESS_SANITIZER
    PIK_ASSERT(block == 0 || block_ == 0);
#endif
    if (block == 0) block = block_;

    if (strategy_ == Type::DCT16X16) {
#ifdef ADDRESS_SANITIZER
      PIK_ASSERT(block <= 3);
#endif
      return block + kQuantKindDCT16Start;
    }

    if (strategy_ == Type::DCT32X32) {
#ifdef ADDRESS_SANITIZER
      PIK_ASSERT(block <= 15);
#endif
      return block + kQuantKindDCT32Start;
    }

    static_assert(kQuantKindDCT8 == size_t(Type::DCT), "QuantKind != type");
    static_assert(kQuantKindID == size_t(Type::IDENTITY), "QuantKind != type");
    static_assert(kQuantKindDCT4 == size_t(Type::DCT4X4), "QuantKind != type");
    return static_cast<size_t>(strategy_);
  }

  float ARQuantScale() const {
    if (strategy_ == Type::DCT32X32) return 0.71122376f;
    if (strategy_ == Type::DCT16X16) return 0.827516904f;
    // TODO(veluca): find better value.
    if (strategy_ == Type::DCT4X4) return 1.2f;
    return 1.0f;
  }

  // Number of 8x8 blocks that this strategy will cover. 0 for non-top-left
  // blocks inside a multi-block transform.
  size_t covered_blocks_x() const {
    if (strategy_ == Type::DCT32X32) return block_ == 0 ? 4 : 0;
    if (strategy_ == Type::DCT16X16) return block_ == 0 ? 2 : 0;
    return 1;
  }
  size_t covered_blocks_y() const {
    if (strategy_ == Type::DCT32X32) return block_ == 0 ? 4 : 0;
    if (strategy_ == Type::DCT16X16) return block_ == 0 ? 2 : 0;
    return 1;
  }

  // 1 / covered_block_x() / covered_block_y(), for fast division.
  // Should only be called with block_ == 0.
  float inverse_covered_blocks() const {
#ifdef ADDRESS_SANITIZER
    PIK_ASSERT(block_ == 0);
#endif
    if (strategy_ == Type::DCT32X32) return 1.0f / 16;
    if (strategy_ == Type::DCT16X16) return 0.25f;
    return 1.0f;
  }

  float InverseNumACCoefficients() const {
#ifdef ADDRESS_SANITIZER
    PIK_ASSERT(block_ == 0);
#endif
    if (strategy_ == Type::DCT32X32) return 1.0f / (32 * 32 - 16);
    if (strategy_ == Type::DCT16X16) return 1.0f / (16 * 16 - 4);
    return 1.0f / (8 * 8 - 1);
  }

  float ARLowestFrequencyScale(size_t bx, size_t by) {
    switch (strategy_) {
      case Type::IDENTITY:
      case Type::DCT:
      case Type::DCT4X4:
        return 1.0f;
      case Type::DCT16X16: {
        return 2 * kBlockDim * IDCTScales<2 * kBlockDim>()[bx] *
               IDCTScales<2 * kBlockDim>()[by] * L1Norm<2 * kBlockDim>()[bx] *
               L1Norm<2 * kBlockDim>()[by];
      }
      case Type::DCT32X32: {
        return 4 * kBlockDim * IDCTScales<4 * kBlockDim>()[bx] *
               IDCTScales<4 * kBlockDim>()[by] * L1Norm<4 * kBlockDim>()[bx] *
               L1Norm<4 * kBlockDim>()[by];
      }
    }
  }

  // Pixel to coefficients and vice-versa
  SIMD_ATTR void TransformFromPixels(const float* pixels, size_t pixels_stride,
                                     float* coefficients,
                                     size_t coefficients_stride) const;
  SIMD_ATTR void TransformToPixels(const float* coefficients,
                                   size_t coefficients_stride, float* pixels,
                                   size_t pixels_stride) const;

  // Same as above, but for DC image.
  SIMD_ATTR void LowestFrequenciesFromDC(const float* PIK_RESTRICT dc,
                                         size_t dc_stride, float* llf,
                                         size_t llf_stride) const;
  SIMD_ATTR void DCFromLowestFrequencies(const float* PIK_RESTRICT block,
                                         size_t block_stride, float* dc,
                                         size_t dc_stride) const;

  // Produces a 2x2-upsampled DC block out of the lowest frequencies
  // (block_size/8) of the image.
  SIMD_ATTR void DC2x2FromLowestFrequencies(const float* PIK_RESTRICT llf,
                                            size_t llf_stride,
                                            float* PIK_RESTRICT dc2x2,
                                            size_t dc2x2_stride) const;

  // Produces the low frequencies (block_size/4) of the images out of a 2x2
  // upsampled DC image, and vice-versa.
  SIMD_ATTR void DC2x2FromLowFrequencies(const float* block,
                                         size_t block_stride, float* dc2x2,
                                         size_t dc2x2_stride) const;
  SIMD_ATTR void LowFrequenciesFromDC2x2(const float* dc2x2,
                                         size_t dc2x2_stride, float* block,
                                         size_t block_stride) const;

  AcStrategy(Type strategy, uint32_t block)
      : strategy_(strategy), block_(block) {
    PIK_ASSERT(strategy == Type::DCT16X16 || strategy == Type::DCT32X32 ||
               block == 0);
    PIK_ASSERT(strategy == Type::DCT32X32 || block < 4);
    PIK_ASSERT(block < 16);
  }

 private:
  Type strategy_;
  uint32_t block_;
};

// Class to use a certain row of the AC strategy.
class AcStrategyRow {
 public:
  AcStrategyRow(const uint8_t* row, size_t y) : row_(row), y_(y) {}
  AcStrategy operator[](size_t x) const {
    return AcStrategy((AcStrategy::Type)(row_[x] >> 4), row_[x] & 0xF);
  }

 private:
  const uint8_t* PIK_RESTRICT row_;
  size_t y_;
};

class AcStrategyImage {
 public:
  // A value that does not represent a valid combined AC strategy value.
  // Used as a sentinel in DecodeAcStrategy.
  static constexpr uint8_t INVALID = 0xF;

  AcStrategyImage() {}
  AcStrategyImage(size_t xsize, size_t ysize) : layers_(xsize, ysize) {
    FillImage((uint8_t)AcStrategy::Type::DCT, &layers_);
  }
  AcStrategyImage(AcStrategyImage&&) = default;
  AcStrategyImage& operator=(AcStrategyImage&&) = default;

  void SetFromRaw(const Rect& rect, const ImageB& raw_layers);

  AcStrategyRow ConstRow(size_t y, size_t x_prefix = 0) const {
    return AcStrategyRow(layers_.ConstRow(y) + x_prefix, y);
  }

  AcStrategyRow ConstRow(const Rect& rect, size_t y) const {
    return ConstRow(rect.y0() + y, rect.x0());
  }

  const ImageB& ConstRaw() const { return layers_; }

  size_t xsize() const { return layers_.xsize(); }
  size_t ysize() const { return layers_.ysize(); }

  AcStrategyImage Copy(const Rect& rect) const {
    AcStrategyImage copy;
    copy.layers_ = CopyImage(rect, layers_);
    return copy;
  }
  AcStrategyImage Copy() const { return Copy(Rect(layers_)); }

  // Count the number of blocks of a given type.
  size_t CountBlocks(AcStrategy::Type type) const;

 private:
  ImageB layers_;
};

// `quant_field` is an initial quantization field for this image. `src` is the
// input image in the XYB color space. `ac_strategy` is the output strategy.
SIMD_ATTR void FindBestAcStrategy(float butteraugli_target,
                                  const ImageF* quant_field, const Image3F& src,
                                  ThreadPool* pool,
                                  AcStrategyImage* ac_strategy,
                                  PikInfo* aux_out);

}  // namespace pik

#endif  // AC_STRATEGY_H_
