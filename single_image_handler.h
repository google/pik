// Copyright 2018 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef SINGLE_PASS_HANDLER_H_
#define SINGLE_PASS_HANDLER_H_

#include <cstddef>
#include <memory>
#include <vector>

#include "ac_strategy.h"
#include "adaptive_quantization.h"
#include "codec.h"
#include "color_correlation.h"
#include "image.h"
#include "multipass_handler.h"

// A multipass handler/manager to encode single images. It will run heuristics
// for quantization, AC strategy and color correlation map only the first time
// we want to encode a lossy pass, and will then re-use the existing heuristics
// for further passes. All the passes of a single image are added together.

namespace pik {

constexpr size_t kMaxNumPasses = 8;

struct PassDefinition {
  // Side of the square of the coefficients that should be kept in each 8x8
  // block. Must be greater than 1, and at most 8. Should be in non-decreasing
  // order.
  size_t num_coefficients;
  // Whether or not we should include only salient blocks.
  // TODO(veluca): ignored for now.
  bool salient_only;
};

struct ProgressiveMode {
  size_t num_passes = 1;
  PassDefinition passes[kMaxNumPasses] = {{8, false}};

  ProgressiveMode() {}

  template <size_t nump>
  ProgressiveMode(const PassDefinition (&p)[nump]) {
    PIK_ASSERT(nump <= kMaxNumPasses);
    num_passes = nump;
    size_t last_ncoeff = 1;
    bool was_salient_only = false;
    for (size_t i = 0; i < nump; i++) {
      PIK_ASSERT(p[i].num_coefficients > last_ncoeff ||
                 (p[i].num_coefficients == last_ncoeff && !p[i].salient_only &&
                  was_salient_only));
      last_ncoeff = p[i].num_coefficients;
      was_salient_only = p[i].salient_only;
      passes[i] = p[i];
    }
  }
};

class SingleImageManager;

class SingleImageHandler : public MultipassHandler {
 public:
  SingleImageHandler(SingleImageManager* manager, const Rect& group_rect,
                     ProgressiveMode mode)
      : manager_(manager),
        group_rect_(group_rect),
        padded_group_rect_(group_rect.x0(), group_rect.y0(),
                           DivCeil(group_rect.xsize(), kBlockDim) * kBlockDim,
                           DivCeil(group_rect.ysize(), kBlockDim) * kBlockDim),
        mode_(mode) {}

  const Rect& GroupRect() override { return group_rect_; }
  const Rect& PaddedGroupRect() override { return padded_group_rect_; };

  std::vector<Image3S> SplitACCoefficients(
      Image3S&& ac, const AcStrategyImage& ac_strategy) override;

  MultipassManager* Manager() override;

 private:
  SingleImageManager* manager_;
  const Rect group_rect_;
  const Rect padded_group_rect_;
  ProgressiveMode mode_;
};

// A MultipassManager for single images.
class SingleImageManager : public MultipassManager {
 public:
  SingleImageManager() { group_handlers_.reserve(16); }

  void StartPass(const PassHeader& pass_header) override {
    current_header_ = pass_header;
  }

  void SetDecodedPass(const Image3F& opsin) override {}
  void SetDecodedPass(CodecInOut* io) override {}
  void DecorrelateOpsin(Image3F* img) override {}
  void RestoreOpsin(Image3F* img) override {}

  void SetProgressiveMode(ProgressiveMode mode) { mode_ = mode; }

  void SetSaliencyMap(std::shared_ptr<ImageF> saliency_map) {
    saliency_map_ = saliency_map;
  }

  void UseAdaptiveReconstruction() override {
    use_adaptive_reconstruction_ = true;
  }

  MultipassHandler* GetGroupHandler(size_t group_id,
                                    const Rect& group_rect) override;

  BlockDictionary GetBlockDictionary(double butteraugli_target,
                                     const Image3F& opsin) override;

  void GetColorCorrelationMap(const Image3F& opsin,
                              const DequantMatrices& dequant,
                              ColorCorrelationMap* cmap) override;

  void GetAcStrategy(float butteraugli_target, const ImageF* quant_field,
                     const DequantMatrices& dequant, const Image3F& src,
                     ThreadPool* pool, AcStrategyImage* ac_strategy,
                     PikInfo* aux_out) override;

  std::shared_ptr<Quantizer> GetQuantizer(
      const CompressParams& cparams, size_t xsize_blocks, size_t ysize_blocks,
      const Image3F& opsin_orig, const Image3F& opsin,
      const PassHeader& pass_header, const GroupHeader& header,
      const ColorCorrelationMap& cmap, const BlockDictionary& block_dictionary,
      const AcStrategyImage& ac_strategy, const ImageB& ar_sigma_lut_ids,
      const DequantMatrices* dequant, ImageF& quant_field, ThreadPool* pool,
      PikInfo* aux_out) override;

  size_t GetNumPasses() override { return mode_.num_passes; }

 private:
  friend class SingleImageHandler;

  float BlockSaliency(size_t row, size_t col) const;

  PassHeader current_header_;
  ProgressiveMode mode_;
  bool use_adaptive_reconstruction_ = false;

  std::shared_ptr<ImageF> saliency_map_;

  std::shared_ptr<Quantizer> quantizer_;
  bool has_quantizer_ = false;
  ColorCorrelationMap cmap_;
  bool has_cmap_ = false;
  AcStrategyImage ac_strategy_;
  bool has_ac_strategy_ = false;

  std::vector<std::unique_ptr<SingleImageHandler>> group_handlers_;
};

}  // namespace pik

#endif  // SINGLE_PASS_HANDLER_H_
