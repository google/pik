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

#include "adaptive_quantization.h"
#include "color_correlation.h"
#include "image.h"
#include "multipass_handler.h"

namespace pik {

enum class ProgressiveMode {
  kFull,
  kDcOnly,
  kLfOnly,
  kSalientHfOnly,
  kNonSalientHfOnly,
  kHfOnly
};

class SingleImageManager;

class SingleImageHandler : public MultipassHandler {
 public:
  SingleImageHandler(SingleImageManager* manager, const Rect& group_rect)
      : manager_(manager),
        group_rect_(group_rect),
        padded_group_rect_(group_rect.x0(), group_rect.y0(),
                           DivCeil(group_rect.xsize(), kBlockDim) * kBlockDim,
                           DivCeil(group_rect.ysize(), kBlockDim) * kBlockDim) {
  }

  void SaveAcStrategy(const AcStrategyImage& as) override;
  void SaveQuantField(const ImageI& qf) override;

  const AcStrategyImage* HintAcStrategy() override;
  const ImageI* HintQuantField() override;

  const Rect& GroupRect() override { return group_rect_; }
  const Rect& PaddedGroupRect() override { return padded_group_rect_; };

  Status GetPreviousPass(const ColorEncoding& color_encoding, ThreadPool* pool,
                         Image3F* out) override;

  MultipassManager* Manager() override;

 private:
  SingleImageManager* manager_;
  const Rect group_rect_;
  const Rect padded_group_rect_;

  AcStrategyImage ac_strategy_hint_;
  ImageI quant_field_hint_;
};

// A MultipassManager for single images.
class SingleImageManager : public MultipassManager {
 public:
  SingleImageManager() { group_handlers_.reserve(16); }

  void StartPass(const PassHeader& pass_header) override {
    current_header_ = pass_header;
  }

  void DecorrelateOpsin(Image3F* img) override;
  void RestoreOpsin(Image3F* img) override;

  void UpdateBiases(Image3F* PIK_RESTRICT biases) override;
  void StoreBiases(const Image3F& PIK_RESTRICT biases) override;

  void SetDecodedPass(const Image3F& opsin) override;
  void SetDecodedPass(CodecInOut* io) override;

  bool IsLastPass() const { return current_header_.is_last; }

  void SetProgressiveMode(ProgressiveMode mode) { mode_ = mode; }

  void SetSaliencyMap(std::shared_ptr<ImageF> saliency_map) {
    saliency_map_ = saliency_map;
  }

  void UseAdaptiveReconstruction() override {
    use_adaptive_reconstruction_ = true;
  }

  MultipassHandler* GetGroupHandler(size_t group_id,
                                    const Rect& group_rect) override;

  void GetColorCorrelationMap(const Image3F& opsin,
                              ColorCorrelationMap* cmap) override;

  void GetAcStrategy(float butteraugli_target, const ImageF* quant_field,
                     const Image3F& src, ThreadPool* pool,
                     AcStrategyImage* ac_strategy, PikInfo* aux_out) override;

  std::shared_ptr<Quantizer> GetQuantizer(
      const CompressParams& cparams, size_t xsize_blocks, size_t ysize_blocks,
      const Image3F& opsin_orig, const Image3F& opsin,
      const NoiseParams& noise_params, const PassHeader& pass_header,
      const GroupHeader& header, const ColorCorrelationMap& cmap,
      const AcStrategyImage& ac_strategy, ImageF& quant_field, ThreadPool* pool,
      PikInfo* aux_out) override;

  void StripInfo(EncCache* cache) override;
  void StripInfoBeforePredictions(EncCache* cache) override;
  void StripDCInfo(PassEncCache* cache) override;

 private:
  friend class SingleImageHandler;

  float BlockSaliency(size_t row, size_t col) const;

  PassHeader current_header_;
  size_t num_passes_ = 0;
  Image3F previous_pass_;
  Image3F last_pass_biases_;
  ProgressiveMode mode_ = ProgressiveMode::kFull;
  bool use_adaptive_reconstruction_ = false;

  std::shared_ptr<ImageF> saliency_map_;

  std::shared_ptr<Quantizer> quantizer_;
  ColorCorrelationMap cmap_;
  AcStrategyImage ac_strategy_;

  std::vector<std::unique_ptr<SingleImageHandler>> group_handlers_;
};

}  // namespace pik

#endif  // SINGLE_PASS_HANDLER_H_
