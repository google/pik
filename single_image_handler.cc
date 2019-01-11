// Copyright 2018 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "single_image_handler.h"
#include "adaptive_quantization.h"
#include "codec.h"
#include "color_correlation.h"
#include "common.h"
#include "image.h"
#include "opsin_image.h"
#include "opsin_inverse.h"
#include "pik_params.h"
#include "profiler.h"
#include "quantizer.h"

namespace pik {

MultipassHandler* SingleImageManager::GetGroupHandler(size_t group_id,
                                                      const Rect& group_rect) {
  if (group_handlers_.size() <= group_id) {
    group_handlers_.resize(group_id + 1);
  }
  if (!group_handlers_[group_id]) {
    group_handlers_[group_id].reset(new SingleImageHandler(this, group_rect));
  }
  return group_handlers_[group_id].get();
}

float SingleImageManager::BlockSaliency(size_t row, size_t col) const {
  auto saliency_map = saliency_map_.get();
  if (saliency_map == nullptr) return 0.0f;
  return saliency_map->Row(row)[col];
}

void SingleImageManager::SetDecodedPass(CodecInOut* io) {
  if (current_header_.is_last) return;
  previous_pass_ =
      PadImageToMultiple(OpsinDynamicsImage(io, Rect(io->color())), kBlockDim);
  if (current_header_.gaborish != GaborishStrength::kOff) {
    previous_pass_ = GaborishInverse(previous_pass_, 0.92718927264540152);
  }
  num_passes_++;
}

void SingleImageManager::SetDecodedPass(const Image3F& opsin) {
  if (current_header_.is_last) return;
  previous_pass_ = CopyImage(opsin);
  num_passes_++;
}

void SingleImageManager::DecorrelateOpsin(Image3F* img) {
  if (num_passes_ == 0) return;
  PIK_ASSERT(SameSize(*img, previous_pass_));
  SubtractFrom(previous_pass_, img);
}

void SingleImageManager::RestoreOpsin(Image3F* img) {
  if (num_passes_ == 0) return;
  PIK_ASSERT(SameSize(*img, previous_pass_));
  AddTo(previous_pass_, img);
}

void SingleImageManager::GetColorCorrelationMap(const Image3F& opsin,
                                                ColorCorrelationMap* cmap) {
  if (!has_cmap_) {
    cmap_ = std::move(*cmap);
    FindBestColorCorrelationMap(opsin, &cmap_);
    has_cmap_ = true;
  }
  *cmap = cmap_.Copy();
}

void SingleImageManager::GetAcStrategy(float butteraugli_target,
                                       const ImageF* quant_field,
                                       const Image3F& src, ThreadPool* pool,
                                       AcStrategyImage* ac_strategy,
                                       PikInfo* aux_out) {
  if (!has_ac_strategy_) {
    FindBestAcStrategy(butteraugli_target, quant_field, src, pool,
                       &ac_strategy_, aux_out);
    has_ac_strategy_ = true;
  }
  *ac_strategy = ac_strategy_.Copy();
}

std::shared_ptr<Quantizer> SingleImageManager::GetQuantizer(
    const CompressParams& cparams, size_t xsize_blocks, size_t ysize_blocks,
    const Image3F& opsin_orig, const Image3F& opsin,
    const NoiseParams& noise_params, const PassHeader& pass_header,
    const GroupHeader& header, const ColorCorrelationMap& cmap,
    const AcStrategyImage& ac_strategy, ImageF& quant_field, ThreadPool* pool,
    PikInfo* aux_out) {
  if (!has_quantizer_) {
    PassHeader hdr = pass_header;
    if (use_adaptive_reconstruction_) {
      hdr.have_adaptive_reconstruction = true;
    }
    quantizer_ = FindBestQuantizer(
        cparams, xsize_blocks, ysize_blocks, opsin_orig, opsin, noise_params,
        hdr, header, cmap, ac_strategy, quant_field, pool, aux_out, this);
    has_quantizer_ = true;
  }
  return quantizer_;
}

void SingleImageManager::StripInfo(EncCache* cache) {
  const constexpr size_t kBlockSize = kBlockDim * kBlockDim;
  switch (mode_) {
    case ProgressiveMode::kDcOnly: {
      FillImage(int16_t(0), &cache->ac);
      break;
    }
    case ProgressiveMode::kLfOnly: {
      for (size_t c = 0; c < cache->ac.kNumPlanes; c++) {
        for (size_t by = 0; by < cache->ac_strategy.ysize(); by++) {
          int16_t* PIK_RESTRICT row = cache->ac.PlaneRow(c, by);
          for (size_t bx = 0; bx < cache->ac_strategy.xsize(); bx++) {
            int16_t* PIK_RESTRICT block = row + kBlockSize * bx;
            for (size_t i = 2; i < kBlockSize; i++) {
              if (i != kBlockDim && i != kBlockDim + 1) block[i] = 0;
            }
          }
        }
      }
      break;
    }
    case ProgressiveMode::kSalientHfOnly:
    case ProgressiveMode::kNonSalientHfOnly: {
      // High frequency components for progressive mode with saliency.
      // For Salient Hf pass, null out non-salient blocks.
      // For Non-Salient Hf pass, null out everything if we debug-skip
      // non-salient data.
      bool null_out_everything = false;
      if (mode_ == ProgressiveMode::kNonSalientHfOnly) {
        if (cache->saliency_debug_skip_nonsalient) {
          null_out_everything = true;
        } else {
          break;  // Nothing else to do for this non-salient Hf pass.
        }
      }
      for (size_t c = 0; c < cache->ac.kNumPlanes; c++) {
        for (size_t by = 0; by < cache->ac_strategy.ysize(); by++) {
          int16_t* PIK_RESTRICT row = cache->ac.PlaneRow(c, by);
          for (size_t bx = 0; bx < cache->ac_strategy.xsize(); bx++) {
            int16_t* PIK_RESTRICT block = row + kBlockSize * bx;
            if (null_out_everything ||
                !(BlockSaliency(by, bx) > cache->saliency_threshold)) {
              // For Salient Hf, null out non-salient blocks.
              // For Non-Salient Hf, null out all blocks if requested by debug
              // flag.
              std::fill(block, block + kBlockSize, 0);
            }
          }
        }
      }
      break;
    }
    case ProgressiveMode::kHfOnly:
    case ProgressiveMode::kFull: {
      break;
    }
  }
}

void SingleImageManager::StripInfoBeforePredictions(EncCache* cache) {
  const constexpr size_t kBlockSize = kBlockDim * kBlockDim;
  switch (mode_) {
    case ProgressiveMode::kDcOnly: {
      break;
    }
    case ProgressiveMode::kLfOnly: {
      break;
    }
    case ProgressiveMode::kHfOnly:
    case ProgressiveMode::kNonSalientHfOnly:
    case ProgressiveMode::kSalientHfOnly: {
      for (size_t c = 0; c < cache->ac.kNumPlanes; c++) {
        for (size_t by = 0; by < cache->ac_strategy.ysize(); by++) {
          float* PIK_RESTRICT row = cache->coeffs.PlaneRow(c, by);
          for (size_t bx = 0; bx < cache->ac_strategy.xsize(); bx++) {
            float* PIK_RESTRICT block = row + kBlockSize * bx;
            // Zero out components that should be zero but might not quite be so
            // e.g. due to numerical noise.
            //
            // Null out only DC and Low-Frequency components
            // (0, 0), (0, 1), (1, 0), (1, 1).
            block[0] = block[1] = 0.0f;
            block[kBlockDim] = block[kBlockDim + 1] = 0.0f;
            if (mode_ == ProgressiveMode::kNonSalientHfOnly &&
                (cache->saliency_debug_skip_nonsalient ||
                 (BlockSaliency(by, bx) > cache->saliency_threshold)))
              std::fill(block, block + kBlockSize, 0);
          }
        }
      }
      break;
    }
    case ProgressiveMode::kFull: {
      break;
    }
  }
}

void SingleImageManager::StripDCInfo(PassEncCache* cache) {
  switch (mode_) {
    case ProgressiveMode::kDcOnly: {
      break;
    }
    case ProgressiveMode::kLfOnly: {
      FillImage(0.0f, &cache->dc_dec);
      FillImage(int16_t(0), &cache->dc);
      break;
    }
    case ProgressiveMode::kHfOnly:
    case ProgressiveMode::kNonSalientHfOnly:
    case ProgressiveMode::kSalientHfOnly: {
      FillImage(0.0f, &cache->dc_dec);
      FillImage(int16_t(0), &cache->dc);
      break;
    }
    case ProgressiveMode::kFull: {
      break;
    }
  }
}

void SingleImageManager::UpdateBiases(Image3F* PIK_RESTRICT biases) {
  PROFILER_ZONE("SingleImage UpdateBiases");
  if (num_passes_ == 0) return;
  AddTo(last_pass_biases_, biases);
}

void SingleImageManager::StoreBiases(const Image3F& PIK_RESTRICT biases) {
  if (!IsLastPass()) {
    last_pass_biases_ = CopyImage(biases);
  }
}

void SingleImageHandler::SaveAcStrategy(const AcStrategyImage& as) {
  if (manager_->IsLastPass()) return;
  ac_strategy_hint_ = as.Copy(BlockGroupRect());
}

void SingleImageHandler::SaveQuantField(const ImageI& qf) {
  if (manager_->IsLastPass()) return;
  quant_field_hint_ = CopyImage(BlockGroupRect(), qf);
}

const AcStrategyImage* SingleImageHandler::HintAcStrategy() {
  if (ac_strategy_hint_.xsize() == 0 || ac_strategy_hint_.ysize() == 0)
    return nullptr;
  return &ac_strategy_hint_;
}

const ImageI* SingleImageHandler::HintQuantField() {
  if (quant_field_hint_.xsize() == 0 || quant_field_hint_.ysize() == 0)
    return nullptr;
  return &quant_field_hint_;
}

Status SingleImageHandler::GetPreviousPass(const ColorEncoding& color_encoding,
                                           ThreadPool* pool, Image3F* out) {
  if (manager_->num_passes_ == 0) return true;
  CodecContext ctx;
  CodecInOut io(&ctx);
  Rect group_rect = GroupRect();
  Image3F opsin = CopyImage(group_rect, manager_->previous_pass_);
  opsin = ConvolveGaborish(std::move(opsin), manager_->current_header_.gaborish,
                           /*pool=*/nullptr);
  Image3F linear(opsin.xsize(), opsin.ysize());
  OpsinToLinear(opsin, Rect(opsin), &linear);
  io.SetFromImage(std::move(linear),
                  ctx.c_linear_srgb[color_encoding.IsGray()]);
  PIK_RETURN_IF_ERROR(io.TransformTo(color_encoding, pool));
  // TODO(veluca): avoid this copy.
  *out = CopyImage(io.color());
  return true;
}

MultipassManager* SingleImageHandler::Manager() { return manager_; }
}  // namespace pik
