// Copyright 2018 Google Inc. All Rights Reserved.
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

#ifndef MULTIPASS_HANDLER_H_
#define MULTIPASS_HANDLER_H_

#include "codec.h"
#include "color_correlation.h"
#include "color_encoding.h"
#include "common.h"
#include "compressed_image_fwd.h"
#include "data_parallel.h"
#include "headers.h"
#include "image.h"
#include "noise.h"
#include "pik_params.h"
#include "quantizer.h"
#include "status.h"

namespace pik {

class MultipassManager;

/* MultipassHandler is a child object of MultipassManager. It is bound to
   specific group (see GetGroupHandler) and is used to perform operations over
   that group region. */
class MultipassHandler {
 public:
  virtual ~MultipassHandler() = default;

  // Save the ac strategy / quant field of this pass.
  virtual void SaveAcStrategy(const AcStrategyImage& af) {}
  virtual void SaveQuantField(const ImageI& qf) {}

  // Give a hint to the ac strategy / quant field encoder/decoder.
  virtual const AcStrategyImage* HintAcStrategy() { return nullptr; }
  virtual const ImageI* HintQuantField() { return nullptr; }

  virtual const Rect& GroupRect() = 0;
  virtual const Rect& PaddedGroupRect() = 0;
  Rect BlockGroupRect() {
    const Rect& r = PaddedGroupRect();
    return Rect(r.x0() / kBlockDim, r.y0() / kBlockDim, r.xsize() / kBlockDim,
                r.ysize() / kBlockDim);
  }

  virtual Status GetPreviousPass(const ColorEncoding& color_encoding,
                                 ThreadPool* pool, Image3F* out) {
    return true;
  }

  // Returns the MultipassManager this handler was created by.
  virtual MultipassManager* Manager() = 0;

  void SetDecoderQuantizer(Quantizer&& quantizer) {
    quantizer_ = std::move(quantizer);
  }

  Quantizer TakeDecoderQuantizer() { return std::move(quantizer_); }

 private:
  Quantizer quantizer_{kBlockDim, 0, 0, 0};
};

/* MultipassManager holds information about passes and manages
   MultipassHandlers. It is assumed that parallelization goes below the manager
   level (at group level), so all the methods of MultipassManager should be
   invoked from single thread. */
class MultipassManager {
 public:
  virtual ~MultipassManager() = default;

  // Modifies img, applying a transformation that reduces its entropy given a
  // reference image, typically the output of a previous pass.
  virtual void DecorrelateOpsin(Image3F* img) = 0;

  // Inverse of DecorrelateOpsin.
  virtual void RestoreOpsin(Image3F* img) = 0;

  // Updates/stores biases for AR on the final pass.
  virtual void UpdateBiases(Image3F* PIK_RESTRICT biases) {}
  virtual void StoreBiases(const Image3F& PIK_RESTRICT biases) {}

  // Called at the start of each pass.
  virtual void StartPass(const PassHeader& header) = 0;

  // Called by the decoder when a pass is done.
  virtual void SetDecodedPass(const Image3F& opsin) = 0;

  // This version is only called if we decoded a lossless pass.
  virtual void SetDecodedPass(CodecInOut* io) = 0;

  // Used *on the encoder only* to forcibly enable adaptive reconstruction in
  // GetQuantizer.
  virtual void UseAdaptiveReconstruction() {}

  // NOTE: not thread safe.
  // Preferably, `group_id` should be small non-negative number.
  // Same `group_rect` should be provided with corresponding `group_id`.
  virtual MultipassHandler* GetGroupHandler(size_t group_id,
                                            const Rect& group_rect) = 0;

  // Methods to retrieve color correlation, ac strategy and quantizer.
  virtual void GetColorCorrelationMap(const Image3F& opsin,
                                      ColorCorrelationMap* cmap) = 0;

  virtual void GetAcStrategy(float butteraugli_target,
                             const ImageF* quant_field, const Image3F* src,
                             const Image3F* coeffs_init, ThreadPool* pool,
                             AcStrategyImage* ac_strategy,
                             PikInfo* aux_out) = 0;

  virtual std::shared_ptr<Quantizer> GetQuantizer(
      const CompressParams& cparams, size_t xsize_blocks, size_t ysize_blocks,
      const Image3F& opsin_orig, const Image3F& opsin,
      const NoiseParams& noise_params, const PassHeader& pass_header,
      const GroupHeader& header, const ColorCorrelationMap& cmap,
      const AcStrategyImage& ac_strategy, ImageF& quant_field, ThreadPool* pool,
      PikInfo* aux_out) = 0;

  // Remove information from the image, just before encoding.
  virtual void StripInfo(EncCache* cache) {}
  virtual void StripInfoBeforePredictions(EncCache* cache) {}
};

}  // namespace pik

#endif  // MULTIPASS_HANDLER_H_
