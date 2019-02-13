// Copyright 2018 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef MULTIPASS_HANDLER_H_
#define MULTIPASS_HANDLER_H_

#include "ac_strategy.h"
#include "block_dictionary.h"
#include "codec.h"
#include "color_correlation.h"
#include "color_encoding.h"
#include "common.h"
#include "compressed_image_fwd.h"
#include "data_parallel.h"
#include "headers.h"
#include "image.h"
#include "pik_params.h"
#include "quantizer.h"
#include "status.h"

// Defines how multi-pass images should be encoded and decoded.

namespace pik {

class MultipassManager;

// MultipassHandler is a child object of MultipassManager. It is bound to
// specific group (see GetGroupHandler) and is used to perform operations over
// that group region.
class MultipassHandler {
 public:
  virtual ~MultipassHandler() = default;

  virtual const Rect& GroupRect() = 0;
  virtual const Rect& PaddedGroupRect() = 0;
  Rect BlockGroupRect() {
    const Rect& r = PaddedGroupRect();
    return Rect(r.x0() / kBlockDim, r.y0() / kBlockDim, r.xsize() / kBlockDim,
                r.ysize() / kBlockDim);
  }

  // Progressive mode.
  virtual std::vector<Image3S> SplitACCoefficients(
      Image3S&& ac, const AcStrategyImage& ac_strategy) {
    std::vector<Image3S> ret;
    ret.push_back(std::move(ac));
    return ret;
  }

  // Returns the MultipassManager this handler was created by.
  virtual MultipassManager* Manager() = 0;

 private:
  DequantMatrices default_matrices_{/*need_inv_matrices=*/false};
  Quantizer quantizer_{&default_matrices_, 0, 0};
};

// MultipassManager holds information about passes and manages
// MultipassHandlers. It is assumed that parallelization goes below the manager
// level (at group level), so all the methods of MultipassManager should be
// invoked from a single thread.
class MultipassManager {
 public:
  virtual ~MultipassManager() = default;

  // Modifies img, applying a transformation that reduces its entropy given a
  // reference image, typically the output of a previous pass.
  virtual void DecorrelateOpsin(Image3F* img) = 0;

  // Inverse of DecorrelateOpsin.
  virtual void RestoreOpsin(Image3F* img) = 0;

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

  // Methods to retrieve color correlation, ac strategy, quantizer, block
  // dictionary and dequant matrices.
  virtual DequantMatrices GetDequantMatrices(double butteraugli_target,
                                             const Image3F& opsin) {
    return FindBestDequantMatrices(butteraugli_target, opsin);
  }

  virtual BlockDictionary GetBlockDictionary(double butteraugli_target,
                                             const Image3F& opsin) = 0;

  virtual void GetColorCorrelationMap(const Image3F& opsin,
                                      const DequantMatrices& dequant,
                                      ColorCorrelationMap* cmap) = 0;

  virtual void GetAcStrategy(float butteraugli_target,
                             const ImageF* quant_field,
                             const DequantMatrices& dequant, const Image3F& src,
                             ThreadPool* pool, AcStrategyImage* ac_strategy,
                             PikInfo* aux_out) = 0;

  virtual std::shared_ptr<Quantizer> GetQuantizer(
      const CompressParams& cparams, size_t xsize_blocks, size_t ysize_blocks,
      const Image3F& opsin_orig, const Image3F& opsin,
      const PassHeader& pass_header, const GroupHeader& header,
      const ColorCorrelationMap& cmap, const BlockDictionary& block_dictionary,
      const AcStrategyImage& ac_strategy, const ImageB& ar_sigma_lut_ids,
      const DequantMatrices* dequant, ImageF& quant_field, ThreadPool* pool,
      PikInfo* aux_out) = 0;

  virtual size_t GetNumPasses() { return 1; }

  // Save the ac strategy / quant field of this pass.
  virtual void SaveAcStrategy(const AcStrategyImage& af) {}
  virtual void SaveQuantField(const ImageI& qf) {}

  // Give a hint to the ac strategy / quant field encoder/decoder.
  virtual const AcStrategyImage* HintAcStrategy() { return nullptr; }
  virtual const ImageI* HintQuantField() { return nullptr; }

  // Previous pass in a specific colorspace.
  virtual Status GetPreviousPass(const ColorEncoding& color_encoding,
                                 ThreadPool* pool, Image3F* out) {
    return true;
  }
};

}  // namespace pik

#endif  // MULTIPASS_HANDLER_H_
