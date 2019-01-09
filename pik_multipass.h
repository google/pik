// Copyright 2018 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_MULTIPASS_H_
#define PIK_MULTIPASS_H_

#include "codec.h"
#include "multipass_handler.h"
#include "padded_bytes.h"
#include "pik_info.h"
#include "pik_params.h"
#include "pik_pass.h"

namespace pik {

// Encodes a stream of images as a PIK multipass image. The first pass, and then
// other frames at an interval of kKeyFrameInterval passes, are stored as normal
// PIK images. Other passes are stored as their difference (in xyb color space),
// PIK-encoded, with respect to the previous pass. More in detail, after
// encoding each frame, the encoder runs a step of the decoder and stores the
// resulting image as a reference for the following pass.
class PikMultipassEncoder {
 public:
  PikMultipassEncoder(const FileHeader& container, PaddedBytes* output,
                      MultipassManager* manager, PikInfo* info = nullptr);

  // On the first call, creates the output container and adds the first pass to
  // it. On subsequent calls, just appends the pass. We build the container
  // only when the first pass is ready, as some information in the pass is
  // required.
  Status AddPass(const CompressParams& params, const PassParams& pass_params,
                 const CodecInOut* frame, ThreadPool* pool);

  // Closes the stream. After calling this, calls to AddPass are invalid,
  // and output contains a valid encoding of all the given passes.
  Status Finalize();

 private:
  Status SetupContainer(const CodecInOut* first_frame);
  void ExtendOutput(size_t bits);

  PaddedBytes* output_;
  PikInfo* info_;
  FileHeader container_;
  size_t bits_ = 0;
  size_t pos_ = 0;
  size_t num_passes_ = 0;
  bool finalized_ = false;
  MultipassManager* multipass_manager_;
};

class PikMultipassDecoder {
 public:
  // input must remain valid thoroughout the lifetime of the Decoder object.
  PikMultipassDecoder(const PaddedBytes* input, MultipassManager* manager,
                      PikInfo* info = nullptr);

  // Returns the next frame, if any, otherwise returns false.
  Status NextPass(CodecInOut* pass, ThreadPool* pool);

 private:
  const PaddedBytes* input_;
  PikInfo* info_;
  BitReader reader_{input_->data(), input_->size()};
  FileHeader container_;
  MultipassManager* multipass_manager_;
};

}  // namespace pik

#endif  // PIK_STREAM_H_
