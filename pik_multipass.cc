// Copyright 2018 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik_multipass.h"
#include "common.h"
#include "compressed_image.h"
#include "headers.h"
#include "image.h"
#include "opsin_image.h"
#include "opsin_inverse.h"
#include "pik_params.h"
#include "pik_pass.h"
#include "write_bits.h"

namespace pik {

PikMultipassEncoder::PikMultipassEncoder(const FileHeader& container,
                                         PaddedBytes* output,
                                         MultipassManager* manager,
                                         PikInfo* info)
    : output_(output),
      info_(info),
      container_(container),
      multipass_manager_(manager) {}

void PikMultipassEncoder::ExtendOutput(size_t bits) {
  bits_ += bits;
  output_->resize(DivCeil(bits_, kBitsPerByte));
}

Status PikMultipassEncoder::AddPass(const CompressParams& params,
                                    const PassParams& pass_params,
                                    const CodecInOut* frame, ThreadPool* pool) {
  if (finalized_) return PIK_FAILURE("AddFrame called after Finalize");
  if (frame->xsize() == 0 || frame->ysize() == 0) {
    return PIK_FAILURE("Empty image.");
  }

  // On the first pass, write out the container.
  if (num_passes_ == 0) {
    size_t extension_bits, total_bits;
    PIK_CHECK(CanEncode(container_, &extension_bits, &total_bits));

    output_->resize(DivCeil(total_bits, kBitsPerByte));
    PIK_RETURN_IF_ERROR(
        WriteFileHeader(container_, extension_bits, &pos_, output_->data()));
  }

  // Check that all the frames have the same shape and depth.
  if (container_.metadata.transcoded.original_bit_depth !=
      frame->original_bits_per_sample()) {
    return PIK_FAILURE("Invalid frame bit depth.");
  }
  if (!SameSize(container_, *frame)) {
    return PIK_FAILURE("Image size changed.");
  }

  size_t image_start = pos_;
  PIK_RETURN_IF_ERROR(PixelsToPikPass(params, pass_params, frame, pool, output_,
                                      pos_, info_, multipass_manager_));
  // TODO(veluca): only do this if we have another pass.
  {
    DecompressParams params;
    BitReader reader(output_->data(), output_->size());
    reader.SkipBits(image_start);
    CodecInOut out(frame->Context());
    PIK_ASSERT(PikPassToPixels(params, *output_, container_, pool, &reader,
                               &out,
                               /*aux_out=*/nullptr, multipass_manager_));
  }
  bits_ = pos_;
  num_passes_++;
  return true;
}

Status PikMultipassEncoder::Finalize() {
  if (finalized_) {
    return PIK_FAILURE("Finalize called multiple times.");
  }
  if (!num_passes_) {
    return PIK_FAILURE("Finalized called with no frames in the stream.");
  }
  finalized_ = true;
  WriteZeroesToByteBoundary(&pos_, output_->data());
  return true;
}

PikMultipassDecoder::PikMultipassDecoder(const PaddedBytes* input,
                                         MultipassManager* manager,
                                         PikInfo* info)
    : input_(input), info_(info), multipass_manager_(manager) {}

Status PikMultipassDecoder::NextPass(CodecInOut* pass, ThreadPool* pool) {
  if (reader_.Position() == 0) {
    PIK_RETURN_IF_ERROR(ReadFileHeader(&reader_, &container_));
    size_t xsize = container_.xsize();
    size_t ysize = container_.ysize();
    static const uint32_t kMaxWidth = (1 << 25) - 1;
    if (xsize > kMaxWidth) {
      return PIK_FAILURE("Image too wide.");
    }
    uint64_t num_pixels = static_cast<uint64_t>(xsize) * ysize;
    static const uint32_t kMaxNumPixels = (1 << 30) - 1;
    if (num_pixels > kMaxNumPixels) {
      return PIK_FAILURE("Image too big.");
    }
    if (pass->dec_c_original.icc.empty()) {
      // Removed by MaybeRemoveProfile; fail unless we successfully restore it.
      PIK_RETURN_IF_ERROR(ColorManagement::SetProfileFromFields(
          &container_.metadata.transcoded.original_color_encoding));
    }
  }
  // TODO(veluca): metadata.

  // Not an error, end of file.
  if (reader_.Position() >= input_->size()) return false;

  DecompressParams params;
  PIK_RETURN_IF_ERROR(PikPassToPixels(params, *input_, container_, pool,
                                      &reader_, pass, info_,
                                      multipass_manager_));
  return true;
}

}  // namespace pik
