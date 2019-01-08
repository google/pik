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

#include <unistd.h>

#include <string>
#include <vector>

#include "data_parallel.h"
#include "pik.h"
#include "pik_params.h"

#undef PROFILER_ENABLED
#define PROFILER_ENABLED 1
#include "adaptive_quantization.h"
#include "common.h"
#include "image.h"
#include "os_specific.h"
#include "pik_multipass.h"
#include "pik_pass.h"
#include "profiler.h"
#include "saliency_map.h"
#include "single_image_handler.h"

namespace pik {

Status PixelsToPik(const CompressParams& cparams, const CodecInOut* io,
                   PaddedBytes* compressed, PikInfo* aux_out,
                   ThreadPool* pool) {
  if (io->xsize() == 0 || io->ysize() == 0) {
    return PIK_FAILURE("Empty image");
  }
  if (!io->HasOriginalBitsPerSample()) {
    return PIK_FAILURE(
        "Pik requires specifying original bit depth "
        "of the pixels to encode as metadata.");
  }
  FileHeader container;
  MakeFileHeader(cparams, io, &container);

  if (!cparams.progressive_mode) {
    size_t extension_bits, total_bits;
    PIK_CHECK(CanEncode(container, &extension_bits, &total_bits));

    compressed->resize(DivCeil(total_bits, kBitsPerByte));
    size_t pos = 0;
    PIK_RETURN_IF_ERROR(
        WriteFileHeader(container, extension_bits, &pos, compressed->data()));
    PassParams pass_params;
    pass_params.is_last = true;
    SingleImageManager transform;
    PIK_RETURN_IF_ERROR(PixelsToPikPass(cparams, pass_params, io, pool,
                                        compressed, pos, aux_out, &transform));
  } else {
    bool lossless = cparams.lossless_mode;
    SingleImageManager transform;
    PikMultipassEncoder encoder(container, compressed, &transform, aux_out);
    CompressParams p = cparams;
    PassParams pass_params;
    p.lossless_mode = false;

    if (ApplyOverride(cparams.adaptive_reconstruction,
                      cparams.butteraugli_distance >=
                          kMinButteraugliForAdaptiveReconstruction)) {
      transform.UseAdaptiveReconstruction();
    }

    // Disable adaptive reconstruction in intermediate passes.
    p.adaptive_reconstruction = Override::kOff;
    pass_params.is_last = false;

    // DC pass.
    transform.SetProgressiveMode(ProgressiveMode::kDcOnly);
    PIK_RETURN_IF_ERROR(encoder.AddPass(p, pass_params, io, pool));

    // Disable gradient map from here on.
    p.gradient = Override::kOff;

    // DC is 0, low frequency predictions are useless.
    p.predict_lf = false;

    // Low frequency pass.
    transform.SetProgressiveMode(ProgressiveMode::kLfOnly);
    PIK_RETURN_IF_ERROR(encoder.AddPass(p, pass_params, io, pool));

    // DC + LF are 0, high frequency predictions are useless.
    p.predict_hf = false;

    // Optional salient-regions high frequency pass.
    auto final_pass_progressive_mode = ProgressiveMode::kHfOnly;
    if (!cparams.saliency_extractor_for_progressive_mode.empty()) {
      std::shared_ptr<ImageF> saliency_map;
      PIK_RETURN_IF_ERROR(
          ProduceSaliencyMap(cparams, compressed, io, pool, &saliency_map));
      final_pass_progressive_mode = ProgressiveMode::kNonSalientHfOnly;
      transform.SetProgressiveMode(ProgressiveMode::kSalientHfOnly);
      transform.SetSaliencyMap(saliency_map);
      PIK_RETURN_IF_ERROR(encoder.AddPass(p, pass_params, io, pool));
    }

    // Final non-lossless pass.
    transform.SetProgressiveMode(final_pass_progressive_mode);
    p.adaptive_reconstruction = cparams.adaptive_reconstruction;
    if (!lossless) {
      pass_params.is_last = true;
    }
    PIK_RETURN_IF_ERROR(encoder.AddPass(p, pass_params, io, pool));
    if (lossless) {
      pass_params.is_last = true;
      p.lossless_mode = true;
      PIK_RETURN_IF_ERROR(encoder.AddPass(p, pass_params, io, pool));
    }
    PIK_RETURN_IF_ERROR(encoder.Finalize());
  }
  return true;
}

Status PikToPixels(const DecompressParams& dparams,
                   const PaddedBytes& compressed, CodecInOut* io,
                   PikInfo* aux_out, ThreadPool* pool) {
  PROFILER_ZONE("PikToPixels uninstrumented");

  // To avoid the complexity of file I/O and buffering, we assume the bitstream
  // is loaded (or for large images/sequences: mapped into) memory.
  BitReader reader(compressed.data(), compressed.size());
  FileHeader container;
  PIK_RETURN_IF_ERROR(ReadFileHeader(&reader, &container));

  // Preview is discardable, i.e. content image does not rely on decoded preview
  // pixels; just skip it, if any.
  size_t preview_size_bits = container.preview.size_bits;
  if (preview_size_bits != 0) {
    reader.SkipBits(preview_size_bits);
  }

  SingleImageManager transform;
  do {
    PIK_RETURN_IF_ERROR(PikPassToPixels(dparams, compressed, container, pool,
                                        &reader, io, aux_out, &transform));
  } while (!transform.IsLastPass());

  if (dparams.check_decompressed_size &&
      reader.Position() != compressed.size()) {
    return PIK_FAILURE("Pik compressed data size mismatch.");
  }

  io->enc_size = compressed.size();

  return true;
}

}  // namespace pik
