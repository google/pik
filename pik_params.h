#ifndef PIK_PARAMS_H_
#define PIK_PARAMS_H_

#include <stdint.h>

namespace pik {

struct CompressParams {
  // Only used for benchmarking (comparing vs libjpeg)
  int jpeg_quality = 100;
  bool jpeg_chroma_subsampling = false;
  bool clear_metadata = false;

  float butteraugli_distance = -1.0f;
  float target_bitrate = 0.0f;
  // 0.0 means search for the adaptive quantization map that matches the
  // butteraugli distance, positive values mean quantize everywhere with that
  // value.
  float uniform_quant = 0.0f;
  // If true, will use a compression method that is reasonably fast and aims to
  // find a trade-off between quality and file size that optimizes the
  // quality-adjusted-bits-per-pixel metric.
  bool fast_mode = false;
  int max_butteraugli_iters = 100;

  bool alpha_channel = false;

};

struct DecompressParams {
  uint64_t max_num_pixels = (1 << 30) - 1;
  // If true, checks at the end of decoding that all of the compressed data
  // was consumed by the decoder.
  bool check_decompressed_size = true;
};
}  // namespace pik

#endif  // PIK_PARAMS_H_
