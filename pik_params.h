#ifndef PIK_PARAMS_H_
#define PIK_PARAMS_H_

#include <stddef.h>
#include <stdint.h>

namespace pik {

// No effect if kDefault, otherwise forces a feature on or off.
enum class Override : int {
  kOn = 1,
  kOff = 0,
  kDefault = -1
};

struct CompressParams {
  // Only used for benchmarking (comparing vs libjpeg)
  int jpeg_quality = 100;
  bool jpeg_chroma_subsampling = false;
  bool clear_metadata = false;

  float butteraugli_distance = 1.0f;
  size_t target_size = 0;
  float target_bitrate = 0.0f;
  bool target_size_search_fast_mode = false;
  // 0.0 means search for the adaptive quantization map that matches the
  // butteraugli distance, positive values mean quantize everywhere with that
  // value.
  float uniform_quant = 0.0f;
  float quant_border_bias = 0.0f;
  // If true, will use a compression method that is reasonably fast and aims to
  // find a trade-off between quality and file size that optimizes the
  // quality-adjusted-bits-per-pixel metric.
  bool fast_mode = false;
  int max_butteraugli_iters = 7;

  bool really_slow_mode = false;
  int max_butteraugli_iters_really_slow_mode = 100;

  Override denoise = Override::kDefault;

  Override apply_noise = Override::kDefault;

  bool use_brunsli_v2 = false;

  // Prints extra information after encoding.
  bool verbose = false;

  float hf_asymmetry = 1.0;
};

struct DecompressParams {
  uint64_t max_num_pixels = (1 << 30) - 1;
  // If true, checks at the end of decoding that all of the compressed data
  // was consumed by the decoder.
  bool check_decompressed_size = true;

  // kDefault := whatever the encoder decided (stored in header).
  Override denoise = Override::kDefault;
};

static constexpr float kMaxButteraugliForHQ = 2.0f;
static constexpr float kMinButteraugliForDither = 1.0f;

}  // namespace pik

#endif  // PIK_PARAMS_H_
