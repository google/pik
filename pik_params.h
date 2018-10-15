#ifndef PIK_PARAMS_H_
#define PIK_PARAMS_H_

#include <stddef.h>
#include <stdint.h>

namespace pik {

// Reasonable default for sRGB, matches common monitors. Butteraugli was tuned
// for this, we scale darker/brighter inputs accordingly.
static constexpr int kDefaultIntensityTarget = 250;
static constexpr float kIntensityMultiplier = 1.0f / kDefaultIntensityTarget;

// No effect if kDefault, otherwise forces a feature (typically a Header flag)
// on or off.
enum class Override : int {
  kOn = 1,
  kOff = 0,
  kDefault = -1
};

static inline bool ApplyOverride(Override o, bool condition) {
  if (o == Override::kOn) condition = true;
  if (o == Override::kOff) condition = false;
  return condition;
}

struct CompressParams {
  // Only used for benchmarking (comparing vs libjpeg)
  int jpeg_quality = 100;
  bool jpeg_chroma_subsampling = false;
  bool clear_metadata = false;

  float butteraugli_distance = 1.0f;
  size_t target_size = 0;
  float target_bitrate = 0.0f;

  // 1+this is multiplied onto quant map in regions of interest (e.g. faces).
  float roi_factor = 0.5f;

  // 0.0 means search for the adaptive quantization map that matches the
  // butteraugli distance, positive values mean quantize everywhere with that
  // value.
  float uniform_quant = 0.0f;
  float quant_border_bias = 0.0f;

  // If true, will use a compression method that is reasonably fast and aims to
  // find a trade-off between quality and file size that optimizes the
  // quality-adjusted-bits-per-pixel metric.
  bool fast_mode = false;
  int max_butteraugli_iters = 11;

  size_t resampling_factor2 = 2;

  bool guetzli_mode = false;
  int max_butteraugli_iters_guetzli_mode = 100;

  Override noise = Override::kDefault;
  Override smooth = Override::kDefault;
  Override gradient = Override::kDefault;
  Override adaptive_reconstruction = Override::kDefault;
  int gaborish = -1;  // 0..7 or -1 for default.

  bool use_ac_strategy = false;

  // Prints extra information after encoding.
  bool verbose = false;

  float hf_asymmetry = 1.0f;

  // Intended intensity target of the viewer after decoding, in nits (cd/m^2).
  // There is no other way of knowing the target brightness - depends on source
  // material. 709 typically targets 100 nits, 2020 PQ up to 10K, but HDR
  // content is more typically mastered to 4K nits. The default requires no
  // scaling for Butteraugli.
  float intensity_target = kDefaultIntensityTarget;
};

struct DecompressParams {
  uint64_t max_num_pixels = (1 << 30) - 1;
  // If true, checks at the end of decoding that all of the compressed data
  // was consumed by the decoder.
  bool check_decompressed_size = true;

  Override noise = Override::kDefault;  // cannot be kOn (needs encoder)
  // (It is not useful to override smooth in the decoder because the residuals
  // are specific to the chosen predictor.)
  Override gradient = Override::kDefault;  // cannot be kOn (needs encoder)
  Override adaptive_reconstruction = Override::kDefault;
  int gaborish = -1;  // 0..7 or -1 for default.
};

// Enable features for distances >= these thresholds:
static constexpr float kMinButteraugliForNoise = 1.4f;  // see pik.cc
static constexpr float kMinButteraugliForGradient = 1.85f;  // see pik.cc
static constexpr float kMinButteraugliForAdaptiveReconstruction = 1.0f;

static constexpr float kMinButteraugliForDefaultQuant = 2.0f;

}  // namespace pik

#endif  // PIK_PARAMS_H_
