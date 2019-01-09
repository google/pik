// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_INFO_H_
#define PIK_INFO_H_

#include <cstddef>
#include <string>
#include <vector>
#include "adaptive_reconstruction_fwd.h"
#include "image.h"
#include "image_io.h"

namespace pik {

struct PikImageSizeInfo {
  PikImageSizeInfo() {}

  void Assimilate(const PikImageSizeInfo& victim) {
    num_clustered_histograms += victim.num_clustered_histograms;
    histogram_size += victim.histogram_size;
    entropy_coded_bits += victim.entropy_coded_bits;
    extra_bits += victim.extra_bits;
    total_size += victim.total_size;
    clustered_entropy += victim.clustered_entropy;
  }
  void Print(size_t num_inputs) const {
    printf("%10zd", total_size);
    if (histogram_size > 0) {
      printf("   [%6.2f %8zd %8zd %8zd %12.3f",
             num_clustered_histograms * 1.0 / num_inputs, histogram_size,
             entropy_coded_bits >> 3, extra_bits >> 3,
             histogram_size + (clustered_entropy + extra_bits) / 8.0f);
      printf("]");
    }
    printf("\n");
  }
  size_t num_clustered_histograms = 0;
  size_t histogram_size = 0;
  size_t entropy_coded_bits = 0;
  size_t extra_bits = 0;
  size_t total_size = 0;
  double clustered_entropy = 0.0f;
};

enum {
  kLayerHeader = 0,
  kLayerQuant,
  kLayerOrder,
  kLayerCmap,
  kLayerDC,
  kLayerAC,
  kNumImageLayers
};
static const char* kImageLayers[kNumImageLayers] = {"header", "quant", "order",
                                                    "cmap",   "DC",    "AC"};

struct TestingAux {
  Image3F* ac_prediction = nullptr;
};

// Metadata and statistics gathered during compression or decompression.
struct PikInfo {
  PikInfo() : layers(kNumImageLayers) {}

  PikInfo(const PikInfo&) = default;

  void Assimilate(const PikInfo& victim) {
    for (int i = 0; i < layers.size(); ++i) {
      layers[i].Assimilate(victim.layers[i]);
    }
    num_blocks += victim.num_blocks;
    num_dct16_blocks += victim.num_dct16_blocks;
    num_dct32_blocks += victim.num_dct32_blocks;
    entropy_estimate += victim.entropy_estimate;
    num_butteraugli_iters += victim.num_butteraugli_iters;
    adaptive_reconstruction_aux.Assimilate(victim.adaptive_reconstruction_aux);
  }

  PikImageSizeInfo TotalImageSize() const {
    PikImageSizeInfo total;
    for (int i = 0; i < layers.size(); ++i) {
      total.Assimilate(layers[i]);
    }
    return total;
  }

  void Print(size_t num_inputs) const {
    if (num_inputs == 0) return;
    printf("Average butteraugli iters: %10.2f\n",
           num_butteraugli_iters * 1.0 / num_inputs);
    for (int i = 0; i < layers.size(); ++i) {
      if (layers[i].total_size > 0) {
        printf("Total layer size %-10s", kImageLayers[i]);
        layers[i].Print(num_inputs);
      }
    }
    printf("Total image size           ");
    TotalImageSize().Print(num_inputs);
    adaptive_reconstruction_aux.Print();
  }

  template <typename Img>
  void DumpImage(const char* label, const Img& image) const {
    if (debug_prefix.empty()) return;
    char pathname[200];
    snprintf(pathname, sizeof(pathname), "%s%s.png", debug_prefix.c_str(),
             label);
    WriteImage(ImageFormatPNG(), image, pathname);
  }

  // This dumps coefficients as a 16-bit PNG with coefficients of a block placed
  // in the area that would contain that block in a normal image. To view the
  // resulting image manually, rescale intensities by using:
  // $ convert -auto-level IMAGE.PNG - | display -
  void DumpCoeffImage(const char* label, const Image3S& coeff_image) const;

  std::vector<PikImageSizeInfo> layers;
  size_t num_blocks = 0;
  // Number of blocks that use larger DCT. Only set in the encoder.
  size_t num_dct16_blocks = 0;
  size_t num_dct32_blocks = 0;
  // Estimate of compressed size according to entropy-given lower bounds.
  float entropy_estimate = 0;
  int num_butteraugli_iters = 0;
  // If not empty, additional debugging information (e.g. debug images) is
  // saved in files with this prefix.
  std::string debug_prefix;

  AdaptiveReconstructionAux adaptive_reconstruction_aux;
  TestingAux testing_aux;
};

// Used to skip image creation if they won't be written to debug directory.
static inline bool WantDebugOutput(const PikInfo* info) {
  // Need valid pointer and filename.
  return info != nullptr && !info->debug_prefix.empty();
}

}  // namespace pik

#endif  // PIK_INFO_H_
