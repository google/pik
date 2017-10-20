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

#ifndef PIK_INFO_H_
#define PIK_INFO_H_

#include <cstddef>
#include <string>
#include <vector>

namespace pik {

struct PikImageSizeInfo {
  void Assimilate(const PikImageSizeInfo& victim) {
    num_clustered_histograms += victim.num_clustered_histograms;
    histogram_size += victim.histogram_size;
    entropy_coded_bits += victim.entropy_coded_bits;
    extra_bits += victim.extra_bits;
    total_size += victim.total_size;
    clustered_entropy += victim.clustered_entropy;
  }
  void Print(size_t num_inputs) const {
    printf("%10zd   [%6.2f %8zd %10zd %10zd %21.10f]\n", total_size,
           num_clustered_histograms * 1.0 / num_inputs, histogram_size,
           entropy_coded_bits >> 3, extra_bits >> 3,
           histogram_size + (clustered_entropy + extra_bits) / 8.0f);
  }
  size_t num_clustered_histograms = 0;
  size_t histogram_size = 0;
  size_t entropy_coded_bits = 0;
  size_t extra_bits = 0;
  size_t total_size = 0;
  double clustered_entropy = 0.0f;
};

static const int kNumImageLayers = 3;
static const char* kImageLayers[] = {
  "quant", "DC", "AC",
};

// Metadata and statistics gathered during compression or decompression.
struct PikInfo {
  PikInfo() : layers(kNumImageLayers) {}
  void Assimilate(const PikInfo& victim) {
    for (int i = 0; i < layers.size(); ++i) {
      layers[i].Assimilate(victim.layers[i]);
    }
    num_butteraugli_iters += victim.num_butteraugli_iters;
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
      printf("Total layer size %-10s", kImageLayers[i]);
      layers[i].Print(num_inputs);
    }
    printf("Total image size           ");
    TotalImageSize().Print(num_inputs);
  }


  std::vector<PikImageSizeInfo> layers;
  int num_butteraugli_iters = 0;
  size_t decoded_size = 0;
  // If not empty, additional debugging information (e.g. debug images) is
  // saved in files with this prefix.
  std::string debug_prefix;
};

}  // namespace pik

#endif  // PIK_INFO_H_
