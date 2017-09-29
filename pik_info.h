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
  size_t num_clustered_histograms = 0;
  size_t histogram_size = 0;
  size_t entropy_coded_bits = 0;
  size_t extra_bits = 0;
  size_t total_size = 0;
  double clustered_entropy = 0.0f;
};

// Metadata and statistics gathered during compression or decompression.
struct PikInfo {
  void Assimilate(const PikInfo& victim) {
    ytob_image.Assimilate(victim.ytob_image);
    quant_image.Assimilate(victim.quant_image);
    dc_image.Assimilate(victim.dc_image);
    ac_image.Assimilate(victim.ac_image);
    num_butteraugli_iters += victim.num_butteraugli_iters;
  }
  PikImageSizeInfo TotalImageSize() const {
    PikImageSizeInfo total;
    total.Assimilate(ytob_image);
    total.Assimilate(quant_image);
    total.Assimilate(dc_image);
    total.Assimilate(ac_image);
    return total;
  }
  PikImageSizeInfo ytob_image;
  PikImageSizeInfo quant_image;
  PikImageSizeInfo dc_image;
  PikImageSizeInfo ac_image;
  int num_butteraugli_iters = 0;
  size_t decoded_size = 0;
  // If not empty, additional debugging information (e.g. debug images) is
  // saved in files with this prefix.
  std::string debug_prefix;
};

}  // namespace pik

#endif  // PIK_INFO_H_
