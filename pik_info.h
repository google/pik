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
  }
  size_t num_clustered_histograms = 0;
  size_t histogram_size = 0;
  size_t entropy_coded_bits = 0;
  size_t extra_bits = 0;
  size_t total_size = 0;
};

// Metadata and statistics gathered during compression or decompression.
struct PikInfo {
  void Assimilate(const PikInfo& victim) {
    ytob_image_size += victim.ytob_image_size;
    quant_image_size += victim.quant_image_size;
    dc_image.Assimilate(victim.dc_image);
    ac_image.Assimilate(victim.ac_image);
    num_butteraugli_iters += victim.num_butteraugli_iters;
  }
  PikImageSizeInfo TotalImageSize() const {
    PikImageSizeInfo total;
    total.Assimilate(dc_image);
    total.Assimilate(ac_image);
    return total;
  }
  size_t ytob_image_size = 0;
  size_t quant_image_size = 0;
  PikImageSizeInfo dc_image;
  PikImageSizeInfo ac_image;
  int num_butteraugli_iters = 0;
  // If not empty, additional debugging information (e.g. debug images) is
  // saved in files with this prefix.
  std::string debug_prefix;
};

}  // namespace pik

#endif  // PIK_INFO_H_
