/*
 * Copyright 2016 Google Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef GUETZLI_PROCESSOR_H_
#define GUETZLI_PROCESSOR_H_

#include <string>
#include <vector>

#include "guetzli/jpeg_data.h"
#include "guetzli/stats.h"

namespace pik {
namespace guetzli {

struct Params {
  float butteraugli_target = 1.0;
  bool clear_metadata = false;
  bool try_420 = false;
  bool force_420 = false;
  bool use_silver_screen = false;
  int zeroing_greedy_lookahead = 3;
  bool new_zeroing_model = true;
};

struct GuetzliOutput {
  JPEGData jpeg;  // used by some Process overloads below
  std::string jpeg_data;
  std::vector<float> distmap;
  double distmap_aggregate;
  double score;
};

// Sets *out to a jpeg encoded string that will decode to an image that is
// visually indistinguishable from the input rgb image.
bool Process(const Params& params, ProcessStats* stats,
             const std::vector<uint8_t>& rgb, int w, int h, std::string* out);

bool Process(const Params& params, const JPEGData& jpeg_in, JPEGData* jpeg_out);

bool Process(const Params& params, const std::string& in_data,
             std::string* out_data);

bool Process(const Params& params, const std::vector<uint8_t>& rgb, int w,
             int h, std::string* out);

// For use by JPEG->PIK converter - avoids serialization to string.
bool Process(const Params& params, const std::vector<uint8_t>& rgb, int w,
             int h, JPEGData* jpg);

}  // namespace guetzli
}  // namespace pik

#endif  // GUETZLI_PROCESSOR_H_
