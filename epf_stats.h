// Copyright 2018 Google Inc. All Rights Reserved.
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

#ifndef EPF_STATS_H_
#define EPF_STATS_H_

#include <stdio.h>
#include <stdlib.h>

#include "descriptive_statistics.h"

namespace pik {

// Per-thread.
struct EpfStats {
  void Assimilate(const EpfStats& other) {
    total += other.total;
    skipped += other.skipped;
    less += other.less;
    greater += other.greater;
    s_quant.Assimilate(other.s_quant);
    s_sigma.Assimilate(other.s_sigma);
  }

  void Print() const {
    const int stats = Stats::kNoSkewKurt + Stats::kNoGeomean;
    printf(
        "EPF total blocks: %zu; skipped: %zu (%f%%); outside %zu|%zu (%f%%)\n"
        "quant: %s\nsigma: %s\n",
        total, skipped, 100.0 * skipped / total, less, greater,
        100.0 * (less + greater) / total, s_quant.ToString(stats).c_str(),
        s_sigma.ToString(stats).c_str());
  }

  size_t total = 0;
  size_t skipped = 0;  // sigma == 0 => no filter

  // Outside LUT range:
  size_t less = 0;
  size_t greater = 0;

  Stats s_quant;
  Stats s_sigma;
};

}  // namespace pik

#endif  // EPF_STATS_H_
