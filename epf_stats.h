// Copyright 2018 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

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
