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

#ifndef ANS_COMMON_H_
#define ANS_COMMON_H_

#include <vector>

#include "compiler_specific.h"

namespace pik {

// Returns the precision (number of bits) that should be used to store
// a histogram count such that Log2Floor(count) == logcount.
PIK_INLINE int GetPopulationCountPrecision(int logcount) {
  return (logcount + 1) >> 1;
}

// Returns a histogram where the counts are positive, differ by at most 1,
// and add up to total_count. The bigger counts (if any) are at the beginning
// of the histogram.
std::vector<int> CreateFlatHistogram(int length, int total_count);

}  // namespace pik

#endif  // ANS_COMMON_H_
