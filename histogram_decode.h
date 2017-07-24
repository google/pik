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

// Library to decode a histogram from the bit-stream.

#ifndef HISTOGRAM_DECODE_H_
#define HISTOGRAM_DECODE_H_

#include <vector>

#include "bit_reader.h"

namespace pik {

// Decodes a histogram from the bit-stream where the sum of all population
// counts is 1 << precision_bits.
// Fills in *counts with the decoded population count values.
// Returns false on decoding error.
bool ReadHistogram(int precision_bits, std::vector<int>* counts,
                   BitReader* input);

}  // namespace pik

#endif  // HISTOGRAM_DECODE_H_
