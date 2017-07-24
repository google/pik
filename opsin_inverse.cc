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

#include "opsin_inverse.h"

#include <stdint.h>
#include <cmath>

#include "opsin_params.h"

namespace pik {

namespace {

const uint8_t* NewOpsinToSrgb8Table(float bias) {
  uint8_t* table = new uint8_t[4096];
  for (int i = 0; i < 4096; ++i) {
    table[i] = std::round(
        std::min(255.0f, std::max(0.0f, OpsinToSrgb8Direct(i / 16.0) + bias)));
  }
  return table;
}

}  // namespace

const uint8_t* OpsinToSrgb8TablePlusQuarter() {
  static const uint8_t* const kOpsinToSrgb8TablePlusQuarter =
      NewOpsinToSrgb8Table(0.25);
  return kOpsinToSrgb8TablePlusQuarter;
}

const uint8_t* OpsinToSrgb8TableMinusQuarter() {
  static const uint8_t* const kOpsinToSrgb8TableMinusQuarter =
      NewOpsinToSrgb8Table(-0.25);
  return kOpsinToSrgb8TableMinusQuarter;
}

// TODO(user): Make this vectorizable.
Image3B OpsinDynamicsInverse(const Image3F& opsin) {
  Image3B srgb(opsin.xsize(), opsin.ysize());
  for (int y = 0; y < opsin.ysize(); ++y) {
    auto row_in = opsin.Row(y);
    auto row_out = srgb.Row(y);
    for (int x = 0; x < opsin.xsize(); ++x) {
      float r, g, b;
      XybToRgb(row_in[0][x], row_in[1][x], row_in[2][x], &r, &g, &b);
      row_out[0][x] = OpsinToSrgb8Direct(r);
      row_out[1][x] = OpsinToSrgb8Direct(g);
      row_out[2][x] = OpsinToSrgb8Direct(b);
    }
  }
  return srgb;
}

}  // namespace pik
