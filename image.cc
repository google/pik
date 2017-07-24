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

#include "image.h"

#include <stdint.h>
#include <memory>
#include <string>

#include "profiler.h"

namespace pik {

ImageB Float255ToByteImage(const ImageF& from) {
  ImageB to(from.xsize(), from.ysize());
  PROFILER_FUNC;
  for (size_t y = 0; y < from.ysize(); ++y) {
    const float* const PIK_RESTRICT row_from = from.Row(y);
    uint8_t* const PIK_RESTRICT row_to = to.Row(y);
    for (size_t x = 0; x < from.xsize(); ++x) {
      float f = std::min(std::max(0.0f, row_from[x]), 255.0f);
      row_to[x] = static_cast<uint8_t>(f + 0.5);
    }
  }
  return to;
}

ImageB ImageFromPacked(const uint8_t* packed, const size_t xsize,
                       const size_t ysize, const size_t bytes_per_row) {
  PIK_ASSERT(bytes_per_row >= xsize);
  ImageB image(xsize, ysize);
  PROFILER_FUNC;
  for (size_t y = 0; y < ysize; ++y) {
    uint8_t* const PIK_RESTRICT row = image.Row(y);
    const uint8_t* const PIK_RESTRICT packed_row = packed + y * bytes_per_row;
    memcpy(row, packed_row, xsize);
  }
  return image;
}


Image3B Float255ToByteImage3(const Image3F& from) {
  Image3B to(from.xsize(), from.ysize());
  PROFILER_FUNC;
  for (size_t y = 0; y < from.ysize(); ++y) {
    const auto from_rows = from.ConstRow(y);
    auto to_rows = to.Row(y);
    for (int c = 0; c < 3; c++) {
      for (size_t x = 0; x < from.xsize(); ++x) {
        float f = std::min(std::max(0.0f, from_rows[c][x]), 255.0f);
        to_rows[c][x] = static_cast<uint8_t>(f + 0.5);
      }
    }
  }
  return to;
}

float Average(const ImageF& img) {
  // TODO(user): Make sure this is numerically stable.
  const size_t xsize = img.xsize();
  const size_t ysize = img.ysize();
  double sum = 0.0f;
  for (size_t y = 0; y < ysize; ++y) {
    auto row = img.Row(y);
    for (size_t x = 0; x < xsize; ++x) {
      sum += row[x];
    }
  }
  return sum / xsize / ysize;
}

float DotProduct(const ImageF& a, const ImageF& b) {
  double sum = 0.0;
  for (int y = 0; y < a.ysize(); ++y) {
    const float* const PIK_RESTRICT row_a = a.Row(y);
    const float* const PIK_RESTRICT row_b = b.Row(y);
    for (int x = 0; x < a.xsize(); ++x) {
      sum += row_a[x] * row_b[x];
    }
  }
  return sum;
}

}  // namespace pik
