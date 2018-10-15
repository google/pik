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

#include "common.h"
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

Image3F PadImageToMultiple(const Image3F& in, const size_t N) {
  PROFILER_FUNC;
  const size_t xsize_blocks = DivCeil(in.xsize(), N);
  const size_t ysize_blocks = DivCeil(in.ysize(), N);
  const size_t xsize = N * xsize_blocks;
  const size_t ysize = N * ysize_blocks;
  Image3F out(xsize, ysize);
  for (int c = 0; c < 3; ++c) {
    int y = 0;
    for (; y < in.ysize(); ++y) {
      const float* PIK_RESTRICT row_in = in.ConstPlaneRow(c, y);
      float* PIK_RESTRICT row_out = out.PlaneRow(c, y);
      memcpy(row_out, row_in, in.xsize() * sizeof(row_in[0]));
      const int lastcol = in.xsize() - 1;
      const float lastval = row_out[lastcol];
      for (int x = in.xsize(); x < xsize; ++x) {
        row_out[x] = lastval;
      }
    }

    // TODO(janwas): no need to copy if we can 'extend' image: if rows are
    // pointers to any memory? Or allocate larger image before IO?
    const int lastrow = in.ysize() - 1;
    for (; y < ysize; ++y) {
      const float* PIK_RESTRICT row_in = out.ConstPlaneRow(c, lastrow);
      float* PIK_RESTRICT row_out = out.PlaneRow(c, y);
      memcpy(row_out, row_in, xsize * sizeof(row_out[0]));
    }
  }
  return out;
}

Image3B Float255ToByteImage3(const Image3F& from) {
  Image3B to(from.xsize(), from.ysize());
  PROFILER_FUNC;
  for (int c = 0; c < 3; c++) {
    for (size_t y = 0; y < from.ysize(); ++y) {
      const float* PIK_RESTRICT row_from = from.ConstPlaneRow(c, y);
      uint8_t* PIK_RESTRICT row_to = to.PlaneRow(c, y);
      for (size_t x = 0; x < from.xsize(); ++x) {
        float f = std::min(std::max(0.0f, row_from[x]), 255.0f);
        row_to[x] = static_cast<uint8_t>(f + 0.5);
      }
    }
  }
  return to;
}

float DotProduct(const ImageF& a, const ImageF& b) {
  double sum = 0.0;
  for (int y = 0; y < a.ysize(); ++y) {
    const float* const PIK_RESTRICT row_a = a.ConstRow(y);
    const float* const PIK_RESTRICT row_b = b.ConstRow(y);
    for (int x = 0; x < a.xsize(); ++x) {
      sum += row_a[x] * row_b[x];
    }
  }
  return sum;
}

}  // namespace pik
