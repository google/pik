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

#include "dct.h"
#include <cmath>

#define PROFILER_ENABLED 1
#include "arch_specific.h"
#include "compiler_specific.h"
#include "profiler.h"

namespace pik {
namespace {

template <class DC_Op>
void TransposedScaledIDCT_Func(const void*, const ConstImageViewF* in,
                               const OutputRegion& output_region,
                               const MutableImageViewF* PIK_RESTRICT out) {
  PROFILER_ZONE("|| IDCT");
  const size_t xsize = output_region.xsize;
  const size_t ysize = output_region.ysize;

  const size_t stride = out->bytes_per_row() / sizeof(float);

  for (int c = 0; c < 3; ++c) {
    // x,y = top-left corner of 8x8 output block; 0,0 is the top-left of
    // the current output tile.
    for (size_t y = 0; y < ysize; y += kBlockHeight) {
      const float* PIK_RESTRICT row_in = in[c].ConstRow(y / kBlockHeight);
      float* PIK_RESTRICT row_out = out[c].Row(y);

      for (size_t x = 0; x < xsize; x += kBlockWidth) {
        ComputeTransposedScaledBlockIDCTFloat(
            FromBlock(row_in + x * kBlockWidth), ToLines(row_out + x, stride),
            DC_Op());
      }
    }
  }
}

}  // namespace

TFNode* AddTransposedScaledIDCT(const TFPorts in_xyb, bool zero_dc,
                                TFBuilder* builder) {
  PIK_CHECK(OutType(in_xyb.node) == TFType::kF32);
  return builder->Add("idct", Borders(), Scale(), {in_xyb}, 3, TFType::kF32,
                      zero_dc ? &TransposedScaledIDCT_Func<DC_Zero>
                              : &TransposedScaledIDCT_Func<DC_Unchanged>);
}

void ComputeBlockDCTFloat(float block[kBlockSize]) {
  ComputeTransposedScaledBlockDCTFloat(block);
  TransposeBlock(block);
  for (size_t y = 0; y < kBlockHeight; ++y) {
    const float scale_y = static_cast<int>(kBlockSize) * kIDCTScales[y];
    for (size_t x = 0; x < kBlockWidth; ++x) {
      block[kBlockWidth * y + x] /= scale_y * kIDCTScales[x];
    }
  }
}

void ComputeBlockIDCTFloat(float block[kBlockSize]) {
  for (int y = 0; y < kBlockHeight; ++y) {
    for (int x = 0; x < kBlockWidth; ++x) {
      block[kBlockWidth * y + x] *= kIDCTScales[y] * kIDCTScales[x];
    }
  }
  TransposeBlock(block);
  ComputeTransposedScaledBlockIDCTFloat(block, DC_Unchanged());
}

void RotateDCT(float angle, float block[kBlockSize]) {
  float a2a = std::cos(angle);
  float a2b = -std::sin(angle);
  float b2a = std::sin(angle);
  float b2b = std::cos(angle);
  for (int y = 0; y < kBlockHeight; y++) {
    for (int x = 0; x < y; x++) {
      if (x >= 2 || y >= 2) continue;
      float a = block[kBlockWidth * y + x];
      float b = block[kBlockWidth * x + y];
      block[kBlockWidth * y + x] = a2a * a + b2a * b;
      block[kBlockWidth * x + y] = a2b * a + b2b * b;
    }
  }
}

}  // namespace pik
