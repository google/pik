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

Image3F TransposedScaledDCT(const Image3F& img) {
  PIK_ASSERT(img.xsize() % 8 == 0);
  PIK_ASSERT(img.ysize() % 8 == 0);
  Image3F coeffs(img.xsize() * 8, img.ysize() / 8);

  for (int c = 0; c < 3; ++c) {
    const size_t stride = img.PlaneRow(c, 1) - img.PlaneRow(c, 0);
    for (size_t y = 0; y < coeffs.ysize(); ++y) {
      const float* PIK_RESTRICT row_in = img.PlaneRow(c, y * 8);
      float* PIK_RESTRICT row_out = coeffs.PlaneRow(c, y);

      for (size_t x = 0; x < coeffs.xsize(); x += 64) {
        ComputeTransposedScaledBlockDCTFloat(FromLines(row_in + x / 8, stride),
                                             ScaleToBlock(row_out + x));
      }
    }
  }
  return coeffs;
}

Image3F TransposedScaledIDCT(const Image3F& coeffs) {
  PIK_ASSERT(coeffs.xsize() % 64 == 0);
  Image3F img(coeffs.xsize() / 8, coeffs.ysize() * 8);

  for (int c = 0; c < 3; ++c) {
    const size_t stride = img.PlaneRow(c, 1) - img.PlaneRow(c, 0);
    for (size_t y = 0; y < coeffs.ysize(); ++y) {
      const float* PIK_RESTRICT row_in = coeffs.PlaneRow(c, y);
      float* PIK_RESTRICT row_out = img.PlaneRow(c, y * 8);

      for (size_t x = 0; x < coeffs.xsize(); x += 64) {
        ComputeTransposedScaledBlockIDCTFloat(FromBlock(row_in + x),
                                              ToLines(row_out + x / 8, stride));
      }
    }
  }
  return img;
}

TFNode* AddTransposedScaledIDCT(const TFPorts in_xyb, TFBuilder* builder) {
  PIK_CHECK(OutType(in_xyb.node) == TFType::kF32);
  return builder->AddClosure(
      "idct", Borders(), Scale(), {in_xyb}, 3, TFType::kF32,
      [](const ConstImageViewF* in, const OutputRegion& output_region,
         const MutableImageViewF* PIK_RESTRICT out) {
        PROFILER_ZONE("|| IDCT");
        const size_t xsize = output_region.xsize;
        const size_t ysize = output_region.ysize;

        const size_t stride = out->bytes_per_row() / sizeof(float);

        for (int c = 0; c < 3; ++c) {
          // x,y = top-left corner of 8x8 output block; 0,0 is the top-left of
          // the current output tile.
          for (size_t y = 0; y < ysize; y += 8) {
            const float* PIK_RESTRICT row_in = in[c].ConstRow(y / 8);
            float* PIK_RESTRICT row_out = out[c].Row(y);

            for (size_t x = 0; x < xsize; x += 8) {
              ComputeTransposedScaledBlockIDCTFloat(
                  FromBlock(row_in + x * 8), ToLines(row_out + x, stride));
            }
          }
        }
      });
}

void ComputeBlockDCTFloat(float block[64]) {
  ComputeTransposedScaledBlockDCTFloat(block);
  TransposeBlock(block);
  for (int y = 0; y < 8; ++y) {
    for (int x = 0; x < 8; ++x) {
      block[8 * y + x] /= 64.0f * kIDCTScales[y] * kIDCTScales[x];
    }
  }
}

void ComputeBlockIDCTFloat(float block[64]) {
  for (int y = 0; y < 8; ++y) {
    for (int x = 0; x < 8; ++x) {
      block[8 * y + x] *= kIDCTScales[y] * kIDCTScales[x];
    }
  }
  TransposeBlock(block);
  ComputeTransposedScaledBlockIDCTFloat(block);
}

void RotateDCT(float angle, float block[64]) {
  float a2a = std::cos(angle);
  float a2b = -std::sin(angle);
  float b2a = std::sin(angle);
  float b2b = std::cos(angle);
  for (int y = 0; y < 8; y++) {
    for (int x = 0; x < y; x++) {
      if (x >= 2 || y >= 2) continue;
      float a = block[8 * y + x];
      float b = block[8 * x + y];
      block[8 * y + x] = a2a * a + b2a * b;
      block[8 * x + y] = a2b * a + b2b * b;
    }
  }
}

}  // namespace pik
