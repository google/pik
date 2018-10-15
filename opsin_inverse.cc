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

#define PROFILER_ENABLED 1
#include "profiler.h"

namespace pik {
namespace {

SIMD_FULL(float)::V inverse_matrix[9];

// Called from non-local static initializer for convenience.
SIMD_ATTR int InitInverseMatrix() {
  const SIMD_FULL(float) d;
  const float* PIK_RESTRICT inverse = GetOpsinAbsorbanceInverseMatrix();
  for (size_t i = 0; i < 9; ++i) {
    inverse_matrix[i] = set1(d, inverse[i]);
  }

  return 0;
}

int dummy = InitInverseMatrix();

}  // namespace

SIMD_ATTR void CenteredOpsinToLinear(const Image3F& opsin, ThreadPool* pool,
                                     Image3F* linear) {
  PIK_CHECK(linear->xsize() != 0);
  PROFILER_FUNC;
  // Opsin is padded to blocks; only produce valid output pixels.
  const size_t xsize = linear->xsize();
  const size_t ysize = linear->ysize();

  const SIMD_FULL(float) d;
  using V = SIMD_FULL(float)::V;

  const auto center_x = set1(d, kXybCenter[0]);
  const auto center_y = set1(d, kXybCenter[1]);
  const auto center_b = set1(d, kXybCenter[2]);

  pool->Run(0, ysize,
            [&](const int task, const int thread) {
              const size_t y = task;

              // Faster than adding stride at end of loop.
              const float* PIK_RESTRICT row_opsin_x = opsin.ConstPlaneRow(0, y);
              const float* PIK_RESTRICT row_opsin_y = opsin.ConstPlaneRow(1, y);
              const float* PIK_RESTRICT row_opsin_b = opsin.ConstPlaneRow(2, y);

              float* PIK_RESTRICT row_linear_r = linear->PlaneRow(0, y);
              float* PIK_RESTRICT row_linear_g = linear->PlaneRow(1, y);
              float* PIK_RESTRICT row_linear_b = linear->PlaneRow(2, y);

              for (size_t x = 0; x < xsize; x += d.N) {
                const auto in_opsin_x = load(d, row_opsin_x + x) + center_x;
                const auto in_opsin_y = load(d, row_opsin_y + x) + center_y;
                const auto in_opsin_b = load(d, row_opsin_b + x) + center_b;
                V linear_r, linear_g, linear_b;
                XybToRgb(d, in_opsin_x, in_opsin_y, in_opsin_b, inverse_matrix,
                         &linear_r, &linear_g, &linear_b);

                store(linear_r, d, row_linear_r + x);
                store(linear_g, d, row_linear_g + x);
                store(linear_b, d, row_linear_b + x);
              }
            },
            "CenteredOpsinToLinear");
}

SIMD_ATTR void CenteredOpsinToOpsin(const Image3F& centered_opsin,
                                    ThreadPool* pool, Image3F* opsin) {
  PIK_CHECK(opsin->xsize() != 0);
  PROFILER_FUNC;
  // Opsin is padded to blocks; only produce valid output pixels.
  const size_t xsize = opsin->xsize();
  const size_t ysize = opsin->ysize();

  const SIMD_FULL(float) d;

  const auto center_x = set1(d, kXybCenter[0]);
  const auto center_y = set1(d, kXybCenter[1]);
  const auto center_b = set1(d, kXybCenter[2]);

  pool->Run(0, ysize,
            [&](const int task, const int thread) {
              const size_t y = task;

              // Faster than adding stride at end of loop.
              const float* PIK_RESTRICT row_copsin_x =
                  centered_opsin.ConstPlaneRow(0, y);
              const float* PIK_RESTRICT row_copsin_y =
                  centered_opsin.ConstPlaneRow(1, y);
              const float* PIK_RESTRICT row_copsin_b =
                  centered_opsin.ConstPlaneRow(2, y);

              float* PIK_RESTRICT row_opsin_x = opsin->PlaneRow(0, y);
              float* PIK_RESTRICT row_opsin_y = opsin->PlaneRow(1, y);
              float* PIK_RESTRICT row_opsin_b = opsin->PlaneRow(2, y);

              for (size_t x = 0; x < xsize; x += d.N) {
                const auto opsin_x = load(d, row_copsin_x + x) + center_x;
                const auto opsin_y = load(d, row_copsin_y + x) + center_y;
                const auto opsin_b = load(d, row_copsin_b + x) + center_b;
                store(opsin_x, d, row_opsin_x + x);
                store(opsin_y, d, row_opsin_y + x);
                store(opsin_b, d, row_opsin_b + x);
              }
            },
            "CenteredOpsinToOpsin");
}

SIMD_ATTR void OpsinToLinear(const Image3F& opsin, ThreadPool* pool,
                             Image3F* linear) {
  PIK_CHECK(linear->xsize() != 0);
  PROFILER_FUNC;
  // Opsin is padded to blocks; only produce valid output pixels.
  const size_t xsize = linear->xsize();
  const size_t ysize = linear->ysize();

  const SIMD_FULL(float) d;
  using V = SIMD_FULL(float)::V;

  pool->Run(0, ysize,
            [&](const int task, const int thread) {
              const size_t y = task;

              // Faster than adding via ByteOffset at end of loop.
              const float* PIK_RESTRICT row_opsin_x = opsin.ConstPlaneRow(0, y);
              const float* PIK_RESTRICT row_opsin_y = opsin.ConstPlaneRow(1, y);
              const float* PIK_RESTRICT row_opsin_b = opsin.ConstPlaneRow(2, y);

              float* PIK_RESTRICT row_linear_r = linear->PlaneRow(0, y);
              float* PIK_RESTRICT row_linear_g = linear->PlaneRow(1, y);
              float* PIK_RESTRICT row_linear_b = linear->PlaneRow(2, y);

              for (size_t x = 0; x < xsize; x += d.N) {
                const auto in_opsin_x = load(d, row_opsin_x + x);
                const auto in_opsin_y = load(d, row_opsin_y + x);
                const auto in_opsin_b = load(d, row_opsin_b + x);
                V linear_r, linear_g, linear_b;
                XybToRgb(d, in_opsin_x, in_opsin_y, in_opsin_b, inverse_matrix,
                         &linear_r, &linear_g, &linear_b);

                store(linear_r, d, row_linear_r + x);
                store(linear_g, d, row_linear_g + x);
                store(linear_b, d, row_linear_b + x);
              }
            },
            "OpsinToLinear");
}

}  // namespace pik
