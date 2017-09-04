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

#include "butteraugli_comparator.h"

#include <stddef.h>
#include <array>
#include <memory>
#include <vector>

#include "compiler_specific.h"
#include "gamma_correct.h"
#include "opsin_inverse.h"
#include "status.h"

namespace pik {

namespace {

// REQUIRES: xsize <= srgb.xsize(), ysize <= srgb.ysize()
std::vector<butteraugli::ImageF> SrgbToLinearRgb(
    const int xsize, const int ysize,
    const Image3B& srgb) {
  PIK_ASSERT(xsize <= srgb.xsize());
  PIK_ASSERT(ysize <= srgb.ysize());
  const float* lut = Srgb8ToLinearTable();
  std::vector<butteraugli::ImageF> planes =
      butteraugli::CreatePlanes<float>(xsize, ysize, 3);
  for (size_t y = 0; y < ysize; ++y) {
    auto row_in = srgb.Row(y);
    for (int c = 0; c < 3; ++c) {
      float* const PIK_RESTRICT row_out = planes[c].Row(y);
      for (size_t x = 0; x < xsize; ++x) {
        row_out[x] = lut[row_in[c][x]];
      }
    }
  }
  return planes;
}

// REQUIRES: xsize <= srgb.xsize(), ysize <= srgb.ysize()
std::vector<butteraugli::ImageF> OpsinToLinearRgb(
    const int xsize, const int ysize,
    const Image3F& opsin) {
  PIK_ASSERT(xsize <= opsin.xsize());
  PIK_ASSERT(ysize <= opsin.ysize());
  std::vector<butteraugli::ImageF> planes =
      butteraugli::CreatePlanes<float>(xsize, ysize, 3);
  for (size_t y = 0; y < ysize; ++y) {
    auto row_in = opsin.Row(y);
    std::array<float*, 3> row_out{{
        planes[0].Row(y), planes[1].Row(y), planes[2].Row(y) }};
    for (int x = 0; x < xsize; ++x) {
      XybToRgb(row_in[0][x], row_in[1][x], row_in[2][x],
               &row_out[0][x], &row_out[1][x], &row_out[2][x]);
    }
  }
  return planes;
}

Image3F Image3FromButteraugliPlanes(
    const std::vector<butteraugli::ImageF>& planes) {
  Image3F img(planes[0].xsize(), planes[0].ysize());
  for (size_t y = 0; y < img.ysize(); ++y) {
    auto row_out = img.Row(y);
    for (int c = 0; c < 3; ++c) {
      auto row_in = planes[c].Row(y);
      for (size_t x = 0; x < img.xsize(); ++x) {
        row_out[c][x] = row_in[x];
      }
    }
  }
  return img;
}

}  // namespace

ButteraugliComparator::ButteraugliComparator(const Image3B& srgb)
    : xsize_(srgb.xsize()),
      ysize_(srgb.ysize()),
      comparator_(SrgbToLinearRgb(xsize_, ysize_, srgb)),
      distance_(0.0),
      distmap_(xsize_, ysize_, 0) {
}

ButteraugliComparator::ButteraugliComparator(const Image3F& opsin)
    : xsize_(opsin.xsize()),
      ysize_(opsin.ysize()),
      comparator_(OpsinToLinearRgb(xsize_, ysize_, opsin)),
      distance_(0.0),
      distmap_(xsize_, ysize_, 0) {
}

void ButteraugliComparator::Compare(const Image3B& srgb) {
  comparator_.Diffmap(SrgbToLinearRgb(xsize_, ysize_, srgb), distmap_);
  distance_ = butteraugli::ButteraugliScoreFromDiffmap(distmap_);
}

void ButteraugliComparator::Mask(Image3F* mask, Image3F* mask_dc) {
  std::vector<butteraugli::ImageF> ba_mask, ba_mask_dc;
  comparator_.Mask(&ba_mask, &ba_mask_dc);
  *mask = Image3FromButteraugliPlanes(ba_mask);
  *mask_dc = Image3FromButteraugliPlanes(ba_mask_dc);
}

}  // namespace pik
