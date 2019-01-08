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

#include "opsin_inverse.h"

namespace pik {

namespace {

Image3F LinearFromOpsin(const Image3F& opsin) {
  Image3F linear(opsin.xsize(), opsin.ysize());
  OpsinToLinear(opsin, Rect(opsin), &linear);
  return linear;
}

}  // namespace

ButteraugliComparator::ButteraugliComparator(const Image3F& opsin,
                                             float hf_asymmetry,
                                             float multiplier)
    : xsize_(opsin.xsize()),
      ysize_(opsin.ysize()),
      comparator_(ScaleImage(multiplier, LinearFromOpsin(opsin)), hf_asymmetry),
      distance_(0.0),
      multiplier_(multiplier),
      distmap_(xsize_, ysize_) {
  FillImage(0.0f, &distmap_);
}

void ButteraugliComparator::Compare(const Image3F& linear_rgb) {
  PIK_CHECK(SameSize(distmap_, linear_rgb));
  if (multiplier_ == 1) {
    comparator_.Diffmap(linear_rgb, distmap_);
  } else {
    comparator_.Diffmap(ScaleImage(multiplier_, linear_rgb), distmap_);
  }
  distance_ = butteraugli::ButteraugliScoreFromDiffmap(distmap_);
}

void ButteraugliComparator::Mask(Image3F* mask, Image3F* mask_dc) {
  comparator_.Mask(mask, mask_dc);
}

}  // namespace pik
