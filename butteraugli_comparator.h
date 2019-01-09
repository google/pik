// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef BUTTERAUGLI_COMPARATOR_H_
#define BUTTERAUGLI_COMPARATOR_H_

#include "butteraugli/butteraugli.h"
#include "image.h"

namespace pik {

class ButteraugliComparator {
 public:
  ButteraugliComparator(const Image3F& opsin, float hf_asymmetry,
                        float multiplier);

  void Compare(const Image3F& linear_rgb);

  const ImageF& distmap() const { return distmap_; }
  float distance() const { return distance_; }

  void Mask(Image3F* mask, Image3F* mask_dc);

 private:
  const int xsize_;
  const int ysize_;
  butteraugli::ButteraugliComparator comparator_;
  float distance_;
  float multiplier_;
  ImageF distmap_;
};

}  // namespace pik

#endif  // BUTTERAUGLI_COMPARATOR_H_
