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

#ifndef BUTTERAUGLI_COMPARATOR_H_
#define BUTTERAUGLI_COMPARATOR_H_

#include "butteraugli/butteraugli.h"
#include "image.h"

namespace pik {

class ButteraugliComparator {
 public:
  ButteraugliComparator(const Image3B& srgb, float hf_asymmetry);
  ButteraugliComparator(const Image3F& opsin, float hf_asymmetry);

  void Compare(const Image3B& srgb);

  const ImageF& distmap() const { return distmap_; }
  float distance() const { return distance_; }

  void Mask(Image3F* mask, Image3F* mask_dc);

 private:
  const int xsize_;
  const int ysize_;
  butteraugli::ButteraugliComparator comparator_;
  float distance_;
  ImageF distmap_;
};

}  // namespace pik

#endif  // BUTTERAUGLI_COMPARATOR_H_
