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

#ifndef BUTTERAUGLI_DISTANCE_H_
#define BUTTERAUGLI_DISTANCE_H_

#include <vector>

#include "image.h"

namespace pik {

// Returns the butteraugli distance between rgb0 and rgb1.
// Both rgb0 and rgb1 are assumed to be in sRGB color space.
// If distmap is not null, it must be the same size as rgb0 and rgb1.
float ButteraugliDistance(const Image3B& rgb0, const Image3B& rgb1,
                          ImageF* distmap);

// Same as above, but rgb0 and rgb1 are linear RGB images.
float ButteraugliDistance(const Image3F& rgb0, const Image3F& rgb1,
                          ImageF* distmap);

// rgb0 and rgb1 are linear RGB images with optional alpha channel.
float ButteraugliDistance(const MetaImageF& rgb0, const MetaImageF& rgb1,
                          ImageF* distmap_out);

}  // namespace pik

#endif  // BUTTERAUGLI_DISTANCE_H_
