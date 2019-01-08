// Copyright 2018 Google Inc. All Rights Reserved.
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

#ifndef GRADIENT_MAP_H_
#define GRADIENT_MAP_H_

#include "compressed_image_fwd.h"
#include "data_parallel.h"
#include "image.h"
#include "padded_bytes.h"
#include "quantizer.h"

namespace pik {

// TODO(user): Add unit tests. Verify that
// ComputeGradientMap(ApplyGradientMap(map)) == map.

// For encoding

// Computes the gradient map for the given image of DC
// values.
void ComputeGradientMap(const Image3F& opsin, bool grayscale,
                        const Quantizer& quantizer, ThreadPool* pool,
                        GradientMap* gradient);

void SerializeGradientMap(const GradientMap& gradient, const Rect& rect,
                          const Quantizer& quantizer, PaddedBytes* compressed);

// For decoding

Status DeserializeGradientMap(size_t xsize_dc, size_t ysize_dc, bool grayscale,
                              const Quantizer& quantizer,
                              const PaddedBytes& compressed, size_t* byte_pos,
                              GradientMap* gradient);

// Applies the gradient map to the decoded DC image.
void ApplyGradientMap(const GradientMap& gradient, const Quantizer& quantizer,
                      Image3F* opsin);

}  // namespace pik

#endif  // GRADIENT_MAP_H_
