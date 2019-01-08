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

#ifndef SALIENCY_MAP_H_
#define SALIENCY_MAP_H_

#include <memory>

#include "codec.h"
#include "image.h"
#include "padded_bytes.h"
#include "pik_params.h"
#include "status.h"

namespace pik {

Status ProduceSaliencyMap(const CompressParams& cparams,
                          const PaddedBytes* compressed, const CodecInOut* io,
                          ThreadPool* pool,
                          std::shared_ptr<ImageF>* out_heatmap);

}  // namespace pik

#endif  // SALIENCY_MAP_H_
