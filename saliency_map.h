// Copyright 2018 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

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
