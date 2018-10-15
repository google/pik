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

#ifndef COLOR_MANAGEMENT_H_
#define COLOR_MANAGEMENT_H_

// ICC profiles and color space conversions.

#include <stdint.h>
#include <memory>
#include <vector>

#include "color_encoding.h"
#include "common.h"
#include "data_parallel.h"
#include "image.h"
#include "padded_bytes.h"
#include "status.h"

namespace pik {

class ColorManagement {
  struct ContextDeleter {
    void operator()(void* p);
  };
  struct TransformDeleter {
    void operator()(void* p);
  };

 public:
  explicit ColorManagement(size_t num_threads);

  using Context = std::unique_ptr<void, ContextDeleter>;
  using Transform = std::unique_ptr<void, TransformDeleter>;

  // For ColorSpaceTransform.
  const std::vector<Context>& GetContexts() const { return contexts_; }

  // Returns false without changing "c" if pp.color_space is unsupported or
  // pp.gamma is outside (0, 1]. Otherwise, sets fields AND c->icc. Used by
  // codecs that provide their own non-ICC metadata.
  Status SetFromParams(const ProfileParams& pp, ColorEncoding* c) const;

  // Returns false without changing "c" if "icc" is invalid. Otherwise, sets
  // fields AND c->icc. Used by image codecs that read embedded ICC profiles.
  Status SetFromProfile(PaddedBytes&& icc, ColorEncoding* c) const;

  // Returns true and clears c->icc if a subsequent SetProfileFromFields
  // will generate an equivalent profile. If so, there is no need to send the
  // (large) profile in the bitstream.
  Status MaybeRemoveProfile(ColorEncoding* c) const;

  // Returns true if c->icc was successfully reconstructed from other fields.
  // This re-establishes the invariant (broken by MaybeRemoveProfile or changing
  // fields) that fields and c->icc are equivalent. Returning false indicates
  // the profile is lost/empty, which means ColorSpaceTransform will fail.
  Status SetProfileFromFields(ColorEncoding* c) const;

 private:
  std::vector<Context> contexts_;
};

// Per-thread state for a color transform.
class ColorSpaceTransform {
 public:
  // Allocates one for every Context, or returns false.
  Status Init(const std::vector<ColorManagement::Context>& contexts,
              const ColorEncoding& c_src, const ColorEncoding& c_dst,
              size_t xsize);

  float* PIK_RESTRICT BufSrc(const size_t thread) {
    return buf_src_.Row(thread);
  }

  float* PIK_RESTRICT BufDst(const size_t thread) {
    return buf_dst_.Row(thread);
  }

  // buf_X can either be from BufX() or caller-allocated, interleaved storage.
  void Run(const size_t thread, const float* buf_src, float* buf_dst);

 private:
  enum class ExtraTF {
    kNone,
    kPQ,
    kHLG,
    kSRGB,
  };

  // One per context - cannot share because of caching.
  std::vector<ColorManagement::Transform> transforms_;

  ImageF buf_src_;
  ImageF buf_dst_;
  size_t xsize_;
  bool skip_lcms_ = false;
  ExtraTF preprocess_ = ExtraTF::kNone;
  ExtraTF postprocess_ = ExtraTF::kNone;
};

}  // namespace pik

#endif  // COLOR_MANAGEMENT_H_
