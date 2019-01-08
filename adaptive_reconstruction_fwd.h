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

#ifndef ADAPTIVE_RECONSTRUCTION_FWD_H_
#define ADAPTIVE_RECONSTRUCTION_FWD_H_

// Breaks the circular dependency between adaptive_reconstruction.h and
// pik_info.h.

#include "epf_stats.h"
#include "image.h"

namespace pik {

// Optional output(s).
struct AdaptiveReconstructionAux {
  void Assimilate(const AdaptiveReconstructionAux& other) {
    epf_stats.Assimilate(other.epf_stats);
    if (other.stretch != -1.0f) stretch = other.stretch;
    if (other.quant_scale != -1.0f) quant_scale= other.quant_scale;
  }

  void Print() const { epf_stats.Print(); }

  // Filled with the multiplier used to scale input pixels to [0, 255].
  float stretch = -1.0f;

  // Set to Quantizer::Scale().
  float quant_scale = -1.0f;

  // If not null, filled with difference between input and filtered image.
  Image3F* residual = nullptr;
  // If not null, filled with raw quant map used to compute sigma.
  ImageI* ac_quant = nullptr;
  // If not null, filled with AC strategy (for detecting DCT16)
  ImageB* ac_strategy = nullptr;

  EpfStats epf_stats;
};

}  // namespace pik

#endif  // ADAPTIVE_RECONSTRUCTION_FWD_H_
