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

#ifndef INSTRUCTION_SETS_H_
#define INSTRUCTION_SETS_H_

// Calls the best specialization of a template supported by the current CPU.
//
// Usage: for each dispatch site, declare a Functor template with a 'Target'
// argument, add a source file defining its operator() and instantiating
// Functor<PIK_TARGET>, and call InstructionSets::Run<Functor>(/*args*/).

#include <utility>  // std::forward

#include "arch_specific.h"  // PIK_TARGET_*
#include "compiler_specific.h"

namespace pik {

// Detects TargetBits and calls specializations of a user-defined functor.
class InstructionSets {
 public:
// Returns bit array of PIK_TARGET_* supported by the current CPU.
// The PIK_TARGET_Portable bit is guaranteed to be set.
#if PIK_ARCH_X64
  static TargetBits Supported();
#else
  static PIK_INLINE TargetBits Supported() { return PIK_TARGET_Portable; }
#endif

  // Chooses the best available "Target" for the current CPU, runs the
  // corresponding Func<Target>::operator()(args) and returns that Target
  // (a single bit). The overhead of dispatching is low, about 4 cycles, but
  // this should only be called infrequently (e.g. hoisting it out of loops).
  template <template <TargetBits> class Func, typename... Args>
  static PIK_INLINE TargetBits Run(Args&&... args) {
#if PIK_ARCH_X64
    const TargetBits supported = Supported();
    if (supported & PIK_TARGET_AVX2) {
      Func<PIK_TARGET_AVX2>()(std::forward<Args>(args)...);
      return PIK_TARGET_AVX2;
    }
    if (supported & PIK_TARGET_SSE41) {
      Func<PIK_TARGET_SSE41>()(std::forward<Args>(args)...);
      return PIK_TARGET_SSE41;
    }
#endif

    // No matching PIK_ARCH or no supported PIK_TARGET:
    Func<PIK_TARGET_Portable>()(std::forward<Args>(args)...);
    return PIK_TARGET_Portable;
  }

  // Calls Func<Target>::operator()(args) for all Target supported by the
  // current CPU, and returns their PIK_TARGET_* bits.
  template <template <TargetBits> class Func, typename... Args>
  static PIK_INLINE TargetBits RunAll(Args&&... args) {
    const TargetBits supported = Supported();

#if PIK_ARCH_X64
    if (supported & PIK_TARGET_AVX2) {
      Func<PIK_TARGET_AVX2>()(std::forward<Args>(args)...);
    }
    if (supported & PIK_TARGET_SSE41) {
      Func<PIK_TARGET_SSE41>()(std::forward<Args>(args)...);
    }
#endif

    Func<PIK_TARGET_Portable>()(std::forward<Args>(args)...);

    return supported;  // i.e. all that were run
  }
};

}  // namespace pik

#endif  // INSTRUCTION_SETS_H_
