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

#ifndef SIMD_DISPATCH_H_
#define SIMD_DISPATCH_H_

// Detects instruction sets available on the current CPU, decides which is best,
// and calls the matching specialization of a template.
//
// Usage: for each dispatch site, declare a Functor class, add a source file
// that specializes its operator()<SIMD_TARGET>, dispatch::Run<Functor>(..).

#include <utility>  // std::forward

#include "simd/port.h"

namespace pik {
namespace dispatch {

// Returns bit array of instruction sets supported by the current CPU,
// e.g. SIMD_SSE4 | SIMD_AVX2.
int SupportedTargets();

// Returns true if the Target's bit is set in "targets" (from SupportedTargets).
template <class Target>
constexpr bool IsSupported(const int targets) {
  return (targets & Target::value);
}
// Special-case for NONE - always supported, but its value is zero.
template <>
constexpr bool IsSupported<NONE>(const int targets) {
  return true;
}

// Chooses "kTarget", the best instruction set supported by the current CPU,
// and returns func.operator()<Target>(args). The dispatch overhead is low,
// about 4 cycles, but this should be called infrequently. The member function
// template (as opposed to a class template) allows stateful functors.
template <class Func, typename... Args>
SIMD_INLINE auto Run(Func&& func, Args&&... args)
    -> decltype(std::forward<Func>(func).template operator()<NONE>(
        std::forward<Args>(args)...)) {
  const int supported = SupportedTargets();
  (void)supported;
  // NOTE: do not check SIMD_ENABLE_*: SIMD_ENABLE might be zero in this
  // translation unit, but the instantiation[s] can still be called.
#if SIMD_ARCH == SIMD_ARCH_X86
  if (supported & SIMD_AVX2) {
    return std::forward<Func>(func).template operator()<AVX2>(
        std::forward<Args>(args)...);
  }
  if (supported & SIMD_SSE4) {
    return std::forward<Func>(func).template operator()<SSE4>(
        std::forward<Args>(args)...);
  }
#elif SIMD_ARCH == SIMD_ARCH_ARM
  if (supported & SIMD_ARM8) {
    return std::forward<Func>(func).template operator()<ARM8>(
        std::forward<Args>(args)...);
  }
#endif

  return std::forward<Func>(func).template operator()<NONE>(
      std::forward<Args>(args)...);
}

// Calls func.operator()<Target>(args) for all instruction sets in "targets"
// (typically the return value of SupportedTargets).
template <class Func, typename... Args>
SIMD_INLINE void ForeachTarget(const int targets, Func&& func, Args&&... args) {
  // NOTE: do not check SIMD_ENABLE_*: SIMD_ENABLE might be zero in this
  // translation unit, but the instantiation[s] can still be called.
#if SIMD_ARCH == SIMD_ARCH_X86
  if (targets & SIMD_SSE4) {
    std::forward<Func>(func).template operator()<SSE4>(
        std::forward<Args>(args)...);
  }
  if (targets & SIMD_AVX2) {
    std::forward<Func>(func).template operator()<AVX2>(
        std::forward<Args>(args)...);
  }
#elif SIMD_ARCH == SIMD_ARCH_ARM
  if (targets & SIMD_ARM8) {
    std::forward<Func>(func).template operator()<ARM8>(
        std::forward<Args>(args)...);
  }
#endif

  std::forward<Func>(func).template operator()<NONE>(
      std::forward<Args>(args)...);
}

}  // namespace dispatch
}  // namespace pik

#endif  // SIMD_DISPATCH_H_
