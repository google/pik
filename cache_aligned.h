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

#ifndef CACHE_ALIGNED_H_
#define CACHE_ALIGNED_H_

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>  // memcpy
#include <algorithm>
#include <memory>
#include <new>

#include "arch_specific.h"
#include "compiler_specific.h"
#include "simd/simd.h"
#include "status.h"

namespace pik {

// Functions that depend on the cache line size.
class CacheAligned {
 public:
  static constexpr size_t kPointerSize = sizeof(void*);
  static constexpr size_t kCacheLineSize = 64;

  // "offset" is added to the allocation size and allocated pointer in an
  // attempt to avoid 2K aliasing of consecutive allocations (e.g. Image).
  static void* Allocate(const size_t payload_size, const size_t offset = 0) {
    PIK_ASSERT(payload_size < (1ULL << 63));
    // Layout: |<alignment>   Avoid2K|<allocated>  left_padding  | <payload>
    // Sizes : |kCacheLineSize offset|kPointerSize kMaxVectorSize| payload_size
    //         ^allocated............^stash...............payload^
    const size_t header_size =
        kCacheLineSize + offset + kPointerSize + kMaxVectorSize;
    void* allocated = malloc(header_size + payload_size);
    if (allocated == nullptr) return nullptr;
    uintptr_t payload = reinterpret_cast<uintptr_t>(allocated) + header_size;
    payload &= ~(kCacheLineSize - 1);  // round down
    const uintptr_t stash = payload - kMaxVectorSize - kPointerSize;
    memcpy(reinterpret_cast<void*>(stash), &allocated, kPointerSize);
    return reinterpret_cast<void*>(payload);
  }

  // Template allows freeing pointer-to-const.
  template <typename T>
  static void Free(T* aligned_pointer) {
    if (aligned_pointer == nullptr) {
      return;
    }
    const uintptr_t payload = reinterpret_cast<uintptr_t>(aligned_pointer);
    PIK_ASSERT(payload % kCacheLineSize == 0);
    const uintptr_t stash = payload - kMaxVectorSize - kPointerSize;
    void* allocated;
    memcpy(&allocated, reinterpret_cast<const void*>(stash), kPointerSize);
    free(allocated);
  }

  // Overwrites "to_items" without loading it into cache (read-for-ownership).
  // Copies kCacheLineSize bytes from/to naturally aligned addresses.
  template <typename T>
  static void StreamCacheLine(const T* PIK_RESTRICT from, T* PIK_RESTRICT to) {
    static_assert(16 % sizeof(T) == 0, "T must fit in a lane");
#if SIMD_TARGET_VALUE != SIMD_NONE
    constexpr size_t kLanes = 16 / sizeof(T);
    const SIMD_NAMESPACE::Part<T, kLanes> d;
    PIK_COMPILER_FENCE;
    const auto v0 = load(d, from + 0 * kLanes);
    const auto v1 = load(d, from + 1 * kLanes);
    const auto v2 = load(d, from + 2 * kLanes);
    const auto v3 = load(d, from + 3 * kLanes);
    // Fences prevent the compiler from reordering loads/stores, which may
    // interfere with write-combining.
    PIK_COMPILER_FENCE;
    stream(v0, d, to + 0 * kLanes);
    stream(v1, d, to + 1 * kLanes);
    stream(v2, d, to + 2 * kLanes);
    stream(v3, d, to + 3 * kLanes);
    PIK_COMPILER_FENCE;
#else
    memcpy(to, from, kCacheLineSize);
#endif
  }
};

template <typename T>
using CacheAlignedUniquePtrT = std::unique_ptr<T, void (*)(T*)>;

using CacheAlignedUniquePtr = CacheAlignedUniquePtrT<uint8_t>;

template <typename T>
inline void DestroyAndAlignedFree(T* t) {
  t->~T();
  CacheAligned::Free(t);
}

template <typename T, typename... Args>
inline CacheAlignedUniquePtrT<T> Allocate(Args&&... args) {
  void* mem = CacheAligned::Allocate(sizeof(T));
  T* t = new (mem) T(std::forward<Args>(args)...);
  return CacheAlignedUniquePtrT<T>(t, &DestroyAndAlignedFree<T>);
}

// Does not invoke constructors.
template <typename T = uint8_t>
inline CacheAlignedUniquePtrT<T> AllocateArray(const size_t entries,
                                               const size_t offset = 0) {
  return CacheAlignedUniquePtrT<T>(
      static_cast<T*>(CacheAligned::Allocate(entries * sizeof(T), offset)),
      CacheAligned::Free);
}

}  // namespace pik

#endif  // CACHE_ALIGNED_H_
