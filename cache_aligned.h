// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef CACHE_ALIGNED_H_
#define CACHE_ALIGNED_H_

// Memory allocator with support for alignment + misalignment.

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
  // To avoid RFOs, match L2 fill size (pairs of lines).
  static constexpr size_t kAlignment = 2 * kCacheLineSize;

  // "offset" is added to the allocation size and allocated pointer in an
  // attempt to avoid 2K aliasing of consecutive allocations (e.g. Image).
  static void* Allocate(const size_t payload_size, const size_t offset = 0) {
    PIK_ASSERT(payload_size < (1ULL << 63));
    // Layout: |<alignment> Avoid2K|<allocated>  left_padding  | <payload>
    // Sizes : |kAlignment   offset|kPointerSize kMaxVectorSize| payload_size
    //         ^allocated..........^stash...............payload^
    const size_t header_size =
        kAlignment + offset + kPointerSize + kMaxVectorSize;
    void* allocated = malloc(header_size + payload_size);
    if (allocated == nullptr) return nullptr;
    uintptr_t payload = reinterpret_cast<uintptr_t>(allocated) + header_size;
    payload &= ~(kAlignment - 1);  // round down
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
    PIK_ASSERT(payload % kAlignment == 0);
    const uintptr_t stash = payload - kMaxVectorSize - kPointerSize;
    void* allocated;
    memcpy(&allocated, reinterpret_cast<const void*>(stash), kPointerSize);
    free(allocated);
  }

  // Overwrites "to_items" without loading it into cache (read-for-ownership).
  // Copies kCacheLineSize bytes from/to naturally aligned addresses.
  template <typename T>
  static SIMD_ATTR void StreamCacheLine(const T* PIK_RESTRICT from,
                                        T* PIK_RESTRICT to) {
    static_assert(16 % sizeof(T) == 0, "T must fit in a lane");
#if SIMD_TARGET_VALUE != SIMD_NONE
    constexpr size_t kLanes = 16 / sizeof(T);
    const SIMD_PART(T, kLanes) d;
    PIK_COMPILER_FENCE;
    const auto v0 = load(d, from + 0 * kLanes);
    const auto v1 = load(d, from + 1 * kLanes);
    const auto v2 = load(d, from + 2 * kLanes);
    const auto v3 = load(d, from + 3 * kLanes);
    static_assert(sizeof(v0) * 4 == kCacheLineSize, "Wrong #vectors");
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

// Avoids the need for a function pointer (deleter) in CacheAlignedUniquePtr.
struct CacheAlignedDeleter {
  void operator()(uint8_t* aligned_pointer) const {
    return CacheAligned::Free(aligned_pointer);
  }
};

using CacheAlignedUniquePtr = std::unique_ptr<uint8_t[], CacheAlignedDeleter>;

// Does not invoke constructors.
static inline CacheAlignedUniquePtr AllocateArray(const size_t entries,
                                           const size_t offset = 0) {
  const size_t size = entries * sizeof(uint8_t);
  return CacheAlignedUniquePtr(
      static_cast<uint8_t*>(CacheAligned::Allocate(size, offset)),
      CacheAlignedDeleter());
}

}  // namespace pik

#endif  // CACHE_ALIGNED_H_
