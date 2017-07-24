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

#include <string.h>  // memcpy
#include <emmintrin.h>
#include <memory>
#include "compiler_specific.h"
#include "status.h"

namespace pik {

// Functions that depend on the cache line size.
class CacheAligned {
 public:
  static constexpr size_t kPointerSize = sizeof(void*);
  static constexpr size_t kCacheLineSize = 64;

  static void* Allocate(const size_t bytes);

  // Template allows freeing pointer-to-const.
  template <typename T>
  static void Free(T* aligned_pointer) {
    if (aligned_pointer == nullptr) {
      return;
    }
    const char* const aligned = reinterpret_cast<const char*>(aligned_pointer);
    PIK_ASSERT(reinterpret_cast<uintptr_t>(aligned) % kCacheLineSize == 0);
    char* allocated;
    memcpy(&allocated, aligned - kPointerSize, kPointerSize);
    PIK_ASSERT(allocated <= aligned - kPointerSize);
    PIK_ASSERT(allocated >= aligned - kCacheLineSize);
    free(allocated);
  }

  // Overwrites "to_items" without loading it into cache (read-for-ownership).
  // Copies kCacheLineSize bytes from/to naturally aligned addresses.
  template <typename T>
  static void StreamCacheLine(const T* PIK_RESTRICT from_items,
                              T* PIK_RESTRICT to_items) {
    static_assert(sizeof(__m128i) % sizeof(T) == 0, "Cannot divide");
    const __m128i* const from = reinterpret_cast<const __m128i*>(from_items);
    __m128i* const to = reinterpret_cast<__m128i*>(to_items);
    PIK_COMPILER_FENCE;
    const __m128i v0 = _mm_load_si128(from + 0);
    const __m128i v1 = _mm_load_si128(from + 1);
    const __m128i v2 = _mm_load_si128(from + 2);
    const __m128i v3 = _mm_load_si128(from + 3);
    // Fences prevent the compiler from reordering loads/stores, which may
    // interfere with write-combining.
    PIK_COMPILER_FENCE;
    _mm_stream_si128(to + 0, v0);
    _mm_stream_si128(to + 1, v1);
    _mm_stream_si128(to + 2, v2);
    _mm_stream_si128(to + 3, v3);
    PIK_COMPILER_FENCE;
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
inline CacheAlignedUniquePtrT<T> AllocateArray(const size_t entries) {
  return CacheAlignedUniquePtrT<T>(
      static_cast<T*>(CacheAligned::Allocate(entries * sizeof(T))),
      CacheAligned::Free);
}

}  // namespace pik

#endif  // CACHE_ALIGNED_H_
