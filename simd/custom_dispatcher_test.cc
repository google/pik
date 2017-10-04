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

// Demo of using intrinsics without -mavx2 compile flags.

#include <stdio.h>
#include <numeric>   // iota
#include <random>

// Which of the intrinsics referenced by simd.h to use (default: all).
// This can be limited to e.g. SIMD_SSE4 for compilers that don't support AVX2.
#define SIMD_ENABLE ~0u

#include "simd/simd.h"
#include "simd/dispatch.h"

// GCC/Clang require certain code generation flags when using intrinsics.
// This file demonstrates the SIMD_ATTR approach for working around this
// requirement. It requires a recent GCC/Clang, or MSVC, which does not need
// extra flags in the first place.
#if SIMD_ENABLE_ANY && defined(SIMD_ATTR_TARGET)
#define SIMD_CUSTOM_DISPATCHER_TEST_ENABLED
#endif

#ifdef SIMD_CUSTOM_DISPATCHER_TEST_ENABLED

namespace pik {
namespace SIMD_NAMESPACE {
namespace {

// Demo of using the best enabled instruction set, and only calling it if
// supported by the current CPU. The implementation doesn't care exactly which
// instruction set it is (width-agnostic).

// Loops from [i, size), NumLanes<V>() at a time. This template avoids
// repeating code for V and vec1. Returns the next i, and assumes "i" is
// appropriately aligned for V.
template <class V, typename T>
SIMD_ATTR SIMD_INLINE size_t MulAddLoop(const T* SIMD_RESTRICT mul_array,
                                        const T* SIMD_RESTRICT add_array,
                                        size_t i, const size_t size,
                                        T* SIMD_RESTRICT x_array) {
  for (; i + NumLanes<V>() <= size; i += NumLanes<V>()) {
    const auto mul = load(V(), mul_array + i);
    const auto add = load(V(), add_array + i);
    auto x = load(V(), x_array + i);
    x = mul_add(mul, x, add);
    store(x, x_array + i);
  }
  return i;
}

// Computes x := x * mul + add over an array.
// Type-agnostic (caller-specified lane type) and width-agnostic.
template <typename T>
SIMD_ATTR SIMD_INLINE void MulAdd(const T* SIMD_RESTRICT mul,
                                  const T* SIMD_RESTRICT add, const size_t size,
                                  T* SIMD_RESTRICT x) {
  using V = vec<T, SIMD_TARGET>;
  printf("MulAdd %s\n", vec_name<V>());
  size_t i = 0;
  // Clang generates only load, store and FMA instructions if size is constexpr.
  i = MulAddLoop<V>(mul, add, i, size, x);
  MulAddLoop<vec1<T>>(mul, add, i, size, x);
}

// Generates input data and verifies the result. Note: top-level functions
// require NOINLINE to avoid propagating the function-specific attributes.
template <typename T>
SIMD_ATTR SIMD_NOINLINE void TestMulAdd() {
  std::mt19937 rng(1234);
  const size_t kSize = 64;
  SIMD_ALIGN T mul[kSize];
  SIMD_ALIGN T x[kSize];
  SIMD_ALIGN T add[kSize];
  for (size_t i = 0; i < kSize; ++i) {
    mul[i] = int32_t(rng()) & 0xF;
    x[i] = int32_t(rng()) & 0xFF;
    add[i] = int32_t(rng()) & 0xFF;
  }
  MulAdd(mul, add, kSize, x);
  const T sum = std::accumulate(x, x + kSize, T(0));
  printf("MulAdd sum %f\n", sum);
}

// Demo of targeting specific instruction sets and only calling them if
// supported.

// Generic implementation, "instantiated" for all supported instruction sets.
// Cannot be a function - that would require different SIMD_ATTR_*.
#define TEST_LOAD_STORE(vec_template)      \
  using V = vec_template<int32_t>;         \
  constexpr size_t N = NumLanes<V>();      \
  SIMD_ALIGN int32_t lanes[N];             \
  std::iota(lanes, lanes + N, 1);          \
  auto v = load(V(), lanes);               \
  SIMD_ALIGN int32_t lanes2[N];            \
  store(v, lanes2);                        \
  for (size_t i = 0; i < N; ++i) {         \
    if (lanes[i] != lanes2[i]) {           \
      printf("Mismatch in lane %zu\n", i); \
      SIMD_TRAP();                         \
    }                                      \
  }

SIMD_ATTR_AVX2 SIMD_NOINLINE void TestLoadStore_AVX2() {
  TEST_LOAD_STORE(vec256);
  printf("AVX2 ok\n");
}

SIMD_ATTR_SSE4 SIMD_NOINLINE void TestLoadStore_SSE4() {
  TEST_LOAD_STORE(vec128);
  printf("SSE4 ok\n");
}

void TestLoadStore_Portable() {
  TEST_LOAD_STORE(vec1);
  printf("Portable ok\n");
}

void RunTests() {
  const int supported = dispatch::SupportedTargets();

  // SIMD_TARGET is the "best" of all targets in SIMD_ENABLE. If it's actually
  // supported, run the code.
  if (dispatch::IsSupported<SIMD_TARGET>(supported)) {
    TestMulAdd<float>();
  }

  // We can also query the bit directly.
  if (supported & SIMD_SSE4) {
    TestLoadStore_SSE4();
  }
  if (dispatch::IsSupported<AVX2>(supported)) {
    TestLoadStore_AVX2();
  }

  TestLoadStore_Portable();
}

}  // namespace
}  // namespace SIMD_NAMESPACE
}  // namespace pik

#endif  // SIMD_CUSTOM_DISPATCHER_TEST_ENABLED

int main() {
#ifdef SIMD_CUSTOM_DISPATCHER_TEST_ENABLED
  pik::SIMD_NAMESPACE::RunTests();
  return 0;
#else
  printf("Compiler does not support SIMD_TARGET_ATTR => demo is disabled.\n");
  return 0;
#endif
}
