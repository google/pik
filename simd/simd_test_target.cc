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

#include <stdarg.h>
#include <stdio.h>
#include <limits>

#include "simd.h"
#include "simd_test_target.h"

namespace simd {
namespace SIMD_NAMESPACE {
namespace {

// Returns a vector with lane i=0..N-1 set to "first" + i. Unique per-lane
// values are required to detect lane-crossing bugs.
template <class V>
SIMD_INLINE V Iota(const Lane<V> first = 0) {
  constexpr size_t N = NumLanes<V>();
  SIMD_ALIGN Lane<V> lanes[N];
  for (size_t i = 0; i < N; ++i) {
    lanes[i] = first + i;
  }
  return load(V(), lanes);
}

// Test failure: prints message to stderr and aborts.
[[noreturn]] inline void TestFailed(const char* format, ...) {
  va_list args;
  va_start(args, format);
  vfprintf(stderr, format, args);
  va_end(args);
  SIMD_TRAP();
}

// Compare non-vector T.
template <typename T>
void AssertEqual(const T expected, const T actual, const int line = -1,
                 const size_t lane = 0, const char* vec_name = "") {
  if (expected == actual) return;
  char expected_buf[30];
  char actual_buf[30];
  ToString(expected, expected_buf);
  ToString(actual, actual_buf);
  TestFailed("line %d, %s lane %zu mismatch: expected %s, got %s.\n", line,
             vec_name, lane, expected_buf, actual_buf);
}

#define ASSERT_EQ(expected, actual) AssertEqual(expected, actual, __LINE__)

// Compare expected vector to vector.
template <class V>
void AssertVecEqual(const V expected, const V actual, const int line) {
  using T = Lane<V>;
  constexpr size_t N = NumLanes<V>();
  SIMD_ALIGN T actual_lanes[N];
  SIMD_ALIGN T expected_lanes[N];
  store(actual, actual_lanes);
  store(expected, expected_lanes);
  const char* name = vec_name<V>();
  for (size_t i = 0; i < N; ++i) {
    AssertEqual(expected_lanes[i], actual_lanes[i], line, i, name);
  }
}

// Compare expected lanes to vector.
template <class V, typename Lane, size_t N>
void AssertVecEqual(const Lane (&expected)[N], const V actual, const int line) {
  static_assert(N == NumLanes<V>(), "Size mismatch");
  AssertVecEqual(load_unaligned(V(), expected), actual, line);
}

#define ASSERT_VEC_EQ(expected, actual) \
  AssertVecEqual(expected, actual, __LINE__)

// Type lists

// Calls Test<T>::operator() for each lane type, passing the expected vec_name.
template <template <typename, class> class Test>
void ForeachUnsignedLaneType() {
  Test<uint8_t, vec<uint8_t>>()();
  Test<uint16_t, vec<uint16_t>>()();
  Test<uint32_t, vec<uint32_t>>()();
  Test<uint64_t, vec<uint64_t>>()();
}

template <template <typename, class> class Test>
void ForeachSignedLaneType() {
  Test<int8_t, vec<int8_t>>()();
  Test<int16_t, vec<int16_t>>()();
  Test<int32_t, vec<int32_t>>()();
  Test<int64_t, vec<int64_t>>()();
}

template <template <typename, class> class Test>
void ForeachFloatLaneType() {
  Test<float, vec<float>>()();
  Test<double, vec<double>>()();
}

template <template <typename, class> class Test>
void ForeachLaneType() {
  ForeachUnsignedLaneType<Test>();
  ForeachSignedLaneType<Test>();
  ForeachFloatLaneType<Test>();
}

namespace examples {

void Copy(const uint8_t* SIMD_RESTRICT from, const size_t size,
          uint8_t* SIMD_RESTRICT to) {
  // Width-agnostic (library-specified NumLanes)
  using V = vec<uint8_t>;
  size_t i = 0;
  for (; i + NumLanes<V>() <= size; i += NumLanes<V>()) {
    const auto bytes = load(V(), from + i);
    store(bytes, to + i);
  }

  for (; i < size; i += NumLanes<vec1<uint8_t>>()) {
    // (Same loop body as above, could factor into a shared template)
    const auto bytes = load(vec1<uint8_t>(), from + i);
    store(bytes, to + i);
  }
}

void TestCopy() {
  RandomState rng = {1234};
  const size_t kSize = 34;
  SIMD_ALIGN uint8_t from[kSize];
  for (size_t i = 0; i < kSize; ++i) {
    from[i] = Random32(&rng) & 0xFF;
  }
  SIMD_ALIGN uint8_t to[kSize];
  Copy(from, kSize, to);
  for (size_t i = 0; i < kSize; ++i) {
    ASSERT_EQ(from[i], to[i]);
  }
}

#if SIMD_ENABLE_ANY

int GetMostSignificantBits(const uint8_t* SIMD_RESTRICT from) {
  // Fixed-size, can use template or type alias.
  static_assert(sizeof(vec128<uint8_t>) == sizeof(u8x16), "Size mismatch");
  const auto bytes = load(u8x16(), from);
  return ext::movemask(bytes);  // 16 bits, one from each byte
}

#endif

void TestGetMostSignificantBits() {
#if SIMD_ENABLE_ANY
  SIMD_ALIGN const uint8_t from[16] = {0, 0, 0xF0, 0x80, 0, 0x10, 0x20, 0x40,
                                       1, 2, 4,    8,    3, 7,    15,   31};
  ASSERT_EQ(0x000C, GetMostSignificantBits(from));
#endif
}

template <typename T>
void MulAdd(const T* SIMD_RESTRICT mul_array, const T* SIMD_RESTRICT add_array,
            const size_t size, T* SIMD_RESTRICT x_array) {
  // Type-agnostic (caller-specified lane type) and width-agnostic (uses
  // best available instruction set).
  using V = vec<T>;
  for (size_t i = 0; i < size; i += NumLanes<V>()) {
    const auto mul = load(V(), mul_array + i);
    const auto add = load(V(), add_array + i);
    auto x = load(V(), x_array + i);
    x = mul_add(mul, x, add);
    store(x, x_array + i);
  }
}

template <typename T>
T SumMulAdd() {
  RandomState rng = {1234};
  const size_t kSize = 64;
  SIMD_ALIGN T mul[kSize];
  SIMD_ALIGN T x[kSize];
  SIMD_ALIGN T add[kSize];
  for (size_t i = 0; i < kSize; ++i) {
    mul[i] = Random32(&rng) & 0xF;
    x[i] = Random32(&rng) & 0xFF;
    add[i] = Random32(&rng) & 0xFF;
  }
  MulAdd(mul, add, kSize, x);
  double sum = 0.0;
  for (auto xi : x) {
    sum += xi;
  }
  return sum;
}

void TestExamples() {
  TestCopy();

  TestGetMostSignificantBits();

  ASSERT_EQ(75598.0f, SumMulAdd<float>());
  ASSERT_EQ(75598.0, SumMulAdd<double>());
}

}  // namespace examples

namespace basic {

// Test the ToString used to output test failures

void TestToString() {
  char buf[32];
  const char* end;

  end = ToString(int64_t(0), buf);
  ASSERT_EQ('0', end[-1]);
  ASSERT_EQ('\0', end[0]);

  end = ToString(int64_t(3), buf);
  ASSERT_EQ('3', end[-1]);
  ASSERT_EQ('\0', end[0]);

  end = ToString(int64_t(-1), buf);
  ASSERT_EQ('-', end[-2]);
  ASSERT_EQ('1', end[-1]);
  ASSERT_EQ('\0', end[0]);

  ToString(0x7FFFFFFFFFFFFFFFLL, buf);
  ASSERT_EQ(true, StringsEqual("9223372036854775807", buf));

  ToString(int64_t(0x8000000000000000ULL), buf);
  ASSERT_EQ(true, StringsEqual("-9223372036854775808", buf));

  ToString(0.0, buf);
  ASSERT_EQ(true, StringsEqual("0.0", buf));
  ToString(4.0, buf);
  ASSERT_EQ(true, StringsEqual("4.0", buf));
  ToString(-1.0, buf);
  ASSERT_EQ(true, StringsEqual("-1.0", buf));
  ToString(-1.25, buf);
  ASSERT_EQ(true, StringsEqual("-1.250", buf));
  ToString(2.125, buf);
  ASSERT_EQ(true, StringsEqual("2.125", buf));
}

template <typename T, class V>
struct TestIsVec {
  void operator()() const {
    static_assert(IsVec<V>::value, "V should be a vector");
    static_assert(!IsVec<T>::value, "T should not be a vector");
  }
};

template <typename T, class V>
struct TestIsUnsigned {
  void operator()() const {
    static_assert(!IsFloat<T>(), "Expected !IsFloat");
    static_assert(!IsSigned<T>(), "Expected !IsSigned");
  }
};

template <typename T, class V>
struct TestIsSigned {
  void operator()() const {
    static_assert(!IsFloat<T>(), "Expected !IsFloat");
    static_assert(IsSigned<T>(), "Expected IsSigned");
  }
};

template <typename T, class V>
struct TestIsFloat {
  void operator()() const {
    static_assert(IsFloat<T>(), "Expected IsFloat");
    static_assert(IsSigned<T>(), "Floats are also considered signed");
  }
};

void TestType() {
  ForeachUnsignedLaneType<TestIsUnsigned>();
  ForeachSignedLaneType<TestIsSigned>();
  ForeachFloatLaneType<TestIsFloat>();
}

template <typename T, class V>
struct TestSize {
  void operator()() const {
#if SIMD_ENABLE_ANY
    // Every instruction set provides 128-bit vectors
    static_assert(sizeof(vec128<T>) == 16, "vec128 size mismatch");
    // Max is at least that large
    static_assert(sizeof(V) >= 16, "V size mismatch");
#endif
  }
};

template <typename T, class V>
struct TestName {
  void operator()() const {
    char expected[7] = {IsFloat<T>() ? 'f' : (IsSigned<T>() ? 'i' : 'u')};
    char* end = ToString(sizeof(T) * 8, expected + 1);
    *end++ = 'x';
    end = ToString(NumLanes<V>::value, end);
    if (!StringsEqual(expected, vec_name<V>())) {
      TestFailed("Name mismatch: expected %s, got %s\n", expected,
                 vec_name<V>());
    }
  }
};

template <typename T, class V>
struct TestSet {
  void operator()() const {
    const size_t N = NumLanes<V>();

    // setzero
    const V v0 = setzero(V());
    T lanes[N] = {T(0)};
    ASSERT_VEC_EQ(lanes, v0);

    // set1
    const V v2 = set1(V(), T(2));
    for (size_t i = 0; i < N; ++i) {
      lanes[i] = 2;
    }
    ASSERT_VEC_EQ(lanes, v2);
  }
};

template <typename T, class V>
struct TestCopyAndAssign {
  void operator()() const {
    // copy V
    const V v3 = Iota<V>(3);
    V v3b(v3);
    ASSERT_VEC_EQ(v3, v3b);

    // assign V
    V v3c;
    v3c = v3;
    ASSERT_VEC_EQ(v3, v3c);
  }
};

// It's fine to cast between signed/unsigned with same-sized T.
template <typename FromT, typename ToT>
void TestCastT() {
  using FromV = vec<FromT>;
  using ToV = vec<ToT>;
  const FromV from = Iota<FromV>(1);
  const ToV to(from);
  ASSERT_VEC_EQ(from, FromV(to));
}

void TestCast() {
  TestCastT<int8_t, uint8_t>();
  TestCastT<uint8_t, int8_t>();

  TestCastT<int16_t, uint16_t>();
  TestCastT<uint16_t, int16_t>();

  TestCastT<int32_t, uint32_t>();
  TestCastT<uint32_t, int32_t>();

  TestCastT<int64_t, uint64_t>();
  TestCastT<uint64_t, int64_t>();

  // No float.
}

template <typename T, class V>
struct TestHalf {
  void operator()() const {
    constexpr size_t N = NumLanes<V>();

    const V v = Iota<V>(1);
    SIMD_ALIGN T lanes[N];
    for (size_t i = 0; i < N; ++i) {
      lanes[i] = 123;
    }
    const auto lo = lower_half(v);
    store(lo, lanes);
    size_t i = 0;
    for (; i < (N + 1) / 2; ++i) {
      ASSERT_EQ(T(i + 1), lanes[i]);
    }
    // Other lanes remain unchanged
    for (; i < N; ++i) {
      ASSERT_EQ(T(123), lanes[i]);
    }

    const auto hi = upper_half(v);
    store(hi, lanes);
    i = 0;
    for (; i < (N + 1) / 2; ++i) {
      ASSERT_EQ(T(i + 1 + (N == 1 ? 0 : N / 2)), lanes[i]);
    }
    // Other lanes remain unchanged
    for (; i < N; ++i) {
      ASSERT_EQ(T(123), lanes[i]);
    }

    // Can convert to full (upper half undefined)
    const V v2 = from_half(lo);
    store(v2, lanes);
    for (i = 0; i < (N + 1) / 2; ++i) {
      ASSERT_EQ(T(i + 1), lanes[i]);
    }
  }
};

void TestBasic() {
  TestToString();
  ForeachLaneType<TestIsVec>();
  ForeachLaneType<TestSize>();
  TestType();
  ForeachLaneType<TestName>();
  ForeachLaneType<TestSet>();
  ForeachLaneType<TestCopyAndAssign>();
  ForeachLaneType<TestHalf>();
  TestCast();
}

}  // namespace basic

namespace arithmetic {

template <typename T, class V>
struct TestPlusMinus {
  void operator()() const {
    constexpr size_t N = NumLanes<V>();
    const V v2 = Iota<V>(2);
    const V v3 = Iota<V>(3);
    const V v4 = Iota<V>(4);

    T lanes[N];
    for (size_t i = 0; i < N; ++i) {
      lanes[i] = (2 + i) + (3 + i);
    }
    ASSERT_VEC_EQ(lanes, v2 + v3);
    ASSERT_VEC_EQ(v3, (v2 + v3) - v2);

    for (size_t i = 0; i < N; ++i) {
      lanes[i] = (2 + i) + (4 + i);
    }
    V sum = v2;
    sum += v4;  // sum == 6,8..
    ASSERT_VEC_EQ(lanes, sum);

    sum -= v4;
    ASSERT_VEC_EQ(v2, sum);
  }
};

template <typename T>
void TestUnsignedSaturatingArithmetic() {
  using V = vec<T>;
  const V v0 = setzero(V());
  const V vi = Iota<V>(1);
  const V vm = set1(V(), UnsignedMax<T>());

  ASSERT_VEC_EQ(v0 + v0, add_sat(v0, v0));
  ASSERT_VEC_EQ(v0 + vi, add_sat(v0, vi));
  ASSERT_VEC_EQ(v0 + vm, add_sat(v0, vm));
  ASSERT_VEC_EQ(vm, add_sat(vi, vm));
  ASSERT_VEC_EQ(vm, add_sat(vm, vm));

  ASSERT_VEC_EQ(v0, sub_sat(v0, v0));
  ASSERT_VEC_EQ(v0, sub_sat(v0, vi));
  ASSERT_VEC_EQ(v0, sub_sat(vi, vi));
  ASSERT_VEC_EQ(v0, sub_sat(vi, vm));
  ASSERT_VEC_EQ(vm - vi, sub_sat(vm, vi));
}

template <typename T>
void TestSignedSaturatingArithmetic() {
  using V = vec<T>;
  constexpr size_t N = NumLanes<V>();

  const V v0 = setzero(V());
  const V vi = Iota<V>(1);
  const V vpm = set1(V(), SignedMax<T>());
  const V vn = Iota<V>(-T(N));
  const V vnm = set1(V(), std::numeric_limits<T>::lowest());

  ASSERT_VEC_EQ(v0, add_sat(v0, v0));
  ASSERT_VEC_EQ(vi, add_sat(v0, vi));
  ASSERT_VEC_EQ(vpm, add_sat(v0, vpm));
  ASSERT_VEC_EQ(vpm, add_sat(vi, vpm));
  ASSERT_VEC_EQ(vpm, add_sat(vpm, vpm));

  ASSERT_VEC_EQ(v0, sub_sat(v0, v0));
  ASSERT_VEC_EQ(v0 - vi, sub_sat(v0, vi));
  ASSERT_VEC_EQ(vn, sub_sat(vn, v0));
  ASSERT_VEC_EQ(vnm, sub_sat(vnm, vi));
  ASSERT_VEC_EQ(vnm, sub_sat(vnm, vpm));
}

void TestSaturatingArithmetic() {
  TestUnsignedSaturatingArithmetic<uint8_t>();
  TestUnsignedSaturatingArithmetic<uint16_t>();
  TestSignedSaturatingArithmetic<int8_t>();
  TestSignedSaturatingArithmetic<int16_t>();
}

template <typename T>
void TestAverageT() {
  using V = vec<T>;
  const V v0 = setzero(V());
  const V v1 = set1(V(), T(1));
  const V v2 = set1(V(), T(2));

  ASSERT_VEC_EQ(v0, avg(v0, v0));
  ASSERT_VEC_EQ(v1, avg(v0, v1));
  ASSERT_VEC_EQ(v1, avg(v1, v1));
  ASSERT_VEC_EQ(v2, avg(v1, v2));
  ASSERT_VEC_EQ(v2, avg(v2, v2));
}

void TestAverage() {
  TestAverageT<uint8_t>();
  TestAverageT<uint16_t>();
}

template <typename T>
void TestUnsignedShifts() {
  using V = vec<T>;
  const size_t N = NumLanes<V>();
  constexpr int kSign = (sizeof(T) * 8) - 1;
  const V v0 = setzero(V());
  const V vi = Iota<V>();
  T lanes[N];

  // Shifting out of right side => zero
  ASSERT_VEC_EQ(v0, vi >> 7);

  // Shifting out of left side => zero
  ASSERT_VEC_EQ(v0, vi << (kSign + 1));

  // Simple left shift
  for (size_t i = 0; i < N; ++i) {
    lanes[i] = i << 1;
  }
  ASSERT_VEC_EQ(lanes, vi << 1);

  // Simple right shift
  for (size_t i = 0; i < N; ++i) {
    lanes[i] = i >> 1;
  }
  ASSERT_VEC_EQ(lanes, vi >> 1);

  for (size_t i = 0; i < N; ++i) {
    lanes[i] = static_cast<T>((i << kSign) & ~T(0));
  }
  ASSERT_VEC_EQ(lanes, vi << kSign);
}

template <typename T>
void TestSignedShifts() {
  using V = vec<T>;
  const size_t N = NumLanes<V>();
  constexpr int kSign = (sizeof(T) * 8) - 1;
  const V v0 = setzero(V());
  const V vi = Iota<V>();
  T lanes[N];

  // Shifting out of right side => zero
  ASSERT_VEC_EQ(v0, vi >> 7);

  // Shifting out of left side => zero
  ASSERT_VEC_EQ(v0, vi << (kSign + 1));

  // Simple left shift
  for (size_t i = 0; i < N; ++i) {
    lanes[i] = i << 1;
  }
  ASSERT_VEC_EQ(lanes, vi << 1);

  // Simple right shift
  for (size_t i = 0; i < N; ++i) {
    lanes[i] = i >> 1;
  }
  ASSERT_VEC_EQ(lanes, vi >> 1);

  // Sign extension
  constexpr T min = std::numeric_limits<T>::min();
  const V vn = Iota<V>(min);
  for (size_t i = 0; i < N; ++i) {
    lanes[i] = (min + i) >> 1;
  }
  ASSERT_VEC_EQ(lanes, vn >> 1);

  // Shifting negative left
  for (size_t i = 0; i < N; ++i) {
    lanes[i] = (min + i) << 1;
  }
  ASSERT_VEC_EQ(lanes, vn << 1);
}

void TestShifts() {
  // No u8.
  TestUnsignedShifts<uint16_t>();
  TestUnsignedShifts<uint32_t>();
  TestUnsignedShifts<uint64_t>();
  // No i8.
  TestSignedShifts<int16_t>();
  TestSignedShifts<int32_t>();
  // No i64/f32/f64.
}

template <typename T>
void TestVariableShiftsT() {
#if SIMD_ENABLE_AVX2
  using V = vec<T>;
  constexpr size_t N = NumLanes<V>();

  const V counts = Iota<V>();

  const V bits = set1(V(), T(1)) << counts;
  SIMD_ALIGN T lanes[N];
  store(bits, lanes);
  for (size_t i = 0; i < N; ++i) {
    ASSERT_EQ(T(1ull << i), lanes[i]);
  }

  store(bits >> counts, lanes);
  for (size_t i = 0; i < N; ++i) {
    ASSERT_EQ(T(1), lanes[i]);
  }
#endif  // SIMD_ENABLE_AVX2
}

void TestVariableShifts() {
  // No u8/u16.
  TestVariableShiftsT<uint32_t>();
  TestVariableShiftsT<uint64_t>();
  TestVariableShiftsT<int32_t>();
  // No i8/i16/i64/f32/f64.
}

template <typename T>
void TestUnsignedMinMax() {
  using V = vec<T>;
  const size_t N = NumLanes<V>();
  const V v0 = setzero(V());
  const V v1 = Iota<V>(1);
  const V v2 = Iota<V>(2);
  const V v_max = Iota<V>(UnsignedMax<T>() - N + 1);
  ASSERT_VEC_EQ(v1, min(v1, v2));
  ASSERT_VEC_EQ(v2, max(v1, v2));
  ASSERT_VEC_EQ(v0, min(v1, v0));
  ASSERT_VEC_EQ(v1, max(v1, v0));
  ASSERT_VEC_EQ(v1, min(v1, v_max));
  ASSERT_VEC_EQ(v_max, max(v1, v_max));
  ASSERT_VEC_EQ(v0, min(v0, v_max));
  ASSERT_VEC_EQ(v_max, max(v0, v_max));
}

template <typename T>
void TestSignedMinMax() {
  using V = vec<T>;
  const size_t N = NumLanes<V>();
  const V v1 = Iota<V>(1);
  const V v2 = Iota<V>(2);
  const V v_neg = Iota<V>(-T(N));
  const V v_neg_max = Iota<V>(std::numeric_limits<T>::lowest());
  ASSERT_VEC_EQ(v1, min(v1, v2));
  ASSERT_VEC_EQ(v2, max(v1, v2));
  ASSERT_VEC_EQ(v_neg, min(v1, v_neg));
  ASSERT_VEC_EQ(v1, max(v1, v_neg));
  ASSERT_VEC_EQ(v_neg_max, min(v1, v_neg_max));
  ASSERT_VEC_EQ(v1, max(v1, v_neg_max));
  ASSERT_VEC_EQ(v_neg_max, min(v_neg, v_neg_max));
  ASSERT_VEC_EQ(v_neg, max(v_neg, v_neg_max));
}

void TestMinMax() {
  TestUnsignedMinMax<uint8_t>();
  TestUnsignedMinMax<uint16_t>();
  TestUnsignedMinMax<uint32_t>();
  // No u64.
  TestSignedMinMax<int8_t>();
  TestSignedMinMax<int16_t>();
  TestSignedMinMax<int32_t>();
  // No i64.
  TestSignedMinMax<float>();
  TestSignedMinMax<double>();
}

template <typename T>
void TestUnsignedMul() {
  using V = vec<T>;
  const size_t N = NumLanes<V>();
  const V v0 = setzero(V());
  const V v1 = set1(V(), T(1));
  const V vi = Iota<V>(1);
  const V vj = Iota<V>(3);
  T lanes[N];
  ASSERT_VEC_EQ(v0, v0 * v0);
  ASSERT_VEC_EQ(v1, v1 * v1);
  ASSERT_VEC_EQ(vi, v1 * vi);
  ASSERT_VEC_EQ(vi, vi * v1);

  for (size_t i = 0; i < N; ++i) {
    lanes[i] = (1 + i) * (1 + i);
  }
  ASSERT_VEC_EQ(lanes, vi * vi);

  for (size_t i = 0; i < N; ++i) {
    lanes[i] = (1 + i) * (3 + i);
  }
  ASSERT_VEC_EQ(lanes, vi * vj);

  const T max = UnsignedMax<T>();
  const V vmax = set1(V(), max);
  ASSERT_VEC_EQ(vmax, vmax * v1);
  ASSERT_VEC_EQ(vmax, v1 * vmax);

  const size_t bits = sizeof(T) * 8;
  const uint64_t mask = (1ull << bits) - 1;
  const T max2 = (uint64_t(max) * max) & mask;
  ASSERT_VEC_EQ(set1(V(), max2), vmax * vmax);
}

template <typename T>
void TestSignedMul() {
  using V = vec<T>;
  const size_t N = NumLanes<V>();
  const V v0 = setzero(V());
  const V v1 = set1(V(), T(1));
  const V vi = Iota<V>(1);
  const V vn = Iota<V>(-T(N));
  T lanes[N];
  ASSERT_VEC_EQ(v0, v0 * v0);
  ASSERT_VEC_EQ(v1, v1 * v1);
  ASSERT_VEC_EQ(vi, v1 * vi);
  ASSERT_VEC_EQ(vi, vi * v1);

  for (size_t i = 0; i < N; ++i) {
    lanes[i] = (1 + i) * (1 + i);
  }
  ASSERT_VEC_EQ(lanes, vi * vi);

  for (size_t i = 0; i < N; ++i) {
    lanes[i] = (-T(N) + i) * (1 + i);
  }
  ASSERT_VEC_EQ(lanes, vn * vi);
  ASSERT_VEC_EQ(lanes, vi * vn);
}

void TestMul() {
  // No u8.
  TestUnsignedMul<uint16_t>();
  TestUnsignedMul<uint32_t>();
  // No u64,i8.
  TestSignedMul<int16_t>();
  TestSignedMul<int32_t>();
  // No i64.
}

void TestMulHi16() {
  using T = int16_t;
  using V = vec<T>;
  constexpr size_t N = NumLanes<V>();
  SIMD_ALIGN T in_lanes[N];
  SIMD_ALIGN T expected_lanes[N];
  const auto vi = Iota<V>(1);
  const auto vni = Iota<V>(-T(N));
  V v;

  const V v0 = setzero(V());
  ASSERT_VEC_EQ(v0, ext::mulhi(v0, v0));
  ASSERT_VEC_EQ(v0, ext::mulhi(v0, vi));
  ASSERT_VEC_EQ(v0, ext::mulhi(vi, v0));

  // Large positive squared
  for (size_t i = 0; i < N; ++i) {
    in_lanes[i] = SignedMax<T>() >> i;
    expected_lanes[i] = (int32_t(in_lanes[i]) * in_lanes[i]) >> 16;
  }
  v = load(V(), in_lanes);
  ASSERT_VEC_EQ(expected_lanes, ext::mulhi(v, v));

  // Large positive * small positive
  for (size_t i = 0; i < N; ++i) {
    expected_lanes[i] = (int32_t(in_lanes[i]) * (1 + i)) >> 16;
  }
  ASSERT_VEC_EQ(expected_lanes, ext::mulhi(v, vi));
  ASSERT_VEC_EQ(expected_lanes, ext::mulhi(vi, v));

  // Large positive * small negative
  for (size_t i = 0; i < N; ++i) {
    expected_lanes[i] = (int32_t(in_lanes[i]) * (i - N)) >> 16;
  }
  ASSERT_VEC_EQ(expected_lanes, ext::mulhi(v, vni));
  ASSERT_VEC_EQ(expected_lanes, ext::mulhi(vni, v));
}

template <typename T, typename T2>
void TestMulEvenT() {
  using V = vec<T>;
  using V2 = vec<T2>;  // wider type, half the lanes
  constexpr size_t N = NumLanes<V>();
  constexpr size_t N2 = NumLanes<V2>();

  const V v0 = setzero(V());
  ASSERT_VEC_EQ(V2(v0), mul_even(v0, v0));

  // vec1 has N=1 and we write to "lane 1" below, though it isn't used by
  // the actual mul_even.
  SIMD_ALIGN T in_lanes[SIMD_MAX(N, 2)];
  SIMD_ALIGN T2 expected[N2];
  for (size_t i = 0; i < N; i += 2) {
    in_lanes[i + 0] = std::numeric_limits<T>::max() >> i;
    in_lanes[i + 1] = 1;  // will be overwritten with upper half of result
    expected[i / 2] = T2(in_lanes[i + 0]) * in_lanes[i + 0];
  }

  const auto v = load(V(), in_lanes);
  ASSERT_VEC_EQ(expected, mul_even(v, v));
}

void TestMulEven() {
  TestMulEvenT<int32_t, int64_t>();
  TestMulEvenT<uint32_t, uint64_t>();
}

template <typename T, class V>
struct TestMulAdd {
  void operator()() const {
    const size_t N = NumLanes<V>();
    const V v0 = setzero(V());
    const V v1 = Iota<V>(1);
    const V v2 = Iota<V>(2);
    T lanes[N];
    ASSERT_VEC_EQ(v0, mul_add(v0, v0, v0));
    ASSERT_VEC_EQ(v2, mul_add(v0, v1, v2));
    ASSERT_VEC_EQ(v2, mul_add(v1, v0, v2));
    for (size_t i = 0; i < N; ++i) {
      lanes[i] = (i + 1) * (i + 2);
    }
    ASSERT_VEC_EQ(lanes, mul_add(v2, v1, v0));
    ASSERT_VEC_EQ(lanes, mul_add(v1, v2, v0));
    for (size_t i = 0; i < N; ++i) {
      lanes[i] = (i + 2) * (i + 2) + (i + 1);
    }
    ASSERT_VEC_EQ(lanes, mul_add(v2, v2, v1));

    ASSERT_VEC_EQ(v0, mul_sub(v0, v0, v0));
    for (size_t i = 0; i < N; ++i) {
      lanes[i] = -T(i + 2);
    }
    ASSERT_VEC_EQ(lanes, mul_sub(v0, v1, v2));
    ASSERT_VEC_EQ(lanes, mul_sub(v1, v0, v2));
    for (size_t i = 0; i < N; ++i) {
      lanes[i] = (i + 1) * (i + 2);
    }
    ASSERT_VEC_EQ(lanes, mul_sub(v1, v2, v0));
    ASSERT_VEC_EQ(lanes, mul_sub(v2, v1, v0));
    for (size_t i = 0; i < N; ++i) {
      lanes[i] = (i + 2) * (i + 2) - (1 + i);
    }
    ASSERT_VEC_EQ(lanes, mul_sub(v2, v2, v1));

    ASSERT_VEC_EQ(v0, nmul_add(v0, v0, v0));
    ASSERT_VEC_EQ(v2, nmul_add(v0, v1, v2));
    ASSERT_VEC_EQ(v2, nmul_add(v1, v0, v2));
    for (size_t i = 0; i < N; ++i) {
      lanes[i] = -T(i + 1) * (i + 2);
    }
    ASSERT_VEC_EQ(lanes, nmul_add(v1, v2, v0));
    ASSERT_VEC_EQ(lanes, nmul_add(v2, v1, v0));
    for (size_t i = 0; i < N; ++i) {
      lanes[i] = -T(i + 2) * (i + 2) + (1 + i);
    }
    ASSERT_VEC_EQ(lanes, nmul_add(v2, v2, v1));
  }
};

template <typename T, class V>
struct TestSquareRoot {
  void operator()() const {
    const V vi = Iota<V>();
    ASSERT_VEC_EQ(vi, sqrt(vi * vi));
  }
};

void TestReciprocalSquareRoot() {
  using V = vec<float>;
  constexpr size_t N = NumLanes<V>();
  const auto v = set1(V(), 123.0f);
  SIMD_ALIGN float lanes[N];
  store(rsqrt_approx(v), lanes);
  for (size_t i = 0; i < N; ++i) {
    float err = lanes[i] - 0.090166f;
    if (err < 0.0f) err = -err;
    ASSERT_EQ(true, err < 1E-4f);
  }
}

template <typename T, class V>
struct TestRound {
  void operator()() const {
    // Integer positive
    {
      const V v = Iota<V>(4.0);
      ASSERT_VEC_EQ(v, round_pos_inf(v));
      ASSERT_VEC_EQ(v, round_neg_inf(v));
    }

    // Integer negative
    {
      const V v = Iota<V>(T(-32.0));
      ASSERT_VEC_EQ(v, round_pos_inf(v));
      ASSERT_VEC_EQ(v, round_neg_inf(v));
    }

    // Huge positive
    {
      const V v = set1(V(), T(1E30));
      ASSERT_VEC_EQ(v, round_pos_inf(v));
      ASSERT_VEC_EQ(v, round_neg_inf(v));
    }

    // Huge negative
    {
      const V v = set1(V(), T(-1E31));
      ASSERT_VEC_EQ(v, round_pos_inf(v));
      ASSERT_VEC_EQ(v, round_neg_inf(v));
    }

    // Above positive
    {
      const V v = Iota<V>(T(2.0001));
      const V v3 = Iota<V>(T(3));
      const V v2 = Iota<V>(T(2));
      ASSERT_VEC_EQ(v3, round_pos_inf(v));
      ASSERT_VEC_EQ(v2, round_neg_inf(v));
    }

    // Below positive
    {
      const V v = Iota<V>(T(3.9999));
      const V v4 = Iota<V>(T(4));
      const V v3 = Iota<V>(T(3));
      ASSERT_VEC_EQ(v4, round_pos_inf(v));
      ASSERT_VEC_EQ(v3, round_neg_inf(v));
    }

    // Above negative
    {
      const V v = Iota<V>(T(-3.9999));
      const V v3 = Iota<V>(T(-3));
      const V v4 = Iota<V>(T(-4));
      ASSERT_VEC_EQ(v3, round_pos_inf(v));
      ASSERT_VEC_EQ(v4, round_neg_inf(v));
    }

    // Below negative
    {
      const V v = Iota<V>(T(-2.0001));
      const V v2 = Iota<V>(T(-2));
      const V v3 = Iota<V>(T(-3));
      ASSERT_VEC_EQ(v2, round_pos_inf(v));
      ASSERT_VEC_EQ(v3, round_neg_inf(v));
    }
  }
};

void TestHorzSum8() {
  using V = vec<uint8_t>;
  const size_t N = NumLanes<V>();
  SIMD_ALIGN uint8_t in_lanes[N];
  uint64_t sums[(N + 7) / 8] = {0};
  for (size_t i = 0; i < N; ++i) {
    const size_t group = i / 8;
    in_lanes[i] = 2 * i + 1;
    sums[group] += in_lanes[i];
  }
  const V v = load(V(), in_lanes);
  ASSERT_VEC_EQ(sums, ext::horz_sum(v));
}

template <typename T>
void TestHorzSumT() {
  using V = vec<T>;
  constexpr size_t N = NumLanes<V>();
  SIMD_ALIGN T in_lanes[N];
  double sum = 0.0;
  for (size_t i = 0; i < N; ++i) {
    in_lanes[i] = 1u << i;
    sum += in_lanes[i];
  }
  const V v = load(V(), in_lanes);
  using SumV = decltype(ext::horz_sum(v));
  const auto expected = set1(SumV(), T(sum));
  ASSERT_VEC_EQ(expected, ext::horz_sum(v));
}

void TestHorzSum() {
  TestHorzSum8();
  // No u16.
  TestHorzSumT<uint32_t>();
  TestHorzSumT<uint64_t>();

  // No i8/i16.
  TestHorzSumT<int32_t>();
  TestHorzSumT<int64_t>();

  TestHorzSumT<float>();
  TestHorzSumT<double>();
}

void TestArithmetic() {
  ForeachLaneType<TestPlusMinus>();
  TestSaturatingArithmetic();

  TestShifts();
  TestVariableShifts();
  TestMinMax();
  TestAverage();
  TestMul();
  TestMulHi16();
  TestMulEven();

  ForeachFloatLaneType<TestMulAdd>();
  ForeachFloatLaneType<TestSquareRoot>();
  TestReciprocalSquareRoot();
  ForeachFloatLaneType<TestRound>();

  TestHorzSum();
}

}  // namespace arithmetic

namespace compare {

template <typename T, class V>
struct TestSignedCompare {
  void operator()() const {
    const size_t N = NumLanes<V>();
    const V v2 = Iota<V>(2);
    const V v2b = Iota<V>(2);
    const V vn = Iota<V>(-T(N));
    const V yes = set1(V(), static_cast<T>(-1));
    const V no = setzero(V());

    ASSERT_VEC_EQ(no, v2 == vn);
    ASSERT_VEC_EQ(yes, v2 == v2b);

    ASSERT_VEC_EQ(yes, v2 > vn);
    ASSERT_VEC_EQ(yes, vn < v2);
    ASSERT_VEC_EQ(no, v2 < vn);
    ASSERT_VEC_EQ(no, vn > v2);
  }
};

template <typename T, class V>
struct TestUnsignedCompare {
  void operator()() const {
    const V v2 = Iota<V>(2);
    const V v2b = Iota<V>(2);
    const V v3 = Iota<V>(3);
    const V yes = set1(V(), T(~0ull));
    const V no = setzero(V());

    ASSERT_VEC_EQ(no, v2 == v3);
    ASSERT_VEC_EQ(yes, v2 == v2b);
  }
};

template <typename T, class V>
struct TestFloatCompare {
  void operator()() const {
    const size_t N = NumLanes<V>();
    const V v2 = Iota<V>(2);
    const V v2b = Iota<V>(2);
    const V vn = Iota<V>(-T(N));
    const V no = setzero(V());

    ASSERT_VEC_EQ(no, v2 == vn);
    ASSERT_VEC_EQ(no, v2 < vn);
    ASSERT_VEC_EQ(no, vn > v2);

    // Equality is represented as 1-bits, which is a NaN, so compare bytes.
    uint8_t yes[sizeof(V)];
    SetBytes(0xFF, &yes);

    SIMD_ALIGN T lanes[NumLanes<V>()];
    store(v2 == v2b, lanes);
    ASSERT_EQ(true, BytesEqual(lanes, yes, sizeof(V)));
    store(v2 > vn, lanes);
    ASSERT_EQ(true, BytesEqual(lanes, yes, sizeof(V)));
    store(vn < v2, lanes);
    ASSERT_EQ(true, BytesEqual(lanes, yes, sizeof(V)));
  }
};

// Returns "bits" after zeroing any upper bits that wouldn't be returned by
// movemask for the given vector "V".
template <class V>
uint32_t ValidBits(const uint32_t bits) {
  const uint64_t mask = (1ull << NumLanes<V>()) - 1;
  return bits & mask;
}

void TestMovemask() {
  using V = vec<uint8_t>;
  SIMD_ALIGN const uint8_t bytes[32] = {
      0x80, 0xFF, 0x7F, 0x00, 0x01, 0x10,      0x20, 0x40, 0x80, 0x02, 0x04,
      0x08, 0xC0, 0xC1, 0xFE, 0x0F, /**/ 0x0F, 0xFE, 0xC1, 0xC0, 0x08, 0x04,
      0x02, 0x80, 0x40, 0x20, 0x10, 0x01,      0x00, 0x7F, 0xFF, 0x80};
  ASSERT_EQ(ValidBits<V>(0xC08E7103), ext::movemask(load(V(), bytes)));

  SIMD_ALIGN const float lanes[8] = {-1.0f,  1E30f, -0.0f, 1E-30f,
                                     1E-30f, -0.0f, 1E30f, -1.0f};
  using VF = vec<float>;
  ASSERT_EQ(ValidBits<VF>(0xa5), ext::movemask(load(VF(), lanes)));

  using VD = vec<double>;
  SIMD_ALIGN const double lanes2[4] = {1E300, -1E-300, -0.0, 1E-10};
  ASSERT_EQ(ValidBits<VD>(6), ext::movemask(load(VD(), lanes2)));
}

template <typename T, class V>
struct TestAllZero {
  void operator()() const {
    constexpr size_t N = NumLanes<V>();

    const T max = std::numeric_limits<T>::max();
    const T min_nonzero = std::numeric_limits<T>::lowest() + 1;

    // all lanes zero
    V v = set1(V(), T(0));
    SIMD_ALIGN T lanes[N];
    store(v, lanes);

    // Set each lane to nonzero and ensure !all_zero
    for (size_t i = 0; i < N; ++i) {
      lanes[i] = max;
      v = load(V(), lanes);
      ASSERT_EQ(false, ext::all_zero(v));

      lanes[i] = min_nonzero;
      v = load(V(), lanes);
      ASSERT_EQ(false, ext::all_zero(v));

      // Reset to all zero
      lanes[i] = T(0);
      v = load(V(), lanes);
      ASSERT_EQ(true, ext::all_zero(v));
    }
  }
};

void TestCompare() {
  ForeachSignedLaneType<TestSignedCompare>();
  ForeachUnsignedLaneType<TestUnsignedCompare>();
  ForeachFloatLaneType<TestFloatCompare>();

  TestMovemask();

  ForeachUnsignedLaneType<TestAllZero>();
  ForeachSignedLaneType<TestAllZero>();
  // No float.
}

}  // namespace compare

namespace logical {

template <typename T, class V>
struct TestLogicalT {
  void operator()() const {
    const V v0 = setzero(V());
    const V vi = Iota<V>();

    ASSERT_VEC_EQ(v0, v0 & vi);
    ASSERT_VEC_EQ(v0, vi & v0);
    ASSERT_VEC_EQ(vi, vi & vi);

    ASSERT_VEC_EQ(vi, v0 | vi);
    ASSERT_VEC_EQ(vi, vi | v0);
    ASSERT_VEC_EQ(vi, vi | vi);

    ASSERT_VEC_EQ(vi, v0 ^ vi);
    ASSERT_VEC_EQ(vi, vi ^ v0);
    ASSERT_VEC_EQ(v0, vi ^ vi);

    ASSERT_VEC_EQ(vi, andnot(v0, vi));
    ASSERT_VEC_EQ(v0, andnot(vi, v0));
    ASSERT_VEC_EQ(v0, andnot(vi, vi));

    V v = vi;
    v &= vi;
    ASSERT_VEC_EQ(vi, v);
    v &= v0;
    ASSERT_VEC_EQ(v0, v);

    v |= vi;
    ASSERT_VEC_EQ(vi, v);
    v |= v0;
    ASSERT_VEC_EQ(vi, v);

    v ^= vi;
    ASSERT_VEC_EQ(v0, v);
    v ^= v0;
    ASSERT_VEC_EQ(v0, v);
  }
};

void TestLogical() { ForeachLaneType<TestLogicalT>(); }

}  // namespace logical

namespace memory {

template <typename T, class V>
struct TestLoadStore {
  void operator()() const {
    constexpr size_t N = NumLanes<V>();
    const V hi = Iota<V>(1 + N);
    const V lo = Iota<V>(1);
    SIMD_ALIGN T lanes[2 * N];
    store(hi, lanes + N);
    store(lo, lanes);

    // Aligned load
    const V lo2 = load(V(), lanes);
    ASSERT_VEC_EQ(lo2, lo);
    // First value goes into least-significant/lowest lane
    ASSERT_EQ(T(1), get_low(lo2));

    // Aligned store
    SIMD_ALIGN T lanes2[2 * N];
    store(lo2, lanes2);
    store(hi, lanes2 + N);
    for (size_t i = 0; i < 2 * N; ++i) {
      ASSERT_EQ(lanes[i], lanes2[i]);
    }

    // Unaligned load
    const V vu = load_unaligned(V(), lanes + 1);
    SIMD_ALIGN T lanes3[N];
    store(vu, lanes3);
    for (size_t i = 0; i < N; ++i) {
      ASSERT_EQ(T(i + 2), lanes3[i]);
    }

    // Unaligned store
    store_unaligned(lo2, lanes2 + N / 2);
    size_t i = 0;
    for (; i < N / 2; ++i) {
      ASSERT_EQ(lanes[i], lanes2[i]);
    }
    for (; i < 3 * N / 2; ++i) {
      ASSERT_EQ(T(i - N / 2 + 1), lanes2[i]);
    }
    // Subsequent values remain unchanged.
    for (; i < 2 * N; ++i) {
      ASSERT_EQ(T(i + 1), lanes2[i]);
    }
  }
};

template <typename T, class V>
struct TestLoadDup128 {
  void operator()() const {
#if SIMD_ENABLE_ANY
    constexpr size_t N = NumLanes<V>();
    const size_t num_lanes_128 = 16 / sizeof(T);
    alignas(16) T lanes[num_lanes_128];
    for (size_t i = 0; i < num_lanes_128; ++i) {
      lanes[i] = 1 + i;
    }
    const V v = load_dup128(V(), lanes);
    SIMD_ALIGN T out[N];
    store(v, out);
    for (size_t i = 0; i < N; ++i) {
      ASSERT_EQ(T(i % num_lanes_128 + 1), out[i]);
    }
#endif
  }
};

template <typename T>
void TestStreamT() {
  using V = vec<T>;
  constexpr size_t N = NumLanes<V>();
  const V v = Iota<V>();
  SIMD_ALIGN T out[N];
  stream(v, out);
  store_fence();
  for (size_t i = 0; i < N; ++i) {
    ASSERT_EQ(T(i), out[i]);
  }
}

void TestStream() {
  // No u8,u16.
  TestStreamT<uint32_t>();
  TestStreamT<uint64_t>();
  // No i8,i16.
  TestStreamT<int32_t>();
  TestStreamT<int64_t>();
  TestStreamT<float>();
  TestStreamT<double>();
}

void TestMemory() {
  ForeachLaneType<TestLoadStore>();
  ForeachLaneType<TestLoadDup128>();
  TestStream();
}

}  // namespace memory

namespace swizzle {

template <typename T, class V>
struct TestShiftBytesT {
  void operator()() const {
    using V8 = vec<uint8_t>;
    constexpr size_t N = NumLanes<V>();

    // Zero remains zero
    const V v0 = setzero(V());
    ASSERT_VEC_EQ(v0, shift_bytes_left<0>(v0));
    ASSERT_VEC_EQ(v0, shift_bytes_left<1>(v0));
    ASSERT_VEC_EQ(v0, shift_bytes_right<0>(v0));
    ASSERT_VEC_EQ(v0, shift_bytes_right<1>(v0));

    // Zero after shifting out the high/low byte
    SIMD_ALIGN uint8_t bytes[sizeof(V)] = {0};
    bytes[sizeof(V) - 1] = 0x7F;
    const V vhi(load(V8(), bytes));
    bytes[sizeof(V) - 1] = 0;
    bytes[0] = 0x7F;
    const V vlo(load(V8(), bytes));
    ASSERT_EQ(true, ext::all_zero(shift_bytes_left<1>(vhi)));
    ASSERT_EQ(true, ext::all_zero(shift_bytes_left<sizeof(V)>(vlo)));
    ASSERT_EQ(true, ext::all_zero(shift_bytes_right<1>(vlo)));
    ASSERT_EQ(true, ext::all_zero(shift_bytes_right<sizeof(V)>(vhi)));

    // Shifting by zero has no effect
    ASSERT_VEC_EQ(vlo, shift_bytes_left<0>(vlo));
    ASSERT_VEC_EQ(vlo, shift_bytes_right<0>(vlo));
    ASSERT_VEC_EQ(vhi, shift_bytes_left<0>(vhi));
    ASSERT_VEC_EQ(vhi, shift_bytes_right<0>(vhi));

    SIMD_ALIGN T in[N];
    const uint8_t* in_bytes = reinterpret_cast<const uint8_t*>(in);
    const V v(Iota<V8>(1));
    store(v, in);

    SIMD_ALIGN T shifted[N];
    const uint8_t* shifted_bytes = reinterpret_cast<const uint8_t*>(shifted);

    const size_t kBlockSize = SIMD_MIN(sizeof(V), 16);
    store(shift_bytes_left<1>(v), shifted);
    for (size_t block = 0; block < sizeof(V); block += kBlockSize) {
      ASSERT_EQ(uint8_t(0), shifted_bytes[block]);
      ASSERT_EQ(true, BytesEqual(in_bytes + block, shifted_bytes + block + 1,
                                 kBlockSize - 1));
    }

    store(shift_bytes_right<1>(v), shifted);
    for (size_t block = 0; block < sizeof(V); block += kBlockSize) {
      ASSERT_EQ(uint8_t(0), shifted_bytes[block + kBlockSize - 1]);
      ASSERT_EQ(true, BytesEqual(in_bytes + block + 1, shifted_bytes + block,
                                 kBlockSize - 1));
    }
  }
};

void TestShiftBytes() {
  ForeachUnsignedLaneType<TestShiftBytesT>();
  ForeachSignedLaneType<TestShiftBytesT>();
  // No float.
}

template <typename T, class V>
struct TestGetSet {
  void operator()() const {
    constexpr size_t N = NumLanes<V>();
    SIMD_ALIGN T lanes[N];

    const V v0 = set_low(V(), T(0));
    ASSERT_EQ(T(0), get_low(v0));
    store(v0, lanes);
    for (size_t i = 1; i < N; ++i) {
      ASSERT_EQ(T(0), lanes[i]);
    }

    V v1 = set_low(V(), T(1));
    ASSERT_EQ(T(1), get_low(v1));
    store(v1, lanes);
    for (size_t i = 1; i < N; ++i) {
      ASSERT_EQ(T(0), lanes[i]);
    }

    v1 = set_low(V(), T(2));
    ASSERT_EQ(T(2), get_low(v1));
  }
};

template <typename T, int kLane>
struct TestBroadcastR {
  void operator()() const {
    using V = vec<T>;
    constexpr size_t N = NumLanes<V>();
    SIMD_ALIGN T in_lanes[N] = {0};
    const size_t kBlockT = SIMD_MIN(sizeof(V), 16) / sizeof(T);
    // Need to set within each 128-bit block
    for (size_t block = 0; block < N; block += kBlockT) {
      in_lanes[block + kLane] = block + 1;
    }
    const V in = load(V(), in_lanes);
    SIMD_ALIGN T out_lanes[N];
    store(broadcast<kLane>(in), out_lanes);
    for (size_t block = 0; block < N; block += kBlockT) {
      for (size_t i = 0; i < kBlockT; ++i) {
        ASSERT_EQ(T(block + 1), out_lanes[block + i]);
      }
    }

    TestBroadcastR<T, kLane - 1>()();
  }
};

template <typename T>
struct TestBroadcastR<T, -1> {
  void operator()() const {}
};

template <typename T>
void TestBroadcastT() {
  // Lane index cannot exceed those for 128-bit but scalar.h lacks vec128, so
  // hard-code 16 directly. For scalar.h, vec = vec1 => N=1.
  TestBroadcastR<T, SIMD_MIN(NumLanes<vec<T>>(), 16 / sizeof(T)) - 1>()();
}

void TestBroadcast() {
  // No u8,u16.
  TestBroadcastT<uint32_t>();
  TestBroadcastT<uint64_t>();
  // No i8,i16.
  TestBroadcastT<int32_t>();
  TestBroadcastT<int64_t>();
  TestBroadcastT<float>();
  TestBroadcastT<double>();
}

template <typename T, class V>
struct TestZip {
  void operator()() const {
// Not supported by scalar.h: zip(f32, f32) would need to return f32x2.
#if SIMD_ENABLE_ANY
    constexpr size_t N = NumLanes<V>();
    SIMD_ALIGN T even_lanes[N];
    SIMD_ALIGN T odd_lanes[N];
    for (size_t i = 0; i < N; ++i) {
      even_lanes[i] = 2 * i + 0;
      odd_lanes[i] = 2 * i + 1;
    }
    const V even = load(V(), even_lanes);
    const V odd = load(V(), odd_lanes);

    SIMD_ALIGN T lo_lanes[N];
    SIMD_ALIGN T hi_lanes[N];
    store(zip_lo(even, odd), lo_lanes);
    store(zip_hi(even, odd), hi_lanes);

    const size_t kBlockT = 16 / sizeof(T);
    for (size_t i = 0; i < N; ++i) {
      const size_t block = i / kBlockT;
      const size_t lo = (i % kBlockT) + block * 2 * kBlockT;
      ASSERT_EQ(T(lo), lo_lanes[i]);
      ASSERT_EQ(T(lo + kBlockT), hi_lanes[i]);
    }
#endif
  }
};

template <typename T, class V>
struct TestShuffleT {
  void operator()() const {
// Not supported by scalar.h (its vector size is always less than 16 bytes)
#if SIMD_ENABLE_ANY
    RandomState rng = {1234};
    SIMD_ALIGN uint8_t in_bytes[sizeof(V)];
    for (size_t i = 0; i < sizeof(V); ++i) {
      in_bytes[i] = Random32(&rng) & 0xFF;
    }
    using V8 = vec<uint8_t>;
    const V in(load(V8(), in_bytes));
    SIMD_ALIGN const uint8_t index_bytes[32] = {
        // Same index as source, multiple outputs from same input,
        // unused input (9), ascending/descending and nonconsecutive neighbors.
        0,  2,  1, 2, 15, 12, 13, 14, 6,  7,  8,  5,  4, 3, 10, 11,
        11, 10, 3, 4, 5,  8,  7,  6,  14, 13, 12, 15, 2, 1, 2,  0};
    const V indices(load(V8(), index_bytes));
    SIMD_ALIGN T out_lanes[NumLanes<V>()];
    store(shuffle_bytes(in, indices), out_lanes);
    const uint8_t* out_bytes = reinterpret_cast<const uint8_t*>(out_lanes);

    for (size_t block = 0; block < sizeof(V); block += 16) {
      for (size_t i = 0; i < 16; ++i) {
        const uint8_t expected = in_bytes[block + index_bytes[block + i]];
        ASSERT_EQ(expected, out_bytes[block + i]);
      }
    }
#endif
  }
};

void TestShuffle() {
  ForeachUnsignedLaneType<TestShuffleT>();
  ForeachSignedLaneType<TestShuffleT>();
  // No float.
}

template <typename T, class V, int kBytes>
struct TestExtractR {
  void operator()() const {
    constexpr size_t N = NumLanes<V>();
    // (Cannot use Iota<> because sizeof(u8x1) != sizeof(u8x2))
    SIMD_ALIGN T in_lanes[2 * N];
    uint8_t* in_bytes = reinterpret_cast<uint8_t*>(in_lanes);
    for (size_t i = 0; i < 2 * sizeof(V); ++i) {
      in_bytes[i] = 1 + i;
    }

    const V lo = load(V(), in_lanes);
    const V hi = load(V(), in_lanes + N);

    SIMD_ALIGN T lanes[N];
    store(extract_concat_bytes<kBytes>(hi, lo), lanes);
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(lanes);

    const size_t kBlockSize = SIMD_MIN(sizeof(V), 16);
    for (size_t i = 0; i < sizeof(V); ++i) {
      const size_t block = i / kBlockSize;
      const size_t lane = i % kBlockSize;
      const size_t first_lo = block * kBlockSize;
      const size_t idx = lane + kBytes;
      const size_t offset = (idx < kBlockSize) ? 0 : sizeof(V) - kBlockSize;
      const bool at_end = idx >= 2 * kBlockSize;
      const uint8_t expected = at_end ? 0 : (first_lo + idx + 1 + offset);
      ASSERT_EQ(expected, bytes[i]);
    }

    TestExtractR<T, V, kBytes - 1>()();
  }
};

template <typename T, class V>
struct TestExtractR<T, V, -1> {
  void operator()() const {}
};

template <typename T, class V>
struct TestExtractT {
  void operator()() const { TestExtractR<T, V, sizeof(vec<T>)>()(); }
};

void TestExtract() {
  ForeachUnsignedLaneType<TestExtractT>();
  ForeachSignedLaneType<TestExtractT>();
  // No float.
}

#if SIMD_ENABLE_ANY

template <class V>
void VerifyLanes32(const V v, const int i3, const int i2, const int i1,
                   const int i0) {
  using T = Lane<V>;
  constexpr size_t N = NumLanes<V>();
  SIMD_ALIGN T lanes[N];
  store(v, lanes);
  const size_t kBlockT = 16 / sizeof(T);
  for (size_t block = 0; block < N; block += kBlockT) {
    ASSERT_EQ(T(block + i3), lanes[block + 3]);
    ASSERT_EQ(T(block + i2), lanes[block + 2]);
    ASSERT_EQ(T(block + i1), lanes[block + 1]);
    ASSERT_EQ(T(block + i0), lanes[block + 0]);
  }
}

template <class V>
void VerifyLanes64(const V v, const int i1, const int i0) {
  constexpr size_t N = NumLanes<V>();
  using T = Lane<V>;
  SIMD_ALIGN T lanes[N];
  store(v, lanes);
  const size_t kBlockT = 16 / sizeof(T);
  for (size_t block = 0; block < N; block += kBlockT) {
    ASSERT_EQ(T(block + i1), lanes[block + 1]);
    ASSERT_EQ(T(block + i0), lanes[block + 0]);
  }
}

template <typename T>
void TestSpecialShuffle32() {
  using V = vec<T>;
  const V v = Iota<V>();

  VerifyLanes32(shuffle_1032(v), 1, 0, 3, 2);
  VerifyLanes32(shuffle_0321(v), 0, 3, 2, 1);
  VerifyLanes32(shuffle_2103(v), 2, 1, 0, 3);
}

template <typename T>
void TestSpecialShuffle64() {
  using V = vec<T>;
  const V v = Iota<V>();
  VerifyLanes64(shuffle_01(v), 0, 1);
}

#endif

void TestSpecialShuffles() {
#if SIMD_ENABLE_ANY
  TestSpecialShuffle32<int32_t>();
  TestSpecialShuffle64<int64_t>();
  TestSpecialShuffle32<float>();
  TestSpecialShuffle64<double>();
#endif
}

template <typename T, class V>
struct TestSelect {
  void operator()() const {
    constexpr size_t N = NumLanes<V>();
    RandomState rng = {1234};
    const T mask0(0);
    const uint64_t ones = ~0ull;
    T mask1;
    CopyBytes(ones, &mask1);

    SIMD_ALIGN T lanes1[N];
    SIMD_ALIGN T lanes2[N];
    SIMD_ALIGN T masks[N];
    for (size_t i = 0; i < N; ++i) {
      lanes1[i] = int32_t(Random32(&rng));
      lanes2[i] = int32_t(Random32(&rng));
      masks[i] = (Random32(&rng) & 1024) ? mask0 : mask1;
    }

    SIMD_ALIGN T out_lanes[sizeof(V)];
    store(select(load(V(), lanes1), load(V(), lanes2), load(V(), masks)),
          out_lanes);
    for (size_t i = 0; i < N; ++i) {
      ASSERT_EQ((masks[i] == mask0) ? lanes1[i] : lanes2[i], out_lanes[i]);
    }
  }
};

// Basic test: iota values remain unchanged. Called by Test*Promote.
template <typename T>
void TestPromote() {
  using HalfV = Half<vec<T>>;
  const HalfV v = Iota<HalfV>();
  const auto wide = promote(v);
  using WideV = decltype(wide);
  const auto expected_wide = Iota<WideV>();
  ASSERT_VEC_EQ(expected_wide, wide);
}

template <typename T>
void TestUnsignedPromote() {
  TestPromote<T>();
  using HalfV = Half<vec<T>>;
  const HalfV v = Iota<HalfV>(1);
  using WideV = decltype(promote(v));
  using WideT = Lane<WideV>;

  // Test extreme value: the maximum.
  const HalfV vpm = set1(HalfV(), UnsignedMax<T>());
  const WideV wpm = set1(WideV(), WideT(UnsignedMax<T>()));
  ASSERT_VEC_EQ(wpm, promote(vpm));
}

template <typename T>
void TestSignedPromote() {
  TestPromote<T>();
  using HalfV = Half<vec<T>>;
  const HalfV v = Iota<HalfV>();
  using WideV = decltype(promote(v));
  using WideT = Lane<WideV>;

  // Test other values: -1, max, min.
  const HalfV vn1 = set1(HalfV(), T(-1));
  const HalfV vpm = set1(HalfV(), SignedMax<T>());
  const HalfV vnm = set1(HalfV(), std::numeric_limits<T>::lowest());
  const WideV wn1 = set1(WideV(), WideT(-1));
  const WideV wpm = set1(WideV(), WideT(SignedMax<T>()));
  const WideV wnm = set1(WideV(), WideT(std::numeric_limits<T>::lowest()));
  ASSERT_VEC_EQ(wn1, promote(vn1));
  ASSERT_VEC_EQ(wpm, promote(vpm));
  ASSERT_VEC_EQ(wnm, promote(vnm));
}

void TestPromote() {
  TestUnsignedPromote<uint8_t>();
  TestUnsignedPromote<uint16_t>();
  TestUnsignedPromote<uint32_t>();
  // No u64.

  TestSignedPromote<int8_t>();
  TestSignedPromote<int16_t>();
  TestSignedPromote<int32_t>();
  // No i64,float.
}

template <typename T>
void TestDemoteToUnsignedT() {
  using V = vec<T>;
  constexpr size_t N = NumLanes<V>();

  // Verify order.
  const V v = Iota<V>(1);
  const auto narrow = demote_to_unsigned(v);
  using NarrowV = decltype(narrow);
  using NarrowT = Lane<NarrowV>;
  SIMD_ALIGN NarrowT lanes[N];
  store(narrow, lanes);
  for (size_t i = 0; i < N; ++i) {
    ASSERT_EQ(NarrowT(1 + i), lanes[i]);
  }

  // Test extreme values of the input.
  const V v0 = setzero(V());
  const V vp1 = set1(V(), T(1));
  const V vpm = set1(V(), SignedMax<T>());
  const V vnm = set1(V(), std::numeric_limits<T>::lowest());

  const NarrowV n0 = set1(NarrowV(), NarrowT(0));
  const NarrowV n1 = set1(NarrowV(), NarrowT(1));
  const NarrowV npm = set1(NarrowV(), UnsignedMax<NarrowT>());
  const NarrowV nnm = n0;
  ASSERT_VEC_EQ(n0, demote_to_unsigned(v0));
  ASSERT_VEC_EQ(n1, demote_to_unsigned(vp1));
  ASSERT_VEC_EQ(npm, demote_to_unsigned(vpm));
  ASSERT_VEC_EQ(nnm, demote_to_unsigned(vnm));
}

template <typename T>
void TestDemoteToSignedT() {
  using V = vec<T>;
  constexpr size_t N = NumLanes<V>();

  // Verify order.
  const V v = Iota<V>(1);
  const auto narrow = demote(v);
  using NarrowV = decltype(narrow);
  using NarrowT = Lane<NarrowV>;
  SIMD_ALIGN NarrowT lanes[N];
  store(narrow, lanes);
  for (size_t i = 0; i < N; ++i) {
    ASSERT_EQ(NarrowT(1 + i), lanes[i]);
  }

  // Test extreme values of the input.
  const V v0 = setzero(V());
  const V vp1 = set1(V(), T(1));
  const V vpm = set1(V(), SignedMax<T>());
  const V vnm = set1(V(), std::numeric_limits<T>::lowest());

  const NarrowV n0 = setzero(NarrowV());
  const NarrowV np1 = set1(NarrowV(), NarrowT(1));
  const NarrowV npm = set1(NarrowV(), SignedMax<NarrowT>());
  const NarrowV nnm = set1(NarrowV(), std::numeric_limits<NarrowT>::lowest());

  ASSERT_VEC_EQ(n0, demote(v0));
  ASSERT_VEC_EQ(np1, demote(vp1));
  ASSERT_VEC_EQ(npm, demote(vpm));
  ASSERT_VEC_EQ(nnm, demote(vnm));
}

void TestDemote() {
  // No i8,i64.
  TestDemoteToUnsignedT<int16_t>();
  TestDemoteToUnsignedT<int32_t>();
  TestDemoteToSignedT<int16_t>();
  TestDemoteToSignedT<int32_t>();
}

void TestSwizzle() {
  TestShiftBytes();
  ForeachLaneType<TestGetSet>();
  TestBroadcast();
  ForeachLaneType<TestZip>();
  TestShuffle();
  TestExtract();
  TestSpecialShuffles();
  ForeachLaneType<TestSelect>();
  TestPromote();
  TestDemote();
}

}  // namespace swizzle

void RunTests() {
  examples::TestExamples();
  basic::TestBasic();
  arithmetic::TestArithmetic();
  compare::TestCompare();
  logical::TestLogical();
  memory::TestMemory();
  swizzle::TestSwizzle();
}

}  // namespace
}  // namespace SIMD_NAMESPACE

// Instantiate for the current target.
template <>
void SimdTest::operator()<SIMD_TARGET>() {
  SIMD_NAMESPACE::RunTests();
  targets |= SIMD_TARGET::value;
}

}  // namespace simd
