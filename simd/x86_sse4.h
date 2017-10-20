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

// 128-bit SSE4 vectors and operations.
// (No include guard nor namespace: this is included from the middle of simd.h.)

// Avoid compile errors when generating deps.mk.
#ifndef SIMD_DEPS

template <typename T>
struct raw_sse4 {
  using type = __m128i;
};
template <>
struct raw_sse4<float> {
  using type = __m128;
};
template <>
struct raw_sse4<double> {
  using type = __m128d;
};

// All 128-bit blocks equal; returned from load_dup128.
template <typename T>
struct dup128x1 {
  using Raw = typename raw_sse4<T>::type;

  explicit dup128x1(const Raw v) : raw(v) {}

  Raw raw;
};

// Returned by set_shift_*_count, also used by AVX2; do not use directly.
template <typename T>
struct shift_left_count {
  __m128i raw;
};

template <typename T>
struct shift_right_count {
  __m128i raw;
};

template <typename T, size_t N = SSE4::NumLanes<T>()>
class vec_sse4 {
  using Raw = typename raw_sse4<T>::type;
  static constexpr size_t kBytes = N * sizeof(T);
  static_assert((kBytes & (kBytes - 1)) == 0, "Size must be 2^j bytes");
  static_assert(kBytes == 0 || (4 <= kBytes && kBytes <= 16), "Invalid size");

 public:
  SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4() {}
  vec_sse4(const vec_sse4&) = default;
  vec_sse4& operator=(const vec_sse4&) = default;
  SIMD_ATTR_SSE4 SIMD_INLINE explicit vec_sse4(const Raw v) : raw(v) {}

  // Used by non-member functions; avoids verbose .raw() for each argument and
  // reduces Clang compile time of the test by 10-20%.
  SIMD_ATTR_SSE4 SIMD_INLINE operator Raw() const { return raw; }  // NOLINT

  // Compound assignment. Only usable if there is a corresponding non-member
  // binary operator overload. For example, only f32 and f64 support division.
  SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4& operator*=(const vec_sse4 other) {
    return *this = (*this * other);
  }
  SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4& operator/=(const vec_sse4 other) {
    return *this = (*this / other);
  }
  SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4& operator+=(const vec_sse4 other) {
    return *this = (*this + other);
  }
  SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4& operator-=(const vec_sse4 other) {
    return *this = (*this - other);
  }
  SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4& operator&=(const vec_sse4 other) {
    return *this = (*this & other);
  }
  SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4& operator|=(const vec_sse4 other) {
    return *this = (*this | other);
  }
  SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4& operator^=(const vec_sse4 other) {
    return *this = (*this ^ other);
  }

  Raw raw;
};

template <typename T, size_t N>
struct VecT<T, N, SSE4> {
  using type = vec_sse4<T, N>;
};

using u8x16 = vec_sse4<uint8_t, 16>;
using u16x8 = vec_sse4<uint16_t, 8>;
using u32x4 = vec_sse4<uint32_t, 4>;
using u64x2 = vec_sse4<uint64_t, 2>;
using i8x16 = vec_sse4<int8_t, 16>;
using i16x8 = vec_sse4<int16_t, 8>;
using i32x4 = vec_sse4<int32_t, 4>;
using i64x2 = vec_sse4<int64_t, 2>;
using f32x4 = vec_sse4<float, 4>;
using f64x2 = vec_sse4<double, 2>;

using u8x8 = vec_sse4<uint8_t, 8>;
using u16x4 = vec_sse4<uint16_t, 4>;
using u32x2 = vec_sse4<uint32_t, 2>;
using u64x1 = vec_sse4<uint64_t, 1>;
using i8x8 = vec_sse4<int8_t, 8>;
using i16x4 = vec_sse4<int16_t, 4>;
using i32x2 = vec_sse4<int32_t, 2>;
using i64x1 = vec_sse4<int64_t, 1>;
using f32x2 = vec_sse4<float, 2>;
using f64x1 = vec_sse4<double, 1>;

using u8x4 = vec_sse4<uint8_t, 4>;
using u16x2 = vec_sse4<uint16_t, 2>;
using u32x1 = vec_sse4<uint32_t, 1>;
using i8x4 = vec_sse4<int8_t, 4>;
using i16x2 = vec_sse4<int16_t, 2>;
using i32x1 = vec_sse4<int32_t, 1>;
using f32x1 = vec_sse4<float, 1>;

// ------------------------------ Set

// Returns an all-zero vector/part.
template <typename T, size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<T, N> setzero(Desc<T, N, SSE4>) {
  return vec_sse4<T, N>(_mm_setzero_si128());
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec<float, N, SSE4> setzero(Desc<float, N, SSE4>) {
  return Vec<float, N, SSE4>(_mm_setzero_ps());
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec<double, N, SSE4> setzero(Desc<double, N, SSE4>) {
  return Vec<double, N, SSE4>(_mm_setzero_pd());
}

// Returns a vector/part with all lanes set to "t".
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec<uint8_t, N, SSE4> set1(Desc<uint8_t, N, SSE4>,
                                                      const uint8_t t) {
  return Vec<uint8_t, N, SSE4>(_mm_set1_epi8(t));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec<uint16_t, N, SSE4> set1(Desc<uint16_t, N, SSE4>,
                                                       const uint16_t t) {
  return Vec<uint16_t, N, SSE4>(_mm_set1_epi16(t));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec<uint32_t, N, SSE4> set1(Desc<uint32_t, N, SSE4>,
                                                       const uint32_t t) {
  return Vec<uint32_t, N, SSE4>(_mm_set1_epi32(t));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec<uint64_t, N, SSE4> set1(Desc<uint64_t, N, SSE4>,
                                                       const uint64_t t) {
  return Vec<uint64_t, N, SSE4>(_mm_set1_epi64x(t));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec<int8_t, N, SSE4> set1(Desc<int8_t, N, SSE4>,
                                                     const int8_t t) {
  return Vec<int8_t, N, SSE4>(_mm_set1_epi8(t));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec<int16_t, N, SSE4> set1(Desc<int16_t, N, SSE4>,
                                                      const int16_t t) {
  return Vec<int16_t, N, SSE4>(_mm_set1_epi16(t));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec<int32_t, N, SSE4> set1(Desc<int32_t, N, SSE4>,
                                                      const int32_t t) {
  return Vec<int32_t, N, SSE4>(_mm_set1_epi32(t));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec<int64_t, N, SSE4> set1(Desc<int64_t, N, SSE4>,
                                                      const int64_t t) {
  return Vec<int64_t, N, SSE4>(_mm_set1_epi64x(t));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec<float, N, SSE4> set1(Desc<float, N, SSE4>,
                                                    const float t) {
  return Vec<float, N, SSE4>(_mm_set1_ps(t));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec<double, N, SSE4> set1(Desc<double, N, SSE4>,
                                                     const double t) {
  return Vec<double, N, SSE4>(_mm_set1_pd(t));
}

// Returns a vector with lane i=[0, N) set to "first" + i. Unique per-lane
// values are required to detect lane-crossing bugs.
template <typename T, size_t N, typename T2>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<T, N> iota(Desc<T, N, SSE4> d,
                                               const T2 first) {
  SIMD_ALIGN T lanes[N];
  for (size_t i = 0; i < N; ++i) {
    lanes[i] = first + i;
  }
  return load(d, lanes);
}

// Returns a vector/part with value "t".
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec<uint32_t, N, SSE4> set(Desc<uint32_t, N, SSE4>,
                                                      const uint32_t t) {
  return Vec<uint32_t, N, SSE4>(_mm_cvtsi32_si128(t));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec<int32_t, N, SSE4> set(Desc<int32_t, N, SSE4>,
                                                     const int32_t t) {
  return Vec<int32_t, N, SSE4>(_mm_cvtsi32_si128(t));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec<float, N, SSE4> set(Desc<float, N, SSE4>,
                                                   const float t) {
  return Vec<float, N, SSE4>(_mm_set_ss(t));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec<uint64_t, N, SSE4> set(Desc<uint64_t, N, SSE4>,
                                                      const uint64_t t) {
  return Vec<uint64_t, N, SSE4>(_mm_cvtsi64_si128(t));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec<int64_t, N, SSE4> set(Desc<int64_t, N, SSE4>,
                                                     const int64_t t) {
  return Vec<int64_t, N, SSE4>(_mm_cvtsi64_si128(t));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec<double, N, SSE4> set(Desc<double, N, SSE4>,
                                                    const double t) {
  return Vec<double, N, SSE4>(_mm_set_sd(t));
}

// Gets the single value stored in a vector/part.
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE uint32_t get(Desc<uint32_t, N, SSE4>,
                                        const Vec<uint32_t, N, SSE4> v) {
  return _mm_cvtsi128_si32(v);
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE int32_t get(Desc<int32_t, N, SSE4>,
                                       const Vec<int32_t, N, SSE4> v) {
  return _mm_cvtsi128_si32(v);
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE float get(Desc<float, N, SSE4>,
                                     const Vec<float, N, SSE4> v) {
  return _mm_cvtss_f32(v);
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE uint64_t get(Desc<uint64_t, N, SSE4>,
                                        const Vec<uint64_t, N, SSE4> v) {
  return _mm_cvtsi128_si64(v);
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE int64_t get(Desc<int64_t, N, SSE4>,
                                       const Vec<int64_t, N, SSE4> v) {
  return _mm_cvtsi128_si64(v);
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE double get(Desc<double, N, SSE4>,
                                      const Vec<double, N, SSE4> v) {
  return _mm_cvtsd_f64(v);
}

// Returns part of a vector (unspecified whether upper or lower).
template <typename T, size_t N, size_t VN>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<T, N> any_part(Desc<T, N, SSE4>,
                                                   const vec_sse4<T, VN> v) {
  return vec_sse4<T, N>(v);
}

// ================================================== ARITHMETIC

// ------------------------------ Addition

// Unsigned
SIMD_ATTR_SSE4 SIMD_INLINE u8x16 operator+(const u8x16 a, const u8x16 b) {
  return u8x16(_mm_add_epi8(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE u16x8 operator+(const u16x8 a, const u16x8 b) {
  return u16x8(_mm_add_epi16(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE u32x4 operator+(const u32x4 a, const u32x4 b) {
  return u32x4(_mm_add_epi32(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE u64x2 operator+(const u64x2 a, const u64x2 b) {
  return u64x2(_mm_add_epi64(a, b));
}

// Signed
SIMD_ATTR_SSE4 SIMD_INLINE i8x16 operator+(const i8x16 a, const i8x16 b) {
  return i8x16(_mm_add_epi8(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE i16x8 operator+(const i16x8 a, const i16x8 b) {
  return i16x8(_mm_add_epi16(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE i32x4 operator+(const i32x4 a, const i32x4 b) {
  return i32x4(_mm_add_epi32(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE i64x2 operator+(const i64x2 a, const i64x2 b) {
  return i64x2(_mm_add_epi64(a, b));
}

// Float
SIMD_ATTR_SSE4 SIMD_INLINE f32x4 operator+(const f32x4 a, const f32x4 b) {
  return f32x4(_mm_add_ps(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE f64x2 operator+(const f64x2 a, const f64x2 b) {
  return f64x2(_mm_add_pd(a, b));
}

// ------------------------------ Subtraction

// Unsigned
SIMD_ATTR_SSE4 SIMD_INLINE u8x16 operator-(const u8x16 a, const u8x16 b) {
  return u8x16(_mm_sub_epi8(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE u16x8 operator-(const u16x8 a, const u16x8 b) {
  return u16x8(_mm_sub_epi16(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE u32x4 operator-(const u32x4 a, const u32x4 b) {
  return u32x4(_mm_sub_epi32(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE u64x2 operator-(const u64x2 a, const u64x2 b) {
  return u64x2(_mm_sub_epi64(a, b));
}

// Signed
SIMD_ATTR_SSE4 SIMD_INLINE i8x16 operator-(const i8x16 a, const i8x16 b) {
  return i8x16(_mm_sub_epi8(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE i16x8 operator-(const i16x8 a, const i16x8 b) {
  return i16x8(_mm_sub_epi16(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE i32x4 operator-(const i32x4 a, const i32x4 b) {
  return i32x4(_mm_sub_epi32(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE i64x2 operator-(const i64x2 a, const i64x2 b) {
  return i64x2(_mm_sub_epi64(a, b));
}

// Float
SIMD_ATTR_SSE4 SIMD_INLINE f32x4 operator-(const f32x4 a, const f32x4 b) {
  return f32x4(_mm_sub_ps(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE f64x2 operator-(const f64x2 a, const f64x2 b) {
  return f64x2(_mm_sub_pd(a, b));
}

// ------------------------------ Saturating addition

// Returns a + b clamped to the destination range.

// Unsigned
SIMD_ATTR_SSE4 SIMD_INLINE u8x16 add_sat(const u8x16 a, const u8x16 b) {
  return u8x16(_mm_adds_epu8(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE u16x8 add_sat(const u16x8 a, const u16x8 b) {
  return u16x8(_mm_adds_epu16(a, b));
}

// Signed
SIMD_ATTR_SSE4 SIMD_INLINE i8x16 add_sat(const i8x16 a, const i8x16 b) {
  return i8x16(_mm_adds_epi8(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE i16x8 add_sat(const i16x8 a, const i16x8 b) {
  return i16x8(_mm_adds_epi16(a, b));
}

// ------------------------------ Saturating subtraction

// Returns a - b clamped to the destination range.

// Unsigned
SIMD_ATTR_SSE4 SIMD_INLINE u8x16 sub_sat(const u8x16 a, const u8x16 b) {
  return u8x16(_mm_subs_epu8(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE u16x8 sub_sat(const u16x8 a, const u16x8 b) {
  return u16x8(_mm_subs_epu16(a, b));
}

// Signed
SIMD_ATTR_SSE4 SIMD_INLINE i8x16 sub_sat(const i8x16 a, const i8x16 b) {
  return i8x16(_mm_subs_epi8(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE i16x8 sub_sat(const i16x8 a, const i16x8 b) {
  return i16x8(_mm_subs_epi16(a, b));
}

// ------------------------------ Average

// Returns (a + b + 1) / 2

// Unsigned
SIMD_ATTR_SSE4 SIMD_INLINE u8x16 avg(const u8x16 a, const u8x16 b) {
  return u8x16(_mm_avg_epu8(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE u16x8 avg(const u16x8 a, const u16x8 b) {
  return u16x8(_mm_avg_epu16(a, b));
}

// ------------------------------ Shift lanes by constant #bits

// Unsigned
template <int kBits>
SIMD_ATTR_SSE4 SIMD_INLINE u16x8 shift_left(const u16x8 v) {
  return u16x8(_mm_slli_epi16(v, kBits));
}
template <int kBits>
SIMD_ATTR_SSE4 SIMD_INLINE u16x8 shift_right(const u16x8 v) {
  return u16x8(_mm_srli_epi16(v, kBits));
}
template <int kBits>
SIMD_ATTR_SSE4 SIMD_INLINE u32x4 shift_left(const u32x4 v) {
  return u32x4(_mm_slli_epi32(v, kBits));
}
template <int kBits>
SIMD_ATTR_SSE4 SIMD_INLINE u32x4 shift_right(const u32x4 v) {
  return u32x4(_mm_srli_epi32(v, kBits));
}
template <int kBits>
SIMD_ATTR_SSE4 SIMD_INLINE u64x2 shift_left(const u64x2 v) {
  return u64x2(_mm_slli_epi64(v, kBits));
}
template <int kBits>
SIMD_ATTR_SSE4 SIMD_INLINE u64x2 shift_right(const u64x2 v) {
  return u64x2(_mm_srli_epi64(v, kBits));
}

// Signed (no i64 shift_right)
template <int kBits>
SIMD_ATTR_SSE4 SIMD_INLINE i16x8 shift_left(const i16x8 v) {
  return i16x8(_mm_slli_epi16(v, kBits));
}
template <int kBits>
SIMD_ATTR_SSE4 SIMD_INLINE i16x8 shift_right(const i16x8 v) {
  return i16x8(_mm_srai_epi16(v, kBits));
}
template <int kBits>
SIMD_ATTR_SSE4 SIMD_INLINE i32x4 shift_left(const i32x4 v) {
  return i32x4(_mm_slli_epi32(v, kBits));
}
template <int kBits>
SIMD_ATTR_SSE4 SIMD_INLINE i32x4 shift_right(const i32x4 v) {
  return i32x4(_mm_srai_epi32(v, kBits));
}
template <int kBits>
SIMD_ATTR_SSE4 SIMD_INLINE i64x2 shift_left(const i64x2 v) {
  return i64x2(_mm_slli_epi64(v, kBits));
}

// ------------------------------ Shift lanes by same variable #bits

template <typename T, size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE shift_left_count<T> set_shift_left_count(
    Desc<T, N, SSE4>, const int bits) {
  return shift_left_count<T>{_mm_cvtsi32_si128(bits)};
}

// Same as shift_left_count on x86, but different on ARM.
template <typename T, size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE shift_right_count<T> set_shift_right_count(
    Desc<T, N, SSE4>, const int bits) {
  return shift_right_count<T>{_mm_cvtsi32_si128(bits)};
}

// Unsigned (no u8)
SIMD_ATTR_SSE4 SIMD_INLINE u16x8
shift_left_same(const u16x8 v, const shift_left_count<uint16_t> bits) {
  return u16x8(_mm_sll_epi16(v, bits.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE u16x8
shift_right_same(const u16x8 v, const shift_right_count<uint16_t> bits) {
  return u16x8(_mm_srl_epi16(v, bits.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE u32x4
shift_left_same(const u32x4 v, const shift_left_count<uint32_t> bits) {
  return u32x4(_mm_sll_epi32(v, bits.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE u32x4
shift_right_same(const u32x4 v, const shift_right_count<uint32_t> bits) {
  return u32x4(_mm_srl_epi32(v, bits.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE u64x2
shift_left_same(const u64x2 v, const shift_left_count<uint64_t> bits) {
  return u64x2(_mm_sll_epi64(v, bits.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE u64x2
shift_right_same(const u64x2 v, const shift_right_count<uint64_t> bits) {
  return u64x2(_mm_srl_epi64(v, bits.raw));
}

// Signed (no i8,i64)
SIMD_ATTR_SSE4 SIMD_INLINE i16x8
shift_left_same(const i16x8 v, const shift_left_count<int16_t> bits) {
  return i16x8(_mm_sll_epi16(v, bits.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE i16x8
shift_right_same(const i16x8 v, const shift_right_count<int16_t> bits) {
  return i16x8(_mm_sra_epi16(v, bits.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE i32x4
shift_left_same(const i32x4 v, const shift_left_count<int32_t> bits) {
  return i32x4(_mm_sll_epi32(v, bits.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE i32x4
shift_right_same(const i32x4 v, const shift_right_count<int32_t> bits) {
  return i32x4(_mm_sra_epi32(v, bits.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE i64x2
shift_left_same(const i64x2 v, const shift_left_count<int64_t> bits) {
  return i64x2(_mm_sll_epi64(v, bits.raw));
}

// ------------------------------ Shift lanes by independent variable #bits

#if SIMD_X86_AVX2

// Unsigned (no u8,u16)
SIMD_ATTR_SSE4 SIMD_INLINE u32x4 shift_left_var(const u32x4 v,
                                                const u32x4 bits) {
  return u32x4(_mm_sllv_epi32(v, bits));
}
SIMD_ATTR_SSE4 SIMD_INLINE u32x4 shift_right_var(const u32x4 v,
                                                 const u32x4 bits) {
  return u32x4(_mm_srlv_epi32(v, bits));
}
SIMD_ATTR_SSE4 SIMD_INLINE u64x2 shift_left_var(const u64x2 v,
                                                const u64x2 bits) {
  return u64x2(_mm_sllv_epi64(v, bits));
}
SIMD_ATTR_SSE4 SIMD_INLINE u64x2 shift_right_var(const u64x2 v,
                                                 const u64x2 bits) {
  return u64x2(_mm_srlv_epi64(v, bits));
}

// Signed (no i8,i16,i64)
SIMD_ATTR_SSE4 SIMD_INLINE i32x4 shift_left_var(const i32x4 v,
                                                const i32x4 bits) {
  return i32x4(_mm_sllv_epi32(v, bits));
}
SIMD_ATTR_SSE4 SIMD_INLINE i32x4 shift_right_var(const i32x4 v,
                                                 const i32x4 bits) {
  return i32x4(_mm_srav_epi32(v, bits));
}
SIMD_ATTR_SSE4 SIMD_INLINE i64x2 shift_left_var(const i64x2 v,
                                                const i64x2 bits) {
  return i64x2(_mm_sllv_epi64(v, bits));
}

#endif  // SIMD_X86_AVX2

// ------------------------------ Minimum

// Unsigned (no u64)
SIMD_ATTR_SSE4 SIMD_INLINE u8x16 min(const u8x16 a, const u8x16 b) {
  return u8x16(_mm_min_epu8(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE u16x8 min(const u16x8 a, const u16x8 b) {
  return u16x8(_mm_min_epu16(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE u32x4 min(const u32x4 a, const u32x4 b) {
  return u32x4(_mm_min_epu32(a, b));
}

// Signed (no i64)
SIMD_ATTR_SSE4 SIMD_INLINE i8x16 min(const i8x16 a, const i8x16 b) {
  return i8x16(_mm_min_epi8(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE i16x8 min(const i16x8 a, const i16x8 b) {
  return i16x8(_mm_min_epi16(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE i32x4 min(const i32x4 a, const i32x4 b) {
  return i32x4(_mm_min_epi32(a, b));
}

// Float
SIMD_ATTR_SSE4 SIMD_INLINE f32x4 min(const f32x4 a, const f32x4 b) {
  return f32x4(_mm_min_ps(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE f64x2 min(const f64x2 a, const f64x2 b) {
  return f64x2(_mm_min_pd(a, b));
}

// ------------------------------ Maximum

// Unsigned (no u64)
SIMD_ATTR_SSE4 SIMD_INLINE u8x16 max(const u8x16 a, const u8x16 b) {
  return u8x16(_mm_max_epu8(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE u16x8 max(const u16x8 a, const u16x8 b) {
  return u16x8(_mm_max_epu16(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE u32x4 max(const u32x4 a, const u32x4 b) {
  return u32x4(_mm_max_epu32(a, b));
}

// Signed (no i64)
SIMD_ATTR_SSE4 SIMD_INLINE i8x16 max(const i8x16 a, const i8x16 b) {
  return i8x16(_mm_max_epi8(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE i16x8 max(const i16x8 a, const i16x8 b) {
  return i16x8(_mm_max_epi16(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE i32x4 max(const i32x4 a, const i32x4 b) {
  return i32x4(_mm_max_epi32(a, b));
}

// Float
SIMD_ATTR_SSE4 SIMD_INLINE f32x4 max(const f32x4 a, const f32x4 b) {
  return f32x4(_mm_max_ps(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE f64x2 max(const f64x2 a, const f64x2 b) {
  return f64x2(_mm_max_pd(a, b));
}

// Returns the closest value to v within [lo, hi].
template <typename T, size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<T, N> clamp(const vec_sse4<T, N> v,
                                                const vec_sse4<T, N> lo,
                                                const vec_sse4<T, N> hi) {
  return min(max(lo, v), hi);
}

// ------------------------------ Integer multiplication

// Unsigned
SIMD_ATTR_SSE4 SIMD_INLINE u16x8 operator*(const u16x8 a, const u16x8 b) {
  return u16x8(_mm_mullo_epi16(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE u32x4 operator*(const u32x4 a, const u32x4 b) {
  return u32x4(_mm_mullo_epi32(a, b));
}

// Signed
SIMD_ATTR_SSE4 SIMD_INLINE i16x8 operator*(const i16x8 a, const i16x8 b) {
  return i16x8(_mm_mullo_epi16(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE i32x4 operator*(const i32x4 a, const i32x4 b) {
  return i32x4(_mm_mullo_epi32(a, b));
}

// "Extensions": useful but quite performance-portable operations. We add
// functions to this namespace in multiple places.
namespace ext {

// Returns the upper 16 bits of a * b in each lane.
SIMD_ATTR_SSE4 SIMD_INLINE i16x8 mulhi(const i16x8 a, const i16x8 b) {
  return i16x8(_mm_mulhi_epi16(a, b));
}

}  // namespace ext

// Multiplies even lanes (0, 2 ..) and places the double-wide result into
// even and the upper half into its odd neighbor lane.
SIMD_ATTR_SSE4 SIMD_INLINE i64x2 mul_even(const i32x4 a, const i32x4 b) {
  return i64x2(_mm_mul_epi32(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE u64x2 mul_even(const u32x4 a, const u32x4 b) {
  return u64x2(_mm_mul_epu32(a, b));
}

// ------------------------------ Floating-point mul / div

SIMD_ATTR_SSE4 SIMD_INLINE f32x4 operator*(const f32x4 a, const f32x4 b) {
  return f32x4(_mm_mul_ps(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE f64x2 operator*(const f64x2 a, const f64x2 b) {
  return f64x2(_mm_mul_pd(a, b));
}

SIMD_ATTR_SSE4 SIMD_INLINE f32x4 operator/(const f32x4 a, const f32x4 b) {
  return f32x4(_mm_div_ps(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE f64x2 operator/(const f64x2 a, const f64x2 b) {
  return f64x2(_mm_div_pd(a, b));
}

// Approximate reciprocal
SIMD_ATTR_SSE4 SIMD_INLINE f32x4 rcp_approx(const f32x4 v) {
  return f32x4(_mm_rcp_ps(v));
}

// ------------------------------ Floating-point multiply-add variants

// Returns mul * x + add
SIMD_ATTR_SSE4 SIMD_INLINE f32x4 mul_add(const f32x4 mul, const f32x4 x,
                                         const f32x4 add) {
#if SIMD_X86_AVX2
  return f32x4(_mm_fmadd_ps(mul, x, add));
#else
  return mul * x + add;
#endif
}
SIMD_ATTR_SSE4 SIMD_INLINE f64x2 mul_add(const f64x2 mul, const f64x2 x,
                                         const f64x2 add) {
#if SIMD_X86_AVX2
  return f64x2(_mm_fmadd_pd(mul, x, add));
#else
  return mul * x + add;
#endif
}

// Returns mul * x - sub
SIMD_ATTR_SSE4 SIMD_INLINE f32x4 mul_sub(const f32x4 mul, const f32x4 x,
                                         const f32x4 sub) {
#if SIMD_X86_AVX2
  return f32x4(_mm_fmsub_ps(mul, x, sub));
#else
  return mul * x - sub;
#endif
}
SIMD_ATTR_SSE4 SIMD_INLINE f64x2 mul_sub(const f64x2 mul, const f64x2 x,
                                         const f64x2 sub) {
#if SIMD_X86_AVX2
  return f64x2(_mm_fmsub_pd(mul, x, sub));
#else
  return mul * x - sub;
#endif
}

// Returns add - mul * x
SIMD_ATTR_SSE4 SIMD_INLINE f32x4 nmul_add(const f32x4 mul, const f32x4 x,
                                          const f32x4 add) {
#if SIMD_X86_AVX2
  return f32x4(_mm_fnmadd_ps(mul, x, add));
#else
  return add - mul * x;
#endif
}
SIMD_ATTR_SSE4 SIMD_INLINE f64x2 nmul_add(const f64x2 mul, const f64x2 x,
                                          const f64x2 add) {
#if SIMD_X86_AVX2
  return f64x2(_mm_fnmadd_pd(mul, x, add));
#else
  return add - mul * x;
#endif
}

// nmul_sub would require an additional negate of mul or x.

// ------------------------------ Floating-point square root

// Full precision square root
SIMD_ATTR_SSE4 SIMD_INLINE f32x4 sqrt(const f32x4 v) {
  return f32x4(_mm_sqrt_ps(v));
}
SIMD_ATTR_SSE4 SIMD_INLINE f64x2 sqrt(const f64x2 v) {
  return f64x2(_mm_sqrt_pd(v));
}

// Approximate reciprocal square root
SIMD_ATTR_SSE4 SIMD_INLINE f32x4 rsqrt_approx(const f32x4 v) {
  return f32x4(_mm_rsqrt_ps(v));
}

// ------------------------------ Floating-point rounding

// Toward nearest integer
SIMD_ATTR_SSE4 SIMD_INLINE f32x4 round_nearest(const f32x4 v) {
  return f32x4(_mm_round_ps(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}
SIMD_ATTR_SSE4 SIMD_INLINE f64x2 round_nearest(const f64x2 v) {
  return f64x2(_mm_round_pd(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}

// Toward +infinity, aka ceiling
SIMD_ATTR_SSE4 SIMD_INLINE f32x4 round_pos_inf(const f32x4 v) {
  return f32x4(_mm_ceil_ps(v));
}
SIMD_ATTR_SSE4 SIMD_INLINE f64x2 round_pos_inf(const f64x2 v) {
  return f64x2(_mm_ceil_pd(v));
}

// Toward -infinity, aka floor
SIMD_ATTR_SSE4 SIMD_INLINE f32x4 round_neg_inf(const f32x4 v) {
  return f32x4(_mm_floor_ps(v));
}
SIMD_ATTR_SSE4 SIMD_INLINE f64x2 round_neg_inf(const f64x2 v) {
  return f64x2(_mm_floor_pd(v));
}

// ------------------------------ Convert i32 <=> f32

SIMD_ATTR_SSE4 SIMD_INLINE f32x4 f32_from_i32(const i32x4 v) {
  return f32x4(_mm_cvtepi32_ps(v));
}
// Uses current rounding mode, which defaults to round-to-nearest.
SIMD_ATTR_SSE4 SIMD_INLINE i32x4 i32_from_f32(const f32x4 v) {
  return i32x4(_mm_cvtps_epi32(v));
}

// ------------------------------ Cast to/from floating-point representation

SIMD_ATTR_SSE4 SIMD_INLINE f32x4 float_from_bits(const u32x4 v) {
  return f32x4(_mm_castsi128_ps(v));
}
SIMD_ATTR_SSE4 SIMD_INLINE f32x4 float_from_bits(const i32x4 v) {
  return f32x4(_mm_castsi128_ps(v));
}
SIMD_ATTR_SSE4 SIMD_INLINE i32x4 bits_from_float(const f32x4 v) {
  return i32x4(_mm_castps_si128(v));
}

SIMD_ATTR_SSE4 SIMD_INLINE f64x2 float_from_bits(const u64x2 v) {
  return f64x2(_mm_castsi128_pd(v));
}
SIMD_ATTR_SSE4 SIMD_INLINE f64x2 float_from_bits(const i64x2 v) {
  return f64x2(_mm_castsi128_pd(v));
}
SIMD_ATTR_SSE4 SIMD_INLINE i64x2 bits_from_float(const f64x2 v) {
  return i64x2(_mm_castpd_si128(v));
}

// ================================================== COMPARE

// Comparisons fill a lane with 1-bits if the condition is true, else 0.

// ------------------------------ Equality

// Unsigned
SIMD_ATTR_SSE4 SIMD_INLINE u8x16 operator==(const u8x16 a, const u8x16 b) {
  return u8x16(_mm_cmpeq_epi8(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE u16x8 operator==(const u16x8 a, const u16x8 b) {
  return u16x8(_mm_cmpeq_epi16(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE u32x4 operator==(const u32x4 a, const u32x4 b) {
  return u32x4(_mm_cmpeq_epi32(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE u64x2 operator==(const u64x2 a, const u64x2 b) {
  return u64x2(_mm_cmpeq_epi64(a, b));
}

// Signed
SIMD_ATTR_SSE4 SIMD_INLINE i8x16 operator==(const i8x16 a, const i8x16 b) {
  return i8x16(_mm_cmpeq_epi8(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE i16x8 operator==(const i16x8 a, const i16x8 b) {
  return i16x8(_mm_cmpeq_epi16(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE i32x4 operator==(const i32x4 a, const i32x4 b) {
  return i32x4(_mm_cmpeq_epi32(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE i64x2 operator==(const i64x2 a, const i64x2 b) {
  return i64x2(_mm_cmpeq_epi64(a, b));
}

// Float
SIMD_ATTR_SSE4 SIMD_INLINE f32x4 operator==(const f32x4 a, const f32x4 b) {
  return f32x4(_mm_cmpeq_ps(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE f64x2 operator==(const f64x2 a, const f64x2 b) {
  return f64x2(_mm_cmpeq_pd(a, b));
}

// ------------------------------ Strict inequality

// Signed/float <
SIMD_ATTR_SSE4 SIMD_INLINE i8x16 operator<(const i8x16 a, const i8x16 b) {
  return i8x16(_mm_cmpgt_epi8(b, a));
}
SIMD_ATTR_SSE4 SIMD_INLINE i16x8 operator<(const i16x8 a, const i16x8 b) {
  return i16x8(_mm_cmpgt_epi16(b, a));
}
SIMD_ATTR_SSE4 SIMD_INLINE i32x4 operator<(const i32x4 a, const i32x4 b) {
  return i32x4(_mm_cmpgt_epi32(b, a));
}
SIMD_ATTR_SSE4 SIMD_INLINE i64x2 operator<(const i64x2 a, const i64x2 b) {
  return i64x2(_mm_cmpgt_epi64(b, a));
}
SIMD_ATTR_SSE4 SIMD_INLINE f32x4 operator<(const f32x4 a, const f32x4 b) {
  return f32x4(_mm_cmplt_ps(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE f64x2 operator<(const f64x2 a, const f64x2 b) {
  return f64x2(_mm_cmplt_pd(a, b));
}

// Signed/float >
SIMD_ATTR_SSE4 SIMD_INLINE i8x16 operator>(const i8x16 a, const i8x16 b) {
  return i8x16(_mm_cmpgt_epi8(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE i16x8 operator>(const i16x8 a, const i16x8 b) {
  return i16x8(_mm_cmpgt_epi16(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE i32x4 operator>(const i32x4 a, const i32x4 b) {
  return i32x4(_mm_cmpgt_epi32(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE i64x2 operator>(const i64x2 a, const i64x2 b) {
  return i64x2(_mm_cmpgt_epi64(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE f32x4 operator>(const f32x4 a, const f32x4 b) {
  return f32x4(_mm_cmpgt_ps(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE f64x2 operator>(const f64x2 a, const f64x2 b) {
  return f64x2(_mm_cmpgt_pd(a, b));
}

// ------------------------------ Weak inequality

// Float <= >=
SIMD_ATTR_SSE4 SIMD_INLINE f32x4 operator<=(const f32x4 a, const f32x4 b) {
  return f32x4(_mm_cmple_ps(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE f64x2 operator<=(const f64x2 a, const f64x2 b) {
  return f64x2(_mm_cmple_pd(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE f32x4 operator>=(const f32x4 a, const f32x4 b) {
  return f32x4(_mm_cmpge_ps(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE f64x2 operator>=(const f64x2 a, const f64x2 b) {
  return f64x2(_mm_cmpge_pd(a, b));
}

// "Extensions": useful but quite performance-portable operations. We add
// functions to this namespace in multiple places.
namespace ext {

// Returns a bit array of the most significant bit of each byte in "v", i.e.
// sum_i=0..15 of (v[i] >> 7) << i; v[0] is the least-significant byte of "v".
// This is useful for testing/branching based on comparison results.
SIMD_ATTR_SSE4 SIMD_INLINE uint32_t movemask(const u8x16 v) {
  return _mm_movemask_epi8(v);
}

// Returns the most significant bit of each float/double lane (see above).
SIMD_ATTR_SSE4 SIMD_INLINE uint32_t movemask(const f32x4 v) {
  return _mm_movemask_ps(v);
}
SIMD_ATTR_SSE4 SIMD_INLINE uint32_t movemask(const f64x2 v) {
  return _mm_movemask_pd(v);
}

// Returns whether all lanes are equal to zero. Supported for all integer V.
template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE bool all_zero(const vec_sse4<T> v) {
  return static_cast<bool>(_mm_testz_si128(v, v));
}

}  // namespace ext

// ================================================== LOGICAL

// ------------------------------ Bitwise AND

template <typename T, size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<T, N> operator&(const vec_sse4<T, N> a,
                                                    const vec_sse4<T, N> b) {
  return vec_sse4<T, N>(_mm_and_si128(a, b));
}
template <>
SIMD_ATTR_SSE4 SIMD_INLINE f32x4 operator&(const f32x4 a, const f32x4 b) {
  return f32x4(_mm_and_ps(a, b));
}
template <>
SIMD_ATTR_SSE4 SIMD_INLINE f64x2 operator&(const f64x2 a, const f64x2 b) {
  return f64x2(_mm_and_pd(a, b));
}

// ------------------------------ Bitwise AND-NOT

// Returns ~not_mask & mask.
template <typename T, size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<T, N> andnot(const vec_sse4<T, N> not_mask,
                                                 const vec_sse4<T, N> mask) {
  return vec_sse4<T, N>(_mm_andnot_si128(not_mask, mask));
}
SIMD_ATTR_SSE4 SIMD_INLINE f32x4 andnot(const f32x4 not_mask,
                                        const f32x4 mask) {
  return f32x4(_mm_andnot_ps(not_mask, mask));
}
SIMD_ATTR_SSE4 SIMD_INLINE f64x2 andnot(const f64x2 not_mask,
                                        const f64x2 mask) {
  return f64x2(_mm_andnot_pd(not_mask, mask));
}

// ------------------------------ Bitwise OR

template <typename T, size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<T, N> operator|(const vec_sse4<T, N> a,
                                                    const vec_sse4<T, N> b) {
  return vec_sse4<T, N>(_mm_or_si128(a, b));
}
template <>
SIMD_ATTR_SSE4 SIMD_INLINE f32x4 operator|(const f32x4 a, const f32x4 b) {
  return f32x4(_mm_or_ps(a, b));
}
template <>
SIMD_ATTR_SSE4 SIMD_INLINE f64x2 operator|(const f64x2 a, const f64x2 b) {
  return f64x2(_mm_or_pd(a, b));
}

// ------------------------------ Bitwise XOR

template <typename T, size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<T, N> operator^(const vec_sse4<T, N> a,
                                                    const vec_sse4<T, N> b) {
  return vec_sse4<T, N>(_mm_xor_si128(a, b));
}
template <>
SIMD_ATTR_SSE4 SIMD_INLINE f32x4 operator^(const f32x4 a, const f32x4 b) {
  return f32x4(_mm_xor_ps(a, b));
}
template <>
SIMD_ATTR_SSE4 SIMD_INLINE f64x2 operator^(const f64x2 a, const f64x2 b) {
  return f64x2(_mm_xor_pd(a, b));
}

// ================================================== MEMORY

// ------------------------------ Load

template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<T> load(Full<T, SSE4>,
                                            const T* SIMD_RESTRICT aligned) {
  return vec_sse4<T>(_mm_load_si128(reinterpret_cast<const __m128i*>(aligned)));
}
template <>
SIMD_ATTR_SSE4 SIMD_INLINE f32x4
load<float>(Full<float, SSE4>, const float* SIMD_RESTRICT aligned) {
  return f32x4(_mm_load_ps(aligned));
}
template <>
SIMD_ATTR_SSE4 SIMD_INLINE f64x2
load<double>(Full<double, SSE4>, const double* SIMD_RESTRICT aligned) {
  return f64x2(_mm_load_pd(aligned));
}

template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<T> load_unaligned(
    Full<T, SSE4>, const T* SIMD_RESTRICT p) {
  return vec_sse4<T>(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
}
template <>
SIMD_ATTR_SSE4 SIMD_INLINE f32x4
load_unaligned<float>(Full<float, SSE4>, const float* SIMD_RESTRICT p) {
  return f32x4(_mm_loadu_ps(p));
}
template <>
SIMD_ATTR_SSE4 SIMD_INLINE f64x2
load_unaligned<double>(Full<double, SSE4>, const double* SIMD_RESTRICT p) {
  return f64x2(_mm_loadu_pd(p));
}

template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<T, 8 / sizeof(T)> load(
    Desc<T, 8 / sizeof(T), SSE4>, const T* SIMD_RESTRICT p) {
  return vec_sse4<T, 8 / sizeof(T)>(
      _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p)));
}
template <>
SIMD_ATTR_SSE4 SIMD_INLINE f32x2 load<float>(Desc<float, 2, SSE4>,
                                             const float* SIMD_RESTRICT p) {
  const __m128 hi = _mm_setzero_ps();
  return f32x2(_mm_loadl_pi(hi, reinterpret_cast<const __m64*>(p)));
}
template <>
SIMD_ATTR_SSE4 SIMD_INLINE f64x1 load<double>(Desc<double, 1, SSE4>,
                                              const double* SIMD_RESTRICT p) {
  const __m128d hi = _mm_setzero_pd();
  return f64x1(_mm_loadl_pd(hi, p));
}

template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<T, 4 / sizeof(T)> load(
    Desc<T, 4 / sizeof(T), SSE4>, const T* SIMD_RESTRICT p) {
  // TODO(janwas): load_ss?
  int32_t bits;
  CopyBytes<4>(p, &bits);
  return vec_sse4<T, 4 / sizeof(T)>(_mm_cvtsi32_si128(bits));
}
template <>
SIMD_ATTR_SSE4 SIMD_INLINE f32x1 load<float>(Desc<float, 1, SSE4>,
                                             const float* SIMD_RESTRICT p) {
  return f32x1(_mm_load_ss(p));
}

// 128-bit SIMD => nothing to duplicate, same as an unaligned load.
template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE dup128x1<T> load_dup128(
    Full<T, SSE4> d, const T* const SIMD_RESTRICT p) {
  return dup128x1<T>(load_unaligned(d, p));
}

// ------------------------------ Store

template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE void store(const vec_sse4<T> v, Full<T, SSE4>,
                                      T* SIMD_RESTRICT aligned) {
  _mm_store_si128(reinterpret_cast<__m128i*>(aligned), v);
}
template <>
SIMD_ATTR_SSE4 SIMD_INLINE void store<float>(const f32x4 v, Full<float, SSE4>,
                                             float* SIMD_RESTRICT aligned) {
  _mm_store_ps(aligned, v);
}
template <>
SIMD_ATTR_SSE4 SIMD_INLINE void store<double>(const f64x2 v, Full<double, SSE4>,
                                              double* SIMD_RESTRICT aligned) {
  _mm_store_pd(aligned, v);
}

template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE void store(const dup128x1<T> v, Full<T, SSE4> d,
                                      T* SIMD_RESTRICT aligned) {
  store(vec_sse4<T>(v.raw), d, aligned);
}

template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE void store_unaligned(const vec_sse4<T> v,
                                                Full<T, SSE4>,
                                                T* SIMD_RESTRICT p) {
  _mm_storeu_si128(reinterpret_cast<__m128i*>(p), v);
}
template <>
SIMD_ATTR_SSE4 SIMD_INLINE void store_unaligned<float>(const f32x4 v,
                                                       Full<float, SSE4>,
                                                       float* SIMD_RESTRICT p) {
  _mm_storeu_ps(p, v);
}
template <>
SIMD_ATTR_SSE4 SIMD_INLINE void store_unaligned<double>(
    const f64x2 v, Full<double, SSE4>, double* SIMD_RESTRICT p) {
  _mm_storeu_pd(p, v);
}

template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE void store_unaligned(const dup128x1<T> v,
                                                Full<T, SSE4> d,
                                                T* SIMD_RESTRICT p) {
  store_unaligned(vec_sse4<T>(v.raw), d, p);
}

template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE void store(const vec_sse4<T, 8 / sizeof(T)> v,
                                      Desc<T, 8 / sizeof(T), SSE4>,
                                      T* SIMD_RESTRICT p) {
  _mm_storel_epi64(reinterpret_cast<__m128i*>(p), v);
}
template <>
SIMD_ATTR_SSE4 SIMD_INLINE void store<float>(const f32x2 v,
                                             Desc<float, 2, SSE4>,
                                             float* SIMD_RESTRICT p) {
  _mm_storel_pi(reinterpret_cast<__m64*>(p), v);
}
template <>
SIMD_ATTR_SSE4 SIMD_INLINE void store<double>(const f64x1 v,
                                              Desc<double, 1, SSE4>,
                                              double* SIMD_RESTRICT p) {
  _mm_storel_pd(p, v);
}

template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE void store(const vec_sse4<T, 4 / sizeof(T)> v,
                                      Desc<T, 4 / sizeof(T), SSE4>,
                                      T* SIMD_RESTRICT p) {
  // _mm_storeu_si32 is documented but unavailable in Clang; CopyBytes generates
  // bad code; type-punning is unsafe; this actually generates MOVD.
  _mm_store_ss(reinterpret_cast<float * SIMD_RESTRICT>(p), _mm_castsi128_ps(v));
}
template <>
SIMD_ATTR_SSE4 SIMD_INLINE void store<float>(const f32x1 v,
                                             Desc<float, 1, SSE4>,
                                             float* SIMD_RESTRICT p) {
  _mm_store_ss(p, v);
}

// ------------------------------ Non-temporal stores

// Same as aligned stores on non-x86.

template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE void stream(const vec_sse4<T> v, Full<T, SSE4>,
                                       T* SIMD_RESTRICT aligned) {
  _mm_stream_si128(reinterpret_cast<__m128i*>(aligned), v);
}
template <>
SIMD_ATTR_SSE4 SIMD_INLINE void stream<float>(const f32x4 v, Full<float, SSE4>,
                                              float* SIMD_RESTRICT aligned) {
  _mm_stream_ps(aligned, v);
}
template <>
SIMD_ATTR_SSE4 SIMD_INLINE void stream<double>(const f64x2 v,
                                               Full<double, SSE4>,
                                               double* SIMD_RESTRICT aligned) {
  _mm_stream_pd(aligned, v);
}

// ================================================== SWIZZLE

// ------------------------------ 'Extract' other half (see any_part)

// These copy hi into lo (smaller instruction encoding than shifts).
template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<T, 8 / sizeof(T)> other_half(
    const vec_sse4<T> v) {
  return vec_sse4<T, 8 / sizeof(T)>(_mm_unpackhi_epi64(v, v));
}
SIMD_ATTR_SSE4 SIMD_INLINE f32x2 other_half(const f32x4 v) {
  return f32x2(_mm_movehl_ps(v, v));
}
SIMD_ATTR_SSE4 SIMD_INLINE f64x1 other_half(const f64x2 v) {
  return f64x1(_mm_unpackhi_pd(v, v));
}

// ------------------------------ Shift vector by constant #bytes

// 0x01..0F, kBytes = 1 => 0x02..0F00
template <int kBytes, typename T>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<T> shift_bytes_left(const vec_sse4<T> v) {
  return vec_sse4<T>(_mm_slli_si128(v, kBytes));
}

// 0x01..0F, kBytes = 1 => 0x0001..0E
template <int kBytes, typename T>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<T> shift_bytes_right(const vec_sse4<T> v) {
  return vec_sse4<T>(_mm_srli_si128(v, kBytes));
}

// ------------------------------ Extract from 2x 128-bit at constant offset

// Extracts 128 bits from <hi, lo> by skipping the least-significant kBytes.
template <int kBytes, typename T>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<T> extract_concat_bytes(
    const vec_sse4<T> hi, const vec_sse4<T> lo) {
  return vec_sse4<T>(_mm_alignr_epi8(hi, lo, kBytes));
}

// ------------------------------ Broadcast/splat any lane

// Unsigned
template <int kLane>
SIMD_ATTR_SSE4 SIMD_INLINE u32x4 broadcast(const u32x4 v) {
  static_assert(0 <= kLane && kLane < 4, "Invalid lane");
  return u32x4(_mm_shuffle_epi32(v, 0x55 * kLane));
}
template <int kLane>
SIMD_ATTR_SSE4 SIMD_INLINE u64x2 broadcast(const u64x2 v) {
  static_assert(0 <= kLane && kLane < 2, "Invalid lane");
  return u64x2(_mm_shuffle_epi32(v, kLane ? 0xEE : 0x44));
}

// Signed
template <int kLane>
SIMD_ATTR_SSE4 SIMD_INLINE i32x4 broadcast(const i32x4 v) {
  static_assert(0 <= kLane && kLane < 4, "Invalid lane");
  return i32x4(_mm_shuffle_epi32(v, 0x55 * kLane));
}
template <int kLane>
SIMD_ATTR_SSE4 SIMD_INLINE i64x2 broadcast(const i64x2 v) {
  static_assert(0 <= kLane && kLane < 2, "Invalid lane");
  return i64x2(_mm_shuffle_epi32(v, kLane ? 0xEE : 0x44));
}

// Float
template <int kLane>
SIMD_ATTR_SSE4 SIMD_INLINE f32x4 broadcast(const f32x4 v) {
  static_assert(0 <= kLane && kLane < 4, "Invalid lane");
  return f32x4(_mm_shuffle_ps(v, v, 0x55 * kLane));
}
template <int kLane>
SIMD_ATTR_SSE4 SIMD_INLINE f64x2 broadcast(const f64x2 v) {
  static_assert(0 <= kLane && kLane < 2, "Invalid lane");
  return f64x2(_mm_shuffle_pd(v, v, 3 * kLane));
}

// ------------------------------ Shuffle bytes with variable indices

// Returns vector of bytes[from[i]]. "from" is also interpreted as bytes:
// either valid indices in [0, 16) or >= 0x80 to zero the i-th output byte.
template <typename T, typename TI>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<T> shuffle_bytes(const vec_sse4<T> bytes,
                                                     const vec_sse4<TI> from) {
  return vec_sse4<T>(_mm_shuffle_epi8(bytes, from));
}

// ------------------------------ Hard-coded shuffles

// Notation: let i32x4 have lanes 3,2,1,0 (0 is least-significant).
// shuffle_0321 rotates one lane to the right (the previous least-significant
// lane is now most-significant). These could also be implemented via
// extract_concat_bytes but the shuffle_abcd notation is more convenient.

// Swap 64-bit halves
SIMD_ATTR_SSE4 SIMD_INLINE u32x4 shuffle_1032(const u32x4 v) {
  return u32x4(_mm_shuffle_epi32(v, 0x4E));
}
SIMD_ATTR_SSE4 SIMD_INLINE i32x4 shuffle_1032(const i32x4 v) {
  return i32x4(_mm_shuffle_epi32(v, 0x4E));
}
SIMD_ATTR_SSE4 SIMD_INLINE f32x4 shuffle_1032(const f32x4 v) {
  return f32x4(_mm_shuffle_ps(v, v, 0x4E));
}
SIMD_ATTR_SSE4 SIMD_INLINE u64x2 shuffle_01(const u64x2 v) {
  return u64x2(_mm_shuffle_epi32(v, 0x4E));
}
SIMD_ATTR_SSE4 SIMD_INLINE i64x2 shuffle_01(const i64x2 v) {
  return i64x2(_mm_shuffle_epi32(v, 0x4E));
}
SIMD_ATTR_SSE4 SIMD_INLINE f64x2 shuffle_01(const f64x2 v) {
  return f64x2(_mm_shuffle_pd(v, v, 1));
}

// Rotate right 32 bits
SIMD_ATTR_SSE4 SIMD_INLINE u32x4 shuffle_0321(const u32x4 v) {
  return u32x4(_mm_shuffle_epi32(v, 0x39));
}
SIMD_ATTR_SSE4 SIMD_INLINE i32x4 shuffle_0321(const i32x4 v) {
  return i32x4(_mm_shuffle_epi32(v, 0x39));
}
SIMD_ATTR_SSE4 SIMD_INLINE f32x4 shuffle_0321(const f32x4 v) {
  return f32x4(_mm_shuffle_ps(v, v, 0x39));
}
// Rotate left 32 bits
SIMD_ATTR_SSE4 SIMD_INLINE u32x4 shuffle_2103(const u32x4 v) {
  return u32x4(_mm_shuffle_epi32(v, 0x93));
}
SIMD_ATTR_SSE4 SIMD_INLINE i32x4 shuffle_2103(const i32x4 v) {
  return i32x4(_mm_shuffle_epi32(v, 0x93));
}
SIMD_ATTR_SSE4 SIMD_INLINE f32x4 shuffle_2103(const f32x4 v) {
  return f32x4(_mm_shuffle_ps(v, v, 0x93));
}

// ------------------------------ Interleave lanes

// Interleaves lanes from halves of the 128-bit blocks of "a" (which provides
// the least-significant lane) and "b". To concatenate two half-width integers
// into one, use zip_lo/hi instead (also works with scalar).

SIMD_ATTR_SSE4 SIMD_INLINE u8x16 interleave_lo(const u8x16 a, const u8x16 b) {
  return u8x16(_mm_unpacklo_epi8(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE u16x8 interleave_lo(const u16x8 a, const u16x8 b) {
  return u16x8(_mm_unpacklo_epi16(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE u32x4 interleave_lo(const u32x4 a, const u32x4 b) {
  return u32x4(_mm_unpacklo_epi32(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE u64x2 interleave_lo(const u64x2 a, const u64x2 b) {
  return u64x2(_mm_unpacklo_epi64(a, b));
}

SIMD_ATTR_SSE4 SIMD_INLINE i8x16 interleave_lo(const i8x16 a, const i8x16 b) {
  return i8x16(_mm_unpacklo_epi8(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE i16x8 interleave_lo(const i16x8 a, const i16x8 b) {
  return i16x8(_mm_unpacklo_epi16(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE i32x4 interleave_lo(const i32x4 a, const i32x4 b) {
  return i32x4(_mm_unpacklo_epi32(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE i64x2 interleave_lo(const i64x2 a, const i64x2 b) {
  return i64x2(_mm_unpacklo_epi64(a, b));
}

SIMD_ATTR_SSE4 SIMD_INLINE f32x4 interleave_lo(const f32x4 a, const f32x4 b) {
  return f32x4(_mm_unpacklo_ps(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE f64x2 interleave_lo(const f64x2 a, const f64x2 b) {
  return f64x2(_mm_unpacklo_pd(a, b));
}

SIMD_ATTR_SSE4 SIMD_INLINE u8x16 interleave_hi(const u8x16 a, const u8x16 b) {
  return u8x16(_mm_unpackhi_epi8(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE u16x8 interleave_hi(const u16x8 a, const u16x8 b) {
  return u16x8(_mm_unpackhi_epi16(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE u32x4 interleave_hi(const u32x4 a, const u32x4 b) {
  return u32x4(_mm_unpackhi_epi32(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE u64x2 interleave_hi(const u64x2 a, const u64x2 b) {
  return u64x2(_mm_unpackhi_epi64(a, b));
}

SIMD_ATTR_SSE4 SIMD_INLINE i8x16 interleave_hi(const i8x16 a, const i8x16 b) {
  return i8x16(_mm_unpackhi_epi8(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE i16x8 interleave_hi(const i16x8 a, const i16x8 b) {
  return i16x8(_mm_unpackhi_epi16(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE i32x4 interleave_hi(const i32x4 a, const i32x4 b) {
  return i32x4(_mm_unpackhi_epi32(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE i64x2 interleave_hi(const i64x2 a, const i64x2 b) {
  return i64x2(_mm_unpackhi_epi64(a, b));
}

SIMD_ATTR_SSE4 SIMD_INLINE f32x4 interleave_hi(const f32x4 a, const f32x4 b) {
  return f32x4(_mm_unpackhi_ps(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE f64x2 interleave_hi(const f64x2 a, const f64x2 b) {
  return f64x2(_mm_unpackhi_pd(a, b));
}

// ------------------------------ Zip lanes

// Same as interleave_*, except that the return lanes are double-width integers;
// this is necessary because the single-lane scalar cannot return two values.

SIMD_ATTR_SSE4 SIMD_INLINE u16x8 zip_lo(const u8x16 a, const u8x16 b) {
  return u16x8(_mm_unpacklo_epi8(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE u32x4 zip_lo(const u16x8 a, const u16x8 b) {
  return u32x4(_mm_unpacklo_epi16(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE u64x2 zip_lo(const u32x4 a, const u32x4 b) {
  return u64x2(_mm_unpacklo_epi32(a, b));
}

SIMD_ATTR_SSE4 SIMD_INLINE i16x8 zip_lo(const i8x16 a, const i8x16 b) {
  return i16x8(_mm_unpacklo_epi8(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE i32x4 zip_lo(const i16x8 a, const i16x8 b) {
  return i32x4(_mm_unpacklo_epi16(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE i64x2 zip_lo(const i32x4 a, const i32x4 b) {
  return i64x2(_mm_unpacklo_epi32(a, b));
}

SIMD_ATTR_SSE4 SIMD_INLINE u16x8 zip_hi(const u8x16 a, const u8x16 b) {
  return u16x8(_mm_unpackhi_epi8(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE u32x4 zip_hi(const u16x8 a, const u16x8 b) {
  return u32x4(_mm_unpackhi_epi16(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE u64x2 zip_hi(const u32x4 a, const u32x4 b) {
  return u64x2(_mm_unpackhi_epi32(a, b));
}

SIMD_ATTR_SSE4 SIMD_INLINE i16x8 zip_hi(const i8x16 a, const i8x16 b) {
  return i16x8(_mm_unpackhi_epi8(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE i32x4 zip_hi(const i16x8 a, const i16x8 b) {
  return i32x4(_mm_unpackhi_epi16(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE i64x2 zip_hi(const i32x4 a, const i32x4 b) {
  return i64x2(_mm_unpackhi_epi32(a, b));
}

// ------------------------------ Promotions (part w/ narrow lanes -> full)

// Unsigned: zero-extend.
SIMD_ATTR_SSE4 SIMD_INLINE u16x8 convert_to(Full<uint16_t, SSE4>,
                                            const u8x8 v) {
  return u16x8(_mm_cvtepu8_epi16(v));
}
SIMD_ATTR_SSE4 SIMD_INLINE u32x4 convert_to(Full<uint32_t, SSE4>,
                                            const u8x4 v) {
  return u32x4(_mm_cvtepu8_epi32(v));
}
SIMD_ATTR_SSE4 SIMD_INLINE i16x8 convert_to(Full<int16_t, SSE4>, const u8x8 v) {
  return i16x8(_mm_cvtepu8_epi16(v));
}
SIMD_ATTR_SSE4 SIMD_INLINE i32x4 convert_to(Full<int32_t, SSE4>, const u8x4 v) {
  return i32x4(_mm_cvtepu8_epi32(v));
}
SIMD_ATTR_SSE4 SIMD_INLINE u32x4 convert_to(Full<uint32_t, SSE4>,
                                            const u16x4 v) {
  return u32x4(_mm_cvtepu16_epi32(v));
}
SIMD_ATTR_SSE4 SIMD_INLINE i32x4 convert_to(Full<int32_t, SSE4>,
                                            const u16x4 v) {
  return i32x4(_mm_cvtepu16_epi32(v));
}
SIMD_ATTR_SSE4 SIMD_INLINE u64x2 convert_to(Full<uint64_t, SSE4>,
                                            const u32x2 v) {
  return u64x2(_mm_cvtepu32_epi64(v));
}

// Signed: replicate sign bit.
SIMD_ATTR_SSE4 SIMD_INLINE i16x8 convert_to(Full<int16_t, SSE4>, const i8x8 v) {
  return i16x8(_mm_cvtepi8_epi16(v));
}
SIMD_ATTR_SSE4 SIMD_INLINE i32x4 convert_to(Full<int32_t, SSE4>, const i8x4 v) {
  return i32x4(_mm_cvtepi8_epi32(v));
}
SIMD_ATTR_SSE4 SIMD_INLINE i32x4 convert_to(Full<int32_t, SSE4>,
                                            const i16x4 v) {
  return i32x4(_mm_cvtepi16_epi32(v));
}
SIMD_ATTR_SSE4 SIMD_INLINE i64x2 convert_to(Full<int64_t, SSE4>,
                                            const i32x2 v) {
  return i64x2(_mm_cvtepi32_epi64(v));
}

// ------------------------------ Demotions (full -> part w/ narrow lanes)

SIMD_ATTR_SSE4 SIMD_INLINE u16x4 convert_to(Part<uint16_t, 4>, const i32x4 v) {
  return u16x4(_mm_packus_epi32(v, v));
}

SIMD_ATTR_SSE4 SIMD_INLINE u8x4 convert_to(Part<uint8_t, 4>, const i32x4 v) {
  const __m128i u16 = _mm_packus_epi32(v, v);
  return u8x4(_mm_packus_epi16(u16, u16));
}

SIMD_ATTR_SSE4 SIMD_INLINE u8x8 convert_to(Part<uint8_t, 8>, const i16x8 v) {
  return u8x8(_mm_packus_epi16(v, v));
}

SIMD_ATTR_SSE4 SIMD_INLINE i16x4 convert_to(Part<int16_t, 4>, const i32x4 v) {
  return i16x4(_mm_packs_epi32(v, v));
}

SIMD_ATTR_SSE4 SIMD_INLINE i8x4 convert_to(Part<int8_t, 4>, const i32x4 v) {
  const __m128i i16 = _mm_packs_epi32(v, v);
  return i8x4(_mm_packs_epi16(i16, i16));
}

SIMD_ATTR_SSE4 SIMD_INLINE i8x8 convert_to(Part<int8_t, 8>, const i16x8 v) {
  return i8x8(_mm_packs_epi16(v, v));
}

// ------------------------------ Select/blend

// Returns mask ? b : a. Due to ARM's semantics, each lane of "mask" must
// equal T(0) or ~T(0) although x86 may only check the most significant bit.
template <typename T, size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<T, N> select(const vec_sse4<T, N> a,
                                                 const vec_sse4<T, N> b,
                                                 const vec_sse4<T, N> mask) {
  return vec_sse4<T, N>(_mm_blendv_epi8(a, b, mask));
}
template <>
SIMD_ATTR_SSE4 SIMD_INLINE f32x4 select<float>(const f32x4 a, const f32x4 b,
                                               const f32x4 mask) {
  return f32x4(_mm_blendv_ps(a, b, mask));
}
template <>
SIMD_ATTR_SSE4 SIMD_INLINE f64x2 select<double>(const f64x2 a, const f64x2 b,
                                                const f64x2 mask) {
  return f64x2(_mm_blendv_pd(a, b, mask));
}

// ------------------------------ AES cipher

// One round of AES. "round_key" is a constant for breaking the symmetry of AES
// (ensures previously equal columns differ afterwards).
SIMD_ATTR_SSE4 SIMD_INLINE u8x16 aes_round(const u8x16 state,
                                           const u8x16 round_key) {
  return u8x16(_mm_aesenc_si128(state, round_key));
}

// ------------------------------ Horizontal sum (reduction)

// "Extensions": useful but quite performance-portable operations. We add
// functions to this namespace in multiple places.
namespace ext {

// Returns 64-bit sums of 8-byte groups.
SIMD_ATTR_SSE4 SIMD_INLINE u64x2 sums_of_u8x8(const u8x16 v) {
  return u64x2(_mm_sad_epu8(v, _mm_setzero_si128()));
}

// Supported for u32x4, i32x4 and f32x4. Returns the sum in each lane.
template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE vec_sse4<T> horz_sum(const vec_sse4<T> v3210) {
  const vec_sse4<T> v1032 = shuffle_1032(v3210);
  const vec_sse4<T> v31_20_31_20 = v3210 + v1032;
  const vec_sse4<T> v20_31_20_31 = shuffle_0321(v31_20_31_20);
  return v20_31_20_31 + v31_20_31_20;
}

SIMD_ATTR_SSE4 SIMD_INLINE u64x2 horz_sum(const u64x2 v10) {
  const u64x2 v01 = shuffle_01(v10);
  return v10 + v01;
}

SIMD_ATTR_SSE4 SIMD_INLINE i64x2 horz_sum(const i64x2 v10) {
  const i64x2 v01 = shuffle_01(v10);
  return v10 + v01;
}

SIMD_ATTR_SSE4 SIMD_INLINE f64x2 horz_sum(const f64x2 v10) {
  const f64x2 v01 = shuffle_01(v10);
  return v10 + v01;
}

}  // namespace ext

// TODO(janwas): wrappers for all intrinsics (in x86 namespace).

#endif  // SIMD_DEPS
