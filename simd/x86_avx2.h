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

// 256-bit AVX2 vectors and operations.
// (No include guard nor namespace: this is included from the middle of simd.h.)

// WARNING: no operations cross 128-bit boundaries. In particular,
// "broadcast", pack and zip may have surprising behavior.

// Avoid compile errors when generating deps.mk.
#ifndef SIMD_DEPS

template <typename T>
struct raw_avx2 {
  using type = __m256i;
};
template <>
struct raw_avx2<float> {
  using type = __m256;
};
template <>
struct raw_avx2<double> {
  using type = __m256d;
};

// All 128-bit blocks equal; returned from load_dup128.
template <typename T>
struct dup128x2 {
  using Raw = typename raw_avx2<T>::type;

  explicit dup128x2(const Raw v) : raw(v) {}

  Raw raw;
};

template <typename T, size_t N = AVX2::NumLanes<T>()>
class vec_avx2 {
  using Raw = typename raw_avx2<T>::type;
  static constexpr size_t kBytes = N * sizeof(T);
  static_assert((kBytes & (kBytes - 1)) == 0, "Size must be 2^j bytes");
  static_assert(kBytes == 0 || (4 <= kBytes && kBytes <= 32), "Invalid size");

 public:
  SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2() {}
  vec_avx2(const vec_avx2&) = default;
  vec_avx2& operator=(const vec_avx2&) = default;
  SIMD_ATTR_AVX2 SIMD_INLINE explicit vec_avx2(const Raw v) : raw(v) {}

  // Used by non-member functions; avoids verbose .raw() for each argument and
  // reduces Clang compile time of the test by 10-20%.
  SIMD_ATTR_AVX2 SIMD_INLINE operator Raw() const { return raw; }  // NOLINT

  // Compound assignment. Only usable if there is a corresponding non-member
  // binary operator overload. For example, only f32 and f64 support division.
  SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2& operator*=(const vec_avx2 other) {
    return *this = (*this * other);
  }
  SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2& operator/=(const vec_avx2 other) {
    return *this = (*this / other);
  }
  SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2& operator+=(const vec_avx2 other) {
    return *this = (*this + other);
  }
  SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2& operator-=(const vec_avx2 other) {
    return *this = (*this - other);
  }
  SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2& operator&=(const vec_avx2 other) {
    return *this = (*this & other);
  }
  SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2& operator|=(const vec_avx2 other) {
    return *this = (*this | other);
  }
  SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2& operator^=(const vec_avx2 other) {
    return *this = (*this ^ other);
  }

  Raw raw;
};

template <typename T, size_t N>
struct VecT<T, N, AVX2> {
  using type = vec_avx2<T, N>;
};

using u8x32 = vec_avx2<uint8_t, 32>;
using u16x16 = vec_avx2<uint16_t, 16>;
using u32x8 = vec_avx2<uint32_t, 8>;
using u64x4 = vec_avx2<uint64_t, 4>;
using i8x32 = vec_avx2<int8_t, 32>;
using i16x16 = vec_avx2<int16_t, 16>;
using i32x8 = vec_avx2<int32_t, 8>;
using i64x4 = vec_avx2<int64_t, 4>;
using f32x8 = vec_avx2<float, 8>;
using f64x4 = vec_avx2<double, 4>;

// ------------------------------ Set

// Returns an all-zero vector.
template <typename T, size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<T, N> setzero(Desc<T, N, AVX2>) {
  return vec_avx2<T, N>(_mm256_setzero_si256());
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE Vec<float, N, AVX2> setzero(Desc<float, N, AVX2>) {
  return Vec<float, N, AVX2>(_mm256_setzero_ps());
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE Vec<double, N, AVX2> setzero(Desc<double, N, AVX2>) {
  return Vec<double, N, AVX2>(_mm256_setzero_pd());
}

template <typename T, size_t N, typename T2>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<T, N> iota(Desc<T, N, AVX2> d,
                                               const T2 first) {
  SIMD_ALIGN T lanes[N];
  for (size_t i = 0; i < N; ++i) {
    lanes[i] = first + i;
  }
  return load(d, lanes);
}

// Returns a vector with all lanes set to "t".
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE Vec<uint8_t, N, AVX2> set1(Desc<uint8_t, N, AVX2>,
                                                      const uint8_t t) {
  return Vec<uint8_t, N, AVX2>(_mm256_set1_epi8(t));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE Vec<uint16_t, N, AVX2> set1(Desc<uint16_t, N, AVX2>,
                                                       const uint16_t t) {
  return Vec<uint16_t, N, AVX2>(_mm256_set1_epi16(t));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE Vec<uint32_t, N, AVX2> set1(Desc<uint32_t, N, AVX2>,
                                                       const uint32_t t) {
  return Vec<uint32_t, N, AVX2>(_mm256_set1_epi32(t));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE Vec<uint64_t, N, AVX2> set1(Desc<uint64_t, N, AVX2>,
                                                       const uint64_t t) {
  return Vec<uint64_t, N, AVX2>(_mm256_set1_epi64x(t));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE Vec<int8_t, N, AVX2> set1(Desc<int8_t, N, AVX2>,
                                                     const int8_t t) {
  return Vec<int8_t, N, AVX2>(_mm256_set1_epi8(t));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE Vec<int16_t, N, AVX2> set1(Desc<int16_t, N, AVX2>,
                                                      const int16_t t) {
  return Vec<int16_t, N, AVX2>(_mm256_set1_epi16(t));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE Vec<int32_t, N, AVX2> set1(Desc<int32_t, N, AVX2>,
                                                      const int32_t t) {
  return Vec<int32_t, N, AVX2>(_mm256_set1_epi32(t));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE Vec<int64_t, N, AVX2> set1(Desc<int64_t, N, AVX2>,
                                                      const int64_t t) {
  return Vec<int64_t, N, AVX2>(_mm256_set1_epi64x(t));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE Vec<float, N, AVX2> set1(Desc<float, N, AVX2>,
                                                    const float t) {
  return Vec<float, N, AVX2>(_mm256_set1_ps(t));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE Vec<double, N, AVX2> set1(Desc<double, N, AVX2>,
                                                     const double t) {
  return Vec<double, N, AVX2>(_mm256_set1_pd(t));
}

// Returns part of a vector (unspecified whether upper or lower).
template <typename T, size_t N, size_t VN>
SIMD_ATTR_AVX2 SIMD_INLINE Vec<T, N, SSE4> any_part(Part<T, N>,
                                                    const vec_avx2<T, VN> v) {
  return Vec<T, N, SSE4>(_mm256_castsi256_si128(v));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE Vec<float, N, SSE4> any_part(Part<float, N>,
                                                        f32x8 v) {
  return Vec<float, N, SSE4>(_mm256_castps256_ps128(v));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE Vec<double, N, SSE4> any_part(Part<double, N>,
                                                         f64x4 v) {
  return Vec<double, N, SSE4>(_mm256_castpd256_pd128(v));
}

// Gets the single value stored in a vector/part.
template <typename T, size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE T get(Part<T, N>, const vec_avx2<T, N> v) {
  const Part<T, 1> d;
  return get(d, any_part(d, v));
}

// ================================================== ARITHMETIC

// ------------------------------ Addition

// Unsigned
SIMD_ATTR_AVX2 SIMD_INLINE u8x32 operator+(const u8x32 a, const u8x32 b) {
  return u8x32(_mm256_add_epi8(a, b));
}
SIMD_ATTR_AVX2 SIMD_INLINE u16x16 operator+(const u16x16 a, const u16x16 b) {
  return u16x16(_mm256_add_epi16(a, b));
}
SIMD_ATTR_AVX2 SIMD_INLINE u32x8 operator+(const u32x8 a, const u32x8 b) {
  return u32x8(_mm256_add_epi32(a, b));
}
SIMD_ATTR_AVX2 SIMD_INLINE u64x4 operator+(const u64x4 a, const u64x4 b) {
  return u64x4(_mm256_add_epi64(a, b));
}

// Signed
SIMD_ATTR_AVX2 SIMD_INLINE i8x32 operator+(const i8x32 a, const i8x32 b) {
  return i8x32(_mm256_add_epi8(a, b));
}
SIMD_ATTR_AVX2 SIMD_INLINE i16x16 operator+(const i16x16 a, const i16x16 b) {
  return i16x16(_mm256_add_epi16(a, b));
}
SIMD_ATTR_AVX2 SIMD_INLINE i32x8 operator+(const i32x8 a, const i32x8 b) {
  return i32x8(_mm256_add_epi32(a, b));
}
SIMD_ATTR_AVX2 SIMD_INLINE i64x4 operator+(const i64x4 a, const i64x4 b) {
  return i64x4(_mm256_add_epi64(a, b));
}

// Float
SIMD_ATTR_AVX2 SIMD_INLINE f32x8 operator+(const f32x8 a, const f32x8 b) {
  return f32x8(_mm256_add_ps(a, b));
}
SIMD_ATTR_AVX2 SIMD_INLINE f64x4 operator+(const f64x4 a, const f64x4 b) {
  return f64x4(_mm256_add_pd(a, b));
}

// ------------------------------ Subtraction

// Unsigned
SIMD_ATTR_AVX2 SIMD_INLINE u8x32 operator-(const u8x32 a, const u8x32 b) {
  return u8x32(_mm256_sub_epi8(a, b));
}
SIMD_ATTR_AVX2 SIMD_INLINE u16x16 operator-(const u16x16 a, const u16x16 b) {
  return u16x16(_mm256_sub_epi16(a, b));
}
SIMD_ATTR_AVX2 SIMD_INLINE u32x8 operator-(const u32x8 a, const u32x8 b) {
  return u32x8(_mm256_sub_epi32(a, b));
}
SIMD_ATTR_AVX2 SIMD_INLINE u64x4 operator-(const u64x4 a, const u64x4 b) {
  return u64x4(_mm256_sub_epi64(a, b));
}

// Signed
SIMD_ATTR_AVX2 SIMD_INLINE i8x32 operator-(const i8x32 a, const i8x32 b) {
  return i8x32(_mm256_sub_epi8(a, b));
}
SIMD_ATTR_AVX2 SIMD_INLINE i16x16 operator-(const i16x16 a, const i16x16 b) {
  return i16x16(_mm256_sub_epi16(a, b));
}
SIMD_ATTR_AVX2 SIMD_INLINE i32x8 operator-(const i32x8 a, const i32x8 b) {
  return i32x8(_mm256_sub_epi32(a, b));
}
SIMD_ATTR_AVX2 SIMD_INLINE i64x4 operator-(const i64x4 a, const i64x4 b) {
  return i64x4(_mm256_sub_epi64(a, b));
}

// Float
SIMD_ATTR_AVX2 SIMD_INLINE f32x8 operator-(const f32x8 a, const f32x8 b) {
  return f32x8(_mm256_sub_ps(a, b));
}
SIMD_ATTR_AVX2 SIMD_INLINE f64x4 operator-(const f64x4 a, const f64x4 b) {
  return f64x4(_mm256_sub_pd(a, b));
}

// ------------------------------ Saturating addition

// Returns a + b clamped to the destination range.

// Unsigned
SIMD_ATTR_AVX2 SIMD_INLINE u8x32 add_sat(const u8x32 a, const u8x32 b) {
  return u8x32(_mm256_adds_epu8(a, b));
}
SIMD_ATTR_AVX2 SIMD_INLINE u16x16 add_sat(const u16x16 a, const u16x16 b) {
  return u16x16(_mm256_adds_epu16(a, b));
}

// Signed
SIMD_ATTR_AVX2 SIMD_INLINE i8x32 add_sat(const i8x32 a, const i8x32 b) {
  return i8x32(_mm256_adds_epi8(a, b));
}
SIMD_ATTR_AVX2 SIMD_INLINE i16x16 add_sat(const i16x16 a, const i16x16 b) {
  return i16x16(_mm256_adds_epi16(a, b));
}

// ------------------------------ Saturating subtraction

// Returns a - b clamped to the destination range.

// Unsigned
SIMD_ATTR_AVX2 SIMD_INLINE u8x32 sub_sat(const u8x32 a, const u8x32 b) {
  return u8x32(_mm256_subs_epu8(a, b));
}
SIMD_ATTR_AVX2 SIMD_INLINE u16x16 sub_sat(const u16x16 a, const u16x16 b) {
  return u16x16(_mm256_subs_epu16(a, b));
}

// Signed
SIMD_ATTR_AVX2 SIMD_INLINE i8x32 sub_sat(const i8x32 a, const i8x32 b) {
  return i8x32(_mm256_subs_epi8(a, b));
}
SIMD_ATTR_AVX2 SIMD_INLINE i16x16 sub_sat(const i16x16 a, const i16x16 b) {
  return i16x16(_mm256_subs_epi16(a, b));
}

// ------------------------------ Average

// Returns (a + b + 1) / 2

// Unsigned
SIMD_ATTR_AVX2 SIMD_INLINE u8x32 avg(const u8x32 a, const u8x32 b) {
  return u8x32(_mm256_avg_epu8(a, b));
}
SIMD_ATTR_AVX2 SIMD_INLINE u16x16 avg(const u16x16 a, const u16x16 b) {
  return u16x16(_mm256_avg_epu16(a, b));
}

// ------------------------------ Shift lanes by constant #bits

// Unsigned
template <int kBits>
SIMD_ATTR_AVX2 SIMD_INLINE u16x16 shift_left(const u16x16 v) {
  return u16x16(_mm256_slli_epi16(v, kBits));
}
template <int kBits>
SIMD_ATTR_AVX2 SIMD_INLINE u16x16 shift_right(const u16x16 v) {
  return u16x16(_mm256_srli_epi16(v, kBits));
}
template <int kBits>
SIMD_ATTR_AVX2 SIMD_INLINE u32x8 shift_left(const u32x8 v) {
  return u32x8(_mm256_slli_epi32(v, kBits));
}
template <int kBits>
SIMD_ATTR_AVX2 SIMD_INLINE u32x8 shift_right(const u32x8 v) {
  return u32x8(_mm256_srli_epi32(v, kBits));
}
template <int kBits>
SIMD_ATTR_AVX2 SIMD_INLINE u64x4 shift_left(const u64x4 v) {
  return u64x4(_mm256_slli_epi64(v, kBits));
}
template <int kBits>
SIMD_ATTR_AVX2 SIMD_INLINE u64x4 shift_right(const u64x4 v) {
  return u64x4(_mm256_srli_epi64(v, kBits));
}

// Signed (no i64 shift_right)
template <int kBits>
SIMD_ATTR_AVX2 SIMD_INLINE i16x16 shift_left(const i16x16 v) {
  return i16x16(_mm256_slli_epi16(v, kBits));
}
template <int kBits>
SIMD_ATTR_AVX2 SIMD_INLINE i16x16 shift_right(const i16x16 v) {
  return i16x16(_mm256_srai_epi16(v, kBits));
}
template <int kBits>
SIMD_ATTR_AVX2 SIMD_INLINE i32x8 shift_left(const i32x8 v) {
  return i32x8(_mm256_slli_epi32(v, kBits));
}
template <int kBits>
SIMD_ATTR_AVX2 SIMD_INLINE i32x8 shift_right(const i32x8 v) {
  return i32x8(_mm256_srai_epi32(v, kBits));
}
template <int kBits>
SIMD_ATTR_AVX2 SIMD_INLINE i64x4 shift_left(const i64x4 v) {
  return i64x4(_mm256_slli_epi64(v, kBits));
}

// ------------------------------ Shift lanes by same variable #bits

template <typename T, size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE shift_left_count<T> set_shift_left_count(
    Desc<T, N, AVX2>, const int bits) {
  return shift_left_count<T>{_mm_cvtsi32_si128(bits)};
}

// Same as shift_left_count on x86, but different on ARM.
template <typename T, size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE shift_right_count<T> set_shift_right_count(
    Desc<T, N, AVX2>, const int bits) {
  return shift_right_count<T>{_mm_cvtsi32_si128(bits)};
}

// Unsigned (no u8)
SIMD_ATTR_AVX2 SIMD_INLINE u16x16
shift_left_same(const u16x16 v, const shift_left_count<uint16_t> bits) {
  return u16x16(_mm256_sll_epi16(v, bits.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE u16x16
shift_right_same(const u16x16 v, const shift_right_count<uint16_t> bits) {
  return u16x16(_mm256_srl_epi16(v, bits.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE u32x8
shift_left_same(const u32x8 v, const shift_left_count<uint32_t> bits) {
  return u32x8(_mm256_sll_epi32(v, bits.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE u32x8
shift_right_same(const u32x8 v, const shift_right_count<uint32_t> bits) {
  return u32x8(_mm256_srl_epi32(v, bits.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE u64x4
shift_left_same(const u64x4 v, const shift_left_count<uint64_t> bits) {
  return u64x4(_mm256_sll_epi64(v, bits.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE u64x4
shift_right_same(const u64x4 v, const shift_right_count<uint64_t> bits) {
  return u64x4(_mm256_srl_epi64(v, bits.raw));
}

// Signed (no i8,i64)
SIMD_ATTR_AVX2 SIMD_INLINE i16x16
shift_left_same(const i16x16 v, const shift_left_count<int16_t> bits) {
  return i16x16(_mm256_sll_epi16(v, bits.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE i16x16
shift_right_same(const i16x16 v, const shift_right_count<int16_t> bits) {
  return i16x16(_mm256_sra_epi16(v, bits.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE i32x8
shift_left_same(const i32x8 v, const shift_left_count<int32_t> bits) {
  return i32x8(_mm256_sll_epi32(v, bits.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE i32x8
shift_right_same(const i32x8 v, const shift_right_count<int32_t> bits) {
  return i32x8(_mm256_sra_epi32(v, bits.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE i64x4
shift_left_same(const i64x4 v, const shift_left_count<int64_t> bits) {
  return i64x4(_mm256_sll_epi64(v, bits.raw));
}

// ------------------------------ Shift lanes by independent variable #bits

// Unsigned (no u8,u16)
SIMD_ATTR_AVX2 SIMD_INLINE u32x8 shift_left_var(const u32x8 v,
                                                const u32x8 bits) {
  return u32x8(_mm256_sllv_epi32(v, bits));
}
SIMD_ATTR_AVX2 SIMD_INLINE u32x8 shift_right_var(const u32x8 v,
                                                 const u32x8 bits) {
  return u32x8(_mm256_srlv_epi32(v, bits));
}
SIMD_ATTR_AVX2 SIMD_INLINE u64x4 shift_left_var(const u64x4 v,
                                                const u64x4 bits) {
  return u64x4(_mm256_sllv_epi64(v, bits));
}
SIMD_ATTR_AVX2 SIMD_INLINE u64x4 shift_right_var(const u64x4 v,
                                                 const u64x4 bits) {
  return u64x4(_mm256_srlv_epi64(v, bits));
}

// Signed (no i8,i16,i64)
SIMD_ATTR_AVX2 SIMD_INLINE i32x8 shift_left_var(const i32x8 v,
                                                const i32x8 bits) {
  return i32x8(_mm256_sllv_epi32(v, bits));
}
SIMD_ATTR_AVX2 SIMD_INLINE i32x8 shift_right_var(const i32x8 v,
                                                 const i32x8 bits) {
  return i32x8(_mm256_srav_epi32(v, bits));
}
SIMD_ATTR_AVX2 SIMD_INLINE i64x4 shift_left_var(const i64x4 v,
                                                const i64x4 bits) {
  return i64x4(_mm256_sllv_epi64(v, bits));
}

// ------------------------------ Minimum

// Unsigned (no u64)
SIMD_ATTR_AVX2 SIMD_INLINE u8x32 min(const u8x32 a, const u8x32 b) {
  return u8x32(_mm256_min_epu8(a, b));
}
SIMD_ATTR_AVX2 SIMD_INLINE u16x16 min(const u16x16 a, const u16x16 b) {
  return u16x16(_mm256_min_epu16(a, b));
}
SIMD_ATTR_AVX2 SIMD_INLINE u32x8 min(const u32x8 a, const u32x8 b) {
  return u32x8(_mm256_min_epu32(a, b));
}

// Signed (no i64)
SIMD_ATTR_AVX2 SIMD_INLINE i8x32 min(const i8x32 a, const i8x32 b) {
  return i8x32(_mm256_min_epi8(a, b));
}
SIMD_ATTR_AVX2 SIMD_INLINE i16x16 min(const i16x16 a, const i16x16 b) {
  return i16x16(_mm256_min_epi16(a, b));
}
SIMD_ATTR_AVX2 SIMD_INLINE i32x8 min(const i32x8 a, const i32x8 b) {
  return i32x8(_mm256_min_epi32(a, b));
}

// Float
SIMD_ATTR_AVX2 SIMD_INLINE f32x8 min(const f32x8 a, const f32x8 b) {
  return f32x8(_mm256_min_ps(a, b));
}
SIMD_ATTR_AVX2 SIMD_INLINE f64x4 min(const f64x4 a, const f64x4 b) {
  return f64x4(_mm256_min_pd(a, b));
}

// ------------------------------ Maximum

// Unsigned (no u64)
SIMD_ATTR_AVX2 SIMD_INLINE u8x32 max(const u8x32 a, const u8x32 b) {
  return u8x32(_mm256_max_epu8(a, b));
}
SIMD_ATTR_AVX2 SIMD_INLINE u16x16 max(const u16x16 a, const u16x16 b) {
  return u16x16(_mm256_max_epu16(a, b));
}
SIMD_ATTR_AVX2 SIMD_INLINE u32x8 max(const u32x8 a, const u32x8 b) {
  return u32x8(_mm256_max_epu32(a, b));
}

// Signed (no i64)
SIMD_ATTR_AVX2 SIMD_INLINE i8x32 max(const i8x32 a, const i8x32 b) {
  return i8x32(_mm256_max_epi8(a, b));
}
SIMD_ATTR_AVX2 SIMD_INLINE i16x16 max(const i16x16 a, const i16x16 b) {
  return i16x16(_mm256_max_epi16(a, b));
}
SIMD_ATTR_AVX2 SIMD_INLINE i32x8 max(const i32x8 a, const i32x8 b) {
  return i32x8(_mm256_max_epi32(a, b));
}

// Float
SIMD_ATTR_AVX2 SIMD_INLINE f32x8 max(const f32x8 a, const f32x8 b) {
  return f32x8(_mm256_max_ps(a, b));
}
SIMD_ATTR_AVX2 SIMD_INLINE f64x4 max(const f64x4 a, const f64x4 b) {
  return f64x4(_mm256_max_pd(a, b));
}

// Returns the closest value to v within [lo, hi].
template <typename T, size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<T, N> clamp(const vec_avx2<T, N> v,
                                                const vec_avx2<T, N> lo,
                                                const vec_avx2<T, N> hi) {
  return min(max(lo, v), hi);
}

// ------------------------------ Integer multiplication

// Unsigned
SIMD_ATTR_AVX2 SIMD_INLINE u16x16 operator*(const u16x16 a, const u16x16 b) {
  return u16x16(_mm256_mullo_epi16(a, b));
}
SIMD_ATTR_AVX2 SIMD_INLINE u32x8 operator*(const u32x8 a, const u32x8 b) {
  return u32x8(_mm256_mullo_epi32(a, b));
}

// Signed
SIMD_ATTR_AVX2 SIMD_INLINE i16x16 operator*(const i16x16 a, const i16x16 b) {
  return i16x16(_mm256_mullo_epi16(a, b));
}
SIMD_ATTR_AVX2 SIMD_INLINE i32x8 operator*(const i32x8 a, const i32x8 b) {
  return i32x8(_mm256_mullo_epi32(a, b));
}

// "Extensions": useful but quite performance-portable operations. We add
// functions to this namespace in multiple places.
namespace ext {

// Returns the upper 16 bits of a * b in each lane.
SIMD_ATTR_AVX2 SIMD_INLINE i16x16 mulhi(const i16x16 a, const i16x16 b) {
  return i16x16(_mm256_mulhi_epi16(a, b));
}

SIMD_ATTR_AVX2 SIMD_INLINE i16x16 mulhrs(const i16x16 a, const i16x16 b) {
  return i16x16(_mm256_mulhrs_epi16(a, b));
}

}  // namespace ext

// Multiplies even lanes (0, 2 ..) and places the double-wide result into
// even and the upper half into its odd neighbor lane.
SIMD_ATTR_AVX2 SIMD_INLINE i64x4 mul_even(const i32x8 a, const i32x8 b) {
  return i64x4(_mm256_mul_epi32(a, b));
}
SIMD_ATTR_AVX2 SIMD_INLINE u64x4 mul_even(const u32x8 a, const u32x8 b) {
  return u64x4(_mm256_mul_epu32(a, b));
}

// ------------------------------ Floating-point mul / div

SIMD_ATTR_AVX2 SIMD_INLINE f32x8 operator*(const f32x8 a, const f32x8 b) {
  return f32x8(_mm256_mul_ps(a, b));
}
SIMD_ATTR_AVX2 SIMD_INLINE f64x4 operator*(const f64x4 a, const f64x4 b) {
  return f64x4(_mm256_mul_pd(a, b));
}

SIMD_ATTR_AVX2 SIMD_INLINE f32x8 operator/(const f32x8 a, const f32x8 b) {
  return f32x8(_mm256_div_ps(a, b));
}
SIMD_ATTR_AVX2 SIMD_INLINE f64x4 operator/(const f64x4 a, const f64x4 b) {
  return f64x4(_mm256_div_pd(a, b));
}

// Approximate reciprocal
SIMD_ATTR_AVX2 SIMD_INLINE f32x8 rcp_approx(const f32x8 v) {
  return f32x8(_mm256_rcp_ps(v));
}

// ------------------------------ Floating-point multiply-add variants

// Returns mul * x + add
SIMD_ATTR_AVX2 SIMD_INLINE f32x8 mul_add(const f32x8 mul, const f32x8 x,
                                         const f32x8 add) {
  return f32x8(_mm256_fmadd_ps(mul, x, add));
}
SIMD_ATTR_AVX2 SIMD_INLINE f64x4 mul_add(const f64x4 mul, const f64x4 x,
                                         const f64x4 add) {
  return f64x4(_mm256_fmadd_pd(mul, x, add));
}

// Returns mul * x - sub
SIMD_ATTR_AVX2 SIMD_INLINE f32x8 mul_sub(const f32x8 mul, const f32x8 x,
                                         const f32x8 sub) {
  return f32x8(_mm256_fmsub_ps(mul, x, sub));
}
SIMD_ATTR_AVX2 SIMD_INLINE f64x4 mul_sub(const f64x4 mul, const f64x4 x,
                                         const f64x4 sub) {
  return f64x4(_mm256_fmsub_pd(mul, x, sub));
}

// Returns add - mul * x
SIMD_ATTR_AVX2 SIMD_INLINE f32x8 nmul_add(const f32x8 mul, const f32x8 x,
                                          const f32x8 add) {
  return f32x8(_mm256_fnmadd_ps(mul, x, add));
}
SIMD_ATTR_AVX2 SIMD_INLINE f64x4 nmul_add(const f64x4 mul, const f64x4 x,
                                          const f64x4 add) {
  return f64x4(_mm256_fnmadd_pd(mul, x, add));
}

// nmul_sub would require an additional negate of mul or x.

// ------------------------------ Floating-point square root

// Full precision square root
SIMD_ATTR_AVX2 SIMD_INLINE f32x8 sqrt(const f32x8 v) {
  return f32x8(_mm256_sqrt_ps(v));
}
SIMD_ATTR_AVX2 SIMD_INLINE f64x4 sqrt(const f64x4 v) {
  return f64x4(_mm256_sqrt_pd(v));
}

// Approximate reciprocal square root
SIMD_ATTR_AVX2 SIMD_INLINE f32x8 rsqrt_approx(const f32x8 v) {
  return f32x8(_mm256_rsqrt_ps(v));
}

// ------------------------------ Floating-point rounding

// Toward nearest integer
SIMD_ATTR_AVX2 SIMD_INLINE f32x8 round_nearest(const f32x8 v) {
  return f32x8(
      _mm256_round_ps(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}
SIMD_ATTR_AVX2 SIMD_INLINE f64x4 round_nearest(const f64x4 v) {
  return f64x4(
      _mm256_round_pd(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}

// Toward +infinity, aka ceiling
SIMD_ATTR_AVX2 SIMD_INLINE f32x8 round_pos_inf(const f32x8 v) {
  return f32x8(_mm256_ceil_ps(v));
}
SIMD_ATTR_AVX2 SIMD_INLINE f64x4 round_pos_inf(const f64x4 v) {
  return f64x4(_mm256_ceil_pd(v));
}

// Toward -infinity, aka floor
SIMD_ATTR_AVX2 SIMD_INLINE f32x8 round_neg_inf(const f32x8 v) {
  return f32x8(_mm256_floor_ps(v));
}
SIMD_ATTR_AVX2 SIMD_INLINE f64x4 round_neg_inf(const f64x4 v) {
  return f64x4(_mm256_floor_pd(v));
}

// ------------------------------ Convert i32 <=> f32

SIMD_ATTR_AVX2 SIMD_INLINE f32x8 f32_from_i32(const i32x8 v) {
  return f32x8(_mm256_cvtepi32_ps(v));
}
// Uses current rounding mode, which defaults to round-to-nearest.
SIMD_ATTR_AVX2 SIMD_INLINE i32x8 i32_from_f32(const f32x8 v) {
  return i32x8(_mm256_cvtps_epi32(v));
}

// ------------------------------ Cast to/from floating-point representation

SIMD_ATTR_AVX2 SIMD_INLINE f32x8 float_from_bits(const u32x8 v) {
  return f32x8(_mm256_castsi256_ps(v));
}
SIMD_ATTR_AVX2 SIMD_INLINE f32x8 float_from_bits(const i32x8 v) {
  return f32x8(_mm256_castsi256_ps(v));
}
SIMD_ATTR_AVX2 SIMD_INLINE i32x8 bits_from_float(const f32x8 v) {
  return i32x8(_mm256_castps_si256(v));
}

SIMD_ATTR_AVX2 SIMD_INLINE f64x4 float_from_bits(const u64x4 v) {
  return f64x4(_mm256_castsi256_pd(v));
}
SIMD_ATTR_AVX2 SIMD_INLINE f64x4 float_from_bits(const i64x4 v) {
  return f64x4(_mm256_castsi256_pd(v));
}
SIMD_ATTR_AVX2 SIMD_INLINE i64x4 bits_from_float(const f64x4 v) {
  return i64x4(_mm256_castpd_si256(v));
}

// ------------------------------ Horizontal sum (reduction)

// "Extensions": useful but quite performance-portable operations. We add
// functions to this namespace in multiple places.
namespace ext {

// Returns 64-bit sums of 8-byte groups.
SIMD_ATTR_AVX2 SIMD_INLINE u64x4 sums_of_u8x8(const u8x32 v) {
  return u64x4(_mm256_sad_epu8(v, _mm256_setzero_si256()));
}

// Returns vector with swapped 128-bit halves: H,L |-> L,H.
template <typename T, size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<T, N> SwapBlocks(const vec_avx2<T, N> vHL) {
  return vec_avx2<T, N>(_mm256_permute2x128_si256(vHL, vHL, 1));
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE f32x8 SwapBlocks(const f32x8 vHL) {
  return f32x8(_mm256_permute2f128_ps(vHL, vHL, 1));
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE f64x4 SwapBlocks(const f64x4 vHL) {
  return f64x4(_mm256_permute2f128_pd(vHL, vHL, 1));
}

// Returns sum{lane[i]} in each lane. "v3210" is a replicated 128-bit block.
template <typename T, size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<T, N> SumsOfLanes(
    char (&tag)[4], const vec_avx2<T, N> v3210) {
  const auto v1032 = shuffle_1032(v3210);
  const auto v31_20_31_20 = v3210 + v1032;
  const auto v20_31_20_31 = shuffle_0321(v31_20_31_20);
  return v20_31_20_31 + v31_20_31_20;
}

template <typename T, size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<T, N> SumsOfLanes(
    char (&tag)[8], const vec_avx2<T, N> v10) {
  const auto v01 = shuffle_01(v10);
  return v10 + v01;
}

// Supported for {uif}32x8, {uif}64x4. Returns the sum in each lane.
template <typename T, size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<T, N> horz_sum(const vec_avx2<T, N> vHL) {
  const vec_avx2<T, N> vLH = SwapBlocks(vHL);
  char tag[sizeof(T)];
  return SumsOfLanes(tag, vLH + vHL);
}

}  // namespace ext

// ================================================== COMPARE

// Comparisons fill a lane with 1-bits if the condition is true, else 0.

// ------------------------------ Equality

// Unsigned
SIMD_ATTR_AVX2 SIMD_INLINE u8x32 operator==(const u8x32 a, const u8x32 b) {
  return u8x32(_mm256_cmpeq_epi8(a, b));
}
SIMD_ATTR_AVX2 SIMD_INLINE u16x16 operator==(const u16x16 a, const u16x16 b) {
  return u16x16(_mm256_cmpeq_epi16(a, b));
}
SIMD_ATTR_AVX2 SIMD_INLINE u32x8 operator==(const u32x8 a, const u32x8 b) {
  return u32x8(_mm256_cmpeq_epi32(a, b));
}
SIMD_ATTR_AVX2 SIMD_INLINE u64x4 operator==(const u64x4 a, const u64x4 b) {
  return u64x4(_mm256_cmpeq_epi64(a, b));
}

// Signed
SIMD_ATTR_AVX2 SIMD_INLINE i8x32 operator==(const i8x32 a, const i8x32 b) {
  return i8x32(_mm256_cmpeq_epi8(a, b));
}
SIMD_ATTR_AVX2 SIMD_INLINE i16x16 operator==(const i16x16 a, const i16x16 b) {
  return i16x16(_mm256_cmpeq_epi16(a, b));
}
SIMD_ATTR_AVX2 SIMD_INLINE i32x8 operator==(const i32x8 a, const i32x8 b) {
  return i32x8(_mm256_cmpeq_epi32(a, b));
}
SIMD_ATTR_AVX2 SIMD_INLINE i64x4 operator==(const i64x4 a, const i64x4 b) {
  return i64x4(_mm256_cmpeq_epi64(a, b));
}

// Float
SIMD_ATTR_AVX2 SIMD_INLINE f32x8 operator==(const f32x8 a, const f32x8 b) {
  return f32x8(_mm256_cmp_ps(a, b, _CMP_EQ_OQ));
}
SIMD_ATTR_AVX2 SIMD_INLINE f64x4 operator==(const f64x4 a, const f64x4 b) {
  return f64x4(_mm256_cmp_pd(a, b, _CMP_EQ_OQ));
}

// ------------------------------ Strict inequality

// Signed/float <
SIMD_ATTR_AVX2 SIMD_INLINE i8x32 operator<(const i8x32 a, const i8x32 b) {
  return i8x32(_mm256_cmpgt_epi8(b, a));
}
SIMD_ATTR_AVX2 SIMD_INLINE i16x16 operator<(const i16x16 a, const i16x16 b) {
  return i16x16(_mm256_cmpgt_epi16(b, a));
}
SIMD_ATTR_AVX2 SIMD_INLINE i32x8 operator<(const i32x8 a, const i32x8 b) {
  return i32x8(_mm256_cmpgt_epi32(b, a));
}
SIMD_ATTR_AVX2 SIMD_INLINE i64x4 operator<(const i64x4 a, const i64x4 b) {
  return i64x4(_mm256_cmpgt_epi64(b, a));
}
SIMD_ATTR_AVX2 SIMD_INLINE f32x8 operator<(const f32x8 a, const f32x8 b) {
  return f32x8(_mm256_cmp_ps(a, b, _CMP_LT_OQ));
}
SIMD_ATTR_AVX2 SIMD_INLINE f64x4 operator<(const f64x4 a, const f64x4 b) {
  return f64x4(_mm256_cmp_pd(a, b, _CMP_LT_OQ));
}

// Signed/float >
SIMD_ATTR_AVX2 SIMD_INLINE i8x32 operator>(const i8x32 a, const i8x32 b) {
  return i8x32(_mm256_cmpgt_epi8(a, b));
}
SIMD_ATTR_AVX2 SIMD_INLINE i16x16 operator>(const i16x16 a, const i16x16 b) {
  return i16x16(_mm256_cmpgt_epi16(a, b));
}
SIMD_ATTR_AVX2 SIMD_INLINE i32x8 operator>(const i32x8 a, const i32x8 b) {
  return i32x8(_mm256_cmpgt_epi32(a, b));
}
SIMD_ATTR_AVX2 SIMD_INLINE i64x4 operator>(const i64x4 a, const i64x4 b) {
  return i64x4(_mm256_cmpgt_epi64(a, b));
}
SIMD_ATTR_AVX2 SIMD_INLINE f32x8 operator>(const f32x8 a, const f32x8 b) {
  return f32x8(_mm256_cmp_ps(a, b, _CMP_GT_OQ));
}
SIMD_ATTR_AVX2 SIMD_INLINE f64x4 operator>(const f64x4 a, const f64x4 b) {
  return f64x4(_mm256_cmp_pd(a, b, _CMP_GT_OQ));
}

// ------------------------------ Weak inequality

// Float <= >=
SIMD_ATTR_AVX2 SIMD_INLINE f32x8 operator<=(const f32x8 a, const f32x8 b) {
  return f32x8(_mm256_cmp_ps(a, b, _CMP_LE_OQ));
}
SIMD_ATTR_AVX2 SIMD_INLINE f64x4 operator<=(const f64x4 a, const f64x4 b) {
  return f64x4(_mm256_cmp_pd(a, b, _CMP_LE_OQ));
}
SIMD_ATTR_AVX2 SIMD_INLINE f32x8 operator>=(const f32x8 a, const f32x8 b) {
  return f32x8(_mm256_cmp_ps(a, b, _CMP_GE_OQ));
}
SIMD_ATTR_AVX2 SIMD_INLINE f64x4 operator>=(const f64x4 a, const f64x4 b) {
  return f64x4(_mm256_cmp_pd(a, b, _CMP_GE_OQ));
}

// "Extensions": useful but quite performance-portable operations. We add
// functions to this namespace in multiple places.
namespace ext {

// Returns a bit array of the most significant bit of each byte in "v", i.e.
// sum_i=0..31 of (v[i] >> 7) << i; v[0] is the least-significant byte of "v".
// This is useful for testing/branching based on comparison results.
SIMD_ATTR_AVX2 SIMD_INLINE uint32_t movemask(const u8x32 v) {
  return _mm256_movemask_epi8(v);
}

// Returns the most significant bit of each float/double lane (see above).
SIMD_ATTR_AVX2 SIMD_INLINE uint32_t movemask(const f32x8 v) {
  return _mm256_movemask_ps(v);
}
SIMD_ATTR_AVX2 SIMD_INLINE uint32_t movemask(const f64x4 v) {
  return _mm256_movemask_pd(v);
}

// Returns whether all lanes are equal to zero. Supported for all integer V.
template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE bool all_zero(const vec_avx2<T> v) {
  return static_cast<bool>(_mm256_testz_si256(v, v));
}

}  // namespace ext

// ================================================== LOGICAL

// ------------------------------ Bitwise AND

template <typename T, size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<T, N> operator&(const vec_avx2<T, N> a,
                                                    const vec_avx2<T, N> b) {
  return vec_avx2<T, N>(_mm256_and_si256(a, b));
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE f32x8 operator&(const f32x8 a, const f32x8 b) {
  return f32x8(_mm256_and_ps(a, b));
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE f64x4 operator&(const f64x4 a, const f64x4 b) {
  return f64x4(_mm256_and_pd(a, b));
}

// ------------------------------ Bitwise AND-NOT

// Returns ~not_mask & mask.
template <typename T, size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<T, N> andnot(const vec_avx2<T, N> not_mask,
                                                 const vec_avx2<T, N> mask) {
  return vec_avx2<T, N>(_mm256_andnot_si256(not_mask, mask));
}
SIMD_ATTR_AVX2 SIMD_INLINE f32x8 andnot(const f32x8 not_mask,
                                        const f32x8 mask) {
  return f32x8(_mm256_andnot_ps(not_mask, mask));
}
SIMD_ATTR_AVX2 SIMD_INLINE f64x4 andnot(const f64x4 not_mask,
                                        const f64x4 mask) {
  return f64x4(_mm256_andnot_pd(not_mask, mask));
}

// ------------------------------ Bitwise OR

template <typename T, size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<T, N> operator|(const vec_avx2<T, N> a,
                                                    const vec_avx2<T, N> b) {
  return vec_avx2<T, N>(_mm256_or_si256(a, b));
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE f32x8 operator|(const f32x8 a, const f32x8 b) {
  return f32x8(_mm256_or_ps(a, b));
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE f64x4 operator|(const f64x4 a, const f64x4 b) {
  return f64x4(_mm256_or_pd(a, b));
}

// ------------------------------ Bitwise XOR

template <typename T, size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<T, N> operator^(const vec_avx2<T, N> a,
                                                    const vec_avx2<T, N> b) {
  return vec_avx2<T, N>(_mm256_xor_si256(a, b));
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE f32x8 operator^(const f32x8 a, const f32x8 b) {
  return f32x8(_mm256_xor_ps(a, b));
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE f64x4 operator^(const f64x4 a, const f64x4 b) {
  return f64x4(_mm256_xor_pd(a, b));
}

// ================================================== MEMORY

// ------------------------------ Load

template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<T> load(Full<T, AVX2>,
                                            const T* SIMD_RESTRICT aligned) {
  return vec_avx2<T>(
      _mm256_load_si256(reinterpret_cast<const __m256i*>(aligned)));
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE f32x8
load<float>(Full<float, AVX2>, const float* SIMD_RESTRICT aligned) {
  return f32x8(_mm256_load_ps(aligned));
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE f64x4
load<double>(Full<double, AVX2>, const double* SIMD_RESTRICT aligned) {
  return f64x4(_mm256_load_pd(aligned));
}

template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<T> load_unaligned(
    Full<T, AVX2>, const T* SIMD_RESTRICT p) {
  return vec_avx2<T>(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)));
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE f32x8
load_unaligned<float>(Full<float, AVX2>, const float* SIMD_RESTRICT p) {
  return f32x8(_mm256_loadu_ps(p));
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE f64x4
load_unaligned<double>(Full<double, AVX2>, const double* SIMD_RESTRICT p) {
  return f64x4(_mm256_loadu_pd(p));
}

// Loads 128 bit and duplicates into both 128-bit halves. This avoids the
// 3-cycle cost of moving data between 128-bit halves and avoids port 5.
template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE dup128x2<T> load_dup128(
    Full<T, AVX2>, const T* const SIMD_RESTRICT p) {
  const Full<T, SSE4> d128;
  // NOTE: Clang 3.9 generates VINSERTF128; 4 yields the desired VBROADCASTI128.
  return dup128x2<T>(_mm256_broadcastsi128_si256(load(d128, p)));
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE dup128x2<float> load_dup128(
    Full<float, AVX2>, const float* const SIMD_RESTRICT p) {
  return dup128x2<float>(
      _mm256_broadcast_ps(reinterpret_cast<const __m128*>(p)));
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE dup128x2<double> load_dup128(
    Full<double, AVX2>, const double* const SIMD_RESTRICT p) {
  return dup128x2<double>(
      _mm256_broadcast_pd(reinterpret_cast<const __m128d*>(p)));
}

// ------------------------------ Store

template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE void store(const vec_avx2<T> v, Full<T, AVX2>,
                                      T* SIMD_RESTRICT aligned) {
  _mm256_store_si256(reinterpret_cast<__m256i*>(aligned), v);
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE void store<float>(const f32x8 v, Full<float, AVX2>,
                                             float* SIMD_RESTRICT aligned) {
  _mm256_store_ps(aligned, v);
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE void store<double>(const f64x4 v, Full<double, AVX2>,
                                              double* SIMD_RESTRICT aligned) {
  _mm256_store_pd(aligned, v);
}

template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE void store(const dup128x2<T> v, Full<T, AVX2> d,
                                      T* SIMD_RESTRICT aligned) {
  return store(vec_avx2<T>(v.raw), d, aligned);
}

template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE void store_unaligned(const vec_avx2<T> v,
                                                Full<T, AVX2>,
                                                T* SIMD_RESTRICT p) {
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v);
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE void store_unaligned<float>(const f32x8 v,
                                                       Full<float, AVX2>,
                                                       float* SIMD_RESTRICT p) {
  _mm256_storeu_ps(p, v);
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE void store_unaligned<double>(
    const f64x4 v, Full<double, AVX2>, double* SIMD_RESTRICT p) {
  _mm256_storeu_pd(p, v);
}

template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE void store_unaligned(const dup128x2<T> v,
                                                Full<T, AVX2> d,
                                                T* SIMD_RESTRICT p) {
  return store_unaligned(vec_avx2<T>(v.raw), d, p);
}

// ------------------------------ Non-temporal stores

// Same as aligned stores on non-x86.

template <typename T, size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE void stream(const vec_avx2<T, N> v, Full<T, AVX2>,
                                       T* SIMD_RESTRICT aligned) {
  _mm256_stream_si256(reinterpret_cast<__m256i*>(aligned), v);
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE void stream<float>(const f32x8 v, Full<float, AVX2>,
                                              float* SIMD_RESTRICT aligned) {
  _mm256_stream_ps(aligned, v);
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE void stream<double>(const f64x4 v,
                                               Full<double, AVX2>,
                                               double* SIMD_RESTRICT aligned) {
  _mm256_stream_pd(aligned, v);
}

// stream(u32/64), store_fence and cache control already defined by x86_sse4.h.

// ================================================== SWIZZLE

// ------------------------------ Extract other half (see any_part)

template <typename T, size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_sse4<T, (N + 1) / 2> other_half(
    const vec_avx2<T, N> v) {
  return vec_sse4<T, (N + 1) / 2>(_mm256_extracti128_si256(v.raw, 1));
}
SIMD_ATTR_AVX2 SIMD_INLINE f32x4 other_half(const f32x8 v) {
  return f32x4(_mm256_extractf128_ps(v.raw, 1));
}
SIMD_ATTR_AVX2 SIMD_INLINE f64x2 other_half(const f64x4 v) {
  return f64x2(_mm256_extractf128_pd(v.raw, 1));
}

// ------------------------------ Shift vector by constant #bytes

// 0x01..0F, kBytes = 1 => 0x02..0F00
template <int kBytes, typename T, size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<T, N> shift_bytes_left(
    const vec_avx2<T, N> v) {
  return vec_avx2<T, N>(_mm256_slli_si256(v, kBytes));
}

// 0x01..0F, kBytes = 1 => 0x0001..0E
template <int kBytes, typename T, size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<T, N> shift_bytes_right(
    const vec_avx2<T, N> v) {
  return vec_avx2<T, N>(_mm256_srli_si256(v, kBytes));
}

// ------------------------------ Extract from 2x 128-bit at constant offset

// Extracts 128 bits from <hi, lo> by skipping the least-significant kBytes.
template <int kBytes, typename T, size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<T, N> extract_concat_bytes(
    const vec_avx2<T, N> hi, const vec_avx2<T, N> lo) {
  return vec_avx2<T, N>(_mm256_alignr_epi8(hi, lo, kBytes));
}

// ------------------------------ Broadcast/splat any lane

// Unsigned
template <int kLane>
SIMD_ATTR_AVX2 SIMD_INLINE u32x8 broadcast(const u32x8 v) {
  static_assert(0 <= kLane && kLane < 4, "Invalid lane");
  return u32x8(_mm256_shuffle_epi32(v, 0x55 * kLane));
}
template <int kLane>
SIMD_ATTR_AVX2 SIMD_INLINE u64x4 broadcast(const u64x4 v) {
  static_assert(0 <= kLane && kLane < 2, "Invalid lane");
  return u64x4(_mm256_shuffle_epi32(v, kLane ? 0xEE : 0x44));
}

// Signed
template <int kLane>
SIMD_ATTR_AVX2 SIMD_INLINE i32x8 broadcast(const i32x8 v) {
  static_assert(0 <= kLane && kLane < 4, "Invalid lane");
  return i32x8(_mm256_shuffle_epi32(v, 0x55 * kLane));
}
template <int kLane>
SIMD_ATTR_AVX2 SIMD_INLINE i64x4 broadcast(const i64x4 v) {
  static_assert(0 <= kLane && kLane < 2, "Invalid lane");
  return i64x4(_mm256_shuffle_epi32(v, kLane ? 0xEE : 0x44));
}

// Float
template <int kLane>
SIMD_ATTR_AVX2 SIMD_INLINE f32x8 broadcast(const f32x8 v) {
  static_assert(0 <= kLane && kLane < 4, "Invalid lane");
  return f32x8(_mm256_shuffle_ps(v, v, 0x55 * kLane));
}
template <int kLane>
SIMD_ATTR_AVX2 SIMD_INLINE f64x4 broadcast(const f64x4 v) {
  static_assert(0 <= kLane && kLane < 2, "Invalid lane");
  return f64x4(_mm256_shuffle_pd(v, v, 15 * kLane));
}

// ------------------------------ Shuffle bytes with variable indices

// Returns vector of bytes[from[i]]. "from" is also interpreted as bytes:
// either valid indices in [0, 16) or >= 0x80 to zero the i-th output byte.
template <typename T, typename TI, size_t N, size_t NI>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<T, N> shuffle_bytes(
    const vec_avx2<T, N> bytes, const vec_avx2<TI, NI> from) {
  return vec_avx2<T, N>(_mm256_shuffle_epi8(bytes, from));
}

// ------------------------------ Hard-coded shuffles

// Notation: let i32x8 have lanes 7,6,5,4,3,2,1,0 (0 is least-significant).
// shuffle_0321 rotates four-lane blocks one lane to the right (the previous
// least-significant lane is now most-significant => 47650321). These could
// also be implemented via extract_concat_bytes but the shuffle_abcd notation
// is more convenient.

// Swap 64-bit halves
SIMD_ATTR_AVX2 SIMD_INLINE u32x8 shuffle_1032(const u32x8 v) {
  return u32x8(_mm256_shuffle_epi32(v, 0x4E));
}
SIMD_ATTR_AVX2 SIMD_INLINE i32x8 shuffle_1032(const i32x8 v) {
  return i32x8(_mm256_shuffle_epi32(v, 0x4E));
}
SIMD_ATTR_AVX2 SIMD_INLINE f32x8 shuffle_1032(const f32x8 v) {
  return f32x8(_mm256_shuffle_ps(v, v, 0x4E));
}
SIMD_ATTR_AVX2 SIMD_INLINE u64x4 shuffle_01(const u64x4 v) {
  return u64x4(_mm256_shuffle_epi32(v, 0x4E));
}
SIMD_ATTR_AVX2 SIMD_INLINE i64x4 shuffle_01(const i64x4 v) {
  return i64x4(_mm256_shuffle_epi32(v, 0x4E));
}
SIMD_ATTR_AVX2 SIMD_INLINE f64x4 shuffle_01(const f64x4 v) {
  return f64x4(_mm256_shuffle_pd(v, v, 5));
}

// Rotate right 32 bits
SIMD_ATTR_AVX2 SIMD_INLINE u32x8 shuffle_0321(const u32x8 v) {
  return u32x8(_mm256_shuffle_epi32(v, 0x39));
}
SIMD_ATTR_AVX2 SIMD_INLINE i32x8 shuffle_0321(const i32x8 v) {
  return i32x8(_mm256_shuffle_epi32(v, 0x39));
}
SIMD_ATTR_AVX2 SIMD_INLINE f32x8 shuffle_0321(const f32x8 v) {
  return f32x8(_mm256_shuffle_ps(v, v, 0x39));
}
// Rotate left 32 bits
SIMD_ATTR_AVX2 SIMD_INLINE u32x8 shuffle_2103(const u32x8 v) {
  return u32x8(_mm256_shuffle_epi32(v, 0x93));
}
SIMD_ATTR_AVX2 SIMD_INLINE i32x8 shuffle_2103(const i32x8 v) {
  return i32x8(_mm256_shuffle_epi32(v, 0x93));
}
SIMD_ATTR_AVX2 SIMD_INLINE f32x8 shuffle_2103(const f32x8 v) {
  return f32x8(_mm256_shuffle_ps(v, v, 0x93));
}

// ------------------------------ Interleave lanes

// Interleaves lanes from halves of the 128-bit blocks of "a" (which provides
// the least-significant lane) and "b". To concatenate two half-width integers
// into one, use zip_lo/hi instead (also works with scalar).

SIMD_ATTR_AVX2 SIMD_INLINE u8x32 interleave_lo(const u8x32 a, const u8x32 b) {
  return u8x32(_mm256_unpacklo_epi8(a, b));
}
SIMD_ATTR_AVX2 SIMD_INLINE u16x16 interleave_lo(const u16x16 a, const u16x16 b) {
  return u16x16(_mm256_unpacklo_epi16(a, b));
}
SIMD_ATTR_AVX2 SIMD_INLINE u32x8 interleave_lo(const u32x8 a, const u32x8 b) {
  return u32x8(_mm256_unpacklo_epi32(a, b));
}
SIMD_ATTR_AVX2 SIMD_INLINE u64x4 interleave_lo(const u64x4 a, const u64x4 b) {
  return u64x4(_mm256_unpacklo_epi64(a, b));
}

SIMD_ATTR_AVX2 SIMD_INLINE i8x32 interleave_lo(const i8x32 a, const i8x32 b) {
  return i8x32(_mm256_unpacklo_epi8(a, b));
}
SIMD_ATTR_AVX2 SIMD_INLINE i16x16 interleave_lo(const i16x16 a, const i16x16 b) {
  return i16x16(_mm256_unpacklo_epi16(a, b));
}
SIMD_ATTR_AVX2 SIMD_INLINE i32x8 interleave_lo(const i32x8 a, const i32x8 b) {
  return i32x8(_mm256_unpacklo_epi32(a, b));
}
SIMD_ATTR_AVX2 SIMD_INLINE i64x4 interleave_lo(const i64x4 a, const i64x4 b) {
  return i64x4(_mm256_unpacklo_epi64(a, b));
}

SIMD_ATTR_AVX2 SIMD_INLINE f32x8 interleave_lo(const f32x8 a, const f32x8 b) {
  return f32x8(_mm256_unpacklo_ps(a, b));
}
SIMD_ATTR_AVX2 SIMD_INLINE f64x4 interleave_lo(const f64x4 a, const f64x4 b) {
  return f64x4(_mm256_unpacklo_pd(a, b));
}

SIMD_ATTR_AVX2 SIMD_INLINE u8x32 interleave_hi(const u8x32 a, const u8x32 b) {
  return u8x32(_mm256_unpackhi_epi8(a, b));
}
SIMD_ATTR_AVX2 SIMD_INLINE u16x16 interleave_hi(const u16x16 a, const u16x16 b) {
  return u16x16(_mm256_unpackhi_epi16(a, b));
}
SIMD_ATTR_AVX2 SIMD_INLINE u32x8 interleave_hi(const u32x8 a, const u32x8 b) {
  return u32x8(_mm256_unpackhi_epi32(a, b));
}
SIMD_ATTR_AVX2 SIMD_INLINE u64x4 interleave_hi(const u64x4 a, const u64x4 b) {
  return u64x4(_mm256_unpackhi_epi64(a, b));
}

SIMD_ATTR_AVX2 SIMD_INLINE i8x32 interleave_hi(const i8x32 a, const i8x32 b) {
  return i8x32(_mm256_unpackhi_epi8(a, b));
}
SIMD_ATTR_AVX2 SIMD_INLINE i16x16 interleave_hi(const i16x16 a, const i16x16 b) {
  return i16x16(_mm256_unpackhi_epi16(a, b));
}
SIMD_ATTR_AVX2 SIMD_INLINE i32x8 interleave_hi(const i32x8 a, const i32x8 b) {
  return i32x8(_mm256_unpackhi_epi32(a, b));
}
SIMD_ATTR_AVX2 SIMD_INLINE i64x4 interleave_hi(const i64x4 a, const i64x4 b) {
  return i64x4(_mm256_unpackhi_epi64(a, b));
}

SIMD_ATTR_AVX2 SIMD_INLINE f32x8 interleave_hi(const f32x8 a, const f32x8 b) {
  return f32x8(_mm256_unpackhi_ps(a, b));
}
SIMD_ATTR_AVX2 SIMD_INLINE f64x4 interleave_hi(const f64x4 a, const f64x4 b) {
  return f64x4(_mm256_unpackhi_pd(a, b));
}

// ------------------------------ Zip lanes

// Same as interleave_*, except that the return lanes are double-width integers;
// this is necessary because the single-lane scalar cannot return two values.

SIMD_ATTR_AVX2 SIMD_INLINE u16x16 zip_lo(const u8x32 a, const u8x32 b) {
  return u16x16(_mm256_unpacklo_epi8(a, b));
}
SIMD_ATTR_AVX2 SIMD_INLINE u32x8 zip_lo(const u16x16 a, const u16x16 b) {
  return u32x8(_mm256_unpacklo_epi16(a, b));
}
SIMD_ATTR_AVX2 SIMD_INLINE u64x4 zip_lo(const u32x8 a, const u32x8 b) {
  return u64x4(_mm256_unpacklo_epi32(a, b));
}

SIMD_ATTR_AVX2 SIMD_INLINE i16x16 zip_lo(const i8x32 a, const i8x32 b) {
  return i16x16(_mm256_unpacklo_epi8(a, b));
}
SIMD_ATTR_AVX2 SIMD_INLINE i32x8 zip_lo(const i16x16 a, const i16x16 b) {
  return i32x8(_mm256_unpacklo_epi16(a, b));
}
SIMD_ATTR_AVX2 SIMD_INLINE i64x4 zip_lo(const i32x8 a, const i32x8 b) {
  return i64x4(_mm256_unpacklo_epi32(a, b));
}

SIMD_ATTR_AVX2 SIMD_INLINE u16x16 zip_hi(const u8x32 a, const u8x32 b) {
  return u16x16(_mm256_unpackhi_epi8(a, b));
}
SIMD_ATTR_AVX2 SIMD_INLINE u32x8 zip_hi(const u16x16 a, const u16x16 b) {
  return u32x8(_mm256_unpackhi_epi16(a, b));
}
SIMD_ATTR_AVX2 SIMD_INLINE u64x4 zip_hi(const u32x8 a, const u32x8 b) {
  return u64x4(_mm256_unpackhi_epi32(a, b));
}

SIMD_ATTR_AVX2 SIMD_INLINE i16x16 zip_hi(const i8x32 a, const i8x32 b) {
  return i16x16(_mm256_unpackhi_epi8(a, b));
}
SIMD_ATTR_AVX2 SIMD_INLINE i32x8 zip_hi(const i16x16 a, const i16x16 b) {
  return i32x8(_mm256_unpackhi_epi16(a, b));
}
SIMD_ATTR_AVX2 SIMD_INLINE i64x4 zip_hi(const i32x8 a, const i32x8 b) {
  return i64x4(_mm256_unpackhi_epi32(a, b));
}

// ------------------------------ Promotions (part w/ narrow lanes -> full)

// Unsigned: zero-extend.
// Note: these have 3 cycle latency; if inputs are already split across the
// 128 bit blocks (in their upper/lower halves), then zip_hi/lo would be faster.
SIMD_ATTR_AVX2 SIMD_INLINE u16x16 convert_to(Full<uint16_t, AVX2>,
                                             const u8x16 v) {
  return u16x16(_mm256_cvtepu8_epi16(v));
}
SIMD_ATTR_AVX2 SIMD_INLINE u32x8 convert_to(Full<uint32_t, AVX2>,
                                            const u8x8 v) {
  return u32x8(_mm256_cvtepu8_epi32(v));
}
SIMD_ATTR_AVX2 SIMD_INLINE i16x16 convert_to(Full<int16_t, AVX2>,
                                             const u8x16 v) {
  return i16x16(_mm256_cvtepu8_epi16(v));
}
SIMD_ATTR_AVX2 SIMD_INLINE i32x8 convert_to(Full<int32_t, AVX2>, const u8x8 v) {
  return i32x8(_mm256_cvtepu8_epi32(v));
}
SIMD_ATTR_AVX2 SIMD_INLINE u32x8 convert_to(Full<uint32_t, AVX2>,
                                            const u16x8 v) {
  return u32x8(_mm256_cvtepu16_epi32(v));
}
SIMD_ATTR_AVX2 SIMD_INLINE i32x8 convert_to(Full<int32_t, AVX2>,
                                            const u16x8 v) {
  return i32x8(_mm256_cvtepu16_epi32(v));
}
SIMD_ATTR_AVX2 SIMD_INLINE u64x4 convert_to(Full<uint64_t, AVX2>,
                                            const u32x4 v) {
  return u64x4(_mm256_cvtepu32_epi64(v));
}

// Signed: replicate sign bit.
// Note: these have 3 cycle latency; if inputs are already split across the
// 128 bit blocks (in their upper/lower halves), then zip_hi/lo followed by
// signed shift would be faster.
SIMD_ATTR_AVX2 SIMD_INLINE i16x16 convert_to(Full<int16_t, AVX2>,
                                             const i8x16 v) {
  return i16x16(_mm256_cvtepi8_epi16(v));
}
SIMD_ATTR_AVX2 SIMD_INLINE i32x8 convert_to(Full<int32_t, AVX2>, const i8x8 v) {
  return i32x8(_mm256_cvtepi8_epi32(v));
}
SIMD_ATTR_AVX2 SIMD_INLINE i32x8 convert_to(Full<int32_t, AVX2>,
                                            const i16x8 v) {
  return i32x8(_mm256_cvtepi16_epi32(v));
}
SIMD_ATTR_AVX2 SIMD_INLINE i64x4 convert_to(Full<int64_t, AVX2>,
                                            const i32x4 v) {
  return i64x4(_mm256_cvtepi32_epi64(v));
}

// ------------------------------ Demotions (full -> part w/ narrow lanes)

SIMD_ATTR_AVX2 SIMD_INLINE u16x8 convert_to(Part<uint16_t, 8>, const i32x8 v) {
  const __m256i u16 = _mm256_packus_epi32(v, v);
  // Concatenating lower halves of both 128-bit blocks afterward is more
  // efficient than an extra input with low block = high block of v.
  return u16x8(_mm256_castsi256_si128(_mm256_permute4x64_epi64(u16, 0x88)));
}

SIMD_ATTR_AVX2 SIMD_INLINE u8x8 convert_to(Part<uint8_t, 8>, const i32x8 v) {
  const __m256i u16_blocks = _mm256_packus_epi32(v, v);
  // Concatenate lower 64 bits of each 128-bit block
  const __m256i u16_concat = _mm256_permute4x64_epi64(u16_blocks, 0x88);
  const __m128i u16 = _mm256_castsi256_si128(u16_concat);
  return u8x8(_mm_packus_epi16(u16, u16));
}

SIMD_ATTR_AVX2 SIMD_INLINE i16x8 convert_to(Part<int16_t, 8>, const i32x8 v) {
  const __m256i i16 = _mm256_packs_epi32(v, v);
  return i16x8(_mm256_castsi256_si128(_mm256_permute4x64_epi64(i16, 0x88)));
}

SIMD_ATTR_AVX2 SIMD_INLINE i8x8 convert_to(Part<int8_t, 8>, const i32x8 v) {
  const __m256i i16_blocks = _mm256_packs_epi32(v, v);
  // Concatenate lower 64 bits of each 128-bit block
  const __m256i i16_concat = _mm256_permute4x64_epi64(i16_blocks, 0x88);
  const __m128i i16 = _mm256_castsi256_si128(i16_concat);
  return i8x8(_mm_packs_epi16(i16, i16));
}

SIMD_ATTR_AVX2 SIMD_INLINE u8x16 convert_to(Part<uint8_t, 16>, const i16x16 v) {
  const __m256i u8 = _mm256_packus_epi16(v, v);
  return u8x16(_mm256_castsi256_si128(_mm256_permute4x64_epi64(u8, 0x88)));
}

SIMD_ATTR_AVX2 SIMD_INLINE i8x16 convert_to(Part<int8_t, 16>, const i16x16 v) {
  const __m256i i8 = _mm256_packs_epi16(v, v);
  return i8x16(_mm256_castsi256_si128(_mm256_permute4x64_epi64(i8, 0x88)));
}

// ------------------------------ Select/blend

// Returns mask ? b : a. Due to ARM's semantics, each lane of "mask" must
// equal T(0) or ~T(0) although x86 may only check the most significant bit.
template <typename T, size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE vec_avx2<T, N> select(const vec_avx2<T, N> a,
                                                 const vec_avx2<T, N> b,
                                                 const vec_avx2<T, N> mask) {
  return vec_avx2<T, N>(_mm256_blendv_epi8(a, b, mask));
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE f32x8 select<float>(const f32x8 a, const f32x8 b,
                                               const f32x8 mask) {
  return f32x8(_mm256_blendv_ps(a, b, mask));
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE f64x4 select<double>(const f64x4 a, const f64x4 b,
                                                const f64x4 mask) {
  return f64x4(_mm256_blendv_pd(a, b, mask));
}

// aes_round already defined by x86_sse4.h.

// TODO(janwas): wrappers for all intrinsics (in x86 namespace).

#endif  // SIMD_DEPS
