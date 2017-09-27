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

// Primary template for 1-8 byte integer lanes.
template <typename T>
struct Raw128 {
  using type = __m128i;
};

template <>
struct Raw128<float> {
  using type = __m128;
};

template <>
struct Raw128<double> {
  using type = __m128d;
};

template <typename Lane>
class vec128 {
  using Raw = typename Raw128<Lane>::type;

 public:
  using T = Lane;

  SIMD_ATTR_SSE4 SIMD_INLINE vec128() {}
  vec128(const vec128&) = default;
  vec128& operator=(const vec128&) = default;
  SIMD_ATTR_SSE4 SIMD_INLINE explicit vec128(const Raw v) : v_(v) {}

  // Used by non-member functions; avoids verbose .raw() for each argument and
  // reduces Clang compile time of the test by 10-20%.
  SIMD_ATTR_SSE4 SIMD_INLINE operator Raw() const { return v_; }  // NOLINT

  // Compound assignment. Only usable if there is a corresponding non-member
  // binary operator overload. For example, only f32 and f64 support division.
  SIMD_ATTR_SSE4 SIMD_INLINE vec128& operator*=(const vec128 other) {
    return *this = (*this * other);
  }
  SIMD_ATTR_SSE4 SIMD_INLINE vec128& operator/=(const vec128 other) {
    return *this = (*this / other);
  }
  SIMD_ATTR_SSE4 SIMD_INLINE vec128& operator+=(const vec128 other) {
    return *this = (*this + other);
  }
  SIMD_ATTR_SSE4 SIMD_INLINE vec128& operator-=(const vec128 other) {
    return *this = (*this - other);
  }
  SIMD_ATTR_SSE4 SIMD_INLINE vec128& operator&=(const vec128 other) {
    return *this = (*this & other);
  }
  SIMD_ATTR_SSE4 SIMD_INLINE vec128& operator|=(const vec128 other) {
    return *this = (*this | other);
  }
  SIMD_ATTR_SSE4 SIMD_INLINE vec128& operator^=(const vec128 other) {
    return *this = (*this ^ other);
  }
  template <typename ShiftArg>  // int or vec128
  SIMD_ATTR_SSE4 SIMD_INLINE vec128& operator<<=(const ShiftArg count) {
    return *this = operator<<(*this, count);
  }
  template <typename ShiftArg>
  SIMD_ATTR_SSE4 SIMD_INLINE vec128& operator>>=(const ShiftArg count) {
    return *this = operator>>(*this, count);
  }

 private:
  Raw v_;
};

template <typename Lane>
struct IsVec<vec128<Lane>> {
  static constexpr bool value = true;
};

using u8x16 = vec128<uint8_t>;
using u16x8 = vec128<uint16_t>;
using u32x4 = vec128<uint32_t>;
using u64x2 = vec128<uint64_t>;

using i8x16 = vec128<int8_t>;
using i16x8 = vec128<int16_t>;
using i32x4 = vec128<int32_t>;
using i64x2 = vec128<int64_t>;

using f32x4 = vec128<float>;
using f64x2 = vec128<double>;

// ------------------------------ Set

// Returns an all-zero vector.
template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE vec128<T> setzero(vec128<T>) {
  return vec128<T>(_mm_setzero_si128());
}
template <>
SIMD_ATTR_SSE4 SIMD_INLINE f32x4 setzero(f32x4) {
  return f32x4(_mm_setzero_ps());
}
template <>
SIMD_ATTR_SSE4 SIMD_INLINE f64x2 setzero(f64x2) {
  return f64x2(_mm_setzero_pd());
}

// Returns a vector with all lanes set to "t".
SIMD_ATTR_SSE4 SIMD_INLINE u8x16 set1(u8x16, const uint8_t t) {
  return u8x16(_mm_set1_epi8(t));
}
SIMD_ATTR_SSE4 SIMD_INLINE u16x8 set1(u16x8, const uint16_t t) {
  return u16x8(_mm_set1_epi16(t));
}
SIMD_ATTR_SSE4 SIMD_INLINE u32x4 set1(u32x4, const uint32_t t) {
  return u32x4(_mm_set1_epi32(t));
}
SIMD_ATTR_SSE4 SIMD_INLINE u64x2 set1(u64x2, const uint64_t t) {
  return u64x2(_mm_set1_epi64x(t));
}
SIMD_ATTR_SSE4 SIMD_INLINE i8x16 set1(i8x16, const int8_t t) {
  return i8x16(_mm_set1_epi8(t));
}
SIMD_ATTR_SSE4 SIMD_INLINE i16x8 set1(i16x8, const int16_t t) {
  return i16x8(_mm_set1_epi16(t));
}
SIMD_ATTR_SSE4 SIMD_INLINE i32x4 set1(i32x4, const int32_t t) {
  return i32x4(_mm_set1_epi32(t));
}
SIMD_ATTR_SSE4 SIMD_INLINE i64x2 set1(i64x2, const int64_t t) {
  return i64x2(_mm_set1_epi64x(t));
}
SIMD_ATTR_SSE4 SIMD_INLINE f32x4 set1(f32x4, const float t) {
  return f32x4(_mm_set1_ps(t));
}
SIMD_ATTR_SSE4 SIMD_INLINE f64x2 set1(f64x2, const double t) {
  return f64x2(_mm_set1_pd(t));
}

// ------------------------------ Half

// Mainly used for promote/demote; can also load/store.
template <typename Lane>
class half128 {
  using Raw = typename Raw128<Lane>::type;

 public:
  using T = Lane;

  SIMD_ATTR_SSE4 SIMD_INLINE half128() {}
  half128(const half128&) = default;
  half128& operator=(const half128&) = default;
  SIMD_ATTR_SSE4 SIMD_INLINE explicit half128(const Raw v) : v_(v) {}

  SIMD_ATTR_SSE4 SIMD_INLINE operator Raw() const { return v_; }  // NOLINT

 private:
  Raw v_;
};

template <class T>
struct HalfT<vec128<T>> {
  using type = half128<T>;
};

template <typename Lane>
struct NumLanes<half128<Lane>> {
  static constexpr size_t value = sizeof(half128<Lane>) / sizeof(Lane) / 2;
  static_assert(value != 0, "NumLanes cannot be zero");
  constexpr operator size_t() const { return value; }
};

using u8x8 = half128<uint8_t>;
using u16x4 = half128<uint16_t>;
using u32x2 = half128<uint32_t>;
// No typedefs for 64-bit lanes because they would conflict with scalar.h.

using i8x8 = half128<int8_t>;
using i16x4 = half128<int16_t>;
using i32x2 = half128<int32_t>;

using f32x2 = half128<float>;

template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE half128<T> setzero(half128<T>) {
  return half128<T>(setzero(vec128<T>()));
}

template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE half128<T> set1(half128<T>, const T t) {
  return half128<T>(set1(vec128<T>(), t));
}

template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE half128<T> lower_half(const vec128<T> v) {
  return half128<T>(v);
}

template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE half128<T> upper_half(const vec128<T> v) {
  return half128<T>(_mm_unpackhi_epi64(v, v));
}
SIMD_ATTR_SSE4 SIMD_INLINE f32x2 upper_half(const f32x4 v) {
  return f32x2(_mm_movehl_ps(v, v));
}
SIMD_ATTR_SSE4 SIMD_INLINE half128<double> upper_half(const f64x2 v) {
  return half128<double>(_mm_unpackhi_pd(v, v));
}

// Returns vec128 with undefined values in the upper half.
template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE vec128<T> from_half(const half128<T> v) {
  return vec128<T>(v);
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
SIMD_ATTR_SSE4 SIMD_INLINE u16x8 operator<<(const u16x8 v, const int bits) {
  return u16x8(_mm_slli_epi16(v, bits));
}
SIMD_ATTR_SSE4 SIMD_INLINE u16x8 operator>>(const u16x8 v, const int bits) {
  return u16x8(_mm_srli_epi16(v, bits));
}
SIMD_ATTR_SSE4 SIMD_INLINE u32x4 operator<<(const u32x4 v, const int bits) {
  return u32x4(_mm_slli_epi32(v, bits));
}
SIMD_ATTR_SSE4 SIMD_INLINE u32x4 operator>>(const u32x4 v, const int bits) {
  return u32x4(_mm_srli_epi32(v, bits));
}
SIMD_ATTR_SSE4 SIMD_INLINE u64x2 operator<<(const u64x2 v, const int bits) {
  return u64x2(_mm_slli_epi64(v, bits));
}
SIMD_ATTR_SSE4 SIMD_INLINE u64x2 operator>>(const u64x2 v, const int bits) {
  return u64x2(_mm_srli_epi64(v, bits));
}

// Signed (no i64 shr)
SIMD_ATTR_SSE4 SIMD_INLINE i16x8 operator<<(const i16x8 v, const int bits) {
  return i16x8(_mm_slli_epi16(v, bits));
}
SIMD_ATTR_SSE4 SIMD_INLINE i16x8 operator>>(const i16x8 v, const int bits) {
  return i16x8(_mm_srai_epi16(v, bits));
}
SIMD_ATTR_SSE4 SIMD_INLINE i32x4 operator<<(const i32x4 v, const int bits) {
  return i32x4(_mm_slli_epi32(v, bits));
}
SIMD_ATTR_SSE4 SIMD_INLINE i32x4 operator>>(const i32x4 v, const int bits) {
  return i32x4(_mm_srai_epi32(v, bits));
}
SIMD_ATTR_SSE4 SIMD_INLINE i64x2 operator<<(const i64x2 v, const int bits) {
  return i64x2(_mm_slli_epi64(v, bits));
}

// ------------------------------ Shift lanes by independent variable #bits

#if SIMD_X86_AVX2

// Unsigned (no u8,u16)
SIMD_ATTR_SSE4 SIMD_INLINE u32x4 operator<<(const u32x4 v, const u32x4 bits) {
  return u32x4(_mm_sllv_epi32(v, bits));
}
SIMD_ATTR_SSE4 SIMD_INLINE u32x4 operator>>(const u32x4 v, const u32x4 bits) {
  return u32x4(_mm_srlv_epi32(v, bits));
}
SIMD_ATTR_SSE4 SIMD_INLINE u64x2 operator<<(const u64x2 v, const u64x2 bits) {
  return u64x2(_mm_sllv_epi64(v, bits));
}
SIMD_ATTR_SSE4 SIMD_INLINE u64x2 operator>>(const u64x2 v, const u64x2 bits) {
  return u64x2(_mm_srlv_epi64(v, bits));
}

// Signed (no i8,i16,i64)
SIMD_ATTR_SSE4 SIMD_INLINE i32x4 operator<<(const i32x4 v, const i32x4 bits) {
  return i32x4(_mm_sllv_epi32(v, bits));
}
SIMD_ATTR_SSE4 SIMD_INLINE i32x4 operator>>(const i32x4 v, const i32x4 bits) {
  return i32x4(_mm_srav_epi32(v, bits));
}
SIMD_ATTR_SSE4 SIMD_INLINE i64x2 operator<<(const i64x2 v, const i64x2 bits) {
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
SIMD_ATTR_SSE4 SIMD_INLINE i32x4 i32_from_f32(const f32x4 v) {
  return i32x4(_mm_cvtps_epi32(v));
}

// ------------------------------ Cast to/from floating-point representation

SIMD_ATTR_SSE4 SIMD_INLINE f32x4 f32_from_bits(const i32x4 v) {
  return f32x4(_mm_castsi128_ps(v));
}
SIMD_ATTR_SSE4 SIMD_INLINE i32x4 bits_from_f32(const f32x4 v) {
  return i32x4(_mm_castps_si128(v));
}

SIMD_ATTR_SSE4 SIMD_INLINE f64x2 f64_from_bits(const i64x2 v) {
  return f64x2(_mm_castsi128_pd(v));
}
SIMD_ATTR_SSE4 SIMD_INLINE i64x2 bits_from_f64(const f64x2 v) {
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
SIMD_ATTR_SSE4 SIMD_INLINE bool all_zero(const vec128<T> v) {
  return static_cast<bool>(_mm_testz_si128(v, v));
}

}  // namespace ext

// ================================================== LOGICAL

// ------------------------------ Bitwise AND

template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE vec128<T> operator&(const vec128<T> a,
                                               const vec128<T> b) {
  return vec128<T>(_mm_and_si128(a, b));
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
template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE vec128<T> andnot(const vec128<T> not_mask,
                                            const vec128<T> mask) {
  return vec128<T>(_mm_andnot_si128(not_mask, mask));
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

template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE vec128<T> operator|(const vec128<T> a,
                                               const vec128<T> b) {
  return vec128<T>(_mm_or_si128(a, b));
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

template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE vec128<T> operator^(const vec128<T> a,
                                               const vec128<T> b) {
  return vec128<T>(_mm_xor_si128(a, b));
}
template <>
SIMD_ATTR_SSE4 SIMD_INLINE f32x4 operator^(const f32x4 a, const f32x4 b) {
  return f32x4(_mm_xor_ps(a, b));
}
template <>
SIMD_ATTR_SSE4 SIMD_INLINE f64x2 operator^(const f64x2 a, const f64x2 b) {
  return f64x2(_mm_xor_pd(a, b));
}

// ================================================== STORE

// ------------------------------ Load all 128

template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE vec128<T> load(vec128<T>,
                                          const T* SIMD_RESTRICT aligned) {
  return vec128<T>(_mm_load_si128(reinterpret_cast<const __m128i*>(aligned)));
}
template <>
SIMD_ATTR_SSE4 SIMD_INLINE f32x4
load<float>(f32x4, const float* SIMD_RESTRICT aligned) {
  return f32x4(_mm_load_ps(aligned));
}
template <>
SIMD_ATTR_SSE4 SIMD_INLINE f64x2
load<double>(f64x2, const double* SIMD_RESTRICT aligned) {
  return f64x2(_mm_load_pd(aligned));
}

template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE vec128<T> load_unaligned(vec128<T>,
                                                    const T* SIMD_RESTRICT p) {
  return vec128<T>(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
}
template <>
SIMD_ATTR_SSE4 SIMD_INLINE f32x4
load_unaligned<float>(f32x4, const float* SIMD_RESTRICT p) {
  return f32x4(_mm_loadu_ps(p));
}
template <>
SIMD_ATTR_SSE4 SIMD_INLINE f64x2
load_unaligned<double>(f64x2, const double* SIMD_RESTRICT p) {
  return f64x2(_mm_loadu_pd(p));
}

// 128-bit SIMD => nothing to duplicate, same as an unaligned load.
template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE vec128<T> load_dup128(
    const vec128<T> v, const T* const SIMD_RESTRICT p) {
  return load_unaligned(v, p);
}

// ------------------------------ Load all 64

template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE half128<T> load_unaligned(half128<T>,
                                                     const T* SIMD_RESTRICT p) {
  return half128<T>(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p)));
}
template <>
SIMD_ATTR_SSE4 SIMD_INLINE f32x2
load_unaligned<float>(f32x2, const float* SIMD_RESTRICT p) {
  const __m128 hi = _mm_setzero_ps();
  return f32x2(_mm_loadl_pi(hi, reinterpret_cast<const __m64*>(p)));
}
template <>
SIMD_ATTR_SSE4 SIMD_INLINE half128<double> load_unaligned<double>(
    half128<double>, const double* SIMD_RESTRICT p) {
  const __m128d hi = _mm_setzero_pd();
  return half128<double>(_mm_loadl_pd(hi, p));
}

template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE half128<T> load(const half128<T> v,
                                           const T* SIMD_RESTRICT p) {
  return load_unaligned(v, p);
}
template <>
SIMD_ATTR_SSE4 SIMD_INLINE f32x2 load<float>(const f32x2 v,
                                             const float* SIMD_RESTRICT p) {
  return load_unaligned(v, p);
}
template <>
SIMD_ATTR_SSE4 SIMD_INLINE half128<double> load<double>(
    const half128<double> v, const double* SIMD_RESTRICT p) {
  return load_unaligned(v, p);
}

// ------------------------------ Store all 128

template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE void store(const vec128<T> v,
                                      T* SIMD_RESTRICT aligned) {
  _mm_store_si128(reinterpret_cast<__m128i*>(aligned), v);
}
template <>
SIMD_ATTR_SSE4 SIMD_INLINE void store<float>(const f32x4 v,
                                             float* SIMD_RESTRICT aligned) {
  _mm_store_ps(aligned, v);
}
template <>
SIMD_ATTR_SSE4 SIMD_INLINE void store<double>(const f64x2 v,
                                              double* SIMD_RESTRICT aligned) {
  _mm_store_pd(aligned, v);
}

template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE void store_unaligned(const vec128<T> v,
                                                T* SIMD_RESTRICT p) {
  _mm_storeu_si128(reinterpret_cast<__m128i*>(p), v);
}
template <>
SIMD_ATTR_SSE4 SIMD_INLINE void store_unaligned<float>(const f32x4 v,
                                                       float* SIMD_RESTRICT p) {
  _mm_storeu_ps(p, v);
}
template <>
SIMD_ATTR_SSE4 SIMD_INLINE void store_unaligned<double>(
    const f64x2 v, double* SIMD_RESTRICT p) {
  _mm_storeu_pd(p, v);
}

// ------------------------------ Store all 64

template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE void store_unaligned(const half128<T> v,
                                                T* SIMD_RESTRICT p) {
  _mm_storel_epi64(reinterpret_cast<__m128i*>(p), v);
}
template <>
SIMD_ATTR_SSE4 SIMD_INLINE void store_unaligned<float>(const f32x2 v,
                                                       float* SIMD_RESTRICT p) {
  _mm_storel_pi(reinterpret_cast<__m64*>(p), v);
}
template <>
SIMD_ATTR_SSE4 SIMD_INLINE void store_unaligned<double>(
    const half128<double> v, double* SIMD_RESTRICT p) {
  _mm_storel_pd(p, v);
}

template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE void store(const half128<T> v, T* SIMD_RESTRICT p) {
  store_unaligned(v, p);
}
template <>
SIMD_ATTR_SSE4 SIMD_INLINE void store<float>(const f32x2 v,
                                             float* SIMD_RESTRICT p) {
  store_unaligned(v, p);
}
template <>
SIMD_ATTR_SSE4 SIMD_INLINE void store<double>(const half128<double> v,
                                              double* SIMD_RESTRICT p) {
  store_unaligned(v, p);
}

// ------------------------------ Non-temporal stores

// Same as aligned stores on non-x86.

template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE void stream(const vec128<T> v,
                                       T* SIMD_RESTRICT aligned) {
  _mm_stream_si128(reinterpret_cast<__m128i*>(aligned), v);
}
template <>
SIMD_ATTR_SSE4 SIMD_INLINE void stream<float>(const f32x4 v,
                                              float* SIMD_RESTRICT aligned) {
  _mm_stream_ps(aligned, v);
}
template <>
SIMD_ATTR_SSE4 SIMD_INLINE void stream<double>(const f64x2 v,
                                               double* SIMD_RESTRICT aligned) {
  _mm_stream_pd(aligned, v);
}

SIMD_ATTR_SSE4 SIMD_INLINE void stream32(const uint32_t t,
                                         uint32_t* SIMD_RESTRICT aligned) {
  _mm_stream_si32(reinterpret_cast<int*>(aligned), t);
}

SIMD_ATTR_SSE4 SIMD_INLINE void stream64(const uint64_t t,
                                         uint64_t* SIMD_RESTRICT aligned) {
  _mm_stream_si64(reinterpret_cast<long long*>(aligned), t);
}

// Ensures previous weakly-ordered stores are visible. No effect on non-x86.
SIMD_ATTR_SSE4 SIMD_INLINE void store_fence() { _mm_sfence(); }

// ------------------------------ Cache control

// Begins loading the cache line containing "p". No effect on non-x86.
template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE void prefetch(const T* p) {
  _mm_prefetch(p, _MM_HINT_T0);
}

// Invalidates and flushes the cache line containing "p". No effect on non-x86.
SIMD_ATTR_SSE4 SIMD_INLINE void flush_cacheline(const void* p) {
  _mm_clflush(p);
}

// ================================================== SWIZZLE

// ------------------------------ Shift vector by constant #bytes

// 0x01..0F, kBytes = 1 => 0x02..0F00
template <int kBytes, typename T>
SIMD_ATTR_SSE4 SIMD_INLINE vec128<T> shift_bytes_left(const vec128<T> v) {
  return vec128<T>(_mm_slli_si128(v, kBytes));
}

// 0x01..0F, kBytes = 1 => 0x0001..0E
template <int kBytes, typename T>
SIMD_ATTR_SSE4 SIMD_INLINE vec128<T> shift_bytes_right(const vec128<T> v) {
  return vec128<T>(_mm_srli_si128(v, kBytes));
}

// ------------------------------ Extract from 2x 128-bit at constant offset

// Extracts 128 bits from <hi, lo> by skipping the least-significant kBytes.
template <int kBytes, typename T>
SIMD_ATTR_SSE4 SIMD_INLINE vec128<T> extract_concat_bytes(const vec128<T> hi,
                                                          const vec128<T> lo) {
  return vec128<T>(_mm_alignr_epi8(hi, lo, kBytes));
}

// ------------------------------ Get/set least-significant lane

template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE T get_low(const vec128<T> v) {
  return static_cast<T>(_mm_cvtsi128_si64(v));
}
template <>
SIMD_ATTR_SSE4 SIMD_INLINE float get_low<float>(const f32x4 v) {
  return _mm_cvtss_f32(v);
}
template <>
SIMD_ATTR_SSE4 SIMD_INLINE double get_low<double>(const f64x2 v) {
  return _mm_cvtsd_f64(v);
}

// Sets least-significant lane and zero-fills the other lanes.
template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE vec128<T> set_low(vec128<T>, const T t) {
  return vec128<T>(_mm_cvtsi64_si128(t));
}
SIMD_ATTR_SSE4 SIMD_INLINE f32x4 set_low(f32x4, const float t) {
  return f32x4(_mm_load_ss(&t));
}
SIMD_ATTR_SSE4 SIMD_INLINE f64x2 set_low(f64x2, const double t) {
  return f64x2(_mm_load_sd(&t));
}

// ------------------------------ Broadcast/splat any lane

// Unsigned
template <int kLane>
SIMD_ATTR_SSE4 SIMD_INLINE u32x4 broadcast(const u32x4 v) {
  static_assert(0 <= kLane && kLane < NumLanes<u32x4>(), "Invalid lane");
  return u32x4(_mm_shuffle_epi32(v, 0x55 * kLane));
}
template <int kLane>
SIMD_ATTR_SSE4 SIMD_INLINE u64x2 broadcast(const u64x2 v) {
  static_assert(0 <= kLane && kLane < NumLanes<u64x2>(), "Invalid lane");
  return u64x2(_mm_shuffle_epi32(v, kLane ? 0xEE : 0x44));
}

// Signed
template <int kLane>
SIMD_ATTR_SSE4 SIMD_INLINE i32x4 broadcast(const i32x4 v) {
  static_assert(0 <= kLane && kLane < NumLanes<i32x4>(), "Invalid lane");
  return i32x4(_mm_shuffle_epi32(v, 0x55 * kLane));
}
template <int kLane>
SIMD_ATTR_SSE4 SIMD_INLINE i64x2 broadcast(const i64x2 v) {
  static_assert(0 <= kLane && kLane < NumLanes<i64x2>(), "Invalid lane");
  return i64x2(_mm_shuffle_epi32(v, kLane ? 0xEE : 0x44));
}

// Float
template <int kLane>
SIMD_ATTR_SSE4 SIMD_INLINE f32x4 broadcast(const f32x4 v) {
  static_assert(0 <= kLane && kLane < NumLanes<f32x4>(), "Invalid lane");
  return f32x4(_mm_shuffle_ps(v, v, 0x55 * kLane));
}
template <int kLane>
SIMD_ATTR_SSE4 SIMD_INLINE f64x2 broadcast(const f64x2 v) {
  static_assert(0 <= kLane && kLane < NumLanes<f64x2>(), "Invalid lane");
  return f64x2(_mm_shuffle_pd(v, v, 3 * kLane));
}

// ------------------------------ Shuffle bytes with variable indices

// Returns vector of bytes[from[i]]. "from" must be valid indices in [0, 16).
template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE vec128<T> shuffle_bytes(const vec128<T> bytes,
                                                   const vec128<T> from) {
  return vec128<T>(_mm_shuffle_epi8(bytes, from));
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

// ------------------------------ Zip/interleave/unpack

// Interleaves halves of the 128-bit parts of "a" (starting in the
// least-significant lane) and "b". To zero-extend, use promote() instead.

SIMD_ATTR_SSE4 SIMD_INLINE u8x16 zip_lo(const u8x16 a, const u8x16 b) {
  return u8x16(_mm_unpacklo_epi8(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE u16x8 zip_lo(const u16x8 a, const u16x8 b) {
  return u16x8(_mm_unpacklo_epi16(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE u32x4 zip_lo(const u32x4 a, const u32x4 b) {
  return u32x4(_mm_unpacklo_epi32(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE u64x2 zip_lo(const u64x2 a, const u64x2 b) {
  return u64x2(_mm_unpacklo_epi64(a, b));
}

SIMD_ATTR_SSE4 SIMD_INLINE i8x16 zip_lo(const i8x16 a, const i8x16 b) {
  return i8x16(_mm_unpacklo_epi8(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE i16x8 zip_lo(const i16x8 a, const i16x8 b) {
  return i16x8(_mm_unpacklo_epi16(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE i32x4 zip_lo(const i32x4 a, const i32x4 b) {
  return i32x4(_mm_unpacklo_epi32(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE i64x2 zip_lo(const i64x2 a, const i64x2 b) {
  return i64x2(_mm_unpacklo_epi64(a, b));
}

SIMD_ATTR_SSE4 SIMD_INLINE f32x4 zip_lo(const f32x4 a, const f32x4 b) {
  return f32x4(_mm_unpacklo_ps(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE f64x2 zip_lo(const f64x2 a, const f64x2 b) {
  return f64x2(_mm_unpacklo_pd(a, b));
}

SIMD_ATTR_SSE4 SIMD_INLINE u8x16 zip_hi(const u8x16 a, const u8x16 b) {
  return u8x16(_mm_unpackhi_epi8(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE u16x8 zip_hi(const u16x8 a, const u16x8 b) {
  return u16x8(_mm_unpackhi_epi16(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE u32x4 zip_hi(const u32x4 a, const u32x4 b) {
  return u32x4(_mm_unpackhi_epi32(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE u64x2 zip_hi(const u64x2 a, const u64x2 b) {
  return u64x2(_mm_unpackhi_epi64(a, b));
}

SIMD_ATTR_SSE4 SIMD_INLINE i8x16 zip_hi(const i8x16 a, const i8x16 b) {
  return i8x16(_mm_unpackhi_epi8(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE i16x8 zip_hi(const i16x8 a, const i16x8 b) {
  return i16x8(_mm_unpackhi_epi16(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE i32x4 zip_hi(const i32x4 a, const i32x4 b) {
  return i32x4(_mm_unpackhi_epi32(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE i64x2 zip_hi(const i64x2 a, const i64x2 b) {
  return i64x2(_mm_unpackhi_epi64(a, b));
}

SIMD_ATTR_SSE4 SIMD_INLINE f32x4 zip_hi(const f32x4 a, const f32x4 b) {
  return f32x4(_mm_unpackhi_ps(a, b));
}
SIMD_ATTR_SSE4 SIMD_INLINE f64x2 zip_hi(const f64x2 a, const f64x2 b) {
  return f64x2(_mm_unpackhi_pd(a, b));
}

// ------------------------------ Cast to double-width lane type

// Returns full-vector given half-vector of half-width lanes.

// Unsigned: zero-extend.
SIMD_ATTR_SSE4 SIMD_INLINE u16x8 promote(const u8x8 v) {
  return u16x8(_mm_cvtepu8_epi16(v));
}
SIMD_ATTR_SSE4 SIMD_INLINE u32x4 promote(const u16x4 v) {
  return u32x4(_mm_cvtepu16_epi32(v));
}
SIMD_ATTR_SSE4 SIMD_INLINE u64x2 promote(const u32x2 v) {
  return u64x2(_mm_cvtepu32_epi64(v));
}

// Signed: replicate sign bit.
SIMD_ATTR_SSE4 SIMD_INLINE i16x8 promote(const i8x8 v) {
  return i16x8(_mm_cvtepi8_epi16(v));
}
SIMD_ATTR_SSE4 SIMD_INLINE i32x4 promote(const i16x4 v) {
  return i32x4(_mm_cvtepi16_epi32(v));
}
SIMD_ATTR_SSE4 SIMD_INLINE i64x2 promote(const i32x2 v) {
  return i64x2(_mm_cvtepi32_epi64(v));
}

// ------------------------------ Cast to half-width lane types

// Returns half-vector of half-width lanes.

SIMD_ATTR_SSE4 SIMD_INLINE u8x8 demote_to_unsigned(const i16x8 v) {
  return u8x8(_mm_packus_epi16(v, v));
}
SIMD_ATTR_SSE4 SIMD_INLINE u16x4 demote_to_unsigned(const i32x4 v) {
  return u16x4(_mm_packus_epi32(v, v));
}

SIMD_ATTR_SSE4 SIMD_INLINE i8x8 demote(const i16x8 v) {
  return i8x8(_mm_packs_epi16(v, v));
}
SIMD_ATTR_SSE4 SIMD_INLINE i16x4 demote(const i32x4 v) {
  return i16x4(_mm_packs_epi32(v, v));
}

// ------------------------------ Select/blend

// Returns mask ? b : a. Due to ARM's semantics, each lane of "mask" must
// equal T(0) or ~T(0) although x86 may only check the most significant bit.
template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE vec128<T> select(const vec128<T> a,
                                            const vec128<T> b,
                                            const vec128<T> mask) {
  return vec128<T>(_mm_blendv_epi8(a, b, mask));
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

// "Extensions": useful but quite performance-portable operations. We add
// functions to this namespace in multiple places.
namespace ext {

// Returns 64-bit sums of 8-byte groups.
SIMD_ATTR_SSE4 SIMD_INLINE u64x2 horz_sum(const u8x16 v) {
  return u64x2(_mm_sad_epu8(v, setzero(u8x16())));
}

// Supported for u32x4, i32x4 and f32x4. Returns the sum in each lane.
template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE vec128<T> horz_sum(const vec128<T> v3210) {
  const vec128<T> v1032 = shuffle_1032(v3210);
  const vec128<T> v31_20_31_20 = v3210 + v1032;
  const vec128<T> v20_31_20_31 = shuffle_0321(v31_20_31_20);
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
