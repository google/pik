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

// ================================================== vec256

// Primary template for 1-8 byte integer lanes.
template <typename T>
struct Raw256 {
  using type = __m256i;
};

template <>
struct Raw256<float> {
  using type = __m256;
};

template <>
struct Raw256<double> {
  using type = __m256d;
};

// Returned by load_dup128.
template <typename T>
struct dup128x2 {
  using Raw = typename Raw256<T>::type;
  explicit dup128x2(const Raw v) : raw(v) {}
  Raw raw;
};

template <typename Lane>
class vec256 {
  using Raw = typename Raw256<Lane>::type;

 public:
  using T = Lane;

  SIMD_ATTR_AVX2 SIMD_INLINE vec256() {}
  vec256(const vec256&) = default;
  vec256& operator=(const vec256&) = default;
  SIMD_ATTR_AVX2 SIMD_INLINE explicit vec256(const Raw v) : v_(v) {}

  // Used by non-member functions; avoids verbose .raw() for each argument and
  // reduces Clang compile time of the test by 10-20%.
  SIMD_ATTR_AVX2 SIMD_INLINE operator Raw() const { return v_; }  // NOLINT

  // Compound assignment. Only usable if there is a corresponding non-member
  // binary operator overload. For example, only f32 and f64 support division.
  SIMD_ATTR_AVX2 SIMD_INLINE vec256& operator*=(const vec256 other) {
    return *this = (*this * other);
  }
  SIMD_ATTR_AVX2 SIMD_INLINE vec256& operator/=(const vec256 other) {
    return *this = (*this / other);
  }
  SIMD_ATTR_AVX2 SIMD_INLINE vec256& operator+=(const vec256 other) {
    return *this = (*this + other);
  }
  SIMD_ATTR_AVX2 SIMD_INLINE vec256& operator-=(const vec256 other) {
    return *this = (*this - other);
  }
  SIMD_ATTR_AVX2 SIMD_INLINE vec256& operator&=(const vec256 other) {
    return *this = (*this & other);
  }
  SIMD_ATTR_AVX2 SIMD_INLINE vec256& operator|=(const vec256 other) {
    return *this = (*this | other);
  }
  SIMD_ATTR_AVX2 SIMD_INLINE vec256& operator^=(const vec256 other) {
    return *this = (*this ^ other);
  }
  template <typename ShiftArg>  // int or vec256
  SIMD_ATTR_AVX2 SIMD_INLINE vec256& operator<<=(const ShiftArg count) {
    return *this = operator<<(*this, count);
  }
  template <typename ShiftArg>
  SIMD_ATTR_AVX2 SIMD_INLINE vec256& operator>>=(const ShiftArg count) {
    return *this = operator>>(*this, count);
  }

 private:
  Raw v_;
};

template <typename Lane>
struct IsVec<vec256<Lane>> {
  static constexpr bool value = true;
};

using u8x32 = vec256<uint8_t>;
using u16x16 = vec256<uint16_t>;
using u32x8 = vec256<uint32_t>;
using u64x4 = vec256<uint64_t>;

using i8x32 = vec256<int8_t>;
using i16x16 = vec256<int16_t>;
using i32x8 = vec256<int32_t>;
using i64x4 = vec256<int64_t>;

using f32x8 = vec256<float>;
using f64x4 = vec256<double>;

// ------------------------------ Set

// Returns an all-zero vector.
template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE vec256<T> setzero(vec256<T>) {
  return vec256<T>(_mm256_setzero_si256());
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE f32x8 setzero(f32x8) {
  return f32x8(_mm256_setzero_ps());
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE f64x4 setzero(f64x4) {
  return f64x4(_mm256_setzero_pd());
}

// Returns a vector with all lanes set to "t".
SIMD_ATTR_AVX2 SIMD_INLINE u8x32 set1(u8x32, const uint8_t t) {
  return u8x32(_mm256_set1_epi8(t));
}
SIMD_ATTR_AVX2 SIMD_INLINE u16x16 set1(u16x16, const uint16_t t) {
  return u16x16(_mm256_set1_epi16(t));
}
SIMD_ATTR_AVX2 SIMD_INLINE u32x8 set1(u32x8, const uint32_t t) {
  return u32x8(_mm256_set1_epi32(t));
}
SIMD_ATTR_AVX2 SIMD_INLINE u64x4 set1(u64x4, const uint64_t t) {
  return u64x4(_mm256_set1_epi64x(t));
}
SIMD_ATTR_AVX2 SIMD_INLINE i8x32 set1(i8x32, const int8_t t) {
  return i8x32(_mm256_set1_epi8(t));
}
SIMD_ATTR_AVX2 SIMD_INLINE i16x16 set1(i16x16, const int16_t t) {
  return i16x16(_mm256_set1_epi16(t));
}
SIMD_ATTR_AVX2 SIMD_INLINE i32x8 set1(i32x8, const int32_t t) {
  return i32x8(_mm256_set1_epi32(t));
}
SIMD_ATTR_AVX2 SIMD_INLINE i64x4 set1(i64x4, const int64_t t) {
  return i64x4(_mm256_set1_epi64x(t));
}
template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE f32x8 set1(f32x8, const T t) {
  return f32x8(_mm256_set1_ps(t));
}
template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE f64x4 set1(f64x4, const T t) {
  return f64x4(_mm256_set1_pd(t));
}

// ------------------------------ Cast to/from vector subset (zero-cost)

template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE vec256<T> to_subset(vec256<T>, const vec256<T> v) {
  return v;
}
template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE vec128<T> to_subset(vec128<T>, const vec256<T> v) {
  return vec128<T>(_mm256_castsi256_si128(v));
}
SIMD_ATTR_AVX2 SIMD_INLINE f32x4 to_subset(f32x4, const f32x8 v) {
  return f32x4(_mm256_castps256_ps128(v));
}
SIMD_ATTR_AVX2 SIMD_INLINE f64x2 to_subset(f64x2, const f64x4 v) {
  return f64x2(_mm256_castpd256_pd128(v));
}

template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE vec64<T> to_subset(vec64<T>, const vec256<T> v) {
  return vec64<T>(to_subset(vec128<T>(), v));
}

template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE vec32<T> to_subset(vec32<T>, const vec256<T> v) {
  return vec32<T>(to_subset(vec128<T>(), v));
}

template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE vec256<T> from_subset(vec256<T>, const vec256<T> v) {
  return v;
}
// Upper lane(s) are undefined.
template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE vec256<T> from_subset(vec256<T>, const vec128<T> v) {
  return vec256<T>(_mm256_castsi128_si256(v));
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE f32x8 from_subset(f32x8, const f32x4 v) {
  return f32x8(_mm256_castps128_ps256(v));
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE f64x4 from_subset(f64x4, const f64x2 v) {
  return f64x4(_mm256_castpd128_pd256(v));
}

template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE vec256<T> from_subset(vec256<T>, const vec64<T> v) {
  return from_subset(vec256<T>(), vec128<T>(v));
}

template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE vec256<T> from_subset(vec256<T>, const vec32<T> v) {
  return from_subset(vec256<T>(), vec128<T>(v));
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
SIMD_ATTR_AVX2 SIMD_INLINE u16x16 operator<<(const u16x16 v, const int bits) {
  return u16x16(_mm256_slli_epi16(v, bits));
}
SIMD_ATTR_AVX2 SIMD_INLINE u16x16 operator>>(const u16x16 v, const int bits) {
  return u16x16(_mm256_srli_epi16(v, bits));
}
SIMD_ATTR_AVX2 SIMD_INLINE u32x8 operator<<(const u32x8 v, const int bits) {
  return u32x8(_mm256_slli_epi32(v, bits));
}
SIMD_ATTR_AVX2 SIMD_INLINE u32x8 operator>>(const u32x8 v, const int bits) {
  return u32x8(_mm256_srli_epi32(v, bits));
}
SIMD_ATTR_AVX2 SIMD_INLINE u64x4 operator<<(const u64x4 v, const int bits) {
  return u64x4(_mm256_slli_epi64(v, bits));
}
SIMD_ATTR_AVX2 SIMD_INLINE u64x4 operator>>(const u64x4 v, const int bits) {
  return u64x4(_mm256_srli_epi64(v, bits));
}

// Signed (no i64 shr)
SIMD_ATTR_AVX2 SIMD_INLINE i16x16 operator<<(const i16x16 v, const int bits) {
  return i16x16(_mm256_slli_epi16(v, bits));
}
SIMD_ATTR_AVX2 SIMD_INLINE i16x16 operator>>(const i16x16 v, const int bits) {
  return i16x16(_mm256_srai_epi16(v, bits));
}
SIMD_ATTR_AVX2 SIMD_INLINE i32x8 operator<<(const i32x8 v, const int bits) {
  return i32x8(_mm256_slli_epi32(v, bits));
}
SIMD_ATTR_AVX2 SIMD_INLINE i32x8 operator>>(const i32x8 v, const int bits) {
  return i32x8(_mm256_srai_epi32(v, bits));
}
SIMD_ATTR_AVX2 SIMD_INLINE i64x4 operator<<(const i64x4 v, const int bits) {
  return i64x4(_mm256_slli_epi64(v, bits));
}

// ------------------------------ Shift lanes by independent variable #bits

// Unsigned (no u8,u16)
SIMD_ATTR_AVX2 SIMD_INLINE u32x8 operator<<(const u32x8 v, const u32x8 bits) {
  return u32x8(_mm256_sllv_epi32(v, bits));
}
SIMD_ATTR_AVX2 SIMD_INLINE u32x8 operator>>(const u32x8 v, const u32x8 bits) {
  return u32x8(_mm256_srlv_epi32(v, bits));
}
SIMD_ATTR_AVX2 SIMD_INLINE u64x4 operator<<(const u64x4 v, const u64x4 bits) {
  return u64x4(_mm256_sllv_epi64(v, bits));
}
SIMD_ATTR_AVX2 SIMD_INLINE u64x4 operator>>(const u64x4 v, const u64x4 bits) {
  return u64x4(_mm256_srlv_epi64(v, bits));
}

// Signed (no i8,i16,i64)
SIMD_ATTR_AVX2 SIMD_INLINE i32x8 operator<<(const i32x8 v, const i32x8 bits) {
  return i32x8(_mm256_sllv_epi32(v, bits));
}
SIMD_ATTR_AVX2 SIMD_INLINE i32x8 operator>>(const i32x8 v, const i32x8 bits) {
  return i32x8(_mm256_srav_epi32(v, bits));
}
SIMD_ATTR_AVX2 SIMD_INLINE i64x4 operator<<(const i64x4 v, const i64x4 bits) {
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
SIMD_ATTR_AVX2 SIMD_INLINE u64x4 horz_sum(const u8x32 v) {
  return u64x4(_mm256_sad_epu8(v, setzero(u8x32())));
}

// Supported for {uif}32x8, {uif}64x4. Returns the sum in each lane.
template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE vec128<T> horz_sum(const vec256<T> v) {
  const vec128<T> v0 = to_subset(vec128<T>(), v);
  const vec128<T> v1 = to_other_half(v);
  return horz_sum(v0 + v1);
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
SIMD_ATTR_AVX2 SIMD_INLINE bool all_zero(const vec256<T> v) {
  return static_cast<bool>(_mm256_testz_si256(v, v));
}

}  // namespace ext

// ================================================== LOGICAL

// ------------------------------ Bitwise AND

template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE vec256<T> operator&(const vec256<T> a,
                                               const vec256<T> b) {
  return vec256<T>(_mm256_and_si256(a, b));
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
template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE vec256<T> andnot(const vec256<T> not_mask,
                                            const vec256<T> mask) {
  return vec256<T>(_mm256_andnot_si256(not_mask, mask));
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

template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE vec256<T> operator|(const vec256<T> a,
                                               const vec256<T> b) {
  return vec256<T>(_mm256_or_si256(a, b));
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

template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE vec256<T> operator^(const vec256<T> a,
                                               const vec256<T> b) {
  return vec256<T>(_mm256_xor_si256(a, b));
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE f32x8 operator^(const f32x8 a, const f32x8 b) {
  return f32x8(_mm256_xor_ps(a, b));
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE f64x4 operator^(const f64x4 a, const f64x4 b) {
  return f64x4(_mm256_xor_pd(a, b));
}

// ================================================== STORE

// ------------------------------ Load all lanes

template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE vec256<T> load(vec256<T>,
                                          const T* SIMD_RESTRICT aligned) {
  return vec256<T>(
      _mm256_load_si256(reinterpret_cast<const __m256i*>(aligned)));
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE f32x8
load<float>(f32x8, const float* SIMD_RESTRICT aligned) {
  return f32x8(_mm256_load_ps(aligned));
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE f64x4
load<double>(f64x4, const double* SIMD_RESTRICT aligned) {
  return f64x4(_mm256_load_pd(aligned));
}

template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE vec256<T> load_unaligned(vec256<T>,
                                                    const T* SIMD_RESTRICT p) {
  return vec256<T>(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)));
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE f32x8
load_unaligned<float>(f32x8, const float* SIMD_RESTRICT p) {
  return f32x8(_mm256_loadu_ps(p));
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE f64x4
load_unaligned<double>(f64x4, const double* SIMD_RESTRICT p) {
  return f64x4(_mm256_loadu_pd(p));
}

// Loads 128 bit and duplicates into both 128-bit halves. This avoids the
// 3-cycle cost of moving data between 128-bit halves and avoids port 5.
template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE dup128x2<T> load_dup128(
    vec256<T>, const T* const SIMD_RESTRICT p) {
  // NOTE: Clang 3.9 generates VINSERTF128; 4 yields the desired VBROADCASTI128.
  return dup128x2<T>(_mm256_broadcastsi128_si256(load(vec128<T>(), p)));
}
SIMD_ATTR_AVX2 SIMD_INLINE dup128x2<float> load_dup128(
    f32x8, const float* const SIMD_RESTRICT p) {
  return dup128x2<float>(
      _mm256_broadcast_ps(reinterpret_cast<const __m128*>(p)));
}
SIMD_ATTR_AVX2 SIMD_INLINE dup128x2<double> load_dup128(
    f64x4, const double* const SIMD_RESTRICT p) {
  return dup128x2<double>(
      _mm256_broadcast_pd(reinterpret_cast<const __m128d*>(p)));
}

// ------------------------------ Store all lanes

template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE void store(const vec256<T> v,
                                      T* SIMD_RESTRICT aligned) {
  _mm256_store_si256(reinterpret_cast<__m256i*>(aligned), v);
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE void store<float>(const f32x8 v,
                                             float* SIMD_RESTRICT aligned) {
  _mm256_store_ps(aligned, v);
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE void store<double>(const f64x4 v,
                                              double* SIMD_RESTRICT aligned) {
  _mm256_store_pd(aligned, v);
}

template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE void store_unaligned(const vec256<T> v,
                                                T* SIMD_RESTRICT p) {
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v);
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE void store_unaligned<float>(const f32x8 v,
                                                       float* SIMD_RESTRICT p) {
  _mm256_storeu_ps(p, v);
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE void store_unaligned<double>(
    const f64x4 v, double* SIMD_RESTRICT p) {
  _mm256_storeu_pd(p, v);
}

// ------------------------------ Non-temporal stores

// Same as aligned stores on non-x86.

template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE void stream(const vec256<T> v,
                                       T* SIMD_RESTRICT aligned) {
  _mm256_stream_si256(reinterpret_cast<__m256i*>(aligned), v);
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE void stream<float>(const f32x8 v,
                                              float* SIMD_RESTRICT aligned) {
  _mm256_stream_ps(aligned, v);
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE void stream<double>(const f64x4 v,
                                               double* SIMD_RESTRICT aligned) {
  _mm256_stream_pd(aligned, v);
}

// stream(u32/64), store_fence and cache control already defined by x86_sse4.h.

// ================================================== SWIZZLE

// ------------------------------ Extract other half (see to_subset)

template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE vec128<T> to_other_half(const vec256<T> v) {
  return vec128<T>(_mm256_extracti128_si256(v, 1));
}
SIMD_ATTR_AVX2 SIMD_INLINE f32x4 to_other_half(const f32x8 v) {
  return f32x4(_mm256_extractf128_ps(v, 1));
}
SIMD_ATTR_AVX2 SIMD_INLINE f64x2 to_other_half(const f64x4 v) {
  return f64x2(_mm256_extractf128_pd(v, 1));
}

// ------------------------------ Shift vector by constant #bytes

// 0x01..0F, kBytes = 1 => 0x02..0F00
template <int kBytes, typename T>
SIMD_ATTR_AVX2 SIMD_INLINE vec256<T> shift_bytes_left(const vec256<T> v) {
  return vec256<T>(_mm256_slli_si256(v, kBytes));
}

// 0x01..0F, kBytes = 1 => 0x0001..0E
template <int kBytes, typename T>
SIMD_ATTR_AVX2 SIMD_INLINE vec256<T> shift_bytes_right(const vec256<T> v) {
  return vec256<T>(_mm256_srli_si256(v, kBytes));
}

// ------------------------------ Extract from 2x 128-bit at constant offset

// Extracts 128 bits from <hi, lo> by skipping the least-significant kBytes.
template <int kBytes, typename T>
SIMD_ATTR_AVX2 SIMD_INLINE vec256<T> extract_concat_bytes(const vec256<T> hi,
                                                          const vec256<T> lo) {
  return vec256<T>(_mm256_alignr_epi8(hi, lo, kBytes));
}

// ------------------------------ Broadcast/splat any lane

// Unsigned
template <int kLane>
SIMD_ATTR_AVX2 SIMD_INLINE u32x8 broadcast(const u32x8 v) {
  static_assert(0 <= kLane && kLane < NumLanes<u32x4>(), "Invalid lane");
  return u32x8(_mm256_shuffle_epi32(v, 0x55 * kLane));
}
template <int kLane>
SIMD_ATTR_AVX2 SIMD_INLINE u64x4 broadcast(const u64x4 v) {
  static_assert(0 <= kLane && kLane < NumLanes<u64x2>(), "Invalid lane");
  return u64x4(_mm256_shuffle_epi32(v, kLane ? 0xEE : 0x44));
}

// Signed
template <int kLane>
SIMD_ATTR_AVX2 SIMD_INLINE i32x8 broadcast(const i32x8 v) {
  static_assert(0 <= kLane && kLane < NumLanes<i32x4>(), "Invalid lane");
  return i32x8(_mm256_shuffle_epi32(v, 0x55 * kLane));
}
template <int kLane>
SIMD_ATTR_AVX2 SIMD_INLINE i64x4 broadcast(const i64x4 v) {
  static_assert(0 <= kLane && kLane < NumLanes<i64x2>(), "Invalid lane");
  return i64x4(_mm256_shuffle_epi32(v, kLane ? 0xEE : 0x44));
}

// Float
template <int kLane>
SIMD_ATTR_AVX2 SIMD_INLINE f32x8 broadcast(const f32x8 v) {
  static_assert(0 <= kLane && kLane < NumLanes<f32x4>(), "Invalid lane");
  return f32x8(_mm256_shuffle_ps(v, v, 0x55 * kLane));
}
template <int kLane>
SIMD_ATTR_AVX2 SIMD_INLINE f64x4 broadcast(const f64x4 v) {
  static_assert(0 <= kLane && kLane < NumLanes<f64x2>(), "Invalid lane");
  return f64x4(_mm256_shuffle_pd(v, v, 15 * kLane));
}

// ------------------------------ Shuffle bytes with variable indices

// Returns vector of bytes[from[i]]. "from" must be valid indices in [0, 16) or
// >= 0x80 to zero the i-th output byte.
template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE vec256<T> shuffle_bytes(const vec256<T> bytes,
                                                   const vec256<T> from) {
  return vec256<T>(_mm256_shuffle_epi8(bytes, from));
}

// ------------------------------ Hard-coded shuffles

// Notation: let i32x8 have lanes 7,6,5,4,3,2,1,0 (0 is least-significant).
// shuffle_0321 rotates four-lane blocks one lane to the right (the previous
// least-significant lane is now most-significant => 47650321). These could
// also be implemented via extract_concat_bytes but the shuffle_abcd notation
// is more convenient.

// Swap 64-bit halves
SIMD_ATTR_AVX2 SIMD_INLINE i32x8 shuffle_1032(const i32x8 v) {
  return i32x8(_mm256_shuffle_epi32(v, 0x4E));
}
SIMD_ATTR_AVX2 SIMD_INLINE f32x8 shuffle_1032(const f32x8 v) {
  return f32x8(_mm256_shuffle_ps(v, v, 0x4E));
}
SIMD_ATTR_AVX2 SIMD_INLINE i64x4 shuffle_01(const i64x4 v) {
  return i64x4(_mm256_shuffle_epi32(v, 0x4E));
}
SIMD_ATTR_AVX2 SIMD_INLINE f64x4 shuffle_01(const f64x4 v) {
  return f64x4(_mm256_shuffle_pd(v, v, 5));
}

// Rotate right 32 bits
SIMD_ATTR_AVX2 SIMD_INLINE i32x8 shuffle_0321(const i32x8 v) {
  return i32x8(_mm256_shuffle_epi32(v, 0x39));
}
SIMD_ATTR_AVX2 SIMD_INLINE f32x8 shuffle_0321(const f32x8 v) {
  return f32x8(_mm256_shuffle_ps(v, v, 0x39));
}
// Rotate left 32 bits
SIMD_ATTR_AVX2 SIMD_INLINE i32x8 shuffle_2103(const i32x8 v) {
  return i32x8(_mm256_shuffle_epi32(v, 0x93));
}
SIMD_ATTR_AVX2 SIMD_INLINE f32x8 shuffle_2103(const f32x8 v) {
  return f32x8(_mm256_shuffle_ps(v, v, 0x93));
}

// ------------------------------ Interleave lanes

// Interleaves lanes from halves of the 128-bit blocks of "a" (which provides
// the least-significant lane) and "b". To concatenate two half-width integers
// into one, use zip_lo/hi instead (also works with vec1).

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
// this is necessary because the single-lane vec1 cannot return two values.

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

// ------------------------------ Promotions (subset w/ narrow lanes -> full)

// Unsigned: zero-extend.
// Note: these have 3 cycle latency; if inputs are already split across the
// 128 bit blocks (in their upper/lower halves), then zip_hi/lo would be faster.
SIMD_ATTR_AVX2 SIMD_INLINE u16x16 convert_to(uint16_t, const u8x16 v) {
  return u16x16(_mm256_cvtepu8_epi16(v));
}
SIMD_ATTR_AVX2 SIMD_INLINE u32x8 convert_to(uint32_t, const u8x8 v) {
  return u32x8(_mm256_cvtepu8_epi32(v));
}
SIMD_ATTR_AVX2 SIMD_INLINE i16x16 convert_to(int16_t, const u8x16 v) {
  return i16x16(_mm256_cvtepu8_epi16(v));
}
SIMD_ATTR_AVX2 SIMD_INLINE i32x8 convert_to(int32_t, const u8x8 v) {
  return i32x8(_mm256_cvtepu8_epi32(v));
}
SIMD_ATTR_AVX2 SIMD_INLINE u32x8 convert_to(uint32_t, const u16x8 v) {
  return u32x8(_mm256_cvtepu16_epi32(v));
}
SIMD_ATTR_AVX2 SIMD_INLINE i32x8 convert_to(int32_t, const u16x8 v) {
  return i32x8(_mm256_cvtepu16_epi32(v));
}
SIMD_ATTR_AVX2 SIMD_INLINE u64x4 convert_to(uint64_t, const u32x4 v) {
  return u64x4(_mm256_cvtepu32_epi64(v));
}

// Signed: replicate sign bit.
// Note: these have 3 cycle latency; if inputs are already split across the
// 128 bit blocks (in their upper/lower halves), then zip_hi/lo followed by
// signed shift would be faster.
SIMD_ATTR_AVX2 SIMD_INLINE i16x16 convert_to(int16_t, const i8x16 v) {
  return i16x16(_mm256_cvtepi8_epi16(v));
}
SIMD_ATTR_AVX2 SIMD_INLINE i32x8 convert_to(int32_t, const i8x8 v) {
  return i32x8(_mm256_cvtepi8_epi32(v));
}
SIMD_ATTR_AVX2 SIMD_INLINE i32x8 convert_to(int32_t, const i16x8 v) {
  return i32x8(_mm256_cvtepi16_epi32(v));
}
SIMD_ATTR_AVX2 SIMD_INLINE i64x4 convert_to(int64_t, const i32x4 v) {
  return i64x4(_mm256_cvtepi32_epi64(v));
}

// ------------------------------ Demotions (full -> subset w/ narrow lanes)

SIMD_ATTR_AVX2 SIMD_INLINE u16x8 convert_to(uint16_t, const i32x8 v) {
  const __m256i u16 = _mm256_packus_epi32(v, v);
  // Concatenating lower halves of both 128-bit blocks afterward is more
  // efficient than an extra input with low block = high block of v.
  return u16x8(_mm256_castsi256_si128(_mm256_permute4x64_epi64(u16, 0x88)));
}

SIMD_ATTR_AVX2 SIMD_INLINE u8x8 convert_to(uint8_t, const i32x8 v) {
  const __m256i u16 = _mm256_packus_epi32(v, v);
  const __m256i u8 = _mm256_packus_epi16(u16, u16);
  // Zeros upper bits.
  const __m256i idx = _mm256_castsi128_si256(_mm_cvtsi64_si128(0x400000000ull));
  const __m256i fixed = _mm256_permutevar8x32_epi32(u8, idx);
  return u8x8(_mm256_castsi256_si128(fixed));
}

SIMD_ATTR_AVX2 SIMD_INLINE i16x8 convert_to(int16_t, const i32x8 v) {
  const __m256i i16 = _mm256_packs_epi32(v, v);
  return i16x8(_mm256_castsi256_si128(_mm256_permute4x64_epi64(i16, 0x88)));
}

SIMD_ATTR_AVX2 SIMD_INLINE i8x8 convert_to(int8_t, const i32x8 v) {
  const __m256i i16 = _mm256_packs_epi32(v, v);
  const __m256i i8 = _mm256_packs_epi16(i16, i16);
  // Zeros upper bits.
  const __m256i idx = _mm256_castsi128_si256(_mm_cvtsi64_si128(0x400000000ull));
  const __m256i fixed = _mm256_permutevar8x32_epi32(i8, idx);
  return i8x8(_mm256_castsi256_si128(fixed));
}

SIMD_ATTR_AVX2 SIMD_INLINE u8x16 convert_to(uint8_t, const i16x16 v) {
  const __m256i u8 = _mm256_packus_epi16(v, v);
  return u8x16(_mm256_castsi256_si128(_mm256_permute4x64_epi64(u8, 0x88)));
}

SIMD_ATTR_AVX2 SIMD_INLINE i8x16 convert_to(int8_t, const i16x16 v) {
  const __m256i i8 = _mm256_packs_epi16(v, v);
  return i8x16(_mm256_castsi256_si128(_mm256_permute4x64_epi64(i8, 0x88)));
}

// ------------------------------ Select/blend

// Returns mask ? b : a. Due to ARM's semantics, each lane of "mask" must
// equal T(0) or ~T(0) although x86 may only check the most significant bit.
template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE vec256<T> select(const vec256<T> a,
                                            const vec256<T> b,
                                            const vec256<T> mask) {
  return vec256<T>(_mm256_blendv_epi8(a, b, mask));
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
