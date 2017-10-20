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

// 128-bit ARM64 NEON vectors and operations.
// (No include guard nor namespace: this is included from the middle of simd.h.)

// Avoid compile errors when generating deps.mk.
#ifndef SIMD_DEPS

template <typename T, size_t N>
struct raw_arm8;

// 128
template <>
struct raw_arm8<uint8_t, 16> {
  using type = uint8x16_t;
};

template <>
struct raw_arm8<uint16_t, 8> {
  using type = uint16x8_t;
};

template <>
struct raw_arm8<uint32_t, 4> {
  using type = uint32x4_t;
};

template <>
struct raw_arm8<uint64_t, 2> {
  using type = uint64x2_t;
};

template <>
struct raw_arm8<int8_t, 16> {
  using type = int8x16_t;
};

template <>
struct raw_arm8<int16_t, 8> {
  using type = int16x8_t;
};

template <>
struct raw_arm8<int32_t, 4> {
  using type = int32x4_t;
};

template <>
struct raw_arm8<int64_t, 2> {
  using type = int64x2_t;
};

template <>
struct raw_arm8<float, 4> {
  using type = float32x4_t;
};

template <>
struct raw_arm8<double, 2> {
  using type = float64x2_t;
};

// 64
template <>
struct raw_arm8<uint8_t, 8> {
  using type = uint8x8_t;
};

template <>
struct raw_arm8<uint16_t, 4> {
  using type = uint16x4_t;
};

template <>
struct raw_arm8<uint32_t, 2> {
  using type = uint32x2_t;
};

template <>
struct raw_arm8<uint64_t, 1> {
  using type = uint64x1_t;
};

template <>
struct raw_arm8<int8_t, 8> {
  using type = int8x8_t;
};

template <>
struct raw_arm8<int16_t, 4> {
  using type = int16x4_t;
};

template <>
struct raw_arm8<int32_t, 2> {
  using type = int32x2_t;
};

template <>
struct raw_arm8<int64_t, 1> {
  using type = int64x1_t;
};

template <>
struct raw_arm8<float, 2> {
  using type = float32x2_t;
};

template <>
struct raw_arm8<double, 1> {
  using type = float64x1_t;
};

// 32 (same as 64)
template <>
struct raw_arm8<uint8_t, 4> {
  using type = uint8x8_t;
};

template <>
struct raw_arm8<uint16_t, 2> {
  using type = uint16x4_t;
};

template <>
struct raw_arm8<uint32_t, 1> {
  using type = uint32x2_t;
};

template <>
struct raw_arm8<int8_t, 4> {
  using type = int8x8_t;
};

template <>
struct raw_arm8<int16_t, 2> {
  using type = int16x4_t;
};

template <>
struct raw_arm8<int32_t, 1> {
  using type = int32x2_t;
};

template <>
struct raw_arm8<float, 1> {
  using type = float32x2_t;
};

// All 128-bit blocks equal; returned from load_dup128.
template <typename T, size_t N = ARM8::NumLanes<T>()>
struct dup128x1 {
  using Raw = typename raw_arm8<T, N>::type;

  explicit dup128x1(const Raw v) : raw(v) {}

  Raw raw;
};

template <typename T, size_t N = ARM8::NumLanes<T>()>
class vec_arm8 {
  using Raw = typename raw_arm8<T, N>::type;
  static constexpr size_t kBytes = N * sizeof(T);
  static_assert((kBytes & (kBytes - 1)) == 0, "Size must be 2^j bytes");
  static_assert(kBytes == 0 || (4 <= kBytes && kBytes <= 16), "Invalid size");

 public:
  SIMD_INLINE vec_arm8() {}
  vec_arm8(const vec_arm8&) = default;
  vec_arm8& operator=(const vec_arm8&) = default;
  SIMD_INLINE explicit vec_arm8(const Raw v) : raw(v) {}

  // Used by non-member functions; avoids verbose .raw() for each argument and
  // reduces Clang compile time of the test by 10-20%.
  SIMD_INLINE operator Raw() const { return raw; }  // NOLINT

  // Compound assignment. Only usable if there is a corresponding non-member
  // binary operator overload. For example, only f32 and f64 support division.
  SIMD_INLINE vec_arm8& operator*=(const vec_arm8 other) {
    return *this = (*this * other);
  }
  SIMD_INLINE vec_arm8& operator/=(const vec_arm8 other) {
    return *this = (*this / other);
  }
  SIMD_INLINE vec_arm8& operator+=(const vec_arm8 other) {
    return *this = (*this + other);
  }
  SIMD_INLINE vec_arm8& operator-=(const vec_arm8 other) {
    return *this = (*this - other);
  }
  SIMD_INLINE vec_arm8& operator&=(const vec_arm8 other) {
    return *this = (*this & other);
  }
  SIMD_INLINE vec_arm8& operator|=(const vec_arm8 other) {
    return *this = (*this | other);
  }
  SIMD_INLINE vec_arm8& operator^=(const vec_arm8 other) {
    return *this = (*this ^ other);
  }

  Raw raw;
};

template <typename T, size_t N>
struct VecT<T, N, ARM8> {
  using type = vec_arm8<T, N>;
};

using u8x16 = vec_arm8<uint8_t, 16>;
using u16x8 = vec_arm8<uint16_t, 8>;
using u32x4 = vec_arm8<uint32_t, 4>;
using u64x2 = vec_arm8<uint64_t, 2>;
using i8x16 = vec_arm8<int8_t, 16>;
using i16x8 = vec_arm8<int16_t, 8>;
using i32x4 = vec_arm8<int32_t, 4>;
using i64x2 = vec_arm8<int64_t, 2>;
using f32x4 = vec_arm8<float, 4>;
using f64x2 = vec_arm8<double, 2>;

using u8x8 = vec_arm8<uint8_t, 8>;
using u16x4 = vec_arm8<uint16_t, 4>;
using u32x2 = vec_arm8<uint32_t, 2>;
using u64x1 = vec_arm8<uint64_t, 1>;
using i8x8 = vec_arm8<int8_t, 8>;
using i16x4 = vec_arm8<int16_t, 4>;
using i32x2 = vec_arm8<int32_t, 2>;
using i64x1 = vec_arm8<int64_t, 1>;
using f32x2 = vec_arm8<float, 2>;
using f64x1 = vec_arm8<double, 1>;

using u8x4 = vec_arm8<uint8_t, 4>;
using u16x2 = vec_arm8<uint16_t, 2>;
using u32x1 = vec_arm8<uint32_t, 1>;
using i8x4 = vec_arm8<int8_t, 4>;
using i16x2 = vec_arm8<int16_t, 2>;
using i32x1 = vec_arm8<int32_t, 1>;
using f32x1 = vec_arm8<float, 1>;

template <typename T>
struct shift_left_count {
  vec_arm8<T> raw;
};

template <typename T>
struct shift_right_count {
  vec_arm8<T> raw;
};

// ------------------------------ Set

// Returns a vector with all lanes set to "t".
SIMD_INLINE u8x16 set1(Full<uint8_t, ARM8>, const uint8_t t) {
  return u8x16(vdupq_n_u8(t));
}
SIMD_INLINE u16x8 set1(Full<uint16_t, ARM8>, const uint16_t t) {
  return u16x8(vdupq_n_u16(t));
}
SIMD_INLINE u32x4 set1(Full<uint32_t, ARM8>, const uint32_t t) {
  return u32x4(vdupq_n_u32(t));
}
SIMD_INLINE u64x2 set1(Full<uint64_t, ARM8>, const uint64_t t) {
  return u64x2(vdupq_n_u64(t));
}
SIMD_INLINE i8x16 set1(Full<int8_t, ARM8>, const int8_t t) {
  return i8x16(vdupq_n_s8(t));
}
SIMD_INLINE i16x8 set1(Full<int16_t, ARM8>, const int16_t t) {
  return i16x8(vdupq_n_s16(t));
}
SIMD_INLINE i32x4 set1(Full<int32_t, ARM8>, const int32_t t) {
  return i32x4(vdupq_n_s32(t));
}
SIMD_INLINE i64x2 set1(Full<int64_t, ARM8>, const int64_t t) {
  return i64x2(vdupq_n_s64(t));
}
SIMD_INLINE f32x4 set1(Full<float, ARM8>, const float t) {
  return f32x4(vdupq_n_f32(t));
}
SIMD_INLINE f64x2 set1(Full<double, ARM8>, const double t) {
  return f64x2(vdupq_n_f64(t));
}

// 64
SIMD_INLINE u8x8 set1(Desc<uint8_t, 8, ARM8>, const uint8_t t) {
  return u8x8(vdup_n_u8(t));
}
SIMD_INLINE u16x4 set1(Desc<uint16_t, 4, ARM8>, const uint16_t t) {
  return u16x4(vdup_n_u16(t));
}
SIMD_INLINE u32x2 set1(Desc<uint32_t, 2, ARM8>, const uint32_t t) {
  return u32x2(vdup_n_u32(t));
}
SIMD_INLINE u64x1 set1(Desc<uint64_t, 1, ARM8>, const uint64_t t) {
  return u64x1(vdup_n_u64(t));
}
SIMD_INLINE i8x8 set1(Desc<int8_t, 8, ARM8>, const int8_t t) {
  return i8x8(vdup_n_s8(t));
}
SIMD_INLINE i16x4 set1(Desc<int16_t, 4, ARM8>, const int16_t t) {
  return i16x4(vdup_n_s16(t));
}
SIMD_INLINE i32x2 set1(Desc<int32_t, 2, ARM8>, const int32_t t) {
  return i32x2(vdup_n_s32(t));
}
SIMD_INLINE i64x1 set1(Desc<int64_t, 1, ARM8>, const int64_t t) {
  return i64x1(vdup_n_s64(t));
}
SIMD_INLINE f32x2 set1(Desc<float, 2, ARM8>, const float t) {
  return f32x2(vdup_n_f32(t));
}
SIMD_INLINE f64x1 set1(Desc<double, 1, ARM8>, const double t) {
  return f64x1(vdup_n_f64(t));
}

// 32
SIMD_INLINE u8x4 set1(Desc<uint8_t, 4, ARM8>, const uint8_t t) {
  return u8x4(vdup_n_u8(t));
}
SIMD_INLINE u16x2 set1(Desc<uint16_t, 2, ARM8>, const uint16_t t) {
  return u16x2(vdup_n_u16(t));
}
SIMD_INLINE u32x1 set1(Desc<uint32_t, 1, ARM8>, const uint32_t t) {
  return u32x1(vdup_n_u32(t));
}
SIMD_INLINE i8x4 set1(Desc<int8_t, 4, ARM8>, const int8_t t) {
  return i8x4(vdup_n_s8(t));
}
SIMD_INLINE i16x2 set1(Desc<int16_t, 2, ARM8>, const int16_t t) {
  return i16x2(vdup_n_s16(t));
}
SIMD_INLINE i32x1 set1(Desc<int32_t, 1, ARM8>, const int32_t t) {
  return i32x1(vdup_n_s32(t));
}
SIMD_INLINE f32x1 set1(Desc<float, 1, ARM8>, const float t) {
  return f32x1(vdup_n_f32(t));
}

// Returns an all-zero vector.
template <typename T, size_t N>
SIMD_INLINE Vec<T, N, ARM8> setzero(Desc<T, N, ARM8> d) {
  return set1(d, 0);
}

// Returns a vector with lane i=[0, N) set to "first" + i. Unique per-lane
// values are required to detect lane-crossing bugs.
template <typename T, size_t N, typename T2>
SIMD_INLINE Vec<T, N, ARM8> iota(Desc<T, N, ARM8> d, const T2 first) {
  SIMD_ALIGN T lanes[N];
  for (size_t i = 0; i < N; ++i) {
    lanes[i] = first + i;
  }
  return load(d, lanes);
}

// Returns a vector/part with value "t".
template <typename T, size_t N>
SIMD_INLINE Vec<T, N, ARM8> set(Desc<T, N, ARM8> d, const T t) {
  return set1(d, t);
}

// Gets the single value stored in a vector/part.
template <typename T, size_t N>
SIMD_INLINE T get(Desc<T, N, ARM8> d, const Vec<T, N, ARM8> v) {
  // TODO(janwas): more efficient implementation?
  SIMD_ALIGN T ret[N];
  store(v, d, &ret);
  return ret[0];
}

// Returns part of a vector (unspecified whether upper or lower).
SIMD_INLINE u8x8 any_part(Desc<uint8_t, 8, ARM8>, const u8x16 v) {
  return u8x8(vget_low_u8(v));
}
SIMD_INLINE u16x4 any_part(Desc<uint16_t, 4, ARM8>, const u16x8 v) {
  return u16x4(vget_low_u16(v));
}
SIMD_INLINE u32x2 any_part(Desc<uint32_t, 2, ARM8>, const u32x4 v) {
  return u32x2(vget_low_u32(v));
}
SIMD_INLINE u64x1 any_part(Desc<uint64_t, 1, ARM8>, const u64x2 v) {
  return u64x1(vget_low_u64(v));
}
SIMD_INLINE i8x8 any_part(Desc<int8_t, 8>, const i8x16 v) {
  return i8x8(vget_low_s8(v));
}
SIMD_INLINE i16x4 any_part(Desc<int16_t, 4>, const i16x8 v) {
  return i16x4(vget_low_s16(v));
}
SIMD_INLINE i32x2 any_part(Desc<int32_t, 2>, const i32x4 v) {
  return i32x2(vget_low_s32(v));
}
SIMD_INLINE i64x1 any_part(Desc<int64_t, 1>, const i64x2 v) {
  return i64x1(vget_low_s64(v));
}
SIMD_INLINE f32x2 any_part(Desc<float, 2>, const f32x4 v) {
  return f32x2(vget_low_f32(v));
}
SIMD_INLINE f64x1 any_part(Desc<double, 1>, const f64x2 v) {
  return f64x1(vget_low_f64(v));
}

SIMD_INLINE u8x4 any_part(Desc<uint8_t, 4, ARM8>, const u8x16 v) {
  return u8x4(vget_low_u8(v));
}
SIMD_INLINE u16x2 any_part(Desc<uint16_t, 2, ARM8>, const u16x8 v) {
  return u16x2(vget_low_u16(v));
}
SIMD_INLINE u32x1 any_part(Desc<uint32_t, 1, ARM8>, const u32x4 v) {
  return u32x1(vget_low_u32(v));
}
SIMD_INLINE i8x4 any_part(Desc<int8_t, 4, ARM8>, const i8x16 v) {
  return i8x4(vget_low_s8(v));
}
SIMD_INLINE i16x2 any_part(Desc<int16_t, 2, ARM8>, const i16x8 v) {
  return i16x2(vget_low_s16(v));
}
SIMD_INLINE i32x1 any_part(Desc<int32_t, 1, ARM8>, const i32x4 v) {
  return i32x1(vget_low_s32(v));
}
SIMD_INLINE f32x1 any_part(Desc<float, 1, ARM8>, const f32x4 v) {
  return f32x1(vget_low_f32(v));
}

// ================================================== ARITHMETIC

// ------------------------------ Addition

// Unsigned
SIMD_INLINE u8x16 operator+(const u8x16 a, const u8x16 b) {
  return u8x16(vaddq_u8(a, b));
}
SIMD_INLINE u16x8 operator+(const u16x8 a, const u16x8 b) {
  return u16x8(vaddq_u16(a, b));
}
SIMD_INLINE u32x4 operator+(const u32x4 a, const u32x4 b) {
  return u32x4(vaddq_u32(a, b));
}
SIMD_INLINE u64x2 operator+(const u64x2 a, const u64x2 b) {
  return u64x2(vaddq_u64(a, b));
}

// Signed
SIMD_INLINE i8x16 operator+(const i8x16 a, const i8x16 b) {
  return i8x16(vaddq_s8(a, b));
}
SIMD_INLINE i16x8 operator+(const i16x8 a, const i16x8 b) {
  return i16x8(vaddq_s16(a, b));
}
SIMD_INLINE i32x4 operator+(const i32x4 a, const i32x4 b) {
  return i32x4(vaddq_s32(a, b));
}
SIMD_INLINE i64x2 operator+(const i64x2 a, const i64x2 b) {
  return i64x2(vaddq_s64(a, b));
}

// Float
SIMD_INLINE f32x4 operator+(const f32x4 a, const f32x4 b) {
  return f32x4(vaddq_f32(a, b));
}
SIMD_INLINE f64x2 operator+(const f64x2 a, const f64x2 b) {
  return f64x2(vaddq_f64(a, b));
}

// ------------------------------ Subtraction

// Unsigned
SIMD_INLINE u8x16 operator-(const u8x16 a, const u8x16 b) {
  return u8x16(vsubq_u8(a, b));
}
SIMD_INLINE u16x8 operator-(const u16x8 a, const u16x8 b) {
  return u16x8(vsubq_u16(a, b));
}
SIMD_INLINE u32x4 operator-(const u32x4 a, const u32x4 b) {
  return u32x4(vsubq_u32(a, b));
}
SIMD_INLINE u64x2 operator-(const u64x2 a, const u64x2 b) {
  return u64x2(vsubq_u64(a, b));
}

// Signed
SIMD_INLINE i8x16 operator-(const i8x16 a, const i8x16 b) {
  return i8x16(vsubq_s8(a, b));
}
SIMD_INLINE i16x8 operator-(const i16x8 a, const i16x8 b) {
  return i16x8(vsubq_s16(a, b));
}
SIMD_INLINE i32x4 operator-(const i32x4 a, const i32x4 b) {
  return i32x4(vsubq_s32(a, b));
}
SIMD_INLINE i64x2 operator-(const i64x2 a, const i64x2 b) {
  return i64x2(vsubq_s64(a, b));
}

// Float
SIMD_INLINE f32x4 operator-(const f32x4 a, const f32x4 b) {
  return f32x4(vsubq_f32(a, b));
}
SIMD_INLINE f64x2 operator-(const f64x2 a, const f64x2 b) {
  return f64x2(vsubq_f64(a, b));
}

// ------------------------------ Saturating addition

// Returns a + b clamped to the destination range.

// Unsigned
SIMD_INLINE u8x16 add_sat(const u8x16 a, const u8x16 b) {
  return u8x16(vqaddq_u8(a, b));
}
SIMD_INLINE u16x8 add_sat(const u16x8 a, const u16x8 b) {
  return u16x8(vqaddq_u16(a, b));
}

// Signed
SIMD_INLINE i8x16 add_sat(const i8x16 a, const i8x16 b) {
  return i8x16(vqaddq_s8(a, b));
}
SIMD_INLINE i16x8 add_sat(const i16x8 a, const i16x8 b) {
  return i16x8(vqaddq_s16(a, b));
}

// ------------------------------ Saturating subtraction

// Returns a - b clamped to the destination range.

// Unsigned
SIMD_INLINE u8x16 sub_sat(const u8x16 a, const u8x16 b) {
  return u8x16(vqsubq_u8(a, b));
}
SIMD_INLINE u16x8 sub_sat(const u16x8 a, const u16x8 b) {
  return u16x8(vqsubq_u16(a, b));
}

// Signed
SIMD_INLINE i8x16 sub_sat(const i8x16 a, const i8x16 b) {
  return i8x16(vqsubq_s8(a, b));
}
SIMD_INLINE i16x8 sub_sat(const i16x8 a, const i16x8 b) {
  return i16x8(vqsubq_s16(a, b));
}

// ------------------------------ Average

// Returns (a + b + 1) / 2

// Unsigned
SIMD_INLINE u8x16 avg(const u8x16 a, const u8x16 b) {
  return u8x16(vrhaddq_u8(a, b));
}
SIMD_INLINE u16x8 avg(const u16x8 a, const u16x8 b) {
  return u16x8(vrhaddq_u16(a, b));
}

// ------------------------------ Shift lanes by constant #bits

// Unsigned
template <int kBits>
SIMD_INLINE u16x8 shift_left(const u16x8 v) {
  return u16x8(vshlq_n_u16(v, kBits));
}
template <int kBits>
SIMD_INLINE u16x8 shift_right(const u16x8 v) {
  return u16x8(vshrq_n_u16(v, kBits));
}
template <int kBits>
SIMD_INLINE u32x4 shift_left(const u32x4 v) {
  return u32x4(vshlq_n_u32(v, kBits));
}
template <int kBits>
SIMD_INLINE u32x4 shift_right(const u32x4 v) {
  return u32x4(vshrq_n_u32(v, kBits));
}
template <int kBits>
SIMD_INLINE u64x2 shift_left(const u64x2 v) {
  return u64x2(vshlq_n_u64(v, kBits));
}
template <int kBits>
SIMD_INLINE u64x2 shift_right(const u64x2 v) {
  return u64x2(vshrq_n_u64(v, kBits));
}

// Signed (no i64 shr)
template <int kBits>
SIMD_INLINE i16x8 shift_left(const i16x8 v) {
  return i16x8(vshlq_n_s16(v, kBits));
}
template <int kBits>
SIMD_INLINE i16x8 shift_right(const i16x8 v) {
  return i16x8(vshrq_n_s16(v, kBits));
}
template <int kBits>
SIMD_INLINE i32x4 shift_left(const i32x4 v) {
  return i32x4(vshlq_n_s32(v, kBits));
}
template <int kBits>
SIMD_INLINE i32x4 shift_right(const i32x4 v) {
  return i32x4(vshrq_n_s32(v, kBits));
}
template <int kBits>
SIMD_INLINE i64x2 shift_left(const i64x2 v) {
  return i64x2(vshlq_n_s64(v, kBits));
}

// ------------------------------ Shift lanes by same variable #bits

// Extra overhead, use _var instead unless SSE4 support is required.

template <typename T>
SIMD_INLINE shift_left_count<T> set_shift_left_count(Full<T, ARM8> d,
                                                     const int bits) {
  return shift_left_count<T>{set1(d, bits)};
}

template <typename T>
SIMD_INLINE shift_right_count<T> set_shift_right_count(Full<T, ARM8> d,
                                                       const int bits) {
  return shift_right_count<T>{set1(d, -bits)};
}

// Unsigned (no u8)
SIMD_INLINE u16x8 shift_left_same(const u16x8 v,
                                  const shift_left_count<uint16_t> bits) {
  return u16x8(vshlq_u16(v, bits.raw));
}
SIMD_INLINE u16x8 shift_right_same(const u16x8 v,
                                   const shift_right_count<uint16_t> bits) {
  return u16x8(vshlq_u16(v, bits.raw));
}
SIMD_INLINE u32x4 shift_left_same(const u32x4 v,
                           const shift_left_count<uint32_t> bits) {
  return u32x4(vshlq_u32(v, bits.raw));
}
SIMD_INLINE u32x4 shift_right_same(const u32x4 v,
                           const shift_right_count<uint32_t> bits) {
  return u32x4(vshlq_u32(v, bits.raw));
}
SIMD_INLINE u64x2 shift_left_same(const u64x2 v,
                           const shift_left_count<uint64_t> bits) {
  return u64x2(vshlq_u64(v, bits.raw));
}
SIMD_INLINE u64x2 shift_right_same(const u64x2 v,
                           const shift_right_count<uint64_t> bits) {
  return u64x2(vshlq_u64(v, bits.raw));
}

// Signed (no i8,i64)
SIMD_INLINE i16x8 shift_left_same(const i16x8 v,
                                  const shift_left_count<int16_t> bits) {
  return i16x8(vshlq_s16(v, bits.raw));
}
SIMD_INLINE i16x8 shift_right_same(const i16x8 v,
                                   const shift_right_count<int16_t> bits) {
  return i16x8(vshlq_s16(v, bits.raw));
}
SIMD_INLINE i32x4 shift_left_same(const i32x4 v,
                           const shift_left_count<int32_t> bits) {
  return i32x4(vshlq_s32(v, bits.raw));
}
SIMD_INLINE i32x4 shift_right_same(const i32x4 v,
                           const shift_right_count<int32_t> bits) {
  return i32x4(vshlq_s32(v, bits.raw));
}
SIMD_INLINE i64x2 shift_left_same(const i64x2 v,
                           const shift_left_count<int64_t> bits) {
  return i64x2(vshlq_s64(v, bits.raw));
}

// ------------------------------ Shift lanes by independent variable #bits

// Unsigned (no u8,u16)
SIMD_INLINE u32x4 shift_left_var(const u32x4 v, const u32x4 bits) {
  return u32x4(vshlq_u32(v, bits.raw));
}
SIMD_INLINE u32x4 shift_right_var(const u32x4 v, const u32x4 bits) {
  return u32x4(vshlq_u32(v, vnegq_s32(vreinterpretq_s32_u32(bits.raw))));
}
SIMD_INLINE u64x2 shift_left_var(const u64x2 v, const u64x2 bits) {
  return u64x2(vshlq_u64(v, bits.raw));
}
SIMD_INLINE u64x2 shift_right_var(const u64x2 v, const u64x2 bits) {
  return u64x2(vshlq_u64(v, vnegq_s64(vreinterpretq_s64_u64(bits.raw))));
}

// Signed (no i8,i16)
SIMD_INLINE i32x4 shift_left_var(const i32x4 v, const i32x4 bits) {
  return i32x4(vshlq_s32(v, bits.raw));
}
SIMD_INLINE i32x4 shift_right_var(const i32x4 v, const i32x4 bits) {
  return i32x4(vshlq_s32(v, vnegq_s32(bits.raw)));
}
SIMD_INLINE i64x2 shift_left_var(const i64x2 v, const i64x2 bits) {
  return i64x2(vshlq_s64(v, bits.raw));
}

// ------------------------------ Minimum

// Unsigned (no u64)
SIMD_INLINE u8x16 min(const u8x16 a, const u8x16 b) {
  return u8x16(vminq_u8(a, b));
}
SIMD_INLINE u16x8 min(const u16x8 a, const u16x8 b) {
  return u16x8(vminq_u16(a, b));
}
SIMD_INLINE u32x4 min(const u32x4 a, const u32x4 b) {
  return u32x4(vminq_u32(a, b));
}

// Signed (no i64)
SIMD_INLINE i8x16 min(const i8x16 a, const i8x16 b) {
  return i8x16(vminq_s8(a, b));
}
SIMD_INLINE i16x8 min(const i16x8 a, const i16x8 b) {
  return i16x8(vminq_s16(a, b));
}
SIMD_INLINE i32x4 min(const i32x4 a, const i32x4 b) {
  return i32x4(vminq_s32(a, b));
}

// Float
SIMD_INLINE f32x4 min(const f32x4 a, const f32x4 b) {
  return f32x4(vminq_f32(a, b));
}
SIMD_INLINE f64x2 min(const f64x2 a, const f64x2 b) {
  return f64x2(vminq_f64(a, b));
}

// ------------------------------ Maximum

// Unsigned (no u64)
SIMD_INLINE u8x16 max(const u8x16 a, const u8x16 b) {
  return u8x16(vmaxq_u8(a, b));
}
SIMD_INLINE u16x8 max(const u16x8 a, const u16x8 b) {
  return u16x8(vmaxq_u16(a, b));
}
SIMD_INLINE u32x4 max(const u32x4 a, const u32x4 b) {
  return u32x4(vmaxq_u32(a, b));
}

// Signed (no i64)
SIMD_INLINE i8x16 max(const i8x16 a, const i8x16 b) {
  return i8x16(vmaxq_s8(a, b));
}
SIMD_INLINE i16x8 max(const i16x8 a, const i16x8 b) {
  return i16x8(vmaxq_s16(a, b));
}
SIMD_INLINE i32x4 max(const i32x4 a, const i32x4 b) {
  return i32x4(vmaxq_s32(a, b));
}

// Float
SIMD_INLINE f32x4 max(const f32x4 a, const f32x4 b) {
  return f32x4(vmaxq_f32(a, b));
}
SIMD_INLINE f64x2 max(const f64x2 a, const f64x2 b) {
  return f64x2(vmaxq_f64(a, b));
}

// ------------------------------ Integer multiplication

// Unsigned
SIMD_INLINE u16x8 operator*(const u16x8 a, const u16x8 b) {
  return u16x8(vmulq_u16(a, b));
}
SIMD_INLINE u32x4 operator*(const u32x4 a, const u32x4 b) {
  return u32x4(vmulq_u32(a, b));
}

// Signed
SIMD_INLINE i16x8 operator*(const i16x8 a, const i16x8 b) {
  return i16x8(vmulq_s16(a, b));
}
SIMD_INLINE i32x4 operator*(const i32x4 a, const i32x4 b) {
  return i32x4(vmulq_s32(a, b));
}

// "Extensions": useful but quite performance-portable operations. We add
// functions to this namespace in multiple places.
namespace ext {

// Returns the upper 16 bits of a * b in each lane.
SIMD_INLINE i16x8 mulhi(const i16x8 a, const i16x8 b) {
  int32x4_t rlo = vmull_s16(vget_low_s16(a), vget_low_s16(b));
  int32x4_t rhi = vmull_high_s16(a, b);
  return i16x8(vuzp2q_s16(vreinterpretq_s16_s32(rlo),
                          vreinterpretq_s16_s32(rhi)));
}

}  // namespace ext

// Multiplies even lanes (0, 2 ..) and places the double-wide result into
// even and the upper half into its odd neighbor lane.
SIMD_INLINE i64x2 mul_even(const i32x4 a, const i32x4 b) {
  int32x4_t a_packed = vuzp1q_s32(a, a);
  int32x4_t b_packed = vuzp1q_s32(b, b);
  return i64x2(vmull_s32(vget_low_s32(a_packed), vget_low_s32(b_packed)));
}
SIMD_INLINE u64x2 mul_even(const u32x4 a, const u32x4 b) {
  uint32x4_t a_packed = vuzp1q_u32(a, a);
  uint32x4_t b_packed = vuzp1q_u32(b, b);
  return u64x2(vmull_u32(vget_low_u32(a_packed), vget_low_u32(b_packed)));
}

// ------------------------------ Floating-point mul / div

SIMD_INLINE f32x4 operator*(const f32x4 a, const f32x4 b) {
  return f32x4(vmulq_f32(a, b));
}
SIMD_INLINE f64x2 operator*(const f64x2 a, const f64x2 b) {
  return f64x2(vmulq_f64(a, b));
}

SIMD_INLINE f32x4 operator/(const f32x4 a, const f32x4 b) {
  return f32x4(vdivq_f32(a, b));
}
SIMD_INLINE f64x2 operator/(const f64x2 a, const f64x2 b) {
  return f64x2(vdivq_f64(a, b));
}

// Approximate reciprocal
SIMD_INLINE f32x4 rcp_approx(const f32x4 v) {
  return f32x4(vrecpeq_f32(v));
}

// ------------------------------ Floating-point multiply-add variants

// Returns mul * x + add
SIMD_INLINE f32x4 mul_add(const f32x4 mul, const f32x4 x,
                                         const f32x4 add) {
  return f32x4(vmlaq_f32(add, mul, x));
}
SIMD_INLINE f64x2 mul_add(const f64x2 mul, const f64x2 x,
                                         const f64x2 add) {
  return f64x2(vmlaq_f64(add, mul, x));
}

// Returns mul * x - sub
SIMD_INLINE f32x4 mul_sub(const f32x4 mul, const f32x4 x,
                                         const f32x4 sub) {
  return mul * x - sub;
}
SIMD_INLINE f64x2 mul_sub(const f64x2 mul, const f64x2 x,
                                         const f64x2 sub) {
  return mul * x - sub;
}

// Returns add - mul * x
SIMD_INLINE f32x4 nmul_add(const f32x4 mul, const f32x4 x,
                                          const f32x4 add) {
  return f32x4(vmlsq_f32(add, mul, x));
}
SIMD_INLINE f64x2 nmul_add(const f64x2 mul, const f64x2 x,
                                          const f64x2 add) {
  return f64x2(vmlsq_f64(add, mul, x));
}

// nmul_sub would require an additional negate of mul or x.

// ------------------------------ Floating-point square root

// Full precision square root
SIMD_INLINE f32x4 sqrt(const f32x4 v) {
  return f32x4(vsqrtq_f32(v));
}
SIMD_INLINE f64x2 sqrt(const f64x2 v) {
  return f64x2(vsqrtq_f64(v));
}

// Approximate reciprocal square root
SIMD_INLINE f32x4 rsqrt_approx(const f32x4 v) {
  return f32x4(vrsqrteq_f32(v));
}

// ------------------------------ Floating-point rounding

// Toward nearest integer
SIMD_INLINE f32x4 round_nearest(const f32x4 v) {
  return f32x4(vrndnq_f32(v));
}
SIMD_INLINE f64x2 round_nearest(const f64x2 v) {
  return f64x2(vrndnq_f64(v));
}

// Toward +infinity, aka ceiling
SIMD_INLINE f32x4 round_pos_inf(const f32x4 v) {
  return f32x4(vrndpq_f32(v));
}
SIMD_INLINE f64x2 round_pos_inf(const f64x2 v) {
  return f64x2(vrndpq_f64(v));
}

// Toward -infinity, aka floor
SIMD_INLINE f32x4 round_neg_inf(const f32x4 v) {
  return f32x4(vrndmq_f32(v));
}
SIMD_INLINE f64x2 round_neg_inf(const f64x2 v) {
  return f64x2(vrndmq_f64(v));
}

// ------------------------------ Convert i32 <=> f32

SIMD_INLINE f32x4 f32_from_i32(const i32x4 v) {
  return f32x4(vcvtq_f32_s32(v));
}
// Uses current rounding mode, which defaults to round-to-nearest.
SIMD_INLINE i32x4 i32_from_f32(const f32x4 v) {
  return i32x4(vcvtnq_s32_f32(v));
}

// ------------------------------ Cast to/from floating-point representation

SIMD_INLINE f32x4 float_from_bits(const u32x4 v) {
  return f32x4(vreinterpretq_f32_u32(v));
}
SIMD_INLINE f32x4 float_from_bits(const i32x4 v) {
  return f32x4(vreinterpretq_f32_s32(v));
}
SIMD_INLINE i32x4 bits_from_float(const f32x4 v) {
  return i32x4(vreinterpretq_s32_f32(v));
}

SIMD_INLINE f64x2 float_from_bits(const u64x2 v) {
  return f64x2(vreinterpretq_f64_u64(v));
}
SIMD_INLINE f64x2 float_from_bits(const i64x2 v) {
  return f64x2(vreinterpretq_f64_s64(v));
}
SIMD_INLINE i64x2 bits_from_float(const f64x2 v) {
  return i64x2(vreinterpretq_s64_f64(v));
}

// ================================================== COMPARE

// Comparisons fill a lane with 1-bits if the condition is true, else 0.

// ------------------------------ Equality

// Unsigned
SIMD_INLINE u8x16 operator==(const u8x16 a, const u8x16 b) {
  return u8x16(vceqq_u8(a, b));
}
SIMD_INLINE u16x8 operator==(const u16x8 a, const u16x8 b) {
  return u16x8(vceqq_u16(a, b));
}
SIMD_INLINE u32x4 operator==(const u32x4 a, const u32x4 b) {
  return u32x4(vceqq_u32(a, b));
}
SIMD_INLINE u64x2 operator==(const u64x2 a, const u64x2 b) {
  return u64x2(vceqq_u64(a, b));
}

// Signed
SIMD_INLINE i8x16 operator==(const i8x16 a, const i8x16 b) {
  return i8x16(vceqq_s8(a, b));
}
SIMD_INLINE i16x8 operator==(const i16x8 a, const i16x8 b) {
  return i16x8(vceqq_s16(a, b));
}
SIMD_INLINE i32x4 operator==(const i32x4 a, const i32x4 b) {
  return i32x4(vceqq_s32(a, b));
}
SIMD_INLINE i64x2 operator==(const i64x2 a, const i64x2 b) {
  return i64x2(vceqq_s64(a, b));
}

// Float
SIMD_INLINE f32x4 operator==(const f32x4 a, const f32x4 b) {
  return f32x4(vceqq_f32(a, b));
}
SIMD_INLINE f64x2 operator==(const f64x2 a, const f64x2 b) {
  return f64x2(vceqq_f64(a, b));
}

// ------------------------------ Strict inequality

// Signed/float <
SIMD_INLINE i8x16 operator<(const i8x16 a, const i8x16 b) {
  return i8x16(vcltq_s8(a, b));
}
SIMD_INLINE i16x8 operator<(const i16x8 a, const i16x8 b) {
  return i16x8(vcltq_s16(a, b));
}
SIMD_INLINE i32x4 operator<(const i32x4 a, const i32x4 b) {
  return i32x4(vcltq_s32(a, b));
}
SIMD_INLINE i64x2 operator<(const i64x2 a, const i64x2 b) {
  return i64x2(vcltq_s64(a, b));
}
SIMD_INLINE f32x4 operator<(const f32x4 a, const f32x4 b) {
  return f32x4(vcltq_f32(a, b));
}
SIMD_INLINE f64x2 operator<(const f64x2 a, const f64x2 b) {
  return f64x2(vcltq_f64(a, b));
}

// Signed/float >
SIMD_INLINE i8x16 operator>(const i8x16 a, const i8x16 b) {
  return i8x16(vcgtq_s8(a, b));
}
SIMD_INLINE i16x8 operator>(const i16x8 a, const i16x8 b) {
  return i16x8(vcgtq_s16(a, b));
}
SIMD_INLINE i32x4 operator>(const i32x4 a, const i32x4 b) {
  return i32x4(vcgtq_s32(a, b));
}
SIMD_INLINE i64x2 operator>(const i64x2 a, const i64x2 b) {
  return i64x2(vcgtq_s64(a, b));
}
SIMD_INLINE f32x4 operator>(const f32x4 a, const f32x4 b) {
  return f32x4(vcgtq_f32(a, b));
}
SIMD_INLINE f64x2 operator>(const f64x2 a, const f64x2 b) {
  return f64x2(vcgtq_f64(a, b));
}

// ------------------------------ Weak inequality

// Float <= >=
SIMD_INLINE f32x4 operator<=(const f32x4 a, const f32x4 b) {
  return f32x4(vcleq_f32(a, b));
}
SIMD_INLINE f64x2 operator<=(const f64x2 a, const f64x2 b) {
  return f64x2(vcleq_f64(a, b));
}
SIMD_INLINE f32x4 operator>=(const f32x4 a, const f32x4 b) {
  return f32x4(vcgeq_f32(a, b));
}
SIMD_INLINE f64x2 operator>=(const f64x2 a, const f64x2 b) {
  return f64x2(vcgeq_f64(a, b));
}

// "Extensions": useful but quite performance-portable operations. We add
// functions to this namespace in multiple places.
namespace ext {

// Returns a bit array of the most significant bit of each byte in "v", i.e.
// sum_i=0..15 of (v[i] >> 7) << i; v[0] is the least-significant byte of "v".
// This is useful for testing/branching based on comparison results.
SIMD_INLINE uint32_t movemask(const u8x16 v) {
  static constexpr uint8x16_t kCollapseMask = {
      1, 2, 4, 8, 0x10, 0x20, 0x40, 0x80,
      1, 2, 4, 8, 0x10, 0x20, 0x40, 0x80,
  };
  int8x16_t signed_v = vreinterpretq_s8_u8(v);
  int8x16_t signed_mask = vshrq_n_s8(signed_v, 7);
  uint8x16_t values = vreinterpretq_u8_s8(signed_mask) & kCollapseMask;

  uint8x8_t c0 = vget_low_u8(vpaddq_u8(values, values));
  uint8x8_t c1 = vpadd_u8(c0, c0);
  uint8x8_t c2 = vpadd_u8(c1, c1);

  return vreinterpret_u16_u8(c2)[0];
}

// Returns the most significant bit of each float/double lane (see above).
SIMD_INLINE uint32_t movemask(const f32x4 v) {
  static constexpr uint32x4_t kCollapseMask = {
    1, 2, 4, 8
  };
  int32x4_t signed_v = vreinterpretq_s32_f32(v);
  int32x4_t signed_mask = vshrq_n_s32(signed_v, 31);
  uint32x4_t values = vreinterpretq_u32_s32(signed_mask) & kCollapseMask;
  return vaddvq_u32(values);
}
SIMD_INLINE uint32_t movemask(const f64x2 v) {
  static constexpr uint64x2_t kCollapseMask = {
    1, 2
  };
  int64x2_t signed_v = vreinterpretq_s64_f64(v);
  int64x2_t signed_mask = vshrq_n_s64(signed_v, 63);
  uint64x2_t values = vreinterpretq_u64_s64(signed_mask) & kCollapseMask;
  return (uint32_t) vaddvq_u64(values);
}

// Returns whether all lanes are equal to zero. Supported for all integer V.
SIMD_INLINE bool all_zero(const u64x2 v) {
  uint32x2_t a = vqmovn_u64(v);
  return vreinterpret_u64_u32(a)[0] == 0;
}
SIMD_INLINE bool all_zero(const i8x16 v) {
  return all_zero(u64x2(vreinterpretq_u64_s8(v)));
}
SIMD_INLINE bool all_zero(const u8x16 v) {
  return all_zero(u64x2(vreinterpretq_u64_u8(v)));
}
SIMD_INLINE bool all_zero(const i16x8 v) {
  return all_zero(u64x2(vreinterpretq_u64_s16(v)));
}
SIMD_INLINE bool all_zero(const u16x8 v) {
  return all_zero(u64x2(vreinterpretq_u64_u16(v)));
}
SIMD_INLINE bool all_zero(const i32x4 v) {
  return all_zero(u64x2(vreinterpretq_u64_s32(v)));
}
SIMD_INLINE bool all_zero(const u32x4 v) {
  return all_zero(u64x2(vreinterpretq_u64_u32(v)));
}
SIMD_INLINE bool all_zero(const i64x2 v) {
  return all_zero(u64x2(vreinterpretq_u64_s64(v)));
}
SIMD_INLINE bool all_zero(const f32x4 v) {
  return all_zero(u64x2(vreinterpretq_u64_f32(v)));
}
SIMD_INLINE bool all_zero(const f64x2 v) {
  return all_zero(u64x2(vreinterpretq_u64_f64(v)));
}

}  // namespace ext

// ================================================== LOGICAL

// ------------------------------ Bitwise AND

SIMD_INLINE i8x16 operator&(const i8x16 a, const i8x16 b) {
  return i8x16(vandq_s8(a, b));
}
SIMD_INLINE u8x16 operator&(const u8x16 a, const u8x16 b) {
  return u8x16(vandq_u8(a, b));
}
SIMD_INLINE i16x8 operator&(const i16x8 a, const i16x8 b) {
  return i16x8(vandq_s16(a, b));
}
SIMD_INLINE u16x8 operator&(const u16x8 a, const u16x8 b) {
  return u16x8(vandq_u16(a, b));
}
SIMD_INLINE i32x4 operator&(const i32x4 a, const i32x4 b) {
  return i32x4(vandq_s32(a, b));
}
SIMD_INLINE u32x4 operator&(const u32x4 a, const u32x4 b) {
  return u32x4(vandq_u32(a, b));
}
SIMD_INLINE i64x2 operator&(const i64x2 a, const i64x2 b) {
  return i64x2(vandq_s64(a, b));
}
SIMD_INLINE u64x2 operator&(const u64x2 a, const u64x2 b) {
  return u64x2(vandq_u64(a, b));
}
SIMD_INLINE f32x4 operator&(const f32x4 a, const f32x4 b) {
  uint32x4_t a_int = vreinterpretq_u32_f32(a);
  uint32x4_t b_int = vreinterpretq_u32_f32(b);
  uint32x4_t r_int = vandq_u32(a_int, b_int);
  return f32x4(vreinterpretq_f32_u32(r_int));
}
SIMD_INLINE f64x2 operator&(const f64x2 a, const f64x2 b) {
  uint64x2_t a_int = vreinterpretq_u64_f64(a);
  uint64x2_t b_int = vreinterpretq_u64_f64(b);
  uint64x2_t r_int = vandq_u64(a_int, b_int);
  return f64x2(vreinterpretq_f64_u64(r_int));
}

// ------------------------------ Bitwise AND-NOT

// Returns ~not_mask & mask.
SIMD_INLINE i8x16 andnot(const i8x16 not_mask, const i8x16 mask) {
  return i8x16(vbicq_s8(mask, not_mask));
}
SIMD_INLINE u8x16 andnot(const u8x16 not_mask, const u8x16 mask) {
  return u8x16(vbicq_u8(mask, not_mask));
}
SIMD_INLINE i16x8 andnot(const i16x8 not_mask, const i16x8 mask) {
  return i16x8(vbicq_s16(mask, not_mask));
}
SIMD_INLINE u16x8 andnot(const u16x8 not_mask, const u16x8 mask) {
  return u16x8(vbicq_u16(mask, not_mask));
}
SIMD_INLINE i32x4 andnot(const i32x4 not_mask, const i32x4 mask) {
  return i32x4(vbicq_s32(mask, not_mask));
}
SIMD_INLINE u32x4 andnot(const u32x4 not_mask, const u32x4 mask) {
  return u32x4(vbicq_u32(mask, not_mask));
}
SIMD_INLINE i64x2 andnot(const i64x2 not_mask, const i64x2 mask) {
  return i64x2(vbicq_s64(mask, not_mask));
}
SIMD_INLINE u64x2 andnot(const u64x2 not_mask, const u64x2 mask) {
  return u64x2(vbicq_u64(mask, not_mask));
}
SIMD_INLINE f32x4 andnot(const f32x4 not_mask, const f32x4 mask) {
  uint32x4_t not_mask_int = vreinterpretq_u32_f32(not_mask);
  uint32x4_t mask_int = vreinterpretq_u32_f32(mask);
  uint32x4_t r_int = vbicq_u32(mask_int, not_mask_int);
  return f32x4(vreinterpretq_f32_u32(r_int));
}
SIMD_INLINE f64x2 andnot(const f64x2 not_mask, const f64x2 mask) {
  uint64x2_t not_mask_int = vreinterpretq_u64_f64(not_mask);
  uint64x2_t mask_int = vreinterpretq_u64_f64(mask);
  uint64x2_t r_int = vbicq_u64(mask_int, not_mask_int);
  return f64x2(vreinterpretq_f64_u64(r_int));
}

// ------------------------------ Bitwise OR

SIMD_INLINE i8x16 operator|(const i8x16 a, const i8x16 b) {
  return i8x16(vorrq_s8(a, b));
}
SIMD_INLINE u8x16 operator|(const u8x16 a, const u8x16 b) {
  return u8x16(vorrq_u8(a, b));
}
SIMD_INLINE i16x8 operator|(const i16x8 a, const i16x8 b) {
  return i16x8(vorrq_s16(a, b));
}
SIMD_INLINE u16x8 operator|(const u16x8 a, const u16x8 b) {
  return u16x8(vorrq_u16(a, b));
}
SIMD_INLINE i32x4 operator|(const i32x4 a, const i32x4 b) {
  return i32x4(vorrq_s32(a, b));
}
SIMD_INLINE u32x4 operator|(const u32x4 a, const u32x4 b) {
  return u32x4(vorrq_u32(a, b));
}
SIMD_INLINE i64x2 operator|(const i64x2 a, const i64x2 b) {
  return i64x2(vorrq_s64(a, b));
}
SIMD_INLINE u64x2 operator|(const u64x2 a, const u64x2 b) {
  return u64x2(vorrq_u64(a, b));
}
SIMD_INLINE f32x4 operator|(const f32x4 a, const f32x4 b) {
  uint32x4_t a_int = vreinterpretq_u32_f32(a);
  uint32x4_t b_int = vreinterpretq_u32_f32(b);
  uint32x4_t r_int = vorrq_u32(a_int, b_int);
  return f32x4(vreinterpretq_f32_u32(r_int));
}
SIMD_INLINE f64x2 operator|(const f64x2 a, const f64x2 b) {
  uint64x2_t a_int = vreinterpretq_u64_f64(a);
  uint64x2_t b_int = vreinterpretq_u64_f64(b);
  uint64x2_t r_int = vorrq_u64(a_int, b_int);
  return f64x2(vreinterpretq_f64_u64(r_int));
}

// ------------------------------ Bitwise XOR

SIMD_INLINE i8x16 operator^(const i8x16 a, const i8x16 b) {
  return i8x16(veorq_s8(a, b));
}
SIMD_INLINE u8x16 operator^(const u8x16 a, const u8x16 b) {
  return u8x16(veorq_u8(a, b));
}
SIMD_INLINE i16x8 operator^(const i16x8 a, const i16x8 b) {
  return i16x8(veorq_s16(a, b));
}
SIMD_INLINE u16x8 operator^(const u16x8 a, const u16x8 b) {
  return u16x8(veorq_u16(a, b));
}
SIMD_INLINE i32x4 operator^(const i32x4 a, const i32x4 b) {
  return i32x4(veorq_s32(a, b));
}
SIMD_INLINE u32x4 operator^(const u32x4 a, const u32x4 b) {
  return u32x4(veorq_u32(a, b));
}
SIMD_INLINE i64x2 operator^(const i64x2 a, const i64x2 b) {
  return i64x2(veorq_s64(a, b));
}
SIMD_INLINE u64x2 operator^(const u64x2 a, const u64x2 b) {
  return u64x2(veorq_u64(a, b));
}
SIMD_INLINE f32x4 operator^(const f32x4 a, const f32x4 b) {
  uint32x4_t a_int = vreinterpretq_u32_f32(a);
  uint32x4_t b_int = vreinterpretq_u32_f32(b);
  uint32x4_t r_int = veorq_u32(a_int, b_int);
  return f32x4(vreinterpretq_f32_u32(r_int));
}
SIMD_INLINE f64x2 operator^(const f64x2 a, const f64x2 b) {
  uint64x2_t a_int = vreinterpretq_u64_f64(a);
  uint64x2_t b_int = vreinterpretq_u64_f64(b);
  uint64x2_t r_int = veorq_u64(a_int, b_int);
  return f64x2(vreinterpretq_f64_u64(r_int));
}

// ================================================== MEMORY

// ------------------------------ Load 128

SIMD_INLINE u8x16 load_unaligned(Full<uint8_t, ARM8>,
                                 const uint8_t* SIMD_RESTRICT aligned) {
  return u8x16(vld1q_u8(aligned));
}
SIMD_INLINE u16x8 load_unaligned(Full<uint16_t, ARM8>,
                                 const uint16_t* SIMD_RESTRICT aligned) {
  return u16x8(vld1q_u16(aligned));
}
SIMD_INLINE u32x4 load_unaligned(Full<uint32_t, ARM8>,
                                 const uint32_t* SIMD_RESTRICT aligned) {
  return u32x4(vld1q_u32(aligned));
}
SIMD_INLINE u64x2 load_unaligned(Full<uint64_t, ARM8>,
                                 const uint64_t* SIMD_RESTRICT aligned) {
  return u64x2(vld1q_u64(aligned));
}
SIMD_INLINE i8x16 load_unaligned(Full<int8_t, ARM8>,
                                 const int8_t* SIMD_RESTRICT aligned) {
  return i8x16(vld1q_s8(aligned));
}
SIMD_INLINE i16x8 load_unaligned(Full<int16_t, ARM8>,
                                 const int16_t* SIMD_RESTRICT aligned) {
  return i16x8(vld1q_s16(aligned));
}
SIMD_INLINE i32x4 load_unaligned(Full<int32_t, ARM8>,
                                 const int32_t* SIMD_RESTRICT aligned) {
  return i32x4(vld1q_s32(aligned));
}
SIMD_INLINE i64x2 load_unaligned(Full<int64_t, ARM8>,
                                 const int64_t* SIMD_RESTRICT aligned) {
  return i64x2(vld1q_s64(aligned));
}
SIMD_INLINE f32x4 load_unaligned(Full<float, ARM8>,
                                 const float* SIMD_RESTRICT aligned) {
  return f32x4(vld1q_f32(aligned));
}
SIMD_INLINE f64x2 load_unaligned(Full<double, ARM8>,
                                 const double* SIMD_RESTRICT aligned) {
  return f64x2(vld1q_f64(aligned));
}

template <typename T>
SIMD_INLINE vec_arm8<T> load(Full<T, ARM8> d, const T* SIMD_RESTRICT p) {
  return load_unaligned(d, p);
}

// 128-bit SIMD => nothing to duplicate, same as an unaligned load.
template <typename T>
SIMD_INLINE dup128x1<T> load_dup128(Full<T, ARM8> d,
                                    const T* const SIMD_RESTRICT p) {
  return dup128x1<T>(load_unaligned(d, p));
}

// ------------------------------ Load 64

SIMD_INLINE u8x8 load(Desc<uint8_t, 8, ARM8>, const uint8_t* SIMD_RESTRICT p) {
  return u8x8(vld1_u8(p));
}
SIMD_INLINE u16x4 load(Desc<uint16_t, 4, ARM8>,
                       const uint16_t* SIMD_RESTRICT p) {
  return u16x4(vld1_u16(p));
}
SIMD_INLINE u32x2 load(Desc<uint32_t, 2, ARM8>,
                       const uint32_t* SIMD_RESTRICT p) {
  return u32x2(vld1_u32(p));
}
SIMD_INLINE u64x1 load(Desc<uint64_t, 1, ARM8>,
                       const uint64_t* SIMD_RESTRICT p) {
  return u64x1(vld1_u64(p));
}
SIMD_INLINE i8x8 load(Desc<int8_t, 8, ARM8>, const int8_t* SIMD_RESTRICT p) {
  return i8x8(vld1_s8(p));
}
SIMD_INLINE i16x4 load(Desc<int16_t, 4, ARM8>, const int16_t* SIMD_RESTRICT p) {
  return i16x4(vld1_s16(p));
}
SIMD_INLINE i32x2 load(Desc<int32_t, 2, ARM8>, const int32_t* SIMD_RESTRICT p) {
  return i32x2(vld1_s32(p));
}
SIMD_INLINE i64x1 load(Desc<int64_t, 1, ARM8>, const int64_t* SIMD_RESTRICT p) {
  return i64x1(vld1_s64(p));
}
SIMD_INLINE f32x2 load(Desc<float, 2, ARM8>, const float* SIMD_RESTRICT p) {
  return f32x2(vld1_f32(p));
}
SIMD_INLINE f64x1 load(Desc<double, 1, ARM8>, const double* SIMD_RESTRICT p) {
  return f64x1(vld1_f64(p));
}

// ------------------------------ Load 32

// In the following load functions, |a| is purposely undefined.
// It is a required parameter to the intrinsic, however
// we don't actually care what is in it, and we don't want
// to introduce extra overhead by initializing it to something.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wuninitialized"

SIMD_INLINE u8x4 load(Desc<uint8_t, 4, ARM8>, const uint8_t* SIMD_RESTRICT p) {
  uint32x2_t a;
  uint32x2_t b = vld1_lane_u32(reinterpret_cast<const uint32_t*>(p), a, 0);
  return u8x4(vreinterpret_u8_u32(b));
}
SIMD_INLINE u16x2 load(Desc<uint16_t, 2, ARM8>,
                       const uint16_t* SIMD_RESTRICT p) {
  uint32x2_t a;
  uint32x2_t b = vld1_lane_u32(reinterpret_cast<const uint32_t*>(p), a, 0);
  return u16x2(vreinterpret_u16_u32(b));
}
SIMD_INLINE u32x1 load(Desc<uint32_t, 1, ARM8>,
                       const uint32_t* SIMD_RESTRICT p) {
  uint32x2_t a;
  uint32x2_t b = vld1_lane_u32(p, a, 0);
  return u32x1(b);
}
SIMD_INLINE i8x4 load(Desc<int8_t, 4, ARM8>, const int8_t* SIMD_RESTRICT p) {
  int32x2_t a;
  int32x2_t b = vld1_lane_s32(reinterpret_cast<const int32_t*>(p), a, 0);
  return i8x4(vreinterpret_s8_s32(b));
}
SIMD_INLINE i16x2 load(Desc<int16_t, 2, ARM8>, const int16_t* SIMD_RESTRICT p) {
  int32x2_t a;
  int32x2_t b = vld1_lane_s32(reinterpret_cast<const int32_t*>(p), a, 0);
  return i16x2(vreinterpret_s16_s32(b));
}
SIMD_INLINE i32x1 load(Desc<int32_t, 1, ARM8>, const int32_t* SIMD_RESTRICT p) {
  int32x2_t a;
  int32x2_t b = vld1_lane_s32(p, a, 0);
  return i32x1(b);
}
SIMD_INLINE f32x1 load(Desc<float, 1, ARM8>, const float* SIMD_RESTRICT p) {
  float32x2_t a;
  float32x2_t b = vld1_lane_f32(p, a, 0);
  return f32x1(b);
}

#pragma clang diagnostic pop

// ------------------------------ Store 128

SIMD_INLINE void store_unaligned(const u8x16 v, Full<uint8_t, ARM8>,
                                 uint8_t* SIMD_RESTRICT aligned) {
  vst1q_u8(aligned, v);
}
SIMD_INLINE void store_unaligned(const u16x8 v, Full<uint16_t, ARM8>,
                                 uint16_t* SIMD_RESTRICT aligned) {
  vst1q_u16(aligned, v);
}
SIMD_INLINE void store_unaligned(const u32x4 v, Full<uint32_t, ARM8>,
                                 uint32_t* SIMD_RESTRICT aligned) {
  vst1q_u32(aligned, v);
}
SIMD_INLINE void store_unaligned(const u64x2 v, Full<uint64_t, ARM8>,
                                 uint64_t* SIMD_RESTRICT aligned) {
  vst1q_u64(aligned, v);
}
SIMD_INLINE void store_unaligned(const i8x16 v, Full<int8_t, ARM8>,
                                 int8_t* SIMD_RESTRICT aligned) {
  vst1q_s8(aligned, v);
}
SIMD_INLINE void store_unaligned(const i16x8 v, Full<int16_t, ARM8>,
                                 int16_t* SIMD_RESTRICT aligned) {
  vst1q_s16(aligned, v);
}
SIMD_INLINE void store_unaligned(const i32x4 v, Full<int32_t, ARM8>,
                                 int32_t* SIMD_RESTRICT aligned) {
  vst1q_s32(aligned, v);
}
SIMD_INLINE void store_unaligned(const i64x2 v, Full<int64_t, ARM8>,
                                 int64_t* SIMD_RESTRICT aligned) {
  vst1q_s64(aligned, v);
}
SIMD_INLINE void store_unaligned(const f32x4 v, Full<float, ARM8>,
                                 float* SIMD_RESTRICT aligned) {
  vst1q_f32(aligned, v);
}
SIMD_INLINE void store_unaligned(const f64x2 v, Full<double, ARM8>,
                                 double* SIMD_RESTRICT aligned) {
  vst1q_f64(aligned, v);
}

template <typename T, size_t N>
SIMD_INLINE void store(Vec<T, N, ARM8> v, Desc<T, N, ARM8> d,
                       T* SIMD_RESTRICT p) {
  store_unaligned(v, d, p);
}

template <typename T, size_t N>
SIMD_INLINE void store(const dup128x1<T, N> v, Full<T, ARM8> d,
                       T* SIMD_RESTRICT aligned) {
  store(vec_arm8<T>(v.raw), d, aligned);
}

// ------------------------------ Store 64

SIMD_INLINE void store(const u8x8 v, Desc<uint8_t, 8, ARM8>,
                       uint8_t* SIMD_RESTRICT p) {
  vst1_u8(p, v);
}
SIMD_INLINE void store(const u16x4 v, Desc<uint16_t, 4, ARM8>,
                       uint16_t* SIMD_RESTRICT p) {
  vst1_u16(p, v);
}
SIMD_INLINE void store(const u32x2 v, Desc<uint32_t, 2, ARM8>,
                       uint32_t* SIMD_RESTRICT p) {
  vst1_u32(p, v);
}
SIMD_INLINE void store(const u64x1 v, Desc<uint64_t, 1, ARM8>,
                       uint64_t* SIMD_RESTRICT p) {
  vst1_u64(p, v);
}
SIMD_INLINE void store(const i8x8 v, Desc<int8_t, 8, ARM8>,
                       int8_t* SIMD_RESTRICT p) {
  vst1_s8(p, v);
}
SIMD_INLINE void store(const i16x4 v, Desc<int16_t, 4, ARM8>,
                       int16_t* SIMD_RESTRICT p) {
  vst1_s16(p, v);
}
SIMD_INLINE void store(const i32x2 v, Desc<int32_t, 2, ARM8>,
                       int32_t* SIMD_RESTRICT p) {
  vst1_s32(p, v);
}
SIMD_INLINE void store(const i64x1 v, Desc<int64_t, 1, ARM8>,
                       int64_t* SIMD_RESTRICT p) {
  vst1_s64(p, v);
}
SIMD_INLINE void store(const f32x2 v, Desc<float, 2, ARM8>,
                       float* SIMD_RESTRICT p) {
  vst1_f32(p, v);
}
SIMD_INLINE void store(const f64x1 v, Desc<double, 1, ARM8>,
                       double* SIMD_RESTRICT p) {
  vst1_f64(p, v);
}

// ------------------------------ Store 32

SIMD_INLINE void store(const u8x4 v, Desc<uint8_t, 4, ARM8>,
                       uint8_t* SIMD_RESTRICT p) {
  uint32x2_t a = vreinterpret_u32_u8(v);
  vst1_lane_u32(p, a, 0);
}
SIMD_INLINE void store(const u16x2 v, Desc<uint16_t, 2, ARM8>,
                       uint16_t* SIMD_RESTRICT p) {
  uint32x2_t a = vreinterpret_u32_u16(v);
  vst1_lane_u32(p, a, 0);
}
SIMD_INLINE void store(const u32x1 v, Desc<uint32_t, 1, ARM8>,
                       uint32_t* SIMD_RESTRICT p) {
  vst1_lane_u32(p, v, 0);
}
SIMD_INLINE void store(const i8x4 v, Desc<int8_t, 4, ARM8>,
                       int8_t* SIMD_RESTRICT p) {
  int32x2_t a = vreinterpret_s32_s8(v);
  vst1_lane_s32(p, a, 0);
}
SIMD_INLINE void store(const i16x2 v, Desc<int16_t, 2, ARM8>,
                       int16_t* SIMD_RESTRICT p) {
  int32x2_t a = vreinterpret_s32_s16(v);
  vst1_lane_s32(p, a, 0);
}
SIMD_INLINE void store(const i32x1 v, Desc<int32_t, 1, ARM8>,
                       int32_t* SIMD_RESTRICT p) {
  vst1_lane_s32(p, v, 0);
}
SIMD_INLINE void store(const f32x1 v, Desc<float, 1, ARM8>,
                       float* SIMD_RESTRICT p) {
  vst1_lane_f32(p, v, 0);
}

// ------------------------------ Non-temporal stores

// Same as aligned stores on non-x86.

template <typename T>
SIMD_INLINE void stream(const vec_arm8<T> v, Full<T, ARM8> d,
                        T* SIMD_RESTRICT aligned) {
  store(v, d, aligned);
}

// ================================================== SWIZZLE

// ------------------------------ 'Extract' other half (see any_part)

// These copy hi into lo
SIMD_INLINE u8x8 other_half(const u8x16 v) { return u8x8(vget_high_u8(v.raw)); }
SIMD_INLINE i8x8 other_half(const i8x16 v) { return i8x8(vget_high_s8(v.raw)); }
SIMD_INLINE u16x4 other_half(const u16x8 v) {
  return u16x4(vget_high_u16(v.raw));
}
SIMD_INLINE i16x4 other_half(const i16x8 v) {
  return i16x4(vget_high_s16(v.raw));
}
SIMD_INLINE u32x2 other_half(const u32x4 v) {
  return u32x2(vget_high_u32(v.raw));
}
SIMD_INLINE i32x2 other_half(const i32x4 v) {
  return i32x2(vget_high_s32(v.raw));
}
SIMD_INLINE u64x1 other_half(const u64x2 v) {
  return u64x1(vget_high_u64(v.raw));
}
SIMD_INLINE i64x1 other_half(const i64x2 v) {
  return i64x1(vget_high_s64(v.raw));
}
SIMD_INLINE f32x2 other_half(const f32x4 v) {
  return f32x2(vget_high_f32(v.raw));
}
SIMD_INLINE f64x1 other_half(const f64x2 v) {
  return f64x1(vget_high_f64(v.raw));
}

// ------------------------------ Extract from 2x 128-bit at constant offset

// Extracts 128 bits from <hi, lo> by skipping the least-significant kBytes.
template <int kBytes>
SIMD_INLINE u8x16 extract_concat_bytes(const u8x16 hi,
                                       const u8x16 lo) {
  static_assert(0 < kBytes && kBytes < 16, "kBytes must be in [1, 15]");
  return u8x16(vextq_u8(lo, hi, kBytes));
}
template <int kBytes>
SIMD_INLINE i8x16 extract_concat_bytes(const i8x16 hi,
                                       const i8x16 lo) {
  static_assert(0 < kBytes && kBytes < 16, "kBytes must be in [1, 15]");
  uint8x16_t u8hi = vreinterpretq_u8_s8(hi);
  uint8x16_t u8lo = vreinterpretq_u8_s8(lo);
  uint8x16_t r = vextq_u8(u8lo, u8hi, kBytes);
  return i8x16(vreinterpretq_s8_u8(r));
}
template <int kBytes>
SIMD_INLINE i16x8 extract_concat_bytes(const i16x8 hi,
                                       const i16x8 lo) {
  static_assert(0 < kBytes && kBytes < 16, "kBytes must be in [1, 15]");
  uint8x16_t u8hi = vreinterpretq_u8_s16(hi);
  uint8x16_t u8lo = vreinterpretq_u8_s16(lo);
  uint8x16_t r = vextq_u8(u8lo, u8hi, kBytes);
  return i16x8(vreinterpretq_s16_u8(r));
}
template <int kBytes>
SIMD_INLINE u16x8 extract_concat_bytes(const u16x8 hi,
                                       const u16x8 lo) {
  static_assert(0 < kBytes && kBytes < 16, "kBytes must be in [1, 15]");
  uint8x16_t u8hi = vreinterpretq_u8_u16(hi);
  uint8x16_t u8lo = vreinterpretq_u8_u16(lo);
  uint8x16_t r = vextq_u8(u8lo, u8hi, kBytes);
  return u16x8(vreinterpretq_u16_u8(r));
}
template <int kBytes>
SIMD_INLINE i32x4 extract_concat_bytes(const i32x4 hi,
                                       const i32x4 lo) {
  static_assert(0 < kBytes && kBytes < 16, "kBytes must be in [1, 15]");
  uint8x16_t u8hi = vreinterpretq_u8_s32(hi);
  uint8x16_t u8lo = vreinterpretq_u8_s32(lo);
  uint8x16_t r = vextq_u8(u8lo, u8hi, kBytes);
  return i32x4(vreinterpretq_s32_u8(r));
}
template <int kBytes>
SIMD_INLINE u32x4 extract_concat_bytes(const u32x4 hi,
                                       const u32x4 lo) {
  static_assert(0 < kBytes && kBytes < 16, "kBytes must be in [1, 15]");
  uint8x16_t u8hi = vreinterpretq_u8_u32(hi);
  uint8x16_t u8lo = vreinterpretq_u8_u32(lo);
  uint8x16_t r = vextq_u8(u8lo, u8hi, kBytes);
  return u32x4(vreinterpretq_u32_u8(r));
}
template <int kBytes>
SIMD_INLINE i64x2 extract_concat_bytes(const i64x2 hi,
                                       const i64x2 lo) {
  static_assert(0 < kBytes && kBytes < 16, "kBytes must be in [1, 15]");
  uint8x16_t u8hi = vreinterpretq_u8_s64(hi);
  uint8x16_t u8lo = vreinterpretq_u8_s64(lo);
  uint8x16_t r = vextq_u8(u8lo, u8hi, kBytes);
  return i64x2(vreinterpretq_s64_u8(r));
}
template <int kBytes>
SIMD_INLINE u64x2 extract_concat_bytes(const u64x2 hi,
                                       const u64x2 lo) {
  static_assert(0 < kBytes && kBytes < 16, "kBytes must be in [1, 15]");
  uint8x16_t u8hi = vreinterpretq_u8_u64(hi);
  uint8x16_t u8lo = vreinterpretq_u8_u64(lo);
  uint8x16_t r = vextq_u8(u8lo, u8hi, kBytes);
  return u64x2(vreinterpretq_u64_u8(r));
}
template <int kBytes>
SIMD_INLINE f32x4 extract_concat_bytes(const f32x4 hi,
                                       const f32x4 lo) {
  static_assert(0 < kBytes && kBytes < 16, "kBytes must be in [1, 15]");
  uint8x16_t u8hi = vreinterpretq_u8_f32(hi);
  uint8x16_t u8lo = vreinterpretq_u8_f32(lo);
  uint8x16_t r = vextq_u8(u8lo, u8hi, kBytes);
  return f32x4(vreinterpretq_f32_u8(r));
}
template <int kBytes>
SIMD_INLINE f64x2 extract_concat_bytes(const f64x2 hi,
                                       const f64x2 lo) {
  static_assert(0 < kBytes && kBytes < 16, "kBytes must be in [1, 15]");
  uint8x16_t u8hi = vreinterpretq_u8_f64(hi);
  uint8x16_t u8lo = vreinterpretq_u8_f64(lo);
  uint8x16_t r = vextq_u8(u8lo, u8hi, kBytes);
  return f64x2(vreinterpretq_f64_u8(r));
}

// ------------------------------ Shift vector by constant #bytes

// 0x01..0F, kBytes = 1 => 0x02..0F00
template <int kBytes, typename T, size_t N>
SIMD_INLINE vec_arm8<T, N> shift_bytes_left(const vec_arm8<T, N> v) {
  return extract_concat_bytes<16 - kBytes>(v, setzero(Full<T, ARM8>()));
}

// 0x01..0F, kBytes = 1 => 0x0001..0E
template <int kBytes, typename T, size_t N>
SIMD_INLINE vec_arm8<T, N> shift_bytes_right(const vec_arm8<T, N> v) {
  return extract_concat_bytes<kBytes>(setzero(Full<T, ARM8>()), v);
}

// ------------------------------ Broadcast/splat any lane

// Unsigned
template <int kLane>
SIMD_INLINE u32x4 broadcast(const u32x4 v) {
  static_assert(0 <= kLane && kLane < 4, "Invalid lane");
  return u32x4(vdupq_laneq_u32(v, kLane));
}
template <int kLane>
SIMD_INLINE u64x2 broadcast(const u64x2 v) {
  static_assert(0 <= kLane && kLane < 2, "Invalid lane");
  return u64x2(vdupq_laneq_u64(v, kLane));
}

// Signed
template <int kLane>
SIMD_INLINE i32x4 broadcast(const i32x4 v) {
  static_assert(0 <= kLane && kLane < 4, "Invalid lane");
  return i32x4(vdupq_laneq_s32(v, kLane));
}
template <int kLane>
SIMD_INLINE i64x2 broadcast(const i64x2 v) {
  static_assert(0 <= kLane && kLane < 2, "Invalid lane");
  return i64x2(vdupq_laneq_s64(v, kLane));
}

// Float
template <int kLane>
SIMD_INLINE f32x4 broadcast(const f32x4 v) {
  static_assert(0 <= kLane && kLane < 4, "Invalid lane");
  return f32x4(vdupq_laneq_f32(v, kLane));
}
template <int kLane>
SIMD_INLINE f64x2 broadcast(const f64x2 v) {
  static_assert(0 <= kLane && kLane < 2, "Invalid lane");
  return f64x2(vdupq_laneq_f64(v, kLane));
}

// ------------------------------ Shuffle bytes with variable indices

template <size_t N>
SIMD_INLINE Vec<uint8_t, N, ARM8> cast_to(Desc<uint8_t, N, ARM8>, u8x16 v) {
  return v;
}
template <size_t N>
SIMD_INLINE Vec<uint8_t, N, ARM8> cast_to(Desc<uint8_t, N, ARM8>, u16x8 v) {
  return Vec<uint8_t, N, ARM8>(vreinterpretq_u8_u16(v));
}
template <size_t N>
SIMD_INLINE Vec<uint8_t, N, ARM8> cast_to(Desc<uint8_t, N, ARM8>, u32x4 v) {
  return Vec<uint8_t, N, ARM8>(vreinterpretq_u8_u32(v));
}
template <size_t N>
SIMD_INLINE Vec<uint8_t, N, ARM8> cast_to(Desc<uint8_t, N, ARM8>, u64x2 v) {
  return Vec<uint8_t, N, ARM8>(vreinterpretq_u8_u64(v));
}

template <size_t N>
SIMD_INLINE Vec<uint8_t, N, ARM8> cast_to(Desc<uint8_t, N, ARM8>, i8x16 v) {
  return Vec<uint8_t, N, ARM8>(vreinterpretq_u8_s8(v));
}
template <size_t N>
SIMD_INLINE Vec<uint8_t, N, ARM8> cast_to(Desc<uint8_t, N, ARM8>, i16x8 v) {
  return Vec<uint8_t, N, ARM8>(vreinterpretq_u8_s16(v));
}
template <size_t N>
SIMD_INLINE Vec<uint8_t, N, ARM8> cast_to(Desc<uint8_t, N, ARM8>, i32x4 v) {
  return Vec<uint8_t, N, ARM8>(vreinterpretq_u8_s32(v));
}
template <size_t N>
SIMD_INLINE Vec<uint8_t, N, ARM8> cast_to(Desc<uint8_t, N, ARM8>, i64x2 v) {
  return Vec<uint8_t, N, ARM8>(vreinterpretq_u8_s32(v));
}

// Returns vector of bytes[from[i]]. "from" is also interpreted as bytes:
// either valid indices in [0, 16) or >= 0x80 to zero the i-th output byte.
// TODO(janwas): replace with template<class V, class VI> that casts args?
template <typename T, typename TI>
SIMD_INLINE vec_arm8<T> shuffle_bytes(const vec_arm8<T> bytes,
                                      const vec_arm8<TI> from) {
  const Full<uint8_t> d8;
  return vec_arm8<T>(vqtbl1q_u8(cast_to(d8, bytes), cast_to(d8, from)));
}

// ------------------------------ Hard-coded shuffles

// Notation: let i32x4 have lanes 3,2,1,0 (0 is least-significant).
// shuffle_0321 rotates one lane to the right (the previous least-significant
// lane is now most-significant). These could also be implemented via
// extract_concat_bytes but the shuffle_abcd notation is more convenient.

// Swap 64-bit halves
template <typename T>
SIMD_INLINE vec_arm8<T> shuffle_1032(const vec_arm8<T> v) {
  return extract_concat_bytes<8>(v, v);
}
template <typename T>
SIMD_INLINE vec_arm8<T> shuffle_01(const vec_arm8<T> v) {
  return extract_concat_bytes<8>(v, v);
}

// Rotate right 32 bits
template <typename T>
SIMD_INLINE vec_arm8<T> shuffle_0321(const vec_arm8<T> v) {
  return extract_concat_bytes<4>(v, v);
}

// Rotate left 32 bits
template <typename T>
SIMD_INLINE vec_arm8<T> shuffle_2103(const vec_arm8<T> v) {
  return extract_concat_bytes<12>(v, v);
}

// ------------------------------ Interleave lanes

// Interleaves lanes from halves of the 128-bit blocks of "a" (which provides
// the least-significant lane) and "b". To concatenate two half-width integers
// into one, use zip_lo/hi instead (also works with scalar).

SIMD_INLINE u8x16 interleave_lo(const u8x16 a, const u8x16 b) {
  return u8x16(vzip1q_u8(a, b));
}
SIMD_INLINE u16x8 interleave_lo(const u16x8 a, const u16x8 b) {
  return u16x8(vzip1q_u16(a, b));
}
SIMD_INLINE u32x4 interleave_lo(const u32x4 a, const u32x4 b) {
  return u32x4(vzip1q_u32(a, b));
}
SIMD_INLINE u64x2 interleave_lo(const u64x2 a, const u64x2 b) {
  return u64x2(vzip1q_u64(a, b));
}

SIMD_INLINE i8x16 interleave_lo(const i8x16 a, const i8x16 b) {
  return i8x16(vzip1q_s8(a, b));
}
SIMD_INLINE i16x8 interleave_lo(const i16x8 a, const i16x8 b) {
  return i16x8(vzip1q_s16(a, b));
}
SIMD_INLINE i32x4 interleave_lo(const i32x4 a, const i32x4 b) {
  return i32x4(vzip1q_s32(a, b));
}
SIMD_INLINE i64x2 interleave_lo(const i64x2 a, const i64x2 b) {
  return i64x2(vzip1q_s64(a, b));
}

SIMD_INLINE f32x4 interleave_lo(const f32x4 a, const f32x4 b) {
  return f32x4(vzip1q_f32(a, b));
}
SIMD_INLINE f64x2 interleave_lo(const f64x2 a, const f64x2 b) {
  return f64x2(vzip1q_f64(a, b));
}

SIMD_INLINE u8x16 interleave_hi(const u8x16 a, const u8x16 b) {
  return u8x16(vzip2q_u8(a, b));
}
SIMD_INLINE u16x8 interleave_hi(const u16x8 a, const u16x8 b) {
  return u16x8(vzip2q_u16(a, b));
}
SIMD_INLINE u32x4 interleave_hi(const u32x4 a, const u32x4 b) {
  return u32x4(vzip2q_u32(a, b));
}
SIMD_INLINE u64x2 interleave_hi(const u64x2 a, const u64x2 b) {
  return u64x2(vzip2q_u64(a, b));
}

SIMD_INLINE i8x16 interleave_hi(const i8x16 a, const i8x16 b) {
  return i8x16(vzip2q_s8(a, b));
}
SIMD_INLINE i16x8 interleave_hi(const i16x8 a, const i16x8 b) {
  return i16x8(vzip2q_s16(a, b));
}
SIMD_INLINE i32x4 interleave_hi(const i32x4 a, const i32x4 b) {
  return i32x4(vzip2q_s32(a, b));
}
SIMD_INLINE i64x2 interleave_hi(const i64x2 a, const i64x2 b) {
  return i64x2(vzip2q_s64(a, b));
}

SIMD_INLINE f32x4 interleave_hi(const f32x4 a, const f32x4 b) {
  return f32x4(vzip2q_f32(a, b));
}
SIMD_INLINE f64x2 interleave_hi(const f64x2 a, const f64x2 b) {
  return f64x2(vzip2q_s64(a, b));
}

//// ------------------------------ Zip lanes
//
//// Same as interleave_*, except that the return lanes are double-width
/// integers; / this is necessary because the single-lane scalar cannot return two
/// values.
//
// SIMD_INLINE u16x8 zip_lo(const u8x16 a, const u8x16 b) {
//  return u16x8(_mm_unpacklo_epi8(a, b));
//}
// SIMD_INLINE u32x4 zip_lo(const u16x8 a, const u16x8 b) {
//  return u32x4(_mm_unpacklo_epi16(a, b));
//}
// SIMD_INLINE u64x2 zip_lo(const u32x4 a, const u32x4 b) {
//  return u64x2(_mm_unpacklo_epi32(a, b));
//}
//
// SIMD_INLINE i16x8 zip_lo(const i8x16 a, const i8x16 b) {
//  return i16x8(_mm_unpacklo_epi8(a, b));
//}
// SIMD_INLINE i32x4 zip_lo(const i16x8 a, const i16x8 b) {
//  return i32x4(_mm_unpacklo_epi16(a, b));
//}
// SIMD_INLINE i64x2 zip_lo(const i32x4 a, const i32x4 b) {
//  return i64x2(_mm_unpacklo_epi32(a, b));
//}
//
// SIMD_INLINE u16x8 zip_hi(const u8x16 a, const u8x16 b) {
//  return u16x8(_mm_unpackhi_epi8(a, b));
//}
// SIMD_INLINE u32x4 zip_hi(const u16x8 a, const u16x8 b) {
//  return u32x4(_mm_unpackhi_epi16(a, b));
//}
// SIMD_INLINE u64x2 zip_hi(const u32x4 a, const u32x4 b) {
//  return u64x2(_mm_unpackhi_epi32(a, b));
//}
//
// SIMD_INLINE i16x8 zip_hi(const i8x16 a, const i8x16 b) {
//  return i16x8(_mm_unpackhi_epi8(a, b));
//}
// SIMD_INLINE i32x4 zip_hi(const i16x8 a, const i16x8 b) {
//  return i32x4(_mm_unpackhi_epi16(a, b));
//}
// SIMD_INLINE i64x2 zip_hi(const i32x4 a, const i32x4 b) {
//  return i64x2(_mm_unpackhi_epi32(a, b));
//}
//
// ------------------------------ Promotions (part w/ narrow lanes -> full)

// Unsigned: zero-extend.
SIMD_INLINE u16x8 convert_to(Full<uint16_t, ARM8>, const u8x8 v) {
  return u16x8(vmovl_u8(v));
}
SIMD_INLINE u32x4 convert_to(Full<uint32_t, ARM8>, const u8x4 v) {
  uint16x8_t a = vmovl_u8(v);
  return u32x4(vmovl_u16(vget_low_u16(a)));
}
SIMD_INLINE u32x4 convert_to(Full<uint32_t, ARM8>, const u16x4 v) {
  return u32x4(vmovl_u16(v));
}
SIMD_INLINE u64x2 convert_to(Full<uint64_t, ARM8>, const u32x2 v) {
  return u64x2(vmovl_u32(v));
}
SIMD_INLINE i16x8 convert_to(Full<int16_t, ARM8>, const u8x8 v) {
  return i16x8(vreinterpretq_s16_u16(vmovl_u8(v)));
}
SIMD_INLINE i32x4 convert_to(Full<int32_t, ARM8>, const u8x4 v) {
  uint16x8_t a = vmovl_u8(v);
  return i32x4(vreinterpretq_s32_u16(vmovl_u16(vget_low_u16(a))));
}
SIMD_INLINE i32x4 convert_to(Full<int32_t, ARM8>, const u16x4 v) {
  return i32x4(vmovl_u16(v));
}

// Signed: replicate sign bit.
SIMD_INLINE i16x8 convert_to(Full<int16_t, ARM8>, const i8x8 v) {
  return i16x8(vmovl_s8(v));
}
SIMD_INLINE i32x4 convert_to(Full<int32_t, ARM8>, const i8x4 v) {
  int16x8_t a = vmovl_s8(v);
  return i32x4(vmovl_s16(vget_low_s16(a)));
}
SIMD_INLINE i32x4 convert_to(Full<int32_t, ARM8>, const i16x4 v) {
  return i32x4(vmovl_s16(v));
}
SIMD_INLINE i64x2 convert_to(Full<int64_t, ARM8>, const i32x2 v) {
  return i64x2(vmovl_s32(v));
}

// ------------------------------ Demotions (full -> part w/ narrow lanes)

SIMD_INLINE u16x4 convert_to(Desc<uint16_t, 4, ARM8>, const i32x4 v) {
  return u16x4(vqmovun_s32(v));
}
SIMD_INLINE u8x8 convert_to(Desc<uint8_t, 8, ARM8>, const u16x8 v) {
  return u8x8(vqmovn_u16(v));
}

SIMD_INLINE u8x8 convert_to(Desc<uint8_t, 8, ARM8>, const i16x8 v) {
  return u8x8(vqmovun_s16(v));
}

SIMD_INLINE i16x4 convert_to(Desc<int16_t, 4, ARM8>, const i32x4 v) {
  return i16x4(vqmovn_s32(v));
}
SIMD_INLINE i8x8 convert_to(Desc<int8_t, 8, ARM8>, const i16x8 v) {
  return i8x8(vqmovn_s16(v));
}

// In the following convert_to functions, |b| is purposely undefined.
// The value a needs to be extended to 128 bits so that vqmovn can be
// used and |b| is undefined so that no extra overhead is introduced.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wuninitialized"

SIMD_INLINE u8x4 convert_to(Desc<uint8_t, 4, ARM8>, const i32x4 v) {
  u16x4 a = convert_to(Desc<uint16_t, 4, ARM8>(), v);
  u16x4 b;
  uint16x8_t c = vcombine_u16(a, b);
  return u8x4(vqmovn_u16(c));
}

SIMD_INLINE i8x4 convert_to(Desc<int8_t, 4, ARM8>, const i32x4 v) {
  i16x4 a = convert_to(Desc<int16_t, 4, ARM8>(), v);
  i16x4 b;
  uint16x8_t c = vcombine_s16(a, b);
  return i8x4(vqmovn_s16(c));
}

#pragma clang diagnostic pop

// ------------------------------ Select/blend

// Returns mask ? b : a. Due to ARM's semantics, each lane of "mask" must
// equal T(0) or ~T(0) although x86 may only check the most significant bit.
SIMD_INLINE u8x16 select(const u8x16 a,
                         const u8x16 b,
                         const u8x16 mask) {
  return u8x16(vbslq_u8(mask, b, a));
}
SIMD_INLINE i8x16 select(const i8x16 a,
                         const i8x16 b,
                         const i8x16 mask) {
  return i8x16(vbslq_s8(mask, b, a));
}
SIMD_INLINE u16x8 select(const u16x8 a,
                         const u16x8 b,
                         const u16x8 mask) {
  return u16x8(vbslq_u16(mask, b, a));
}
SIMD_INLINE i16x8 select(const i16x8 a,
                         const i16x8 b,
                         const i16x8 mask) {
  return i16x8(vbslq_s16(mask, b, a));
}
SIMD_INLINE u32x4 select(const u32x4 a,
                         const u32x4 b,
                         const u32x4 mask) {
  return u32x4(vbslq_u32(mask, b, a));
}
SIMD_INLINE i32x4 select(const i32x4 a,
                         const i32x4 b,
                         const i32x4 mask) {
  return i32x4(vbslq_s32(mask, b, a));
}
SIMD_INLINE u64x2 select(const u64x2 a,
                         const u64x2 b,
                         const u64x2 mask) {
  return u64x2(vbslq_u64(mask, b, a));
}
SIMD_INLINE i64x2 select(const i64x2 a,
                         const i64x2 b,
                         const i64x2 mask) {
  return i64x2(vbslq_s64(mask, b, a));
}
SIMD_INLINE f32x4 select(const f32x4 a,
                         const f32x4 b,
                         const f32x4 mask) {
  return f32x4(vbslq_f32(mask, b, a));
}
SIMD_INLINE f64x2 select(const f64x2 a,
                         const f64x2 b,
                         const f64x2 mask) {
  return f64x2(vbslq_f64(mask, b, a));
}

// ------------------------------ AES cipher

// One round of AES. "round_key" is a constant for breaking the symmetry of AES
// (ensures previously equal columns differ afterwards).
SIMD_INLINE u8x16 aes_round(const u8x16 state,
                            const u8x16 round_key) {
  return u8x16(vaesmcq_u8(vaeseq_u8(state, round_key)));
}

// ------------------------------ Horizontal sum (reduction)

// "Extensions": useful but quite performance-portable operations. We add
// functions to this namespace in multiple places.
namespace ext {

// Returns 64-bit sums of 8-byte groups.
SIMD_INLINE u64x2 sums_of_u8x8(const u8x16 v) {
  uint16x8_t a = vpaddlq_u8(v);
  uint32x4_t b = vpaddlq_u16(a);
  return u64x2(vpaddlq_u32(b));
}

// Supported for 32b and 64b vector types. Returns the sum in each lane.
SIMD_INLINE u32x4 horz_sum(const u32x4 v) {
  return u32x4(vdupq_n_u32(vaddvq_u32(v)));
}
SIMD_INLINE i32x4 horz_sum(const i32x4 v) {
  return i32x4(vdupq_n_s32(vaddvq_s32(v)));
}
SIMD_INLINE u64x2 horz_sum(const u64x2 v) {
  return u64x2(vdupq_n_u64(vaddvq_u64(v)));
}
SIMD_INLINE i64x2 horz_sum(const i64x2 v) {
  return i64x2(vdupq_n_s64(vaddvq_s64(v)));
}
SIMD_INLINE f32x4 horz_sum(const f32x4 v) {
  return f32x4(vdupq_n_f32(vaddvq_f32(v)));
}
SIMD_INLINE f64x2 horz_sum(const f64x2 v) {
  return f64x2(vdupq_n_f64(vaddvq_f64(v)));
}

}  // namespace ext

// TODO(user): wrappers for all intrinsics (in neon namespace).

#endif  // SIMD_DEPS
