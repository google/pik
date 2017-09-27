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

// Single-element vectors and operations.
// (No include guard nor namespace: this is included from the middle of simd.h.)

// (We can't just use built-in types because comparison and shift operators
// need to be overloaded.)
template <typename Lane>
class vec1 {
  using Raw = Lane;

 public:
  using T = Lane;

  SIMD_INLINE vec1() {}
  vec1(const vec1&) = default;
  vec1& operator=(const vec1&) = default;
  SIMD_INLINE explicit vec1(const Raw& v) : v_(v) {}

  SIMD_INLINE operator Raw() const { return v_; }  // NOLINT

  SIMD_INLINE vec1& operator*=(const vec1 other) {
    return *this = (*this * other);
  }
  SIMD_INLINE vec1& operator/=(const vec1 other) {
    return *this = (*this / other);
  }
  SIMD_INLINE vec1& operator+=(const vec1 other) {
    return *this = (*this + other);
  }
  SIMD_INLINE vec1& operator-=(const vec1 other) {
    return *this = (*this - other);
  }
  SIMD_INLINE vec1& operator&=(const vec1 other) {
    return *this = (*this & other);
  }
  SIMD_INLINE vec1& operator|=(const vec1 other) {
    return *this = (*this | other);
  }
  SIMD_INLINE vec1& operator^=(const vec1 other) {
    return *this = (*this ^ other);
  }
  template <typename ShiftArg>  // int or vec1
  SIMD_INLINE vec1& operator<<=(const ShiftArg count) {
    return *this = (*this << count);
  }
  template <typename ShiftArg>
  SIMD_INLINE vec1& operator>>=(const ShiftArg count) {
    return *this = (*this >> count);
  }

 private:
  Raw v_;
};

template <typename Lane>
struct IsVec<vec1<Lane>> {
  static constexpr bool value = true;
};

using u8x1 = vec1<uint8_t>;
using u16x1 = vec1<uint16_t>;
using u32x1 = vec1<uint32_t>;
using u64x1 = vec1<uint64_t>;

using i8x1 = vec1<int8_t>;
using i16x1 = vec1<int16_t>;
using i32x1 = vec1<int32_t>;
using i64x1 = vec1<int64_t>;

using f32x1 = vec1<float>;
using f64x1 = vec1<double>;

// ------------------------------ Set

template <typename T>
SIMD_INLINE vec1<T> setzero(vec1<T>) {
  return vec1<T>(T(0));
}

template <typename T>
SIMD_INLINE vec1<T> set1(vec1<T>, const T t) {
  return vec1<T>(t);
}

// ------------------------------ Half

template <typename T>
vec1<T> lower_half(const vec1<T> v) {
  return v;
}

template <typename T>
vec1<T> upper_half(const vec1<T> v) {
  return v;
}

template <typename T>
vec1<T> from_half(const vec1<T> v) {
  return v;
}

// ================================================== ARITHMETIC

template <typename T>
SIMD_INLINE vec1<T> operator+(const vec1<T> a, const vec1<T> b) {
  return vec1<T>(T(a) + T(b));
}

template <typename T>
SIMD_INLINE vec1<T> operator-(const vec1<T> a, const vec1<T> b) {
  return vec1<T>(T(a) - T(b));
}

// ------------------------------ Saturating addition

// Returns a + b clamped to the destination range.

// Unsigned
SIMD_INLINE u8x1 add_sat(const u8x1 a, const u8x1 b) {
  return u8x1(SIMD_MIN(SIMD_MAX(0, int(a) + int(b)), 255));
}
SIMD_INLINE u16x1 add_sat(const u16x1 a, const u16x1 b) {
  return u16x1(SIMD_MIN(SIMD_MAX(0, int(a) + int(b)), 65535));
}

// Signed
SIMD_INLINE i8x1 add_sat(const i8x1 a, const i8x1 b) {
  return i8x1(SIMD_MIN(SIMD_MAX(-128, int(a) + int(b)), 127));
}
SIMD_INLINE i16x1 add_sat(const i16x1 a, const i16x1 b) {
  return i16x1(SIMD_MIN(SIMD_MAX(-32768, int(a) + int(b)), 32767));
}

// ------------------------------ Saturating subtraction

// Returns a - b clamped to the destination range.

// Unsigned
SIMD_INLINE u8x1 sub_sat(const u8x1 a, const u8x1 b) {
  return u8x1(SIMD_MIN(SIMD_MAX(0, int(a) - int(b)), 255));
}
SIMD_INLINE u16x1 sub_sat(const u16x1 a, const u16x1 b) {
  return u16x1(SIMD_MIN(SIMD_MAX(0, int(a) - int(b)), 65535));
}

// Signed
SIMD_INLINE i8x1 sub_sat(const i8x1 a, const i8x1 b) {
  return i8x1(SIMD_MIN(SIMD_MAX(-128, int(a) - int(b)), 127));
}
SIMD_INLINE i16x1 sub_sat(const i16x1 a, const i16x1 b) {
  return i16x1(SIMD_MIN(SIMD_MAX(-32768, int(a) - int(b)), 32767));
}

// ------------------------------ Average

// Returns (a + b + 1) / 2

SIMD_INLINE u8x1 avg(const u8x1 a, const u8x1 b) {
  return u8x1((a + b + 1) / 2);
}
SIMD_INLINE u16x1 avg(const u16x1 a, const u16x1 b) {
  return u16x1((a + b + 1) / 2);
}

// ------------------------------ Shift lanes by constant #bits

// Shift counts >= type bits are undefined in C, but we want to match the
// sensible behavior of SSE2 (zeroing).

template <typename T>
SIMD_INLINE vec1<T> operator<<(const vec1<T> v, const int bits) {
  if (bits >= sizeof(T) * 8) {
    return vec1<T>(0);
  }
  return vec1<T>(T(v) << bits);
}

template <typename T>
SIMD_INLINE vec1<T> operator>>(const vec1<T> v, const int bits) {
  if (bits >= sizeof(T) * 8) {
    return vec1<T>(0);
  }
  return vec1<T>(T(v) >> bits);
}

// ------------------------------ Shift lanes by independent variable #bits

template <typename T>
SIMD_INLINE u32x1 operator<<(const u32x1 v, const u32x1 bits) {
  return operator<<(v, static_cast<int>(bits));
}
template <typename T>
SIMD_INLINE u32x1 operator>>(const u32x1 v, const u32x1 bits) {
  return operator>>(v, static_cast<int>(bits));
}

// ------------------------------ min/max

template <typename T>
SIMD_INLINE vec1<T> min(const vec1<T> a, const vec1<T> b) {
  return vec1<T>(SIMD_MIN(a, b));
}

template <typename T>
SIMD_INLINE vec1<T> max(const vec1<T> a, const vec1<T> b) {
  return vec1<T>(SIMD_MAX(a, b));
}

// ------------------------------ mul/div

template <typename T>
SIMD_INLINE vec1<T> operator*(const vec1<T> a, const vec1<T> b) {
  if (IsFloat<T>()) {
    return vec1<T>(static_cast<T>(double(a) * double(b)));
  } else if (IsSigned<T>()) {
    return vec1<T>(static_cast<T>(int64_t(a) * int64_t(b)));
  } else {
    return vec1<T>(static_cast<T>(uint64_t(a) * uint64_t(b)));
  }
}

template <typename T>
SIMD_INLINE vec1<T> operator/(const vec1<T> a, const vec1<T> b) {
  return vec1<T>(T(a) / T(b));
}

// "Extensions": useful but quite performance-portable operations. We add
// functions to this namespace in multiple places.
namespace ext {

// Returns the upper 16 bits of a * b in each lane.
SIMD_INLINE i16x1 mulhi(const i16x1 a, const i16x1 b) {
  return i16x1((int32_t(a) * int32_t(b)) >> 16);
}

}  // namespace ext

// Multiplies even lanes (0, 2 ..) and returns the double-wide result.
SIMD_INLINE i64x1 mul_even(const i32x1 a, const i32x1 b) {
  const int64_t a64 = a;
  return i64x1(a64 * int32_t(b));
}
SIMD_INLINE u64x1 mul_even(const u32x1 a, const u32x1 b) {
  const uint64_t a64 = a;
  return u64x1(a64 * uint32_t(b));
}

// Approximate reciprocal
SIMD_INLINE f32x1 rcp_approx(const f32x1 v) { return f32x1(1.0f / float(v)); }

// ------------------------------ Floating-point multiply-add variants

template <typename T>
SIMD_INLINE vec1<T> mul_add(const vec1<T> mul, const vec1<T> x,
                            const vec1<T> add) {
  return mul * x + add;
}

template <typename T>
SIMD_INLINE vec1<T> mul_sub(const vec1<T> mul, const vec1<T> x,
                            const vec1<T> sub) {
  return mul * x - sub;
}

template <typename T>
SIMD_INLINE vec1<T> nmul_add(const vec1<T> mul, const vec1<T> x,
                             const vec1<T> add) {
  return add - mul * x;
}

// nmul_sub would require an additional negate of mul or x.

// ------------------------------ Floating-point square root

// Approximate reciprocal square root
SIMD_INLINE f32x1 rsqrt_approx(const f32x1 v) {
  float f = v;
  const float half = f * 0.5f;
  uint32_t bits;
  CopyBytes(f, &bits);
  // Initial guess based on log2(f)
  bits = 0x5F3759DF - (bits >> 1);
  CopyBytes(bits, &f);
  // One Newton-Raphson iteration
  return f32x1(f * (1.5f - (half * f * f)));
}

// Square root
SIMD_INLINE f32x1 sqrt(const f32x1 v) { return rsqrt_approx(v) * v; }
SIMD_INLINE f64x1 sqrt(const f64x1 v) {
  const double f = v;
  const float s = sqrt(f32x1(f));
  return f64x1(s);
}

// ------------------------------ Floating-point rounding

// Toward nearest integer
SIMD_INLINE f32x1 round_nearest(const f32x1 v) {
  return f32x1(static_cast<int>(float(v) + 0.5f));
}
SIMD_INLINE f64x1 round_nearest(const f64x1 v) {
  return f64x1(static_cast<int>(double(v) + 0.5));
}

template <typename Float, typename Bits, int kMantissaBits, int kExponentBits,
          class V>
V Ceiling(const V v) {
  const Bits kExponentMask = (1ull << kExponentBits) - 1;
  const Bits kMantissaMask = (1ull << kMantissaBits) - 1;
  const Bits kBias = kExponentMask / 2;

  Float f = v;
  const bool positive = f > 0.0f;

  Bits bits;
  CopyBytes(v, &bits);

  const int exponent = ((bits >> kMantissaBits) & kExponentMask) - kBias;
  // Already an integer.
  if (exponent >= kMantissaBits) return v;
  // |v| <= 1 => 0 or 1.
  if (exponent < 0) return V(positive);

  const Bits mantissa_mask = kMantissaMask >> exponent;
  // Already an integer
  if ((bits & mantissa_mask) == 0) return v;

  // Clear fractional bits and round up
  if (positive) bits += (kMantissaMask + 1) >> exponent;
  bits &= ~mantissa_mask;

  CopyBytes(bits, &f);
  return V(f);
}

template <typename Float, typename Bits, int kMantissaBits, int kExponentBits,
          class V>
V Floor(const V v) {
  const Bits kExponentMask = (1ull << kExponentBits) - 1;
  const Bits kMantissaMask = (1ull << kMantissaBits) - 1;
  const Bits kBias = kExponentMask / 2;

  Float f = v;
  const bool negative = f < 0.0f;

  Bits bits;
  CopyBytes(v, &bits);

  const int exponent = ((bits >> kMantissaBits) & kExponentMask) - kBias;
  // Already an integer.
  if (exponent >= kMantissaBits) return v;
  // |v| <= 1 => -1 or 0.
  if (exponent < 0) return V(negative ? -1.0 : 0.0f);

  const Bits mantissa_mask = kMantissaMask >> exponent;
  // Already an integer
  if ((bits & mantissa_mask) == 0) return v;

  // Clear fractional bits and round down
  if (negative) bits += (kMantissaMask + 1) >> exponent;
  bits &= ~mantissa_mask;

  CopyBytes(bits, &f);
  return V(f);
}

// Toward +infinity, aka ceiling
SIMD_INLINE f32x1 round_pos_inf(const f32x1 v) {
  return Ceiling<float, uint32_t, 23, 8>(v);
}
SIMD_INLINE f64x1 round_pos_inf(const f64x1 v) {
  return Ceiling<double, uint64_t, 52, 11>(v);
}

// Toward -infinity, aka floor
SIMD_INLINE f32x1 round_neg_inf(const f32x1 v) {
  return Floor<float, uint32_t, 23, 8>(v);
}
SIMD_INLINE f64x1 round_neg_inf(const f64x1 v) {
  return Floor<double, uint64_t, 52, 11>(v);
}

// ------------------------------ Convert i32 <=> f32

SIMD_INLINE f32x1 f32_from_i32(const i32x1 v) {
  return f32x1(static_cast<float>(v));
}
SIMD_INLINE i32x1 i32_from_f32(const f32x1 v) {
  return i32x1(static_cast<int>(v));
}

// ------------------------------ Cast to/from floating-point representation

SIMD_INLINE f32x1 f32_from_bits(const i32x1 v) {
  float f;
  CopyBytes(v, &f);
  return f32x1(f);
}
SIMD_INLINE i32x1 bits_from_f32(const f32x1 v) {
  int32_t i;
  CopyBytes(v, &i);
  return i32x1(i);
}

SIMD_INLINE f64x1 f64_from_bits(const i64x1 v) {
  double f;
  CopyBytes(v, &f);
  return f64x1(f);
}
SIMD_INLINE i64x1 bits_from_f64(const f64x1 v) {
  int64_t i;
  CopyBytes(v, &i);
  return i64x1(i);
}

// "Extensions": useful but quite performance-portable operations. We add
// functions to this namespace in multiple places.
namespace ext {

// Sum of all lanes, i.e. the only one.
template <typename T>
SIMD_INLINE vec1<T> horz_sum(const vec1<T> v0) {
  return v0;
}
SIMD_INLINE vec1<uint64_t> horz_sum(const u8x1 v0) {
  return vec1<uint64_t>(int(v0));
}

}  // namespace ext

// ================================================== COMPARE

// Comparisons fill a lane with 1-bits if the condition is true, else 0.
template <class V>
V ComparisonResult(const bool result) {
  typename V::T ret;
  SetBytes(result ? 0xFF : 0, &ret);
  return V(ret);
}

template <typename T>
SIMD_INLINE vec1<T> operator==(const vec1<T> a, const vec1<T> b) {
  return ComparisonResult<vec1<T>>(T(a) == T(b));
}

template <typename T>
SIMD_INLINE vec1<T> operator<(const vec1<T> a, const vec1<T> b) {
  return ComparisonResult<vec1<T>>(T(a) < T(b));
}
template <typename T>
SIMD_INLINE vec1<T> operator>(const vec1<T> a, const vec1<T> b) {
  return ComparisonResult<vec1<T>>(T(a) > T(b));
}

template <typename T>
SIMD_INLINE vec1<T> operator<=(const vec1<T> a, const vec1<T> b) {
  return ComparisonResult<vec1<T>>(T(a) <= T(b));
}
template <typename T>
SIMD_INLINE vec1<T> operator>=(const vec1<T> a, const vec1<T> b) {
  return ComparisonResult<vec1<T>>(T(a) >= T(b));
}

// "Extensions": useful but quite performance-portable operations. We add
// functions to this namespace in multiple places.
namespace ext {

// Returns a bit array of the most significant bit of each byte in "v", i.e.
// sum_i=0..15 of (v[i] >> 7) << i; v[0] is the least-significant byte of "v".
// This is useful for testing/branching based on comparison results.
SIMD_INLINE uint32_t movemask(const u8x1 v) { return v >> 7; }

// Returns the most significant bit of each float/double lane (see above).
SIMD_INLINE uint32_t movemask(const f32x1 v) { return v < 0.0f; }
SIMD_INLINE uint32_t movemask(const f64x1 v) { return v < 0.0; }

// Returns whether all lanes are equal to zero. Supported for all integer T.
template <typename T>
SIMD_INLINE bool all_zero(const vec1<T> v) {
  return static_cast<T>(v) == 0;
}

}  // namespace ext

// ================================================== LOGICAL

template <typename Bits>
struct BitwiseOp {
  template <typename T, class Op>
  vec1<T> operator()(const vec1<T> a, const vec1<T> b, const Op& op) const {
    static_assert(sizeof(T) == sizeof(Bits), "Float/int size mismatch");
    Bits ia, ib;
    CopyBytes(a, &ia);
    CopyBytes(b, &ib);
    ia = op(ia, ib);
    T ret;
    CopyBytes(ia, &ret);
    return vec1<T>(ret);
  }
};

// ------------------------------ Bitwise AND

template <typename T>
SIMD_INLINE vec1<T> operator&(const vec1<T> a, const vec1<T> b) {
  return vec1<T>(T(a) & T(b));
}
template <>
SIMD_INLINE f32x1 operator&(const f32x1 a, const f32x1 b) {
  return BitwiseOp<int32_t>()(a, b, [](int32_t i, int32_t j) { return i & j; });
}
template <>
SIMD_INLINE f64x1 operator&(const f64x1 a, const f64x1 b) {
  return BitwiseOp<int64_t>()(a, b, [](int64_t i, int64_t j) { return i & j; });
}

// ------------------------------ Bitwise AND-NOT

// Returns ~a & b.
template <typename T>
SIMD_INLINE vec1<T> andnot(const vec1<T> a, const vec1<T> b) {
  return vec1<T>(~T(a) & T(b));
}
template <>
SIMD_INLINE f32x1 andnot(const f32x1 a, const f32x1 b) {
  return BitwiseOp<int32_t>()(a, b,
                              [](int32_t i, int32_t j) { return ~i & j; });
}
template <>
SIMD_INLINE f64x1 andnot(const f64x1 a, const f64x1 b) {
  return BitwiseOp<int64_t>()(a, b,
                              [](int64_t i, int64_t j) { return ~i & j; });
}

// ------------------------------ Bitwise OR

template <typename T>
SIMD_INLINE vec1<T> operator|(const vec1<T> a, const vec1<T> b) {
  return vec1<T>(T(a) | T(b));
}
template <>
SIMD_INLINE f32x1 operator|(const f32x1 a, const f32x1 b) {
  return BitwiseOp<int32_t>()(a, b, [](int32_t i, int32_t j) { return i | j; });
}
template <>
SIMD_INLINE f64x1 operator|(const f64x1 a, const f64x1 b) {
  return BitwiseOp<int64_t>()(a, b, [](int64_t i, int64_t j) { return i | j; });
}

// ------------------------------ Bitwise XOR

template <typename T>
SIMD_INLINE vec1<T> operator^(const vec1<T> a, const vec1<T> b) {
  return vec1<T>(T(a) ^ T(b));
}
template <>
SIMD_INLINE f32x1 operator^(const f32x1 a, const f32x1 b) {
  return BitwiseOp<int32_t>()(a, b, [](int32_t i, int32_t j) { return i ^ j; });
}
template <>
SIMD_INLINE f64x1 operator^(const f64x1 a, const f64x1 b) {
  return BitwiseOp<int64_t>()(a, b, [](int64_t i, int64_t j) { return i ^ j; });
}

// ================================================== LOAD/STORE

// ------------------------------ Load all lanes

template <typename T>
SIMD_INLINE vec1<T> load(vec1<T>, const T* SIMD_RESTRICT aligned) {
  T t;
  CopyBytes(*aligned, &t);
  return vec1<T>(t);
}

template <typename T>
SIMD_INLINE vec1<T> load_unaligned(const vec1<T> v, const T* SIMD_RESTRICT p) {
  return load(v, p);
}

// no load_dup128: that requires at least 128-bit vectors.

// ------------------------------ Store all lanes

template <typename T>
SIMD_INLINE void store(const vec1<T> v, T* SIMD_RESTRICT aligned) {
  const T t = v;
  CopyBytes(t, aligned);
}

template <typename T>
SIMD_INLINE void store_unaligned(const vec1<T> v, T* SIMD_RESTRICT p) {
  return store(v, p);
}

// ------------------------------ "Non-temporal" stores

template <typename T>
SIMD_INLINE void stream(const vec1<T> v, T* SIMD_RESTRICT aligned) {
  return store(v, aligned);
}

#if !SIMD_ENABLE_SSE4

SIMD_INLINE void stream32(const uint32_t t, uint32_t* SIMD_RESTRICT aligned) {
  CopyBytes(t, aligned);
}

SIMD_INLINE void stream64(const uint64_t t, uint64_t* SIMD_RESTRICT aligned) {
  CopyBytes(t, aligned);
}

SIMD_INLINE void store_fence() {}

// ------------------------------ Cache control

template <typename T>
SIMD_INLINE void prefetch(const T*) {}

SIMD_INLINE void flush_cacheline(const void*) {}

#endif

// ================================================== SWIZZLE

// ------------------------------ Shift vector by constant #bytes

// 0x10..01, kBytes = 1 => 0x0F..0100
template <int kBytes, typename T>
SIMD_INLINE vec1<T> shift_bytes_left(const vec1<T> v) {
  static_assert(0 <= kBytes && kBytes <= sizeof(v), "Invalid kBytes");
  const vec1<T> lh[2] = {set1(vec1<T>(), T(0)), v};
  T ret;
  CopyBytesWithOffset(lh[0], sizeof(v) - kBytes, &ret);
  return vec1<T>(ret);
}

// 0x10..01, kBytes = 1 => 0x0010..02
template <int kBytes, typename T>
SIMD_INLINE vec1<T> shift_bytes_right(const vec1<T> v) {
  static_assert(0 <= kBytes && kBytes <= sizeof(v), "Invalid kBytes");
  const vec1<T> lh[2] = {v, set1(vec1<T>(), T(0))};
  T ret;
  CopyBytesWithOffset(lh[0], kBytes, &ret);
  return vec1<T>(ret);
}

// ------------------------------ Extract from 2x 128-bit at constant offset

// Extracts a vector from <hi, lo> by skipping the least-significant kBytes.
template <int kBytes, typename T>
SIMD_INLINE vec1<T> extract_concat_bytes(const vec1<T> hi, const vec1<T> lo) {
  static_assert(0 <= kBytes && kBytes <= sizeof(lo), "Invalid kBytes");
  const vec1<T> lh[2] = {lo, hi};
  T ret;
  CopyBytesWithOffset(lh[0], kBytes, &ret);
  return vec1<T>(ret);
}

// ------------------------------ Get/set least-significant lane

template <typename T>
SIMD_INLINE T get_low(const vec1<T> v) {
  return static_cast<T>(v);
}

template <typename T>
SIMD_INLINE vec1<T> set_low(vec1<T>, const T t) {
  return vec1<T>(t);
}

// ------------------------------ Broadcast/splat any lane

template <int kLane, typename T>
SIMD_INLINE vec1<T> broadcast(const vec1<T> v) {
  static_assert(kLane == 0, "Scalar only has one lane");
  return v;
}

// ------------------------------ Shuffle bytes with variable indices

// Returns vector of bytes[from[i]]. "from" must be valid indices in [0, 16).
template <typename T>
SIMD_INLINE vec1<T> shuffle_bytes(const vec1<T> v, const vec1<T> from) {
  T v_lanes;
  store(v, &v_lanes);
  T from_lanes;
  store(from, &from_lanes);
  const uint8_t* v_bytes = reinterpret_cast<const uint8_t*>(v_lanes);
  const uint8_t* from_bytes = reinterpret_cast<const uint8_t*>(from_lanes);
  T ret = 0;
  uint8_t* out_bytes = reinterpret_cast<uint8_t*>(&ret);
  for (size_t i = 0; i < sizeof(vec1<T>); ++i) {
    const int index = from_bytes[i];
    out_bytes[i] = index < sizeof(vec1<T>) ? v_bytes[index] : 0;
  }
  return vec1<T>(ret);
}

// ------------------------------ Zip/interleave/unpack

// Not supported - integer unpack could return double-wide lanes, but for
// floats/doubles the two resulting lanes do not fit in vec1.

// ------------------------------ Cast to double-width lane type

// Unsigned: zero-extend.
SIMD_INLINE u16x1 promote(const u8x1 v) { return u16x1(uint16_t(v)); }
SIMD_INLINE u32x1 promote(const u16x1 v) { return u32x1(uint32_t(v)); }
SIMD_INLINE u64x1 promote(const u32x1 v) { return u64x1(uint64_t(v)); }

// Signed: replicate sign bit.
SIMD_INLINE i16x1 promote(const i8x1 v) { return i16x1(int8_t(v)); }
SIMD_INLINE i32x1 promote(const i16x1 v) { return i32x1(int32_t(v)); }
SIMD_INLINE i64x1 promote(const i32x1 v) { return i64x1(int64_t(v)); }

// ------------------------------ Cast to half-width lane types

// Converts to half-width type after saturating.

SIMD_INLINE u8x1 demote_to_unsigned(const i16x1 v) {
  return u8x1(SIMD_MIN(SIMD_MAX(0, int32_t(v)), 255));
}
SIMD_INLINE u16x1 demote_to_unsigned(const i32x1 v) {
  return u16x1(SIMD_MIN(SIMD_MAX(0, int32_t(v)), 65535));
}

SIMD_INLINE i8x1 demote(const i16x1 v) {
  return i8x1(SIMD_MIN(SIMD_MAX(-128, int32_t(v)), 127));
}
SIMD_INLINE i16x1 demote(const i32x1 v) {
  return i16x1(SIMD_MIN(SIMD_MAX(-32768, int32_t(v)), 32767));
}

// ------------------------------ Select/blend

// Returns mask ? b : a. Each lane of "mask" must equal T(0) or ~T(0).
template <typename T>
SIMD_INLINE vec1<T> select(const vec1<T> a, const vec1<T> b,
                           const vec1<T> mask) {
  return (mask & b) | andnot(mask, a);
}
