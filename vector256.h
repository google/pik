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

#ifndef VECTOR256_H_
#define VECTOR256_H_

// Defines SIMD vector classes ("V4x64U") with overloaded arithmetic operators:
// const V4x64U masked_sum = (a + b) & m;
// This is shorter and more readable than compiler intrinsics:
// const __m256i masked_sum = _mm256_and_si256(_mm256_add_epi64(a, b), m);
// There is typically no runtime cost for these abstractions.
//
// The naming convention is VNxBBT where N is the number of lanes, BB the
// number of bits per lane and T is the lane type: unsigned integer (U),
// signed integer (I), or floating-point (F).

// WARNING: this is a "restricted" header because it is included from
// translation units compiled with different flags. This header and its
// dependencies must not define any function unless it is static inline and/or
// within namespace PIK_TARGET_NAME. See arch_specific.h for details.

#include <stddef.h>
#include <stdint.h>

#include "arch_specific.h"
#include "compiler_specific.h"

// For auto-dependency generation, we need to include all headers but not their
// contents (otherwise compilation fails because -mavx2 is not specified).
#ifndef PIK_DISABLE_TARGET_SPECIFIC

// (This include cannot be moved within a namespace due to conflicts with
// other system headers - see vector128.h)
#include <immintrin.h>

namespace pik {
// To prevent ODR violations when including this from multiple translation
// units (TU) that are compiled with different flags, the contents must reside
// in a namespace whose name is unique to the TU. NOTE: this behavior is
// incompatible with precompiled modules and requires textual inclusion instead.
namespace PIK_TARGET_NAME {

// Primary template for 256-bit AVX2 vectors; only specializations are used.
template <typename T>
class V256 {};

template <>
class V256<uint8_t> {
 public:
  using T = uint8_t;
  static constexpr size_t N = 32;

  // Leaves v_ uninitialized - typically used for output parameters.
  PIK_INLINE V256() {}

  // Broadcasts i to all lanes.
  PIK_INLINE explicit V256(T i)
      : v_(_mm256_broadcastb_epi8(_mm_cvtsi32_si128(i))) {}

  // Copy from other vector.
  PIK_INLINE explicit V256(const V256& other) : v_(other.v_) {}
  template <typename U>
  PIK_INLINE explicit V256(const V256<U>& other) : v_(other) {}
  PIK_INLINE V256& operator=(const V256& other) {
    v_ = other.v_;
    return *this;
  }

  // Convert from/to intrinsics.
  PIK_INLINE V256(const __m256i& v) : v_(v) {}
  PIK_INLINE V256& operator=(const __m256i& v) {
    v_ = v;
    return *this;
  }
  PIK_INLINE operator __m256i() const { return v_; }

  // There are no greater-than comparison instructions for unsigned T.
  PIK_INLINE V256 operator==(const V256& other) const {
    return V256(_mm256_cmpeq_epi8(v_, other.v_));
  }

  PIK_INLINE V256& operator+=(const V256& other) {
    v_ = _mm256_add_epi8(v_, other.v_);
    return *this;
  }
  PIK_INLINE V256& operator-=(const V256& other) {
    v_ = _mm256_sub_epi8(v_, other.v_);
    return *this;
  }

  PIK_INLINE V256& operator&=(const V256& other) {
    v_ = _mm256_and_si256(v_, other.v_);
    return *this;
  }
  PIK_INLINE V256& operator|=(const V256& other) {
    v_ = _mm256_or_si256(v_, other.v_);
    return *this;
  }
  PIK_INLINE V256& operator^=(const V256& other) {
    v_ = _mm256_xor_si256(v_, other.v_);
    return *this;
  }

 private:
  __m256i v_;
};

template <>
class V256<uint16_t> {
 public:
  using T = uint16_t;
  static constexpr size_t N = 16;

  // Leaves v_ uninitialized - typically used for output parameters.
  PIK_INLINE V256() {}

  // Lane 0 (p_0) is the lowest.
  PIK_INLINE V256(T p_F, T p_E, T p_D, T p_C, T p_B, T p_A, T p_9, T p_8, T p_7,
                  T p_6, T p_5, T p_4, T p_3, T p_2, T p_1, T p_0)
      : v_(_mm256_set_epi16(p_F, p_E, p_D, p_C, p_B, p_A, p_9, p_8, p_7, p_6,
                            p_5, p_4, p_3, p_2, p_1, p_0)) {}

  // Broadcasts i to all lanes.
  PIK_INLINE explicit V256(T i)
      : v_(_mm256_broadcastw_epi16(_mm_cvtsi32_si128(i))) {}

  // Copy from other vector.
  PIK_INLINE explicit V256(const V256& other) : v_(other.v_) {}
  template <typename U>
  PIK_INLINE explicit V256(const V256<U>& other) : v_(other) {}
  PIK_INLINE V256& operator=(const V256& other) {
    v_ = other.v_;
    return *this;
  }

  // Convert from/to intrinsics.
  PIK_INLINE V256(const __m256i& v) : v_(v) {}
  PIK_INLINE V256& operator=(const __m256i& v) {
    v_ = v;
    return *this;
  }
  PIK_INLINE operator __m256i() const { return v_; }

  // There are no greater-than comparison instructions for unsigned T.
  PIK_INLINE V256 operator==(const V256& other) const {
    return V256(_mm256_cmpeq_epi16(v_, other.v_));
  }

  PIK_INLINE V256& operator+=(const V256& other) {
    v_ = _mm256_add_epi16(v_, other.v_);
    return *this;
  }
  PIK_INLINE V256& operator-=(const V256& other) {
    v_ = _mm256_sub_epi16(v_, other.v_);
    return *this;
  }

  PIK_INLINE V256& operator&=(const V256& other) {
    v_ = _mm256_and_si256(v_, other.v_);
    return *this;
  }
  PIK_INLINE V256& operator|=(const V256& other) {
    v_ = _mm256_or_si256(v_, other.v_);
    return *this;
  }
  PIK_INLINE V256& operator^=(const V256& other) {
    v_ = _mm256_xor_si256(v_, other.v_);
    return *this;
  }

  PIK_INLINE V256& operator<<=(const int count) {
    v_ = _mm256_slli_epi16(v_, count);
    return *this;
  }

  PIK_INLINE V256& operator>>=(const int count) {
    v_ = _mm256_srli_epi16(v_, count);
    return *this;
  }

 private:
  __m256i v_;
};

template <>
class V256<int16_t> {
 public:
  using T = int16_t;
  static constexpr size_t N = 16;

  // Leaves v_ uninitialized - typically used for output parameters.
  PIK_INLINE V256() {}

  // Lane 0 (p_0) is the lowest.
  PIK_INLINE V256(T p_F, T p_E, T p_D, T p_C, T p_B, T p_A, T p_9, T p_8, T p_7,
                  T p_6, T p_5, T p_4, T p_3, T p_2, T p_1, T p_0)
      : v_(_mm256_set_epi16(p_F, p_E, p_D, p_C, p_B, p_A, p_9, p_8, p_7, p_6,
                            p_5, p_4, p_3, p_2, p_1, p_0)) {}

  // Broadcasts i to all lanes.
  PIK_INLINE explicit V256(T i)
      : v_(_mm256_broadcastw_epi16(_mm_cvtsi32_si128(i))) {}

  // Copy from other vector.
  PIK_INLINE explicit V256(const V256& other) : v_(other.v_) {}
  template <typename U>
  PIK_INLINE explicit V256(const V256<U>& other) : v_(other) {}
  PIK_INLINE V256& operator=(const V256& other) {
    v_ = other.v_;
    return *this;
  }

  // Convert from/to intrinsics.
  PIK_INLINE V256(const __m256i& v) : v_(v) {}
  PIK_INLINE V256& operator=(const __m256i& v) {
    v_ = v;
    return *this;
  }
  PIK_INLINE operator __m256i() const { return v_; }

  PIK_INLINE V256 operator==(const V256& other) const {
    return V256(_mm256_cmpeq_epi16(v_, other.v_));
  }
  PIK_INLINE V256 operator<(const V256& other) const {
    return V256(_mm256_cmpgt_epi16(other.v_, v_));
  }
  PIK_INLINE V256 operator>(const V256& other) const {
    return V256(_mm256_cmpgt_epi16(v_, other.v_));
  }

  PIK_INLINE V256& operator+=(const V256& other) {
    v_ = _mm256_add_epi16(v_, other.v_);
    return *this;
  }
  PIK_INLINE V256& operator-=(const V256& other) {
    v_ = _mm256_sub_epi16(v_, other.v_);
    return *this;
  }

  PIK_INLINE V256& operator&=(const V256& other) {
    v_ = _mm256_and_si256(v_, other.v_);
    return *this;
  }
  PIK_INLINE V256& operator|=(const V256& other) {
    v_ = _mm256_or_si256(v_, other.v_);
    return *this;
  }
  PIK_INLINE V256& operator^=(const V256& other) {
    v_ = _mm256_xor_si256(v_, other.v_);
    return *this;
  }

  PIK_INLINE V256& operator<<=(const int count) {
    v_ = _mm256_slli_epi16(v_, count);
    return *this;
  }

  PIK_INLINE V256& operator>>=(const int count) {
    v_ = _mm256_srai_epi16(v_, count);
    return *this;
  }

 private:
  __m256i v_;
};

template <>
class V256<uint32_t> {
 public:
  using T = uint32_t;
  static constexpr size_t N = 8;

  // Leaves v_ uninitialized - typically used for output parameters.
  PIK_INLINE V256() {}

  // Lane 0 (p_0) is the lowest.
  PIK_INLINE V256(T p_7, T p_6, T p_5, T p_4, T p_3, T p_2, T p_1, T p_0)
      : v_(_mm256_set_epi32(p_7, p_6, p_5, p_4, p_3, p_2, p_1, p_0)) {}

  // Broadcasts i to all lanes.
  PIK_INLINE explicit V256(T i)
      : v_(_mm256_broadcastd_epi32(_mm_cvtsi32_si128(i))) {}

  // Copy from other vector.
  PIK_INLINE explicit V256(const V256& other) : v_(other.v_) {}
  template <typename U>
  PIK_INLINE explicit V256(const V256<U>& other) : v_(other) {}
  PIK_INLINE V256& operator=(const V256& other) {
    v_ = other.v_;
    return *this;
  }

  // Convert from/to intrinsics.
  PIK_INLINE V256(const __m256i& v) : v_(v) {}
  PIK_INLINE V256& operator=(const __m256i& v) {
    v_ = v;
    return *this;
  }
  PIK_INLINE operator __m256i() const { return v_; }

  // There are no greater-than comparison instructions for unsigned T.
  PIK_INLINE V256 operator==(const V256& other) const {
    return V256(_mm256_cmpeq_epi32(v_, other.v_));
  }

  PIK_INLINE V256& operator+=(const V256& other) {
    v_ = _mm256_add_epi32(v_, other.v_);
    return *this;
  }
  PIK_INLINE V256& operator-=(const V256& other) {
    v_ = _mm256_sub_epi32(v_, other.v_);
    return *this;
  }

  PIK_INLINE V256& operator&=(const V256& other) {
    v_ = _mm256_and_si256(v_, other.v_);
    return *this;
  }
  PIK_INLINE V256& operator|=(const V256& other) {
    v_ = _mm256_or_si256(v_, other.v_);
    return *this;
  }
  PIK_INLINE V256& operator^=(const V256& other) {
    v_ = _mm256_xor_si256(v_, other.v_);
    return *this;
  }

  PIK_INLINE V256& operator<<=(const int count) {
    v_ = _mm256_slli_epi32(v_, count);
    return *this;
  }

  PIK_INLINE V256& operator>>=(const int count) {
    v_ = _mm256_srli_epi32(v_, count);
    return *this;
  }

 private:
  __m256i v_;
};

template <>
class V256<int32_t> {
 public:
  using T = int32_t;
  static constexpr size_t N = 8;

  // Leaves v_ uninitialized - typically used for output parameters.
  PIK_INLINE V256() {}

  // Lane 0 (p_0) is the lowest.
  PIK_INLINE V256(T p_7, T p_6, T p_5, T p_4, T p_3, T p_2, T p_1, T p_0)
      : v_(_mm256_set_epi32(p_7, p_6, p_5, p_4, p_3, p_2, p_1, p_0)) {}

  // Broadcasts i to all lanes.
  PIK_INLINE explicit V256(T i)
      : v_(_mm256_broadcastd_epi32(_mm_cvtsi32_si128(i))) {}

  // Copy from other vector.
  PIK_INLINE explicit V256(const V256& other) : v_(other.v_) {}
  template <typename U>
  PIK_INLINE explicit V256(const V256<U>& other) : v_(other) {}
  PIK_INLINE V256& operator=(const V256& other) {
    v_ = other.v_;
    return *this;
  }

  // Convert from/to intrinsics.
  PIK_INLINE V256(const __m256i& v) : v_(v) {}
  PIK_INLINE V256& operator=(const __m256i& v) {
    v_ = v;
    return *this;
  }
  PIK_INLINE operator __m256i() const { return v_; }

  PIK_INLINE V256 operator==(const V256& other) const {
    return V256(_mm256_cmpeq_epi32(v_, other.v_));
  }
  PIK_INLINE V256 operator<(const V256& other) const {
    return V256(_mm256_cmpgt_epi32(other.v_, v_));
  }
  PIK_INLINE V256 operator>(const V256& other) const {
    return V256(_mm256_cmpgt_epi32(v_, other.v_));
  }

  PIK_INLINE V256& operator+=(const V256& other) {
    v_ = _mm256_add_epi32(v_, other.v_);
    return *this;
  }
  PIK_INLINE V256& operator-=(const V256& other) {
    v_ = _mm256_sub_epi32(v_, other.v_);
    return *this;
  }

  PIK_INLINE V256& operator&=(const V256& other) {
    v_ = _mm256_and_si256(v_, other.v_);
    return *this;
  }
  PIK_INLINE V256& operator|=(const V256& other) {
    v_ = _mm256_or_si256(v_, other.v_);
    return *this;
  }
  PIK_INLINE V256& operator^=(const V256& other) {
    v_ = _mm256_xor_si256(v_, other.v_);
    return *this;
  }

  PIK_INLINE V256& operator<<=(const int count) {
    v_ = _mm256_slli_epi32(v_, count);
    return *this;
  }

  PIK_INLINE V256& operator>>=(const int count) {
    v_ = _mm256_srai_epi32(v_, count);
    return *this;
  }

 private:
  __m256i v_;
};

template <>
class V256<uint64_t> {
 public:
  using T = uint64_t;
  static constexpr size_t N = 4;

  // Leaves v_ uninitialized - typically used for output parameters.
  PIK_INLINE V256() {}

  // Lane 0 (p_0) is the lowest.
  PIK_INLINE V256(T p_3, T p_2, T p_1, T p_0)
      : v_(_mm256_set_epi64x(p_3, p_2, p_1, p_0)) {}

  // Broadcasts i to all lanes.
  PIK_INLINE explicit V256(T i)
      : v_(_mm256_broadcastq_epi64(_mm_cvtsi64_si128(i))) {}

  // Copy from other vector.
  PIK_INLINE explicit V256(const V256& other) : v_(other.v_) {}
  template <typename U>
  PIK_INLINE explicit V256(const V256<U>& other) : v_(other) {}
  PIK_INLINE V256& operator=(const V256& other) {
    v_ = other.v_;
    return *this;
  }

  // Convert from/to intrinsics.
  PIK_INLINE V256(const __m256i& v) : v_(v) {}
  PIK_INLINE V256& operator=(const __m256i& v) {
    v_ = v;
    return *this;
  }
  PIK_INLINE operator __m256i() const { return v_; }

  // There are no greater-than comparison instructions for unsigned T.
  PIK_INLINE V256 operator==(const V256& other) const {
    return V256(_mm256_cmpeq_epi64(v_, other.v_));
  }

  PIK_INLINE V256& operator+=(const V256& other) {
    v_ = _mm256_add_epi64(v_, other.v_);
    return *this;
  }
  PIK_INLINE V256& operator-=(const V256& other) {
    v_ = _mm256_sub_epi64(v_, other.v_);
    return *this;
  }

  PIK_INLINE V256& operator&=(const V256& other) {
    v_ = _mm256_and_si256(v_, other.v_);
    return *this;
  }
  PIK_INLINE V256& operator|=(const V256& other) {
    v_ = _mm256_or_si256(v_, other.v_);
    return *this;
  }
  PIK_INLINE V256& operator^=(const V256& other) {
    v_ = _mm256_xor_si256(v_, other.v_);
    return *this;
  }

  PIK_INLINE V256& operator<<=(const int count) {
    v_ = _mm256_slli_epi64(v_, count);
    return *this;
  }

  PIK_INLINE V256& operator>>=(const int count) {
    v_ = _mm256_srli_epi64(v_, count);
    return *this;
  }

 private:
  __m256i v_;
};

template <>
class V256<float> {
 public:
  using T = float;
  static constexpr size_t N = 8;

  // Leaves v_ uninitialized - typically used for output parameters.
  PIK_INLINE V256() {}

  // Lane 0 (p_0) is the lowest.
  PIK_INLINE V256(T p_7, T p_6, T p_5, T p_4, T p_3, T p_2, T p_1, T p_0)
      : v_(_mm256_set_ps(p_7, p_6, p_5, p_4, p_3, p_2, p_1, p_0)) {}

  // Broadcasts to all lanes.
  PIK_INLINE explicit V256(T f) : v_(_mm256_set1_ps(f)) {}

  // Copy from other vector.
  PIK_INLINE explicit V256(const V256& other) : v_(other.v_) {}
  template <typename U>
  PIK_INLINE explicit V256(const V256<U>& other) : v_(other) {}
  PIK_INLINE V256& operator=(const V256& other) {
    v_ = other.v_;
    return *this;
  }

  // Convert from/to intrinsics.
  PIK_INLINE V256(const __m256& v) : v_(v) {}
  PIK_INLINE V256& operator=(const __m256& v) {
    v_ = v;
    return *this;
  }
  PIK_INLINE operator __m256() const { return v_; }

  PIK_INLINE V256 operator==(const V256& other) const {
    return V256(_mm256_cmp_ps(v_, other.v_, 0));
  }
  PIK_INLINE V256 operator<(const V256& other) const {
    return V256(_mm256_cmp_ps(v_, other.v_, 1));
  }
  PIK_INLINE V256 operator>(const V256& other) const {
    return V256(_mm256_cmp_ps(other.v_, v_, 1));
  }

  PIK_INLINE V256& operator*=(const V256& other) {
    v_ = _mm256_mul_ps(v_, other.v_);
    return *this;
  }
  PIK_INLINE V256& operator/=(const V256& other) {
    v_ = _mm256_div_ps(v_, other.v_);
    return *this;
  }
  PIK_INLINE V256& operator+=(const V256& other) {
    v_ = _mm256_add_ps(v_, other.v_);
    return *this;
  }
  PIK_INLINE V256& operator-=(const V256& other) {
    v_ = _mm256_sub_ps(v_, other.v_);
    return *this;
  }

  PIK_INLINE V256& operator&=(const V256& other) {
    v_ = _mm256_and_ps(v_, other.v_);
    return *this;
  }
  PIK_INLINE V256& operator|=(const V256& other) {
    v_ = _mm256_or_ps(v_, other.v_);
    return *this;
  }
  PIK_INLINE V256& operator^=(const V256& other) {
    v_ = _mm256_xor_ps(v_, other.v_);
    return *this;
  }

 private:
  __m256 v_;
};

template <>
class V256<double> {
 public:
  using T = double;
  static constexpr size_t N = 4;

  // Leaves v_ uninitialized - typically used for output parameters.
  PIK_INLINE V256() {}

  // Lane 0 (p_0) is the lowest.
  PIK_INLINE V256(T p_3, T p_2, T p_1, T p_0)
      : v_(_mm256_set_pd(p_3, p_2, p_1, p_0)) {}

  // Broadcasts to all lanes.
  PIK_INLINE explicit V256(T f) : v_(_mm256_set1_pd(f)) {}

  // Copy from other vector.
  PIK_INLINE explicit V256(const V256& other) : v_(other.v_) {}
  template <typename U>
  PIK_INLINE explicit V256(const V256<U>& other) : v_(other) {}
  PIK_INLINE V256& operator=(const V256& other) {
    v_ = other.v_;
    return *this;
  }

  // Convert from/to intrinsics.
  PIK_INLINE V256(const __m256d& v) : v_(v) {}
  PIK_INLINE V256& operator=(const __m256d& v) {
    v_ = v;
    return *this;
  }
  PIK_INLINE operator __m256d() const { return v_; }

  PIK_INLINE V256 operator==(const V256& other) const {
    return V256(_mm256_cmp_pd(v_, other.v_, 0));
  }
  PIK_INLINE V256 operator<(const V256& other) const {
    return V256(_mm256_cmp_pd(v_, other.v_, 1));
  }
  PIK_INLINE V256 operator>(const V256& other) const {
    return V256(_mm256_cmp_pd(other.v_, v_, 1));
  }

  PIK_INLINE V256& operator*=(const V256& other) {
    v_ = _mm256_mul_pd(v_, other.v_);
    return *this;
  }
  PIK_INLINE V256& operator/=(const V256& other) {
    v_ = _mm256_div_pd(v_, other.v_);
    return *this;
  }
  PIK_INLINE V256& operator+=(const V256& other) {
    v_ = _mm256_add_pd(v_, other.v_);
    return *this;
  }
  PIK_INLINE V256& operator-=(const V256& other) {
    v_ = _mm256_sub_pd(v_, other.v_);
    return *this;
  }

  PIK_INLINE V256& operator&=(const V256& other) {
    v_ = _mm256_and_pd(v_, other.v_);
    return *this;
  }
  PIK_INLINE V256& operator|=(const V256& other) {
    v_ = _mm256_or_pd(v_, other.v_);
    return *this;
  }
  PIK_INLINE V256& operator^=(const V256& other) {
    v_ = _mm256_xor_pd(v_, other.v_);
    return *this;
  }

 private:
  __m256d v_;
};

// Nonmember functions for any V256 via member functions.

template <typename T>
PIK_INLINE V256<T> operator*(const V256<T>& left, const V256<T>& right) {
  V256<T> t(left);
  return t *= right;
}

// Beware: only partially pipelined before Skylake!
template <typename T>
PIK_INLINE V256<T> operator/(const V256<T>& left, const V256<T>& right) {
  V256<T> t(left);
  return t /= right;
}

template <typename T>
PIK_INLINE V256<T> operator+(const V256<T>& left, const V256<T>& right) {
  V256<T> t(left);
  return t += right;
}

template <typename T>
PIK_INLINE V256<T> operator-(const V256<T>& left, const V256<T>& right) {
  V256<T> t(left);
  return t -= right;
}

template <typename T>
PIK_INLINE V256<T> operator&(const V256<T>& left, const V256<T>& right) {
  V256<T> t(left);
  return t &= right;
}

template <typename T>
PIK_INLINE V256<T> operator|(const V256<T> left, const V256<T>& right) {
  V256<T> t(left);
  return t |= right;
}

template <typename T>
PIK_INLINE V256<T> operator^(const V256<T>& left, const V256<T>& right) {
  V256<T> t(left);
  return t ^= right;
}

template <typename T>
PIK_INLINE V256<T> operator<<(const V256<T>& v, const int count) {
  V256<T> t(v);
  return t <<= count;
}

template <typename T>
PIK_INLINE V256<T> operator>>(const V256<T>& v, const int count) {
  V256<T> t(v);
  return t >>= count;
}

// We do not provide operator<<(V, __m128i) because it has 4 cycle latency
// (to broadcast the shift count). It is faster to use sllv_epi64 etc. instead.

using V32x8U = V256<uint8_t>;
using V16x16U = V256<uint16_t>;
using V16x16I = V256<int16_t>;
using V8x32U = V256<uint32_t>;
using V8x32I = V256<int32_t>;
using V4x64U = V256<uint64_t>;
using V8x32F = V256<float>;
using V4x64F = V256<double>;

// Load/Store for any V256.

// We differentiate between targets' vector types via template specialization.
// Calling Load<V>(floats) is more natural than Load(V8x32F(), floats) and may
// generate better code in unoptimized builds. The primary template can only
// be defined once, even if multiple vector headers are included.
#ifndef PIK_DEFINED_PRIMARY_TEMPLATE_FOR_LOAD
#define PIK_DEFINED_PRIMARY_TEMPLATE_FOR_LOAD
template <class V>
PIK_INLINE V Load(const typename V::T* const PIK_RESTRICT from) {
  return V();  // must specialize for each type.
}
template <class V>
PIK_INLINE V LoadUnaligned(const typename V::T* const PIK_RESTRICT from) {
  return V();  // must specialize for each type.
}
#endif

template <>
PIK_INLINE V32x8U Load(const V32x8U::T* const PIK_RESTRICT from) {
  const __m256i* const PIK_RESTRICT p = reinterpret_cast<const __m256i*>(from);
  return V32x8U(_mm256_load_si256(p));
}
template <>
PIK_INLINE V16x16U Load(const V16x16U::T* const PIK_RESTRICT from) {
  const __m256i* const PIK_RESTRICT p = reinterpret_cast<const __m256i*>(from);
  return V16x16U(_mm256_load_si256(p));
}
template <>
PIK_INLINE V16x16I Load(const V16x16I::T* const PIK_RESTRICT from) {
  const __m256i* const PIK_RESTRICT p = reinterpret_cast<const __m256i*>(from);
  return V16x16I(_mm256_load_si256(p));
}
template <>
PIK_INLINE V8x32U Load(const V8x32U::T* const PIK_RESTRICT from) {
  const __m256i* const PIK_RESTRICT p = reinterpret_cast<const __m256i*>(from);
  return V8x32U(_mm256_load_si256(p));
}
template <>
PIK_INLINE V8x32I Load(const V8x32I::T* const PIK_RESTRICT from) {
  const __m256i* const PIK_RESTRICT p = reinterpret_cast<const __m256i*>(from);
  return V8x32I(_mm256_load_si256(p));
}
template <>
PIK_INLINE V4x64U Load(const V4x64U::T* const PIK_RESTRICT from) {
  const __m256i* const PIK_RESTRICT p = reinterpret_cast<const __m256i*>(from);
  return V4x64U(_mm256_load_si256(p));
}
template <>
PIK_INLINE V8x32F Load(const V8x32F::T* const PIK_RESTRICT from) {
  return V8x32F(_mm256_load_ps(from));
}
template <>
PIK_INLINE V4x64F Load(const V4x64F::T* const PIK_RESTRICT from) {
  return V4x64F(_mm256_load_pd(from));
}

template <>
PIK_INLINE V32x8U LoadUnaligned(const V32x8U::T* const PIK_RESTRICT from) {
  const __m256i* const PIK_RESTRICT p = reinterpret_cast<const __m256i*>(from);
  return V32x8U(_mm256_loadu_si256(p));
}
template <>
PIK_INLINE V16x16U LoadUnaligned(const V16x16U::T* const PIK_RESTRICT from) {
  const __m256i* const PIK_RESTRICT p = reinterpret_cast<const __m256i*>(from);
  return V16x16U(_mm256_loadu_si256(p));
}
template <>
PIK_INLINE V16x16I LoadUnaligned(const V16x16I::T* const PIK_RESTRICT from) {
  const __m256i* const PIK_RESTRICT p = reinterpret_cast<const __m256i*>(from);
  return V16x16I(_mm256_loadu_si256(p));
}
template <>
PIK_INLINE V8x32U LoadUnaligned(const V8x32U::T* const PIK_RESTRICT from) {
  const __m256i* const PIK_RESTRICT p = reinterpret_cast<const __m256i*>(from);
  return V8x32U(_mm256_loadu_si256(p));
}
template <>
PIK_INLINE V8x32I LoadUnaligned(const V8x32I::T* const PIK_RESTRICT from) {
  const __m256i* const PIK_RESTRICT p = reinterpret_cast<const __m256i*>(from);
  return V8x32I(_mm256_loadu_si256(p));
}
template <>
PIK_INLINE V4x64U LoadUnaligned(const V4x64U::T* const PIK_RESTRICT from) {
  const __m256i* const PIK_RESTRICT p = reinterpret_cast<const __m256i*>(from);
  return V4x64U(_mm256_loadu_si256(p));
}
template <>
PIK_INLINE V8x32F LoadUnaligned(const V8x32F::T* const PIK_RESTRICT from) {
  return V8x32F(_mm256_loadu_ps(from));
}
template <>
PIK_INLINE V4x64F LoadUnaligned(const V4x64F::T* const PIK_RESTRICT from) {
  return V4x64F(_mm256_loadu_pd(from));
}

// "to" must be vector-aligned.
template <typename T>
PIK_INLINE void Store(const V256<T>& v, T* const PIK_RESTRICT to) {
  _mm256_store_si256(reinterpret_cast<__m256i * PIK_RESTRICT>(to), v);
}
PIK_INLINE void Store(const V256<float>& v, float* const PIK_RESTRICT to) {
  _mm256_store_ps(to, v);
}
PIK_INLINE void Store(const V256<double>& v, double* const PIK_RESTRICT to) {
  _mm256_store_pd(to, v);
}

template <typename T>
PIK_INLINE void StoreUnaligned(const V256<T>& v, T* const PIK_RESTRICT to) {
  _mm256_storeu_si256(reinterpret_cast<__m256i * PIK_RESTRICT>(to), v);
}
PIK_INLINE void StoreUnaligned(const V256<float>& v,
                               float* const PIK_RESTRICT to) {
  _mm256_storeu_ps(to, v);
}
PIK_INLINE void StoreUnaligned(const V256<double>& v,
                               double* const PIK_RESTRICT to) {
  _mm256_storeu_pd(to, v);
}

// Writes directly to (aligned) memory, bypassing the cache. This is useful for
// data that will not be read again in the near future.
template <typename T>
PIK_INLINE void Stream(const V256<T>& v, T* const PIK_RESTRICT to) {
  _mm256_stream_si256(reinterpret_cast<__m256i * PIK_RESTRICT>(to), v);
}
PIK_INLINE void Stream(const V256<float>& v, float* const PIK_RESTRICT to) {
  _mm256_stream_ps(to, v);
}
PIK_INLINE void Stream(const V256<double>& v, double* const PIK_RESTRICT to) {
  _mm256_stream_pd(to, v);
}

// Miscellaneous functions.

template <typename T>
PIK_INLINE V256<T> RotateLeft(const V256<T>& v, const int count) {
  constexpr size_t num_bits = sizeof(T) * 8;
  return (v << count) | (v >> (num_bits - count));
}

template <typename T>
PIK_INLINE V256<T> AndNot(const V256<T>& neg_mask, const V256<T>& values) {
  return V256<T>(_mm256_andnot_si256(neg_mask, values));
}
template <>
PIK_INLINE V256<float> AndNot(const V256<float>& neg_mask,
                              const V256<float>& values) {
  return V256<float>(_mm256_andnot_ps(neg_mask, values));
}
template <>
PIK_INLINE V256<double> AndNot(const V256<double>& neg_mask,
                               const V256<double>& values) {
  return V256<double>(_mm256_andnot_pd(neg_mask, values));
}

PIK_INLINE V8x32F Select(const V8x32F& a, const V8x32F& b, const V8x32F& mask) {
  return V8x32F(_mm256_blendv_ps(a, b, mask));
}

PIK_INLINE V4x64F Select(const V4x64F& a, const V4x64F& b, const V4x64F& mask) {
  return V4x64F(_mm256_blendv_pd(a, b, mask));
}

// Uses the current rounding mode, which defaults to round-to-nearest.
PIK_INLINE V8x32I RoundToInt(const V8x32F& x) {
  return V8x32I(_mm256_cvtps_epi32(x));
}

PIK_INLINE V8x32F FloatFromInt(const V8x32I& x) {
  return V8x32F(_mm256_cvtepi32_ps(x));
}

// 3 cycle latency due to 128-bit lane split. For single-cycle latency,
// broadcast to both lanes and then unpack/shuffle within each.
PIK_INLINE V8x32I IntFromU8(const __m128i& x) {
  return V8x32I(_mm256_cvtepu8_epi32(x));
}

PIK_INLINE V8x32I IntFromU16(const __m128i& x) {
  return V8x32I(_mm256_cvtepu16_epi32(x));
}

// Returns (((a * b) >> 14) + 1) >> 1.
PIK_INLINE V16x16I MulHRS16(const V16x16I& a, const V16x16I& b) {
  return V16x16I(_mm256_mulhrs_epi16(a, b));
}

// Returns hi[11] lo[11] .. hi[8] lo[8] | hi[3] lo[3] .. hi[0] lo[0]
PIK_INLINE V8x32I UnpackLo(const V16x16I& lo, const V16x16I& hi) {
  return V8x32I(_mm256_unpacklo_epi16(lo, hi));
}

PIK_INLINE V8x32U BitsFromFloat(const V8x32F& x) {
  return V8x32U(_mm256_castps_si256(x));
}

PIK_INLINE V8x32F FloatFromBits(const V8x32I& bits) {
  return V8x32F(_mm256_castsi256_ps(bits));
}

PIK_INLINE V8x32F FloatFromBits(const V8x32U& bits) {
  return V8x32F(_mm256_castsi256_ps(bits));
}

PIK_INLINE V8x32F Abs(const V8x32F& x, const V8x32U& kSignBit) {
  return AndNot(FloatFromBits(kSignBit), x);
}

// Returns mul1 * mul2 + add.
PIK_INLINE V8x32F MulAdd(const V8x32F& mul1, const V8x32F& mul2,
                         const V8x32F& add) {
  return V8x32F(_mm256_fmadd_ps(mul1, mul2, add));
}

PIK_INLINE V4x64F MulAdd(const V4x64F& mul1, const V4x64F& mul2,
                         const V4x64F& add) {
  return V4x64F(_mm256_fmadd_pd(mul1, mul2, add));
}

// Returns mul1 * mul2 - sub.
PIK_INLINE V8x32F MulSub(const V8x32F& mul1, const V8x32F& mul2,
                         const V8x32F& sub) {
  return V8x32F(_mm256_fmsub_ps(mul1, mul2, sub));
}

// Returns a 12-bit approximation of 1/x.
PIK_INLINE V8x32F Reciprocal12(const V8x32F& x) {
  return V8x32F(_mm256_rcp_ps(x));
}

// Returns a 12-bit approximation of x^(-1/2). Multiply the result by x to
// approximate sqrt(x).
PIK_INLINE V8x32F ReciprocalSquareRoot12(const V8x32F& x) {
  return V8x32F(_mm256_rsqrt_ps(x));
}

// Beware: only partially pipelined before Skylake!
PIK_INLINE V8x32F SquareRoot(const V8x32F& x) {
  return V8x32F(_mm256_sqrt_ps(x));
}

// Returns a 12-bit approximation of sqrt(x).
PIK_INLINE V8x32F SquareRoot12(const V8x32F& x) {
  return ReciprocalSquareRoot12(x) * x;
}

// Min/Max

PIK_INLINE V32x8U Min(const V32x8U& v0, const V32x8U& v1) {
  return V32x8U(_mm256_min_epu8(v0, v1));
}

PIK_INLINE V32x8U Max(const V32x8U& v0, const V32x8U& v1) {
  return V32x8U(_mm256_max_epu8(v0, v1));
}

PIK_INLINE V16x16U Min(const V16x16U& v0, const V16x16U& v1) {
  return V16x16U(_mm256_min_epu16(v0, v1));
}

PIK_INLINE V16x16U Max(const V16x16U& v0, const V16x16U& v1) {
  return V16x16U(_mm256_max_epu16(v0, v1));
}

PIK_INLINE V16x16I Min(const V16x16I& v0, const V16x16I& v1) {
  return V16x16I(_mm256_min_epi16(v0, v1));
}

PIK_INLINE V16x16I Max(const V16x16I& v0, const V16x16I& v1) {
  return V16x16I(_mm256_max_epi16(v0, v1));
}

PIK_INLINE V8x32U Min(const V8x32U& v0, const V8x32U& v1) {
  return V8x32U(_mm256_min_epu32(v0, v1));
}

PIK_INLINE V8x32U Max(const V8x32U& v0, const V8x32U& v1) {
  return V8x32U(_mm256_max_epu32(v0, v1));
}

PIK_INLINE V8x32I Min(const V8x32I& v0, const V8x32I& v1) {
  return V8x32I(_mm256_min_epi32(v0, v1));
}

PIK_INLINE V8x32I Max(const V8x32I& v0, const V8x32I& v1) {
  return V8x32I(_mm256_max_epi32(v0, v1));
}

PIK_INLINE V8x32F Min(const V8x32F& v0, const V8x32F& v1) {
  return V8x32F(_mm256_min_ps(v0, v1));
}

PIK_INLINE V8x32F Max(const V8x32F& v0, const V8x32F& v1) {
  return V8x32F(_mm256_max_ps(v0, v1));
}

PIK_INLINE V4x64F Min(const V4x64F& v0, const V4x64F& v1) {
  return V4x64F(_mm256_min_pd(v0, v1));
}

PIK_INLINE V4x64F Max(const V4x64F& v0, const V4x64F& v1) {
  return V4x64F(_mm256_max_pd(v0, v1));
}

}  // namespace PIK_TARGET_NAME
}  // namespace pik

#endif  // #ifndef PIK_DISABLE_TARGET_SPECIFIC
#endif  // VECTOR256_H_
