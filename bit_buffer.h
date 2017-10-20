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

#ifndef BIT_BUFFER_H_
#define BIT_BUFFER_H_

// FIFO queues for serializing multi-bit codes to/from byte-aligned memory.
//
// Uses a 64-bit SSE lane as the buffer. Unlike regular shifts, SSE shifts do
// not depend on the CL register (successive shifts are independent) and also
// allow shifting by 64 bits (avoids conditional branches). It is faster to use
// 128-bit vectors rather than AVX-2 (easier to initialize).
//
// Both the source and sink read and write 32-bits at a time, with fewer
// conditional branches (amortized over several Insert/Extract). It is faster
// for the sink to insert values into the lower part of an SSE register.
// To maintain the FIFO property, we use byte swapping.

#include <stddef.h>
#include <stdint.h>
#include <cstring>

#include "arch_specific.h"
#include "byte_order.h"
#include "compiler_specific.h"
#include "simd/simd.h"

namespace pik {
namespace SIMD_NAMESPACE {

// Accumulates variable-sized codes and writes them to memory in 32-bit units.
class BitSink {
  static const Part<uint64_t, 2> d;

 public:
  // There are 32 lower bits to fill before reaching the upper bits.
  BitSink(uint8_t* const PIK_RESTRICT storage)
      : buffer_(setzero(d)), upper_bits_used_(-32), write_pos_(storage) {}

  // It is safe to insert a total of <= 32 bits after construction, or the last
  // call to CanWrite32 that returned false.
  template <int num_bits>
  void Insert(const uint64_t code) {
    buffer_ = shift_left<num_bits>(buffer_);
    buffer_ |= set(d, code);
    upper_bits_used_ += num_bits;
  }

  // (Slightly less efficient code: num_bits is loaded into a vector.)
  void InsertVariableCount(const int num_bits, const uint64_t code) {
    buffer_ = shift_left_same(buffer_, set_shift_left_count(d, num_bits));
    buffer_ |= set(d, code);
    upper_bits_used_ += num_bits;
  }

  // Returns how many bits have been inserted but not yet written.
  size_t BitsPending() const { return upper_bits_used_ + 32; }

  // Returns whether at least 32 bits have been inserted into the buffer.
  bool CanWrite32() const { return upper_bits_used_ >= 0; }

  // Writes the oldest 32 bits to memory in big-endian order, such that
  // BitSource will return them in FIFO order. Precondition: CanWrite32.
  // Postcondition: the buffer has space for at least 32 more bits.
  void Write32() {
    const auto oldest_32 =
        shift_right_same(buffer_, set_shift_right_count(d, upper_bits_used_));
    const uint32_t out = PIK_BSWAP32(uint32_t(get(d, oldest_32)));
    memcpy(write_pos_, &out, 4);
    write_pos_ += 4;
    upper_bits_used_ -= 32;
  }

  // Writes any remaining bits to (byte-aligned) memory. Returns STL-style end.
  // Do not call other functions after this.
  uint8_t* const PIK_RESTRICT Finalize() {
    // Left-align oldest bit, inserting zeros at the bottom.
    buffer_ = shift_left_same(buffer_,
                              set_shift_left_count(d, 32 - upper_bits_used_));
    const uint64_t out = PIK_BSWAP64(get(d, buffer_));

    // Copy exactly the required number of bytes (other threads may be writing
    // at subsequent positions).
    const int bytes = (BitsPending() + 7) / 8;
    switch (bytes) {
      case 8:
        memcpy(write_pos_, &out, 8);
        return write_pos_ + 8;
      case 7:
        memcpy(write_pos_, &out, 4);
        write_pos_[4] = static_cast<uint8_t>(out >> 32);
        write_pos_[5] = static_cast<uint8_t>(out >> 40);
        write_pos_[6] = static_cast<uint8_t>(out >> 48);
        return write_pos_ + 7;
      case 6:
        memcpy(write_pos_, &out, 4);
        write_pos_[4] = static_cast<uint8_t>(out >> 32);
        write_pos_[5] = static_cast<uint8_t>(out >> 40);
        return write_pos_ + 6;
      case 5:
        memcpy(write_pos_, &out, 4);
        write_pos_[4] = static_cast<uint8_t>(out >> 32);
        return write_pos_ + 5;
      case 4:
        memcpy(write_pos_, &out, 4);
        return write_pos_ + 4;
      case 3:
        write_pos_[0] = static_cast<uint8_t>(out);
        write_pos_[1] = static_cast<uint8_t>(out >> 8);
        write_pos_[2] = static_cast<uint8_t>(out >> 16);
        return write_pos_ + 3;
      case 2:
        write_pos_[0] = static_cast<uint8_t>(out);
        write_pos_[1] = static_cast<uint8_t>(out >> 8);
        return write_pos_ + 2;
      case 1:
        write_pos_[0] = static_cast<uint8_t>(out);
        return write_pos_ + 1;
      case 0:
        return write_pos_;
      default:
        PIK_UNREACHABLE;
    }
  }

 private:
  Part<uint64_t, 2>::V buffer_;
  int upper_bits_used_;
  uint8_t* PIK_RESTRICT write_pos_;
};

// Reads from memory in 64 or 32-bit units and extracts variable-size codes.
class BitSource {
  static const Part<uint64_t, 2> d;

 public:
  // Reads exactly 64 bits.
  BitSource(const uint8_t* const PIK_RESTRICT from) : read_pos_(from + 8) {
    uint64_t bits;
    memcpy(&bits, from, 8);
    buffer_ = set(d, PIK_BSWAP64(bits));
    // There are 32 upper bits to extract before reaching the lower bits.
    lower_bits_extracted_ = -32;
  }

  // Returns the next code ("num_bits" wide). It is safe to extract a total of
  // <= 64 bits after construction, or <= 32 bits after the last call to
  // CanRead32 that returned false.
  template <int num_bits>
  size_t Extract() {
    const auto bits = shift_right<64 - num_bits>(buffer_);
    const uint64_t code = get(d, bits);
    buffer_ = shift_left<num_bits>(buffer_);
    lower_bits_extracted_ += num_bits;
    return code;
  }

  // (Slightly less efficient code: num_bits is loaded into a vector.)
  size_t ExtractVariableCount(const int num_bits) {
    const auto bits =
        shift_right_same(buffer_, set_shift_right_count(d, 64 - num_bits));
    const uint64_t code = get(d, bits);
    buffer_ = shift_left_same(buffer_, set_shift_left_count(d, num_bits));
    lower_bits_extracted_ += num_bits;
    return code;
  }

  // Returns whether the buffer has space for reading 32 more bits.
  bool CanRead32() const { return (lower_bits_extracted_ >= 0); }

  // Reads the next 32 bits into the buffer. Precondition: CanRead32 =>
  // lower_bits_extracted_ >= 0 => buffer_[0,32) == 0.
  void Read32() {
    const auto shift = set_shift_left_count(d, lower_bits_extracted_);
    uint32_t bits;
    memcpy(&bits, read_pos_, 4);
    read_pos_ += 4;
    const auto vbits = set(d, uint64_t(PIK_BSWAP32(bits)));
    // The upper half may have had some zeros at the bottom, so match that.
    buffer_ += shift_left_same(vbits, shift);
    lower_bits_extracted_ -= 32;
  }

  // Returns the byte-aligned end position after all extracted bits.
  const uint8_t* const PIK_RESTRICT Finalize() const {
    const size_t excess_bytes = (32 - lower_bits_extracted_) / 8;
    return read_pos_ - excess_bytes;
  }

 private:
  Part<uint64_t, 2>::V buffer_;
  int lower_bits_extracted_;
  const uint8_t* PIK_RESTRICT read_pos_;
};

}  // namespace SIMD_NAMESPACE
}  // namespace pik

#endif  // BIT_BUFFER_H_
