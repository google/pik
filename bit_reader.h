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

#ifndef BIT_READER_H_
#define BIT_READER_H_

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>  // memcpy

#include "compiler_specific.h"
#include "status.h"

namespace pik {

// Adapter for reading individual bits from a fixed memory buffer, can Peek
// up to 30 bits at a time. Reads 4 bytes (or len % 4 at the end) of input at a
// time into its accumulator. Performs bounds-checking, returns all-zero values
// after the memory buffer is depleted.
class BitReader {
 private:
 public:
  // data is not necessarily 4-byte aligned nor padded to RoundUp(len, 4).
  BitReader(const uint8_t* const PIK_RESTRICT data, const size_t len)
      : data32_(reinterpret_cast<const uint32_t*>(data)),
        len32_(len >> 2),
        len_mod4_(len % 4),
        val_(0),
        pos32_(0),
        bit_pos_(64) {
    FillBitBuffer();
  }

  void FillBitBuffer() {
    if (PIK_UNLIKELY(bit_pos_ >= 32)) {
      bit_pos_ -= 32;
      val_ >>= 32;

      if (PIK_LIKELY(pos32_ < len32_)) {
        // Read unaligned (memcpy avoids ubsan warning)
        uint32_t next;
        memcpy(&next, data32_ + pos32_, sizeof(next));
        val_ |= static_cast<uint64_t>(next) << 32;
      } else if (pos32_ == len32_) {
        // Only read the valid bytes.
        const uint8_t* bytes =
            reinterpret_cast<const uint8_t*>(data32_ + pos32_);
        uint64_t next = 0;
        for (size_t i = 0; i < len_mod4_; ++i) {
          // Pre-shifted by 32 so we can inject into val_ directly.
          // Assumes little-endian byte order.
          next |= static_cast<uint64_t>(bytes[i]) << (i * 8 + 32);
        }
        val_ |= next;
      }
      ++pos32_;
    }
  }

  void Advance(int num_bits) {
    bit_pos_ += num_bits;
  }

  template<int N>
  int PeekFixedBits() const {
    static_assert(N <= 30, "At most 30 bits may be read.");
    PIK_ASSERT(N + bit_pos_ <= 64);
    return (val_ >> bit_pos_) & ((1 << N) - 1);
  }

  int PeekBits(int nbits) const {
    PIK_ASSERT(nbits <= 30);
    PIK_ASSERT(nbits + bit_pos_ <= 64);
    return (val_ >> bit_pos_) & ((1 << nbits) - 1);
  }

  int ReadBits(int nbits) {
    PIK_ASSERT(nbits <= 30);
    FillBitBuffer();
    int bits = PeekBits(nbits);
    bit_pos_ += nbits;
    return bits;
  }

  uint16_t GetNextWord() {
    return static_cast<uint16_t>(ReadBits(16));
  }

  void JumpToByteBoundary() {
    int rem = bit_pos_ % 8;
    if (rem > 0) ReadBits(8 - rem);
  }

  // Returns the (rounded up) number of bytes consumed so far.
  size_t Position() const {
    size_t bits_read = 32 * pos32_ + bit_pos_ - 64;
    return (bits_read + 7) / 8;
  }

 private:
  // *32 counters/pointers are in units of 4 bytes, or 32 bits.
  const uint32_t* const PIK_RESTRICT data32_;
  const size_t len32_;
  const size_t len_mod4_;
  uint64_t val_;
  size_t pos32_;
  size_t bit_pos_;
};

}  // namespace pik

#endif  // BIT_READER_H_
