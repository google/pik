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

#include "compiler_specific.h"
#include "status.h"

namespace pik {

// Adapter for reading individual bits from a fixed memory buffer, can read up
// to 30 bits at a time. Reads 4 bytes of input at a time into its accumulator.
// Performs bounds-checking, returns only 0 bit values after memory buffer
// is depleted.
class BitReader {
 public:
  BitReader(const uint8_t* const PIK_RESTRICT data, const size_t len)
      : data32_(reinterpret_cast<const uint32_t*>(data)),
        len32_(len >> 2),
        val_(static_cast<uint64_t>(data32_[0]) << 32),
        pos32_(1),
        bit_pos_(32) {
    PIK_ASSERT(len % 4 == 0);
  }

  void FillBitBuffer() {
    if (PIK_UNLIKELY(bit_pos_ >= 32)) {
      val_ >>= 32;
      if (pos32_ < len32_) {
        val_ |= static_cast<uint64_t>(data32_[pos32_]) << 32;
      }
      ++pos32_;
      bit_pos_ -= 32;
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

  // Returns the byte position, aligned to 4 bytes, where the next chunk of
  // data should be read from after all symbols have been decoded.
  size_t Position() const {
    size_t bits_read = 32 * pos32_ + bit_pos_ - 64;
    size_t bytes_read = (bits_read + 7) / 8;
    return (bytes_read + 3) & ~3;
  }

 private:
  // *32 counters/pointers are in units of 4 bytes, or 32 bits.
  const uint32_t* const PIK_RESTRICT data32_;
  const size_t len32_;
  uint64_t val_;
  size_t pos32_;
  size_t bit_pos_;
};

}  // namespace pik

#endif  // BIT_READER_H_
