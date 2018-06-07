#ifndef BRUNSLI_V2_INPUT_H_
#define BRUNSLI_V2_INPUT_H_

#include <stdint.h>
#include <cstddef>

namespace pik {

// TODO(user) Make this work with odd lengths, use 64-bit buffer and
// load 32-bit pieces at a time.
struct BrunsliV2Input {
  BrunsliV2Input(const uint8_t* data, size_t len) :
      data_(reinterpret_cast<const uint16_t*>(data)),
      len_(len >> 1), pos_(0), val_(0), bit_pos_(0), error_(0) {}

  void InitBitReader() {
    val_ = GetNextWord();
  }

  uint16_t GetNextWord() {
    uint16_t val = 0;
    if (pos_ < len_) {
      val = data_[pos_];
    } else {
      error_ = 1;
    }
    ++pos_;
    return val;
  }

  int PeekBits(int nbits) const {
    if (bit_pos_ + nbits > 16 && pos_ < len_) {
      return (((data_[pos_] << 16) | val_) >> bit_pos_) & ((1 << nbits) - 1);
    } else {
      return (val_ >> bit_pos_) & ((1 << nbits) - 1);
    }
  }

  int ReadBits(int nbits) {
    if (bit_pos_ + nbits > 16) {
      uint32_t new_bits = GetNextWord();
      val_ |= new_bits << 16;
    }
    int retval = (val_ >> bit_pos_) & ((1 << nbits) - 1);
    bit_pos_ += nbits;
    if (bit_pos_ > 16) {
      bit_pos_ -= 16;
      val_ >>= 16;
    }
    return retval;
  }

  const uint16_t* data_;
  const size_t len_;
  size_t pos_;
  uint32_t val_;
  int bit_pos_;
  int error_;
};

}  // namespace pik

#endif  // BRUNSLI_V2_INPUT_H_
