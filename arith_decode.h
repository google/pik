#ifndef ARITH_DECODE_H_
#define ARITH_DECODE_H_

#include <stdint.h>

#include "distributions.h"

namespace pik {

// A class used for entropy decoding a sequence of binary values.
// skal@ wrote the original version, szabadka@ ported it for brunsli.
class BinaryArithmeticDecoder {
 public:
  BinaryArithmeticDecoder() : low_(0), high_(~0), value_(0) {}

  template <class Reader>
  void Init(Reader* in) {
    value_ = in->GetNextWord();
    value_ = (value_ << 16) | in->GetNextWord();
  }

  // Returns the next bit decoded from the bit stream, based on the given N-bit
  // precision probability, i.e. P(bit = 0) = prob / 2^N. This probability must
  // be the same as the one used by the encoder.
  template <int N, class Reader>
  int ReadBit(int prob, Reader* in) {
    const uint32_t diff = high_ - low_;
    const uint32_t split = low_ + (((uint64_t)diff * prob) >> N);
    int bit;
    if (value_ > split) {
      low_ = split + 1;
      bit = 1;
    } else {
      high_ = split;
      bit = 0;
    }
    if (((low_ ^ high_) >> 16) == 0) {
      value_ = (value_ << 16) | in->GetNextWord();
      low_ <<= 16;
      high_ <<= 16;
      high_ |= 0xffff;
    }
    return bit;
  }

  template <class Reader>
  int ReadBit(Prob* p, Reader* in) {
    int val = ReadBit<8>(p->get_proba(), in);
    p->Add(val);
    return val;
  }

 private:
  uint32_t low_;
  uint32_t high_;
  uint32_t value_;
};

}  // namespace pik

#endif  // ARITH_DECODE_H_
