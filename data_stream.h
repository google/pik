#ifndef DATA_STREAM_H_
#define DATA_STREAM_H_

#include <stddef.h>
#include <stdint.h>
#include <memory>
#include <vector>

#include "ans_encode.h"
#include "distributions.h"
#include "entropy_source.h"
#include "status.h"

namespace pik {

// Manages the multiplexing of the ANS-coded and arithmetic coded bits.
class DataStream {
 public:
  explicit DataStream(int num_contexts)
      : pos_(3),
        bw_pos_(0),
        ac_pos0_(1),
        ac_pos1_(2),
        low_(0),
        high_(~0),
        bw_val_(0),
        bw_bitpos_(0),
        num_contexts_(num_contexts) {}

  void Resize(int max_num_code_words) {
    code_words_.resize(max_num_code_words);
  }

  void ResizeForBlock() {
    if (pos_ + kSlackForOneBlock > code_words_.size()) {
      static const float kGrowMult = 1.2;
      const size_t new_size =
          kGrowMult * code_words_.capacity() + kSlackForOneBlock;
      code_words_.resize(new_size);
    }
  }

  void AddCode(int code, int band, int context, EntropySource* s) {
    int histo_ix = band * num_contexts_ + context;
    CodeWord word;
    word.context = histo_ix;
    word.code = code;
    word.nbits = 0;
    word.value = 0;
    PIK_ASSERT(pos_ < code_words_.size());
    code_words_[pos_++] = word;
    s->AddCode(code, histo_ix);
  }

  void AddBits(int nbits, int bits) {
    bw_val_ |= (bits << bw_bitpos_);
    bw_bitpos_ += nbits;
    if (bw_bitpos_ > 16) {
      CodeWord word;
      word.context = 0;
      word.code = 0;
      word.nbits = 16;
      word.value = bw_val_ & 0xffff;
      code_words_[bw_pos_] = word;
      bw_pos_ = pos_;
      ++pos_;
      bw_val_ >>= 16;
      bw_bitpos_ -= 16;
    }
  }

  void FlushArithmeticCoder() {
    code_words_[ac_pos0_].value = high_ >> 16;
    code_words_[ac_pos1_].value = high_ & 0xffff;
    code_words_[ac_pos0_].nbits = 16;
    code_words_[ac_pos1_].nbits = 16;
    low_ = 0;
    high_ = ~0;
  }

  void FlushBitWriter() {
    code_words_[bw_pos_].nbits = 16;
    code_words_[bw_pos_].value = bw_val_ & 0xffff;
  }

  // Encodes the next bit to the bit stream, based on the 8-bit precision
  // probability, i.e. P(bit = 0) = prob / 256. Statistics are updated in 'p'.
  void AddBit(Prob* const p, int bit) {
    const uint8_t prob = p->get_proba();
    p->Add(bit);
    const uint32_t diff = high_ - low_;
    const uint32_t split = low_ + (((uint64_t)diff * prob) >> 8);
    if (bit) {
      low_ = split + 1;
    } else {
      high_ = split;
    }
    if (((low_ ^ high_) >> 16) == 0) {
      code_words_[ac_pos0_].value = high_ >> 16;
      code_words_[ac_pos0_].nbits = 16;
      ac_pos0_ = ac_pos1_;
      ac_pos1_ = pos_;
      ++pos_;
      low_ <<= 16;
      high_ <<= 16;
      high_ |= 0xffff;
    }
  }

  void EncodeCodeWords(EntropySource* s, size_t* storage_ix, uint8_t* storage) {
    FlushBitWriter();
    FlushArithmeticCoder();
    uint16_t* out = reinterpret_cast<uint16_t*>(storage);
    const uint16_t* out_start = out;
    if (s != nullptr) {
      ANSCoder ans;
      for (int i = pos_ - 1; i >= 0; --i) {
        CodeWord* const word = &code_words_[i];
        if (word->nbits == 0) {
          const ANSEncSymbolInfo info =
              s->GetANSTable(word->context)[word->code];
          word->value = ans.PutSymbol(info, &word->nbits);
        }
      }
      const uint32_t state = ans.GetState();
      *out++ = (state >> 16) & 0xffff;
      *out++ = (state >>  0) & 0xffff;
    }
    for (int i = 0; i < pos_; ++i) {
      CodeWord word = code_words_[i];
      if (word.nbits) {
        *out++ = word.value;
      }
    }
    *storage_ix += (out - out_start) * 16;
  }

 private:
  struct CodeWord {
    uint32_t context;
    uint16_t value;
    uint8_t code;
    uint8_t nbits;
  };
  static const size_t kSlackForOneBlock = 1024;

  int pos_;
  int bw_pos_;
  int ac_pos0_;
  int ac_pos1_;
  uint32_t low_;
  uint32_t high_;
  uint32_t bw_val_;
  int bw_bitpos_;
  const int num_contexts_;
  std::vector<CodeWord> code_words_;
};

}  // namespace pik

#endif  // DATA_STREAM_H_
