// Library to decode the ANS population counts from the bit-stream and build a
// decoding table from them.

#ifndef ANS_DECODE_H_
#define ANS_DECODE_H_

#include <stddef.h>
#include <stdint.h>
#include <vector>

#include "ans_params.h"
#include "bit_reader.h"
#include "brunsli_v2_input.h"
#include "byte_order.h"

namespace pik {

struct ANSSymbolInfo {
  uint16_t offset_;
  uint16_t freq_;
  uint8_t symbol_;
};

struct ANSDecodingData {
  ANSDecodingData() {}

  bool ReadFromBitStream(BitReader* input);

  ANSSymbolInfo map_[ANS_TAB_SIZE];
};

class ANSDecoder {
 public:
  ANSDecoder() : state_(0) {}

  void Init(BrunsliV2Input* in) {
    state_ = in->GetNextWord();
    state_ = (state_ << 16) | in->GetNextWord();
  }

  int ReadSymbol(const ANSDecodingData& code, BrunsliV2Input* in) {
    const uint32_t res = state_ & (ANS_TAB_SIZE - 1);
    const ANSSymbolInfo& s = code.map_[res];
    state_ = s.freq_ * (state_ >> ANS_LOG_TAB_SIZE) + s.offset_;
    if (state_ < (1u << 16)) {
      state_ = (state_ << 16) | in->GetNextWord();
    }
    return s.symbol_;
  }
  uint32_t GetState() const { return state_; }
  bool CheckCRC() const { return state_ == (ANS_SIGNATURE << 16); }

 private:
  uint32_t state_;
};

struct ANSCode {
  struct ANSSymbolInfo {
    uint16_t offset;
    uint16_t freq;
  };
  std::vector<uint16_t> map;
  // indexed by (entropy_code_id << ANS_LOG_TAB_SIZE) + symbol.
  std::vector<ANSSymbolInfo> info;
};

bool DecodeANSCodes(const size_t num_histograms, const size_t max_alphabet_size,
                    const uint8_t* symbol_lut, size_t symbol_lut_size,
                    BitReader* in, ANSCode* result);

class ANSSymbolReader {
 public:
  ANSSymbolReader(const ANSCode* code) : code_(code) {}

  PIK_INLINE int ReadSymbol(const int histo_idx, BitReader* PIK_RESTRICT br) {
    if (PIK_UNLIKELY(symbols_left_ == 0)) {
      state_ = br->ReadBits(16);
      state_ = (state_ << 16) | br->ReadBits(16);
      br->FillBitBuffer();
      symbols_left_ = kANSBufferSize;
    }
    const uint32_t res = state_ & (ANS_TAB_SIZE - 1);
    const int histo_offset = histo_idx << ANS_LOG_TAB_SIZE;

#if PIK_BYTE_ORDER_LITTLE
    uint32_t s32;
    memcpy(&s32, &code_->map[histo_offset + res], sizeof(s32));
    const size_t symbol = s32 & 0xFFFF;

    memcpy(&s32, &code_->info[histo_offset + symbol], sizeof(s32));
    const uint32_t offset = s32 & 0xFFFF;
    const uint32_t freq = s32 >> 16;
    state_ = freq * (state_ >> ANS_LOG_TAB_SIZE) + res - offset;
#else
    const uint16_t symbol = code_->map[histo_offset + res];
    const ANSCode::ANSSymbolInfo s = code_->info[histo_offset + symbol];
    state_ = s.freq * (state_ >> ANS_LOG_TAB_SIZE) + res - s.offset;
#endif
    --symbols_left_;
    if (PIK_UNLIKELY(state_ < (1u << 16))) {
      state_ = (state_ << 16) | br->PeekFixedBits<16>();
      br->Advance(16);
    }
    return symbol;
  }

  bool CheckANSFinalState() { return state_ == (ANS_SIGNATURE << 16); }

 private:
  size_t symbols_left_ = 0;
  uint32_t state_ = ANS_SIGNATURE << 16;
  const ANSCode* code_;
};

}  // namespace pik

#endif  // ANS_DECODE_H_
