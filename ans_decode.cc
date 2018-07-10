#include "ans_decode.h"

#include <vector>

#include "ans_common.h"
#include "fast_log.h"

namespace pik {
namespace {

bool ANSBuildMapTable(const std::vector<int>& counts,
                      ANSSymbolInfo map[ANS_TAB_SIZE]) {
  int i;
  int pos = 0;
  for (i = 0; i < counts.size(); ++i) {
    int j;
    for (j = 0; j < counts[i]; ++j, ++pos) {
      map[pos].symbol_ = i;
      map[pos].freq_ = counts[i];
      map[pos].offset_ = j;
    }
  }
  return (pos == ANS_TAB_SIZE);
}

// Decodes a number in the range [0..65535], by reading 1 - 20 bits.
inline int DecodeVarLenUint16(BitReader* input) {
  if (input->ReadBits(1)) {
    int nbits = static_cast<int>(input->ReadBits(4));
    if (nbits == 0) {
      return 1;
    } else {
      return static_cast<int>(input->ReadBits(nbits)) + (1 << nbits);
    }
  }
  return 0;
}

bool ReadHistogram(int precision_bits, std::vector<int>* counts,
                   BitReader* input) {
  int simple_code = input->ReadBits(1);
  if (simple_code == 1) {
    int i;
    int symbols[2] = { 0 };
    int max_symbol = 0;
    const int num_symbols = input->ReadBits(1) + 1;
    for (i = 0; i < num_symbols; ++i) {
      symbols[i] = DecodeVarLenUint16(input);
      if (symbols[i] > max_symbol) max_symbol = symbols[i];
    }
    counts->resize(max_symbol + 1);
    if (num_symbols == 1) {
      (*counts)[symbols[0]] = 1 << precision_bits;
    } else {
      if (symbols[0] == symbols[1]) {  // corrupt data
        return false;
      }
      (*counts)[symbols[0]] = input->ReadBits(precision_bits);
      (*counts)[symbols[1]] = (1 << precision_bits) - (*counts)[symbols[0]];
    }
  } else {
    int is_flat = input->ReadBits(1);
    if (is_flat == 1) {
      int alphabet_size = input->ReadBits(precision_bits);
      if (alphabet_size == 0) {
        return PIK_FAILURE("Invalid alphabet size for flat histogram.");
      }
      *counts = CreateFlatHistogram(alphabet_size, 1 << precision_bits);
      return true;
    }
    int length = DecodeVarLenUint16(input) + 3;
    counts->resize(length);
    int total_count = 0;
    static const uint8_t huff[64][2] = {
      {2, 6}, {3, 7}, {3, 4}, {4, 1}, {2, 6}, {3, 8}, {3, 5}, {4, 3},
      {2, 6}, {3, 7}, {3, 4}, {4, 2}, {2, 6}, {3, 8}, {3, 5}, {5, 0},
      {2, 6}, {3, 7}, {3, 4}, {4, 1}, {2, 6}, {3, 8}, {3, 5}, {4, 3},
      {2, 6}, {3, 7}, {3, 4}, {4, 2}, {2, 6}, {3, 8}, {3, 5}, {6, 9},
      {2, 6}, {3, 7}, {3, 4}, {4, 1}, {2, 6}, {3, 8}, {3, 5}, {4, 3},
      {2, 6}, {3, 7}, {3, 4}, {4, 2}, {2, 6}, {3, 8}, {3, 5}, {5, 0},
      {2, 6}, {3, 7}, {3, 4}, {4, 1}, {2, 6}, {3, 8}, {3, 5}, {4, 3},
      {2, 6}, {3, 7}, {3, 4}, {4, 2}, {2, 6}, {3, 8}, {3, 5}, {6, 10},
    };
    std::vector<int> logcounts(counts->size());
    int omit_log = -1;
    int omit_pos = -1;
    for (int i = 0; i < logcounts.size(); ++i) {
      input->FillBitBuffer();
      int idx = input->PeekFixedBits<6>();
      input->Advance(huff[idx][0]);
      logcounts[i] = huff[idx][1];
      if (logcounts[i] > omit_log) {
        omit_log = logcounts[i];
        omit_pos = i;
      }
    }
    for (int i = 0; i < logcounts.size(); ++i) {
      int code = logcounts[i];
      if (i == omit_pos) {
        continue;
      } else if (code == 0) {
        continue;
      } else if (code == 1) {
        (*counts)[i] = 1;
      } else {
        int bitcount = GetPopulationCountPrecision(code - 1);
        (*counts)[i] = (1 << (code - 1)) +
            (input->ReadBits(bitcount) << (code - 1 - bitcount));
      }
      total_count += (*counts)[i];
    }
    PIK_ASSERT(omit_pos >= 0);
    (*counts)[omit_pos] = (1 << precision_bits) - total_count;
    if ((*counts)[omit_pos] <= 0) {
      // The histogram we've read sums to more than total_count (including at
      // least 1 for the omitted value).
      return false;
    }
  }
  return true;
}

} // namespace

bool ANSDecodingData::ReadFromBitStream(BitReader* input) {
  std::vector<int> counts;
  return (ReadHistogram(ANS_LOG_TAB_SIZE, &counts, input) &&
          ANSBuildMapTable(counts, map_));
}

bool DecodeANSCodes(const size_t num_histograms, const size_t max_alphabet_size,
                    const uint8_t* symbol_lut, size_t symbol_lut_size,
                    BitReader* in, ANSCode* result) {
  PIK_ASSERT(max_alphabet_size <= ANS_TAB_SIZE);
  result->map.resize((num_histograms << ANS_LOG_TAB_SIZE) + 1);
  result->info.resize(num_histograms << ANS_LOG_TAB_SIZE);
  for (size_t c = 0; c < num_histograms; ++c) {
    std::vector<int> counts;
    if (!ReadHistogram(ANS_LOG_TAB_SIZE, &counts, in)) {
      return PIK_FAILURE("Invalid histogram bitstream.");
    }
    if (counts.size() > max_alphabet_size) {
      return PIK_FAILURE("Alphabet size is too long.");
    }
    const size_t histo_offset = c << ANS_LOG_TAB_SIZE;
    uint32_t offset = 0;
    for (size_t i = 0, pos = 0; i < counts.size(); ++i) {
      size_t symbol = i;
      if (symbol_lut != nullptr && symbol < symbol_lut_size) {
        symbol = symbol_lut[symbol];
      }
      const size_t symbol_idx = histo_offset + symbol;
      const uint32_t freq = counts[i];
#if PIK_BYTE_ORDER_LITTLE
      const uint32_t s32 = offset + (freq << 16);
      memcpy(&result->info[symbol_idx], &s32, sizeof(s32));
#else
      result->info[symbol_idx].offset = static_cast<uint16_t>(offset);
      result->info[symbol_idx].freq = static_cast<uint16_t>(freq);
#endif
      offset += counts[i];
      if (offset > ANS_TAB_SIZE) {
        return PIK_FAILURE("Invalid ANS histogram data.");
      }
      for (size_t j = 0; j < counts[i]; ++j, ++pos) {
        result->map[histo_offset + pos] = symbol;
      }
    }
  }
  return true;
}

}  // namespace pik
