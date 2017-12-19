#include "ans_decode.h"

#include <vector>

#include "fast_log.h"
#include "histogram_decode.h"

namespace pik {

bool DecodeANSCodes(const size_t num_histograms, const size_t max_alphabet_size,
                    const uint8_t* symbol_lut, size_t symbol_lut_size,
                    BitReader* in, ANSCode* result) {
  PIK_ASSERT(max_alphabet_size <= ANS_TAB_SIZE);
  result->map.resize(num_histograms << ANS_LOG_TAB_SIZE);
  result->info.resize(num_histograms << ANS_LOG_TAB_SIZE);
  for (int c = 0; c < num_histograms; ++c) {
    std::vector<int> counts;
    if (!ReadHistogram(ANS_LOG_TAB_SIZE, &counts, in)) {
      return PIK_FAILURE("Invalid histogram bitstream.");
    }
    if (counts.size() > max_alphabet_size) {
      return PIK_FAILURE("Alphabet size is too long.");
    }
    const int histo_offset = c << ANS_LOG_TAB_SIZE;
    int offset = 0;
    for (int i = 0, pos = 0; i < counts.size(); ++i) {
      int symbol = i;
      if (symbol_lut != nullptr && symbol < symbol_lut_size) {
        symbol = symbol_lut[symbol];
      }
      const int symbol_idx = histo_offset + symbol;
      result->info[symbol_idx].offset = offset;
      result->info[symbol_idx].freq = counts[i];
      offset += counts[i];
      if (offset > ANS_TAB_SIZE) {
        return PIK_FAILURE("Invalid ANS histogram data.");
      }
      for (int j = 0; j < counts[i]; ++j, ++pos) {
        result->map[histo_offset + pos] = symbol;
      }
    }
  }
  return true;
}

}  // namespace pik
