#include "jpeg_quant_tables.h"

#include "status.h"

namespace pik {

void FillQuantMatrix(bool is_chroma, uint32_t q, uint8_t dst[64]) {
  PIK_ASSERT(q >= 0 && q < kMaxQFactor);
  const uint8_t* const in = kDefaultQuantMatrix[is_chroma];
  for (int i = 0; i < 64; ++i) {
    const uint32_t v = (in[i] * q + 32) >> 6;
    // clamp to prevent illegal quantizer values
    dst[i] = (v < 1) ? 1 : (v > 255) ? 255u : v;
  }
}

uint32_t FindBestMatrix(const int* src, bool is_chroma, uint8_t dst[64]) {
  uint32_t best_q = 0;
  uint32_t best_err = ~0;
  for (uint32_t q = 0; q < kMaxQFactor; ++q) {
    FillQuantMatrix(is_chroma, q, dst);
    uint32_t err = 0;
    for (int k = 0; k < 64; ++k) {
      err += (src[k] - dst[k]) * (src[k] - dst[k]);
      if (err >= best_err) break;
    }
    if (err < best_err) {
      best_err = err;
      best_q = q;
    }
  }
  FillQuantMatrix(is_chroma, best_q, dst);
  return best_q;
}

}  // namespace pik
