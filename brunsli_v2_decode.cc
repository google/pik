#include "brunsli_v2_decode.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "ans_decode.h"
#include "arith_decode.h"
#include "bit_reader.h"
#include "brunsli_v2_common.h"
#include "brunsli_v2_input.h"
#include "common.h"
#include "compiler_specific.h"
#include "context.h"
#include "context_map_decode.h"
#include "dc_predictor_slow.h"
#include "distributions.h"
#include "fast_log.h"
#include "guetzli/jpeg_error.h"
#include "jpeg_quant_tables.h"
#include "lehmer_code.h"

namespace pik {

static const int kNumDirectCodes = 8;

int DecodeVarint(BrunsliV2Input* input, int max_bits) {
  int n = 0;
  for (int b = 0; b < max_bits; ++b) {
    if (b + 1 != max_bits && !input->ReadBits(1)) {
      break;
    }
    n |= input->ReadBits(1) << b;
  }
  return n;
}

bool DecodeQuantTables(BrunsliV2Input* input, guetzli::JPEGData* jpg) {
  int num_quant_tables = input->ReadBits(2) + 1;
  jpg->quant.resize(num_quant_tables);
  for (int i = 0; i < num_quant_tables; ++i) {
    guetzli::JPEGQuantTable* q = &jpg->quant[i];
    if (!input->ReadBits(1)) {
      const int short_code = input->ReadBits(3);
      for (int k = 0; k < kDCTBlockSize; ++k) {
        q->values[k] = kStockQuantizationTables[i > 0][short_code][k];
      }
    } else {
      const int qfactor = input->ReadBits(6);
      uint8_t predictor[kDCTBlockSize];
      FillQuantMatrix(i > 0, qfactor, predictor);
      int last_diff = 0;
      for (int k = 0; k < kDCTBlockSize; ++k) {
        int v = 0;
        if (input->ReadBits(1)) {
          const int sign = input->ReadBits(1);
          v = DecodeVarint(input, 16) + 1;
          if (sign) v = -v;
        }
        v += last_diff;
        last_diff = v;
        const int j = guetzli::kJPEGNaturalOrder[k];
        const int quant_value = predictor[j] + v;
        q->values[j] = quant_value;
        if (quant_value <= 0) {
          return false;
        }
      }
    }
  }
  for (int i = 0; i < jpg->components.size(); ++i) {
    guetzli::JPEGComponent* c = &jpg->components[i];
    c->quant_idx = input->ReadBits(2);
  }
  return true;
}

void DecodeFrameType(uint32_t frame_code, guetzli::JPEGData* jpg) {
  for (int i = 0; i < jpg->components.size(); ++i) {
    guetzli::JPEGComponent* c = &jpg->components[i];
    c->v_samp_factor = (frame_code & 0xf) + 1;
    frame_code >>= 4;
    c->h_samp_factor = (frame_code & 0xf) + 1;
    frame_code >>= 4;
  }
  for (int i = 0; i < jpg->components.size(); ++i) {
    guetzli::JPEGComponent* c = &jpg->components[i];
    jpg->max_h_samp_factor = std::max(jpg->max_h_samp_factor, c->h_samp_factor);
    jpg->max_v_samp_factor = std::max(jpg->max_v_samp_factor, c->v_samp_factor);
  }
  jpg->MCU_rows = DivCeil(jpg->height, jpg->max_v_samp_factor * 8);
  jpg->MCU_cols = DivCeil(jpg->width, jpg->max_h_samp_factor * 8);
  for (int i = 0; i < jpg->components.size(); ++i) {
    guetzli::JPEGComponent* c = &jpg->components[i];
    c->width_in_blocks = jpg->MCU_cols * c->h_samp_factor;
    c->height_in_blocks = jpg->MCU_rows * c->v_samp_factor;
    c->num_blocks = c->width_in_blocks * c->height_in_blocks;
  }
}

bool DecodeCoeffOrder(int* order, BrunsliV2Input* in) {
  int lehmer[kDCTBlockSize] = {0};
  static const int kSpan = 16;
  for (int i = 0; i < kDCTBlockSize; i += kSpan) {
    if (!in->ReadBits(1)) continue;  // span is all-zero
    const int start = (i > 0) ? i : 1;
    const int end = i + kSpan;
    for (int j = start; j < end; ++j) {
      int v = 0;
      while (v <= kDCTBlockSize) {
        const int bits = in->ReadBits(3);
        v += bits;
        if (bits < 7) break;
      }
      if (v > kDCTBlockSize) v = kDCTBlockSize;
      lehmer[j] = v;
    }
  }
  int end = kDCTBlockSize - 1;
  while (end > 0 && lehmer[end] == 0) {
    --end;
  }
  for (int i = 1; i <= end; ++i) {
    --lehmer[i];
  }
  DecodeLehmerCode(lehmer, kDCTBlockSize, order);
  for (int k = 0; k < kDCTBlockSize; ++k) {
    order[k] = guetzli::kJPEGNaturalOrder[order[k]];
  }
  return true;
}

int DecodeNumNonzeros(Prob* const p, BinaryArithmeticDecoder* ac,
                      BrunsliV2Input* in) {
  const int kMaxBits = 6;
  int val = 1;
  for (int b = 0; b < kMaxBits; ++b) {
    const int bit = ac->ReadBit<8>(p[val - 1].get_proba(), in);
    p[val - 1].Add(bit);
    val = 2 * val + bit;
  }
  return val - (1 << kMaxBits);
}

bool DecodeDC(int mcu_cols, int mcu_rows, int num_components,
              const int h_samp[guetzli::kMaxComponents],
              const int v_samp[guetzli::kMaxComponents],
              const std::vector<uint8_t>& context_map,
              const std::vector<ANSDecodingData>& entropy_codes,
              coeff_t* all_coeffs[guetzli::kMaxComponents],
              std::vector<bool>* const block_state, BrunsliV2Input* in) {
  std::vector<ComponentStateDC> comps(num_components);
  int total_num_blocks = 0;
  for (int i = 0; i < num_components; ++i) {
    comps[i].SetWidth(mcu_cols * h_samp[i]);
    total_num_blocks += mcu_cols * mcu_rows * h_samp[i] * v_samp[i];
  }
  block_state->resize(total_num_blocks);

  BinaryArithmeticDecoder ac;
  ANSDecoder ans;
  ans.Init(in);
  in->InitBitReader();
  ac.Init(in);

  // We decode DC components in the following interleaved manner:
  //   v_samp[0] rows from component 0
  //   v_samp[1] rows from component 1
  //   v_samp[2] rows from component 2
  //   v_samp[3] rows from component 3 (if present)
  //
  // E.g. in a YUV420 image, we decode 2 rows of DC components from Y and then
  // 1 row of DC components from U and 1 row of DC components from V.
  int block_ipos = 0;
  for (int mcu_y = 0; mcu_y < mcu_rows; ++mcu_y) {
    for (int i = 0; i < num_components; ++i) {
      ComponentStateDC* const c = &comps[i];
      const uint8_t* const cur_context_map = &context_map[i * kNumAvrgContexts];
      const int width = c->width;
      int y = mcu_y * v_samp[i];
      int block_ix = y * width;
      coeff_t* coeffs = &all_coeffs[i][block_ix * kDCTBlockSize];
      int* const prev_sgn = &c->prev_sign[1];
      int* const prev_abs = &c->prev_abs_coeff[2];
      for (int iy = 0; iy < v_samp[i]; ++iy, ++y) {
        for (int x = 0; x < width; ++x) {
          const int is_empty_ctx =
              IsEmptyBlockContext(&c->prev_is_nonempty[1], x);
          Prob* const is_empty_p = &c->is_empty_block_prob[is_empty_ctx];
          const bool is_empty_block =
              !ac.ReadBit<8>(is_empty_p->get_proba(), in);
          is_empty_p->Add(!is_empty_block);
          c->prev_is_nonempty[x + 1] = !is_empty_block;
          (*block_state)[block_ipos] = is_empty_block;
          int absval = 0;
          int sign = 0;
          if (!is_empty_block) {
            int is_zero = 0;
            Prob* const p = &c->is_zero_prob;
            is_zero = ac.ReadBit<8>(p->get_proba(), in);
            p->Add(is_zero);
            if (!is_zero) {
              const int avrg_ctx = WeightedAverageContextDC(prev_abs, x);
              const int sign_ctx = prev_sgn[x] * 3 + prev_sgn[x - 1];
              Prob* const sign_p = &c->sign_prob[sign_ctx];
              sign = ac.ReadBit<8>(sign_p->get_proba(), in);
              sign_p->Add(sign);
              const int entropy_ix = cur_context_map[avrg_ctx];
              int code = ans.ReadSymbol(entropy_codes[entropy_ix], in);
              if (code < kNumDirectCodes) {
                absval = code + 1;
              } else {
                const int nbits = code - kNumDirectCodes;
                Prob* const p = &c->first_extra_bit_prob[nbits];
                const int first_extra_bit = ac.ReadBit<8>(p->get_proba(), in);
                p->Add(first_extra_bit);
                int extra_bits_val = first_extra_bit << nbits;
                if (nbits > 0) {
                  extra_bits_val |= in->ReadBits(nbits);
                }
                absval = kNumDirectCodes - 1 + (2 << nbits) + extra_bits_val;
              }
            }
          }
          prev_abs[x] = absval;
          prev_sgn[x] = absval ? sign + 1 : 0;
          coeffs[0] = (1 - 2 * sign) * absval;
          ++block_ipos;
          ++block_ix;
          coeffs += kDCTBlockSize;
        }
      }
    }
  }
  if (!ans.CheckCRC()) return false;
  if (in->error_) return false;
  return true;
}

bool DecodeAC(const int mcu_cols, const int mcu_rows, const int num_components,
              const int h_samp[guetzli::kMaxComponents],
              const int v_samp[guetzli::kMaxComponents],
              const int all_quant[guetzli::kMaxComponents][kDCTBlockSize],
              const std::vector<int> context_bits,
              const std::vector<uint8_t>& context_map,
              const std::vector<ANSDecodingData>& entropy_codes,
              const std::vector<bool> block_state,
              coeff_t* all_coeffs[guetzli::kMaxComponents],
              BrunsliV2Input* in) {
  int num_contexts = num_components;
  std::vector<ComponentState> comps(num_components);
  for (int i = 0; i < num_components; ++i) {
    comps[i].SetWidth(mcu_cols * h_samp[i]);
    comps[i].context_offset = num_contexts * kNumAvrgContexts;
    num_contexts += kNumNonzeroContextSkip[context_bits[i]];
    ComputeACPredictMultipliers(&all_quant[i][0], &comps[i].mult_row[0],
                                &comps[i].mult_col[0]);
  }

  BinaryArithmeticDecoder ac;
  ANSDecoder ans;
  ans.Init(in);
  in->InitBitReader();
  ac.Init(in);

  for (int i = 0; i < num_components; ++i) {
    if (!DecodeCoeffOrder(&comps[i].order[0], in)) {
      return false;
    }
  }

  int block_ipos = 0;
  for (int mcu_y = 0; mcu_y < mcu_rows; ++mcu_y) {
    for (int i = 0; i < num_components; ++i) {
      ComponentState* const c = &comps[i];
      const uint8_t* const cur_context_map = &context_map[c->context_offset];
      const int cur_ctx_bits = context_bits[i];
      const int width = c->width;
      int y = mcu_y * v_samp[i];
      int block_ix = y * width;
      coeff_t* coeffs = &all_coeffs[i][block_ix * kDCTBlockSize];
      const coeff_t* prev_row_coeffs =
          &all_coeffs[i][(block_ix - width) * kDCTBlockSize];
      const coeff_t* prev_col_coeffs =
          &all_coeffs[i][(block_ix - 1) * kDCTBlockSize];
      int prev_row_delta = (1 - 2 * (y & 1)) * (width + 3) * kDCTBlockSize;
      for (int iy = 0; iy < v_samp[i]; ++iy, ++y) {
        int* prev_sgn = &c->prev_sign[kDCTBlockSize];
        int* prev_abs =
            &c->prev_abs_coeff[((y & 1) * (width + 3) + 2) * kDCTBlockSize];
        for (int x = 0; x < width; ++x) {
          const bool is_empty_block = block_state[block_ipos];
          int last_nz = 0;
          if (!is_empty_block) {
            const int nzero_ctx =
                NumNonzerosContext(&c->prev_num_nonzeros[1], x, y);
            last_nz =
                DecodeNumNonzeros(c->num_nonzero_prob[nzero_ctx], &ac, in);
          }
          for (int k = kDCTBlockSize - 1; k > last_nz; --k) {
            prev_sgn[k] = 0;
            prev_abs[k] = 0;
          }
          int num_nzeros = 0;
          for (int k = last_nz; k >= 1; --k) {
            int is_zero = 0;
            if (k < last_nz) {
              const int bucket = kNonzeroBuckets[num_nzeros - 1];
              const int is_zero_ctx = bucket * kDCTBlockSize + k;
              Prob* const p = &c->is_zero_prob[is_zero_ctx];
              is_zero = ac.ReadBit<8>(p->get_proba(), in);
              p->Add(is_zero);
            }
            int absval = 0;
            int sign = 1;
            const int k_nat = c->order[k];
            if (!is_zero) {
              absval = 1;
              int avrg_ctx = 0;
              int sign_ctx = kMaxAverageContext;
              if (k_nat < 8) {
                if (y > 0) {
                  const int ctx =
                      ACPredictContextRow(prev_row_coeffs + k_nat,
                                          coeffs + k_nat, &c->mult_col[k_nat]);
                  avrg_ctx = std::abs(ctx);
                  sign_ctx += ctx;
                }
              } else if ((k_nat & 7) == 0) {
                if (x > 0) {
                  const int ctx =
                      ACPredictContextCol(prev_col_coeffs + k_nat,
                                          coeffs + k_nat, &c->mult_row[k_nat]);
                  avrg_ctx = std::abs(ctx);
                  sign_ctx += ctx;
                }
              } else {
                avrg_ctx = WeightedAverageContext(prev_abs + k, prev_row_delta);
                sign_ctx = prev_sgn[k] * 3 + prev_sgn[k - kDCTBlockSize];
              }
              sign_ctx = sign_ctx * kDCTBlockSize + k;
              Prob* const sign_p = &c->sign_prob[sign_ctx];
              sign = ac.ReadBit<8>(sign_p->get_proba(), in);
              sign_p->Add(sign);
              prev_sgn[k] = sign + 1;
              sign = 1 - 2 * sign;
              const int zdens_ctx =
                  ZeroDensityContext(num_nzeros, k, cur_ctx_bits);
              const int histo_ix = zdens_ctx * kNumAvrgContexts + avrg_ctx;
              const int entropy_ix = cur_context_map[histo_ix];
              int code = ans.ReadSymbol(entropy_codes[entropy_ix], in);
              if (code < kNumDirectCodes) {
                absval = code + 1;
              } else {
                int nbits = code - kNumDirectCodes;
                Prob* p = &c->first_extra_bit_prob[k * 10 + nbits];
                int first_extra_bit = ac.ReadBit<8>(p->get_proba(), in);
                p->Add(first_extra_bit);
                int extra_bits_val = first_extra_bit << nbits;
                if (nbits > 0) {
                  extra_bits_val |= in->ReadBits(nbits);
                }
                absval = kNumDirectCodes - 1 + (2 << nbits) + extra_bits_val;
              }
              ++num_nzeros;
            } else {
              prev_sgn[k] = 0;
            }
            int coeff = sign * absval;
            coeffs[k_nat] = coeff;
            prev_abs[k] = absval;
          }
          c->prev_num_nonzeros[x + 1] = num_nzeros;
          ++block_ipos;
          ++block_ix;
          coeffs += kDCTBlockSize;
          prev_sgn += kDCTBlockSize;
          prev_abs += kDCTBlockSize;
          prev_row_coeffs += kDCTBlockSize;
          prev_col_coeffs += kDCTBlockSize;
        }
        prev_row_delta *= -1;
      }
    }
  }
  if (!ans.CheckCRC()) return false;
  if (in->error_) return false;
  return true;
}

struct JPEGDecodingState {
  std::vector<int> context_bits;
  std::vector<uint8_t> context_map;
  std::vector<ANSDecodingData> entropy_codes;
  std::vector<bool> block_state;
};

bool DecodeBase128(const uint8_t* data, const size_t len, size_t* pos,
                   size_t* val) {
  int shift = 0;
  uint64_t b;
  *val = 0;
  do {
    if (*pos >= len || shift > 57) {
      return false;
    }
    b = data[(*pos)++];
    *val |= (b & 0x7f) << shift;
    shift += 7;
  } while (b & 0x80);
  return true;
}

bool DecodeDataLength(const uint8_t* data, const size_t len, size_t* pos,
                      size_t* data_len) {
  if (!DecodeBase128(data, len, pos, data_len)) {
    return false;
  }
  return *data_len <= len && *pos <= len - *data_len;
}

bool DecodeHeader(const uint8_t* data, const size_t len, size_t* pos,
                  guetzli::JPEGData* jpg) {
  if (*pos >= len || data[(*pos)++] != kBrunsliHeaderMarker) {
    return PIK_FAILURE("Invalid brunsli V2 header.");
  }
  size_t marker_len = 0;
  if (!DecodeDataLength(data, len, pos, &marker_len)) {
    return PIK_FAILURE("Invalid marker length.");
  }
  size_t marker_end = *pos + marker_len;
  size_t height = 0;
  size_t width = 0;
  size_t comp_code = 0xff;   // Invalid value.
  size_t frame_code = 0xff;  // Invalid frame type value.
  while (*pos < marker_end) {
    uint8_t marker = data[(*pos)++];
    if ((marker & 0x80) != 0 || ((marker & 0x5) != 0) || marker <= 0x02) {
      return PIK_FAILURE("Invalid marker");
    }
    bool ok = false;
    switch (marker) {
      case 0x08:
        ok = (width == 0) && DecodeBase128(data, len, pos, &width);
        break;
      case 0x10:
        ok = (height == 0) && DecodeBase128(data, len, pos, &height);
        break;
      case 0x18:
        ok = (comp_code == 0xff) && DecodeBase128(data, len, pos, &comp_code);
        break;
      case 0x20:
        ok = (frame_code == 0xff) && DecodeBase128(data, len, pos, &frame_code);
        break;
      default: {
        // Skip the unknown marker.
        size_t val = 0;
        ok = DecodeBase128(data, len, pos, &val);
        if ((marker & 0x7) == 2) {
          ok = ok && val <= len && *pos <= len - val;
          *pos += val;
        }
      } break;
    }
    if (!ok) {
      return PIK_FAILURE("Invalid brunsli v2 header.");
    }
  }
  const int version = (comp_code >> 2);
  const int ncomp = (comp_code & 3) + 1;
  if ((version != 1 && (width == 0 || height == 0)) || version > 1 ||
      frame_code == 0xff) {
    return PIK_FAILURE("Invalid brunsli v2 header.");
  }
  if (*pos != marker_end) {
    return PIK_FAILURE("Invalid brunsli v2 header.");
  }
  jpg->width = width;
  jpg->height = height;
  if (version != 1) {
    jpg->components.resize(ncomp);
  }
  jpg->version = version;
  DecodeFrameType(frame_code, jpg);
  return true;
}

bool DecodeQuantDataSection(const uint8_t* data, const size_t len,
                            guetzli::JPEGData* jpg) {
  if (len == 0) {
    return false;
  }
  BrunsliV2Input input(data, len);
  input.InitBitReader();
  if (!DecodeQuantTables(&input, jpg)) {
    return false;
  }
  return !input.error_;
}

bool DecodeHistogramDataSection(const uint8_t* data, const size_t len,
                                JPEGDecodingState* s, guetzli::JPEGData* jpg) {
  if (jpg->components.empty()) {
    // Histogram data can not be decoded without knowing the number of
    // components from the header.
    return false;
  }
  if (len == 0) {
    return false;
  }
  BitReader input(data, len);
  int num_contexts = jpg->components.size();
  s->context_bits.resize(jpg->components.size());
  for (int i = 0; i < jpg->components.size(); ++i) {
    s->context_bits[i] = std::min(input.ReadBits(3), 6);
    num_contexts += kNumNonzeroContextSkip[s->context_bits[i]];
  }
  s->context_map.resize(num_contexts * kNumAvrgContexts);
  size_t num_histograms;
  if (!DecodeContextMap(&s->context_map, &num_histograms, &input)) {
    return false;
  }
  s->entropy_codes.resize(num_histograms);
  for (int i = 0; i < num_histograms; ++i) {
    if (!s->entropy_codes[i].ReadFromBitStream(&input)) {
      return false;
    }
  }
  return ((input.Position() + 3) & ~3) <= len;
}

size_t Neg(size_t x) { return -static_cast<int64_t>(x); }

void UnpredictDC(coeff_t* const PIK_RESTRICT coeffs, const size_t xsize,
                 const size_t ysize) {
  for (int y = 0; y < ysize; y++) {
    coeff_t* const PIK_RESTRICT row = coeffs + y * xsize * kDCTBlockSize;
    for (int x = 0; x < xsize; x++) {
      const coeff_t prediction = MinCostPredict<coeff_t>(
          row + x * kDCTBlockSize, x, y, xsize, -kDCTBlockSize,
          Neg(xsize * kDCTBlockSize), false, 0);
      row[x * kDCTBlockSize] += prediction;
    }
  }
}

static void UnpredictDCWithYPixel(const coeff_t* const PIK_RESTRICT row_y,
                                  coeff_t* const PIK_RESTRICT row_u,
                                  coeff_t* const PIK_RESTRICT row_v, size_t x,
                                  size_t y, size_t xsize, size_t ysize) {
  const int predictor =
      GetUVPredictor(row_y + x * kDCTBlockSize, x, y, xsize, -kDCTBlockSize,
                     Neg(xsize * kDCTBlockSize));
  const coeff_t prediction_u =
      MinCostPredict(row_u + x * kDCTBlockSize, x, y, xsize, -kDCTBlockSize,
                     Neg(xsize * kDCTBlockSize), true, predictor);
  row_u[x * kDCTBlockSize] += prediction_u;
  const coeff_t prediction_v =
      MinCostPredict(row_v + x * kDCTBlockSize, x, y, xsize, -kDCTBlockSize,
                     Neg(xsize * kDCTBlockSize), true, predictor);
  row_v[x * kDCTBlockSize] += prediction_v;
}

void UnpredictDCWithY(const coeff_t* const PIK_RESTRICT coeffs_y,
                      coeff_t* const PIK_RESTRICT coeffs_u,
                      coeff_t* const PIK_RESTRICT coeffs_v, const size_t xsize,
                      const size_t ysize) {
  if (xsize == 0) {
    return;
  }
  for (int y = 0; y < ysize; y++) {
    const coeff_t* const PIK_RESTRICT row_y =
        coeffs_y + y * xsize * kDCTBlockSize;
    coeff_t* const PIK_RESTRICT row_u = coeffs_u + y * xsize * kDCTBlockSize;
    coeff_t* const PIK_RESTRICT row_v = coeffs_v + y * xsize * kDCTBlockSize;
    UnpredictDCWithYPixel(row_y, row_u, row_v, 0, y, xsize, ysize);
    if (xsize >= 2) {
      UnpredictDCWithYPixel(row_y, row_u, row_v, 1, y, xsize, ysize);
    }
    for (int x = 2; x < xsize - 1; x++) {
      UnpredictDCWithYPixel(row_y, row_u, row_v, x, y, xsize, ysize);
    }
    if (xsize >= 3) {
      UnpredictDCWithYPixel(row_y, row_u, row_v, xsize - 1, y, xsize, ysize);
    }
  }
}

bool DecodeDCDataSection(const uint8_t* data, const size_t len,
                         JPEGDecodingState* s, guetzli::JPEGData* jpg) {
  if (jpg->width == 0 || jpg->height == 0 || jpg->MCU_rows == 0 ||
      jpg->MCU_cols == 0 || jpg->components.empty() || jpg->quant.empty() ||
      s->context_map.empty()) {
    // DC data can not be decoded without knowing the width and the height
    // and the number of components from the header, the quantization tables
    // from the quant data section and the context map from the histogram data
    // section.
    return false;
  }
  coeff_t* coeffs[guetzli::kMaxComponents] = {nullptr};
  int h_samp[guetzli::kMaxComponents] = {0};
  int v_samp[guetzli::kMaxComponents] = {0};
  for (int i = 0; i < jpg->components.size(); ++i) {
    guetzli::JPEGComponent* c = &jpg->components[i];
    coeffs[i] = &c->coeffs[0];
    h_samp[i] = c->h_samp_factor;
    v_samp[i] = c->v_samp_factor;
  }
  BrunsliV2Input in(data, len);
  if (!DecodeDC(jpg->MCU_cols, jpg->MCU_rows, jpg->components.size(), h_samp,
                v_samp, s->context_map, s->entropy_codes, coeffs,
                &s->block_state, &in)) {
    return false;
  }
  // Unpredict
  const bool use_uv_prediction = jpg->components.size() == 3 &&
                                 jpg->max_h_samp_factor == 1 &&
                                 jpg->max_v_samp_factor == 1;
  guetzli::JPEGComponent* c0 = &jpg->components[0];
  UnpredictDC(&c0->coeffs[0], c0->width_in_blocks, c0->height_in_blocks);
  if (use_uv_prediction) {
    UnpredictDCWithY(&c0->coeffs[0], &jpg->components[1].coeffs[0],
                     &jpg->components[2].coeffs[0], c0->width_in_blocks,
                     c0->height_in_blocks);
  } else {
    for (int i = 1; i < jpg->components.size(); ++i) {
      guetzli::JPEGComponent* c = &jpg->components[i];
      UnpredictDC(&c->coeffs[0], c->width_in_blocks, c->height_in_blocks);
    }
  }
  return true;
}

bool DecodeACDataSection(const uint8_t* data, const size_t len,
                         JPEGDecodingState* s, guetzli::JPEGData* jpg) {
  if (jpg->width == 0 || jpg->height == 0 || jpg->MCU_rows == 0 ||
      jpg->MCU_cols == 0 || jpg->components.empty() || jpg->quant.empty() ||
      s->block_state.empty() || s->context_map.empty()) {
    // AC data can not be decoded without knowing the width and the height
    // and the number of components from the header, the quantization tables
    // from the quant data section, the context map from the histogram data
    // section and the block states from the DC data section.
    return false;
  }
  coeff_t* coeffs[guetzli::kMaxComponents] = {nullptr};
  int quant[guetzli::kMaxComponents][kDCTBlockSize];
  int h_samp[guetzli::kMaxComponents] = {0};
  int v_samp[guetzli::kMaxComponents] = {0};
  for (int i = 0; i < jpg->components.size(); ++i) {
    guetzli::JPEGComponent* c = &jpg->components[i];
    if (c->quant_idx >= jpg->quant.size()) {
      return false;
    }
    const guetzli::JPEGQuantTable& q = jpg->quant[c->quant_idx];
    coeffs[i] = &c->coeffs[0];
    memcpy(&quant[i][0], &q.values[0], kDCTBlockSize * sizeof(quant[0][0]));
    for (int k = 0; k < kDCTBlockSize; ++k) {
      if (quant[i][k] == 0) {
        return false;
      }
    }
    h_samp[i] = c->h_samp_factor;
    v_samp[i] = c->v_samp_factor;
  }
  BrunsliV2Input in(data, len);
  if (!DecodeAC(jpg->MCU_cols, jpg->MCU_rows, jpg->components.size(), h_samp,
                v_samp, quant, s->context_bits, s->context_map,
                s->entropy_codes, s->block_state, coeffs, &in)) {
    return false;
  }
  return true;
}

bool BrunsliV2DecodeJpegData(const uint8_t* data, const size_t len,
                             guetzli::JPEGData* jpg) {
  size_t pos = 0;

  if (!DecodeHeader(data, len, &pos, jpg)) return false;

  for (int i = 0; i < jpg->components.size(); ++i) {
    guetzli::JPEGComponent* c = &jpg->components[i];
    c->coeffs.resize(static_cast<size_t>(c->num_blocks) * kDCTBlockSize);
  }
  std::set<int> markers_seen;
  JPEGDecodingState s;
  bool have_quant_section = false;
  bool have_dc_section = false;
  while (pos < len) {
    uint8_t marker = data[pos++];
    // There are 15 valid marker bytes for compatibility with the protocol
    // buffer wire format (field_numbers 1 to 15, wire type 2).
    if ((marker & 0x80) != 0 || (marker & 0x7) != 2 || marker == 0x02) {
      return PIK_FAILURE("Invalid brunsli v2 marker.");
    }
    size_t marker_len = 0;
    if (!DecodeDataLength(data, len, &pos, &marker_len)) {
      return PIK_FAILURE("Invalid brunsli v2 marker length.");
    }
    if (markers_seen.insert(marker).second == false) {
      printf("Duplicate marker %d\n", marker);
      return PIK_FAILURE("Duplicate brunsli v2 marker.");
    }
    bool ok = false;
    switch (marker) {
      case kBrunsliHeaderMarker:
        // Header can appear only at the start of the data.
        ok = false;
        break;
      case kBrunsliQuantDataMarker:
        ok = DecodeQuantDataSection(&data[pos], marker_len, jpg);
        have_quant_section = true;
        break;
      case kBrunsliHistogramDataMarker:
        ok = DecodeHistogramDataSection(&data[pos], marker_len, &s, jpg);
        break;
      case kBrunsliDCDataMarker:
        ok = have_quant_section &&
             DecodeDCDataSection(&data[pos], marker_len, &s, jpg);
        have_dc_section = ok;
        break;
      case kBrunsliACDataMarker:
        ok = have_dc_section &&
             DecodeACDataSection(&data[pos], marker_len, &s, jpg);
        break;
      default:
        // We skip unrecognized marker segments.
        ok = true;
        break;
    }
    if (!ok) {
      return PIK_FAILURE("Invalid brunsli v2 data.");
    }
    pos += marker_len;
  }
  if (pos != len) {
    return PIK_FAILURE("Brunsli v2 length mismatch.");
  }
  return true;
}

}  // namespace pik
