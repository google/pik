#include "brunsli_v2_encode.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ans_encode.h"
#include "ans_params.h"
#include "brunsli_v2_common.h"
#include "cluster.h"
#include "context.h"
#include "context_map_encode.h"
#include "data_stream.h"
#include "dc_predictor_slow.h"
#include "distributions.h"
#include "entropy_source.h"
#include "fast_log.h"
#include "guetzli/jpeg_data.h"
#include "guetzli/jpeg_error.h"
#include "jpeg_quant_tables.h"
#include "lehmer_code.h"
#include "status.h"
#include "write_bits.h"

namespace pik {

static const int kNumDirectCodes = 8;

size_t Base128Size(size_t val) {
  size_t size = 1;
  for (; val >= 128; val >>= 7) ++size;
  return size;
}

void EncodeBase128(size_t val, uint8_t* data, size_t* pos) {
  do {
    data[(*pos)++] = (val & 0x7f) | (val >= 128 ? 0x80 : 0);
    val >>= 7;
  } while (val > 0);
}

void EncodeBase128Fix(size_t val, size_t len, uint8_t* data) {
  for (int i = 0; i < len; ++i) {
    *data++ = (val & 0x7f) | (i + 1 < len ? 0x80 : 0);
    val >>= 7;
  }
}

int GetQuantTableId(const guetzli::JPEGQuantTable& q, bool is_chroma,
                    uint8_t dst[kDCTBlockSize]) {
  for (int j = 0; j < kNumStockQuantTables; ++j) {
    bool match_found = true;
    for (int k = 0; match_found && k < kDCTBlockSize; ++k) {
      if (q.values[k] != kStockQuantizationTables[is_chroma][j][k]) {
        match_found = false;
      }
    }
    if (match_found) {
      return j;
    }
  }
  return kNumStockQuantTables + FindBestMatrix(&q.values[0], is_chroma, dst);
}

void EncodeVarint(int n, int max_bits, size_t* storage_ix, uint8_t* storage) {
  int b;
  PIK_ASSERT(n < (1 << max_bits));
  for (b = 0; n != 0 && b < max_bits; ++b) {
    if (b + 1 != max_bits) {
      WriteBits(1, 1, storage_ix, storage);
    }
    WriteBits(1, n & 1, storage_ix, storage);
    n >>= 1;
  }
  if (b < max_bits) {
    WriteBits(1, 0, storage_ix, storage);
  }
}

bool EncodeQuantTables(const guetzli::JPEGData& jpg, size_t* storage_ix,
                       uint8_t* storage) {
  if (jpg.quant.empty() || jpg.quant.size() > 4) {
    // If ReadJpeg() succeeded with JPEG_READ_ALL mode, this should not happen.
    return false;
  }
  WriteBits(2, jpg.quant.size() - 1, storage_ix, storage);
  for (int i = 0; i < jpg.quant.size(); ++i) {
    const guetzli::JPEGQuantTable& q = jpg.quant[i];
    uint8_t predictor[kDCTBlockSize];
    const int code = GetQuantTableId(q, i > 0, predictor);
    WriteBits(1, (code >= kNumStockQuantTables),
              storage_ix, storage);
    if (code < kNumStockQuantTables) {
      WriteBits(3, code, storage_ix, storage);
    } else {
      PIK_ASSERT(code - kNumStockQuantTables < (1 << 6));
      WriteBits(6, code - kNumStockQuantTables, storage_ix, storage);
      int last_diff = 0;  // difference predictor
      for (int k = 0; k < kDCTBlockSize; ++k) {
        const int j = guetzli::kJPEGNaturalOrder[k];
        if (q.values[j] == 0) {
          // Note: ReadJpeg() checks this case and discards such jpeg files.
          return false;
        }
        const int new_diff = q.values[j] - predictor[j];
        int diff = new_diff  - last_diff;
        last_diff = new_diff;
        WriteBits(1, diff != 0, storage_ix, storage);
        if (diff) {
          WriteBits(1, diff < 0, storage_ix, storage);
          if (diff < 0) diff = -diff;
          diff -= 1;
          EncodeVarint(diff, 16, storage_ix, storage);
        }
      }
    }
  }
  for (int i = 0; i < jpg.components.size(); ++i) {
    WriteBits(2, jpg.components[i].quant_idx, storage_ix, storage);
  }
  return true;
}

int SelectContextBits(size_t num_symbols) {
  static const int kContextBits[33] = {
    0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 1,
    1, 2, 2, 3,
    3, 3, 3, 4,
    4, 4, 4, 5,
    5, 5, 6, 6,
    6, 6, 6, 6,
  };
  int log2_size = Log2FloorNonZero(num_symbols);
  return kContextBits[log2_size];
}

void ComputeCoeffOrder(const coeff_t* coeffs, const int num_blocks, int* order,
                       size_t* approx_total_nonzeros) {
  // For faster compression we only go over a sample of the blocks here. We skip
  // this many blocks after each sampled one.
  static const int kSkipBlocks = 4;
  static const int kSkipCoeffs = kSkipBlocks * kDCTBlockSize;
  int num_zeros[kDCTBlockSize] = { 0 };
  if (num_blocks >= 1024) {
    for (int i = 0; i < num_blocks; ++i) {
      for (int k = 0; k < kDCTBlockSize; ++k) {
        if (*coeffs++ == 0) ++num_zeros[k];
      }
      i += kSkipBlocks;
      coeffs += kSkipCoeffs;
    }
  }
  size_t total_zeros = num_zeros[0];
  num_zeros[0] = 0;  // DC coefficient is always the first one.
  std::vector<std::pair<int, int> > pos_and_val(kDCTBlockSize);
  for (int i = 0; i < kDCTBlockSize; ++i) {
    pos_and_val[i].first = i;
    pos_and_val[i].second = num_zeros[guetzli::kJPEGNaturalOrder[i]];
    total_zeros += num_zeros[i];
  }
  std::stable_sort(
      pos_and_val.begin(), pos_and_val.end(),
      [](const std::pair<int, int>& a, const std::pair<int, int>& b) -> bool {
        return a.second < b.second; });
  for (int i = 0; i < kDCTBlockSize; ++i) {
    order[i] = guetzli::kJPEGNaturalOrder[pos_and_val[i].first];
  }
  *approx_total_nonzeros += (kDCTBlockSize * (num_blocks + kSkipBlocks) -
                             ((kSkipBlocks + 1) * total_zeros));
}

struct JPEGCodingState {
  JPEGCodingState() : entropy_source(kNumAvrgContexts),
                      data_stream_dc(kNumAvrgContexts),
                      data_stream_ac(kNumAvrgContexts) {}
  EntropySource entropy_source;
  DataStream data_stream_dc;
  DataStream data_stream_ac;
  std::vector<bool> block_state;
  std::vector<int> context_bits;
};

void EncodeNumNonzeros(int val, Prob* p, DataStream* data_stream) {
  int ctx = 1;
  const int kMaxBits = 6;
  for (int mask = 1 << (kMaxBits - 1); mask != 0; mask >>= 1) {
    const int bit = !!(val & mask);
    data_stream->AddBit(&p[ctx - 1], bit);
    ctx = 2 * ctx + bit;
  }
}

// Or'ing of all coeffs [1..63], for quick zero-test:
coeff_t CollectAllCoeffs(const coeff_t coeffs[kDCTBlockSize]) {
  coeff_t all_coeffs = 0;
  for (int k = 1; all_coeffs == 0 && k < kDCTBlockSize; ++k) {
    all_coeffs |= coeffs[k];
  }
  return all_coeffs;
}

void EncodeDC(const int mcu_cols,
              const int mcu_rows,
              const int num_components,
              const int h_samp[guetzli::kMaxComponents],
              const int v_samp[guetzli::kMaxComponents],
              const coeff_t* all_dc_pred_errors[guetzli::kMaxComponents],
              const coeff_t* all_ac_coeffs[guetzli::kMaxComponents],
              std::vector<bool>* block_state,
              EntropySource* entropy_source,
              DataStream* data_stream) {
  std::vector<ComponentStateDC> comps(num_components);
  int total_num_blocks = 0;
  for (int i = 0; i < num_components; ++i) {
    comps[i].SetWidth(mcu_cols * h_samp[i]);
    total_num_blocks += mcu_cols * mcu_rows * h_samp[i] * v_samp[i];
  }
  entropy_source->Resize(num_components);
  data_stream->Resize(3 * total_num_blocks + 128);
  block_state->resize(total_num_blocks);

  // We encode image components in the following interleaved manner:
  //   v_samp[0] rows of 8x8 blocks from component 0
  //   v_samp[1] rows of 8x8 blocks from component 1
  //   v_samp[2] rows of 8x8 blocks from component 2
  //   v_samp[3] rows of 8x8 blocks from component 3 (if present)
  //
  // E.g. in a YUV420 image, we encode 2 rows of 8x8 blocks from Y and then
  // 1 row of 8x8 blocks from U and 1 row of 8x8 blocks from V.
  //
  // In the terminology of the JPEG standard, we encode one row of MCUs at a
  // time, but within this MCU row, we encode the components non-interleaved.
  int block_ipos = 0;
  for (int mcu_y = 0; mcu_y < mcu_rows; ++mcu_y) {
    for (int i = 0; i < num_components; ++i) {
      ComponentStateDC* c = &comps[i];
      const int width = c->width;
      int y = mcu_y * v_samp[i];
      int block_ix = y * width;
      int* prev_sgn = &c->prev_sign[1];
      int* prev_abs = &c->prev_abs_coeff[2];
      const coeff_t* dc_coeffs_in =  &all_dc_pred_errors[i][block_ix];
      const coeff_t* ac_coeffs_in = &all_ac_coeffs[i][block_ix * kDCTBlockSize];
      for (int iy = 0; iy < v_samp[i]; ++iy, ++y) {
        for (int x = 0; x < width; ++x) {
          data_stream->ResizeForBlock();
          const coeff_t coeff = dc_coeffs_in[0];
          const int sign = (coeff > 0) ? 1 : (coeff < 0) ? 2 : 0;
          const int absval = (sign == 2) ? -coeff : coeff;
          const coeff_t all_coeffs = coeff | CollectAllCoeffs(ac_coeffs_in);
          const bool is_empty_block = (all_coeffs == 0);
          const int is_empty_ctx =
              IsEmptyBlockContext(&c->prev_is_nonempty[1], x);
          Prob* const is_empty_p = &c->is_empty_block_prob[is_empty_ctx];
          data_stream->AddBit(is_empty_p, !is_empty_block);
          c->prev_is_nonempty[x + 1] = !is_empty_block;
          (*block_state)[block_ipos] = is_empty_block;
          if (!is_empty_block) {
            const int is_zero = (coeff == 0);
            Prob* const p = &c->is_zero_prob;
            data_stream->AddBit(p, is_zero);
            if (!is_zero) {
              const int avrg_ctx = WeightedAverageContextDC(prev_abs, x);
              const int sign_ctx = prev_sgn[x] * 3 + prev_sgn[x - 1];
              Prob* const sign_p = &c->sign_prob[sign_ctx];
              data_stream->AddBit(sign_p, sign - 1);
              const int zdens_ctx = i;
              if (absval <= kNumDirectCodes) {
                data_stream->AddCode(absval - 1, zdens_ctx, avrg_ctx,
                                     entropy_source);
              } else {
                int nbits = Log2FloorNonZero(absval - kNumDirectCodes + 1) - 1;
                data_stream->AddCode(kNumDirectCodes + nbits,
                                     zdens_ctx, avrg_ctx, entropy_source);
                int extra_bits = absval - (kNumDirectCodes - 1 + (2 << nbits));
                int first_extra_bit = (extra_bits >> nbits) & 1;
                Prob* p = &c->first_extra_bit_prob[nbits];
                data_stream->AddBit(p, first_extra_bit);
                if (nbits > 0) {
                  extra_bits &= (1 << nbits) - 1;
                  data_stream->AddBits(nbits, extra_bits);
                }
              }
            }
          }
          prev_sgn[x] = sign;
          prev_abs[x] = absval;
          ++block_ix;
          ++block_ipos;
          ++dc_coeffs_in;
          ac_coeffs_in += kDCTBlockSize;
        }
      }
    }
  }
}

void EncodeCoeffOrder(const int* order, DataStream* data_stream) {
  int order_zigzag[kDCTBlockSize];
  for (int i = 0; i < kDCTBlockSize; ++i) {
    order_zigzag[i] = guetzli::kJPEGZigZagOrder[order[i]];
  }
  int lehmer[kDCTBlockSize];
  ComputeLehmerCode(order_zigzag, kDCTBlockSize, lehmer);
  int end = kDCTBlockSize - 1;
  while (end >= 1 && lehmer[end] == 0) {
    --end;
  }
  for (int i = 1; i <= end; ++i) {
    ++lehmer[i];
  }
  static const int kSpan = 16;
  for (int i = 0; i < kDCTBlockSize; i += kSpan) {
    const int start = (i > 0) ? i : 1;
    const int end = i + kSpan;
    int has_non_zero = 0;
    for (int j = start; j < end; ++j) has_non_zero |= lehmer[j];
    if (!has_non_zero) {   // all zero in the span -> escape
      data_stream->AddBits(1, 0);
      continue;
    } else {
      data_stream->AddBits(1, 1);
    }
    for (int j = start; j < end; ++j) {
      int v;
      PIK_ASSERT(lehmer[j] <= kDCTBlockSize);
      for (v = lehmer[j]; v >= 7; v -= 7) {
        data_stream->AddBits(3, 7);
      }
      data_stream->AddBits(3, v);
    }
  }
}

void EncodeAC(const int mcu_cols,
              const int mcu_rows,
              const int num_components,
              const int h_samp[guetzli::kMaxComponents],
              const int v_samp[guetzli::kMaxComponents],
              const coeff_t* all_coeffs_in[guetzli::kMaxComponents],
              const int all_quant[guetzli::kMaxComponents][kDCTBlockSize],
              const std::vector<bool>& block_state,
              std::vector<int>* context_bits,
              EntropySource* entropy_source,
              DataStream* data_stream) {
  int num_code_words = 0;
  int num_contexts = num_components;
  int total_num_blocks = 0;
  std::vector<ComponentState> comps(num_components);
  std::vector<int> context_offsets(1 + num_components);
  context_bits->resize(num_components);
  for (int i = 0; i < num_components; ++i) {
    const int num_blocks = mcu_cols * mcu_rows * h_samp[i] * v_samp[i];
    size_t approx_total_nonzeros = 0;
    ComputeCoeffOrder(all_coeffs_in[i], num_blocks, &comps[i].order[0],
                      &approx_total_nonzeros);

    (*context_bits)[i] = SelectContextBits(approx_total_nonzeros + 1);
    comps[i].context_offset = num_contexts;
    context_offsets[i + 1] = num_contexts;
    num_contexts += kNumNonzeroContextSkip[(*context_bits)[i]];

    ComputeACPredictMultipliers(&all_quant[i][0],
                                &comps[i].mult_row[0],
                                &comps[i].mult_col[0]);

    comps[i].SetWidth(mcu_cols * h_samp[i]);
    num_code_words += 2 * approx_total_nonzeros + 1024 + 3 * num_blocks;
    total_num_blocks += num_blocks;
  }

  entropy_source->Resize(num_contexts);
  data_stream->Resize(num_code_words);

  for (int i = 0; i < num_components; ++i) {
    EncodeCoeffOrder(&comps[i].order[0], data_stream);
  }

  // We encode image components in the following interleaved manner:
  //   v_samp[0] rows of 8x8 blocks from component 0
  //   v_samp[1] rows of 8x8 blocks from component 1
  //   v_samp[2] rows of 8x8 blocks from component 2
  //   v_samp[3] rows of 8x8 blocks from component 3 (if present)
  //
  // E.g. in a YUV420 image, we encode 2 rows of 8x8 blocks from Y and then
  // 1 row of 8x8 blocks from U and 1 row of 8x8 blocks from V.
  //
  // In the terminology of the JPEG standard, we encode one row of MCUs at a
  // time, but within this MCU row, we encode the components non-interleaved.
  int block_ipos = 0;
  for (int mcu_y = 0; mcu_y < mcu_rows; ++mcu_y) {
    for (int i = 0; i < num_components; ++i) {
      ComponentState* const c = &comps[i];
      const int cur_ctx_bits = (*context_bits)[i];
      const int* cur_order = c->order;
      const int width = c->width;
      int y = mcu_y * v_samp[i];
      int block_ix = y * width;
      const coeff_t* coeffs_in = &all_coeffs_in[i][block_ix * kDCTBlockSize];
      const coeff_t* prev_row_coeffs =
          &all_coeffs_in[i][(block_ix - width) * kDCTBlockSize];
      const coeff_t* prev_col_coeffs =
          &all_coeffs_in[i][(block_ix - 1) * kDCTBlockSize];
      int prev_row_delta =
          (1 - 2 * (y & 1)) * (width + 3) * kDCTBlockSize;
      for (int iy = 0; iy < v_samp[i]; ++iy, ++y) {
        int* prev_sgn = &c->prev_sign[kDCTBlockSize];
        int* prev_abs =
            &c->prev_abs_coeff[((y & 1) * (width + 3) + 2) * kDCTBlockSize];
        for (int x = 0; x < width; ++x) {
          data_stream->ResizeForBlock();
          coeff_t coeffs[kDCTBlockSize] = { 0 };
          int last_nz = 0;
          const bool is_empty_block = block_state[block_ipos];
          if (!is_empty_block) {
            for (int k = 1; k < kDCTBlockSize; ++k) {
              const int k_nat = cur_order[k];
              coeffs[k] = coeffs_in[k_nat];
              if (coeffs[k]) last_nz = k;
            }
            const int nzero_context =
                NumNonzerosContext(&c->prev_num_nonzeros[1], x, y);
            EncodeNumNonzeros(last_nz, c->num_nonzero_prob[nzero_context],
                              data_stream);
          }
          for (int k = kDCTBlockSize - 1; k > last_nz; --k) {
            prev_sgn[k] = 0;
            prev_abs[k] = 0;
          }
          int num_nzeros = 0;
          coeff_t encoded_coeffs[kDCTBlockSize] = { 0 };
          for (int k = last_nz; k >= 1; --k) {
            coeff_t coeff = coeffs[k];
            const int is_zero = (coeff == 0);
            if (k < last_nz) {
              const int bucket = kNonzeroBuckets[num_nzeros - 1];
              const int is_zero_ctx = bucket * kDCTBlockSize + k;
              Prob* const p = &c->is_zero_prob[is_zero_ctx];
              data_stream->AddBit(p, is_zero);
            }
            if (!is_zero) {
              const int sign = (coeff > 0 ? 0 : 1);
              const int absval = sign ? -coeff : coeff;

              const int k_nat = cur_order[k];
              int avrg_ctx = 0;
              int sign_ctx = kMaxAverageContext;
              if (k_nat < 8) {
                if (y > 0) {
                  const int ctx = ACPredictContextRow(prev_row_coeffs + k_nat,
                                                      encoded_coeffs + k_nat,
                                                      &c->mult_col[k_nat]);
                  avrg_ctx = std::abs(ctx);
                  sign_ctx += ctx;
                }
              } else if ((k_nat & 7) == 0) {
                if (x > 0) {
                  const int ctx = ACPredictContextCol(prev_col_coeffs + k_nat,
                                                      encoded_coeffs + k_nat,
                                                      &c->mult_row[k_nat]);
                  avrg_ctx = std::abs(ctx);
                  sign_ctx += ctx;
                }
              } else {
                avrg_ctx = WeightedAverageContext(prev_abs + k,
                                                  prev_row_delta);
                sign_ctx = prev_sgn[k] * 3 + prev_sgn[k - kDCTBlockSize];
              }
              sign_ctx = sign_ctx * kDCTBlockSize + k;
              Prob* const sign_p = &c->sign_prob[sign_ctx];
              data_stream->AddBit(sign_p, sign);
              prev_sgn[k] = sign + 1;
              const int zdens_ctx = c->context_offset +
                  ZeroDensityContext(num_nzeros, k, cur_ctx_bits);
              if (absval <= kNumDirectCodes) {
                data_stream->AddCode(absval - 1, zdens_ctx, avrg_ctx,
                                     entropy_source);
              } else {
                const int base_code = absval - kNumDirectCodes + 1;
                const int nbits = Log2FloorNonZero(base_code) - 1;
                data_stream->AddCode(kNumDirectCodes + nbits,
                                     zdens_ctx, avrg_ctx, entropy_source);
                const int extra_bits = base_code - (2 << nbits);
                const int first_extra_bit = (extra_bits >> nbits) & 1;
                Prob* const p = &c->first_extra_bit_prob[k * 10 + nbits];
                data_stream->AddBit(p, first_extra_bit);
                if (nbits > 0) {
                  const int left_over_bits = extra_bits & ((1 << nbits) - 1);
                  data_stream->AddBits(nbits, left_over_bits);
                }
              }
              ++num_nzeros;
              encoded_coeffs[k_nat] = coeff;
              prev_abs[k] = absval;
            } else {
              prev_sgn[k] = 0;
              prev_abs[k] = 0;
            }
          }
          c->prev_num_nonzeros[x + 1] = num_nzeros;
          ++block_ix;
          ++block_ipos;
          coeffs_in += kDCTBlockSize;
          prev_sgn += kDCTBlockSize;
          prev_abs += kDCTBlockSize;
          prev_row_coeffs += kDCTBlockSize;
          prev_col_coeffs += kDCTBlockSize;
        }
        prev_row_delta *= -1;
      }
    }
  }
  entropy_source->ClusterHistograms(context_offsets);
}

bool ProcessCoefficients(const guetzli::JPEGData& jpg,
                         JPEGCodingState* s) {
  const coeff_t* dc_pred_errors[guetzli::kMaxComponents] = { nullptr };
  const coeff_t* ac_coeffs[guetzli::kMaxComponents] = { nullptr };
  int quant[guetzli::kMaxComponents][kDCTBlockSize];
  int h_samp[guetzli::kMaxComponents] = { 0 };
  int v_samp[guetzli::kMaxComponents] = { 0 };
  std::vector<std::vector<coeff_t> > dc_errors(jpg.components.size());

  // The maximum absolute value brunsli can encode is 2054 (8 values for direct
  // codes and num bits from 1 to 10, so a total of 8 + 2 + 4 + ... + 1024).
  static const int kBrunsliMaxDCAbsVal = 2054;

  const bool use_uv_prediction =
      jpg.components.size() == 3 && jpg.max_h_samp_factor == 1 &&
      jpg.max_v_samp_factor == 1;

  for (int i = 0; i < jpg.components.size(); ++i) {
    const guetzli::JPEGComponent& c = jpg.components[i];
    const guetzli::JPEGComponent& c0 = jpg.components[0];
    if (c.quant_idx >= jpg.quant.size()) {
      return false;
    }
    const guetzli::JPEGQuantTable& q = jpg.quant[c.quant_idx];
    const int width = c.width_in_blocks;
    const int height = c.height_in_blocks;
    const bool use_uv_predictor = i > 0 && use_uv_prediction;
    const coeff_t* coeffs = &c.coeffs[0];
    const coeff_t* coeffs_y = &c0.coeffs[0];
    dc_errors[i].resize(width * height);
    coeff_t* pred_errors = &dc_errors[i][0];
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        int predictor = 0;
        if (use_uv_predictor) {
          predictor = GetUVPredictor(coeffs_y, x, y, width, -kDCTBlockSize,
                                     -width * kDCTBlockSize);
        }
        const coeff_t prediction =
            MinCostPredict(coeffs, x, y, width, -kDCTBlockSize,
                           -width * kDCTBlockSize, use_uv_predictor, predictor);
        int err = coeffs[0] - prediction;
        if (std::abs(err) > kBrunsliMaxDCAbsVal) {
          return false;
        }
        coeffs += kDCTBlockSize;
        coeffs_y += kDCTBlockSize;
        *pred_errors++ = err;
      }
    }

    dc_pred_errors[i] = &dc_errors[i][0];
    ac_coeffs[i] = &c.coeffs[0];
    memcpy(&quant[i][0], &q.values[0], kDCTBlockSize * sizeof(quant[0][0]));
    h_samp[i] = c.h_samp_factor;
    v_samp[i] = c.v_samp_factor;
  }

  EncodeDC(jpg.MCU_cols, jpg.MCU_rows, jpg.components.size(), h_samp, v_samp,
           dc_pred_errors, ac_coeffs, &s->block_state, &s->entropy_source,
           &s->data_stream_dc);

  EncodeAC(jpg.MCU_cols, jpg.MCU_rows, jpg.components.size(), h_samp, v_samp,
           ac_coeffs, quant, s->block_state, &s->context_bits,
           &s->entropy_source, &s->data_stream_ac);

  return true;
}

uint32_t FrameTypeCode(const guetzli::JPEGData& jpg) {
  uint32_t code = 0;
  int shift = 0;
  for (int i = 0; i < jpg.components.size() && i < 4; ++i) {
    uint32_t h_samp = jpg.components[i].h_samp_factor - 1;
    uint32_t v_samp = jpg.components[i].v_samp_factor - 1;
    code |= (h_samp << (shift + 4)) | (v_samp << shift);
    shift += 8;
  }
  return code;
}

Status EncodeHeader(const guetzli::JPEGData& jpg, JPEGCodingState* s,
                    uint8_t* data, size_t* len) {
  if ((jpg.version != 1 && (jpg.width == 0 || jpg.height == 0)) ||
      jpg.components.empty() ||
      jpg.components.size() > guetzli::kMaxComponents) {
    return false;
  }
  int version = jpg.version;
  size_t pos = 0;
  data[pos++] = 0x08;
  EncodeBase128(jpg.width, data, &pos);
  data[pos++] = 0x10;
  EncodeBase128(jpg.height, data, &pos);
  data[pos++] = 0x18;
  EncodeBase128((jpg.components.size() - 1) | (version << 2), data, &pos);
  data[pos++] = 0x20;
  EncodeBase128(FrameTypeCode(jpg), data, &pos);
  *len = pos;
  return true;
}

Status EncodeQuantData(const guetzli::JPEGData& jpg, JPEGCodingState* s,
                       uint8_t* data, size_t* len) {
  // Initialize storage.
  size_t storage_ix = 0;
  data[0] = 0;
  PIK_RETURN_IF_ERROR(EncodeQuantTables(jpg, &storage_ix, data));
  *len = (storage_ix + 7) >> 3;
  // Brunsli V2 quant data section must contain an even number of bytes.
  // TODO(user): Remove this restriction by making BrunsliV2Input work
  // with odd lengths.
  if (*len & 1) {
    data[*len] = 0;
    ++(*len);
  }
  return true;
}

Status EncodeHistogramData(const guetzli::JPEGData& jpg, JPEGCodingState* s,
                           uint8_t* data, size_t* len) {
  // Initialize storage.
  size_t storage_ix = 0;
  data[0] = 0;
  for (int i = 0; i < s->context_bits.size(); ++i) {
    WriteBits(3, s->context_bits[i], &storage_ix, data);
  }
  s->entropy_source.EncodeContextMap(&storage_ix, data);
  s->entropy_source.BuildAndStoreEntropyCodes(&storage_ix, data);
  *len = (storage_ix + 7) >> 3;
  // Brunsli V2 histogram data section must be padded to 4 bytes alignment.
  const size_t misalign = *len & 3;
  if (misalign != 0) {
    for (size_t i = 0; i < 4 - misalign; ++i) {
      data[*len] = 0;
      ++(*len);
    }
  }
  return true;
}

Status EncodeDCData(const guetzli::JPEGData& jpg, JPEGCodingState* s,
                    uint8_t* data, size_t* len) {
  // Initialize storage.
  size_t storage_ix = 0;
  data[0] = 0;
  s->data_stream_dc.EncodeCodeWords(&s->entropy_source, &storage_ix, data);
  *len = (storage_ix + 7) >> 3;
  return true;
}

Status EncodeACData(const guetzli::JPEGData& jpg, JPEGCodingState* s,
                    uint8_t* data, size_t* len) {
  // Initialize storage.
  size_t storage_ix = 0;
  data[0] = 0;
  s->data_stream_ac.EncodeCodeWords(&s->entropy_source, &storage_ix, data);
  *len = (storage_ix + 7) >> 3;
  return true;
}

typedef Status (*EncodeSectionDataFn)(const guetzli::JPEGData& jpg,
                                      JPEGCodingState* s, uint8_t* data,
                                      size_t* len);

Status EncodeSection(const guetzli::JPEGData& jpg, JPEGCodingState* s,
                     uint8_t marker, EncodeSectionDataFn write_section,
                     size_t section_size_bytes, size_t len, uint8_t* data,
                     size_t* pos) {
  // Write the marker byte for the section.
  const size_t pos_start = *pos;
  data[(*pos)++] = marker;

  // Skip enough bytes for a valid (though not necessarily optimal) base-128
  // encoding of the size of the section.
  *pos += section_size_bytes;

  size_t section_size = len - *pos;
  PIK_RETURN_IF_ERROR(write_section(jpg, s, &data[*pos], &section_size));
  *pos += section_size;

  if ((section_size >> (7 * section_size_bytes)) > 0) {
    printf("Section 0x%x size %zu too large for %zu bytes base128 number.",
           marker, section_size, section_size_bytes);
    return false;
  }

  // Write the final size of the section after the marker byte.
  EncodeBase128Fix(section_size, section_size_bytes, &data[pos_start + 1]);
  return true;
}

size_t BrunsliV2MaximumEncodedSize(const guetzli::JPEGData& jpg) {
  // Rough estimate is 1.2 * uncompressed size plus some more for the header.
  size_t hdr_size = 1 << 20;
  for (const std::string& data : jpg.app_data) {
    hdr_size += data.size();
  }
  for (const std::string& data : jpg.com_data) {
    hdr_size += data.size();
  }
  hdr_size += jpg.tail_data.size();
  return 1.2 * jpg.width * jpg.height * jpg.components.size() + hdr_size;
}

bool BrunsliV2EncodeJpegData(const guetzli::JPEGData& jpg,
                             const size_t header_size, PaddedBytes* out) {
  size_t pos = header_size;
  const size_t len = BrunsliV2MaximumEncodedSize(jpg);
  uint8_t* data = out->data();
  if (!EncodeSection(jpg, nullptr, kBrunsliHeaderMarker, EncodeHeader,
                     1, len, data, &pos)) {
    return PIK_FAILURE("Encode header");
  }
  if (!EncodeSection(jpg, nullptr, kBrunsliQuantDataMarker, EncodeQuantData,
                     2, len, data, &pos)) {
    return PIK_FAILURE("Encode quant");
  }
  JPEGCodingState s;
  if (!ProcessCoefficients(jpg, &s)) {
    return PIK_FAILURE("ProcessCoefficients");
  }
  if (!EncodeSection(jpg, &s, kBrunsliHistogramDataMarker, EncodeHistogramData,
                     Base128Size(len - pos), len, data, &pos)) {
    return PIK_FAILURE("Histogram");
  }
  if (!EncodeSection(jpg, &s, kBrunsliDCDataMarker, EncodeDCData,
                     Base128Size(len - pos), len, data, &pos)) {
    return PIK_FAILURE("DC");
  }
  if (!EncodeSection(jpg, &s, kBrunsliACDataMarker, EncodeACData,
                     Base128Size(len - pos), len, data, &pos)) {
    return PIK_FAILURE("AC");
  }
  out->resize(pos);
  return true;
}

}  // namespace pik
