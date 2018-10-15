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

#include "entropy_coder.h"

#include <stdint.h>
#include <string.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include "ans_params.h"
#include "bit_reader.h"
#include "common.h"
#include "compiler_specific.h"
#include "context_map_decode.h"
#include "dc_predictor.h"
#include "dc_predictor_slow.h"
#include "fast_log.h"
#include "status.h"
#include "write_bits.h"

namespace pik {

// Reorder the skip+bits symbols by decreasing population-count
// (keeping the first end-of-block symbol in place).
// Round-trip:
//  skip_and_bits = (SKIP << 4) | BITS
//  symbol = kSkipAndBitsSymbol[skip_and_bits]
//  SKIP = kSkipLut[symbol]
//  BITS = kBitsLut[symbol]
constexpr uint8_t kSkipAndBitsSymbol[256] = {
    0,   1,   2,   3,   5,   10,  17,  32,  68,  83,  84,  85,  86,  87,  88,
    89,  90,  4,   7,   12,  22,  31,  43,  60,  91,  92,  93,  94,  95,  96,
    97,  98,  99,  6,   14,  26,  36,  48,  66,  100, 101, 102, 103, 104, 105,
    106, 107, 108, 109, 8,   19,  34,  44,  57,  78,  110, 111, 112, 113, 114,
    115, 116, 117, 118, 119, 9,   27,  39,  52,  61,  79,  120, 121, 122, 123,
    124, 125, 126, 127, 128, 129, 11,  28,  41,  53,  64,  80,  130, 131, 132,
    133, 134, 135, 136, 137, 138, 139, 13,  33,  46,  63,  72,  140, 141, 142,
    143, 144, 145, 146, 147, 148, 149, 150, 15,  35,  47,  65,  69,  151, 152,
    153, 154, 155, 156, 157, 158, 159, 160, 161, 16,  37,  51,  62,  74,  162,
    163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 18,  38,  50,  59,  75,
    173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 20,  40,  54,  76,
    82,  184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 23,  42,  55,
    77,  195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 24,  45,
    56,  70,  207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 25,
    49,  58,  71,  219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230,
    29,  67,  81,  231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242,
    21,  30,  73,  243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254,
    255,
};

constexpr uint8_t kSkipLut[256] = {
    0x0, 0x0, 0x0, 0x0, 0x1, 0x0, 0x2, 0x1, 0x3, 0x4, 0x0, 0x5, 0x1, 0x6, 0x2,
    0x7, 0x8, 0x0, 0x9, 0x3, 0xa, 0xf, 0x1, 0xb, 0xc, 0xd, 0x2, 0x4, 0x5, 0xe,
    0xf, 0x1, 0x0, 0x6, 0x3, 0x7, 0x2, 0x8, 0x9, 0x4, 0xa, 0x5, 0xb, 0x1, 0x3,
    0xc, 0x6, 0x7, 0x2, 0xd, 0x9, 0x8, 0x4, 0x5, 0xa, 0xb, 0xc, 0x3, 0xd, 0x9,
    0x1, 0x4, 0x8, 0x6, 0x5, 0x7, 0x2, 0xe, 0x0, 0x7, 0xc, 0xd, 0x6, 0xf, 0x8,
    0x9, 0xa, 0xb, 0x3, 0x4, 0x5, 0xe, 0xa, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
    0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x2, 0x2, 0x2, 0x2, 0x2, 0x2,
    0x2, 0x2, 0x2, 0x2, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x4,
    0x4, 0x4, 0x4, 0x4, 0x4, 0x4, 0x4, 0x4, 0x4, 0x5, 0x5, 0x5, 0x5, 0x5, 0x5,
    0x5, 0x5, 0x5, 0x5, 0x6, 0x6, 0x6, 0x6, 0x6, 0x6, 0x6, 0x6, 0x6, 0x6, 0x6,
    0x7, 0x7, 0x7, 0x7, 0x7, 0x7, 0x7, 0x7, 0x7, 0x7, 0x7, 0x8, 0x8, 0x8, 0x8,
    0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x9, 0x9, 0x9, 0x9, 0x9, 0x9, 0x9, 0x9,
    0x9, 0x9, 0x9, 0xa, 0xa, 0xa, 0xa, 0xa, 0xa, 0xa, 0xa, 0xa, 0xa, 0xa, 0xb,
    0xb, 0xb, 0xb, 0xb, 0xb, 0xb, 0xb, 0xb, 0xb, 0xb, 0xb, 0xc, 0xc, 0xc, 0xc,
    0xc, 0xc, 0xc, 0xc, 0xc, 0xc, 0xc, 0xc, 0xd, 0xd, 0xd, 0xd, 0xd, 0xd, 0xd,
    0xd, 0xd, 0xd, 0xd, 0xd, 0xe, 0xe, 0xe, 0xe, 0xe, 0xe, 0xe, 0xe, 0xe, 0xe,
    0xe, 0xe, 0xe, 0xf, 0xf, 0xf, 0xf, 0xf, 0xf, 0xf, 0xf, 0xf, 0xf, 0xf, 0xf,
    0xf,
};

constexpr uint8_t kBitsLut[256] = {
    0x0, 0x1, 0x2, 0x3, 0x1, 0x4, 0x1, 0x2, 0x1, 0x1, 0x5, 0x1, 0x3, 0x1, 0x2,
    0x1, 0x1, 0x6, 0x1, 0x2, 0x1, 0x0, 0x4, 0x1, 0x1, 0x1, 0x3, 0x2, 0x2, 0x1,
    0x1, 0x5, 0x7, 0x2, 0x3, 0x2, 0x4, 0x2, 0x2, 0x3, 0x2, 0x3, 0x2, 0x6, 0x4,
    0x2, 0x3, 0x3, 0x5, 0x2, 0x3, 0x3, 0x4, 0x4, 0x3, 0x3, 0x3, 0x5, 0x3, 0x4,
    0x7, 0x5, 0x4, 0x4, 0x5, 0x4, 0x6, 0x2, 0x8, 0x5, 0x4, 0x4, 0x5, 0x2, 0x5,
    0x5, 0x4, 0x4, 0x6, 0x6, 0x6, 0x3, 0x5, 0x9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf,
    0x0, 0x8, 0x9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf, 0x0, 0x7, 0x8, 0x9, 0xa, 0xb,
    0xc, 0xd, 0xe, 0xf, 0x0, 0x7, 0x8, 0x9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf, 0x0,
    0x7, 0x8, 0x9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf, 0x0, 0x7, 0x8, 0x9, 0xa, 0xb,
    0xc, 0xd, 0xe, 0xf, 0x0, 0x6, 0x7, 0x8, 0x9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf,
    0x0, 0x6, 0x7, 0x8, 0x9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf, 0x0, 0x6, 0x7, 0x8,
    0x9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf, 0x0, 0x6, 0x7, 0x8, 0x9, 0xa, 0xb, 0xc,
    0xd, 0xe, 0xf, 0x0, 0x6, 0x7, 0x8, 0x9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf, 0x0,
    0x5, 0x6, 0x7, 0x8, 0x9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf, 0x0, 0x5, 0x6, 0x7,
    0x8, 0x9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf, 0x0, 0x5, 0x6, 0x7, 0x8, 0x9, 0xa,
    0xb, 0xc, 0xd, 0xe, 0xf, 0x0, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xa, 0xb, 0xc,
    0xd, 0xe, 0xf, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xa, 0xb, 0xc, 0xd, 0xe,
    0xf,
};

// REQUIRED: 0 <= skip, bits <= 15
constexpr uint8_t SkipAndBitsSymbol(int skip, int bits) {
  return kSkipAndBitsSymbol[(skip << 4) | bits];
}

// Size of batch of Lehmer-transformed order of coefficients.
// If all codes in the batch are zero, then span is encoded with a single bit.
constexpr int32_t kCoeffOrderCodeSpan = 16;

void ShrinkDC(const Rect& rect_dc, const Image3S& dc,
              Image3S* PIK_RESTRICT tmp_residuals) {
  const size_t xsize = rect_dc.xsize();
  const size_t ysize = rect_dc.ysize();
  PIK_ASSERT(tmp_residuals->xsize() >= xsize);
  PIK_ASSERT(tmp_residuals->ysize() >= ysize);
  const Rect tmp_rect(0, 0, xsize, ysize);

  ShrinkY(rect_dc, dc.Plane(1), tmp_rect, tmp_residuals->MutablePlane(1));

  ImageS tmp_xz(xsize * 2, ysize);

  // Interleave X and Z into XZ for ShrinkXB.
  for (size_t y = 0; y < ysize; ++y) {
    const int16_t* PIK_RESTRICT row_x = rect_dc.ConstRow(dc.Plane(0), y);
    const int16_t* PIK_RESTRICT row_z = rect_dc.ConstRow(dc.Plane(2), y);
    int16_t* PIK_RESTRICT row_xz = tmp_xz.Row(y);

    for (size_t x = 0; x < xsize; ++x) {
      row_xz[2 * x + 0] = row_x[x];
      row_xz[2 * x + 1] = row_z[x];
    }
  }

  ImageS tmp_xz_residuals(xsize * 2, ysize);
  ShrinkXB(rect_dc, dc.Plane(1), tmp_xz, &tmp_xz_residuals);

  // Deinterleave XZ into residuals X and Z.
  for (size_t y = 0; y < ysize; ++y) {
    const int16_t* PIK_RESTRICT row_xz = tmp_xz_residuals.ConstRow(y);
    int16_t* PIK_RESTRICT row_out_x = tmp_residuals->PlaneRow(0, y);
    int16_t* PIK_RESTRICT row_out_z = tmp_residuals->PlaneRow(2, y);

    for (size_t x = 0; x < xsize; ++x) {
      row_out_x[x] = row_xz[2 * x + 0];
      row_out_z[x] = row_xz[2 * x + 1];
    }
  }
}

void ExpandDC(const Rect& rect_dc, Image3S* PIK_RESTRICT dc,
              ImageS* PIK_RESTRICT tmp_y, ImageS* PIK_RESTRICT tmp_xz_residuals,
              ImageS* PIK_RESTRICT tmp_xz_expanded) {
  const size_t xsize = rect_dc.xsize();
  const size_t ysize = rect_dc.ysize();
  PIK_ASSERT(xsize <= tmp_y->xsize() && ysize <= tmp_y->ysize());
  PIK_ASSERT(SameSize(*tmp_xz_residuals, *tmp_xz_expanded));

  ExpandY(rect_dc, dc->Plane(1), tmp_y);

  // The predictor expects a single image with interleaved X and Z.
  for (size_t y = 0; y < ysize; ++y) {
    const int16_t* PIK_RESTRICT row0 = rect_dc.ConstRow(dc->Plane(0), y);
    const int16_t* PIK_RESTRICT row2 = rect_dc.ConstRow(dc->Plane(2), y);
    int16_t* PIK_RESTRICT row_xz = tmp_xz_residuals->Row(y);

    for (size_t x = 0; x < xsize; ++x) {
      row_xz[2 * x + 0] = row0[x];
      row_xz[2 * x + 1] = row2[x];
    }
  }

  ExpandXB(xsize, ysize, *tmp_y, *tmp_xz_residuals, tmp_xz_expanded);

  if (rect_dc.x0() == 0 && rect_dc.y0() == 0 && SameSize(*dc, *tmp_y)) {
    // Avoid copying Y; tmp_y remains valid for next call.
    dc->MutablePlane(1)->Swap(*tmp_y);
  } else {
    for (size_t y = 0; y < ysize; ++y) {
      const int16_t* PIK_RESTRICT row_from = tmp_y->ConstRow(y);
      int16_t* PIK_RESTRICT row_to = rect_dc.Row(dc->MutablePlane(1), y);
      memcpy(row_to, row_from, xsize * sizeof(row_to[0]));
    }
  }

  // Deinterleave |tmp_xz_expanded| and copy into |dc|.
  for (size_t y = 0; y < ysize; ++y) {
    const int16_t* PIK_RESTRICT row_xz = tmp_xz_expanded->ConstRow(y);
    int16_t* PIK_RESTRICT row_out0 = rect_dc.Row(dc->MutablePlane(0), y);
    int16_t* PIK_RESTRICT row_out2 = rect_dc.Row(dc->MutablePlane(2), y);

    for (size_t x = 0; x < xsize; ++x) {
      row_out0[x] = row_xz[2 * x + 0];
      row_out2[x] = row_xz[2 * x + 1];
    }
  }
}

template <int N>
void ComputeCoeffOrderFast(int32_t* PIK_RESTRICT order) {
  constexpr int block_size = N * N;
  for (size_t i = 0; i < kOrderContexts; ++i) {
    memcpy(&order[i * block_size], NaturalCoeffOrder<N>(),
           block_size * sizeof(order[0]));
  }
}
template void ComputeCoeffOrderFast<8>(int32_t* PIK_RESTRICT order);
template void ComputeCoeffOrderFast<16>(int32_t* PIK_RESTRICT order);

template <int N>
void ComputeCoeffOrder(const Image3S& ac, const Image3B& block_ctx,
                       int32_t* PIK_RESTRICT order) {
  constexpr int block_size = N * N;
  size_t xsize_blocks = ac.xsize() / block_size;
  size_t ysize_blocks = ac.ysize();
  const int32_t* natural_coeff_order = NaturalCoeffOrder<N>();

  // Count number of zero coefficients, separately for each DCT band.
  int32_t num_zeros[block_size * kOrderContexts] = {0};
  for (int c = 0; c < 3; ++c) {
    for (size_t by = 0; by < ysize_blocks; ++by) {
      const int16_t* PIK_RESTRICT row = ac.PlaneRow(c, by);
      const uint8_t* PIK_RESTRICT row_ctx = block_ctx.PlaneRow(c, by);
      for (size_t bx = 0; bx < xsize_blocks; ++bx) {
        size_t x = bx * block_size;
        size_t offset = row_ctx[bx] * block_size;
        size_t start = 1;
        if (row_ctx[bx] >= kIdentityOrderContextStart) start = 0;
        for (size_t k = start; k < block_size; ++k) {
          if (row[x + k] == 0) ++num_zeros[offset + k];
        }
      }
    }
  }

  for (uint8_t ctx = 0; ctx < kOrderContexts; ++ctx) {
    struct PosAndCount {
      uint32_t pos;
      uint32_t count;
    };

    // Apply zig-zag order.
    PosAndCount pos_and_val[block_size];
    size_t offset = ctx * block_size;
    for (size_t i = 0; i < block_size; ++i) {
      size_t pos = natural_coeff_order[i];
      pos_and_val[i].pos = pos;
      // We don't care for the exact number -> quantize number of zeros,
      // to get less permuted order.
      pos_and_val[i].count = num_zeros[offset + pos] / 8;
    }

    // Stable-sort -> elements with same number of zeros will preserve their
    // order.
    auto comparator = [](const PosAndCount& a, const PosAndCount& b) -> bool {
      return a.count < b.count;
    };
    std::stable_sort(pos_and_val, pos_and_val + block_size, comparator);

    // Grab indices.
    for (size_t i = 0; i < block_size; ++i) {
      order[ctx * block_size + i] = pos_and_val[i].pos;
    }
  }
}
template void ComputeCoeffOrder<8>(const Image3S&, const Image3B&,
                                   int32_t* PIK_RESTRICT);
template void ComputeCoeffOrder<16>(const Image3S&, const Image3B&,
                                    int32_t* PIK_RESTRICT);

template <int N>
void EncodeCoeffOrder(const int32_t* PIK_RESTRICT order, bool encode_first,
                      size_t* PIK_RESTRICT storage_ix, uint8_t* storage) {
  constexpr int block_size = N * N;
  int32_t order_zigzag[block_size];
  const int32_t* natural_coeff_order_lut = NaturalCoeffOrderLut<N>();
  for (size_t i = 0; i < block_size; ++i) {
    order_zigzag[i] = natural_coeff_order_lut[order[i]];
  }
  int32_t lehmer[block_size];
  ComputeLehmerCode(order_zigzag, block_size, lehmer);
  int32_t end = block_size - 1;
  while (end >= 1 && lehmer[end] == 0) {
    --end;
  }
  for (int32_t i = 1; i <= end; ++i) {
    ++lehmer[i];
  }
  for (int32_t i = 0; i < block_size; i += kCoeffOrderCodeSpan) {
    const int32_t start = (i > 0) ? i : (encode_first ? 0 : 1);
    const int32_t end = i + kCoeffOrderCodeSpan;
    int32_t has_non_zero = 0;
    for (int32_t j = start; j < end; ++j) has_non_zero |= lehmer[j];
    if (!has_non_zero) {  // all zero in the span -> escape
      WriteBits(1, 0, storage_ix, storage);
      continue;
    } else {
      WriteBits(1, 1, storage_ix, storage);
    }
    for (int32_t j = start; j < end; ++j) {
      int32_t v;
      PIK_ASSERT(lehmer[j] <= block_size);
      for (v = lehmer[j]; v >= 7; v -= 7) {
        WriteBits(3, 7, storage_ix, storage);
      }
      WriteBits(3, v, storage_ix, storage);
    }
  }
}

// Fills "tmp_num_nzeros" with per-block count of non-zero coefficients in
// "coeffs" within "rect".
template <int N>
void ExtractNumNZeroes(const Rect& rect, const ImageS& coeffs,
                       const ImageB& ac_strategy,
                       ImageI* PIK_RESTRICT tmp_num_nzeros) {
  constexpr int block_size = N * N;
  PIK_CHECK(coeffs.xsize() % block_size == 0);
  const size_t xsize_blocks = rect.xsize() / block_size;
  const size_t ysize_blocks = rect.ysize();
  for (size_t by = 0; by < ysize_blocks; ++by) {
    const int16_t* PIK_RESTRICT coeffs_row = rect.ConstRow(coeffs, by);
    int32_t* PIK_RESTRICT output_row = tmp_num_nzeros->Row(by);
    const uint8_t* ac_strategy_row =
        ac_strategy.ConstRow(rect.y0() + by) + rect.x0() / block_size;
    for (size_t bx = 0; bx < xsize_blocks; ++bx) {
      size_t num_nzeros = 0;
      size_t start_value = 1;
      if (!AcStrategyType::IsDct(ac_strategy_row[bx])) start_value = 0;
      for (size_t i = start_value; i < block_size; ++i) {
        num_nzeros += (coeffs_row[bx * block_size + i] != 0);
      }
      output_row[bx] = static_cast<int32_t>(num_nzeros);
    }
  }
}

template <int N>
std::string EncodeCoeffOrders(const int32_t* order, PikInfo* pik_info) {
  constexpr int block_size = N * N;
  std::string encoded_coeff_order(kOrderContexts * 1024, 0);
  uint8_t* storage = reinterpret_cast<uint8_t*>(&encoded_coeff_order[0]);
  size_t storage_ix = 0;
  for (size_t c = 0; c < kOrderContexts; c++) {
    EncodeCoeffOrder<N>(&order[c * block_size],
                        /*encode_first=*/c >= kIdentityOrderContextStart,
                        &storage_ix, storage);
  }
  PIK_CHECK(storage_ix < encoded_coeff_order.size() * kBitsPerByte);
  encoded_coeff_order.resize((storage_ix + 7) / kBitsPerByte);
  if (pik_info) {
    pik_info->layers[kLayerOrder].total_size += encoded_coeff_order.size();
  }
  return encoded_coeff_order;
}
template std::string EncodeCoeffOrders<8>(const int32_t*, PikInfo*);

// Number of clusters is encoded with VarLenUint8 - see EncodeContextMap and
// DecodeContextMap.
constexpr size_t kMaxClusters = 256;
// Currently we limit number of clusters even more - for most datasets "optimal"
// number of clusters is less than 64, so increasing this number does not
// produce smaller output.
// TODO(user): find image that would require more clusters.
// TODO(user): revise this number when non-DCT-8x8 contexts are added / used.
static const size_t kClustersLimit = 64;
static const size_t kNumStaticZdensContexts = 7;
// Should depend on N.
static const size_t kNumStaticContexts = 12 + 4 * kNumStaticZdensContexts;

template <int N>
std::vector<uint8_t> StaticContextMap() {
  constexpr int block_size = N * N;
  static const int32_t kStaticZdensContextMap[kZeroDensityContextCount] = {
      0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 1,
      1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 1, 1, 2, 2, 2, 2,
      2, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 5, 2, 2, 2, 2, 2, 3, 3, 3, 3,
      3, 6, 6, 6, 6, 5, 5, 2, 2, 2, 2, 2, 3, 3, 3, 3, 6, 6, 6, 6, 5,
      5, 2, 2, 2, 2, 2, 3, 3, 3, 6, 6, 6, 6, 5, 5, 2, 2, 2, 2, 2,
  };
  static const uint8_t kStaticQuantContextMap[kQuantFieldContexts] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 3, 4, 4, 4, 4, 5, 5,
      5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
      6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
      7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
      7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
      7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
  };
  PIK_ASSERT(kNumStaticContexts <= kMaxClusters);
  std::vector<uint8_t> context_map(kNumContexts);
  memcpy(&context_map[0], kStaticQuantContextMap,
         sizeof(kStaticQuantContextMap));  // [0..7]
  for (size_t c = 0; c < kOrderContexts; ++c) {
    const size_t ctx = std::min<size_t>(c, 3);
    for (size_t i = 0; i < block_size - 1; ++i) {
      context_map[NonZeroContext<N>(i, c)] = 8 + ctx;  // [8..11]
    }
    uint32_t zero_density_context_base = ZeroDensityContextsOffset<N>(c);
    for (size_t i = 0; i < kZeroDensityContextCount; ++i) {
      context_map[zero_density_context_base + i] =
          12 + ctx * kNumStaticZdensContexts + kStaticZdensContextMap[i];
    }
  }
  return context_map;
}

PIK_INLINE int32_t PredictFromTopAndLeft(
    const int32_t* const PIK_RESTRICT row_top,
    const int32_t* const PIK_RESTRICT row, size_t x, int32_t default_val) {
  if (x == 0) {
    return row_top == nullptr ? default_val : row_top[x];
  }
  if (row_top == nullptr) {
    return row[x - 1];
  }
  return (row_top[x] + row[x - 1] + 1) / 2;
}

void TokenizeQuantField(const Rect& rect, const ImageI& quant_field,
                        const ImageB& ac_strategy,
                        std::vector<Token>* PIK_RESTRICT output) {
  const size_t xsize = rect.xsize();
  const size_t ysize = rect.ysize();

  // Fixed quant_field with filled gaps.
  std::vector<int32_t> current(xsize);
  std::vector<int32_t> last(xsize);

  output->reserve(output->size() + xsize * ysize);

  // Compute actual quant values from prediction residuals.
  for (size_t by = 0; by < ysize; ++by) {
    const int32_t* PIK_RESTRICT row_src = rect.ConstRow(quant_field, by);
    int32_t* PIK_RESTRICT row_fixed = current.data();
    const int32_t* PIK_RESTRICT row_last = (by == 0) ? nullptr : last.data();
    for (size_t bx = 0; bx < xsize; ++bx) {
      int32_t quant = row_src[bx];
      int32_t predicted_quant =
          PredictFromTopAndLeft(row_last, row_fixed, bx, 32);
      row_fixed[bx] = quant;
      output->emplace_back(QuantContext(predicted_quant), quant - 1, 0, 0);
    }
    last.swap(current);
  }
}

template <int N>
void TokenizeCoefficients(const int32_t* orders, const Rect& rect,
                          const Image3S& coeffs, const Image3B& block_ctx,
                          const ImageB& ac_strategy,
                          std::vector<Token>* PIK_RESTRICT output) {
  constexpr int block_size = N * N;
  const size_t xsize_blocks = rect.xsize();
  const size_t ysize_blocks = rect.ysize();
  // Transform block coordinates to coefficient layout coordinates.
  Rect literal_rect_ac(rect.x0() * block_size, rect.y0(),
                       xsize_blocks * block_size, ysize_blocks);

  // TODO(user): update the estimate: usually less coefficients are used.
  output->reserve(output->size() +
                  3 * xsize_blocks * ysize_blocks * block_size);

  ImageI tmp_num_nzeros(xsize_blocks, ysize_blocks);
  for (int c = 0; c < 3; ++c) {
    ExtractNumNZeroes<N>(literal_rect_ac, coeffs.Plane(c), ac_strategy,
                         &tmp_num_nzeros);
    for (size_t by = 0; by < ysize_blocks; ++by) {
      const int16_t* PIK_RESTRICT row =
          literal_rect_ac.ConstPlaneRow(coeffs, c, by);
      const uint8_t* PIK_RESTRICT ctx_row =
          rect.ConstRow(block_ctx.Plane(c), by);
      int32_t* PIK_RESTRICT row_nzeros = tmp_num_nzeros.Row(by);
      const int32_t* PIK_RESTRICT row_nzeros_top =
          (by == 0) ? nullptr : tmp_num_nzeros.ConstRow(by - 1);
      const uint8_t* PIK_RESTRICT ac_strategy_row =
          rect.ConstRow(ac_strategy, by);
      for (size_t bx = 0; bx < xsize_blocks; ++bx) {
        int32_t predicted_nzeros =
            PredictFromTopAndLeft(row_nzeros_top, row_nzeros, bx, 32);
        const int bctx = ctx_row[bx];
        const int32_t* order = &orders[bctx * block_size];
        int32_t nzero_ctx = NonZeroContext<N>(predicted_nzeros, bctx);
        size_t num_nzeros = row_nzeros[bx];
        output->emplace_back(nzero_ctx, num_nzeros, 0, 0);
        if (num_nzeros == 0) continue;
        const int16_t* PIK_RESTRICT block = row + bx * block_size;
        bool is_dct = AcStrategyType::IsDct(ac_strategy_row[bx]);
        int r = 0;
        const int histo_offset = ZeroDensityContextsOffset<N>(bctx);
        size_t last_k = 0;
        for (size_t k = is_dct; k < block_size; ++k) {
          PIK_ASSERT(num_nzeros > 0);
          int16_t coeff = block[order[k]];
          if (coeff == 0) {
            if (++r == 16) {
              output->emplace_back(
                  histo_offset + ZeroDensityContext(num_nzeros, last_k),
                  SkipAndBitsSymbol(15, 0), 0, 0);
              // Skip 15, encode 0-bit coefficient -> 16 zeros in total.
              r = 0;
              last_k = k + !is_dct;
            }
            continue;
          }
          int nbits, bits;
          EncodeVarLenInt(coeff, &nbits, &bits);
          PIK_ASSERT(nbits > 0);
          PIK_ASSERT(nbits <= 15);
          int symbol = SkipAndBitsSymbol(r, nbits);
          output->emplace_back(
              histo_offset + ZeroDensityContext(num_nzeros, last_k), symbol,
              nbits, bits);
          r = 0;
          last_k = k + !is_dct;
          if (--num_nzeros == 0) break;
        }
        PIK_ASSERT(num_nzeros == 0);
      }
    }
  }
}
template void TokenizeCoefficients<8>(const int32_t*, const Rect&,
                                      const Image3S&, const Image3B&,
                                      const ImageB&,
                                      std::vector<Token>* PIK_RESTRICT);

namespace {

inline double CrossEntropy(const uint32_t* counts, const size_t counts_len,
                           const uint32_t* codes, const size_t codes_len) {
  double sum = 0.0f;
  uint32_t total_count = 0;
  uint32_t total_codes = 0;
  for (size_t i = 0; i < codes_len; ++i) {
    if (codes[i] > 0) {
      if (i < counts_len && counts[i] > 0) {
        sum -= counts[i] * std::log2(codes[i]);
        total_count += counts[i];
      }
      total_codes += codes[i];
    }
  }
  if (total_codes > 0) {
    sum += total_count * std::log2(total_codes);
  }
  return sum;
}

inline double ShannonEntropy(const uint32_t* data, const size_t data_size) {
  return CrossEntropy(data, data_size, data, data_size);
}

class HistogramBuilder {
 public:
  explicit HistogramBuilder(const size_t num_contexts)
      : histograms_(num_contexts) {}

  void VisitSymbol(int symbol, int histo_idx) {
    PIK_ASSERT(histo_idx < histograms_.size());
    histograms_[histo_idx].Add(symbol);
  }

  template <class EntropyEncodingData>
  void BuildAndStoreEntropyCodes(std::vector<EntropyEncodingData>* codes,
                                 std::vector<uint8_t>* context_map,
                                 size_t* storage_ix, uint8_t* storage,
                                 PikImageSizeInfo* info) const {
    std::vector<Histogram> clustered_histograms(histograms_);
    context_map->resize(histograms_.size());
    if (histograms_.size() > 1) {
      std::vector<uint32_t> histogram_symbols;
      ClusterHistograms(histograms_, histograms_.size(), 1, std::vector<int>(),
                        kClustersLimit, &clustered_histograms,
                        &histogram_symbols);
      for (size_t c = 0; c < histograms_.size(); ++c) {
        (*context_map)[c] = static_cast<uint8_t>(histogram_symbols[c]);
      }
      if (storage_ix != nullptr && storage != nullptr) {
        EncodeContextMap(*context_map, clustered_histograms.size(), storage_ix,
                         storage);
      }
    }
    if (info) {
      for (size_t i = 0; i < clustered_histograms.size(); ++i) {
        info->clustered_entropy += clustered_histograms[i].ShannonEntropy();
      }
    }
    for (size_t c = 0; c < clustered_histograms.size(); ++c) {
      EntropyEncodingData code;
      code.BuildAndStore(clustered_histograms[c].data_.data(),
                         clustered_histograms[c].data_.size(), storage_ix,
                         storage);
      codes->emplace_back(std::move(code));
    }
  }

 private:
  struct Histogram {
    Histogram() {
      data_.reserve(256);
      total_count_ = 0;
    }
    void Clear() {
      memset(data_.data(), 0, data_.size() * sizeof(data_[0]));
      total_count_ = 0;
    }
    void Add(int symbol) {
      if (symbol >= data_.size()) {
        data_.resize(symbol + 1);
      }
      ++data_[symbol];
      ++total_count_;
    }
    void AddHistogram(const Histogram& other) {
      if (other.data_.size() > data_.size()) {
        data_.resize(other.data_.size());
      }
      for (size_t i = 0; i < other.data_.size(); ++i) {
        data_[i] += other.data_[i];
      }
      total_count_ += other.total_count_;
    }
    float PopulationCost() const {
      std::vector<int> counts(data_.size());
      for (size_t i = 0; i < data_.size(); ++i) {
        counts[i] = data_[i];
      }
      return ANSPopulationCost(counts.data(), counts.size(), total_count_);
    }
    double ShannonEntropy() const {
      return pik::ShannonEntropy(data_.data(), data_.size());
    }

    std::vector<uint32_t> data_;
    uint32_t total_count_;
  };
  std::vector<Histogram> histograms_;
};

}  // namespace

std::string BuildAndEncodeHistograms(
    size_t num_contexts, const std::vector<std::vector<Token> >& tokens,
    std::vector<ANSEncodingData>* codes, std::vector<uint8_t>* context_map,
    PikImageSizeInfo* info) {
  // Build histograms.
  HistogramBuilder builder(num_contexts);
  for (size_t i = 0; i < tokens.size(); ++i) {
    for (size_t j = 0; j < tokens[i].size(); ++j) {
      const Token token = tokens[i][j];
      builder.VisitSymbol(token.symbol, token.context);
    }
  }
  // Encode histograms.
  const size_t max_out_size = 1024 * (num_contexts + 4);
  std::string output(max_out_size, 0);
  size_t storage_ix = 0;
  uint8_t* storage = reinterpret_cast<uint8_t*>(&output[0]);
  storage[0] = 0;
  builder.BuildAndStoreEntropyCodes(codes, context_map, &storage_ix, storage,
                                    info);
  // Close the histogram bit stream.
  size_t jump_bits = ((storage_ix + 7) & ~7) - storage_ix;
  WriteBits(jump_bits, 0, &storage_ix, storage);
  PIK_ASSERT(storage_ix % kBitsPerByte == 0);
  const size_t histo_bytes = storage_ix >> 3;
  PIK_CHECK(histo_bytes <= max_out_size);
  output.resize(histo_bytes);
  if (info) {
    info->num_clustered_histograms += codes->size();
    info->histogram_size += histo_bytes;
    info->total_size += histo_bytes;
  }
  return output;
}

template <int N>
std::string BuildAndEncodeHistogramsFast(
    const std::vector<std::vector<Token> >& tokens,
    std::vector<ANSEncodingData>* codes, std::vector<uint8_t>* context_map,
    PikImageSizeInfo* info) {
  *context_map = StaticContextMap<N>();
  // Build histograms from tokens.
  std::vector<uint32_t> histograms(kNumStaticContexts << 8);
  for (size_t i = 0; i < tokens.size(); ++i) {
    for (size_t j = 0; j < tokens[i].size(); ++j) {
      const Token token = tokens[i][j];
      const uint32_t histo_idx = (*context_map)[token.context];
      ++histograms[(histo_idx << 8) + token.symbol];
    }
  }
  if (info) {
    for (size_t c = 0; c < kNumStaticContexts; ++c) {
      info->clustered_entropy += ShannonEntropy(&histograms[c << 8], 256);
    }
  }
  const size_t max_out_size = kNumStaticContexts * 1024;
  std::string output(max_out_size, 0);
  size_t storage_ix = 0;
  uint8_t* storage = reinterpret_cast<uint8_t*>(&output[0]);
  storage[0] = 0;
  // Encode the histograms.
  EncodeContextMap(*context_map, kNumStaticContexts, &storage_ix, storage);
  for (size_t c = 0; c < kNumStaticContexts; ++c) {
    ANSEncodingData code;
    code.BuildAndStore(&histograms[c << 8], 256, &storage_ix, storage);
    codes->emplace_back(std::move(code));
  }
  // Close the histogram bit stream.
  WriteZeroesToByteBoundary(&storage_ix, storage);
  const size_t histo_bytes = (storage_ix >> 3);
  PIK_CHECK(histo_bytes <= max_out_size);
  output.resize(histo_bytes);
  if (info) {
    info->num_clustered_histograms += codes->size();
    info->histogram_size += histo_bytes;
  }
  return output;
}
template std::string BuildAndEncodeHistogramsFast<8>(
    const std::vector<std::vector<Token> >&, std::vector<ANSEncodingData>*,
    std::vector<uint8_t>*, PikImageSizeInfo*);

std::string WriteTokens(const std::vector<Token>& tokens,
                        const std::vector<ANSEncodingData>& codes,
                        const std::vector<uint8_t>& context_map,
                        PikImageSizeInfo* pik_info) {
  const size_t max_out_size = 4 * tokens.size() + 4096;
  std::string output(max_out_size, 0);
  size_t storage_ix = 0;
  uint8_t* storage = reinterpret_cast<uint8_t*>(&output[0]);
  storage[0] = 0;
  size_t num_extra_bits = 0;
  PIK_ASSERT(kANSBufferSize <= (1 << 16));
  for (int start = 0; start < tokens.size(); start += kANSBufferSize) {
    std::vector<uint32_t> out;
    out.reserve(kANSBufferSize);
    const int end = std::min<int>(start + kANSBufferSize, tokens.size());
    ANSCoder ans;
    for (int i = end - 1; i >= start; --i) {
      const Token token = tokens[i];
      const uint8_t histo_idx = context_map[token.context];
      const ANSEncSymbolInfo info = codes[histo_idx].ans_table[token.symbol];
      uint8_t nbits = 0;
      uint32_t bits = ans.PutSymbol(info, &nbits);
      if (nbits == 16) {
        out.push_back(((i - start) << 16) | bits);
      }
    }
    const uint32_t state = ans.GetState();
    WriteBits(16, (state >> 16) & 0xffff, &storage_ix, storage);
    WriteBits(16, state & 0xffff, &storage_ix, storage);
    int tokenidx = start;
    for (int i = out.size(); i >= 0; --i) {
      int nextidx = i > 0 ? start + (out[i - 1] >> 16) : end;
      for (; tokenidx < nextidx; ++tokenidx) {
        const Token token = tokens[tokenidx];
        WriteBits(token.nbits, token.bits, &storage_ix, storage);
        num_extra_bits += token.nbits;
      }
      if (i > 0) {
        WriteBits(16, out[i - 1] & 0xffff, &storage_ix, storage);
      }
    }
  }
  const size_t out_size = (storage_ix + 7) >> 3;
  PIK_CHECK(out_size <= max_out_size);
  output.resize(out_size);
  if (pik_info) {
    pik_info->entropy_coded_bits += storage_ix - num_extra_bits;
    pik_info->extra_bits += num_extra_bits;
    pik_info->total_size += out_size;
  }
  return output;
}

std::string EncodeImageData(const Rect& rect, const Image3S& img,
                            PikImageSizeInfo* info) {
  const size_t xsize = rect.xsize();
  const size_t ysize = rect.ysize();

  std::vector<std::vector<Token> > tokens(1);
  tokens[0].reserve(3 * ysize * xsize);
  for (int c = 0; c < 3; ++c) {
    for (size_t y = 0; y < ysize; ++y) {
      const int16_t* const PIK_RESTRICT row = rect.ConstRow(img.Plane(c), y);
      for (size_t x = 0; x < xsize; ++x) {
        int nbits, bits;
        EncodeVarLenInt(row[x], &nbits, &bits);
        tokens[0].emplace_back(Token(c, nbits, nbits, bits));
      }
    }
  }
  std::vector<ANSEncodingData> codes;
  std::vector<uint8_t> context_map;
  const std::string enc_hist =
      BuildAndEncodeHistograms(3, tokens, &codes, &context_map, info);
  const std::string enc_img = WriteTokens(tokens[0], codes, context_map, info);
  return enc_hist + enc_img;
}

bool DecodeHistograms(BitReader* br, const size_t num_contexts,
                      const size_t max_alphabet_size, ANSCode* code,
                      std::vector<uint8_t>* context_map) {
  size_t num_histograms = 1;
  context_map->resize(num_contexts);
  if (num_contexts > 1) {
    PIK_RETURN_IF_ERROR(DecodeContextMap(context_map, &num_histograms, br));
  }
  if (!DecodeANSCodes(num_histograms, max_alphabet_size, br, code)) {
    return PIK_FAILURE("Histo DecodeANSCodes");
  }
  br->JumpToByteBoundary();
  return true;
}

// See also EncodeImageData.
bool DecodeImageData(BitReader* PIK_RESTRICT br,
                     const std::vector<uint8_t>& context_map,
                     ANSSymbolReader* PIK_RESTRICT decoder, const Rect& rect,
                     Image3S* PIK_RESTRICT img) {
  const size_t xsize = rect.xsize();
  const size_t ysize = rect.ysize();
  PIK_ASSERT(xsize <= img->xsize() && ysize <= img->ysize());
  for (int c = 0; c < 3; ++c) {
    const int histo_idx = context_map[c];

    for (size_t y = 0; y < ysize; ++y) {
      int16_t* PIK_RESTRICT row = rect.Row(img->MutablePlane(c), y);

      for (size_t x = 0; x < xsize; ++x) {
        br->FillBitBuffer();
        int s = decoder->ReadSymbol(histo_idx, br);
        if (s > 0) {
          int bits = br->PeekBits(s);
          br->Advance(s);
          s = DecodeVarLenInt(s, bits);
        }
        row[x] = s;
      }
    }
  }
  br->JumpToByteBoundary();
  return true;
}

template <int N>
bool DecodeCoeffOrder(int32_t* order, bool decode_first, BitReader* br) {
  constexpr int block_size = N * N;
  int32_t lehmer[block_size] = {0};
  for (int32_t i = 0; i < block_size; i += kCoeffOrderCodeSpan) {
    br->FillBitBuffer();
    const int32_t has_non_zero = br->ReadBits(1);
    if (!has_non_zero) continue;
    const int32_t start = (i > 0) ? i : (decode_first ? 0 : 1);
    const int32_t end = i + kCoeffOrderCodeSpan;
    for (int32_t j = start; j < end; ++j) {
      int32_t v = 0;
      while (v <= block_size) {
        br->FillBitBuffer();
        const int32_t bits = br->ReadBits(3);
        v += bits;
        if (bits < 7) break;
      }
      if (v > block_size) v = block_size;
      lehmer[j] = v;
    }
  }
  int32_t end = block_size - 1;
  while (end > 0 && lehmer[end] == 0) {
    --end;
  }
  for (int32_t i = 1; i <= end; ++i) {
    --lehmer[i];
  }
  DecodeLehmerCode(lehmer, block_size, order);
  const int32_t* natural_coeff_order = NaturalCoeffOrder<N>();
  for (size_t k = 0; k < block_size; ++k) {
    order[k] = natural_coeff_order[order[k]];
  }
  return true;
}
template bool DecodeCoeffOrder<8>(int32_t*, bool, BitReader*);

bool DecodeImage(BitReader* PIK_RESTRICT br, const Rect& rect,
                 Image3S* PIK_RESTRICT img) {
  std::vector<uint8_t> context_map;
  ANSCode code;
  PIK_RETURN_IF_ERROR(DecodeHistograms(br, 3, 32, &code, &context_map));
  ANSSymbolReader decoder(&code);
  PIK_RETURN_IF_ERROR(DecodeImageData(br, context_map, &decoder, rect, img));
  if (!decoder.CheckANSFinalState()) {
    return PIK_FAILURE("ANS checksum failure.");
  }
  return true;
}

bool DecodeQuantField(BitReader* PIK_RESTRICT br,
                      ANSSymbolReader* PIK_RESTRICT decoder,
                      const std::vector<uint8_t>& context_map,
                      const Rect& rect_qf,
                      const ImageB& PIK_RESTRICT ac_strategy,
                      ImageI* PIK_RESTRICT quant_field) {
  const size_t xsize = rect_qf.xsize();
  const size_t ysize = rect_qf.ysize();

  for (size_t by = 0; by < ysize; ++by) {
    int32_t* PIK_RESTRICT row_quant = rect_qf.Row(quant_field, by);
    const int32_t* PIK_RESTRICT row_quant_top =
        (by == 0) ? nullptr : rect_qf.ConstRow(*quant_field, by - 1);
    for (size_t bx = 0; bx < xsize; ++bx) {
      int32_t predicted_quant =
          PredictFromTopAndLeft(row_quant_top, row_quant, bx, 32);
      br->FillBitBuffer();
      int32_t quant_ctx = QuantContext(predicted_quant);
      row_quant[bx] = decoder->ReadSymbol(context_map[quant_ctx], br) + 1;
    }
  }
  return true;
}

template <int N>
bool DecodeCoefficients(BitReader* PIK_RESTRICT br,
                        ANSSymbolReader* PIK_RESTRICT decoder,
                        const Image3B& tmp_block_ctx,
                        const std::vector<uint8_t>& context_map,
                        const int32_t* PIK_RESTRICT coeff_order,
                        const Rect& rect_ac, Image3S* PIK_RESTRICT ac,
                        const Rect& rect_bm,
                        const ImageB& PIK_RESTRICT ac_strategy,
                        Image3I* PIK_RESTRICT tmp_num_nzeroes) {
  constexpr int block_size = N * N;
  const size_t xsize_blocks = rect_bm.xsize();
  const size_t ysize_blocks = rect_bm.ysize();

  for (int c = 0; c < 3; ++c) {
    for (size_t by = 0; by < ysize_blocks; ++by) {
      const uint8_t* PIK_RESTRICT ac_strategy_row =
          rect_bm.ConstRow(ac_strategy, by);
      const uint8_t* PIK_RESTRICT row_bctx = tmp_block_ctx.ConstPlaneRow(c, by);
      int16_t* PIK_RESTRICT row_ac = rect_ac.PlaneRow(ac, c, by);
      int32_t* PIK_RESTRICT row_nzeros = tmp_num_nzeroes->PlaneRow(c, by);
      const int32_t* PIK_RESTRICT row_nzeros_top =
          (by == 0) ? nullptr : tmp_num_nzeroes->ConstPlaneRow(c, by - 1);

      for (size_t bx = 0; bx < xsize_blocks; ++bx) {
        int16_t* PIK_RESTRICT block_ac = row_ac + bx * block_size;
        memset(block_ac, 0, block_size * sizeof(row_ac[0]));
        int32_t predicted_nzeros =
            PredictFromTopAndLeft(row_nzeros_top, row_nzeros, bx, 32);
        const size_t block_ctx = row_bctx[bx];
        const size_t nzero_ctx = NonZeroContext<N>(predicted_nzeros, block_ctx);
        br->FillBitBuffer();
        row_nzeros[bx] = decoder->ReadSymbol(context_map[nzero_ctx], br);
        size_t num_nzeros = row_nzeros[bx];
        if (num_nzeros > block_size) {
          return PIK_FAILURE("Invalid AC: nzeros too large");
        }
        if (num_nzeros == 0) continue;
        const int histo_offset = ZeroDensityContextsOffset<N>(block_ctx);
        const size_t order_offset = block_ctx * block_size;
        const int* PIK_RESTRICT block_order = &coeff_order[order_offset];
        PIK_ASSERT(block_ctx < kOrderContexts);
        bool is_dct = AcStrategyType::IsDct(ac_strategy_row[bx]);
        for (size_t k = is_dct; k < block_size && num_nzeros > 0; ++k) {
          int context =
              histo_offset + ZeroDensityContext(num_nzeros, k - is_dct);
          br->FillBitBuffer();
          int symbol = decoder->ReadSymbol(context_map[context], br);
          int nbits = kBitsLut[symbol];
          int skip = kSkipLut[symbol];
          k += skip;
          if (nbits == 0) {
            // NB: currently format does not prohibit unoptimal code.
            // PIK_ASSERT(skip == 15);
            continue;
          }
          if (PIK_UNLIKELY(k + num_nzeros > block_size)) {
            return PIK_FAILURE("Invalid AC data.");
          }
          int32_t bits = br->PeekBits(nbits);
          br->Advance(nbits);
          int32_t coeff = DecodeVarLenInt(nbits, bits);
          --num_nzeros;
          block_ac[block_order[k]] = coeff;
        }
        if (num_nzeros != 0) {
          return PIK_FAILURE("Invalid AC: nzeros not 0.");
        }
      }
    }
  }
  return true;
}

template <int N>
bool DecodeAC(const Image3B& tmp_block_ctx, const ANSCode& code,
              const std::vector<uint8_t>& context_map,
              const int32_t* PIK_RESTRICT coeff_order,
              BitReader* PIK_RESTRICT br, const Rect& rect_ac,
              Image3S* PIK_RESTRICT ac, const Rect& rect_qf,
              ImageI* PIK_RESTRICT quant_field,
              const ImageB& PIK_RESTRICT ac_strategy,
              Image3I* PIK_RESTRICT tmp_num_nzeroes) {
  constexpr int block_size = N * N;
  PIK_ASSERT(SameSize(rect_qf, rect_ac));
  // Transform block coordinates to coefficient layout coordinates.
  Rect literal_rect_ac(rect_ac.x0() * block_size, rect_ac.y0(),
                       rect_ac.xsize() * block_size, rect_ac.ysize());

  PIK_ASSERT(literal_rect_ac.IsInside(*ac));
  PIK_ASSERT(rect_qf.IsInside(*quant_field));

  PIK_ASSERT(SameSize(tmp_block_ctx, *tmp_num_nzeroes));

  ANSSymbolReader decoder(&code);
  PIK_RETURN_IF_ERROR(DecodeQuantField(br, &decoder, context_map, rect_qf,
                                       ac_strategy, quant_field));
  PIK_RETURN_IF_ERROR(DecodeCoefficients<N>(
      br, &decoder, tmp_block_ctx, context_map, coeff_order, literal_rect_ac,
      ac, rect_qf, ac_strategy, tmp_num_nzeroes));

  br->JumpToByteBoundary();
  if (!decoder.CheckANSFinalState()) {
    return PIK_FAILURE("ANS checksum failure.");
  }
  return true;
}
template bool DecodeAC<8>(const Image3B&, const ANSCode&,
                          const std::vector<uint8_t>&,
                          const int32_t* PIK_RESTRICT, BitReader* PIK_RESTRICT,
                          const Rect&, Image3S* PIK_RESTRICT, const Rect&,
                          ImageI* PIK_RESTRICT, const ImageB& PIK_RESTRICT,
                          Image3I* PIK_RESTRICT);

}  // namespace pik
