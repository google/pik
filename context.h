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

#ifndef CONTEXT_H_
#define CONTEXT_H_

#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>
#include <vector>

#include "distributions.h"
#include "fast_log.h"
#include "status.h"

namespace pik {

using coeff_t = int16_t;

static const int kDCTBlockSize = 64;
static const int kMaxAverageContext = 8;
static const int kNumAvrgContexts = kMaxAverageContext + 1;
static const int kInterSymbolContexts = 63;

static const uint8_t kNonzeroBuckets[64] = {
   0,   1,   2,   3,   4,   4,   5,   5,   5,   6,   6,   6,   6,   7,   7,   7,
   7,   7,   7,   7,   7,   8,   8,   8,   8,   8,   8,   8,   8,   8,   8,   8,
   9,   9,   9,   9,   9,   9,   9,   9,   9,   9,   9,   9,   9,  10,  10,  10,
  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,
};
static const uint8_t kNumNonzeroBuckets = 11;

static const uint8_t kFreqContext[7][64] = {
  { 0, },

  { 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,
    1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
    1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,  0,  0, },

  { 0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,
    2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  3,  3,  3,  3,  3,  3,
    3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
    3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  1,  1,  1, },

  { 0,  1,  1,  2,  2,  2,  3,  3,  3,  3,  4,  4,  4,  4,  4,  4,
    5,  5,  5,  5,  5,  5,  5,  5,  6,  6,  6,  6,  6,  6,  6,  6,
    6,  6,  6,  6,  6,  6,  6,  6,  7,  7,  7,  7,  7,  7,  7,  7,
    7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  2,  2,  2, },

  { 0,  1,  2,  3,  4,  4,  5,  5,  6,  6,  7,  7,  8,  8,  8,  8,
    9,  9,  9,  9, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12,
   13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14,
   15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, },

  { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
   16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23,
   24, 24, 24, 24, 25, 25, 25, 25, 26, 26, 26, 26, 27, 27, 27, 27,
   28, 28, 28, 28, 29, 29, 29, 29, 30, 30, 30, 30, 31, 31, 31, 31, },

  { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
   16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
   32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
   48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, },
};

static const uint16_t kNumNonzeroContext[7][64] = {
 { 0,   1,   1,   2,   2,   2,   3,   3,   3,   3,   4,   4,   4,   4,   4,   4,
   5,   5,   5,   5,   5,   5,   5,   5,   6,   6,   6,   6,   6,   6,   6,   6,
   6,   6,   6,   6,   6,   6,   6,   6,   7,   7,   7,   7,   7,   7,   7,   7,
   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7
 },
 { 0,   2,   2,   4,   4,   4,   6,   6,   6,   6,   8,   8,  8,   8,   8,   8,
  10,  10,  10,  10,  10,  10,  10,  10,  12,  12,  12,  12,  12,  12,  12,  12,
  12,  12,  12,  12,  12,  12,  12,  12,  14,  14,  14,  14,  14,  14,  14,  14,
  14,  14,  14,  14,  14,  14,  14,  14,  14,  14,  14,  14,  14,  14,  14,  14
 },
 { 0,   4,   4,   8,   8,   8,  12,  12,  12,  12,  16,  16,  16,  16,  16,  16,
  20,  20,  20,  20,  20,  20,  20,  20,  24,  24,  24,  24,  24,  24,  24,  24,
  24,  24,  24,  24,  24,  24,  24,  24,  28,  28,  28,  28,  28,  28,  28,  28,
  28,  28,  28,  28,  28,  28,  28,  28,  28,  28,  28,  28,  28,  28,  28,  28
 },
 { 0,   8,   8,  16,  16,  16,  24,  24,  24,  24,  32,  32,  32,  32,  32,  32,
  40,  40,  40,  40,  40,  40,  40,  40,  48,  48,  48,  48,  48,  48,  48,  48,
  48,  48,  48,  48,  48,  48,  48,  48,  55,  55,  55,  55,  55,  55,  55,  55,
  55,  55,  55,  55,  55,  55,  55,  55,  55,  55,  55,  55,  55,  55,  55,  55
 },
 { 0,  16,  16,  32,  32,  32,  48,  48,  48,  48,  64,  64,  64,  64,  64,  64,
  80,  80,  80,  80,  80,  80,  80,  80,  95,  95,  95,  95,  95,  95,  95,  95,
  95,  95,  95,  95,  95,  95,  95,  95, 109, 109, 109, 109, 109, 109, 109, 109,
 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109
 },
 { 0,  32,  32,  64,  64,  64,  96,  96,  96,  96, 127, 127, 127, 127, 127, 127,
 157, 157, 157, 157, 157, 157, 157, 157, 185, 185, 185, 185, 185, 185, 185, 185,
 185, 185, 185, 185, 185, 185, 185, 185, 211, 211, 211, 211, 211, 211, 211, 211,
 211, 211, 211, 211, 211, 211, 211, 211, 211, 211, 211, 211, 211, 211, 211, 211
 },
 { 0,  64,  64, 127, 127, 127, 188, 188, 188, 188, 246, 246, 246, 246, 246, 246,
 300, 300, 300, 300, 300, 300, 300, 300, 348, 348, 348, 348, 348, 348, 348, 348,
 348, 348, 348, 348, 348, 348, 348, 348, 388, 388, 388, 388, 388, 388, 388, 388,
 388, 388, 388, 388, 388, 388, 388, 388, 388, 388, 388, 388, 388, 388, 388, 388
 }
};

constexpr uint16_t kNumNonzeroContextSkip[7] = {8, 15, 31, 61, 120, 231, 412};

/* This function is used for entropy-sources pre-clustering.
 *
 * Ideally, each combination of |nonzeros_left| and |k| should go to its own
 * bucket; but it implies (64 * 63 / 2) == 2016 buckets. If there is other
 * dimension (e.g. block context), then number of primary clusters becomes too
 * big.
 *
 * To solve this problem, |nonzeros_left| and |k| values are clustered. It is
 * known that their sum is less than 64, consequently, the total number buckets
 * is less than A(64) * B(64).
 *
 * |bits| controls the granularity of pre-clustering. When |bits| is 0, all |k|
 * values are put together. When |bits| is 6, then all 64 |k| values go to
 * different buckets.
 *
 * Also see the test code, where more compact presentation is expanded into
 * those lookup tables.
 */
inline int ZeroDensityContext(int nonzeros_left, int k, int bits) {
  PIK_ASSERT(bits < 7);
  PIK_ASSERT(nonzeros_left + k < 64);
  return kNumNonzeroContext[bits][nonzeros_left] + kFreqContext[bits][k];
}

// Returns the context for the absolute value of the prediction error of
// the next DC coefficient in column x, using the one row size ringbuffer of
// previous absolute prediction errors in vals.
inline int WeightedAverageContextDC(const int* vals, int x) {
  // Since vals is a ringbuffer, vals[x] and vals[x + 1] refer to the
  // previous row.
  int sum = 1 + vals[x - 2] + vals[x - 1] + vals[x] + vals[x + 1];
  if ((sum >> kMaxAverageContext) != 0) {
    return kMaxAverageContext;
  }
  return Log2FloorNonZero(sum);
}

inline int WeightedAverageContext(const int* vals, int prev_row_delta) {
  int sum = 4 +
      vals[0] +
      (vals[-kDCTBlockSize] + vals[prev_row_delta]) * 2 +
      vals[-2 * kDCTBlockSize] +
      vals[prev_row_delta - kDCTBlockSize] +
      vals[prev_row_delta + kDCTBlockSize];
  if ((sum >> (kMaxAverageContext + 2)) != 0) {
    return kMaxAverageContext;
  }
  return Log2FloorNonZero(sum) - 2;
}

static const int kACPredictPrecisionBits = 13;
static const int kACPredictPrecision = 1 << kACPredictPrecisionBits;

void ComputeACPredictMultipliers(const int* quant,
                                 int* mult_row,
                                 int* mult_col);

// Returns a context value from the AC prediction in the range
// [-kMaxAverageContext, kMaxAverageContext]
inline int ACPredictContext(int p_orig) {
  int p = unsigned(p_orig) << 1;
  static const int kMaxPred = (1 << (kMaxAverageContext + 1)) - 1;
  if (p_orig >= 0) {
    if (p > kMaxPred) {
      return kMaxAverageContext;
    }
    return Log2FloorNonZero(++p);
  }
  p = -p;
  if (p > kMaxPred) {
    return -kMaxAverageContext;
  } else {
    return -Log2FloorNonZero(++p);
  }
}

inline int ACPredictContextCol(const coeff_t* prev,
                               const coeff_t* cur,
                               const int* mult) {
  int p = prev[0] - ((cur[1] + prev[1]) * mult[1] +
                     (cur[2] - prev[2]) * mult[2] +
                     (cur[3] + prev[3]) * mult[3] +
                     (cur[4] - prev[4]) * mult[4] +
                     (cur[5] + prev[5]) * mult[5] +
                     (cur[6] - prev[6]) * mult[6] +
                     (cur[7] + prev[7]) * mult[7]) / kACPredictPrecision;
  return ACPredictContext(p);
}

inline int ACPredictContextRow(const coeff_t* prev,
                               const coeff_t* cur,
                               const int* mult) {
  int p = prev[0] - ((cur[ 8] + prev[ 8]) * mult[ 8] +
                     (cur[16] - prev[16]) * mult[16] +
                     (cur[24] + prev[24]) * mult[24] +
                     (cur[32] - prev[32]) * mult[32] +
                     (cur[40] + prev[40]) * mult[40] +
                     (cur[48] - prev[48]) * mult[48] +
                     (cur[56] + prev[56]) * mult[56]) / kACPredictPrecision;
  return ACPredictContext(p);
}

inline int NumNonzerosContext(const int* prev, int x, int y) {
  return (y == 0 ? (prev[x - 1] >> 1) :
          x == 0 ? (prev[0] >> 1) :
          (prev[x - 1] + prev[x] + 1) >> 2);
}

// Context for the emptyness of a block is the number of non-empty blocks in the
// previous and up neighborhood (blocks beyond the border are assumed
// non-empty).
static const int kNumIsEmptyBlockContexts = 3;
inline int IsEmptyBlockContext(const int* prev, int x) {
  return prev[x - 1] + prev[x];
}

// Holds all encoding/decoding state for an image component that is needed to
// switch between components during interleaved encoding/decoding.
struct ComponentStateDC {
  ComponentStateDC() :
      width(0),
      is_empty_block_prob(kNumIsEmptyBlockContexts),
      sign_prob(9),
      first_extra_bit_prob(10) {
    InitAll();
  }

  void SetWidth(int w) {
    width = w;
    prev_is_nonempty.resize(w + 1, 1);
    prev_abs_coeff.resize(w + 3);
    prev_sign.resize(w + 1);
  }

  int width;
  Prob is_zero_prob;
  std::vector<Prob> is_empty_block_prob;
  std::vector<Prob> sign_prob;
  std::vector<Prob> first_extra_bit_prob;
  std::vector<int> prev_is_nonempty;
  std::vector<int> prev_abs_coeff;
  std::vector<int> prev_sign;

 protected:
  void InitAll();
};

struct ComponentState {
  ComponentState() :
      width(0),
      is_zero_prob(kNumNonzeroBuckets * kDCTBlockSize),
      sign_prob((2 * kMaxAverageContext + 1) * kDCTBlockSize),
      first_extra_bit_prob(10 * kDCTBlockSize) {
    InitAll();
  }

  void SetWidth(int w) {
    width = w;
    prev_is_nonempty.resize(w + 1, 1);
    prev_num_nonzeros.resize(w + 1);
    prev_abs_coeff.resize(kDCTBlockSize * 2 * (w + 3));
    prev_sign.resize(kDCTBlockSize * (w + 1));
  }

  // Returns the size of the object after constructor and SetWidth(w).
  // Used in estimating peak heap memory usage of the brunsli codec.
  static size_t SizeInBytes(int w) {
    return (4 + (10 + 3 * w) * kDCTBlockSize + 2 * w) * sizeof(int) +
        ((kNumNonzeroBuckets + 2 * kMaxAverageContext + 11) * kDCTBlockSize +
         32 * kInterSymbolContexts) * sizeof(Prob);
  }

  int width;
  int context_offset;
  int order[kDCTBlockSize];
  int mult_row[kDCTBlockSize];
  int mult_col[kDCTBlockSize];
  std::vector<Prob> is_zero_prob;
  std::vector<Prob> sign_prob;
  Prob num_nonzero_prob[32][kInterSymbolContexts];
  std::vector<Prob> first_extra_bit_prob;
  std::vector<int> prev_is_nonempty;
  std::vector<int> prev_num_nonzeros;
  std::vector<int> prev_abs_coeff;
  std::vector<int> prev_sign;

 protected:
  void InitAll();
};

}  // namespace pik

#endif  // CONTEXT_H_
