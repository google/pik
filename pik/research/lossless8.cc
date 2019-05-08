// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

// @author Alexander Rhatushnyak

#include "pik/lossless8.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "pik/lossless_c3p.h"
#include "pik/lossless_entropy.h"

namespace pik {

namespace {

// Multiple prediciton modes, each has its own 4 predictors and heuristics.
typedef enum { PM_Regular, PM_West, PM_North } PredictMode;

const int mulWeights0and1_R_[] = {
    34, 36,  // when errors are small,
    31, 37,  // we assume they are random noise,
    33, 37,  // and penalize predictors 0 and 1
    36, 40,  //
    39, 44,  //
    42, 46,  //
    43, 47,  //
    43, 42,  //
};

const int mulWeights3teNE_R_[] = {
    28, 0, 24, 15, 24, 19, 24, 16, 23, 12, 23, 12, 25, 11, 32, 11,
};

const int mulWeights0and1_W_[] = {
    27, 31,  // when errors are small,
    33, 31,  // we assume they are random noise,
    40, 34,  // and penalize predictors 0 and 1
    43, 36, 52, 43, 59, 45, 63, 43, 65, 28,
};

const int mulWeights3teNE_W_[] = {
    31, 0, 31, 21, 29, 19, 28, 13, 26, 14, 28, 24, 32, 26, 43, 35,
};

const int mulWeights0and1_N_[] = {
    43, 23,  // when errors are small,
    38, 21,  // we assume they are random noise,
    35, 24,  // and penalize predictors 0 and 1
    34, 27, 35, 29, 33, 31, 28, 31, 23, 31,
};

const int mulWeights3teNE_N_[] = {
    27, 0, 23, 29, 26, 34, 29, 29, 30, 13, 35, 13, 40, 11, 51, 9,
};

const int kWithSign = 7;
const int kNumContexts = 8 + kWithSign + 2;
const int kMaxError = 101;
const int kMaxSumErrors = kMaxError * 7 + 1;

// The maximum number of total image pixels supported by the lossless format.
const uint32_t kMaxLossless8Pixels = 1 << 28;
// The maximum dimension of total pixels in Y.
const uint32_t kMaxLossless8PixelsY = 1 << 16;

// Left shift a signed integer by the shift amount.
PIK_INLINE int LshInt(int value, unsigned shift) {
  // Cast to unsigned and back to avoid undefined behavior of signed left shift.
  return static_cast<int>(static_cast<unsigned>(value) << shift);
}

// TODO(lode): split state variables needed for encoder from those for decoder
//             and perform one-time global initialization where possible.
struct State {
  const size_t groupSizeX;
  const size_t groupSizeY;
  const size_t groupSize2plus;

  // Multiplier (by shift) of pixel, prediction and error values, to perform
  // prediction computations in higher precision.
  const int PBits = 3;  // SET ME TO ZERO FOR A FASTER VERSION WITH NO ROUNDING!
  const int toRound = ((1 << PBits) >> 1);
  // Mask used to round the division by right shifting with PBits.
  const int toRound_m1 = (toRound ? toRound - 1 : 0);

  // uint64_t gqe[kNumContexts];  // global quantized errors (all groups) counts

  // The uncompressed data per context, the total sum of the size of all
  // kNumContexts edata arrays must be the original image group size, the
  // context decides from which of the arrays each next byte is taken.
  // size should be [2][] instead of [kNumContexts][] in the Production edition
  std::vector<uint8_t> edata[kNumContexts];

  // Buffers used to hold the compressed data while entropy encoding or
  // decoding.
  std::vector<uint8_t> compressedDataTmpBuf;
  uint8_t* compressedData;

  // Errors of the 4 predictors for current and previous scanline.
  // Range 0..kMaxError
  std::vector<uint8_t> errors0;  // Errors of predictor 0.
  std::vector<uint8_t> errors1;  // Errors of predictor 1
  std::vector<uint8_t> errors2;  // Errors of predictor 2
  std::vector<uint8_t> errors3;  // Errors of predictor 3
  std::vector<int16_t> trueErr;  // True errors. Their range is -255...255
  std::vector<uint8_t> quantizedError;  // The range is 0...14, all are
                                        // even due to quantizedInit()

// Whether to use a 1D table or a 2D table to compute encoded value from
// actual & prediction (in encoder) or compute the actual value from
// encoded & prediction (in decoder). The simple transform is analogous to
// subtracting actual - prediction and moving the sign bit to the LSB, the
// complex transform uses a more complex 2D lookup table given actual &
// prediction instead.
// FRWRD = the transform used in encoder, BKWRD = the transform used in decoder.
#ifdef SIMPLE_signToLSB_TRANSFORM  // to fully disable, "=i;" in the init macros

  uint8_t signLSB_forwardTransform[256],
      signLSB_backwardTransform[256];  // const
#define ToLSB_FRWRD signLSB_forwardTransform[err & 255]
#define ToLSB_BKWRD (prediction - signLSB_backwardTransform[q]) & 255

#define signToLSB_FORWARD_INIT                                             \
  do {                                                                     \
    for (int i = 0; i < 256; ++i)                                          \
      signLSB_forwardTransform[i] = (i & 128 ? (255 - i) * 2 + 1 : i * 2); \
  } while (0)

#define signToLSB_BACKWARD_INIT                                         \
  do {                                                                  \
    for (int i = 0; i < 256; ++i)                                       \
      signLSB_backwardTransform[i] = (i & 1 ? 255 - (i >> 1) : i >> 1); \
  } while (0)

#else
  uint8_t signLSB_forwardTransform[1 << 16], signLSB_backwardTransform[1 << 16];
#define ToLSB_FRWRD signLSB_forwardTransform[prediction * 256 + truePixelValue]
#define ToLSB_BKWRD \
  signLSB_backwardTransform[((prediction + toRound_m1) >> PBits) * 256 + q]

#define CalcDistanceFromPredictionAndTPV \
  do {                                   \
    if (p >= 128) {                      \
      if (v >= p)                        \
        d = (v - p) * 2;                 \
      else if (v >= p - (255 - p))       \
        d = (p - v) * 2 - 1;             \
      else                               \
        d = 255 - v;                     \
    } else {                             \
      if (v < p)                         \
        d = (p - v) * 2 - 1;             \
      else if (v > p * 2)                \
        d = v;                           \
      else                               \
        d = (v - p) * 2;                 \
    }                                    \
  } while (0)

#define signToLSB_FORWARD_INIT                     \
  do {                                             \
    for (int p = 0; p < 256; ++p) {                \
      for (int d, v = 0; v < 256; ++v) {           \
        CalcDistanceFromPredictionAndTPV;          \
        signLSB_forwardTransform[p * 256 + v] = d; \
      }                                            \
    }                                              \
  } while (0)

#define signToLSB_BACKWARD_INIT                     \
  do {                                              \
    for (int p = 0; p < 256; ++p) {                 \
      for (int d, v = 0; v < 256; ++v) {            \
        CalcDistanceFromPredictionAndTPV;           \
        signLSB_backwardTransform[p * 256 + d] = v; \
      }                                             \
    }                                               \
  } while (0)
#endif

  // Heuristics table for prediction.
  uint8_t quantizedTable[256], diff2error[512 * 2];  // const
  uint16_t error2weight[kMaxSumErrors];              // const

  // Their range is -255...510 rather than 0...255!
  // And -510..510 after subtracting truePixelValue
  int prediction0, prediction1, prediction2, prediction3;
  int numColors[3], planeMethod, maxerrShift, maxTpv;  // Tpv is truePixelValue
  int width;                                           // width-1 actually

  // The current row of the image for prediction.
  uint8_t* PIK_RESTRICT rowImg;
  // Previous rows of the image for prediction.
  uint8_t const *PIK_RESTRICT rowPrev, *PIK_RESTRICT rowPP;

  State(size_t groupSizeX, size_t groupSizeY)
      : groupSizeX(groupSizeX),
        groupSizeY(groupSizeY),
        groupSize2plus(groupSizeX * groupSizeY * 9 / 8) {
    for (size_t i = 0; i < kNumContexts; i++) {
      // Also prevent uninitialized values in case of invalid compressed data
      edata[i].resize(groupSizeX * groupSizeY, 0);
    }
    quantizedError.resize(groupSizeX * 2);
    compressedDataTmpBuf.resize(groupSize2plus);
    for (int j = 0; j < kMaxSumErrors; ++j) {
      error2weight[j] =
          150 * 512 / (58 + j * std::sqrt(j + 50));  // const init!  150 58 50
    }
    errors0.resize(groupSizeX * 2 + 4);
    errors1.resize(groupSizeX * 2 + 4);
    errors2.resize(groupSizeX * 2 + 4);
    errors3.resize(groupSizeX * 2 + 4);
    trueErr.resize(groupSizeX * 2);

    for (int j = -512; j <= 511; ++j)
      diff2error[512 + j] = std::min(j < 0 ? -j : j, kMaxError);  // const init!
    for (int j = 0; j <= 255; ++j)
      quantizedTable[j] = quantizedInit(j);  // const init!
    // for (int i=0; i < 512; i += 16, printf("\n"))
    //   for (int j=i; j < i + 16; ++j)  printf("%2d, ", quantizedTable[j]);
    signToLSB_FORWARD_INIT;   // const init!
    signToLSB_BACKWARD_INIT;  // const init!
  }

  PIK_INLINE int quantized(int x) {
    assert(0 <= x && x <= 255);
    return quantizedTable[x];
  }

  PIK_INLINE int quantizedInit(int x) {
    assert(0 <= x && x <= 255);
    x = (x + 1) >> 1;
    int res = (x >= 4 ? 4 : x);
    if (x >= 6) res = 5;  // no 'else' to reduce code size
    if (x >= 9) res = 6;
    if (x >= 15) res = 7;
    return res * 2;
  }

  PIK_INLINE int predictY0(size_t x, size_t yc, size_t yp, int* maxErr) {
    *maxErr = (x == 0 ? kNumContexts - 3
                      : x == 1 ? quantizedError[yc]
                               : std::max(quantizedError[yc],
                                          quantizedError[yc - 1]));
    prediction1 = prediction2 = prediction3 = (x > 0 ? rowImg[x - 1] : 27)
                                              << PBits;
    prediction0 =
        (x <= 1 ? prediction1
                : prediction1 +
                      LshInt(rowImg[x - 1] - rowImg[x - 2], PBits) * 5 / 16);
    return (prediction0 < 0 ? 0 : prediction0 > maxTpv ? maxTpv : prediction0);
  }

  PIK_INLINE int predictX0(size_t x, size_t yc, size_t yp, int* maxErr) {
    *maxErr =
        std::max(quantizedError[yp], quantizedError[yp + (x < width ? 1 : 0)]);
    prediction1 = prediction2 = prediction3 = rowPrev[x] << PBits;
    prediction0 =
        (((rowPrev[x] * 7 + rowPrev[x + (x < width ? 1 : 0)]) << PBits) + 4) >>
        3;
    return prediction0;
  }

  PIK_INLINE int predict_R_(size_t x, size_t yc, size_t yp, int* maxErr) {
    if (!rowPrev)
      return predictY0(x, yc, yp, maxErr);  // OK for Prototype edition
    if (x == 0)
      return predictX0(x, yc, yp, maxErr);  // tobe fixed in Production

    int N = rowPrev[x] << PBits, W = rowImg[x - 1] << PBits,
        NW = rowPrev[x - 1] << PBits;
    int a1 = (x < width ? 1 : 0), NE = rowPrev[x + a1] << PBits;
    int weight0 = errors0[yp] + errors0[yp - 1] + errors0[yp + a1];
    int weight1 = errors1[yp] + errors1[yp - 1] + errors1[yp + a1];
    int weight2 = errors2[yp] + errors2[yp - 1] + errors2[yp + a1];
    int weight3 = errors3[yp] + errors3[yp - 1] + errors3[yp + a1];

    uint8_t mxe = quantizedError[yc];
    mxe = std::max(mxe, quantizedError[yp]);
    mxe = std::max(mxe, quantizedError[yp - 1]);
    mxe = std::max(mxe, quantizedError[yp + a1]);
    if (x > 1) mxe = std::max(mxe, quantizedError[yc - 1]);
    int mE = mxe;  // at this point 0 <= mxe <= 14,  and  mxe % 2 == 0

    weight0 = error2weight[weight0] * mulWeights0and1_R_[0 + mE];
    weight1 = error2weight[weight1] * mulWeights0and1_R_[1 + mE];
    weight2 = error2weight[weight2] * 32;  // Baseline
    weight3 = error2weight[weight3] * mulWeights3teNE_R_[0 + mE];

    int teW = trueErr[yc];
    int teN = trueErr[yp];
    int sumWN = teN + teW;  //  -510<<PBits <= sumWN <= 510<<PBits
    int teNW = trueErr[yp - 1];
    int teNE = trueErr[yp + a1];

    if (mE) {
      if (sumWN * 40 + teNW * 20 + teNE * mulWeights3teNE_R_[1 + mE] <= 0) ++mE;
    } else {
      if (N == W && N == NE)
        mE = ((sumWN | teNE | teNW) == 0 ? kNumContexts - 1 : 1);
    }
    *maxErr = mE;

    prediction0 = W - (sumWN + teNW) / 4;  // 7/32 works better than 1/4 ?
    prediction1 =
        N - (sumWN + teNE) / 4;  // predictors 0 & 1 rely on true errors
    prediction2 = W + NE - N;
    int t = (teNE * 3 + teNW * 4 + 7) >> 5;
    prediction3 = N + (N - (rowPP[x] << PBits)) * 23 / 32 + (W - NW) / 16 - t;
    assert(LshInt(-255, PBits) <= prediction0 && prediction0 <= 510 << PBits);
    assert(LshInt(-255, PBits) <= prediction1 && prediction1 <= 510 << PBits);
    assert(LshInt(-255, PBits) <= prediction2 && prediction2 <= 510 << PBits);
    assert(LshInt(-255, PBits) <= prediction3 && prediction3 <= 510 << PBits);

    int sumWeights = weight0 + weight1 + weight2 + weight3;
    // assert(sumWeights>0);  // true if min(error2weight)*min(mulWeights*_R_)>0

    int prediction = (prediction0 * weight0 + prediction1 * weight1 +
                      (sumWeights >> 3) + prediction2 * weight2 +
                      prediction3 * weight3) /  // biased rounding: >>3
                     sumWeights;

    if (((teN ^ teW) | (teN ^ teNW)) > 0)  // if all three have the same sign
      return (prediction < 0 ? 0 : prediction > maxTpv ? maxTpv : prediction);

    int max = (W > N ? W : N);
    int min = W + N - max;
    if (NE > max) max = NE;
    if (NE < min) min = NE;
    return (prediction < min ? min : prediction > max ? max : prediction);
  }

  PIK_INLINE int predict_W_(size_t x, size_t yc, size_t yp, int* maxErr) {
    if (!rowPrev)
      return predictY0(x, yc, yp, maxErr);  // OK for Prototype edition
    if (x == 0)
      return predictX0(x, yc, yp, maxErr);  // tobe fixed in Production

    int N = rowPrev[x] << PBits, W = rowImg[x - 1] << PBits,
        NW = rowPrev[x - 1] << PBits;
    int a1 = (x < width ? 1 : 0), NE = rowPrev[x + a1] << PBits;
    int weight0 = (errors0[yp] * 3 >> 1) + errors0[yp - 1] + errors0[yp + a1];
    int weight1 = (errors1[yp] * 3 >> 1) + errors1[yp - 1] + errors1[yp + a1];
    int weight2 = (errors2[yp] * 3 >> 1) + errors2[yp - 1] + errors2[yp + a1];
    int weight3 = (errors3[yp] * 3 >> 1) + errors3[yp - 1] + errors3[yp + a1];

    uint8_t mxe = quantizedError[yc];
    mxe = std::max(mxe, quantizedError[yp]);
    mxe = std::max(mxe, quantizedError[yp - 1]);
    mxe = std::max(mxe, quantizedError[yp + a1]);
    if (x > 1) mxe = std::max(mxe, quantizedError[yc - 1]);
    int mE = mxe;  // at this point 0 <= mxe <= 14,  and  mxe % 2 == 0

    weight0 = error2weight[weight0] * mulWeights0and1_W_[0 + mE];
    weight1 = error2weight[weight1] * mulWeights0and1_W_[1 + mE];
    weight2 = error2weight[weight2] * 32;  // Baseline
    weight3 = error2weight[weight3] * mulWeights3teNE_W_[0 + mE];

    int teW = trueErr[yc];
    int teN = trueErr[yp];
    int sumWN = teN + teW;  //  -510<<PBits <= sumWN <= 510<<PBits
    int teNW = trueErr[yp - 1];
    int teNE = trueErr[yp + a1];

    if (mE) {
      if (sumWN * 40 + (teNW + teNE) * mulWeights3teNE_W_[1 + mE] <= 0) ++mE;
    } else {
      if (N == W && N == NE)
        mE = ((sumWN | teNE | teNW) == 0 ? kNumContexts - 1 : 1);
    }
    *maxErr = mE;

    prediction0 =
        W - (sumWN + teNW) * 9 / 32;  // pr's 0 & 1 rely on true errors
    prediction1 =
        N - (sumWN + teNE) * 171 / 512;  // clamping not needed, is it?
    prediction2 = W + NE - N;
    prediction3 =
        N + ((N - (rowPP[x] << PBits)) >> 1) + ((W - NW) * 19 - teNW * 13) / 64;
    assert(LshInt(-255, PBits) <= prediction0 && prediction0 <= 510 << PBits);
    assert(LshInt(-255, PBits) <= prediction1 && prediction1 <= 510 << PBits);
    assert(LshInt(-255, PBits) <= prediction2 && prediction2 <= 510 << PBits);
    assert(LshInt(-255, PBits) <= prediction3 && prediction3 <= 510 << PBits);

    int sumWeights = weight0 + weight1 + weight2 + weight3;
    // assert(sumWeights>0);  // true if min(error2weight)*min(mulWeights*_W_)>0

    int prediction =
        (prediction0 * weight0 + prediction1 * weight1 + (sumWeights >> 1) +
         prediction2 * weight2 + prediction3 * weight3) /
        sumWeights;

    if (((teN ^ teW) | (teN ^ teNE)) > 0)  // if all three have the same sign
      return (prediction < 0 ? 0 : prediction > maxTpv ? maxTpv : prediction);

    int max = (W > N ? W : N);
    int min = W + N - max;
    if (NE > max) max = NE;
    if (NE < min) min = NE;
    return (prediction < min ? min : prediction > max ? max : prediction);
  }

  PIK_INLINE int predict_N_(size_t x, size_t yc, size_t yp, int* maxErr) {
    if (!rowPrev)
      return predictY0(x, yc, yp, maxErr);  // OK for Prototype edition
    if (x == 0)
      return predictX0(x, yc, yp, maxErr);  // tobe fixed in Production

    int N = rowPrev[x] << PBits, W = rowImg[x - 1] << PBits;  //, NW is not used
    int a1 = (x < width ? 1 : 0), NE = rowPrev[x + a1] << PBits;
    int weight0 = errors0[yp] + errors0[yp - 1] + errors0[yp + a1];
    int weight1 = errors1[yp] + errors1[yp - 1] + errors1[yp + a1];
    int weight2 = errors2[yp] + errors2[yp - 1] + errors2[yp + a1];
    int weight3 = errors3[yp] + errors3[yp - 1] + errors3[yp + a1];

    uint8_t mxe = quantizedError[yc];
    mxe = std::max(mxe, quantizedError[yp]);
    mxe = std::max(mxe, quantizedError[yp - 1]);
    mxe = std::max(mxe, quantizedError[yp + a1]);
    if (x > 1) mxe = std::max(mxe, quantizedError[yc - 1]);
    int mE = mxe;  // at this point 0 <= mxe <= 14,  and  mxe % 2 == 0

    weight0 = error2weight[weight0] * mulWeights0and1_N_[0 + mE];
    weight1 = error2weight[weight1] * mulWeights0and1_N_[1 + mE];
    weight2 = error2weight[weight2] * 32;  // Baseline
    weight3 = error2weight[weight3] * mulWeights3teNE_N_[0 + mE];

    int teW = trueErr[yc];
    int teN = trueErr[yp];
    int sumWN = teN + teW;  //  -510<<PBits <= sumWN <= 510<<PBits
    int teNW = trueErr[yp - 1];
    int teNE = trueErr[yp + a1];

    if (mE) {
      if (sumWN * 40 + teNW * 23 + teNE * mulWeights3teNE_N_[1 + mE] <= 0) ++mE;
    } else {
      if (N == W && N == NE)
        mE = ((sumWN | teNE | teNW) == 0 ? kNumContexts - 1 : 1);
    }
    *maxErr = mE;

    prediction0 = N - (sumWN + teNW + teNE) / 4;  // if bigger than 1/4,
                                                  // clamping would be needed!
    prediction1 =
        W - ((teW * 2 + teNW) >> 2);  // pr's 0 & 1 rely on true errors
    prediction2 = W + NE - N;
    prediction3 = N + ((N - (rowPP[x] << PBits)) * 47) / 64 - (teN >> 2);
    assert(LshInt(-255, PBits) <= prediction0 && prediction0 <= 510 << PBits);
    assert(LshInt(-255, PBits) <= prediction1 && prediction1 <= 510 << PBits);
    assert(LshInt(-255, PBits) <= prediction2 && prediction2 <= 510 << PBits);
    assert(LshInt(-255, PBits) <= prediction3 && prediction3 <= 510 << PBits);

    int sumWeights = weight0 + weight1 + weight2 + weight3;
    // assert(sumWeights>0);  // true if min(error2weight)*min(mulWeights*_N_)>0

    int prediction =
        (prediction0 * weight0 + prediction1 * weight1 + (sumWeights >> 1) +
         prediction2 * weight2 + prediction3 * weight3) /
        sumWeights;

    if (((teN ^ teW) | (teN ^ teNE)) > 0)  // if all three have the same sign
      return (prediction < 0 ? 0 : prediction > maxTpv ? maxTpv : prediction);

    int max = (W > N ? W : N);
    int min = W + N - max;
    if (NE > max) max = NE;
    if (NE < min) min = NE;
    return (prediction < min ? min : prediction > max ? max : prediction);
  }

#define Update_Size_And_Errors                                    \
  do {                                                            \
    esize[maxErr] = s + 1;                                        \
    trueErr[yc + x] = err;                                        \
    q = quantized(q);                                             \
    quantizedError[yc + x] = q;                                   \
    uint8_t* dp = &diff2error[512 - truePixelValue];              \
    errors0[1 + yp + x] +=                                        \
        (errors0[yc + x] = dp[(prediction0 + toRound) >> PBits]); \
    errors1[1 + yp + x] +=                                        \
        (errors1[yc + x] = dp[(prediction1 + toRound) >> PBits]); \
    errors2[1 + yp + x] +=                                        \
        (errors2[yc + x] = dp[(prediction2 + toRound) >> PBits]); \
    errors3[1 + yp + x] +=                                        \
        (errors3[yc + x] = dp[(prediction3 + toRound) >> PBits]); \
  } while (0)

#define AfterPredictWhenCompressing                    \
  do {                                                 \
    maxErr >>= maxerrShift;                            \
    assert(0 <= maxErr && maxErr <= kNumContexts - 1); \
    int q, truePixelValue = rowImg[x];                 \
    int err = prediction - (truePixelValue << PBits);  \
    size_t s = esize[maxErr];                          \
    prediction = (prediction + toRound_m1) >> PBits;   \
    assert(0 <= prediction && prediction <= 255);      \
    edata[maxErr][s] = q = ToLSB_FRWRD;                \
    Update_Size_And_Errors; /* ++gqe[maxErr]; */       \
  } while (0)

#define AfterPredictWhenCompressing3                   \
  do {                                                 \
    maxErr >>= maxerrShift;                            \
    assert(0 <= maxErr && maxErr <= kNumContexts - 1); \
    int q, truePixelValue = rowImg[x];                 \
    if (planeToCompress != planeToUse) {               \
      truePixelValue -= (int)rowUse[x] - 0x80;         \
      truePixelValue &= 0xff;                          \
      rowImg[x] = truePixelValue;                      \
    }                                                  \
    int err = prediction - (truePixelValue << PBits);  \
    size_t s = esize[maxErr];                          \
    prediction = (prediction + toRound_m1) >> PBits;   \
    assert(0 <= prediction && prediction <= 255);      \
    edata[maxErr][s] = q = ToLSB_FRWRD;                \
    Update_Size_And_Errors; /* ++gqe[maxErr]; */       \
  } while (0)

#define AfterPredictWhenDecompressing                            \
  do {                                                           \
    maxErr >>= maxerrShift;                                      \
    assert(0 <= maxErr && maxErr <= kNumContexts - 1);           \
    assert(0 <= prediction && prediction <= 255 << PBits);       \
    size_t s = esize[maxErr];                                    \
    int err, q = edata[maxErr][s], truePixelValue = ToLSB_BKWRD; \
    rowImg[x] = truePixelValue;                                  \
    err = prediction - (truePixelValue << PBits);                \
    Update_Size_And_Errors;                                      \
  } while (0)

#define setRowImgPointers(imgRow)                                 \
  do {                                                            \
    yc = groupSizeX - yc;                                         \
    yp = groupSizeX - yc;                                         \
    rowImg = imgRow(groupY + y) + groupX;                         \
    rowPrev = (y == 0 ? NULL : imgRow(groupY + y - 1) + groupX);  \
    rowPP = (y <= 1 ? rowPrev : imgRow(groupY + y - 2) + groupX); \
  } while (0)

#define setRowImgPointers3(rowUse, imgRow)                                     \
  do {                                                                         \
    yc = groupSizeX - yc;                                                      \
    yp = groupSizeX - yc;                                                      \
    rowImg = imgRow(planeToCompress, groupY + y) + groupX;                     \
    rowUse = imgRow(planeToUse, groupY + y) + groupX;                          \
    rowPrev =                                                                  \
        (y == 0 ? NULL : imgRow(planeToCompress, groupY + y - 1) + groupX);    \
    rowPP =                                                                    \
        (y <= 1 ? rowPrev : imgRow(planeToCompress, groupY + y - 2) + groupX); \
  } while (0)

#define setRowImgPointers3dec(imgRow)                                         \
  do {                                                                        \
    yc = groupSizeX - yc;                                                     \
    yp = groupSizeX - yc;                                                     \
    rowImg = imgRow(planeToDecompress, groupY + y) + groupX;                  \
    rowPrev =                                                                 \
        (y == 0 ? NULL : imgRow(planeToDecompress, groupY + y - 1) + groupX); \
    rowPP = (y <= 1 ? rowPrev                                                 \
                    : imgRow(planeToDecompress, groupY + y - 2) + groupX);    \
  } while (0)

  bool Grayscale8bit_compress(const ImageB& img_in, pik::PaddedBytes* bytes) {
    // The code modifies the image for palette so must copy for now.
    ImageB img = CopyImage(img_in);

    size_t esize[kNumContexts], xsize = img.xsize(), ysize = img.ysize();
    std::vector<uint8_t> temp_buffer(groupSize2plus);
    compressedData = temp_buffer.data();
    size_t compressedCapacity = temp_buffer.size();

    int freqs[256];
    memset(freqs, 0, sizeof(freqs));
    for (size_t y = 0; y < ysize; ++y) {
      uint8_t* const PIK_RESTRICT rowImg = img.Row(y);
      for (size_t x = 0; x < xsize; ++x)  // UNROLL and PARALLELIZE ME!
        ++freqs[rowImg[x]];               // They can also be used for guessing
                                          // photo/nonphoto
    }
    int palette[256], count = 0;
    for (int i = 0; i < 256; ++i)
      palette[i] = count, count += (freqs[i] ? 1 : 0);
    int havePalette = (count < 255 ? 1 : 0);  // 255? or 256?
    maxTpv = (havePalette ? std::min(255, count + 1) : 255) << PBits;

    if (havePalette) {
      for (size_t y = 0; y < ysize; ++y) {
        uint8_t* const PIK_RESTRICT rowImg = img.Row(y);
        for (size_t x = 0; x < xsize; ++x)  // UNROLL and PARALLELIZE ME!
          rowImg[x] = palette[rowImg[x]];
      }
    }

    for (size_t groupY = 0; groupY < ysize; groupY += groupSizeY) {
      for (size_t groupX = 0; groupX < xsize; groupX += groupSizeX) {
        memset(esize, 0, sizeof(esize));
        size_t yEnd = std::min(groupSizeY, ysize - groupY);
        width = std::min(groupSizeX, xsize - groupX) - 1;
        size_t area = yEnd * (width + 1);
        maxerrShift =
            (area > 25600
                 ? 0
                 : area > 12800 ? 1 : area > 4000 ? 2 : area > 400 ? 3 : 4);

        double fromN = 0, fromW = 0;
        for (size_t y = 1; y < yEnd; ++y) {
          rowImg = img.Row(groupY + y) + groupX;
          rowPrev = img.Row(groupY + y - 1) + groupX;
          uint32_t fromNx = 0, fromWx = 0;
          for (size_t x = 1; x <= width; ++x) {
            int c = rowImg[x];
            int N = rowPrev[x];
            int W = rowImg[x - 1];
            N -= c;
            W -= c;
            fromNx += N * N;
            fromWx += W * W;
          }
          fromN += fromNx;
          fromW += fromWx;
        }
        PredictMode pMode = PM_Regular;
        if (fromW * 5 < fromN * 4)
          pMode = PM_West;  // no 'else' to reduce codesize
        if (fromN * 5 < fromW * 4) pMode = PM_North;  // if (fromN < fromW*0.8)
        // printf("%c ", pMode);

        if (pMode == PM_Regular) {  // Regular mode
          for (size_t y = 0, yc = 0, yp; y < yEnd; ++y) {
            setRowImgPointers(img.Row);
            for (size_t x = 0; x <= width; ++x) {
              int maxErr,
                  prediction = predict_R_(x, yc + x - 1, yp + x, &maxErr);
              AfterPredictWhenCompressing;
            }
          }
        } else if (pMode == PM_West) {  // 'West predicts better' mode
          for (size_t y = 0, yc = 0, yp; y < yEnd; ++y) {
            setRowImgPointers(img.Row);
            for (size_t x = 0; x <= width; ++x) {
              int maxErr,
                  prediction = predict_W_(x, yc + x - 1, yp + x, &maxErr);
              AfterPredictWhenCompressing;
            }
          }
        } else if (pMode == PM_North) {  // 'North predicts better' mode
          for (size_t y = 0, yc = 0, yp; y < yEnd; ++y) {
            setRowImgPointers(img.Row);
            for (size_t x = 0; x <= width; ++x) {
              int maxErr,
                  prediction = predict_N_(x, yc + x - 1, yp + x, &maxErr);
              AfterPredictWhenCompressing;
            }
          }
        } else {
        }  // TODO: other prediction modes!

        size_t pos = 0;
        if (groupY == 0 && groupX == 0) {
          pos += EncodeVarInt(xsize * 2 + havePalette, &compressedData[pos]);
          pos += EncodeVarInt(ysize, &compressedData[pos]);
          if (havePalette) {  // Save bit 1 if color is present, bit 0 if not
            const int kBitsPerByte = 8;
            for (int i = 0; i < 256 / kBitsPerByte; ++i) {
              int code = 0;
              for (int j = kBitsPerByte - 1; j >= 0; --j)
                code = code * 2 + (freqs[i * 8 + j] ? 1 : 0);
              compressedData[pos++] = code;
            }  // for i
          }    // if (havePalette)
        }      // if (groupY...)
        compressedData[pos++] = pMode;
        int nC = ((kNumContexts - 1) >> maxerrShift) + 1;
        std::vector<size_t> sizes(nC);
        std::vector<const uint8_t*> streams(nC);
        for (int i = 0; i < nC; ++i) {
          sizes[i] = esize[i];
          streams[i] = &edata[i][0];
        }
        if (!CompressWithEntropyCode(&pos, sizes.data(), streams.data(),
                                     sizes.size(), compressedCapacity,
                                     &compressedData[0])) {
          return PIK_FAILURE("lossless8 entropy encode failed");
        }
        size_t current = bytes->size();
        bytes->resize(bytes->size() + pos);
        memcpy(bytes->data() + current, &compressedData[0], pos);
      }  // groupX
    }    // groupY
    // for (int i=0; i<kNumContexts; ++i) printf("%3d
    // ",gqe[i]*1000/(xsize*ysize)); printf("\n");

    return true;
  }

  bool Grayscale8bit_decompress(const Span<const uint8_t> bytes,
                                size_t* bytes_pos, ImageB* result) {
    if (*bytes_pos > bytes.size()) return PIK_FAILURE("lossless8");
    size_t compressedSize = bytes.size() - *bytes_pos;
    const uint8_t* compressedData = bytes.data() + *bytes_pos;
    size_t esize[kNumContexts], xsize, ysize, pos = 0;
    xsize = DecodeVarInt(compressedData, compressedSize, &pos);
    ysize = DecodeVarInt(compressedData, compressedSize, &pos);
    int havePalette = xsize & 1, count = 256, palette[256];
    if (havePalette) {
      const uint8_t* p = &compressedData[pos];
      pos += 32;
      if (pos >= compressedSize) return PIK_FAILURE("lossless8");
      count = 0;
      for (int i = 0; i < 256; ++i)
        if (p[i >> 3] & (1 << (i & 7))) palette[count++] = i;
    }
    maxTpv = std::min(255, count + 1) << PBits;
    xsize >>= 1;
    if (!xsize || !ysize) return PIK_FAILURE("lossless8");
    // Check maximum supported image size. Too large would run out of memory.
    // We use a division to avoid overflow when multiplying.
    if (ysize > kMaxLossless8PixelsY || xsize > kMaxLossless8Pixels / ysize)
      return PIK_FAILURE("lossless8");

    pik::ImageB img(xsize, ysize);

    for (size_t groupY = 0; groupY < ysize; groupY += groupSizeY) {
      for (size_t groupX = 0; groupX < xsize; groupX += groupSizeX) {
        size_t yEnd = std::min(groupSizeY, ysize - groupY);
        width = std::min(groupSizeX, xsize - groupX) - 1;
        size_t area = yEnd * (width + 1);
        maxerrShift =
            (area > 25600
                 ? 0
                 : area > 12800 ? 1 : area > 4000 ? 2 : area > 400 ? 3 : 4);
        size_t decompressedSize = 0;  // is used only for the assert()

        PredictMode pMode = static_cast<PredictMode>(compressedData[pos++]);
        int nC = ((kNumContexts - 1) >> maxerrShift) + 1;
        if (!DecompressWithEntropyCode(compressedSize, compressedData, area, nC,
                                       edata, &pos)) {
          return PIK_FAILURE("lossless8 entropy decode failed");
        }
        for (int i = 0; i < nC; ++i) {
          decompressedSize += edata[i].size();
        }
        if (decompressedSize != area) return PIK_FAILURE("lossless8");
        if (groupY + groupSizeY >= ysize && groupX + groupSizeX >= xsize) {
          /* if the last group */
          // if (inpSize != pos) return PIK_FAILURE("lossless8");
        }
        memset(esize, 0, sizeof(esize));

        if (pMode == PM_Regular) {
          for (size_t y = 0, yc = 0, yp; y < yEnd; ++y) {
            setRowImgPointers(img.Row);
            for (size_t x = 0; x <= width; ++x) {
              int maxErr,
                  prediction = predict_R_(x, yc + x - 1, yp + x, &maxErr);
              AfterPredictWhenDecompressing;
            }
          }
        } else if (pMode == PM_West) {
          for (size_t y = 0, yc = 0, yp; y < yEnd; ++y) {
            setRowImgPointers(img.Row);
            for (size_t x = 0; x <= width; ++x) {
              int maxErr,
                  prediction = predict_W_(x, yc + x - 1, yp + x, &maxErr);
              AfterPredictWhenDecompressing;
            }
          }
        } else if (pMode == PM_North) {
          for (size_t y = 0, yc = 0, yp; y < yEnd; ++y) {
            setRowImgPointers(img.Row);
            for (size_t x = 0; x <= width; ++x) {
              int maxErr,
                  prediction = predict_N_(x, yc + x - 1, yp + x, &maxErr);
              AfterPredictWhenDecompressing;
            }
          }
        }
      }  // groupX
    }    // groupY
    if (havePalette) {
      for (size_t y = 0; y < ysize; ++y) {
        uint8_t* const PIK_RESTRICT rowImg = img.Row(y);
        for (size_t x = 0; x < xsize; ++x)  // UNROLL and PARALLELIZE ME!
          rowImg[x] = palette[rowImg[x]];
      }
    }
    *bytes_pos += pos;
    *result = std::move(img);
    return true;
  }

  bool dcmprs512x512(pik::Image3B* img, int planeToDecompress, size_t* pos,
                     size_t groupY, size_t groupX,
                     const uint8_t* compressedData, size_t compressedSize) {
    size_t esize[kNumContexts], xsize = img->xsize(), ysize = img->ysize();
    size_t yEnd = std::min(groupSizeY, ysize - groupY);
    width = std::min(groupSizeX, xsize - groupX) - 1;
    size_t area = yEnd * (width + 1);
    maxerrShift =
        (area > 25600
             ? 0
             : area > 12800 ? 1 : area > 4000 ? 2 : area > 400 ? 3 : 4);
    maxTpv = ((ncMap[planeMethod] & (1 << planeToDecompress))
                  ? numColors[planeToDecompress] - 1
                  : 255)
             << PBits;
    size_t decompressedSize = 0;  // is used only for the assert()

    PredictMode pMode = static_cast<PredictMode>(compressedData[(*pos)++]);
    int nC = ((kNumContexts - 1) >> maxerrShift) + 1;
    if (!DecompressWithEntropyCode(compressedSize, compressedData, area, nC,
                                   edata, pos)) {
      return PIK_FAILURE("lossless8 entropy decode failed");
    }
    for (int i = 0; i < nC; ++i) {
      decompressedSize += edata[i].size();
    }
    if (decompressedSize != area) return PIK_FAILURE("lossless8");
    // if (groupY + groupSize >= ysize && groupX + groupSize >= xsize)
    //  /* if the last group */  assert(inpSize == *pos);

    memset(esize, 0, sizeof(esize));

    if (pMode == PM_Regular) {
      for (size_t y = 0, yc = 0, yp; y < yEnd; ++y) {
        setRowImgPointers3dec(img->PlaneRow);
        for (size_t x = 0; x <= width; ++x) {
          int maxErr, prediction = predict_R_(x, yc + x - 1, yp + x, &maxErr);
          AfterPredictWhenDecompressing;
        }
      }
    } else if (pMode == PM_West) {
      for (size_t y = 0, yc = 0, yp; y < yEnd; ++y) {
        setRowImgPointers3dec(img->PlaneRow);
        for (size_t x = 0; x <= width; ++x) {
          int maxErr, prediction = predict_W_(x, yc + x - 1, yp + x, &maxErr);
          AfterPredictWhenDecompressing;
        }
      }
    } else if (pMode == PM_North) {
      for (size_t y = 0, yc = 0, yp; y < yEnd; ++y) {
        setRowImgPointers3dec(img->PlaneRow);
        for (size_t x = 0; x <= width; ++x) {
          int maxErr, prediction = predict_N_(x, yc + x - 1, yp + x, &maxErr);
          AfterPredictWhenDecompressing;
        }
      }
    }
    return true;
  }

  bool Colorful8bit_decompress(const Span<const uint8_t> bytes,
                               size_t* bytes_pos, Image3B* result) {
    if (*bytes_pos > bytes.size()) return PIK_FAILURE("lossless8");
    size_t compressedSize = bytes.size() - *bytes_pos;
    const uint8_t* compressedData = bytes.data() + *bytes_pos;

    size_t xsize, ysize, pos0 = 0, imageMethod = 0;
    xsize = DecodeVarInt(compressedData, compressedSize, &pos0);
    ysize = DecodeVarInt(compressedData, compressedSize, &pos0);
    if (!xsize || !ysize) return PIK_FAILURE("lossless8");
    // Check maximum supported image size. Too large would run out of memory.
    // We use a division to avoid overflow when multiplying.
    if (ysize > kMaxLossless8PixelsY || xsize > kMaxLossless8Pixels / ysize)
      return PIK_FAILURE("lossless8");

    pik::Image3B img(xsize, ysize);
    std::vector<int> palette(0x100 * 3);

    numColors[0] = numColors[1] = numColors[2] = 0x100;
    size_t pos = pos0;
    if (xsize * ysize > 4 * 0x100) {  // TODO: smarter decision making here
      if (pos >= compressedSize) return PIK_FAILURE("lossless8: out of bounds");
      const uint8_t* p = &compressedData[pos];
      imageMethod = *p++;
      if (imageMethod) {
        ++pos;
        if (pos + 3 >= compressedSize)
          return PIK_FAILURE("lossless8: out of bounds");
        numColors[0] = compressedData[pos++] + 1;
        numColors[1] = compressedData[pos++] + 1;
        numColors[2] = compressedData[pos++] + 1;
        p = &compressedData[pos];
        const uint8_t* p_end = compressedData + compressedSize;
        for (int channel = 0; channel < 3; ++channel) {
          if (imageMethod & (1 << channel)) {
            for (int sb = channel << 8, stop = sb + numColors[channel],
                     color = 0, x = 0;
                 x < 0x100; x += 8) {
              if (p >= p_end) return PIK_FAILURE("lossless8");
              for (int b = *p++, j = 0; j < 8; ++j)
                palette[sb] = color++, sb += b & 1, b >>= 1;
              if (sb >= stop) break;
              if (sb + 0x100 - 8 - x == stop) {
                for (int i = x; i < 0x100 - 8; ++i) palette[sb++] = color++;
                break;
              }
            }
          }
        }
      }
      pos = p - &compressedData[0];
    }

    decompress3planes(uint8_t, "lossless8");

// Disabled, because it is actually useful that the decoder supports decoding
// its own stream when contained inside a bigger stream and knows the correct
// end position.
#if 0
      assert(pos == compressedSize);
#endif

    for (int channel = 0; channel < 3; ++channel) {
      if (imageMethod & (1 << channel)) {
        int* p = &palette[0x100 * channel];
        for (size_t y = 0; y < ysize; ++y) {
          uint8_t* const PIK_RESTRICT rowImg = img.PlaneRow(channel, y);
          for (size_t x = 0; x < xsize; ++x)  // UNROLL AND PARALLELIZE ME!
            rowImg[x] = p[rowImg[x]];
        }
      }
    }
    *bytes_pos += pos;
    *result = std::move(img);
    return true;
  }

  bool cmprs512x512(pik::Image3B* img, int planeToCompress, int planeToUse,
                    size_t groupY, size_t groupX, size_t compressedCapacity,
                    uint8_t* compressedOutput, size_t* csize) {
    size_t esize[kNumContexts], xsize = img->xsize(), ysize = img->ysize();
    memset(esize, 0, sizeof(esize));
    size_t yEnd = std::min(groupSizeY, ysize - groupY);
    width = std::min(groupSizeX, xsize - groupX) - 1;
    size_t area = yEnd * (width + 1);
    maxerrShift =
        (area > 25600
             ? 0
             : area > 12800 ? 1 : area > 4000 ? 2 : area > 400 ? 3 : 4);
    maxTpv =
        (planeToCompress == planeToUse ? numColors[planeToCompress] - 1 : 255)
        << PBits;

    double fromN = 0, fromW = 0;
    for (size_t y = 1; y < yEnd; ++y) {
      rowImg = img->PlaneRow(planeToCompress, groupY + y) + groupX;
      rowPrev = img->PlaneRow(planeToCompress, groupY + y - 1) + groupX;
      uint32_t fromNx = 0, fromWx = 0;
      for (size_t x = 1; x <= width; ++x) {
        int c = rowImg[x];
        int N = rowPrev[x];
        int W = rowImg[x - 1];
        N -= c;
        W -= c;
        fromNx += N * N;
        fromWx += W * W;
      }
      fromN += fromNx;
      fromW += fromWx;
    }
    PredictMode pMode = PM_Regular;
    if (fromW * 5 < fromN * 4) pMode = PM_West;  // no 'else' to reduce codesize
    if (fromN * 5 < fromW * 4) pMode = PM_North;  // if (fromN < fromW*0.8)
    // printf("%c ", pMode);

    if (pMode == PM_Regular) {  // Regular mode
      for (size_t y = 0, yc = 0, yp; y < yEnd; ++y) {
        uint8_t const* PIK_RESTRICT rowUse;
        setRowImgPointers3(rowUse, img->PlaneRow);
        for (size_t x = 0; x <= width; ++x) {
          int maxErr, prediction = predict_R_(x, yc + x - 1, yp + x, &maxErr);
          AfterPredictWhenCompressing3;
        }
      }
    } else if (pMode == PM_West) {  // 'West predicts better' mode
      for (size_t y = 0, yc = 0, yp; y < yEnd; ++y) {
        uint8_t const* PIK_RESTRICT rowUse;
        setRowImgPointers3(rowUse, img->PlaneRow);
        for (size_t x = 0; x <= width; ++x) {
          int maxErr, prediction = predict_W_(x, yc + x - 1, yp + x, &maxErr);
          AfterPredictWhenCompressing3;
        }
      }
    } else if (pMode == PM_North) {  // 'North predicts better' mode
      for (size_t y = 0, yc = 0, yp; y < yEnd; ++y) {
        uint8_t const* PIK_RESTRICT rowUse;
        setRowImgPointers3(rowUse, img->PlaneRow);
        for (size_t x = 0; x <= width; ++x) {
          int maxErr, prediction = predict_N_(x, yc + x - 1, yp + x, &maxErr);
          AfterPredictWhenCompressing3;
        }
      }
    } else {
    }  // TODO: other prediction modes!

    size_t pos = 0;
    compressedOutput[pos++] = pMode;
    int nC = ((kNumContexts - 1) >> maxerrShift) + 1;
    std::vector<size_t> sizes(nC);
    std::vector<const uint8_t*> streams(nC);
    for (int i = 0; i < nC; ++i) {
      sizes[i] = esize[i];
      streams[i] = &edata[i][0];
    }
    if (!CompressWithEntropyCode(&pos, sizes.data(), streams.data(),
                                 sizes.size(), compressedCapacity,
                                 &compressedOutput[0])) {
      return PIK_FAILURE("lossless8 entropy encode failed");
    }
    *csize = pos;
    return true;
  }

  bool Colorful8bit_compress(const Image3B& img_in, pik::PaddedBytes* bytes) {
    // The code modifies the image for palette so must copy for now.
    Image3B img = CopyImage(img_in);

    std::vector<uint8_t> temp_buffer(groupSize2plus * 6);
    compressedData = temp_buffer.data();

    size_t xsize = img.xsize(), ysize = img.ysize(), pos;
    pos = EncodeVarInt(xsize, &compressedData[0]);
    pos += EncodeVarInt(ysize, &compressedData[pos]);
    FWr(&compressedData[0], pos);
    numColors[0] = numColors[1] = numColors[2] = 0x100;

    if (xsize * ysize > 4 * 0x100) {  // TODO: smarter decision making here
      // Let's check whether the image should be 'palettized',
      // because the range is 64k, but 25% or more of the range is unused.
      uint8_t flags = 0, bits[3 * 0x100 / 8], *pb = &bits[0];
      uint32_t palette123[3 * 0x100];

#if 1  // Enable/disable the CompactChannel transform(per-channel palettization)
      memset(bits, 0, sizeof(bits));
      memset(palette123, 0, sizeof(palette123));
      for (int channel = 0; channel < 3; ++channel) {
        uint32_t i, first, count, *palette = &palette123[0x100 * channel];
        for (size_t y = 0; y < ysize; ++y) {
          uint8_t* const PIK_RESTRICT rowImg = img.PlaneRow(channel, y);
          for (size_t x = 0; x < xsize; ++x)  // UNROLL AND PARALLELIZE ME!
            palette[rowImg[x]] = 1;
        }
        // count the number of pixel values present in the image
        for (i = 0; i < 0x100; ++i)
          if (palette[i]) break;
        for (first = i, count = 0; i < 0x100; ++i)
          if (palette[i]) palette[i] = count++;
        // printf("count=%5d, %f%%\n", count, count * 100. / 256);
        numColors[channel] = count;
        if (count >= 255) continue;  // TODO: better decision making
        flags += 1 << channel;
        palette[first] = 1;
        for (int sb = 0, x = 0; x < 0x100; x += 8) {
          uint32_t b = 0, v;
          for (int y = x + 7; y >= x; --y)
            v = (palette[y] ? 1 : 0), b += b + v, sb += v;
          *pb++ = b;  // TODO: Compress the bits, not store!
          if (sb >= count || sb + 0x100 - 8 - x == count) break;
        }
        palette[first] = 0;
      }  // for channel
#endif

      FWrByte(flags);  // As of now (Feb.2019) ImageMethod==flags
      if (flags) {
        for (int channel = 0; channel < 3; ++channel) {
          if (flags & (1 << channel)) {
            uint32_t* palette = &palette123[0x100 * channel];
            for (size_t y = 0; y < ysize; ++y) {
              uint8_t* const PIK_RESTRICT rowImg = img.PlaneRow(channel, y);
              for (size_t x = 0; x < xsize; ++x)  // UNROLL AND PARALLELIZE ME!
                rowImg[x] = palette[rowImg[x]];
            }
          }
        }
        compressedData[0] = numColors[0] - 1;
        compressedData[1] = numColors[1] - 1;
        compressedData[2] = numColors[2] - 1;
        FWr(&compressedData[0], 3);
        FWr(&bits[0], sizeof(uint8_t) * (pb - &bits[0]));
      } else {
        numColors[0] = numColors[1] = numColors[2] = 0x100;
      }
    }  // if (xsize*ysize > 4*0x100)

    compress3planes(uint8_t);

    return true;
  }

};  // struct State

}  // namespace

static constexpr size_t kGroupSize = 512;

bool Grayscale8bit_compress(const ImageB& img, PaddedBytes* bytes) {
  std::unique_ptr<State> state(new State(kGroupSize, kGroupSize));
  return state->Grayscale8bit_compress(img, bytes);
}

bool Grayscale8bit_decompress(const Span<const uint8_t> bytes, size_t* pos,
                              ImageB* result) {
  std::unique_ptr<State> state(new State(kGroupSize, kGroupSize));
  return state->Grayscale8bit_decompress(bytes, pos, result);
}

bool Colorful8bit_compress(const Image3B& img, PaddedBytes* bytes) {
  std::unique_ptr<State> state(new State(kGroupSize, kGroupSize));
  return state->Colorful8bit_compress(img, bytes);
}

bool Colorful8bit_decompress(const Span<const uint8_t> bytes, size_t* pos,
                             Image3B* result) {
  std::unique_ptr<State> state(new State(kGroupSize, kGroupSize));
  return state->Colorful8bit_decompress(bytes, pos, result);
}

}  // namespace pik
