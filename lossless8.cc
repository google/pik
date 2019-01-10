// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

// @author Alexander Rhatushnyak

#include "lossless8.h"

#include <cmath>

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "entropy_coder.h"

namespace pik {

namespace {

size_t encodeVarInt(size_t value, uint8_t* output) {
  size_t outputSize = 0;
  // While more than 7 bits of data are left,
  // store 7 bits and set the next byte flag
  while (value > 127) {
    // |128: Set the next byte flag
    output[outputSize++] = ((uint8_t)(value & 127)) | 128;
    // Remove the seven bits we just wrote
    value >>= 7;
  }
  output[outputSize++] = ((uint8_t)value) & 127;
  return outputSize;
}

size_t decodeVarInt(const uint8_t* input, size_t inputSize, size_t* pos) {
  size_t i, ret = 0;
  for (i = 0; *pos + i < inputSize && i < 10; ++i) {
    ret |= uint64_t(input[*pos + i] & 127) << uint64_t(7 * i);
    // If the next-byte flag is not set, stop
    if ((input[*pos + i] & 128) == 0) break;
  }
  // TODO: Return a decoding error if i == 10.
  *pos += i + 1;
  return ret;
}

// Entropy encode with pik ANS
// TODO(user): move this to ans_encode.h
bool EntropyEncode(const uint8_t* data, size_t size,
                   std::vector<uint8_t>* result) {
  static const int kAlphabetSize = 256;
  static const int kContext = 0;

  std::vector<int> histogram(kAlphabetSize, 0);
  for (size_t i = 0; i < size; i++) {
    histogram[data[i]]++;
  }
  size_t cost_bound =
      1000 + 4 * size + 8 +
      ((size_t)ANSPopulationCost(histogram.data(), kAlphabetSize, size) + 7) /
          8;
  result->resize(cost_bound, 0);

  uint8_t* storage = result->data();
  size_t pos = 0;

  pos += encodeVarInt(size, storage + pos);

  std::vector<ANSEncodingData> encoding_codes(1);
  size_t bitpos = 0;
  encoding_codes[0].BuildAndStore(&histogram[0], histogram.size(), &bitpos,
                                  storage + pos);

  std::vector<uint8_t> dummy_context_map;
  dummy_context_map.push_back(0);  // only 1 histogram
  ANSSymbolWriter writer(encoding_codes, dummy_context_map, &bitpos,
                         storage + pos);
  for (size_t i = 0; i < size; i++) {
    writer.VisitSymbol(data[i], kContext);
  }
  writer.FlushToBitStream();
  pos += ((bitpos + 7) >> 3);
  result->resize(pos);

  return true;
}

// Entropy decode with pik ANS
// TODO(user): move this to ans_decode.h
bool EntropyDecode(const uint8_t* data, size_t size,
                   std::vector<uint8_t>* result) {
  static const int kContext = 0;
  size_t pos = 0;
  size_t num_symbols = decodeVarInt(data, size, &pos);
  if (pos >= size) {
    return PIK_FAILURE("lossless8");
  }
  // TODO(user): instead take expected decoded size as function parameter
  if (num_symbols > 16777216) {
    // Avoid large allocations, we never expect this many symbols for
    // the limited group sizes.
    return PIK_FAILURE("lossless8");
  }

  BitReader br(data + pos, size - pos);
  ANSCode codes;
  if (!DecodeANSCodes(1, 256, &br, &codes)) {
    return PIK_FAILURE("lossless8");
  }

  result->resize(num_symbols);
  ANSSymbolReader reader(&codes);
  for (size_t i = 0; i < num_symbols; i++) {
    br.FillBitBuffer();
    int read_symbol = reader.ReadSymbol(kContext, &br);
    (*result)[i] = read_symbol;
  }
  if (!reader.CheckANSFinalState()) {
    return PIK_FAILURE("lossless8");
  }

  return true;
}

static bool IsRLECompressible(const uint8_t* data, size_t size) {
  if (size < 4) return false;
  uint8_t first = data[0];
  for (size_t i = 1; i < size; i++) {
    if (data[i] != first) return false;
  }
  return true;
}

// TODO(user): avoid the copying between std::vector and data.
// Entropy encode with pik ANS
static bool EntropyEncode(const uint8_t* data, size_t size, size_t out_capacity,
                          uint8_t* out, size_t* out_size) {
  if (IsRLECompressible(data, size)) {
    *out_size = 1;  // Indicate the codec should use RLE instead,
    return true;
  }
  std::vector<uint8_t> result;
  if (!EntropyEncode(data, size, &result)) {
    return PIK_FAILURE("lossless8");  // Encoding error
  }
  if (result.size() > size) {
    *out_size = 0;  // Indicate the codec should use uncompressed mode instead.
    return true;
  }
  if (result.size() > out_capacity) {
    return PIK_FAILURE("lossless8");  // Error: not enough capacity
  }
  memcpy(out, result.data(), result.size());
  *out_size = result.size();
  return true;
}

// Entropy decode with pik ANS
static bool EntropyDecode(const uint8_t* data, size_t size, size_t out_capacity,
                          uint8_t* out, size_t* out_size) {
  std::vector<uint8_t> result;
  if (!EntropyDecode(data, size, &result)) {
    return PIK_FAILURE("lossless8");  // Decoding error
  }
  if (result.size() > out_capacity) {
    return PIK_FAILURE("lossless8");  // Error: not enough capacity
  }
  memcpy(out, result.data(), result.size());
  *out_size = result.size();
  return true;
}

static const int mulWeights0and1_R_[] = {
    34, 36,  // when errors are small,
    31, 37,  // we assume they are random noise,
    33, 37,  // and penalize predictors 0 and 1
    36, 40, 39, 44, 42, 46, 43, 47, 43, 42,
};

static const int mulWeights3teNE_R_[] = {
    28, 0, 24, 15, 24, 19, 24, 16, 23, 12, 23, 12, 25, 11, 32, 11,
};

static const int mulWeights0and1_W_[] = {
    27, 31,  // when errors are small,
    33, 31,  // we assume they are random noise,
    40, 34,  // and penalize predictors 0 and 1
    43, 36, 52, 43, 59, 45, 63, 43, 65, 28,
};

static const int mulWeights3teNE_W_[] = {
    31, 0, 31, 21, 29, 19, 28, 13, 26, 14, 28, 24, 32, 26, 43, 35,
};

static const int mulWeights0and1_N_[] = {
    43, 23,  // when errors are small,
    38, 21,  // we assume they are random noise,
    35, 24,  // and penalize predictors 0 and 1
    34, 27, 35, 29, 33, 31, 28, 31, 23, 31,
};

static const int mulWeights3teNE_N_[] = {
    27, 0, 23, 29, 26, 34, 29, 29, 30, 13, 35, 13, 40, 11, 51, 9,
};

static const int WITHSIGN = 7, NUMCONTEXTS = 8 + WITHSIGN + 2, kGroupSize = 512,
                 kGroupSize2plus = kGroupSize * kGroupSize * 9 / 8;
static const int MAXERROR = 101, MaxSumErrors = MAXERROR * 7 + 1, NumRuns = 1;

// TODO(user): split state variables needed for encoder from those for decoder
//             and perform one-time global initialization where possible.
struct State {
  const int PBits = 3,  // SET ME TO ZERO FOR A FASTER VERSION WITH NO ROUNDING!
      toRound = ((1 << PBits) >> 1), toRound_m1 = (toRound ? toRound - 1 : 0);
  typedef enum { PM_Regular, PM_West, PM_North } PredictMode;

  // uint64_t gqe[NUMCONTEXTS];  //global quantized errors (all groups)
  // frequencies

  uint8_t edata[NUMCONTEXTS]
               [kGroupSize * kGroupSize],  // size can be *2/NUMCONTEXTS in the
                                           // Production edition
      compressedDataTmpBuf[kGroupSize2plus], *compressedData;
  uint8_t
      errors0[kGroupSize * 2 + 4];  // Errors of predictor 0. Range 0..MAXERROR
  uint8_t errors1[kGroupSize * 2 + 4];     // Errors of predictor 1
  uint8_t errors2[kGroupSize * 2 + 4];     // Errors of predictor 2
  uint8_t errors3[kGroupSize * 2 + 4];     // Errors of predictor 3
  int16_t trueErr[kGroupSize * 2];         // Their range is -255...255
  uint8_t quantizedError[kGroupSize * 2];  // The range is 0...14, all are
                                           // even due to quantizedInit()

#ifdef SIMPLE_signToLSB_TRANSFORM  // to fully disable, "=i;" in the init macros

  uint8_t signLSB_forwardTransform[256],
      signLSB_backwardTransform[256];  // const
#define ToLSB_FRWRD signLSB_forwardTransform[err & 255]
#define ToLSB_BKWRD (prediction - signLSB_backwardTransform[err]) & 255

#define signToLSB_FORWARD_INIT  \
  for (int i = 0; i < 256; ++i) \
    signLSB_forwardTransform[i] = (i & 128 ? (255 - i) * 2 + 1 : i * 2);

#define signToLSB_BACKWARD_INIT \
  for (int i = 0; i < 256; ++i) \
    signLSB_backwardTransform[i] = (i & 1 ? 255 - (i >> 1) : i >> 1);

#else
  uint8_t signLSB_forwardTransform[1 << 16], signLSB_backwardTransform[1 << 16];
#define ToLSB_FRWRD signLSB_forwardTransform[prediction * 256 + truePixelValue]
#define ToLSB_BKWRD \
  signLSB_backwardTransform[((prediction + toRound_m1) >> PBits) * 256 + q]

#define signToLSB_FORWARD_INIT                                               \
  for (int p = 0; p < 256; ++p) {                                            \
    signLSB_forwardTransform[p * 256 + p] = 0;                               \
    for (int v, top = p, btm = p, d = 1; d < 256; ++d) {                     \
      v = (d & 1 ? (btm > 0 ? --btm : ++top) : (top < 255 ? ++top : --btm)); \
      signLSB_forwardTransform[p * 256 + v] = d;                             \
    }                                                                        \
  }

#define signToLSB_BACKWARD_INIT                                              \
  for (int p = 0; p < 256; ++p) {                                            \
    signLSB_backwardTransform[p * 256] = p;                                  \
    for (int v, top = p, btm = p, d = 1; d < 256; ++d) {                     \
      v = (d & 1 ? (btm > 0 ? --btm : ++top) : (top < 255 ? ++top : --btm)); \
      signLSB_backwardTransform[p * 256 + d] = v;                            \
    }                                                                        \
  }
#endif

  uint8_t quantizedTable[256], diff2error[512 * 2];  // const
  uint16_t error2weight[MaxSumErrors];               // const

  State() {
    for (int j = 0; j < MaxSumErrors; ++j)
      error2weight[j] =
          150 * 512 / (58 + j * std::sqrt(j + 50));  // const init!  150 58 50

    for (int j = -512; j <= 511; ++j)
      diff2error[512 + j] = std::min(j < 0 ? -j : j, MAXERROR);  // const init!
    for (int j = 0; j <= 255; ++j)
      quantizedTable[j] = quantizedInit(j);  // const init!
    // for (int i=0; i < 512; i += 16, printf("\n"))
    //   for (int j=i; j < i + 16; ++j)  printf("%2d, ", quantizedTable[j]);
    signToLSB_FORWARD_INIT       // const init!
        signToLSB_BACKWARD_INIT  // const init!
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

  int prediction0,
      prediction1,  // Their range is -255...510 rather than 0...255!
      prediction2,
      prediction3;  // And -510..510 after subtracting truePixelValue
  int maxerrShift, pb255, width;  //  width-1  actually

  uint8_t* PIK_RESTRICT rowImg;
  uint8_t const *PIK_RESTRICT rowPrev, *PIK_RESTRICT rowPP;

  PIK_INLINE int predictY0(size_t x, size_t yc, size_t yp, int* maxErr) {
    *maxErr = (x == 0 ? NUMCONTEXTS - 3
                      : x == 1 ? quantizedError[yc]
                               : std::max(quantizedError[yc],
                                          quantizedError[yc - 1]));
    prediction1 = prediction2 = prediction3 = (x > 0 ? rowImg[x - 1] : 27)
                                              << PBits;
    prediction0 =
        (x <= 1 ? prediction1
                : prediction1 +
                      ((rowImg[x - 1] - rowImg[x - 2]) << PBits) * 5 / 16);
    return (prediction0 < 0 ? 0 : prediction0 > pb255 ? pb255 : prediction0);
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
    if (x == 0LL)
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
        mE = ((sumWN | teNE | teNW) == 0 ? NUMCONTEXTS - 1 : 1);
    }
    *maxErr = mE;

    prediction0 = W - (sumWN + teNW) / 4;  // 7/32 works better than 1/4 ?
    prediction1 =
        N - (sumWN + teNE) / 4;  // predictors 0 & 1 rely on true errors
    prediction2 = W + NE - N;
    int t = (teNE * 3 + teNW * 4 + 7) >> 5;
    prediction3 = N + (N - (rowPP[x] << PBits)) * 23 / 32 + (W - NW) / 16 - t;
    assert(-255 << PBits <= prediction0 && prediction0 <= 510 << PBits);
    assert(-255 << PBits <= prediction1 && prediction1 <= 510 << PBits);
    assert(-255 << PBits <= prediction2 && prediction2 <= 510 << PBits);
    assert(-255 << PBits <= prediction3 && prediction3 <= 510 << PBits);

    int sumWeights = weight0 + weight1 + weight2 + weight3;
    // assert(sumWeights>0);  // true if min(error2weight)*min(mulWeights**_R_)
    // > 0

    int prediction = (prediction0 * weight0 + prediction1 * weight1 +
                      (sumWeights >> 3) + prediction2 * weight2 +
                      prediction3 * weight3) /  // biased rounding: >>3
                     sumWeights;

    if (((teN ^ teW) | (teN ^ teNW)) > 0)  // if all three have the same sign
      return (prediction < 0 ? 0 : prediction > pb255 ? pb255 : prediction);

    int max = (W > N ? W : N);
    int min = W + N - max;
    if (NE > max) max = NE;
    if (NE < min) min = NE;
    return (prediction < min ? min : prediction > max ? max : prediction);
  }

  PIK_INLINE int predict_W_(size_t x, size_t yc, size_t yp, int* maxErr) {
    if (!rowPrev)
      return predictY0(x, yc, yp, maxErr);  // OK for Prototype edition
    if (x == 0LL)
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
        mE = ((sumWN | teNE | teNW) == 0 ? NUMCONTEXTS - 1 : 1);
    }
    *maxErr = mE;

    prediction0 =
        W - (sumWN + teNW) * 9 / 32;  // pr's 0 & 1 rely on true errors
    prediction1 =
        N - (sumWN + teNE) * 171 / 512;  // clamping not needed, is it?
    prediction2 = W + NE - N;
    prediction3 =
        N + ((N - (rowPP[x] << PBits)) >> 1) + ((W - NW) * 19 - teNW * 13) / 64;
    assert(-255 << PBits <= prediction0 && prediction0 <= 510 << PBits);
    assert(-255 << PBits <= prediction1 && prediction1 <= 510 << PBits);
    assert(-255 << PBits <= prediction2 && prediction2 <= 510 << PBits);
    assert(-255 << PBits <= prediction3 && prediction3 <= 510 << PBits);

    int sumWeights = weight0 + weight1 + weight2 + weight3;
    // assert(sumWeights>0);  // true if min(error2weight)*min(mulWeights**_W_)
    // > 0

    int prediction =
        (prediction0 * weight0 + prediction1 * weight1 + (sumWeights >> 1) +
         prediction2 * weight2 + prediction3 * weight3) /
        sumWeights;

    if (((teN ^ teW) | (teN ^ teNE)) > 0)  // if all three have the same sign
      return (prediction < 0 ? 0 : prediction > pb255 ? pb255 : prediction);

    int max = (W > N ? W : N);
    int min = W + N - max;
    if (NE > max) max = NE;
    if (NE < min) min = NE;
    return (prediction < min ? min : prediction > max ? max : prediction);
  }

  PIK_INLINE int predict_N_(size_t x, size_t yc, size_t yp, int* maxErr) {
    if (!rowPrev)
      return predictY0(x, yc, yp, maxErr);  // OK for Prototype edition
    if (x == 0LL)
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
        mE = ((sumWN | teNE | teNW) == 0 ? NUMCONTEXTS - 1 : 1);
    }
    *maxErr = mE;

    prediction0 = N - (sumWN + teNW + teNE) / 4;  // if bigger than 1/4,
                                                  // clamping would be needed!
    prediction1 =
        W - ((teW * 2 + teNW) >> 2);  // pr's 0 & 1 rely on true errors
    prediction2 = W + NE - N;
    prediction3 = N + ((N - (rowPP[x] << PBits)) * 47) / 64 - (teN >> 2);
    assert(-255 << PBits <= prediction0 && prediction0 <= 510 << PBits);
    assert(-255 << PBits <= prediction1 && prediction1 <= 510 << PBits);
    assert(-255 << PBits <= prediction2 && prediction2 <= 510 << PBits);
    assert(-255 << PBits <= prediction3 && prediction3 <= 510 << PBits);

    int sumWeights = weight0 + weight1 + weight2 + weight3;
    // assert(sumWeights>0);  // true if min(error2weight)*min(mulWeights**_N_)
    // > 0

    int prediction =
        (prediction0 * weight0 + prediction1 * weight1 + (sumWeights >> 1) +
         prediction2 * weight2 + prediction3 * weight3) /
        sumWeights;

    if (((teN ^ teW) | (teN ^ teNE)) > 0)  // if all three have the same sign
      return (prediction < 0 ? 0 : prediction > pb255 ? pb255 : prediction);

    int max = (W > N ? W : N);
    int min = W + N - max;
    if (NE > max) max = NE;
    if (NE < min) min = NE;
    return (prediction < min ? min : prediction > max ? max : prediction);
  }

#define Update_Size_And_Errors                                  \
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
      (errors3[yc + x] = dp[(prediction3 + toRound) >> PBits]);

#define AfterPredictWhenCompressing                 \
  maxErr >>= maxerrShift;                           \
  assert(0 <= maxErr && maxErr <= NUMCONTEXTS - 1); \
  int q, truePixelValue = rowImg[x];                \
  int err = prediction - (truePixelValue << PBits); \
  size_t s = esize[maxErr];                         \
  prediction = (prediction + toRound_m1) >> PBits;  \
  assert(0 <= prediction && prediction <= 255);     \
  edata[maxErr][s] = q = ToLSB_FRWRD;               \
  Update_Size_And_Errors  // ++gqe[maxErr];

#define AfterPredictWhenCompressing3                \
  maxErr >>= maxerrShift;                           \
  assert(0 <= maxErr && maxErr <= NUMCONTEXTS - 1); \
  int q, truePixelValue = rowImg[x];                \
  if (planeToCompress != planeToUse) {              \
    truePixelValue -= (int)rowUse[x] - 0x80;        \
    truePixelValue &= 0xff;                         \
    rowImg[x] = truePixelValue;                     \
  }                                                 \
  int err = prediction - (truePixelValue << PBits); \
  size_t s = esize[maxErr];                         \
  prediction = (prediction + toRound_m1) >> PBits;  \
  assert(0 <= prediction && prediction <= 255);     \
  edata[maxErr][s] = q = ToLSB_FRWRD;               \
  Update_Size_And_Errors  // ++gqe[maxErr];

#define AfterPredictWhenDecompressing                          \
  maxErr >>= maxerrShift;                                      \
  assert(0 <= maxErr && maxErr <= NUMCONTEXTS - 1);            \
  assert(0 <= prediction && prediction <= 255 << PBits);       \
  size_t s = esize[maxErr];                                    \
  int err, q = edata[maxErr][s], truePixelValue = ToLSB_BKWRD; \
  rowImg[x] = truePixelValue;                                  \
  err = prediction - (truePixelValue << PBits);                \
  Update_Size_And_Errors

#define setRowImgPointers(imgRow)                              \
  yc ^= kGroupSize, yp = kGroupSize - yc;                      \
  rowImg = imgRow(groupY + y) + groupX;                        \
  rowPrev = (y == 0 ? NULL : imgRow(groupY + y - 1) + groupX); \
  rowPP = (y <= 1 ? rowPrev : imgRow(groupY + y - 2) + groupX);

#define setRowImgPointers3(imgRow)                                        \
  yc ^= kGroupSize, yp = kGroupSize - yc;                                 \
  uint8_t const* PIK_RESTRICT rowUse;                                     \
  rowImg = imgRow(planeToCompress, groupY + y) + groupX;                  \
  rowUse = imgRow(planeToUse, groupY + y) + groupX;                       \
  rowPrev =                                                               \
      (y == 0 ? NULL : imgRow(planeToCompress, groupY + y - 1) + groupX); \
  rowPP = (y <= 1 ? rowPrev : imgRow(planeToCompress, groupY + y - 2) + groupX);

#define setRowImgPointers3dec(imgRow)                                       \
  yc ^= kGroupSize, yp = kGroupSize - yc;                                   \
  rowImg = imgRow(planeToDecompress, groupY + y) + groupX;                  \
  rowPrev =                                                                 \
      (y == 0 ? NULL : imgRow(planeToDecompress, groupY + y - 1) + groupX); \
  rowPP =                                                                   \
      (y <= 1 ? rowPrev : imgRow(planeToDecompress, groupY + y - 2) + groupX);

  bool Grayscale8bit_compress(const ImageB& img_in, pik::PaddedBytes* bytes) {
    clock_t start = clock();

    // The code modifies the image for palette so must copy for now.
    ImageB img = CopyImage(img_in);

    size_t esize[NUMCONTEXTS], xsize = img.xsize(), ysize = img.ysize();
    std::vector<uint8_t> temp_buffer(kGroupSize2plus);
    compressedData = temp_buffer.data();

    for (int run = 0; run < NumRuns; ++run) {
      int freqs[256];
      memset(freqs, 0, sizeof(freqs));
      for (size_t y = 0; y < ysize; ++y) {
        uint8_t* const PIK_RESTRICT rowImg = img.Row(y);
        for (size_t x = 0; x < xsize; ++x)  // UNROLL and PARALLELIZE ME!
          ++freqs[rowImg[x]];  // They can also be used for guessing
                               // photo/nonphoto
      }
      int palette[256], count = 0;
      for (int i = 0; i < 256; ++i)
        palette[i] = count, count += (freqs[i] ? 1 : 0);
      int havePalette = (count < 255 ? 1 : 0);  // 255? or 256?
      pb255 = (havePalette ? std::min(255, count + 1) : 255) << PBits;

      if (havePalette)
        for (size_t y = 0; y < ysize; ++y) {
          uint8_t* const PIK_RESTRICT rowImg = img.Row(y);
          for (size_t x = 0; x < xsize; ++x)  // UNROLL and PARALLELIZE ME!
            rowImg[x] = palette[rowImg[x]];
        }

      for (size_t groupY = 0; groupY < ysize; groupY += kGroupSize) {
        for (size_t groupX = 0; groupX < xsize; groupX += kGroupSize) {
          memset(esize, 0, sizeof(esize));
          size_t yEnd = std::min((size_t)kGroupSize, ysize - groupY);
          width = std::min((size_t)kGroupSize, xsize - groupX) - 1;
          size_t area = yEnd * (width + 1);
          maxerrShift =
              (area > 25600
                   ? 0
                   : area > 12800 ? 1 : area > 4000 ? 2 : area > 400 ? 3 : 4);

          uint64_t fromN = 0, fromW = 0;
          for (size_t y = 1; y < yEnd; ++y) {
            rowImg = img.Row(groupY + y) + groupX;
            rowPrev = img.Row(groupY + y - 1) + groupX;
            for (size_t x = 1; x <= width; ++x) {
              int c = rowImg[x];
              int N = rowPrev[x];
              int W = rowImg[x - 1];
              N -= c;
              W -= c;
              fromN += N * N;
              fromW += W * W;
            }
          }
          PredictMode pMode = PM_Regular;
          if (fromW * 5 < fromN * 4)
            pMode = PM_West;  // no 'else' to reduce codesize
          if (fromN * 5 < fromW * 4)
            pMode = PM_North;  // if (fromN < fromW*0.8)
          // printf("%c ", pMode);

          if (pMode == PM_Regular)  // Regular mode
            for (size_t y = 0, yc = 0, yp; y < yEnd; ++y) {
              setRowImgPointers(img.Row) for (size_t x = 0; x <= width; ++x) {
                int maxErr,
                    prediction = predict_R_(x, yc + x - 1, yp + x, &maxErr);
                AfterPredictWhenCompressing
              }
            }
          else if (pMode == PM_West)  // 'West predicts better' mode
            for (size_t y = 0, yc = 0, yp; y < yEnd; ++y) {
              setRowImgPointers(img.Row) for (size_t x = 0; x <= width; ++x) {
                int maxErr,
                    prediction = predict_W_(x, yc + x - 1, yp + x, &maxErr);
                AfterPredictWhenCompressing
              }
            }
          else if (pMode == PM_North)  // 'North predicts better' mode
            for (size_t y = 0, yc = 0, yp; y < yEnd; ++y) {
              setRowImgPointers(img.Row) for (size_t x = 0; x <= width; ++x) {
                int maxErr,
                    prediction = predict_N_(x, yc + x - 1, yp + x, &maxErr);
                AfterPredictWhenCompressing
              }
            }
          else {
          }  // TODO: other prediction modes!

          size_t pos = 0;
          if (groupY == 0 && groupX == 0) {
            pos += encodeVarInt(xsize * 2 + havePalette, &compressedData[pos]);
            pos += encodeVarInt(ysize, &compressedData[pos]);
            if (havePalette) {  // Save bit 1 if color is present, bit 0 if not
              const int kBitsPerByte = 8;
              for (int i = 0; i < 256 / kBitsPerByte; ++i) {
                int code = 0;
                for (int j = kBitsPerByte - 1; j >= 0; --j)
                  code =
                      code * 2 + (freqs[i * 8 + j] ? 1 : 0);  // color=YES bits
                compressedData[pos++] = code;
              }  // for i
            }    // if (havePalette)
          }      // if (groupY...)
          int nC = ((NUMCONTEXTS - 1) >> maxerrShift) + 1;
          for (int i = 0; i < nC; ++i) {
            if (esize[i]) {
              // size_t cs = FSE_compress(&compressedDataTmpBuf[0],
              // sizeof(compressedDataTmpBuf), &edata[i][0], esize[i]);
              size_t cs;
              if (!EntropyEncode(&edata[i][0], esize[i],
                                 sizeof(compressedDataTmpBuf),
                                 &compressedDataTmpBuf[0], &cs)) {
                return PIK_FAILURE("lossless8");
              }
              size_t s = (cs <= 1 ? (esize[i] - 1) * 3 + 1 + cs : cs * 3);
              pos +=
                  encodeVarInt(i > 0 ? s : s * 3 + pMode, &compressedData[pos]);
              if (cs == 1)
                compressedData[pos++] = edata[i][0];
              else if (cs == 0)
                memcpy(&compressedData[pos], &edata[i][0], esize[i]),
                    pos += esize[i];
              else
                memcpy(&compressedData[pos], &compressedDataTmpBuf[0], cs),
                    pos += cs;
            } else
              pos += encodeVarInt(i > 0 ? 0 : pMode, &compressedData[pos]);
          }  // i
          if (run == 0) {
            size_t current = bytes->size();
            bytes->resize(bytes->size() + pos);
            memcpy(bytes->data() + current, &compressedData[0], pos);
          }
        }  // groupX
      }    // groupY
    }      // run
    // for (int i=0; i<NUMCONTEXTS; ++i) printf("%3d
    // ",gqe[i]*1000/(xsize*ysize)); printf("\n");

    if (NumRuns > 1)
      printf("%d runs, %1.5f seconds", NumRuns,
             ((double)clock() - start) / CLOCKS_PER_SEC);
    return true;
  }

  bool Grayscale8bit_decompress(const PaddedBytes& bytes, size_t* bytes_pos,
                                ImageB* result) {
    if (*bytes_pos > bytes.size()) return PIK_FAILURE("lossless8");
    size_t compressedSize = bytes.size() - *bytes_pos;
    const uint8_t* compressedData = bytes.data() + *bytes_pos;

    size_t maxDecodedSize = kGroupSize * kGroupSize;  // Size of an edata entry

    clock_t start = clock();
    size_t esize[NUMCONTEXTS], xsize, ysize, pos = 0;
    xsize = decodeVarInt(compressedData, compressedSize, &pos);
    ysize = decodeVarInt(compressedData, compressedSize, &pos);
    int havePalette = xsize & 1, count = 256, palette[256];
    if (havePalette) {
      const uint8_t* p = &compressedData[pos];
      pos += 32;
      if (pos >= compressedSize) return PIK_FAILURE("lossless8");
      count = 0;
      for (int i = 0; i < 256; ++i)
        if (p[i >> 3] & (1 << (i & 7))) palette[count++] = i;
    }
    pb255 = std::min(255, count + 1) << PBits;
    xsize >>= 1;
    if (!xsize || !ysize) return PIK_FAILURE("lossless8");
    // Too large, would run out of memory. Chosen as reasonable limit for pik
    // while being below default fuzzer memory limit. We check for total pixel
    // size, and an additional restriction to ysize, because large ysize
    // consumes more memory due to the scanline padding.
    if (uint64_t(xsize) * uint64_t(ysize) >= 268435456ull || ysize >= 65536) {
      return PIK_FAILURE("lossless8");
    }
    pik::ImageB img(xsize, ysize);

    for (int run = 0; run < NumRuns; ++run) {
      if (NumRuns > 1) pos = 0;
      for (size_t groupY = 0; groupY < ysize; groupY += kGroupSize) {
        for (size_t groupX = 0; groupX < xsize; groupX += kGroupSize) {
          size_t yEnd = std::min((size_t)kGroupSize, ysize - groupY);
          width = std::min((size_t)kGroupSize, xsize - groupX) - 1;
          size_t area = yEnd * (width + 1);
          maxerrShift =
              (area > 25600
                   ? 0
                   : area > 12800 ? 1 : area > 4000 ? 2 : area > 400 ? 3 : 4);
          size_t decompressedSize = 0;  // is used only for the assert()

          if (NumRuns > 1 && groupY == 0 && groupX == 0) {
            decodeVarInt(compressedData, compressedSize,
                         &pos);  // just skip them
            decodeVarInt(compressedData, compressedSize, &pos);
            if (havePalette) pos += 32;
          }
          PredictMode pMode;
          int nC = ((NUMCONTEXTS - 1) >> maxerrShift) + 1;
          for (int i = 0; i < nC; ++i) {
            size_t cs = decodeVarInt(compressedData, compressedSize, &pos);
            if (i == 0) pMode = (PredictMode)(cs % 3), cs /= 3;
            if (cs == 0) continue;
            int mode = cs % 3;
            cs /= 3;
            if (mode == 2) {
              if (pos >= compressedSize) return PIK_FAILURE("lossless8");
              if (cs > maxDecodedSize) return PIK_FAILURE("lossless8");
              memset(&edata[i][0], compressedData[pos++], ++cs),
                  decompressedSize += cs;
            } else if (mode == 1) {
              if (pos + cs > compressedSize) return PIK_FAILURE("lossless8");
              if (cs > maxDecodedSize) return PIK_FAILURE("lossless8");
              memcpy(&edata[i][0], &compressedData[pos], ++cs),
                  decompressedSize += cs, pos += cs;
            } else {
              if (pos + cs > compressedSize) return PIK_FAILURE("lossless8");
              size_t ds;
              if (!EntropyDecode(&compressedData[pos], cs, maxDecodedSize,
                                 &edata[i][0], &ds)) {
                return PIK_FAILURE("lossless8");
              }
              pos += cs;
              decompressedSize += ds;
            }
          }
          if (decompressedSize != area) return PIK_FAILURE("lossless8");
          if (groupY + kGroupSize >= ysize && groupX + kGroupSize >= xsize) {
            /* if the last group */
            // if (inpSize != pos) return PIK_FAILURE("lossless8");
          }

          memset(esize, 0, sizeof(esize));

          if (pMode == PM_Regular)
            for (size_t y = 0, yc = 0, yp; y < yEnd; ++y) {
              setRowImgPointers(img.Row) for (size_t x = 0; x <= width; ++x) {
                int maxErr,
                    prediction = predict_R_(x, yc + x - 1, yp + x, &maxErr);
                AfterPredictWhenDecompressing
              }
            }
          else if (pMode == PM_West)
            for (size_t y = 0, yc = 0, yp; y < yEnd; ++y) {
              setRowImgPointers(img.Row) for (size_t x = 0; x <= width; ++x) {
                int maxErr,
                    prediction = predict_W_(x, yc + x - 1, yp + x, &maxErr);
                AfterPredictWhenDecompressing
              }
            }
          else if (pMode == PM_North)
            for (size_t y = 0, yc = 0, yp; y < yEnd; ++y) {
              setRowImgPointers(img.Row) for (size_t x = 0; x <= width; ++x) {
                int maxErr,
                    prediction = predict_N_(x, yc + x - 1, yp + x, &maxErr);
                AfterPredictWhenDecompressing
              }
            }
        }  // groupX
      }    // groupY
      if (havePalette)
        for (size_t y = 0; y < ysize; ++y) {
          uint8_t* const PIK_RESTRICT rowImg = img.Row(y);
          for (size_t x = 0; x < xsize; ++x)  // UNROLL and PARALLELIZE ME!
            rowImg[x] = palette[rowImg[x]];
        }
    }  // run
    if (NumRuns > 1)
      printf("%d runs, %1.5f seconds", NumRuns,
             ((double)clock() - start) / CLOCKS_PER_SEC);
    *result = std::move(img);
    return true;
  }

  const int PL1 = 0, PL2 = 1, PL3 = 2;

  enum PlaneMethods_30 {  // 8/30 are redundant (left for encoder's convenience)
    RR_G_B = 0,           // p1=R  p2=G  p3=B
    RR_GmR_B = 1,         // p2-p1  p3
    RR_G_BmR = 2,         //   p2  p3-p1
    RR_GmR_BmR = 3,       // p2-p1 p3-p1

    RR_GmB_B = 4,  // == 21   p2-p3 @ p2
    RR_G_GmB = 5,  // ~= 12   p2-p3 @ p3

    RR_GmR_Bm2 = 6,  //  p2-p1  p3-(p1+p2)/2
    RR_Gm2_BmR = 7,  // p2-(p1+p3)/2   p3-p1
    RR_G_Bm2 = 8,    //   p2    p3-(p1+p2)/2
    RR_Gm2_B = 9,    // p2-(p1+p3)/2     p3

    R_GG_B = 10,  // p1=G  p2=R  p3=B
    RmG_GG_B = 11,
    R_GG_BmG = 12,
    RmG_GG_BmG = 13,

    RmB_GG_B = 14,  // == 22
    R_GG_RmB = 15,  // ~=  2

    RmG_GG_Bm2 = 16,
    Rm2_GG_BmG = 17,
    R_GG_Bm2 = 18,
    Rm2_GG_B = 19,

    R_G_BB = 20,  // p1=B  p2=R  p3=G
    R_GmB_BB = 21,
    RmB_G_BB = 22,
    RmB_GmB_BB = 23,

    RmG_G_BB = 24,  // == 11
    R_RmG_BB = 25,  // ~=  1

    RmB_Gm2_BB = 26,
    Rm2_GmB_BB = 27,
    R_Gm2_BB = 28,
    Rm2_G_BB = 29,
  };

  bool dcmprs512x512(pik::Image3B* img, int planeToDecompress, size_t& pos,
                     size_t groupY, size_t groupX,
                     const uint8_t* compressedData, size_t compressedSize,
                     size_t maxDecodedSize) {
    size_t esize[NUMCONTEXTS], xsize = img->xsize(), ysize = img->ysize();
    size_t yEnd = std::min((size_t)kGroupSize, ysize - groupY);
    width = std::min((size_t)kGroupSize, xsize - groupX) - 1;
    size_t area = yEnd * (width + 1);
    maxerrShift =
        (area > 25600
             ? 0
             : area > 12800 ? 1 : area > 4000 ? 2 : area > 400 ? 3 : 4);
    size_t decompressedSize = 0;  // is used only for the assert()

    PredictMode pMode;
    int nC = ((NUMCONTEXTS - 1) >> maxerrShift) + 1;
    for (int i = 0; i < nC; ++i) {
      size_t cs = decodeVarInt(compressedData, compressedSize, &pos);
      if (i == 0) pMode = (PredictMode)(cs % 3), cs /= 3;
      if (cs == 0) continue;
      int mode = cs % 3;
      cs /= 3;
      if (mode == 2) {
        if (pos >= compressedSize) return PIK_FAILURE("lossless8");
        if (cs > maxDecodedSize) return PIK_FAILURE("lossless8");
        memset(&edata[i][0], compressedData[pos++], ++cs),
            decompressedSize += cs;
      } else if (mode == 1) {
        if (pos + cs > compressedSize) return PIK_FAILURE("lossless8");
        if (cs > maxDecodedSize) return PIK_FAILURE("lossless8");
        memcpy(&edata[i][0], &compressedData[pos], ++cs),
            decompressedSize += cs, pos += cs;
      } else {
        if (pos + cs > compressedSize) return PIK_FAILURE("lossless8");
        size_t ds;
        if (!EntropyDecode(&compressedData[pos], cs, maxDecodedSize,
                           &edata[i][0], &ds)) {
          return PIK_FAILURE("lossless8");
        }
        pos += cs;
        decompressedSize += ds;
      }
    }
    if (decompressedSize != area) return PIK_FAILURE("lossless8");
    // if (groupY + kGroupSize >= ysize && groupX + kGroupSize >= xsize)
    //  /* if the last group */  assert(inpSize == pos);

    memset(esize, 0, sizeof(esize));

    if (pMode == PM_Regular)
      for (size_t y = 0, yc = 0, yp; y < yEnd; ++y) {
        setRowImgPointers3dec(img->PlaneRow) for (size_t x = 0; x <= width;
                                                  ++x) {
          int maxErr, prediction = predict_R_(x, yc + x - 1, yp + x, &maxErr);
          AfterPredictWhenDecompressing
        }
      }
    else if (pMode == PM_West)
      for (size_t y = 0, yc = 0, yp; y < yEnd; ++y) {
        setRowImgPointers3dec(img->PlaneRow) for (size_t x = 0; x <= width;
                                                  ++x) {
          int maxErr, prediction = predict_W_(x, yc + x - 1, yp + x, &maxErr);
          AfterPredictWhenDecompressing
        }
      }
    else if (pMode == PM_North)
      for (size_t y = 0, yc = 0, yp; y < yEnd; ++y) {
        setRowImgPointers3dec(img->PlaneRow) for (size_t x = 0; x <= width;
                                                  ++x) {
          int maxErr, prediction = predict_N_(x, yc + x - 1, yp + x, &maxErr);
          AfterPredictWhenDecompressing
        }
      }
    return true;
  }

  bool Colorful8bit_decompress(const PaddedBytes& bytes, size_t* bytes_pos,
                               Image3B* result) {
    if (*bytes_pos > bytes.size()) return PIK_FAILURE("lossless8");
    size_t compressedSize = bytes.size() - *bytes_pos;
    const uint8_t* compressedData = bytes.data() + *bytes_pos;

    size_t maxDecodedSize = kGroupSize * kGroupSize;  // Size of an edata entry

    size_t xsize, ysize, pos0 = 0, imageMethod = 0;
    xsize = decodeVarInt(compressedData, compressedSize, &pos0);
    ysize = decodeVarInt(compressedData, compressedSize, &pos0);
    if (!xsize || !ysize) return PIK_FAILURE("lossless8");
    // Too large, would run out of memory. Chosen as reasonable limit for pik
    // while being below default fuzzer memory limit. We check for total pixel
    // size, and an additional restriction to ysize, because large ysize
    // consumes more memory due to the scanline padding.
    if (uint64_t(xsize) * uint64_t(ysize) >= 268435456ull || ysize >= 65536) {
      return PIK_FAILURE("lossless8");
    }
    pik::Image3B img(xsize, ysize);
    std::vector<int> palette(0x100 * 3);

    clock_t start = clock();
    pb255 = 255 << PBits;
    for (int run = 0; run < NumRuns; ++run) {
      size_t pos = pos0;
      if (xsize * ysize > 4 * 0x100) {  // TODO: smarter decision making here
        const uint8_t* p = &compressedData[pos];
        imageMethod = *p++;
        if (imageMethod) {
          int numColors[3];
          ++pos;
          numColors[0] = decodeVarInt(compressedData, compressedSize, &pos);
          numColors[1] = decodeVarInt(compressedData, compressedSize, &pos);
          numColors[2] = decodeVarInt(compressedData, compressedSize, &pos);
          if (numColors[0] > 256) return PIK_FAILURE("lossless8");
          if (numColors[1] > 256) return PIK_FAILURE("lossless8");
          if (numColors[2] > 256) return PIK_FAILURE("lossless8");
          p = &compressedData[pos];
          const uint8_t* p_end = compressedData + compressedSize;
          for (int channel = 0; channel < 3; ++channel)
            if (imageMethod & (1 << channel))
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
        pos = p - &compressedData[0];
      }
      for (size_t groupY = 0; groupY < ysize; groupY += kGroupSize) {
        for (size_t groupX = 0; groupX < xsize; groupX += kGroupSize) {
          uint8_t *PIK_RESTRICT row1, *PIK_RESTRICT row2, *PIK_RESTRICT row3;
          size_t yEnd = std::min((size_t)kGroupSize, ysize - groupY);
          size_t xEnd = std::min((size_t)kGroupSize, xsize - groupX);
          if (!dcmprs512x512(&img, PL1, pos, groupY, groupX, compressedData,
                             compressedSize, maxDecodedSize))
            return PIK_FAILURE("lossless8");
          if (!dcmprs512x512(&img, PL2, pos, groupY, groupX, compressedData,
                             compressedSize, maxDecodedSize))
            return PIK_FAILURE("lossless8");
          if (!dcmprs512x512(&img, PL3, pos, groupY, groupX, compressedData,
                             compressedSize, maxDecodedSize))
            return PIK_FAILURE("lossless8");
          if (pos >= compressedSize) return PIK_FAILURE("lossless8");
          int planeMethod = compressedData[pos++];

#define T3bgn                                      \
  for (size_t y = 0; y < yEnd; ++y) {              \
    row1 = img.PlaneRow(PL1, groupY + y) + groupX; \
    row2 = img.PlaneRow(PL2, groupY + y) + groupX; \
    row3 = img.PlaneRow(PL3, groupY + y) + groupX; \
    for (size_t x = 0; x < xEnd; ++x) {            \
      int R = row1[x], G = row2[x], B = row3[x];   \
      (void)R;                                     \
      (void)G;                                     \
      (void)B;

// Close T3bgn above; not using a #define confuses brace matching of editor.
#define CC \
  }        \
  }

          switch (planeMethod) {
            case 0:
            case 10:
            case 20:
              break;
            case 1:
              T3bgn G += R + 0x80;
              row2[x] = G;
              CC break;
            case 2:
              T3bgn B += R + 0x80;
              row3[x] = B;
              CC break;
            case 3:
              T3bgn G += R + 0x80;
              B += R + 0x80;
              row2[x] = G;
              row3[x] = B;
              CC break;
            case 22:
            case 4:
              T3bgn row2[x] = G + B + 0x80;
              CC break;
            case 5:
              T3bgn row3[x] = G - B + 0x80;
              CC break;
            case 6:
              T3bgn row2[x] = G = (G + R + 0x80) & 0xff;
              row3[x] = B + ((R + G) >> 1) + 0x80;
              CC break;
            case 7:
              T3bgn row3[x] = B = (B + R + 0x80) & 0xff;
              row2[x] = G + ((R + B) >> 1) + 0x80;
              CC break;
            case 8:
              T3bgn row3[x] = B + ((R + G) >> 1) + 0x80;
              CC break;
            case 9:
              T3bgn row2[x] = G + ((R + B) >> 1) + 0x80;
              CC break;

            case 24:
            case 11:
              T3bgn R += G + 0x80;
              row1[x] = R;
              CC break;
            case 12:
              T3bgn B += G + 0x80;
              row3[x] = B;
              CC break;
            case 13:
              T3bgn R += G + 0x80;
              B += G + 0x80;
              row1[x] = R;
              row3[x] = B;
              CC break;
            case 21:
            case 14:
              T3bgn row1[x] = R + B + 0x80;
              CC break;
            case 15:
              T3bgn row3[x] = R - B + 0x80;
              CC break;

            case 16:
              T3bgn row1[x] = R = (R + G + 0x80) & 0xff;
              row3[x] = B + ((R + G) >> 1) + 0x80;
              CC break;
            case 17:
              T3bgn row3[x] = B = (B + G + 0x80) & 0xff;
              row1[x] = R + ((B + G) >> 1) + 0x80;
              CC break;
            case 18:
              T3bgn row3[x] = B + ((R + G) >> 1) + 0x80;
              CC break;
            case 19:
              T3bgn row1[x] = R + ((B + G) >> 1) + 0x80;
              CC break;

            case 23:
              T3bgn G += B + 0x80;
              R += B + 0x80;
              row1[x] = R;
              row2[x] = G;
              CC break;
            case 25:
              T3bgn row2[x] = R - G + 0x80;
              CC break;
            case 26:
              T3bgn row1[x] = R = (R + B + 0x80) & 0xff;
              row2[x] = G + ((B + R) >> 1) + 0x80;
              CC break;
            case 27:
              T3bgn row2[x] = G = (G + B + 0x80) & 0xff;
              row1[x] = R + ((B + G) >> 1) + 0x80;
              CC break;
            case 28:
              T3bgn row2[x] = G + ((B + R) >> 1) + 0x80;
              CC break;
            case 29:
              T3bgn row1[x] = R + ((B + G) >> 1) + 0x80;
              CC break;
          }
        }  // groupX
      }    // groupY
// Disabled, because it is actually useful that the decoder supports decoding
// its own stream when contained inside a bigger stream and knows the correct
// end position.
#if 0
      assert(pos == compressedSize);
#endif

      for (int channel = 0; channel < 3; ++channel)
        if (imageMethod & (1 << channel)) {
          int* p = &palette[0x100 * channel];
          for (size_t y = 0; y < ysize; ++y) {
            uint8_t* const PIK_RESTRICT rowImg = img.PlaneRow(channel, y);
            for (size_t x = 0; x < xsize; ++x)  // UNROLL AND PARALLELIZE ME!
              rowImg[x] = p[rowImg[x]];
          }
        }
    }  // run
    if (NumRuns > 1)
      printf("%d runs, %1.5f seconds", NumRuns,
             ((double)clock() - start) / CLOCKS_PER_SEC);
    *result = std::move(img);
    return true;
  }

  uint32_t cmprs512x512(pik::Image3B& img, int planeToCompress, int planeToUse,
                        size_t groupY, size_t groupX,
                        uint8_t* compressedOutput) {
    size_t esize[NUMCONTEXTS], xsize = img.xsize(), ysize = img.ysize();
    memset(esize, 0, sizeof(esize));
    size_t yEnd = std::min((size_t)kGroupSize, ysize - groupY);
    width = std::min((size_t)kGroupSize, xsize - groupX) - 1;
    size_t area = yEnd * (width + 1);
    maxerrShift =
        (area > 25600
             ? 0
             : area > 12800 ? 1 : area > 4000 ? 2 : area > 400 ? 3 : 4);

    uint64_t fromN = 0, fromW = 0;
    for (size_t y = 1; y < yEnd; ++y) {
      rowImg = img.PlaneRow(planeToCompress, groupY + y) + groupX;
      rowPrev = img.PlaneRow(planeToCompress, groupY + y - 1) + groupX;
      for (size_t x = 1; x <= width; ++x) {
        int c = rowImg[x];
        int N = rowPrev[x];
        int W = rowImg[x - 1];
        N -= c;
        W -= c;
        fromN += N * N;
        fromW += W * W;
      }
    }
    PredictMode pMode = PM_Regular;
    if (fromW * 5 < fromN * 4) pMode = PM_West;  // no 'else' to reduce codesize
    if (fromN * 5 < fromW * 4)
      pMode = PM_North;  // if (fromN < fromW*0.8)
                         // printf("%c ", pMode);

    if (pMode == PM_Regular)  // Regular mode
      for (size_t y = 0, yc = 0, yp; y < yEnd; ++y) {
        setRowImgPointers3(img.PlaneRow) for (size_t x = 0; x <= width; ++x) {
          int maxErr, prediction = predict_R_(x, yc + x - 1, yp + x, &maxErr);
          AfterPredictWhenCompressing3
        }
      }
    else if (pMode == PM_West)  // 'West predicts better' mode
      for (size_t y = 0, yc = 0, yp; y < yEnd; ++y) {
        setRowImgPointers3(img.PlaneRow) for (size_t x = 0; x <= width; ++x) {
          int maxErr, prediction = predict_W_(x, yc + x - 1, yp + x, &maxErr);
          AfterPredictWhenCompressing3
        }
      }
    else if (pMode == PM_North)  // 'North predicts better' mode
      for (size_t y = 0, yc = 0, yp; y < yEnd; ++y) {
        setRowImgPointers3(img.PlaneRow) for (size_t x = 0; x <= width; ++x) {
          int maxErr, prediction = predict_N_(x, yc + x - 1, yp + x, &maxErr);
          AfterPredictWhenCompressing3
        }
      }
    else {
    }  // TODO: other prediction modes!

    size_t pos = 0;
    int nC = ((NUMCONTEXTS - 1) >> maxerrShift) + 1;
    for (int i = 0; i < nC; ++i) {
      if (esize[i]) {
        // size_t cs = FSE_compress(&compressedDataTmpBuf[0],
        // sizeof(compressedDataTmpBuf), &edata[i][0], esize[i]);
        size_t cs;
        if (!EntropyEncode(&edata[i][0], esize[i], sizeof(compressedDataTmpBuf),
                           &compressedDataTmpBuf[0], &cs)) {
          return PIK_FAILURE("lossless8");
        }

        size_t s = (cs <= 1 ? (esize[i] - 1) * 3 + 1 + cs : cs * 3);
        pos += encodeVarInt(i > 0 ? s : s * 3 + pMode, &compressedOutput[pos]);
        if (cs == 1)
          compressedOutput[pos++] = edata[i][0];
        else if (cs == 0)
          memcpy(&compressedOutput[pos], &edata[i][0], esize[i]),
              pos += esize[i];
        else
          memcpy(&compressedOutput[pos], &compressedDataTmpBuf[0], cs),
              pos += cs;
      } else {
        pos += encodeVarInt(i > 0 ? 0 : pMode, &compressedOutput[pos]);
      }
    }  // i
    return pos;
  }

#define FWr(buf, bufsize)                            \
  {                                                  \
    if (run == 0) {                                  \
      size_t current = bytes->size();                \
      bytes->resize(bytes->size() + bufsize);        \
      memcpy(bytes->data() + current, buf, bufsize); \
    }                                                \
  }

#define FWrByte(b)    \
  {                   \
    uint8_t byte = b; \
    FWr(&byte, 1);    \
  }

  bool Colorful8bit_compress(const Image3B& img_in, pik::PaddedBytes* bytes) {
    clock_t start = clock();

    // The code modifies the image for palette so must copy for now.
    Image3B img = CopyImage(img_in);

    std::vector<uint8_t> temp_buffer(kGroupSize2plus * 6);
    compressedData = temp_buffer.data();

    pb255 = 255 << PBits;
    for (int run = 0; run < NumRuns; ++run) {
      size_t xsize = img.xsize(), ysize = img.ysize(), pos;
      pos = encodeVarInt(xsize, &compressedData[0]);
      pos += encodeVarInt(ysize, &compressedData[pos]);
      FWr(&compressedData[0], pos) int numColors[3] = {0xff, 0xff, 0xff};

      if (xsize * ysize > 4 * 0x100) {  // TODO: smarter decision making here
        // Let's check whether the image should be 'palettized',
        // because the range is 64k, but 25% or more of the range is unused.
        uint8_t flags = 0, bits[3 * 0x100 / 8], *pb = &bits[0];
        uint32_t palette123[3 * 0x100];

#if 1
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
          if (count >= 240) {
            flags = 0;
            break;
          }  // TODO: decision making

          flags += 1 << channel;
          numColors[channel] = count;
          palette[first] = 1;
          for (int sb = 0, x = 0; x < 0x100;
               x += 8) {  // Compress the bits, not store!
            uint32_t b = 0, v;
            for (int y = x + 7; y >= x; --y)
              v = (palette[y] ? 1 : 0), b += b + v, sb += v;
            *pb++ = b;
            if (sb >= count || sb + 0x100 - 8 - x == count) break;
          }
          palette[first] = 0;
        }  // for channel
#endif
        FWrByte(flags);  // As of now (Dec.2018) ImageMethod==flags
        if (flags) {
          for (int channel = 0; channel < 3; ++channel) {
            uint32_t* palette = &palette123[0x100 * channel];
            for (size_t y = 0; y < ysize; ++y) {
              uint8_t* const PIK_RESTRICT rowImg = img.PlaneRow(channel, y);
              for (size_t x = 0; x < xsize; ++x)  // UNROLL AND PARALLELIZE ME!
                rowImg[x] = palette[rowImg[x]];
            }
          }
          pos = encodeVarInt(numColors[0], &compressedData[0]);
          pos += encodeVarInt(numColors[1], &compressedData[pos]);
          pos += encodeVarInt(numColors[2], &compressedData[pos]);
          FWr(&compressedData[0], pos);
          FWr(&bits[0], sizeof(uint8_t) * (pb - &bits[0]));
        }  // if (flags)
      }    // if (xsize*ysize > 4*0x100)
      uint8_t* compressedData2 = &compressedData[kGroupSize2plus];
      uint8_t* compressedData3 = &compressedData[kGroupSize2plus * 2];
      uint8_t* cd4 = &compressedData[kGroupSize2plus * 3];
      uint8_t* cd5 = &compressedData[kGroupSize2plus * 4];
      uint8_t* cd6 = &compressedData[kGroupSize2plus * 5];
      for (size_t groupY = 0; groupY < ysize; groupY += kGroupSize) {
        for (size_t groupX = 0; groupX < xsize; groupX += kGroupSize) {
          size_t S1, S2, S3, S4, S5, S6, s1, s2, s3, p1, p2, p3;
          uint8_t *cd1, *cd2, *cd3;
          int planeMethod;  // Here we try guessing which of the 30 PlaneMethods
                            // is best, after trying just six color planes.

          s1 = cmprs512x512(img, PL1, PL1, groupY, groupX, compressedData);
          s2 = cmprs512x512(img, PL2, PL2, groupY, groupX, compressedData2);
          s3 = cmprs512x512(img, PL3, PL3, groupY, groupX, compressedData3);

          S1 = s2, p1 = PL2, cd1 = compressedData2, planeMethod = 10;
          S2 = s1, p2 = PL1, cd2 = compressedData;
          S3 = s3, p3 = PL3, cd3 = compressedData3;
          if (s1 < s2 * 63 / 64 && s1 < s3) {
            S1 = s1, p1 = PL1, cd1 = compressedData, planeMethod = 0;
            S2 = s2, p2 = PL2, cd2 = compressedData2;
            S3 = s3, p3 = PL3, cd3 = compressedData3;
          } else if (s3 < s2 * 63 / 64 && s3 < s1) {
            S1 = s3, p1 = PL3, cd1 = compressedData3, planeMethod = 20;
            S2 = s1, p2 = PL1, cd2 = compressedData;
            S3 = s2, p3 = PL2, cd3 = compressedData2;
          }
          S4 = cmprs512x512(img, p2, p1, groupY, groupX, cd4); /* R-G+0x80 */
          S5 = cmprs512x512(img, p3, p1, groupY, groupX, cd5); /* B-G+0x80 */
          if (p1 == PL1)
            FWr(cd1, S1)

                if (S4 >= S2 && S5 >= S3) {
              S6 =
                  cmprs512x512(img, p2, p3, groupY, groupX, cd6); /* R-B+0x80 */
              if (S6 >= S2 && S6 >= S3)
                FWr(cd2, S2) else if (S3 > S2 && S3 > S6)
                    FWr(cd2, S2) else FWr(cd6, S6) if (p1 == PL2)
                        FWr(cd1, S1) if (S6 >= S2 && S6 >= S3) {
                  FWr(cd3, S3)
                }
              else if (S3 > S2 && S3 > S6) {
                FWr(cd6, S6) planeMethod += 5;
              } else {
                FWr(cd3, S3) planeMethod += 4;
              }
            }
          else {
            size_t yEnd = std::min((size_t)kGroupSize, ysize - groupY);
            size_t xEnd = std::min((size_t)kGroupSize, xsize - groupX);
            if (S5 < S4) {
              for (size_t y = 0; y < yEnd; ++y) {
                uint8_t* PIK_RESTRICT row1 =
                    img.PlaneRow(p1, groupY + y) + groupX;
                uint8_t* PIK_RESTRICT row2 =
                    img.PlaneRow(p2, groupY + y) + groupX;
                for (size_t x = 0; x < xEnd; ++x) {
                  uint32_t v1 = row1[x], v2 = (row2[x] + v1 + 0x80) & 0xff;
                  row2[x] = ((v1 + v2) >> 1) - v1 + 0x80;
                }
              }
              S6 = cmprs512x512(img, p3, p2, groupY, groupX,
                                cd6); /* B-(R+G)/2 */
              if (S4 < S2)
                FWr(cd4, S4) else FWr(cd2, S2) if (p1 == PL2)
                    FWr(cd1, S1) if (S3 <= S5 && S3 <= S6) {
                  FWr(cd3, S3) planeMethod += 1;
                }
              else if (S5 <= S6) {
                FWr(cd5, S5) planeMethod += (S4 < S2 ? 3 : 2);
              } else {
                FWr(cd6, S6) planeMethod += (S4 < S2 ? 6 : 8);
              }
            } else {
              for (size_t y = 0; y < yEnd; ++y) {
                uint8_t* PIK_RESTRICT row1 =
                    img.PlaneRow(p1, groupY + y) + groupX;
                uint8_t* PIK_RESTRICT row3 =
                    img.PlaneRow(p3, groupY + y) + groupX;
                for (size_t x = 0; x < xEnd; ++x) {
                  uint32_t v1 = row1[x], v3 = (row3[x] + v1 + 0x80) & 0xff;
                  row3[x] = ((v1 + v3) >> 1) - v1 + 0x80;
                }
              }
              S6 = cmprs512x512(img, p2, p3, groupY, groupX,
                                cd6); /* R-(B+G)/2 */
              if (S2 <= S4 && S2 <= S6) {
                FWr(cd2, S2) planeMethod += 2;
              } else if (S4 <= S6) {
                FWr(cd4, S4) planeMethod += (S5 < S3 ? 3 : 1);
              } else {
                FWr(cd6, S6) planeMethod += (S5 < S3 ? 7 : 9);
              }
              if (p1 == PL2)
                FWr(cd1, S1) if (S5 < S3) FWr(cd5, S5) else FWr(cd3, S3)
            }
          }
          if (p1 == PL3)
            FWr(cd1, S1) FWrByte(planeMethod);  // printf("%2d ", planeMethod);
        }                                       // groupX
      }                                         // groupY
    }                                           // run
    if (NumRuns > 1)
      printf("%d runs, %1.5f seconds", NumRuns,
             ((double)clock() - start) / CLOCKS_PER_SEC);
    return true;
  }

};  // struct State

}  // namespace

bool Grayscale8bit_compress(const ImageB& img, PaddedBytes* bytes) {
  std::unique_ptr<State> state(new State());
  return state->Grayscale8bit_compress(img, bytes);
}

bool Grayscale8bit_decompress(const PaddedBytes& bytes, size_t* pos,
                              ImageB* result) {
  std::unique_ptr<State> state(new State());
  return state->Grayscale8bit_decompress(bytes, pos, result);
}

bool Colorful8bit_compress(const Image3B& img, PaddedBytes* bytes) {
  std::unique_ptr<State> state(new State());
  return state->Colorful8bit_compress(img, bytes);
}

bool Colorful8bit_decompress(const PaddedBytes& bytes, size_t* pos,
                             Image3B* result) {
  std::unique_ptr<State> state(new State());
  return state->Colorful8bit_decompress(bytes, pos, result);
}

}  // namespace pik
