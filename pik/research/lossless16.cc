// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

// @author Alexander Rhatushnyak

#include "pik/lossless16.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "pik/lossless_c3p.h"
#include "pik/lossless_entropy.h"

namespace pik {

namespace {

const int kPcpEnabled = 1 << 16;  // Per-channel palette
const int kWithSign = 4;
const int kBitsMax = 13;
const int kNumContexts = 1 + kWithSign + kBitsMax;
const int kMaxError = 0x3fbf;
const int kMaxSumErrors = (kMaxError + 1) * 21 / 4;
const int mulWeights[kNumContexts * 3] = {
    30, 0, 28, 0, 4,  0, 4,  0, 8,  11, 11, 14, 15, 21, 21, 21, 22, 23,
    31, 0, 31, 0, 31, 0, 34, 0, 35, 32, 33, 33, 33, 31, 28, 25, 22, 9,
    30, 0, 32, 0, 56, 0, 57, 0, 51, 51, 50, 44, 44, 35, 34, 31, 29, 25,
};

// The maximum number of total image pixels supported by the lossless format.
const uint64_t kMaxLossless16Pixels = 1ull << 27ull;
// The maximum dimension of total pixels in Y.
const uint64_t kMaxLossless16PixelsY = 1ull << 16ull;

// TODO(lode): split state variables needed for encoder from those for decoder
//             and run const init just once!  ~65536*3 'const' values in State()
struct State {
  const size_t groupSizeX;
  const size_t groupSizeY;
  const size_t groupSize2plus;

  int prediction0, prediction1, prediction2, prediction3, width;
  uint32_t numColors[3], planeMethod, maxTpv, maxTpvDiv2;
  uint16_t *PIK_RESTRICT rowImg, *PIK_RESTRICT rowPrev;

  std::vector<uint16_t> edata[kNumContexts];
  std::vector<uint8_t> compressedDataTmpBuf;
  uint8_t* compressedData;
  std::vector<int32_t> errors0;  // Errors of predictor 0
  std::vector<int32_t> errors1;  // Errors of predictor 1
  std::vector<int32_t> errors2;  // Errors of predictor 2
  std::vector<int32_t> errors3;  // Errors of predictor 3
  std::vector<uint8_t> nbitErr;
  std::vector<int32_t> trueErr;

  uint16_t error2weight[kMaxSumErrors];  // const
  uint8_t numBitsTable[256];             // const

#define CalcDistanceFromPredictionAndTPV \
  do {                                   \
    if (p > maxTpvDiv2) {                \
      if (v >= p)                        \
        d = (v - p) * 2;                 \
      else if (v >= p - (maxTpv - p))    \
        d = (p - v) * 2 - 1;             \
      else                               \
        d = maxTpv - v;                  \
    } else {                             \
      if (v < p)                         \
        d = (p - v) * 2 - 1;             \
      else if (v <= p * 2)               \
        d = (v - p) * 2;                 \
      else                               \
        d = v;                           \
    }                                    \
  } while (false)

#define CalcTPVfromPredictionAndDistance \
  do {                                   \
    if (p > maxTpvDiv2) {                \
      if (d > (maxTpv - p) * 2)          \
        v = maxTpv - d;                  \
      else if (d & 1)                    \
        v = p - 1 - (d >> 1);            \
      else                               \
        v = p + (d >> 1);                \
    } else {                             \
      if (d > p * 2)                     \
        v = d;                           \
      else if (d & 1)                    \
        v = p - 1 - (d >> 1);            \
      else                               \
        v = p + (d >> 1);                \
    }                                    \
  } while (false)

  PIK_INLINE int fwd_sign2LSB(int p, int v) {
    int d;
    CalcDistanceFromPredictionAndTPV;
    return d;
  }

  PIK_INLINE int bck_sign2LSB(int p, int d) {
    int v;
    CalcTPVfromPredictionAndDistance;
    return v;
  }

  State(size_t groupSizeX, size_t groupSizeY)
      : groupSizeX(groupSizeX),
        groupSizeY(groupSizeY),
        groupSize2plus(groupSizeX * groupSizeY * 9 / 8) {
    for (size_t i = 0; i < kNumContexts; i++) {
      // Also prevent uninitialized values in case of invalid compressed data
      edata[i].resize(groupSizeX * groupSizeY, 0);
    }
    compressedDataTmpBuf.resize(groupSize2plus);
    errors0.resize(groupSizeX * 2 + 4);
    errors1.resize(groupSizeX * 2 + 4);
    errors2.resize(groupSizeX * 2 + 4);
    errors3.resize(groupSizeX * 2 + 4);
    nbitErr.resize(groupSizeX * 2);
    trueErr.resize(groupSizeX * 2);

    for (int i = 0; i < 256; ++i) {
      numBitsTable[i] = numbitsInit(i);  // const init!
    }

    error2weight[0] = 0xffff;
    for (int j = 1; j < kMaxSumErrors; ++j) {
      error2weight[j] = 180 * 256 / j;  // const init!  180
    }
  }

  PIK_INLINE int numbitsInit(int x) {
    assert(0 <= x && x <= 255);
    int res = 0;
    if (x >= 16) res = 4, x >>= 4;
    if (x >= 4) res += 2, x >>= 2;
    return (res + std::min(x, 2));
  }

  PIK_INLINE int numBits(int x) {
    assert(0 <= x && x <= 0xffff);
    if (x < 256) return numBitsTable[x];
    return std::min(8 + numBitsTable[x >> 8], kBitsMax);
  }

  PIK_INLINE int predict1y0(size_t x, size_t yp, size_t yp1, int* maxErr) {
    *maxErr = (x == 0 ? kNumContexts - 1
                      : x == 1 ? nbitErr[yp - 1]
                               : std::max(nbitErr[yp - 1], nbitErr[yp - 2]));
    prediction0 = prediction1 = prediction2 = prediction3 =
        (x == 0 ? 14 * 256  // 14
                : x == 1 ? rowImg[x - 1]
                         : rowImg[x - 1] + (rowImg[x - 1] - rowImg[x - 2]) / 4);
    return (prediction0 < 0 ? 0 : prediction0 > maxTpv ? maxTpv : prediction0);
  }

  PIK_INLINE int predict1x0(size_t x, size_t yp, size_t yp1, int* maxErr) {
    *maxErr = std::max(nbitErr[yp1], nbitErr[yp1 + (x < width ? 1 : 0)]);
    prediction0 = prediction2 = prediction3 = rowPrev[x];
    prediction1 = (rowPrev[x] * 3 + rowPrev[x + (x < width ? 1 : 0)] + 2) >> 2;
    return prediction1;
  }

  PIK_INLINE int predict1(size_t x, size_t yp, size_t yp1, int* maxErr) {
    if (!rowPrev) return predict1y0(x, yp, yp1, maxErr);
    if (x == 0LL) return predict1x0(x, yp, yp1, maxErr);
    int weight0 = ((errors0[yp1] * 9) >> 3) + errors0[yp1 - 1];
    int weight1 = ((errors1[yp1] * 9) >> 3) + errors1[yp1 - 1];
    int weight2 = ((errors2[yp1] * 9) >> 3) + errors2[yp1 - 1];
    int weight3 = ((errors3[yp1] * 9) >> 3) + errors3[yp1 - 1];
    uint8_t mxe = nbitErr[yp - 1];
    mxe = std::max(mxe, nbitErr[yp1]);
    mxe = std::max(mxe, nbitErr[yp1 - 1]);
    int N = rowPrev[x], W = rowImg[x - 1],
        NE = N;  // NW = rowPrev[x - 1] unused!
    if (x < width) {
      mxe = std::max(mxe, nbitErr[yp1 + 1]), NE = rowPrev[x + 1];
      weight0 += errors0[yp1 + 1];
      weight1 += errors1[yp1 + 1];
      weight2 += errors2[yp1 + 1];
      weight3 += errors3[yp1 + 1];
    }

    int mE = mxe;
    weight0 = error2weight[weight0] * mulWeights[mE];
    weight1 = error2weight[weight1] * 33 + 1;
    weight2 = error2weight[weight2] * mulWeights[mE + kNumContexts];
    weight3 = error2weight[weight3] * mulWeights[mE + kNumContexts * 2];

    int teW = trueErr[yp - 1];  // range: -0xffff...0xffff
    int teN = trueErr[yp1];
    int teNW = trueErr[yp1 - 1];
    int sumWN = teN + teW;  // range: -0x1fffe...0x1fffe
    int teNE = (x < width ? trueErr[yp1 + 1] : teNW);

    prediction0 = N - sumWN * 3 / 4;                 // 24/32
    prediction1 = W - (sumWN + teNW) * 5 / 16;       // 10/32
    prediction2 = W + (((NE - N) * 13 + 8) >> 4);    // 26/32
    prediction3 = N - (teN + teNW + teNE) * 7 / 32;  //  7/32
    int sumWeights = weight0 + weight1 + weight2 + weight3;
    int64_t s = sumWeights >> 2;
    s += static_cast<int64_t>(prediction0) * weight0;
    s += static_cast<int64_t>(prediction1) * weight1;
    s += static_cast<int64_t>(prediction2) * weight2;
    s += static_cast<int64_t>(prediction3) * weight3;
    int prediction = s / sumWeights;

    if (mxe && mxe <= kWithSign * 2) {
      if (sumWN * 2 + teNW + teNE < 0) --mxe;  // 2 2 1 1
    }
    *maxErr = mxe;

    int mx = (W > NE ? W : NE), mn = W + NE - mx;
    if (N > mx) mx = N - 1;
    if (N < mn) mn = N + 1;
    prediction = std::max(mn, std::min(mx, prediction));
    return prediction;
  }

#define Update_Errors_0_1_2_3                                    \
  do {                                                           \
    err = prediction0 - truePixelValue;                          \
    if (err < 0) err = -err; /* abs() and min()? worse speed! */ \
    if (err > kMaxError) err = kMaxError;                        \
    errors0[yp + x] = err;                                       \
    errors0[1 + yp1 + x] += err;                                 \
    err = prediction1 - truePixelValue;                          \
    if (err < 0) err = -err;                                     \
    if (err > kMaxError) err = kMaxError;                        \
    errors1[yp + x] = err;                                       \
    errors1[1 + yp1 + x] += err;                                 \
    err = prediction2 - truePixelValue;                          \
    if (err < 0) err = -err;                                     \
    if (err > kMaxError) err = kMaxError;                        \
    errors2[yp + x] = err;                                       \
    errors2[1 + yp1 + x] += err;                                 \
    err = prediction3 - truePixelValue;                          \
    if (err < 0) err = -err;                                     \
    if (err > kMaxError) err = kMaxError;                        \
    errors3[yp + x] = err;                                       \
    errors3[1 + yp1 + x] += err;                                 \
  } while (false)

#define Update_Size_And_Errors                                        \
  do {                                                                \
    ++esize[maxErr];                                                  \
    trueErr[yp + x] = err;                                            \
    err = numBits(err >= 0 ? err : -err);                             \
    nbitErr[yp + x] = (err <= kWithSign ? err * 2 : err + kWithSign); \
    Update_Errors_0_1_2_3;                                            \
  } while (false)

  const uint16_t smt0[64] = {
      0x2415, 0x1d7d, 0x1f71, 0x46fe, 0x24f1, 0x3f15, 0x4a65, 0x6236,
      0x242c, 0x34ce, 0x4872, 0x5cf6, 0x4857, 0x64fe, 0x6745, 0x7986,
      0x24ad, 0x343c, 0x499a, 0x5fb5, 0x49a9, 0x61e8, 0x6e1f, 0x78ae,
      0x4ba3, 0x6332, 0x6c8b, 0x7ccd, 0x6819, 0x8247, 0x83f2, 0x8cce,
      0x247e, 0x3277, 0x391f, 0x5ea3, 0x4694, 0x5168, 0x67e3, 0x784b,
      0x474b, 0x5072, 0x666b, 0x6cb3, 0x6514, 0x7ba6, 0x83e4, 0x8cef,
      0x48bf, 0x6363, 0x6677, 0x7b76, 0x67f9, 0x7e0d, 0x826f, 0x8a52,
      0x659f, 0x7d6f, 0x7f8e, 0x8f66, 0x7ed6, 0x9169, 0x9269, 0x90e4,
  };

  uint8_t* Palette_compress(int numChannels, uint8_t* pb,
                            std::vector<uint32_t>* palette123,
                            const uint32_t* firstColors) {
    for (int channel = 0; channel < numChannels; ++channel) {
      if (numColors[channel] != 0x10000) {
        uint32_t* palette = &((*palette123)[0x10000 * channel]);
        uint8_t* pb0 = pb;
        pb += 2;  // reserve 2 bytes for  (Compressed Size)*2 + Method
        palette[firstColors[channel]] = 1;
        uint32_t nc = numColors[channel], x1 = 0, x2 = 0xffffffff;
        int x, smt[64], context6 = 0, sumv = 0;
        for (int i = 0; i < 64; ++i) smt[i] = smt0[i] << 11;  // 1<<(15+11);
        for (x = 0; x < 0x10000; ++x) {
          int v = (palette[x] ? 1 : 0);
          uint32_t pr = smt[context6] >> 11;
          uint32_t xmid =
              x1 + ((x2 - x1) >> 16) * pr + (((x2 - x1) & 0xffff) * pr >> 16);
          assert(pr >= 0 && pr <= 0xffff && xmid >= x1 && xmid < x2);
          if (v) {
            x2 = xmid;
          } else {
            x1 = xmid + 1;
          }
          if (((x1 ^ x2) & 0xff000000) == 0) {
            do {
              *pb++ = x1 >> 24;
              x1 <<= 8;
              x2 = (x2 << 8) + 255;
            } while (((x1 ^ x2) & 0xff000000) == 0);
            if (pb >= pb0 + 2 + 0x10000) break;
          }
          int p0 = smt[context6];
          p0 += ((v << (16 + 11)) - p0) * 5 >> 7;  // Learning rate
          smt[context6] = p0;
          context6 = (context6 * 2 + v) & 0x3f;
          sumv += v;
          if (sumv == nc || sumv + 0x10000 - 1 - x == nc) break;
        }
        *pb++ = static_cast<uint8_t>((x1 >> 24) & 0xFF);
        // if (count > 512) {  for (int i = 0; i < 64; ++i)
        //                        printf("0x%x,", smt[i]);   printf("\n"); }
        int method = 0;
        if (pb - (pb0 + 2) >= ((x + 7) >> 3)) {  // Store, no compression
          method = 1;
          pb = pb0 + 2;
          for (int sumv = 0, x = 0; x < 0x10000; x += 8) {
            uint32_t b = 0, v;
            for (int y = x + 7; y >= x; --y)
              v = (palette[y] ? 1 : 0), b += b + v, sumv += v;
            *pb++ = b;
            if (sumv >= nc || sumv + 0x10000 - 8 - x == nc) break;
          }
        }
        int compressedSize = (pb - (pb0 + 2)) * 2 + method;
        pb0[0] = static_cast<uint8_t>(compressedSize & 0xFF);
        pb0[1] = static_cast<uint8_t>((compressedSize >> 8) & 0xFF);
        palette[firstColors[channel]] = 0;
      }
    }
    return pb;
  }

  void PerChannelPalette_compress_1(ImageU* img, PaddedBytes* bytes) {
    const int numChannels = 1, channel = 0;
    size_t xsize = img->xsize(), ysize = img->ysize();
    std::vector<uint32_t> palette123(0x10000 * numChannels);
    memset(palette123.data(), 0, 0x10000 * numChannels * sizeof(uint32_t));
    uint8_t bits[0x10010 / 8], flags = 0, compressedData[6];
    memset(bits, 0, sizeof(bits));
    uint32_t firstColors[3];

    uint32_t i, count, *palette = &palette123[0x10000 * channel];
    for (size_t y = 0; y < ysize; ++y) {
      uint16_t* const PIK_RESTRICT rowImg = img->Row(y);
      for (size_t x = 0; x < xsize; ++x)  // UNROLL AND PARALLELIZE ME!
        palette[rowImg[x]] = 1;
    }
    // count the number of pixel values present in the channel
    for (i = 0; i < 0x10000; ++i)
      if (palette[i]) break;
    for (firstColors[channel] = i, count = 0; i < 0x10000; ++i)
      if (palette[i]) palette[i] = count++;
    // printf(" count=%5d, %f%%\n", count, count * 100. / 65536);
    if (count < 0x10000 / 16)  // TODO(arhatushnyak): smarter decision making
      flags = 1 << channel;
    numColors[channel] = flags ? count : 0x10000;

    FWrByte(flags);  // As of Mar.2019, ImageMethod==flags
    if (flags != 0) {
      uint8_t* pb =
          Palette_compress(numChannels, &bits[0], &palette123, &firstColors[0]);
      // Apply the channel's "palette"
      uint32_t* palette = &palette123[0x10000 * channel];
      for (size_t y = 0; y < ysize; ++y) {
        uint16_t* const PIK_RESTRICT rowImg = img->Row(y);
        for (size_t x = 0; x < xsize; ++x)  // UNROLL AND PARALLELIZE ME!
          rowImg[x] = palette[rowImg[x]];
      }
      compressedData[0] = (numColors[channel] - 1) & 255;
      compressedData[1] = (numColors[channel] - 1) >> 8;
      FWr(&compressedData[0], 2);
      FWr(&bits[0], sizeof(uint8_t) * (pb - &bits[0]));
    }
  }

  void PerChannelPalette_compress_3(Image3U* img, PaddedBytes* bytes) {
    const int numChannels = 3;
    size_t xsize = img->xsize(), ysize = img->ysize();
    std::vector<uint32_t> palette123(0x10000 * numChannels);
    memset(palette123.data(), 0, 0x10000 * numChannels * sizeof(uint32_t));
    uint8_t bits[3 * 0x10010 / 8], flags = 0, compressedData[6];
    memset(bits, 0, sizeof(bits));
    uint32_t firstColors[3], sum = 0;

    for (int channel = 0; channel < numChannels; ++channel) {
      uint32_t i, count, *palette = &palette123[0x10000 * channel];
      for (size_t y = 0; y < ysize; ++y) {
        const uint16_t* const PIK_RESTRICT rowImg =
            img->ConstPlaneRow(channel, y);
        for (size_t x = 0; x < xsize; ++x)  // UNROLL AND PARALLELIZE ME!
          palette[rowImg[x]] = 1;
      }
      // count the number of pixel values present in the channel
      for (i = 0; i < 0x10000; ++i)
        if (palette[i]) break;
      for (firstColors[channel] = i, count = 0; i < 0x10000; ++i)
        if (palette[i]) palette[i] = count++;
      // printf(" count=%5d, %f%%\n", count, count * 100. / 65536);
      numColors[channel] = count;
      if (count < 0x10000 / 2)  // TODO(arhatushnyak): smarter decision making
        flags += 1 << channel;
      sum += count;
    }  // for channel

    if (sum < 0x10000 * 9 / 4) flags = 7;
    FWrByte(flags);  // As of Mar.2019, ImageMethod==flags
    if (flags != 0) {
      int pos = 0;
      // Apply the channel's "palette"
      for (int channel = 0; channel < numChannels; ++channel) {
        if ((flags & (1 << channel)) != 0) {
          uint32_t* palette = &palette123[0x10000 * channel];
          for (size_t y = 0; y < ysize; ++y) {
            uint16_t* PIK_RESTRICT rowImg = img->PlaneRow(channel, y);
            for (size_t x = 0; x < xsize; ++x)  // UNROLL AND PARALLELIZE ME!
              rowImg[x] = palette[rowImg[x]];
          }
          compressedData[pos++] = (numColors[channel] - 1) & 255;
          compressedData[pos++] = (numColors[channel] - 1) >> 8;
        } else {
          numColors[channel] = 0x10000;
        }
      }

      uint8_t* pb =
          Palette_compress(numChannels, &bits[0], &palette123, &firstColors[0]);
      FWr(&compressedData[0], pos);
      FWr(&bits[0], sizeof(uint8_t) * (pb - &bits[0]));
    }  // if (flags)
    else
      numColors[0] = numColors[1] = numColors[2] = 0x10000;
  }

  bool PerChannelPalette_decompress(const uint8_t* compressedData,
                                    size_t compressedSize, size_t* pos,
                                    int numChannels, int imageMethod,
                                    std::vector<int>* palette) {
    const uint8_t* p_end = compressedData + compressedSize;
    const uint8_t* p = &compressedData[*pos];
    for (int channel = 0; channel < numChannels; ++channel) {
      if (imageMethod & (1 << channel)) {
        if (p + 1 >= p_end) return PIK_FAILURE("lossless16");
        numColors[channel] = p[0] + p[1] * 256 + 1;
        p += 2;
      } else
        numColors[channel] = 0x10000;
    }
    for (int channel = 0; channel < numChannels; ++channel) {
      if (imageMethod & (1 << channel)) {
        if (p + 1 >= p_end) return PIK_FAILURE("lossless16");
        int methodAndSize = p[0] + p[1] * 256, cSize = methodAndSize >> 1;
        p += 2;
        const uint8_t *p00 = p, *pcEnd = p00 + cSize;
        if (pcEnd >= p_end) return PIK_FAILURE("lossless16");
        if (methodAndSize & 1) {
          int x = 0, sumv = channel << 16, stop = sumv + numColors[channel];
          while (x < 0x10000) {
            if (p >= pcEnd) return PIK_FAILURE("lossless16");
            for (int b = *p++, i = 0; i < 8; ++i)
              (*palette)[sumv] = x++, sumv += b & 1, b >>= 1;
            if (sumv >= stop) break;
            if (sumv + 0x10000 - x == stop) {
              while (x < 0x10000) (*palette)[sumv++] = x++;
            }
          }  // while x
          continue;
        }  // if (methodAndSize & 1)

        uint32_t smt[64], x1 = 0, x2 = 0xffffffff, xr = 0, context6 = 0,
                          sumv = channel << 16,
                          stop = sumv + numColors[channel];
        for (int i = 0; i < 4; ++i) xr = (xr << 8) + (p >= pcEnd ? 0xFF : *p++);
        for (int i = 0; i < 64; ++i) smt[i] = smt0[i] << 11;
        for (int x = 0; x < 0x10000;) {
          int v;
          uint32_t pr = smt[context6] >> 11;

          uint32_t xmid =
              x1 + ((x2 - x1) >> 16) * pr + (((x2 - x1) & 0xffff) * pr >> 16);
          assert(pr >= 0 && pr <= 0xffff && xmid >= x1 && xmid < x2);
          if (xr <= xmid) {
            x2 = xmid, v = 1;
          } else {
            x1 = xmid + 1, v = 0;
          }

          while (((x1 ^ x2) & 0xff000000) ==
                 0) {  // Binary arithm decompression
            xr = (xr << 8) + (p >= pcEnd ? 0xFF : *p++);
            x1 <<= 8;
            x2 = (x2 << 8) + 255;
          }

          int p0 = smt[context6];
          p0 += ((v << (16 + 11)) - p0) * 5 >> 7;  // Learning rate
          smt[context6] = p0;
          context6 = (context6 * 2 + v) & 0x3f;
          (*palette)[sumv] = x++;
          sumv += v;
          if (sumv == stop) break;
          if (sumv + 0x10000 - x == stop) {
            while (x < 0x10000) (*palette)[sumv++] = x++;
          }
        }  // for x
        p = p00 + cSize;
      }  // if (imageMethod & ...
    }
    *pos = p - &compressedData[0];
    return true;
  }

  // Compresses 16-bit streams by splitting them into separate bytes,
  // possibly using the MSB stream as context for multiple LSB streams.
  bool Compress16BitStreams(bool contextMode, size_t esize[kNumContexts],
                            size_t compressedCapacity,
                            uint8_t* compressedOutput, size_t* pos) {
    std::vector<size_t> sizes(kNumContexts * 3, 0);
    std::vector<const uint8_t*> streams(kNumContexts * 3);
    std::vector<std::vector<uint8_t>> mem(kNumContexts * 3);
    size_t j = kNumContexts;
    for (size_t i = 0; i < kNumContexts; ++i) {
      size_t S = esize[i];
      mem[i].resize(S);
      if (S == 0) {
        continue;
      }
      streams[i] = mem[i].data();
      uint16_t* d = &edata[i][0];

      size_t numzero = 0;
      // first, compress MSBs (most significant bytes)
      for (size_t x = 0; x < S; ++x) {
        mem[i][x] = d[x] >> 8;
        sizes[i]++;
        if (mem[i][x] == 0) numzero++;
      }
      size_t smallValue = 0;
      size_t numSmall = 0;
      // Whether to force only a single LSB stream, or let it depend on the
      // MSB stream for context
      bool force_single =
          contextMode ? (i > 9 || S < 128) : (S < 120 || numzero < 120);
      if (force_single) {
        numSmall = S;
      } else {
        smallValue = numzero > (S * 13 >> 4) ? 2 : 1;
        for (size_t x = 0; x < S; ++x) {
          if (mem[i][x] < smallValue) numSmall++;
        }
      }
      size_t numLarge = S - numSmall;

      if (numSmall) {
        mem[j].resize(S);
        streams[j] = mem[j].data();
      }

      if (numLarge) {
        mem[j + 1].resize(S);
        streams[j + 1] = mem[j + 1].data();
      }

      // then, compress LSBs (least significant bytes)
      if (numLarge == 0) {
        for (size_t x = 0; x < S; ++x) {
          mem[j][x] = d[x] & 255;  // All
          sizes[j]++;
        }
      } else {
        for (size_t x = 0; x < S; ++x) {
          if (mem[i][x] < smallValue) {
            // LSBs such that MSB<2
            mem[j][sizes[j]++] = d[x] & 255;
          } else {
            // LSBs such that MSB>=2
            mem[j + 1][sizes[j + 1]++] = d[x] & 255;
          }
        }
      }
      if (numSmall) j++;
      if (numLarge) j++;
    }  // for i
    // Compress all the MSB's first ...
    PIK_RETURN_IF_ERROR(
        CompressWithEntropyCode(pos, sizes.data(), streams.data(), kNumContexts,
                                compressedCapacity, compressedOutput));
    // ... then all the LSB streams
    PIK_RETURN_IF_ERROR(CompressWithEntropyCode(
        pos, sizes.data() + kNumContexts, streams.data() + kNumContexts,
        j - kNumContexts, compressedCapacity, compressedOutput));

    return true;
  }

  // Compresses 16-bit streams by splitting them into separate bytes
  bool Decompress16BitStreams(size_t compressedSize,
                              const uint8_t* compressedData,
                              size_t expected_size, bool contextMode,
                              size_t* pos) {
    size_t decompressedSize = 0;  // is used only for return PIK_FAILURE
    std::vector<std::vector<uint8_t>> streams(kNumContexts * 3);
    // Decompress most significant bytes
    if (!DecompressWithEntropyCode(compressedSize, compressedData,
                                   expected_size, kNumContexts, streams.data(),
                                   pos)) {
      return PIK_FAILURE("entropy decode failed");
    }
    std::vector<size_t> smallCounts(kNumContexts, 0);
    // Values considered small for the MSB context
    std::vector<size_t> smallValues(kNumContexts, 0);
    // Amount of streams used for least significant bytes
    size_t numLSBStreams = 0;

    // Compute MSB statistics/context
    for (size_t i = 0; i < kNumContexts; ++i) {
      uint8_t* msb = streams[i].data();
      size_t size = streams[i].size();
      if (size == 0) continue;

      decompressedSize += size;

      size_t numZero = 0;
      for (size_t x = 0; x < size; ++x) {
        if (msb[x] == 0) numZero++;
      }
      size_t numSmall = 0;
      // Whether to force only a single LSB stream, or let it depend on the
      // MSB stream for context
      bool force_single =
          contextMode ? (i > 9 || size < 128) : (size < 120 || numZero < 120);
      if (force_single) {
        numSmall = size;
      } else {
        smallValues[i] = numZero > (size * 13 >> 4) ? 2 : 1;
        for (size_t x = 0; x < size; ++x) {
          if (msb[x] < smallValues[i]) numSmall++;
        }
      }
      size_t numLarge = size - numSmall;

      smallCounts[i] = numSmall;

      if (numSmall) numLSBStreams++;
      if (numLarge) numLSBStreams++;
    }  // for i
    if (decompressedSize != expected_size) {
      return PIK_FAILURE("wrong decompressed size");
    }

    // Decompress least significant bytes
    if (!DecompressWithEntropyCode(compressedSize, compressedData,
                                   expected_size, numLSBStreams,
                                   streams.data() + kNumContexts, pos)) {
      return PIK_FAILURE("entropy decode failed");
    }

    // Create the 16-bit values from the MSB's and LSB's
    size_t j = kNumContexts;
    for (int i = 0; i < kNumContexts; ++i) {
      // first, decompress MSBs (most significant bytes)
      uint8_t* msb = streams[i].data();
      size_t size = streams[i].size();
      if (size == 0) continue;

      uint16_t* dst = edata[i].data();

      size_t numSmall = smallCounts[i];
      size_t numLarge = size - numSmall;

      uint8_t* lsb0 = nullptr;
      if (numSmall) {
        if (j >= kNumContexts + numLSBStreams) {
          return PIK_FAILURE("LSB stream out of bounds");
        }
        if (streams[j].size() != numSmall) {
          return PIK_FAILURE("invalid LSB stream size");
        }
        lsb0 = streams[j].data();
        j++;
      }
      uint8_t* lsb1 = nullptr;
      if (numLarge) {
        if (j >= kNumContexts + numLSBStreams) {
          return PIK_FAILURE("LSB stream out of bounds");
        }
        if (streams[j].size() != numLarge) {
          return PIK_FAILURE("invalid LSB stream size");
        }
        lsb1 = streams[j].data();
        j++;
      }

      if (numLarge == 0) {  // All LSBs at once
        for (size_t k = 0; k < size; k++) {
          // MSB*256 + LSB
          dst[k] = msb[k] * 256 + lsb0[k];
        }
      } else {
        for (size_t k = 0; k < size; k++) {
          // Guaranteed not out of bounds because the amount of small values
          // was counted earlier and sizes of lsb0 and lsb1 checked.
          dst[k] = msb[k] * 256 + (msb[k] < smallValues[i] ? *lsb0++ : *lsb1++);
        }
      }
    }  // for i

    return true;
  }

  bool Grayscale16bit_compress(const ImageU& img_in, PaddedBytes* bytes) {
    // The code modifies the image for palette so must copy for now.
    ImageU img = CopyImage(img_in);

    size_t esize[kNumContexts], xsize = img.xsize(), ysize = img.ysize();
    std::vector<uint8_t> temp_buffer(groupSize2plus * 2);
    compressedData = temp_buffer.data();
    size_t compressedCapacity = temp_buffer.size();

    size_t pos = EncodeVarInt(xsize, &compressedData[0]);
    pos += EncodeVarInt(ysize, &compressedData[pos]);
    FWr(&compressedData[0], pos);

    numColors[0] = 0x10000;
    if (xsize * ysize >
        kPcpEnabled) {  // TODO(arhatushnyak): smarter decision making here
      PerChannelPalette_compress_1(&img, bytes);
    }
    maxTpv = numColors[0] - 1;
    maxTpvDiv2 = maxTpv >> 1;

    for (size_t groupY = 0; groupY < ysize; groupY += groupSizeY) {
      for (size_t groupX = 0; groupX < xsize; groupX += groupSizeX) {
        memset(esize, 0, sizeof(esize));
        for (size_t y = 0, yp = 0, yp1 = groupSizeY,
                    yEnd = std::min(groupSizeY, ysize - groupY);
             y < yEnd; ++y, yp = groupSizeX - yp, yp1 = groupSizeX - yp) {
          rowImg = img.Row(groupY + y) + groupX;
          rowPrev = (y == 0 ? nullptr : img.Row(groupY + y - 1) + groupX);
          width = std::min(groupSizeX, xsize - groupX) - 1;
          for (size_t x = 0; x <= width; ++x) {
            int maxErr, prediction = predict1(x, yp + x, yp1 + x, &maxErr);
            assert(0 <= maxErr && maxErr <= kNumContexts - 1);
            assert(0 <= prediction && prediction <= 0xffff);

            int truePixelValue = static_cast<int>(rowImg[x]);
            size_t s = esize[maxErr];
            edata[maxErr][s] = fwd_sign2LSB(prediction, truePixelValue);
            int err = prediction - truePixelValue;
            Update_Size_And_Errors;
          }  // x
        }    // y
        size_t pos = 0;

        if (!Compress16BitStreams(true, esize, compressedCapacity,
                                  compressedData, &pos)) {
          return false;
        }
        FWr(&compressedData[0], pos);
      }  // groupX
    }    // groupY
    return true;
  }

  bool Grayscale16bit_decompress(const Span<const uint8_t> bytes,
                                 size_t* bytes_pos, ImageU* result) {
    if (*bytes_pos > bytes.size()) return PIK_FAILURE("lossless16");
    size_t compressedSize = bytes.size() - *bytes_pos;
    const uint8_t* compressedData = bytes.data() + *bytes_pos;

    size_t esize[kNumContexts], xsize, ysize, pos0 = 0, imageMethod = 0;
    xsize = DecodeVarInt(compressedData, compressedSize, &pos0);
    ysize = DecodeVarInt(compressedData, compressedSize, &pos0);
    if (!xsize || !ysize) return PIK_FAILURE("lossless16");
    // Check maximum supported image size. Too large would run out of memory.
    // We use a division to avoid overflow when multiplying.
    if (ysize > kMaxLossless16PixelsY || xsize > kMaxLossless16Pixels / ysize)
      return PIK_FAILURE("lossless16");

    pik::ImageU img(xsize, ysize);
    std::vector<int> palette(0x10000);

    size_t pos = pos0;
    numColors[0] = 0x10000;
    if (xsize * ysize >
        kPcpEnabled) {  // TODO(arhatushnyak): smarter decision making here
      if (pos >= compressedSize)
        return PIK_FAILURE("lossless16: out of bounds");
      imageMethod = compressedData[pos++];
      if (imageMethod != 0) {
        PIK_RETURN_IF_ERROR(PerChannelPalette_decompress(
            compressedData, compressedSize, &pos, 1, imageMethod, &palette));
      }
    }
    maxTpv = numColors[0] - 1;
    maxTpvDiv2 = maxTpv >> 1;

    for (size_t groupY = 0; groupY < ysize; groupY += groupSizeY) {
      for (size_t groupX = 0; groupX < xsize; groupX += groupSizeX) {
        size_t expected_size = std::min(groupSizeY, ysize - groupY) *
                               std::min(groupSizeX, xsize - groupX);

        if (!Decompress16BitStreams(compressedSize, compressedData,
                                    expected_size, true, &pos)) {
          return false;
        }

// Disabled, because it is actually useful that the decoder supports decoding
// its own stream when contained inside a bigger stream and knows the correct
// end position.
#if 0
          if (groupY + groupSizeY >= ysize &&
              groupX + groupSizeX >= xsize)  // if last group
            assert(pos == compressedSize);
#endif

        memset(esize, 0, sizeof(esize));
        for (size_t y = 0, yEnd = std::min(groupSizeY, ysize - groupY), yp = 0,
                    yp1 = groupSizeY;
             y < yEnd; ++y, yp = groupSizeX - yp, yp1 = groupSizeX - yp) {
          rowImg = img.Row(groupY + y) + groupX;
          rowPrev = (y == 0 ? nullptr : img.Row(groupY + y - 1) + groupX);
          width = std::min(groupSizeX, xsize - groupX) - 1;
          for (size_t x = 0; x <= width; ++x) {
            int maxErr, prediction = predict1(x, yp + x, yp1 + x, &maxErr);
            assert(0 <= maxErr && maxErr <= kNumContexts - 1);
            assert(0 <= prediction && prediction <= 0xffff);

            size_t s = esize[maxErr];
            int truePixelValue = bck_sign2LSB(prediction, edata[maxErr][s]);
            rowImg[x] = truePixelValue;
            int err = prediction - truePixelValue;

            Update_Size_And_Errors;
          }  // x
        }    // y
      }      // groupX
    }        // groupY
    *bytes_pos += pos;
    if (imageMethod & 1) {
      for (size_t y = 0; y < ysize; ++y) {
        uint16_t* const PIK_RESTRICT rowImg = img.Row(y);
        for (size_t x = 0; x < xsize; ++x)  // UNROLL AND PARALLELIZE ME!
          rowImg[x] = palette[rowImg[x]];
      }
    }
    *result = std::move(img);
    return true;
  }

  bool dcmprs512x512(pik::Image3U* img, int planeToDecompress, size_t* pos,
                     size_t groupY, size_t groupX,
                     const uint8_t* compressedData, size_t compressedSize) {
    maxTpv = ((ncMap[planeMethod] & (1 << planeToDecompress))
                  ? numColors[planeToDecompress] - 1
                  : 0xffff);
    maxTpvDiv2 = maxTpv >> 1;

    size_t esize[kNumContexts], xsize = img->xsize(), ysize = img->ysize();
    memset(esize, 0, sizeof(esize));

    size_t expected_size = std::min(groupSizeY, ysize - groupY) *
                           std::min(groupSizeX, xsize - groupX);

    if (!Decompress16BitStreams(compressedSize, compressedData, expected_size,
                                false, pos)) {
      return false;
    }

    size_t yEnd = std::min(groupSizeY, ysize - groupY);
    width = std::min(groupSizeX, xsize - groupX) - 1;
    size_t area = yEnd * (width + 1);
    int maxerrShift =
        (area > 25600
             ? 0
             : area > 12800 ? 1 : area > 2800 ? 2 : area > 512 ? 3 : 4);
    int maxerrAdd = (1 << maxerrShift) - 1;

    for (size_t y = 0, yp = 0, yp1 = groupSizeY; y < yEnd;
         ++y, yp = groupSizeX - yp, yp1 = groupSizeX - yp) {
      rowImg = img->PlaneRow(planeToDecompress, groupY + y) + groupX;
      rowPrev =
          (y == 0 ? nullptr
                  : img->PlaneRow(planeToDecompress, groupY + y - 1) + groupX);
      for (size_t x = 0; x <= width; ++x) {
        int maxErr, prediction = predict1(x, yp + x, yp1 + x, &maxErr);
        maxErr = (maxErr + maxerrAdd) >> maxerrShift;
        assert(0 <= maxErr && maxErr <= kNumContexts - 1);
        assert(0 <= prediction && prediction <= 0xffff);

        size_t s = esize[maxErr];
        int truePixelValue = bck_sign2LSB(prediction, edata[maxErr][s]);
        rowImg[x] = truePixelValue;
        int err = prediction - truePixelValue;

        Update_Size_And_Errors;
      }  // x
    }    // y
    return true;
  }

  bool Colorful16bit_decompress(const Span<const uint8_t> bytes,
                                size_t* bytes_pos, Image3U* result) {
    if (*bytes_pos > bytes.size()) return PIK_FAILURE("lossless16");
    size_t compressedSize = bytes.size() - *bytes_pos;
    const uint8_t* compressedData = bytes.data() + *bytes_pos;

    size_t xsize, ysize, pos0 = 0, imageMethod = 0;
    xsize = DecodeVarInt(compressedData, compressedSize, &pos0);
    ysize = DecodeVarInt(compressedData, compressedSize, &pos0);
    if (!xsize || !ysize) return PIK_FAILURE("lossless16");
    // Check maximum supported image size. Too large would run out of memory.
    // We use a division to avoid overflow when multiplying.
    if (ysize > kMaxLossless16PixelsY || xsize > kMaxLossless16Pixels / ysize)
      return PIK_FAILURE("lossless16");

    pik::Image3U img(xsize, ysize);
    std::vector<int> palette(0x10000 * 3);

    size_t pos = pos0;
    if (pos >= compressedSize) return PIK_FAILURE("lossless16: out of bounds");

    numColors[0] = numColors[1] = numColors[2] = 0x10000;
    if (xsize * ysize >
        kPcpEnabled) {  // TODO(arhatushnyak): smarter decision making here
      imageMethod = compressedData[pos++];
      if (imageMethod) {
        PIK_RETURN_IF_ERROR(PerChannelPalette_decompress(
            compressedData, compressedSize, &pos, 3, imageMethod, &palette));
      }
    }  // if (xsize*ysize ...

    decompress3planes(uint16_t, "lossless16");

// Disabled, because it is actually useful that the decoder supports decoding
// its own stream when contained inside a bigger stream and knows the correct
// end position.
#if 0
          assert(pos == compressedSize);
#endif

    for (int channel = 0; channel < 3; ++channel) {
      if (imageMethod & (1 << channel)) {
        int* p = &palette[0x10000 * channel];
        for (size_t y = 0; y < ysize; ++y) {
          uint16_t* const PIK_RESTRICT rowImg = img.PlaneRow(channel, y);
          for (size_t x = 0; x < xsize; ++x)  // UNROLL AND PARALLELIZE ME!
            rowImg[x] = p[rowImg[x]];
        }
      }
    }
    *bytes_pos += pos;
    *result = std::move(img);
    return true;
  }

#define Unfinished_StoreUncompressibleLSBs 0
  // 507017622 ==> 505782136  -- 99.756%  on imagecompression.info 16-bit
  // 434261965 ==> 433161316  -- 99.746%  on imagecompression.info 16-bit linear
  // 126989561 ==> 126934961  -- 99.957%  on Lode's /dc4/16bit/
  //   2284466 ==>   2261066  -- 98.976%  on Lode's /dc3/16bit/
  //
  // Brute-force, try 1...8:
  // 505632343  99.727%
  // 432940889  99.696%
  // 126915753  99.942%
  //   2258972  98.884%

#if Unfinished_StoreUncompressibleLSBs
  int maxFqei;
  uint16_t saved[512][512];

  bool cmprs512x512(pik::Image3U* img, int planeToCompress, int planeToUse,
                    size_t groupY, size_t groupX, size_t compressedCapacity,
                    uint8_t* compressedOutput, size_t* csize) {
    cmprs512x512_0(img, planeToCompress, planeToUse, groupY, groupX,
                   compressedCapacity, compressedOutput, csize);

    int m = (maxFqei <= kWithSign * 2 ? maxFqei / 2 : maxFqei - kWithSign);
    if (m <= 2) return true;

    m = std::min(5, (m * 3 - 1) >> 2);  //  3,4,5,6,7+  ===>  2,2,3,4,5

    int bestmi = -99;
    // const int start = 1, stop = 8;  // VERY SLOW! Try 8 values
    const int start = m, stop = m;
    for (int mi = start; mi <= stop; ++mi) {
      size_t xsize = img->xsize(), ysize = img->ysize(), csize0 = *csize;
      size_t yEnd = std::min(groupSizeY, ysize - groupY);
      width = std::min(groupSizeX, xsize - groupX) - 1;
      for (size_t y = 0; y < yEnd; ++y) {
        rowImg = img->PlaneRow(planeToCompress, groupY + y) + groupX;
        for (size_t x = 0; x <= width; ++x)
          saved[y][x] = rowImg[x], rowImg[x] >>= mi;
      }

      cmprs512x512_0(img, planeToCompress, planeToCompress,  // ! no planeToUse!
                     groupY, groupX, compressedCapacity, compressedOutput,
                     csize);
      for (size_t y = 0; y < yEnd; ++y) {
        rowImg = img->PlaneRow(planeToCompress, groupY + y) + groupX;
        for (size_t x = 0; x <= width; ++x) rowImg[x] = saved[y][x];
      }

      *csize +=
          (((width + 1) * yEnd * mi) + 7) >> 3;  // TODO store bits, no fake!
      if (csize0 <= *csize)
        *csize = csize0;
      else
        bestmi = mi;
    }
    if (bestmi != -99) printf("%d  %d\n", bestmi - m, bestmi);
    return true;
  }
#else
#define cmprs512x512 cmprs512x512_0
#endif

  bool cmprs512x512_0(pik::Image3U* img, int planeToCompress, int planeToUse,
                      size_t groupY, size_t groupX, size_t compressedCapacity,
                      uint8_t* compressedOutput, size_t* csize) {
#if Unfinished_StoreUncompressibleLSBs
    int fqe[kNumContexts];  // frequences of quantized errors
    memset(fqe, 0, sizeof(fqe));
#endif
    size_t esize[kNumContexts], xsize = img->xsize(), ysize = img->ysize();
    memset(esize, 0, sizeof(esize));
    size_t yEnd = std::min(groupSizeY, ysize - groupY);
    width = std::min(groupSizeX, xsize - groupX) - 1;
    size_t area = yEnd * (width + 1);
    int maxerrShift =
        (area > 25600
             ? 0
             : area > 12800 ? 1 : area > 2800 ? 2 : area > 512 ? 3 : 4);
    int maxerrAdd = (1 << maxerrShift) - 1;
    maxTpv = (planeToCompress == planeToUse ? numColors[planeToCompress] - 1
                                            : 0xffff);
    maxTpvDiv2 = maxTpv >> 1;

    for (size_t y = 0, yp = 0, yp1 = groupSizeY; y < yEnd;
         ++y, yp = groupSizeX - yp, yp1 = groupSizeX - yp) {
      rowImg = img->PlaneRow(planeToCompress, groupY + y) + groupX;
      rowPrev = (!y ? nullptr
                    : img->PlaneRow(planeToCompress, groupY + y - 1) + groupX);
      uint16_t* PIK_RESTRICT rowUse =
          img->PlaneRow(planeToUse, groupY + y) + groupX;
      for (size_t x = 0; x <= width; ++x) {
        int maxErr, prediction = predict1(x, yp + x, yp1 + x, &maxErr);
        maxErr = (maxErr + maxerrAdd) >> maxerrShift;
        assert(0 <= maxErr && maxErr <= kNumContexts - 1);
        assert(0 <= prediction && prediction <= 0xffff);
        int truePixelValue = static_cast<int>(rowImg[x]);
        if (planeToCompress != planeToUse) {
          truePixelValue -= static_cast<int>(rowUse[x]) - 0x8000;
          truePixelValue &= 0xffff;
          rowImg[x] = truePixelValue;
        }
        int err = prediction - truePixelValue;
        size_t s = esize[maxErr];
        edata[maxErr][s] = fwd_sign2LSB(prediction, truePixelValue);
        Update_Size_And_Errors;
#if Unfinished_StoreUncompressibleLSBs
        ++fqe[nbitErr[yp + x]];
#endif
      }  // x
    }    // y

    size_t pos = 0;
#if Unfinished_StoreUncompressibleLSBs
    int maxFqe = 0;
    maxFqei = 0;
    for (int i = 1; i <= kWithSign; ++i)
      fqe[i * 2] += fqe[i * 2 - 1], fqe[i * 2 - 1] = 0;
    for (int i = 0; i < kNumContexts; ++i)
      if (fqe[i] > maxFqe) maxFqe = fqe[i], maxFqei = i;
#endif

    if (!Compress16BitStreams(false, esize, compressedCapacity,
                              compressedOutput, &pos)) {
      return false;
    }

    *csize = pos;
    return true;
  }

  bool Colorful16bit_compress(const Image3U& img_in, PaddedBytes* bytes) {
    // The code modifies the image for palette so must copy for now.
    Image3U img = CopyImage(img_in);

    std::vector<uint8_t> temp_buffer(groupSize2plus * 2 * 6);
    compressedData = temp_buffer.data();

    size_t xsize = img.xsize(), ysize = img.ysize(), pos;
    pos = EncodeVarInt(xsize, &compressedData[0]);
    pos += EncodeVarInt(ysize, &compressedData[pos]);
    FWr(&compressedData[0], pos);

    numColors[0] = numColors[1] = numColors[2] = 0x10000;
    if (xsize * ysize >
        kPcpEnabled) {  // TODO(arhatushnyak): smarter decision making here
      PerChannelPalette_compress_3(&img, bytes);
    }

    compress3planes(uint16_t);

    return true;
  }
};  // struct State

}  // namespace

static constexpr size_t kGroupSize = 512;

bool Grayscale16bit_compress(const ImageU& img, PaddedBytes* bytes) {
  std::unique_ptr<State> state(new State(kGroupSize, kGroupSize));
  return state->Grayscale16bit_compress(img, bytes);
}

bool Grayscale16bit_decompress(const Span<const uint8_t> bytes, size_t* pos,
                               ImageU* result) {
  std::unique_ptr<State> state(new State(kGroupSize, kGroupSize));
  return state->Grayscale16bit_decompress(bytes, pos, result);
}

bool Colorful16bit_compress(const Image3U& img, PaddedBytes* bytes) {
  std::unique_ptr<State> state(new State(kGroupSize, kGroupSize));
  return state->Colorful16bit_compress(img, bytes);
}

bool Colorful16bit_decompress(const Span<const uint8_t> bytes, size_t* pos,
                              Image3U* result) {
  std::unique_ptr<State> state(new State(kGroupSize, kGroupSize));
  return state->Colorful16bit_decompress(bytes, pos, result);
}

}  // namespace pik
