// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

// @author Alexander Rhatushnyak

#include "lossless16.h"

#include <cmath>

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "entropy_coder.h"

namespace pik {

namespace {

static const int kGroupSize = 512,
                 kGroupSize2plus = kGroupSize * kGroupSize * 9 / 8,
                 WithSIGN_1 = 4, BitsMAX_1 = 13,
                 NUMCONTEXTS_1 = 1 + WithSIGN_1 + BitsMAX_1, WithSIGN_3 = 3,
                 BitsMAX_3 = 13, NUMCONTEXTS_3 = 1 + WithSIGN_3 + BitsMAX_3,
                 MAXERROR = 0x3fbf, MaxSumErrors = (MAXERROR + 1) * 4,
                 NumRuns = 1;

enum PlaneMethods {
  NoPlaneTransform = 0,
  SubtractFirstPlane = 1,
  SubtractAverageOfTwoPlanes = 2,
};

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
// TODO(lode): move this to ans_encode.h
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
// TODO(lode): move this to ans_decode.h
bool EntropyDecode(const uint8_t* data, size_t size,
                   std::vector<uint8_t>* result) {
  static const int kAlphabetSize = 256;
  static const int kContext = 0;
  size_t pos = 0;
  size_t num_symbols = decodeVarInt(data, size, &pos);
  if (pos >= size) {
    return PIK_FAILURE("lossless16");
  }
  // TODO(lode): instead take expected decoded size as function parameter
  if (num_symbols > 16777216) {
    // Avoid large allocations, we never expect this many symbols for
    // the limited group sizes.
    return PIK_FAILURE("lossless16");
  }

  BitReader br(data + pos, size - pos);
  ANSCode codes;
  if (!DecodeANSCodes(1, kAlphabetSize, &br, &codes)) {
    return PIK_FAILURE("lossless16");
  }

  result->resize(num_symbols);
  ANSSymbolReader reader(&codes);
  for (size_t i = 0; i < num_symbols; i++) {
    br.FillBitBuffer();
    int read_symbol = reader.ReadSymbol(kContext, &br);
    (*result)[i] = read_symbol;
  }
  if (!reader.CheckANSFinalState()) {
    return PIK_FAILURE("lossless16");
  }

  return true;
}

// TODO(lode): avoid the copying between std::vector and data.
// Entropy encode with pik ANS
static size_t EntropyEncode(const uint8_t* data, size_t size, uint8_t* out,
                            size_t out_capacity) {
  std::vector<uint8_t> result;
  if (!EntropyEncode(data, size, &result)) {
    return 0;  // Encoding error
  }
  if (result.size() > out_capacity) {
    return 0;  // Error: not enough capacity
  }
  memcpy(out, result.data(), result.size());
  return result.size();
}

// Entropy decode with pik ANS
static size_t EntropyDecode(const uint8_t* data, size_t size, uint8_t* out,
                            size_t out_capacity) {
  std::vector<uint8_t> result;
  if (!EntropyDecode(data, size, &result)) {
    return 0;  // Decoding error
  }
  if (result.size() > out_capacity) {
    return 0;  // Error: not enough capacity
  }
  memcpy(out, result.data(), result.size());
  return result.size();
}

// TODO(lode): split state variables needed for encoder from those for decoder
//             and perform one-time global initialization where possible.
struct State {
  int width, prediction0, prediction1, prediction2, prediction3, WithSIGN,
      BitsMAX, NUMCONTEXTS;
  uint16_t *PIK_RESTRICT rowImg, *PIK_RESTRICT rowPrev;

  uint16_t edata[NUMCONTEXTS_1 > NUMCONTEXTS_3 ? NUMCONTEXTS_1 : NUMCONTEXTS_3]
                [kGroupSize * kGroupSize];
  uint8_t compressedDataTmpBuf[kGroupSize2plus], *compressedData;
  int32_t errors0[kGroupSize * 2];  // Errors of predictor 0
  int32_t errors1[kGroupSize * 2];  // Errors of predictor 1
  int32_t errors2[kGroupSize * 2];  // Errors of predictor 2
  int32_t errors3[kGroupSize * 2];  // Errors of predictor 3
  uint8_t nbitErr[kGroupSize * 2];
  int32_t trueErr[kGroupSize * 2];

  uint16_t sign_LSB_forward_transform[0x10000],
      error2weight[MaxSumErrors],  // const
      sign_LSB_backward_transform[0x10000];
  uint8_t numBitsTable[256];  // const

  State() {
    for (int i = 0; i < 256; ++i)
      numBitsTable[i] = numbitsInit(i);  // const init
    error2weight[0] = 0xffff;
    for (int j = 1; j < MaxSumErrors; ++j)
      error2weight[j] = 181 * 256 / j;  // 181          // const init!

    // For compress
    for (int i = 0; i < 256 * 256; ++i)
      sign_LSB_forward_transform[i] =
          (i & 32768 ? (0xffff - i) * 2 + 1 : i * 2);  // const init!

    // For decompress
    for (int i = 0; i < 256 * 256; ++i)
      sign_LSB_backward_transform[i] =
          (i & 1 ? 0xffff - (i >> 1) : i >> 1);  // const init!
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
    return std::min(8 + numBitsTable[x >> 8], BitsMAX);
  }

  PIK_INLINE int predict1y0(size_t x, size_t yp, size_t yp1, int& maxErr) {
    maxErr = (x == 0 ? NUMCONTEXTS - 1
                     : x == 1 ? nbitErr[yp - 1]
                              : std::max(nbitErr[yp - 1], nbitErr[yp - 2]));
    prediction0 = prediction1 = prediction2 = prediction3 =
        (x == 0 ? 14 * 256  // 14
                : x == 1 ? rowImg[x - 1]
                         : rowImg[x - 1] + (rowImg[x - 1] - rowImg[x - 2]) / 4);
    return (prediction0 < 0 ? 0 : prediction0 > 0xffff ? 0xffff : prediction0);
  }

  PIK_INLINE int predict1x0(size_t x, size_t yp, size_t yp1, int& maxErr) {
    maxErr = std::max(nbitErr[yp1], nbitErr[yp1 + (x < width ? 1 : 0)]);
    prediction0 = prediction2 = prediction3 = rowPrev[x];
    prediction1 = (rowPrev[x] * 3 + rowPrev[x + (x < width ? 1 : 0)] + 2) >> 2;
    return prediction1;
  }

  PIK_INLINE int predict1(size_t x, size_t yp, size_t yp1, int& maxErr) {
    if (!rowPrev) return predict1y0(x, yp, yp1, maxErr);
    if (x == 0LL) return predict1x0(x, yp, yp1, maxErr);
    int weight0 = errors0[yp - 1] + errors0[yp1] + errors0[yp1 - 1];
    int weight1 = errors1[yp - 1] + errors1[yp1] + errors1[yp1 - 1];
    int weight2 = errors2[yp - 1] + errors2[yp1] + errors2[yp1 - 1];
    int weight3 = errors3[yp - 1] + errors3[yp1] + errors3[yp1 - 1];
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

    weight0 = error2weight[weight0] + 1;
    weight1 = error2weight[weight1] + 1;
    weight2 = error2weight[weight2];
    weight3 = error2weight[weight3];

    int teW = trueErr[yp - 1];  // range: -0xffff...0xffff
    int teN = trueErr[yp1];
    int teNW = trueErr[yp1 - 1];
    int sumWN = teN + teW;  // range: -0x1fffe...0x1fffe
    int teNE = (x < width ? trueErr[yp1 + 1] : 0);

    prediction0 = N - sumWN * 3 / 4;                          // 24/32
    prediction1 = W - (sumWN + teNW) * 11 / 32;               // 11/32
    prediction2 = W + (((NE - N) * 13 + 7) >> 4);             // 26/32
    prediction3 = N - (((teN + teNW + teNE) * 7 + 29) >> 5);  //  7/32
    int sumWeights = weight0 + weight1 + weight2 + weight3;
    int64_t s = sumWeights * 3 / 8;
    s += ((int64_t)prediction0) * weight0;
    s += ((int64_t)prediction1) * weight1;
    s += ((int64_t)prediction2) * weight2;
    s += ((int64_t)prediction3) * weight3;
    int prediction = s / sumWeights;

    if (mxe && mxe <= WithSIGN * 2) {
      if (teW * 3 + teN * 2 + teNW + teNE < 0) --mxe;  // 3 2 1 1
    }
    maxErr = mxe;

    int mx = std::max(N - 28, std::max(W, NE));  // 28
    int mn = std::min(N + 28, std::min(W, NE));  // 28
    prediction = std::max(mn, std::min(mx, prediction));
    return prediction;
  }

  bool IsRLE(const uint8_t* data, size_t size) {
    if (size < 4) return false;
    uint8_t first = data[0];
    for (size_t i = 1; i < size; i++) {
      if (data[i] != first) return false;
    }
    return true;
  }

  bool compressWithEntropyCode(size_t* pos, size_t S, uint8_t* compressedBuf) {
    uint8_t* src = &compressedBuf[*pos + 8];
    size_t cs;
    if (IsRLE(src, S)) {
      cs = 1;  // use RLE encoding instead
    } else {
      cs = EntropyEncode(src, S, &compressedDataTmpBuf[0],
                         sizeof(compressedDataTmpBuf));
      if (!cs) return PIK_FAILURE("lossless16");  // error
    }
    if (cs >= S) cs = 0;  // EntropyCode worse than original, use memcpy.
    *pos += encodeVarInt(cs <= 1 ? (S - 1) * 3 + 1 + cs : cs * 3,
                         &compressedBuf[*pos]);
    uint8_t* dst = &compressedBuf[*pos];
    if (cs == 1)
      compressedBuf[(*pos)++] = *src;
    else if (cs == 0)
      memmove(dst, src, S), *pos += S;
    else
      memcpy(dst, &compressedDataTmpBuf[0], cs), *pos += cs;
    return true;
  }

  // Returns decompressed size, or 0 on error.
  size_t decompressWithEntropyCode(uint8_t* dst, size_t dst_capacity,
                                   const uint8_t* src, size_t src_capacity,
                                   size_t cs, size_t* pos) {
    size_t mode = cs % 3, ds;
    cs /= 3;
    if (mode == 2) {
      if (*pos >= src_capacity) return 0;
      if (cs + 1 > dst_capacity) return 0;
      memset(dst, src[(*pos)++], ++cs);
      return cs;
    }
    if (mode == 1) {
      if (*pos + cs + 1 > src_capacity) return 0;
      if (cs + 1 > dst_capacity) return 0;
      memcpy(dst, &src[*pos], ++cs);
      *pos += cs;
      return cs;
    }
    if (*pos + cs > src_capacity) return 0;
    ds = EntropyDecode(&src[*pos], cs, dst, dst_capacity);
    *pos += cs;
    return ds;
  }

#define Update_Errors_0_1_2_3                                  \
  err = prediction0 - truePixelValue;                          \
  if (err < 0) err = -err; /* abs() and min()? worse speed! */ \
  if (err > MAXERROR) err = MAXERROR;                          \
  errors0[yp + x] = err;                                       \
  err = prediction1 - truePixelValue;                          \
  if (err < 0) err = -err;                                     \
  if (err > MAXERROR) err = MAXERROR;                          \
  errors1[yp + x] = err;                                       \
  err = prediction2 - truePixelValue;                          \
  if (err < 0) err = -err;                                     \
  if (err > MAXERROR) err = MAXERROR;                          \
  errors2[yp + x] = err;                                       \
  err = prediction3 - truePixelValue;                          \
  if (err < 0) err = -err;                                     \
  if (err > MAXERROR) err = MAXERROR;                          \
  errors3[yp + x] = err;

#define Update_Size_And_Errors                                    \
  ++esize[maxErr];                                                \
  trueErr[yp + x] = err;                                          \
  err = numBits(err >= 0 ? err : -err);                           \
  nbitErr[yp + x] = (err <= WithSIGN ? err * 2 : err + WithSIGN); \
  Update_Errors_0_1_2_3

  bool Grayscale16bit_compress(const ImageU& img_in, PaddedBytes* bytes) {
    WithSIGN = WithSIGN_1, BitsMAX = BitsMAX_1, NUMCONTEXTS = NUMCONTEXTS_1;

    // The code modifies the image for palette so must copy for now.
    ImageU img = CopyImage(img_in);

    size_t xsize = img.xsize(), ysize = img.ysize();
    std::vector<size_t> esize(NUMCONTEXTS);
    std::vector<uint8_t> temp_buffer(kGroupSize2plus * 2);
    compressedData = temp_buffer.data();

#if 0  // Let's look whether the image was dequantized, i.e. the range is ~64k,
       // but there are only ~1000 values or so.
      int mn = 65535, mx = 0, palette[65536];
      memset(palette, 0, sizeof(palette));
      for (size_t y = 0; y < ysize; ++y) {
        uint16_t *const PIK_RESTRICT rowImg = img.Row(y);
        for (size_t x = 0; x < xsize; ++x) {
          int v = rowImg[x];
          ++palette[v];
          mn = std::min(mn, v);
          mx = std::max(mx, v);
        }
      }
      // count the number of pixel values present in the image
      int count = 0;
      for (int i = mn; i <= mx; ++i)
        if (palette[i]) palette[i] = count++;
      printf("min=%4d  max=%5d  range=%5d,  count=%5d, %f%%\n", mn, mx,
             mx + 1 - mn, count, count * 100. / (mx + 1 - mn));

#if 1
      // Re-quantize!   Lossy!   to make it lossless, we'd need up to (count*2)
      // bytes to store the 'palette'
      for (size_t y = 0; y < ysize; ++y) {
        uint16_t *const PIK_RESTRICT rowImg = img.Row(y);
        for (size_t x = 0; x < xsize; ++x) {
          int v = rowImg[x];
          rowImg[x] = palette[v];
        }
      }
#endif
#endif

    clock_t start = clock();
    for (int run = 0; run < NumRuns; ++run) {
      for (size_t groupY = 0; groupY < ysize; groupY += kGroupSize) {
        for (size_t groupX = 0; groupX < xsize; groupX += kGroupSize) {
          memset(esize.data(), 0, esize.size() * sizeof(esize[0]));
          for (size_t y = 0,
                      yEnd = std::min((size_t)kGroupSize, ysize - groupY),
                      yp = 0, yp1;
               y < yEnd; ++y, yp ^= kGroupSize, yp1 = kGroupSize - yp) {
            rowImg = img.Row(groupY + y) + groupX;
            rowPrev = (y == 0 ? NULL : img.Row(groupY + y - 1) + groupX);
            width = std::min((size_t)kGroupSize, xsize - groupX) - 1;
            for (size_t x = 0; x <= width; ++x) {
              int maxErr, prediction = predict1(x, yp + x, yp1 + x, maxErr);
              // maxErr=0; // SETTING it 0 here DISABLES ERROR CONTEXT MODELING!
              assert(0 <= maxErr && maxErr <= NUMCONTEXTS_1 - 1);
              assert(0 <= prediction && prediction <= 0xffff);

              int truePixelValue = (int)rowImg[x];
              int err = prediction - truePixelValue;
              size_t s = esize[maxErr];
              edata[maxErr][s] = sign_LSB_forward_transform[err & 0xffff];

              Update_Size_And_Errors
            }  // x
          }    // y
          size_t pos = 0;
          if (groupY + groupX == 0) {
            pos += encodeVarInt(xsize, &compressedData[pos]);
            pos += encodeVarInt(ysize, &compressedData[pos]);
          }
          for (int i = 0; i < NUMCONTEXTS_1; ++i) {
            size_t S = esize[i];
            if (S == 0) {
              pos += encodeVarInt(0, &compressedData[pos]);
              continue;
            }
            uint16_t* d = &edata[i][0];
            // first, compress MSBs (most significant bytes)
            uint8_t* p = &compressedData[pos + 8];
            for (size_t x = 0; x < S; ++x) p[x] = d[x] >> 8;
            if (!compressWithEntropyCode(&pos, S, compressedData))
              return PIK_FAILURE("lossless16");

            if (i > 9 || S < 128) {  //  9  128
              // then, compress LSBs (least significant bytes)
              p = &compressedData[pos + 8];
              for (size_t x = 0; x < S; ++x) p[x] = d[x] & 255;  // All
              if (!compressWithEntropyCode(&pos, S, compressedData))
                return PIK_FAILURE("lossless16");
            } else {
              p = &compressedData[pos + 8];
              size_t y = 0;
              for (size_t x = 0; x < S; ++x)
                if (d[x] < 256) p[y++] = d[x] & 255;  // LSBs such that MSB==0
              if (y) {
                if (!compressWithEntropyCode(&pos, y, compressedData))
                  return PIK_FAILURE("lossless16");
              }

              p = &compressedData[pos + 8];
              y = 0;
              for (size_t x = 0; x < S; ++x)
                if (d[x] >= 256) p[y++] = d[x] & 255;  // LSBs such that MSB!=0
              if (y) {
                if (!compressWithEntropyCode(&pos, y, compressedData))
                  return PIK_FAILURE("lossless16");
              }
            }  // if (i > 9)
          }    // i
          if (run == 0) {
            size_t current = bytes->size();
            bytes->resize(bytes->size() + pos);
            memcpy(bytes->data() + current, &compressedData[0], pos);
          }
        }  // groupX
      }    // groupY
    }      // run
    if (NumRuns > 1)
      printf("%d runs, %1.5f seconds", NumRuns,
             ((double)clock() - start) / CLOCKS_PER_SEC);
    return true;
  }

  bool Grayscale16bit_decompress(const PaddedBytes& bytes, size_t* bytes_pos,
                                 ImageU* result) {
    WithSIGN = WithSIGN_1, BitsMAX = BitsMAX_1, NUMCONTEXTS = NUMCONTEXTS_1;
    if (*bytes_pos > bytes.size()) return PIK_FAILURE("lossless16");
    size_t compressedSize = bytes.size() - *bytes_pos;
    const uint8_t* compressedData = bytes.data() + *bytes_pos;

    // Size of an edata entry
    size_t maxDecodedSize = kGroupSize * kGroupSize;
    // Size of a compressedDataTmpBuf entry
    size_t maxDecodedSize2 = kGroupSize2plus;

    size_t esize[NUMCONTEXTS_1], xsize, ysize, pos = 0;
    xsize = decodeVarInt(compressedData, compressedSize, &pos);
    ysize = decodeVarInt(compressedData, compressedSize, &pos);
    if (!xsize || !ysize) return PIK_FAILURE("lossless16");
    // Too large, would run out of memory. Chosen as reasonable limit for pik
    // while being below default fuzzer memory limit. We check for total pixel
    // size, and an additional restriction to ysize, because large ysize
    // consumes more memory due to the scanline padding.
    if (uint64_t(xsize) * uint64_t(ysize) >= 134217728ull || ysize >= 65536) {
      return PIK_FAILURE("lossless16");
    }
    pik::ImageU img(xsize, ysize);

    clock_t start = clock();
    for (int run = 0; run < NumRuns; ++run) {
      size_t pos = 0;
      for (size_t groupY = 0; groupY < ysize; groupY += kGroupSize) {
        for (size_t groupX = 0; groupX < xsize; groupX += kGroupSize) {
          size_t decompressedSize = 0;  // is used only for the assert()

          if (groupY + groupX == 0) {
            decodeVarInt(compressedData, compressedSize, &pos);
            decodeVarInt(compressedData, compressedSize, &pos);
          }
          for (int i = 0; i < NUMCONTEXTS_1; ++i) {
            size_t cs = decodeVarInt(compressedData, compressedSize, &pos), ds,
                   ds1, ds2, ds3;
            if (cs == 0) continue;
            // first, decompress MSBs (most significant bytes)
            ds1 = decompressWithEntropyCode((uint8_t*)&edata[i][0],
                                            maxDecodedSize, compressedData,
                                            compressedSize, cs, &pos);
            if (!ds1) return PIK_FAILURE("lossless16");

            if (i > 9 || ds1 < 128) {  // All LSBs at once
              cs = decodeVarInt(compressedData, compressedSize, &pos);
              ds2 = decompressWithEntropyCode(&compressedDataTmpBuf[0],
                                              maxDecodedSize2, compressedData,
                                              compressedSize, cs, &pos);
              if (ds1 != ds2) return PIK_FAILURE("lossless16");
              uint16_t* dst = &edata[i][0];
              uint8_t* p = (uint8_t*)dst;
              for (int j = ds1 - 1; j >= 0; --j)
                dst[j] = p[j] * 256 + compressedDataTmpBuf[j];  // MSB*256 + LSB
            } else {
              uint16_t* dst = &edata[i][0];
              uint8_t* p = (uint8_t*)dst;
              ds2 = ds3 = 0;
              for (int j = ds1 - 1; j >= 0; --j)
                if (p[j])
                  ++ds3;
                else
                  ++ds2;

              if (ds2) {  // LSBs such that MSB==0
                cs = decodeVarInt(compressedData, compressedSize, &pos);
                ds = decompressWithEntropyCode(&compressedDataTmpBuf[0],
                                               maxDecodedSize2, compressedData,
                                               compressedSize, cs, &pos);
                if (!ds) return PIK_FAILURE("lossless16");
                if (ds != ds2) return PIK_FAILURE("lossless16");
              }

              if (ds3) {  // LSBs such that MSB!=0
                cs = decodeVarInt(compressedData, compressedSize, &pos);
                ds = decompressWithEntropyCode(&compressedDataTmpBuf[ds2],
                                               maxDecodedSize2, compressedData,
                                               compressedSize, cs, &pos);
                if (!ds) return PIK_FAILURE("lossless16");
                if (ds != ds3) return PIK_FAILURE("lossless16");
              }
              uint8_t *p2 = &compressedDataTmpBuf[ds2 - 1],
                      *p3 = &compressedDataTmpBuf[ds1 - 1];  // Note ds1=ds2+ds3
              for (int j = ds1 - 1; j >= 0; --j)
                dst[j] = p[j] * 256 + (p[j] == 0 ? *p2-- : *p3--);
            }
            decompressedSize += ds1;
          }  // for i
          if (!(decompressedSize ==
                std::min((size_t)kGroupSize, ysize - groupY) *
                    std::min((size_t)kGroupSize, xsize - groupX))) {
            return PIK_FAILURE("lossless16");
          }
// Disabled, because it is actually useful that the decoder supports decoding
// its own stream when contained inside a bigger stream and knows the correct
// end position.
#if 0
          if (groupY + kGroupSize >= ysize &&
              groupX + kGroupSize >= xsize)  // if last group
            assert(pos == compressedSize);
#endif

          memset(esize, 0, sizeof(esize));
          for (size_t y = 0,
                      yEnd = std::min((size_t)kGroupSize, ysize - groupY),
                      yp = 0, yp1;
               y < yEnd; ++y, yp ^= kGroupSize, yp1 = kGroupSize - yp) {
            rowImg = img.Row(groupY + y) + groupX;
            rowPrev = (y == 0 ? NULL : img.Row(groupY + y - 1) + groupX);
            width = std::min((size_t)kGroupSize, xsize - groupX) - 1;
            for (size_t x = 0; x <= width; ++x) {
              int maxErr, prediction = predict1(x, yp + x, yp1 + x, maxErr);
              // maxErr=0; // SETTING it 0 here DISABLES ERROR CONTEXT MODELING!
              assert(0 <= maxErr && maxErr <= NUMCONTEXTS_1 - 1);
              assert(0 <= prediction && prediction <= 0xffff);

              size_t s = esize[maxErr];
              int err = edata[maxErr][s];
              int truePixelValue =
                  (prediction - sign_LSB_backward_transform[err]) & 0xffff;
              rowImg[x] = truePixelValue;
              err = prediction - truePixelValue;

              Update_Size_And_Errors
            }  // x
          }    // y
        }      // groupX
      }        // groupY
      *bytes_pos += pos;
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

  bool dcmprs512x512(pik::Image3U* img, int planeToDecompress, size_t& pos,
                     size_t groupY, size_t groupX,
                     const uint8_t* compressedData, size_t compressedSize,
                     size_t maxDecodedSize, size_t maxDecodedSize2) {
    size_t esize[NUMCONTEXTS_3], xsize = img->xsize(), ysize = img->ysize();
    memset(esize, 0, sizeof(esize));
    size_t decompressedSize = 0;  // is used only for the assert()
    for (int i = 0; i < NUMCONTEXTS_3; ++i) {
      size_t cs = decodeVarInt(compressedData, compressedSize, &pos), ds, ds1,
             ds2, ds3;
      if (cs == 0) continue;
      // first, decompress MSBs (most significant bytes)
      ds1 = decompressWithEntropyCode((uint8_t*)&edata[i][0], maxDecodedSize,
                                      compressedData, compressedSize, cs, &pos);
      if (!ds1) return PIK_FAILURE("lossless16");
      uint32_t freq[256];
      memset(freq, 0, sizeof(freq));
      uint16_t* dst = &edata[i][0];
      uint8_t* p = (uint8_t*)dst;
      for (int j = 0; j < ds1; ++j) ++freq[p[j]];

      if (ds1 < 120 || freq[0] < 120) {  // All LSBs at once
        cs = decodeVarInt(compressedData, compressedSize, &pos);
        ds2 =
            decompressWithEntropyCode(&compressedDataTmpBuf[0], maxDecodedSize2,
                                      compressedData, compressedSize, cs, &pos);
        if (ds1 != ds2) return PIK_FAILURE("lossless16");
        for (int j = ds1 - 1; j >= 0; --j)
          dst[j] = p[j] * 256 + compressedDataTmpBuf[j];  // MSB*256 + LSB
      } else {
        uint32_t c = (freq[0] > (ds1 * 13 >> 4) ? 2 : 1);
        ds2 = freq[0] + (c == 2 ? freq[1] : 0);
        ds3 = ds1 - ds2;
        if (ds2) {  // LSBs such that MSB==0
          cs = decodeVarInt(compressedData, compressedSize, &pos);
          ds = decompressWithEntropyCode(&compressedDataTmpBuf[0],
                                         maxDecodedSize2, compressedData,
                                         compressedSize, cs, &pos);
          if (!ds) return PIK_FAILURE("lossless16");
          if (ds != ds2) return PIK_FAILURE("lossless16");
        }

        if (ds3) {  // LSBs such that MSB!=0
          cs = decodeVarInt(compressedData, compressedSize, &pos);
          ds = decompressWithEntropyCode(&compressedDataTmpBuf[ds2],
                                         maxDecodedSize2, compressedData,
                                         compressedSize, cs, &pos);
          if (!ds) return PIK_FAILURE("lossless16");
          if (ds != ds3) return PIK_FAILURE("lossless16");
        }
        uint8_t *p2 = &compressedDataTmpBuf[ds2 - 1],
                *p3 = &compressedDataTmpBuf[ds1 - 1];  // Note ds1=ds2+ds3
        for (int j = ds1 - 1; j >= 0; --j)
          dst[j] = p[j] * 256 + (p[j] < c ? *p2-- : *p3--);
      }
      decompressedSize += ds1;
    }  // for i
    if (!(decompressedSize ==
          std::min((size_t)kGroupSize, ysize - groupY) *
              std::min((size_t)kGroupSize, xsize - groupX))) {
      return PIK_FAILURE("lossless16");
    }

    size_t yEnd = std::min((size_t)kGroupSize, ysize - groupY);
    width = std::min((size_t)kGroupSize, xsize - groupX) - 1;
    size_t area = yEnd * (width + 1);
    int maxerrShift =
        (area > 25600
             ? 0
             : area > 12800 ? 1 : area > 2800 ? 2 : area > 512 ? 3 : 4);
    int maxerrAdd = (1 << maxerrShift) - 1;

    for (size_t y = 0, yp = 0, yp1; y < yEnd;
         ++y, yp ^= kGroupSize, yp1 = kGroupSize - yp) {
      rowImg = img->PlaneRow(planeToDecompress, groupY + y) + groupX;
      rowPrev =
          (y == 0 ? NULL
                  : img->PlaneRow(planeToDecompress, groupY + y - 1) + groupX);
      for (size_t x = 0; x <= width; ++x) {
        int maxErr, prediction = predict1(x, yp + x, yp1 + x, maxErr);
        maxErr = (maxErr + maxerrAdd) >> maxerrShift;
        // maxErr=0; // SETTING it 0 here DISABLES ERROR CONTEXT MODELING!
        assert(0 <= maxErr && maxErr <= NUMCONTEXTS_3 - 1);
        assert(0 <= prediction && prediction <= 0xffff);

        size_t s = esize[maxErr];
        int err = edata[maxErr][s];
        int truePixelValue =
            (prediction - sign_LSB_backward_transform[err]) & 0xffff;
        rowImg[x] = truePixelValue;
        err = prediction - truePixelValue;

        Update_Size_And_Errors
      }  // x
    }    // y
    return true;
  }

  bool Colorful16bit_decompress(const PaddedBytes& bytes, size_t* bytes_pos,
                                Image3U* result) {
    WithSIGN = WithSIGN_3, BitsMAX = BitsMAX_3, NUMCONTEXTS = NUMCONTEXTS_3;
    if (*bytes_pos > bytes.size()) return PIK_FAILURE("lossless16");
    size_t compressedSize = bytes.size() - *bytes_pos;
    const uint8_t* compressedData = bytes.data() + *bytes_pos;

    // Size of an edata entry
    size_t maxDecodedSize = kGroupSize * kGroupSize;
    // Size of a compressedDataTmpBuf entry
    size_t maxDecodedSize2 = kGroupSize2plus;

    size_t xsize, ysize, pos0 = 0, imageMethod = 0;
    xsize = decodeVarInt(compressedData, compressedSize, &pos0);
    ysize = decodeVarInt(compressedData, compressedSize, &pos0);
    if (!xsize || !ysize) return PIK_FAILURE("lossless16");
    // Too large, would run out of memory. Chosen as reasonable limit for pik
    // while being below default fuzzer memory limit. We check for total pixel
    // size, and an additional restriction to ysize, because large ysize
    // consumes more memory due to the scanline padding.
    if (uint64_t(xsize) * uint64_t(ysize) >= 134217728ull || ysize >= 65536) {
      return PIK_FAILURE("lossless16");
    }
    pik::Image3U img(xsize, ysize);
    std::vector<int> palette(0x10000 * 3);

    clock_t start = clock();
    for (int run = 0; run < NumRuns; ++run) {
      size_t pos = pos0;
      if (xsize * ysize > 256 * 256) {  // TODO: smarter decision making here
        const uint8_t* p = &compressedData[pos];
        imageMethod = *p++;
        if (imageMethod) {
          int numColors[3];
          ++pos;
          numColors[0] = decodeVarInt(compressedData, compressedSize, &pos);
          numColors[1] = decodeVarInt(compressedData, compressedSize, &pos);
          numColors[2] = decodeVarInt(compressedData, compressedSize, &pos);
          if (numColors[0] > 65536) return PIK_FAILURE("lossless16");
          if (numColors[1] > 65536) return PIK_FAILURE("lossless16");
          if (numColors[2] > 65536) return PIK_FAILURE("lossless16");
          p = &compressedData[pos];
          const uint8_t* p_end = compressedData + compressedSize;
          for (int channel = 0; channel < 3; ++channel)
            if (imageMethod & (1 << channel))
              for (int sb = channel << 16, stop = sb + numColors[channel],
                       color = 0, x = 0;
                   x < 0x10000; x += 8) {
                if (p >= p_end) return PIK_FAILURE("lossless16");
                for (int b = *p++, j = 0; j < 8; ++j)
                  palette[sb] = color++, sb += b & 1, b >>= 1;
                if (sb >= stop) break;
                if (sb + 0x10000 - 8 - x == stop) {
                  for (int i = x; i < 0x10000 - 8; ++i) palette[sb++] = color++;
                  break;
                }
              }
        }
        pos = p - &compressedData[0];
      }
      for (size_t groupY = 0; groupY < ysize; groupY += kGroupSize) {
        for (size_t groupX = 0; groupX < xsize; groupX += kGroupSize) {
          uint16_t *PIK_RESTRICT row1, *PIK_RESTRICT row2, *PIK_RESTRICT row3;
          size_t yEnd = std::min((size_t)kGroupSize, ysize - groupY);
          size_t xEnd = std::min((size_t)kGroupSize, xsize - groupX);
          if (!dcmprs512x512(&img, PL1, pos, groupY, groupX, compressedData,
                             compressedSize, maxDecodedSize, maxDecodedSize2))
            return PIK_FAILURE("lossless16");
          if (!dcmprs512x512(&img, PL2, pos, groupY, groupX, compressedData,
                             compressedSize, maxDecodedSize, maxDecodedSize2))
            return PIK_FAILURE("lossless16");
          if (!dcmprs512x512(&img, PL3, pos, groupY, groupX, compressedData,
                             compressedSize, maxDecodedSize, maxDecodedSize2))
            return PIK_FAILURE("lossless16");
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
              T3bgn G += R + 0x8000;
              row2[x] = G;
              CC break;
            case 2:
              T3bgn B += R + 0x8000;
              row3[x] = B;
              CC break;
            case 3:
              T3bgn G += R + 0x8000;
              B += R + 0x8000;
              row2[x] = G;
              row3[x] = B;
              CC break;
            case 22:
            case 4:
              T3bgn row2[x] = G + B + 0x8000;
              CC break;
            case 5:
              T3bgn row3[x] = G - B + 0x8000;
              CC break;
            case 6:
              T3bgn row2[x] = G = (G + R + 0x8000) & 0xffff;
              row3[x] = B + ((R + G) >> 1) + 0x8000;
              CC break;
            case 7:
              T3bgn row3[x] = B = (B + R + 0x8000) & 0xffff;
              row2[x] = G + ((R + B) >> 1) + 0x8000;
              CC break;
            case 8:
              T3bgn row3[x] = B + ((R + G) >> 1) + 0x8000;
              CC break;
            case 9:
              T3bgn row2[x] = G + ((R + B) >> 1) + 0x8000;
              CC break;

            case 24:
            case 11:
              T3bgn R += G + 0x8000;
              row1[x] = R;
              CC break;
            case 12:
              T3bgn B += G + 0x8000;
              row3[x] = B;
              CC break;
            case 13:
              T3bgn R += G + 0x8000;
              B += G + 0x8000;
              row1[x] = R;
              row3[x] = B;
              CC break;
            case 21:
            case 14:
              T3bgn row1[x] = R + B + 0x8000;
              CC break;
            case 15:
              T3bgn row3[x] = R - B + 0x8000;
              CC break;

            case 16:
              T3bgn row1[x] = R = (R + G + 0x8000) & 0xffff;
              row3[x] = B + ((R + G) >> 1) + 0x8000;
              CC break;
            case 17:
              T3bgn row3[x] = B = (B + G + 0x8000) & 0xffff;
              row1[x] = R + ((B + G) >> 1) + 0x8000;
              CC break;
            case 18:
              T3bgn row3[x] = B + ((R + G) >> 1) + 0x8000;
              CC break;
            case 19:
              T3bgn row1[x] = R + ((B + G) >> 1) + 0x8000;
              CC break;

            case 23:
              T3bgn G += B + 0x8000;
              R += B + 0x8000;
              row1[x] = R;
              row2[x] = G;
              CC break;
            case 25:
              T3bgn row2[x] = R - G + 0x8000;
              CC break;
            case 26:
              T3bgn row1[x] = R = (R + B + 0x8000) & 0xffff;
              row2[x] = G + ((B + R) >> 1) + 0x8000;
              CC break;
            case 27:
              T3bgn row2[x] = G = (G + B + 0x8000) & 0xffff;
              row1[x] = R + ((B + G) >> 1) + 0x8000;
              CC break;
            case 28:
              T3bgn row2[x] = G + ((B + R) >> 1) + 0x8000;
              CC break;
            case 29:
              T3bgn row1[x] = R + ((B + G) >> 1) + 0x8000;
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
          int* p = &palette[0x10000 * channel];
          for (size_t y = 0; y < ysize; ++y) {
            uint16_t* const PIK_RESTRICT rowImg = img.PlaneRow(channel, y);
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

  uint32_t cmprs512x512(pik::Image3U& img, int planeToCompress, int planeToUse,
                        size_t groupY, size_t groupX,
                        uint8_t* compressedOutput) {
    size_t esize[NUMCONTEXTS_3], xsize = img.xsize(), ysize = img.ysize();
    memset(esize, 0, sizeof(esize));
    size_t yEnd = std::min((size_t)kGroupSize, ysize - groupY);
    width = std::min((size_t)kGroupSize, xsize - groupX) - 1;
    size_t area = yEnd * (width + 1);
    int maxerrShift =
        (area > 25600
             ? 0
             : area > 12800 ? 1 : area > 2800 ? 2 : area > 512 ? 3 : 4);
    int maxerrAdd = (1 << maxerrShift) - 1;

    for (size_t y = 0, yp = 0, yp1; y < yEnd;
         ++y, yp ^= kGroupSize, yp1 = kGroupSize - yp) {
      rowImg = img.PlaneRow(planeToCompress, groupY + y) + groupX;
      rowPrev =
          (!y ? NULL : img.PlaneRow(planeToCompress, groupY + y - 1) + groupX);
      uint16_t* PIK_RESTRICT rowUse =
          img.PlaneRow(planeToUse, groupY + y) + groupX;
      for (size_t x = 0; x <= width; ++x) {
        int maxErr, prediction = predict1(x, yp + x, yp1 + x, maxErr);
        maxErr = (maxErr + maxerrAdd) >> maxerrShift;
        assert(0 <= maxErr && maxErr <= NUMCONTEXTS_3 - 1);
        assert(0 <= prediction && prediction <= 0xffff);
        int truePixelValue = (int)rowImg[x];
        if (planeToCompress != planeToUse) {
          truePixelValue -= (int)rowUse[x] - 0x8000;
          truePixelValue &= 0xffff;
          rowImg[x] = truePixelValue;
        }
        int err = prediction - truePixelValue;
        size_t s = esize[maxErr];
        edata[maxErr][s] = sign_LSB_forward_transform[err & 0xffff];
        Update_Size_And_Errors
      }  // x
    }    // y

    size_t pos = 0;
    for (int i = 0; i < NUMCONTEXTS_3; ++i) {
      size_t c = 0, S = esize[i];

      if (S == 0) {
        pos += encodeVarInt(0, &compressedOutput[pos]);
        continue;
      }
      uint16_t* d = &edata[i][0];
      // first, compress MSBs (most significant bytes)
      uint8_t* p = &compressedOutput[pos + 8];
      for (size_t x = 0; x < S; ++x) p[x] = d[x] >> 8, c += (p[x] ? 0 : 1);
      if (!compressWithEntropyCode(&pos, S, compressedOutput))
        return PIK_FAILURE("lossless16");
      if (S < 120 || c < 120) {  // 120
        // then, compress LSBs (least significant bytes)
        p = &compressedOutput[pos + 8];
        for (size_t x = 0; x < S; ++x) p[x] = d[x] & 255;  // All LSBs!
        if (!compressWithEntropyCode(&pos, S, compressedOutput))
          return PIK_FAILURE("lossless16");
      } else {
        c = (c > (S * 13 >> 4) ? 2 : 1) << 8;
        p = &compressedOutput[pos + 8];
        size_t y = 0;
        for (size_t x = 0; x < S; ++x)
          if (d[x] < c) p[y++] = d[x] & 255;  // LSBs such that MSB<2
        if (y) {
          if (!compressWithEntropyCode(&pos, y, compressedOutput))
            return PIK_FAILURE("lossless16");
        }

        p = &compressedOutput[pos + 8];
        y = 0;
        for (size_t x = 0; x < S; ++x)
          if (d[x] >= c) p[y++] = d[x] & 255;  // LSBs such that MSB>=2
        if (y) {
          if (!compressWithEntropyCode(&pos, y, compressedOutput))
            return PIK_FAILURE("lossless16");
        }
      }  // if (S < 120)
    }    // for i
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

  bool Colorful16bit_compress(const Image3U& img_in, PaddedBytes* bytes) {
    WithSIGN = WithSIGN_3, BitsMAX = BitsMAX_3, NUMCONTEXTS = NUMCONTEXTS_3;
    clock_t start = clock();

    // The code modifies the image for palette so must copy for now.
    Image3U img = CopyImage(img_in);

    std::vector<uint8_t> temp_buffer(kGroupSize2plus * 2 * 6);
    compressedData = temp_buffer.data();

    for (int run = 0; run < NumRuns; ++run) {
      size_t xsize = img.xsize(), ysize = img.ysize(), pos;
      pos = encodeVarInt(xsize, &compressedData[0]);
      pos += encodeVarInt(ysize, &compressedData[pos]);
      FWr(&compressedData[0], pos) int numColors[3] = {0xffff, 0xffff, 0xffff};

      if (xsize * ysize > 256 * 256) {  // TODO: smarter decision making here
        // Let's check whether the image should be 'palettized',
        // because the range is 64k, but 25% or more of the range is unused.
        uint8_t flags = 0, bits[3 * 0x10000 / 8], *pb = &bits[0];
        std::vector<uint32_t> palette123(0x10000 * 3);
#if 1
        memset(bits, 0, sizeof(bits));
        memset(palette123.data(), 0, 0x10000 * 3 * sizeof(uint32_t));
        for (int channel = 0; channel < 3; ++channel) {
          uint32_t i, first, count, *palette = &palette123[0x10000 * channel];
          for (size_t y = 0; y < ysize; ++y) {
            uint16_t* const PIK_RESTRICT rowImg = img.PlaneRow(channel, y);
            for (size_t x = 0; x < xsize; ++x)  // UNROLL AND PARALLELIZE ME!
              palette[rowImg[x]] = 1;
          }
          // count the number of pixel values present in the image
          for (i = 0; i < 0x10000; ++i)
            if (palette[i]) break;
          for (first = i, count = 0; i < 0x10000; ++i)
            if (palette[i]) palette[i] = count++;
          // printf("count=%5d, %f%%\n", count, count * 100. / 65536);
          if (count >= 65536 * 3 / 4) {
            flags = 0;
            break;
          }  // TODO: decision making

          flags += 1 << channel;
          numColors[channel] = count;
          palette[first] = 1;
          for (int sb = 0, x = 0; x < 0x10000;
               x += 8) {  // Compress the bits, not store!
            uint32_t b = 0, v;
            for (int y = x + 7; y >= x; --y)
              v = (palette[y] ? 1 : 0), b += b + v, sb += v;
            *pb++ = b;
            if (sb >= count || sb + 0x10000 - 8 - x == count) break;
          }
          palette[first] = 0;
        }  // for channel
#endif
        FWrByte(flags);  // As of now (Dec.2018) ImageMethod==flags
        if (flags) {
          for (int channel = 0; channel < 3; ++channel) {
            uint32_t* palette = &palette123[0x10000 * channel];
            for (size_t y = 0; y < ysize; ++y) {
              uint16_t* const PIK_RESTRICT rowImg = img.PlaneRow(channel, y);
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
      }    // if (xsize*ysize > 256*256)
      uint8_t* compressedData2 = &compressedData[kGroupSize2plus * 2];
      uint8_t* compressedData3 = &compressedData[kGroupSize2plus * 4];
      uint8_t* cd4 = &compressedData[kGroupSize2plus * 6];
      uint8_t* cd5 = &compressedData[kGroupSize2plus * 8];
      uint8_t* cd6 = &compressedData[kGroupSize2plus * 10];
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
          S4 = cmprs512x512(img, p2, p1, groupY, groupX, cd4); /* R-G+0x8000 */
          S5 = cmprs512x512(img, p3, p1, groupY, groupX, cd5); /* B-G+0x8000 */
          if (p1 == PL1)
            FWr(cd1, S1)

                if (S4 >= S2 && S5 >= S3) {
              S6 = cmprs512x512(img, p2, p3, groupY, groupX,
                                cd6); /* R-B+0x8000 */
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
                uint16_t* PIK_RESTRICT row1 =
                    img.PlaneRow(p1, groupY + y) + groupX;
                uint16_t* PIK_RESTRICT row2 =
                    img.PlaneRow(p2, groupY + y) + groupX;
                for (size_t x = 0; x < xEnd; ++x) {
                  uint32_t v1 = row1[x], v2 = (row2[x] + v1 + 0x8000) & 0xffff;
                  row2[x] = ((v1 + v2) >> 1) - v1 + 0x8000;
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
                uint16_t* PIK_RESTRICT row1 =
                    img.PlaneRow(p1, groupY + y) + groupX;
                uint16_t* PIK_RESTRICT row3 =
                    img.PlaneRow(p3, groupY + y) + groupX;
                for (size_t x = 0; x < xEnd; ++x) {
                  uint32_t v1 = row1[x], v3 = (row3[x] + v1 + 0x8000) & 0xffff;
                  row3[x] = ((v1 + v3) >> 1) - v1 + 0x8000;
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

bool Grayscale16bit_compress(const ImageU& img, PaddedBytes* bytes) {
  std::unique_ptr<State> state(new State());
  return state->Grayscale16bit_compress(img, bytes);
}

bool Grayscale16bit_decompress(const PaddedBytes& bytes, size_t* pos,
                               ImageU* result) {
  std::unique_ptr<State> state(new State());
  return state->Grayscale16bit_decompress(bytes, pos, result);
}

bool Colorful16bit_compress(const Image3U& img, PaddedBytes* bytes) {
  std::unique_ptr<State> state(new State());
  return state->Colorful16bit_compress(img, bytes);
}

bool Colorful16bit_decompress(const PaddedBytes& bytes, size_t* pos,
                              Image3U* result) {
  std::unique_ptr<State> state(new State());
  return state->Colorful16bit_decompress(bytes, pos, result);
}

}  // namespace pik
