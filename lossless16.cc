// @author Alexander Rhatushnyak

#include "lossless16.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "entropy_coder.h"

namespace pik {

namespace {

const int groupSize = 512, NumRuns = 1, MAXERROR = 0x3fbf,
          MaxSumErrors = (MAXERROR + 1) * 4, WithSIGN = 4, BitsMAX = 13,
          NUMCONTEXTS = 1 + WithSIGN + BitsMAX;

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
  static const int kAlphabetSize = 256;
  static const int kContext = 0;
  size_t pos = 0;
  size_t num_symbols = decodeVarInt(data, size, &pos);
  if (pos >= size) {
    return false;
  }
  // TODO(user): instead take expected decoded size as function parameter
  if (num_symbols > 16777216) {
    // Avoid large allocations, we never expect this many symbols for
    // the limited group sizes.
    return false;
  }

  BitReader br(data + pos, size - pos);
  ANSCode codes;
  if (!DecodeANSCodes(1, kAlphabetSize, &br, &codes)) {
    return false;
  }

  result->resize(num_symbols);
  ANSSymbolReader reader(&codes);
  for (size_t i = 0; i < num_symbols; i++) {
    br.FillBitBuffer();
    int read_symbol = reader.ReadSymbol(kContext, &br);
    (*result)[i] = read_symbol;
  }
  if (!reader.CheckANSFinalState()) {
    return false;
  }

  return true;
}

// TODO(user): avoid the copying between std::vector and data.
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

// TODO(user): split state variables needed for encoder from those for decoder
//             and perform one-time global initialization where possible.
struct State {
  int prediction0, prediction1, prediction2, prediction3;

  uint16_t edata[NUMCONTEXTS][groupSize * groupSize];
  uint8_t compressedDataTmpBuf[groupSize * (groupSize + 128)], *compressedData;
  int32_t errors0[groupSize * 2];  // Errors of predictor 0
  int32_t errors1[groupSize * 2];  // Errors of predictor 1
  int32_t errors2[groupSize * 2];  // Errors of predictor 2
  int32_t errors3[groupSize * 2];  // Errors of predictor 3
  uint8_t nbitErr[groupSize * 2];
  int32_t trueErr[groupSize * 2];

  uint16_t error2weight[MaxSumErrors], sign_LSB_forward_transform[0x10000],
      sign_LSB_backward_transform[0x10000];
  uint8_t numBitsTable[256];  // const

  State() {
    for (int i = 0; i < 256; ++i)
      numBitsTable[i] = numbitsInit(i);  // const
                                         // init!
    error2weight[0] = 0xffff;
    for (int j = 1; j < MaxSumErrors; ++j)
      error2weight[j] = 181 * 256 / j;  // 120          // const init!

    for (int i = 0; i < 256 * 256; ++i)
      sign_LSB_forward_transform[i] =
          (i & 32768 ? (0xffff - i) * 2 + 1 : i * 2);  // const init!

    for (int i = 0; i < 256 * 256; ++i)
      sign_LSB_backward_transform[i] =
          (i & 1 ? 0xffff - (i >> 1) : i >> 1);  // const init!
  }

  PIK_INLINE int numBits(int x) {
    assert(0 <= x && x <= 0xffff);
    if (x < 256) return numBitsTable[x];
    return std::min(8 + numBitsTable[x >> 8], BitsMAX);
  }

  PIK_INLINE int numbitsInit(int x) {
    assert(0 <= x && x <= 255);
    int res = 0;
    if (x >= 16) res = 4, x >>= 4;
    if (x >= 4) res += 2, x >>= 2;
    return (res + std::min(x, 2));
  }

  PIK_INLINE int predictY0(size_t x, const uint16_t* PIK_RESTRICT rowImg,
                           const uint16_t* PIK_RESTRICT rowPrev, size_t yp,
                           size_t yp1, size_t width, int* maxErr) {
    *maxErr = (x == 0 ? NUMCONTEXTS - 1
                      : x == 1 ? nbitErr[yp - 1]
                               : std::max(nbitErr[yp - 1], nbitErr[yp - 2]));
    prediction0 = prediction1 = prediction2 = prediction3 =
        (x == 0 ? 14 * 256  // 14
                : x == 1 ? rowImg[x - 1]
                         : rowImg[x - 1] + (rowImg[x - 1] - rowImg[x - 2]) / 4);
    return (prediction0 < 0 ? 0 : prediction0 > 0xffff ? 0xffff : prediction0);
  }

  PIK_INLINE int predictX0(size_t x, const uint16_t* PIK_RESTRICT rowImg,
                           const uint16_t* PIK_RESTRICT rowPrev, size_t yp,
                           size_t yp1, size_t width, int* maxErr) {
    *maxErr = std::max(nbitErr[yp1], nbitErr[yp1 + (x < width ? 1 : 0)]);
    prediction0 = prediction1 = prediction2 = prediction3 =
        (rowPrev[x] * 7 + rowPrev[x + (x < width ? 1 : 0)] + 6) >> 3;  // 6
    return prediction0;
  }

  PIK_INLINE int predict(size_t x, const uint16_t* PIK_RESTRICT rowImg,
                         const uint16_t* PIK_RESTRICT rowPrev, size_t yp,
                         size_t yp1, size_t width,  // width-1 actually
                         int* maxErr) {
    if (!rowPrev) return predictY0(x, rowImg, rowPrev, yp, yp1, width, maxErr);
    if (x == 0LL) return predictX0(x, rowImg, rowPrev, yp, yp1, width, maxErr);
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
    *maxErr = mxe;

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

  bool compressWithEntropyCode(size_t* pos, size_t S) {
    uint8_t* src = &compressedData[*pos + 8];
    size_t cs;
    if (IsRLE(src, S)) {
      cs = 1;  // use RLE encoding instead
    } else {
      cs = EntropyEncode(src, S, &compressedDataTmpBuf[0],
                         sizeof(compressedDataTmpBuf));
      if (!cs) return false;  // error
    }
    if (cs >= S) cs = 0;  // EntropyCode worse than original, use memcpy.
    *pos += encodeVarInt(cs <= 1 ? (S - 1) * 3 + 1 + cs : cs * 3,
                         &compressedData[*pos]);
    uint8_t* dst = &compressedData[*pos];
    if (cs == 1)
      compressedData[(*pos)++] = *src;
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

#define Update_Size_And_Errors                                    \
  ++esize[maxErr];                                                \
  trueErr[yp + x] = err;                                          \
  err = numBits(err >= 0 ? err : -err);                           \
  nbitErr[yp + x] = (err <= WithSIGN ? err * 2 : err + WithSIGN); \
  err = prediction0 - truePixelValue;                             \
  if (err < 0) err = -err; /* abs() and min()? worse speed! */    \
  if (err > MAXERROR) err = MAXERROR;                             \
  errors0[yp + x] = err;                                          \
  err = prediction1 - truePixelValue;                             \
  if (err < 0) err = -err;                                        \
  if (err > MAXERROR) err = MAXERROR;                             \
  errors1[yp + x] = err;                                          \
  err = prediction2 - truePixelValue;                             \
  if (err < 0) err = -err;                                        \
  if (err > MAXERROR) err = MAXERROR;                             \
  errors2[yp + x] = err;                                          \
  err = prediction3 - truePixelValue;                             \
  if (err < 0) err = -err;                                        \
  if (err > MAXERROR) err = MAXERROR;                             \
  errors3[yp + x] = err;

  bool Grayscale16bit_compress(const ImageU& img, PaddedBytes* bytes) {
    size_t esize[NUMCONTEXTS], xsize = img.xsize(), ysize = img.ysize();
    std::vector<uint8_t> temp_buffer(groupSize * (groupSize + 128) * 2);
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
      for (size_t groupY = 0; groupY < ysize; groupY += groupSize) {
        for (size_t groupX = 0; groupX < xsize; groupX += groupSize) {
          memset(esize, 0, sizeof(esize));
          for (size_t y = 0, yEnd = std::min((size_t)groupSize, ysize - groupY),
                      yp = 0, yp1;
               y < yEnd; ++y, yp ^= groupSize, yp1 = groupSize - yp) {
            const uint16_t* PIK_RESTRICT rowImg = img.Row(groupY + y) + groupX;
            const uint16_t* PIK_RESTRICT rowPrev =
                (y == 0 ? NULL : img.Row(groupY + y - 1) + groupX);
            for (size_t x = 0, xEnd = std::min((size_t)(groupSize - 1),
                                               xsize - groupX - 1);
                 x <= xEnd; ++x) {
              int maxErr;
              int prediction =
                  predict(x, rowImg, rowPrev, yp + x, yp1 + x, xEnd, &maxErr);
              // maxErr=0; // SETTING it 0 here DISABLES ERROR CONTEXT MODELING!
              assert(0 <= maxErr && maxErr <= NUMCONTEXTS - 1);
              assert(0 <= prediction && prediction <= 0xffff);

              int truePixelValue = (int)rowImg[x];
              int err = prediction - truePixelValue;
              size_t s = esize[maxErr];
              edata[maxErr][s] = sign_LSB_forward_transform[err & 0xffff];

              Update_Size_And_Errors
            }  // x
          }    // y
          size_t pos = 0;
          if (groupY == 0 && groupX == 0) {
            pos += encodeVarInt(xsize, &compressedData[pos]);
            pos += encodeVarInt(ysize, &compressedData[pos]);
          }
          for (int i = 0; i < NUMCONTEXTS; ++i) {
            size_t S = esize[i];
            if (S == 0) {
              pos += encodeVarInt(0, &compressedData[pos]);
              continue;
            }
            uint16_t* d = &edata[i][0];
            // first, compress MSBs (most significant bytes)
            uint8_t* p = &compressedData[pos + 8];
            for (size_t x = 0; x < S; ++x) p[x] = d[x] >> 8;
            if (!compressWithEntropyCode(&pos, S)) return false;

            if (i > 8 || S < 160) {  //  8  160
              // then, compress LSBs (least significant bytes)
              p = &compressedData[pos + 8];
              for (size_t x = 0; x < S; ++x) p[x] = d[x] & 255;  // All
              if (!compressWithEntropyCode(&pos, S)) return false;
            } else {
              p = &compressedData[pos + 8];
              size_t y = 0;
              for (size_t x = 0; x < S; ++x)
                if (d[x] < 256) p[y++] = d[x] & 255;  // LSBs such that MSB==0
              if (y) {
                if (!compressWithEntropyCode(&pos, y)) return false;
              }

              p = &compressedData[pos + 8];
              y = 0;
              for (size_t x = 0; x < S; ++x)
                if (d[x] >= 256) p[y++] = d[x] & 255;  // LSBs such that MSB!=0
              if (y) {
                if (!compressWithEntropyCode(&pos, y)) return false;
              }
            }  // if (i > 8)
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
    if (*bytes_pos > bytes.size()) return false;
    size_t compressedSize = bytes.size() - *bytes_pos;
    const uint8_t* compressed = bytes.data() + *bytes_pos;

    // Size of an edata entry
    size_t maxDecodedSize = groupSize * groupSize;
    // Size of a compressedDataTmpBuf entry
    size_t maxDecodedSize2 = groupSize * (groupSize + 128);

    size_t esize[NUMCONTEXTS], xsize, ysize, pos = 0;
    xsize = decodeVarInt(compressed, compressedSize, &pos);
    ysize = decodeVarInt(compressed, compressedSize, &pos);
    if (!xsize || !ysize) return false;
    // Too large, would run out of memory. Chosen as reasonable limit for pik
    // while being below default fuzzer memory limit. We check for total pixel
    // size, and an additional restriction to ysize, because large ysize
    // consumes more memory due to the scanline padding.
    if (uint64_t(xsize) * uint64_t(ysize) >= 134217728ull || ysize >= 65536) {
      return false;
    }
    pik::ImageU img(xsize, ysize);

    clock_t start = clock();
    for (int run = 0; run < NumRuns; ++run) {
      pos = 0;
      for (size_t groupY = 0; groupY < ysize; groupY += groupSize) {
        for (size_t groupX = 0; groupX < xsize; groupX += groupSize) {
          size_t decompressedSize = 0;  // is used only for the assert()

          if (groupY == 0 && groupX == 0) {
            decodeVarInt(compressed, compressedSize, &pos);
            decodeVarInt(compressed, compressedSize, &pos);
          }
          for (int i = 0; i < NUMCONTEXTS; ++i) {
            size_t cs = decodeVarInt(compressed, compressedSize, &pos);
            size_t ds, ds1, ds2, ds3;
            if (cs == 0) continue;
            // first, decompress MSBs (most significant bytes)
            ds1 = decompressWithEntropyCode((uint8_t*)&edata[i][0],
                                            maxDecodedSize, compressed,
                                            compressedSize, cs, &pos);
            if (!ds1) return false;

            if (i > 8 || ds1 < 160) {  // All LSBs at once
              cs = decodeVarInt(compressed, compressedSize, &pos);
              ds2 = decompressWithEntropyCode(&compressedDataTmpBuf[0],
                                              maxDecodedSize2, compressed,
                                              compressedSize, cs, &pos);
              if (ds1 != ds2) return false;
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
                cs = decodeVarInt(compressed, compressedSize, &pos);
                ds = decompressWithEntropyCode(&compressedDataTmpBuf[0],
                                               maxDecodedSize2, compressed,
                                               compressedSize, cs, &pos);
                if (!ds) return false;
                if (ds != ds2) return false;
              }

              if (ds3) {  // LSBs such that MSB!=0
                cs = decodeVarInt(compressed, compressedSize, &pos);
                ds = decompressWithEntropyCode(&compressedDataTmpBuf[ds2],
                                               maxDecodedSize2, compressed,
                                               compressedSize, cs, &pos);
                if (!ds) return false;
                if (ds != ds3) return false;
              }
              uint8_t *p2 = &compressedDataTmpBuf[ds2 - 1],
                      *p3 = &compressedDataTmpBuf[ds1 - 1];  // Note ds1=ds2+ds3
              for (int j = ds1 - 1; j >= 0; --j)
                dst[j] = p[j] * 256 + (p[j] == 0 ? *p2-- : *p3--);
            }
            decompressedSize += ds1;
          }
          if (!(decompressedSize ==
                std::min((size_t)groupSize, ysize - groupY) *
                    std::min((size_t)groupSize, xsize - groupX))) {
            return false;
          }
// Disabled, because it is actually useful that the decoder supports decoding
// its own stream when contained inside a bigger stream and knows the correct
// end position.
#if 0
          if (groupY + groupSize >= ysize &&
              groupX + groupSize >= xsize)  // if last group
            assert(pos == compressedSize);
#endif

          memset(esize, 0, sizeof(esize));
          for (size_t y = 0, yEnd = std::min((size_t)groupSize, ysize - groupY),
                      yp = 0, yp1;
               y < yEnd; ++y, yp ^= groupSize, yp1 = groupSize - yp) {
            uint16_t* const PIK_RESTRICT rowImg = img.Row(groupY + y) + groupX;
            uint16_t* const PIK_RESTRICT rowPrev =
                (y == 0 ? NULL : img.Row(groupY + y - 1) + groupX);
            for (size_t x = 0, xEnd = std::min((size_t)(groupSize - 1),
                                               xsize - groupX - 1);
                 x <= xEnd; ++x) {
              int maxErr;
              int prediction =
                  predict(x, rowImg, rowPrev, yp + x, yp1 + x, xEnd, &maxErr);
              // maxErr=0; // SETTING it 0 here DISABLES ERROR CONTEXT MODELING!
              assert(0 <= maxErr && maxErr <= NUMCONTEXTS - 1);
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
};

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

}  // namespace pik
