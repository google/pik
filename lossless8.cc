// @author Alexander Rhatushnyak

#include "lossless8.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "entropy_coder.h"

namespace pik {

const int WITHSIGN = 8, NUMCONTEXTS = 9 + WITHSIGN, MAXERROR = 0x5f,
          MaxSumErrors = (MAXERROR + 1) * 4, groupSize = 512, NumRuns = 1;

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

size_t decodeVarInt(const uint8_t* input, size_t inputSize, size_t& pos) {
  size_t i, ret = 0;
  for (i = 0; pos + i < inputSize && i < 10; ++i) {
    ret |= uint64_t(input[pos + i] & 127) << uint64_t(7 * i);
    // If the next-byte flag is not set, stop
    if ((input[pos + i] & 128) == 0) break;
  }
  pos += i + 1;
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
  size_t num_symbols = decodeVarInt(data, size, pos);
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
  if (!DecodeANSCodes(1, 256, &br, &codes)) {
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
    return false;  // Encoding error
  }
  if (result.size() > size) {
    *out_size = 0;  // Indicate the codec should use uncompressed mode instead.
    return true;
  }
  if (result.size() > out_capacity) {
    return false;  // Error: not enough capacity
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
    return false;  // Decoding error
  }
  if (result.size() > out_capacity) {
    return false;  // Error: not enough capacity
  }
  memcpy(out, result.data(), result.size());
  *out_size = result.size();
  return true;
}

// TODO(user): split state variables needed for encoder from those for decoder
//             and perform one-time global initialization where possible.
struct State {
  int prediction0,
      prediction1,  // Their range is -255...510 rather than 0...255!
      prediction2,
      prediction3;  // And -510..510 after subtracting truePixelValue

  uint8_t edata[NUMCONTEXTS][groupSize * groupSize],
      compressedDataTmpBuf[groupSize * (groupSize + 128)], *compressedData;
  uint8_t errors0[groupSize * 2];  // Errors of predictor 0. Range: 0...MAXERROR
  uint8_t errors1[groupSize * 2];  // Errors of predictor 1
  uint8_t errors2[groupSize * 2];  // Errors of predictor 2
  uint8_t errors3[groupSize * 2];  // Errors of predictor 3
  uint8_t quantizedError[groupSize * 2];  // The range is 0...16, all are
                                          // even due to quantizedInit()
  int16_t trueErr[groupSize * 2];         // Their range is -255...255

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
#define ToLSB_BKWRD signLSB_backwardTransform[prediction * 256 + err]

#define signToLSB_FORWARD_INIT                                               \
  for (int p = 0; p < 256; ++p) {                                            \
    signLSB_forwardTransform[p * 256 + p] = 0;                               \
    for (int v, top = p, btm = p, d = 1; d < 256; ++d) {                     \
      v = (d & 1 ? (top < 255 ? ++top : --btm) : (btm > 0 ? --btm : ++top)); \
      signLSB_forwardTransform[p * 256 + v] = d;                             \
    }                                                                        \
  }

#define signToLSB_BACKWARD_INIT                                              \
  for (int p = 0; p < 256; ++p) {                                            \
    signLSB_backwardTransform[p * 256] = p;                                  \
    for (int v, top = p, btm = p, d = 1; d < 256; ++d) {                     \
      v = (d & 1 ? (top < 255 ? ++top : --btm) : (btm > 0 ? --btm : ++top)); \
      signLSB_backwardTransform[p * 256 + d] = v;                            \
    }                                                                        \
  }
#endif
  uint8_t quantizedTable[256 * 2], diff2error[512 * 2];  // const
  uint16_t error2weight[MaxSumErrors];                   // const

  State() {
    for (int j = 0; j < MaxSumErrors; ++j)
      // const init!  // 331 49 25.9
      error2weight[j] = 331 * 256 / (49 + j * std::sqrt(j + 25.9));
    for (int j = -510; j <= 510; ++j)
      diff2error[512 + j] = std::min(j < 0 ? -j : j, MAXERROR);  // const init!
    for (int j = -255; j <= 255; ++j)
      quantizedTable[256 + j] = quantizedInit(j);  // const init!
    signToLSB_FORWARD_INIT                         // const init!
        signToLSB_BACKWARD_INIT                    // const init!
  }

  PIK_INLINE int quantized(int x) {
    assert(-255 <= x && x <= 255);
    return quantizedTable[256 + x];
  }

  PIK_INLINE int quantizedInit(int x) {
    assert(-255 <= x && x <= 255);
    if (x < 0) x = -x;
    int res = 0;
    if (x >= 16) res = 4, x >>= 4;
    if (x >= 4) res += 2, x >>= 2;
    res += (x > 2 ? 2 : x);
    if (res >= 2) {
      res = (res - 1) * 2 +
            ((x >> (res - 2)) & 1);  // 2 bits => 2..3,
                                     // 3 => 4..5   4 => 6..7   5+ bits => 8
      if (res > 8) res = 8;
    }
    return res * 2;
  }

  PIK_INLINE int predictY0(size_t x, const uint8_t* PIK_RESTRICT rowImg,
                           const uint8_t* PIK_RESTRICT rowPrev, size_t yc,
                           size_t yp, size_t width, int& maxErr) {
    maxErr = (x == 0 ? NUMCONTEXTS - 3
                     : x == 1 ? quantizedError[yc]
                              : std::max(quantizedError[yc],
                                         quantizedError[yc - 1]));
    prediction0 = prediction1 = prediction2 = prediction3 =
        (x == 0 ? 38
                : x == 1 ? rowImg[x - 1]
                         : rowImg[x - 1] +
                               (rowImg[x - 1] - rowImg[x - 2]) * 3 / 8);
    return (prediction0 < 0 ? 0 : prediction0 > 255 ? 255 : prediction0);
  }

  PIK_INLINE int predictX0(size_t x, const uint8_t* PIK_RESTRICT rowImg,
                           const uint8_t* PIK_RESTRICT rowPrev, size_t yc,
                           size_t yp, size_t width, int& maxErr) {
    maxErr =
        std::max(quantizedError[yp], quantizedError[yp + (x < width ? 1 : 0)]);
    prediction0 = prediction1 = prediction2 = prediction3 =
        (rowPrev[x] * 7 + rowPrev[x + (x < width ? 1 : 0)] + 6) >> 3;
    return prediction0;
  }

  PIK_INLINE int predict(size_t x, const uint8_t* PIK_RESTRICT rowImg,
                         const uint8_t* PIK_RESTRICT rowPrev, size_t yc,
                         size_t yp, size_t width,  // width-1 actually
                         int& maxErr) {
    if (!rowPrev) return predictY0(x, rowImg, rowPrev, yc, yp, width, maxErr);
    if (x == 0LL) return predictX0(x, rowImg, rowPrev, yc, yp, width, maxErr);

    int N = rowPrev[x],
        W = rowImg[x - 1];  // NW = rowPrev[x - 1];  is not used!
    int a1 = (x < width ? 1 : 0), NE = rowPrev[x + a1];
    int weight0 =
        errors0[yc] + errors0[yp] + errors0[yp - 1] + errors0[yp + a1];
    int weight1 =
        errors1[yc] + errors1[yp] + errors1[yp - 1] + errors1[yp + a1];
    int weight2 =
        errors2[yc] + errors2[yp] + errors2[yp - 1] + errors2[yp + a1];
    int weight3 =
        errors3[yc] + errors3[yp] + errors3[yp - 1] + errors3[yp + a1];
    weight0 = error2weight[weight0] + 42;  // 42
    weight1 = error2weight[weight1] + 72;  // 72
    weight2 = error2weight[weight2] + 12;  // 12
    weight3 = error2weight[weight3];

    uint8_t mxe = quantizedError[yc];
    mxe = std::max(mxe, quantizedError[yp]);
    mxe = std::max(mxe, quantizedError[yp - 1]);
    mxe = std::max(mxe, quantizedError[yp + a1]);
    int teW = trueErr[yc];
    int teN = trueErr[yp];
    int sumWN = teN + teW;  //  -510 <= sumWN <= 510
    int teNW = trueErr[yp - 1];
    int teNE = trueErr[yp + a1];

    maxErr = mxe;
    if (mxe && sumWN * 40 + teNW * 23 + teNE * 12 < -11)  // 40 23 12 -11
      --maxErr;

    prediction0 =
        N - sumWN / 2;  // if bigger than 1/2, clamping would be needed!
    prediction1 = W - (sumWN + teNW) /
                          3;  // 1/3. Note bigger than 1/3 is seemingly better,
                              // but would require clamping to -255...510
    prediction2 = W + NE - N;
    prediction3 = N - (teN + teNW + teNE) / 3;  // TODO:  N - refinedDiv3[...]
    assert(-255 <= prediction0 && prediction0 <= 510);
    assert(-255 <= prediction1 && prediction1 <= 510);
    assert(-255 <= prediction2 && prediction2 <= 510);
    assert(-255 <= prediction3 && prediction3 <= 510);

    int sumWeights = weight0 + weight1 + weight2 + weight3;
    int prediction =
        (prediction0 * weight0 + prediction1 * weight1 + sumWeights * 3 / 8 +
         prediction2 * weight2 + prediction3 * weight3) /
        sumWeights;

    if (((teN ^ teW) | (teN ^ teNE)) >= 0)  // if all three have the same sign
      return (prediction < 0 ? 0 : prediction > 255 ? 255 : prediction);

    int mx = (W > N ? W : N);
    int mn = W + N - mx;
    if (NE > mx) mx = NE;
    if (NE < mn) mn = NE;
    return (prediction < mn ? mn : prediction > mx ? mx : prediction);
  }

#define Update_Size_And_Errors                     \
  esize[maxErr] = s + 1;                           \
  trueErr[yc + x] = err;                           \
  quantizedError[yc + x] = quantized(err);         \
  uint8_t* dp = &diff2error[512 - truePixelValue]; \
  errors0[yc + x] = dp[prediction0];               \
  errors1[yc + x] = dp[prediction1];               \
  errors2[yc + x] = dp[prediction2];               \
  errors3[yc + x] = dp[prediction3];

  bool Grayscale8bit_compress(const ImageB& img, pik::PaddedBytes* bytes) {
    size_t esize[NUMCONTEXTS], xsize = img.xsize(), ysize = img.ysize();
    clock_t start = clock();
    std::vector<uint8_t> temp_buffer(groupSize * (groupSize + 128));
    compressedData = temp_buffer.data();
    for (int run = 0; run < NumRuns; ++run) {
      for (size_t groupY = 0; groupY < ysize; groupY += groupSize) {
        for (size_t groupX = 0; groupX < xsize; groupX += groupSize) {
          memset(esize, 0, sizeof(esize));
          size_t yEnd = std::min((size_t)groupSize, ysize - groupY);
          size_t xEnd = std::min((size_t)(groupSize - 1), xsize - groupX - 1);
          for (size_t y = 0, yc = 0, yp; y < yEnd;
               ++y, yc ^= groupSize, yp = groupSize - yc) {
            const uint8_t* PIK_RESTRICT rowImg = img.Row(groupY + y) + groupX;
            const uint8_t* PIK_RESTRICT rowPrev =
                (y == 0 ? NULL : img.Row(groupY + y - 1) + groupX);
            for (size_t x = 0; x <= xEnd; ++x) {
              int maxErr, prediction = predict(x, rowImg, rowPrev, yc + x - 1,
                                               yp + x, xEnd, maxErr);
              // maxErr = 0; // SETTING it 0 here DISABLES ERROR CONTEXT
              // MODELING!
              assert(0 <= maxErr && maxErr <= NUMCONTEXTS - 1);
              assert(0 <= prediction && prediction <= 255);

              int truePixelValue = rowImg[x], err = prediction - truePixelValue;
              size_t s = esize[maxErr];
              edata[maxErr][s] = ToLSB_FRWRD;

              Update_Size_And_Errors
            }  // x
          }    // y
          size_t pos = 0;
          if (groupY == 0 && groupX == 0) {
            pos += encodeVarInt(xsize, &compressedData[pos]);
            pos += encodeVarInt(ysize, &compressedData[pos]);
          }
          for (int i = 0; i < NUMCONTEXTS; ++i) {
            if (esize[i]) {
              // size_t cs = FSE_compress(&compressedDataTmpBuf[0],
              // sizeof(compressedDataTmpBuf), &edata[i][0], esize[i]);
              size_t cs;
              if (!EntropyEncode(&edata[i][0], esize[i],
                                 sizeof(compressedDataTmpBuf),
                                 &compressedDataTmpBuf[0], &cs)) {
                return false;
              }
              pos +=
                  encodeVarInt(cs <= 1 ? (esize[i] - 1) * 3 + 1 + cs : cs * 3,
                               &compressedData[pos]);
              if (cs == 1)
                compressedData[pos++] = edata[i][0];
              else if (cs == 0)
                memcpy(&compressedData[pos], &edata[i][0], esize[i]),
                    pos += esize[i];
              else
                memcpy(&compressedData[pos], &compressedDataTmpBuf[0], cs),
                    pos += cs;
            } else
              pos += encodeVarInt(0, &compressedData[pos]);
          }  // i
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

  bool Grayscale8bit_decompress(const PaddedBytes& bytes, size_t* bytes_pos,
                                ImageB* result) {
    if (*bytes_pos > bytes.size()) return false;
    size_t compressedSize = bytes.size() - *bytes_pos;
    const uint8_t* compressed = bytes.data() + *bytes_pos;

    size_t maxDecodedSize = groupSize * groupSize;  // Size of an edata entry

    size_t esize[NUMCONTEXTS], xsize, ysize, pos = 0;
    xsize = decodeVarInt(compressed, compressedSize, pos);
    ysize = decodeVarInt(compressed, compressedSize, pos);
    if (!xsize || !ysize) return false;
    // Too large, would run out of memory. Chosen as reasonable limit for pik
    // while being below default fuzzer memory limit. We check for total pixel
    // size, and an additional restriction to ysize, because large ysize
    // consumes more memory due to the scanline padding.
    if (uint64_t(xsize) * uint64_t(ysize) >= 268435456ull || ysize >= 65536) {
      return false;
    }
    pik::ImageB img(xsize, ysize);
    clock_t start = clock();
    for (int run = 0; run < NumRuns; ++run) {
      pos = 0;
      for (size_t groupY = 0; groupY < ysize; groupY += groupSize) {
        for (size_t groupX = 0; groupX < xsize; groupX += groupSize) {
          size_t decompressedSize = 0;  // is used only for the assert()

          if (groupY == 0 && groupX == 0) {
            decodeVarInt(compressed, compressedSize, pos);  // just skip them
            decodeVarInt(compressed, compressedSize, pos);
          }
          for (int i = 0; i < NUMCONTEXTS; ++i) {
            size_t cs = decodeVarInt(compressed, compressedSize, pos),
                   mode = cs % 3;
            if (cs == 0) continue;
            cs /= 3;
            if (mode == 2) {
              if (pos >= compressedSize) return false;
              if (cs > maxDecodedSize) return false;
              memset(&edata[i][0], compressed[pos++], ++cs),
                  decompressedSize += cs;
            } else if (mode == 1) {
              if (pos + cs > compressedSize) return false;
              if (cs > maxDecodedSize) return false;
              memcpy(&edata[i][0], &compressed[pos], ++cs),
                  decompressedSize += cs, pos += cs;
            } else {
              if (pos + cs > compressedSize) return false;
              size_t ds;
              if (!EntropyDecode(&compressed[pos], cs, maxDecodedSize,
                                 &edata[i][0], &ds)) {
                return false;
              }
              pos += cs;
              decompressedSize += ds;
            }
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
              groupX + groupSize >= xsize)  // if the last group
            assert(compressedSize == pos);
#endif

          memset(esize, 0, sizeof(esize));
          size_t yEnd = std::min((size_t)groupSize, ysize - groupY);
          size_t xEnd = std::min((size_t)(groupSize - 1), xsize - groupX - 1);
          for (size_t y = 0, yc = 0, yp; y < yEnd;
               ++y, yc ^= groupSize, yp = groupSize - yc) {
            uint8_t* const PIK_RESTRICT rowImg = img.Row(groupY + y) + groupX;
            uint8_t* const PIK_RESTRICT rowPrev =
                (y == 0 ? NULL : img.Row(groupY + y - 1) + groupX);
            for (size_t x = 0; x <= xEnd; ++x) {
              int maxErr, prediction = predict(x, rowImg, rowPrev, yc + x - 1,
                                               yp + x, xEnd, maxErr);
              // maxErr = 0; // SETTING it 0 here DISABLES ERROR CONTEXT
              // MODELING!
              assert(0 <= maxErr && maxErr <= NUMCONTEXTS - 1);
              assert(0 <= prediction && prediction <= 255);

              size_t s = esize[maxErr];
              int err = edata[maxErr][s], truePixelValue = ToLSB_BKWRD;
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

bool Grayscale8bit_compress(const ImageB& img, PaddedBytes* bytes) {
  std::unique_ptr<State> state(new State());
  return state->Grayscale8bit_compress(img, bytes);
}

bool Grayscale8bit_decompress(const PaddedBytes& bytes, size_t* pos,
                              ImageB* result) {
  std::unique_ptr<State> state(new State());
  return state->Grayscale8bit_decompress(bytes, pos, result);
}

}  // namespace pik
