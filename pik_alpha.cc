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

#include "pik_alpha.h"

#include <cstdio>

#include <memory>
#include <vector>

#include "brotli/decode.h"
#include "brotli/encode.h"
#include "bit_reader.h"
#include "fast_log.h"
#include "write_bits.h"

namespace pik {
namespace {

bool BrotliDecompress(const std::vector<uint8_t>& in,
                      size_t max_output_size,
                      size_t* bytes_read,
                      std::vector<uint8_t>* out) {
  std::unique_ptr<BrotliDecoderState, decltype(BrotliDecoderDestroyInstance)*>
      s(BrotliDecoderCreateInstance(nullptr, nullptr, nullptr),
      BrotliDecoderDestroyInstance);
  if (!s) return PIK_FAILURE("BrotliDecoderCreateInstance failed");

  const size_t kBufferSize = 128 * 1024;
  std::vector<uint8_t> temp_buffer(kBufferSize);

  size_t insize = in.size();
  size_t avail_in = insize;
  const uint8_t* next_in = &in[0];
  BrotliDecoderResult code;

  while (1) {
    size_t out_size;
    size_t avail_out = kBufferSize;
    uint8_t* next_out = temp_buffer.data();
    code = BrotliDecoderDecompressStream(
        s.get(), &avail_in, &next_in, &avail_out, &next_out, nullptr);
    out_size = next_out - temp_buffer.data();
    out->resize(out->size() + out_size);
    if (out->size() > max_output_size)
      return PIK_FAILURE("Too long alpha stream");
    memcpy(out->data() + out->size() - out_size, temp_buffer.data(), out_size);
    if (code != BROTLI_DECODER_RESULT_NEEDS_MORE_OUTPUT) break;
  }
  if (code != BROTLI_DECODER_RESULT_SUCCESS)
    return PIK_FAILURE("Decompressing alpha stream failed");
  *bytes_read += (insize - avail_in);
  return true;
}

bool BrotliCompress(int quality, const std::vector<uint8_t>& in,
                    std::vector<uint8_t>* out) {
  std::unique_ptr<BrotliEncoderState, decltype(BrotliEncoderDestroyInstance)*>
      enc(BrotliEncoderCreateInstance(nullptr, nullptr, nullptr),
      BrotliEncoderDestroyInstance);
  if (!enc) return PIK_FAILURE("BrotliEncoderCreateInstance failed");

  BrotliEncoderSetParameter(enc.get(), BROTLI_PARAM_QUALITY, quality);
  BrotliEncoderSetParameter(enc.get(), BROTLI_PARAM_LGWIN, 24);
  BrotliEncoderSetParameter(enc.get(), BROTLI_PARAM_LGBLOCK, 0);

  const size_t kBufferSize = 128 * 1024;
  std::vector<uint8_t> temp_buffer(kBufferSize);

  size_t insize = in.size();
  size_t avail_in = insize;
  const uint8_t* next_in = &in[0];

  size_t total_out = 0;

  while (1) {
    size_t out_size;
    size_t avail_out = kBufferSize;
    uint8_t* next_out = temp_buffer.data();
    if (!BrotliEncoderCompressStream(
        enc.get(), BROTLI_OPERATION_FINISH,
        &avail_in, &next_in, &avail_out, &next_out, &total_out)) {
      return PIK_FAILURE("Failed to compress alpha stream");
    }
    out_size = next_out - temp_buffer.data();
    out->resize(out->size() + out_size);
    memcpy(&(*out)[0] + out->size() - out_size, temp_buffer.data(), out_size);
    if (BrotliEncoderIsFinished(enc.get())) break;
  }

  return true;
}

ImageU FilterAlpha(const ImageU& in, int bit_depth) {
  ImageU out(in.xsize(), in.ysize());
  const size_t mask = (1 << bit_depth) - 1;
  for (size_t y = 0; y < in.ysize(); ++y) {
    const uint16_t* PIK_RESTRICT row_in = in.Row(y);
    uint16_t* PIK_RESTRICT row_out = out.Row(y);
    row_out[0] = row_in[0];
    for (size_t x = 1; x < in.xsize(); ++x) {
      row_out[x] = (row_in[x] - row_in[x - 1]) & mask;
    }
  }
  return out;
}

void UnfilterAlpha(int bit_depth, ImageU* img) {
  const size_t mask = (1 << bit_depth) - 1;
  for (size_t y = 0; y < img->ysize(); ++y) {
    uint16_t* PIK_RESTRICT row = img->Row(y);
    for (size_t x = 1; x < img->xsize(); ++x) {
      row[x] = (row[x] + row[x - 1]) & mask;
    }
  }
}

bool FilterAndBrotliEncode(const CompressParams& params,
                           const ImageU& plane, int bit_depth,
                           std::vector<uint8_t>* out) {
  const size_t xsize = plane.xsize();
  const size_t ysize = plane.ysize();
  const size_t stride = bit_depth / 8;
  ImageU filtered = FilterAlpha(plane, bit_depth);
  std::vector<uint8_t> data(xsize * ysize * stride);
  for (size_t y = 0; y < ysize; ++y) {
    uint16_t* PIK_RESTRICT row = filtered.Row(y);
    for (size_t x = 0; x < xsize; ++x) {
      for (int i = 0; i < stride; ++i) {
        data[(y * xsize + x) * stride + i] = (row[x] >> (8 * i)) & 255;
      }
    }
  }
  int quality = params.fast_mode ? 9 : 11;
  return BrotliCompress(quality, data, out);
}

bool BrotliDecodeAndUnfilter(const std::vector<uint8_t>& brotli,
                             const int bit_depth, ImageU* plane) {
  const size_t num_pixels = plane->xsize() * plane->ysize();
  const size_t stride = bit_depth / 8;
  std::vector<uint8_t> data;
  size_t bytes_read;
  if (!BrotliDecompress(brotli, num_pixels * stride, &bytes_read, &data)) {
    return false;
  }
  if (data.size() != num_pixels * stride) {
    return PIK_FAILURE("Incorrect size of the alpha stream");
  }
  for (size_t y = 0; y < plane->ysize(); ++y) {
    uint16_t* const PIK_RESTRICT row = plane->Row(y);
    for (size_t x = 0; x < plane->xsize(); ++x) {
      row[x] = 0;
      for (int i = 0; i < stride; ++i) {
        row[x] += data[(y * plane->xsize() + x) * stride + i] << (8 * i);
      }
    }
  }
  UnfilterAlpha(bit_depth, plane);
  return true;
}

struct AlphaRow {
  bool IsOpaque() const {
    return zeros_l == 0 && zeros_r == 0 && border_l.empty() && border_r.empty();
  }
  int zeros_l = 0;
  int zeros_r = 0;
  std::vector<uint16_t> border_l;
  std::vector<uint16_t> border_r;
};

struct AlphaImage {
  AlphaImage(const size_t xs, const size_t ys) :
      xsize(xs), ysize(ys), rows(ys) {}

  const size_t xsize;
  const size_t ysize;
  // Only the top border_t and the buttom border_b rows have any non-opaque
  // alpha values.
  int border_t = 0;
  int border_b = 0;
  std::vector<AlphaRow> rows;
};

bool TransformAlphaRow(const uint16_t* row, size_t xsize, int bit_depth,
                       AlphaRow* row_out) {
  int opaque = (1 << bit_depth) - 1;
  int x = 0;
  for (; row[x] == 0 && x < xsize; ++x) {
    ++row_out->zeros_l;
  }
  for (; x < xsize; ++x) {
    const uint16_t ref = x == 0 ? 0 : row[x - 1];
    if (row[x] > ref && row[x] < opaque) {
      row_out->border_l.push_back(row[x] - ref);
    } else {
      break;
    }
  }
  for (; row[x] == opaque && x < xsize; ++x) {}
  for (; x < xsize; ++x) {
    const uint16_t ref = x + 1 == xsize ? 0 : row[x + 1];
    if (row[x] > ref && row[x] < opaque) {
      row_out->border_r.push_back(row[x] - ref);
    } else {
      break;
    }
  }
  for (; row[x] == 0 && x < xsize; ++x) {
    ++row_out->zeros_r;
  }
  return x == xsize;
}

bool ReconstructAlphaRow(const AlphaRow& row_in, int bit_depth, size_t xsize,
                         uint16_t* row_out) {
  const int zl = row_in.zeros_l;
  const int zr = row_in.zeros_r;
  const size_t bl = row_in.border_l.size();
  const size_t br = row_in.border_r.size();
  const int opaque = (1 << bit_depth) - 1;

  if (zl < 0 || zr < 0 || zl + zr + bl + br > xsize) {
    return PIK_FAILURE("Invalid alpha image row.");
  }

  for (int i = 0; i < zl; ++i) {
    row_out[i] = 0;
  }
  for (int i = 0; i < zr; ++i) {
    row_out[xsize - 1 - i] = 0;
  }

  for (int i = 0, x = zl; i < bl; ++i, ++x) {
    const uint16_t ref = x == 0 ? 0 : row_out[x - 1];
    row_out[x] = ref + row_in.border_l[i];
    if (row_out[x] > opaque) {
      return PIK_FAILURE("Invalid alpha value 1.");
    }
  }
  for (int i = br - 1, x = xsize - zr - 1; i >= 0; --i, --x) {
    const uint16_t ref = x + 1 == xsize ? 0 : row_out[x + 1];
    row_out[x] = ref + row_in.border_r[i];
    if (row_out[x] > opaque) {
      return PIK_FAILURE("Invalid alpha value 2.");
    }
  }

  for (int x = zl + bl; x < xsize - zr - br; ++x) {
    row_out[x] = opaque;
  }
  return true;
}

bool TransformAlpha(const ImageU& plane, int bit_depth, AlphaImage* out) {
  for (int y = 0; y < plane.ysize(); ++y) {
    if (!TransformAlphaRow(plane.Row(y), plane.xsize(), bit_depth,
                           &out->rows[y])) {
      return false;  // not an error
    }
  }
  int y = 0;
  for (; y < plane.ysize(); ++y) {
    if ((y == 0 || out->rows[y].zeros_l <= out->rows[y - 1].zeros_l) &&
        !out->rows[y].IsOpaque()) {
      ++out->border_t;
    } else {
      break;
    }
  }
  for (; y < plane.ysize(); ++y) {
    if (!out->rows[y].IsOpaque()) {
      break;
    }
  }
  for (; y < plane.ysize(); ++y) {
    const int prev_zeros_l = out->rows[y - 1].zeros_l;
    const int cur_zeros_l = out->rows[y].zeros_l;
    if (cur_zeros_l >= prev_zeros_l) {
      ++out->border_b;
    } else {
      break;
    }
  }
  return y == plane.ysize();
}

bool ReconstructAlpha(const AlphaImage& alpha, int bit_depth, ImageU* plane) {
  PIK_ASSERT(alpha.xsize == plane->xsize());
  PIK_ASSERT(alpha.ysize == plane->ysize());
  for (int y = 0; y < plane->ysize(); ++y) {
    if (!ReconstructAlphaRow(alpha.rows[y], bit_depth,
                             plane->xsize(), plane->Row(y))) {
      return false;
    }
  }
  return true;
}

void EncodeNonNegative(int n, size_t* storage_ix, uint8_t* storage) {
  WriteBits(1, n != 0, storage_ix, storage);
  if (n > 0) {
    int nbits = Log2FloorNonZero(n);
    WriteBits(nbits + 1, (1 << nbits) - 1, storage_ix, storage);
    if (nbits > 0) {
      WriteBits(nbits, n - (1 << nbits), storage_ix, storage);
    }
  }
}

int DecodeNonNegative(BitReader* br) {
  if (!br->ReadBits(1)) return 0;
  int nbits = 0;
  while (br->ReadBits(1) && nbits < 30) {
    ++nbits;
  }
  return (1 << nbits) + br->ReadBits(nbits);
}

void EncodeSigned(int n, size_t* storage_ix, uint8_t* storage) {
  EncodeNonNegative(std::abs(n), storage_ix, storage);
  if (n != 0) {
    WriteBits(1, n < 0 ? 1 : 0, storage_ix, storage);
  }
}

int DecodeSigned(BitReader* br) {
  int absval = DecodeNonNegative(br);
  int sign = absval == 0 ? 1 : 1 - 2 * br->ReadBits(1);
  return sign * absval;
}

void EncodeTransformedAlpha(const AlphaImage& alpha, int bit_depth,
                            std::vector<uint8_t>* out) {
  const size_t max_out_size = 4 * alpha.ysize * alpha.xsize + 1024;
  out->resize(max_out_size);
  size_t storage_ix = 0;
  uint8_t* storage = &(*out)[0];
  storage[0] = 0;
  const int w_bits = Log2FloorNonZero(alpha.xsize) + 1;
  const int h_bits = Log2FloorNonZero(alpha.ysize) + 1;
  WriteBits(h_bits, alpha.border_t, &storage_ix, storage);
  WriteBits(h_bits, alpha.border_b, &storage_ix, storage);
  WriteBits(w_bits, alpha.rows[0].zeros_l, &storage_ix, storage);
  for (int y = 0; y < alpha.ysize; ++y) {
    if (y >= alpha.border_t && y < alpha.ysize - alpha.border_b) {
      // This is a fully opaque line.
      PIK_CHECK(alpha.rows[y].IsOpaque());
      continue;
    }
    const AlphaRow& r = alpha.rows[y];
    if (y > 0) {
      const int prev_zeros_l = alpha.rows[y - 1].zeros_l;
      const int zeros_l_diff = r.zeros_l - prev_zeros_l;
      const int sign = y < alpha.border_t ? -1 : 1;
      EncodeNonNegative(sign * zeros_l_diff, &storage_ix, storage);
    }
    EncodeSigned(r.zeros_r - r.zeros_l, &storage_ix, storage);
    const int bl = r.border_l.size();
    const int br = r.border_r.size();
    EncodeNonNegative(bl, &storage_ix, storage);
    EncodeSigned(br - bl, &storage_ix, storage);
    const int y_mirror = y < alpha.border_t ? -1 : alpha.ysize - 1 - y;
    for (int i = 0; i < bl; ++i) {
      if (y_mirror >= 0 && y_mirror < alpha.border_t &&
          i < alpha.rows[y_mirror].border_l.size()) {
        EncodeSigned(r.border_l[i] - alpha.rows[y_mirror].border_l[i],
                     &storage_ix, storage);
      } else {
        WriteBits(bit_depth, r.border_l[i], &storage_ix, storage);
      }
    }
    for (int i = br - 1; i >= 0; --i) {
      const int j = br - 1 - i;
      const int ref = j < bl ? r.border_l[j] : 0;
      EncodeSigned(r.border_r[i] - ref, &storage_ix, storage);
    }
  }
  out->resize((storage_ix >> 3) + 4);
}

bool DecodeTransformedAlpha(const std::vector<uint8_t>& data, int bit_depth,
                            AlphaImage* alpha) {
  // TODO(user): This now wastes 4 bytes for simplicity.
  if (data.size() <= 4) {
    return PIK_FAILURE("Transformed alpha data is empty.");
  }
  const size_t data_size_aligned = data.size() - 4;
  BitReader br(data.data(), data_size_aligned);
  const int w_bits = Log2FloorNonZero(alpha->xsize) + 1;
  const int h_bits = Log2FloorNonZero(alpha->ysize) + 1;
  alpha->border_t = br.ReadBits(h_bits);
  alpha->border_b = br.ReadBits(h_bits);
  alpha->rows[0].zeros_l = br.ReadBits(w_bits);
  for (int y = 0; y < alpha->ysize; ++y) {
    if (y >= alpha->border_t && y < alpha->ysize - alpha->border_b) {
      // This is a fully opaque line.
      continue;
    }
    AlphaRow* r = &(alpha->rows)[y];
    if (y > 0) {
      const int prev_zeros_l = alpha->rows[y - 1].zeros_l;
      if (y < alpha->border_t) {
        r->zeros_l = prev_zeros_l - DecodeNonNegative(&br);
      } else {
        r->zeros_l = DecodeNonNegative(&br) + prev_zeros_l;
      }
    }
    r->zeros_r = r->zeros_l + DecodeSigned(&br);
    const int b_l = DecodeNonNegative(&br);
    const int b_r = b_l + DecodeSigned(&br);
    if (b_r < 0) {
      return PIK_FAILURE("Invalid alpha image row.");
    }
    r->border_l.resize(b_l);
    r->border_r.resize(b_r);
    const int y_mirror = y < alpha->border_t ? -1 : alpha->ysize - 1 - y;
    for (int i = 0; i < b_l; ++i) {
      if (y_mirror >= 0 && y_mirror < alpha->border_t &&
          i < alpha->rows[y_mirror].border_l.size()) {
        r->border_l[i] = alpha->rows[y_mirror].border_l[i] + DecodeSigned(&br);
      } else {
        r->border_l[i] = br.ReadBits(bit_depth);
      }
    }
    for (int i = b_r - 1; i >= 0; --i) {
      const int j = b_r - 1 - i;
      const int ref = j < b_l ? r->border_l[j] : 0;
      r->border_r[i] = ref + DecodeSigned(&br);
    }
  }
  if (br.Position() > data_size_aligned) {
    return PIK_FAILURE("Premature EOF in alpha image.");
  }
  return true;
}

}  // namespace

bool AlphaToPik(const CompressParams& params, const ImageU& plane,
                int bit_depth, std::unique_ptr<Alpha>* out_alpha) {
  std::unique_ptr<Alpha> alpha(new Alpha);

  alpha->mode = Alpha::kModeBrotli;
  if (!FilterAndBrotliEncode(params, plane, bit_depth, &alpha->encoded)) {
    return false;
  }

  // Try alternative encoding that assumes that the opaque pixels form a convex
  // shape in the middle.
  AlphaImage tr_alpha(plane.xsize(), plane.ysize());
  if (TransformAlpha(plane, bit_depth, &tr_alpha)) {
    std::vector<uint8_t> tr_out;
    EncodeTransformedAlpha(tr_alpha, bit_depth, &tr_out);
    if (tr_out.size() < alpha->encoded.size()) {
      alpha->encoded = tr_out;
      alpha->mode = Alpha::kModeTransform;
    }
  }

  out_alpha->swap(alpha);
  return true;
}

bool PikToAlpha(const DecompressParams& params, const Alpha& alpha,
                ImageU* plane) {
  if (alpha.mode >= Alpha::kModeInvalid) {
    return PIK_FAILURE("Invalid alpha mode");
  }
  if (alpha.bytes_per_alpha != 1 && alpha.bytes_per_alpha != 2) {
    return PIK_FAILURE("Invalid bytes_per_alpha");
  }

  const int bit_depth = alpha.bytes_per_alpha * 8;
  if (alpha.mode == Alpha::kModeBrotli) {
    return BrotliDecodeAndUnfilter(alpha.encoded, bit_depth, plane);
  }

  AlphaImage tr_alpha(plane->xsize(), plane->ysize());
  return DecodeTransformedAlpha(alpha.encoded, bit_depth, &tr_alpha) &&
         ReconstructAlpha(tr_alpha, bit_depth, plane);
}

}  // namespace pik
