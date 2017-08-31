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

#include <vector>

#include "brotli/decode.h"
#include "brotli/encode.h"

namespace pik {
namespace {

bool BrotliDecompress(const std::vector<uint8_t>& in,
                      size_t max_output_size,
                      std::vector<uint8_t>* out) {
  BrotliDecoderState* s =
      BrotliDecoderCreateInstance(nullptr, nullptr, nullptr);
  if (!s) return false;

  const size_t kBufferSize = 128 * 1024;
  uint8_t* temp_buffer = reinterpret_cast<uint8_t*>(malloc(kBufferSize));

  size_t insize = in.size();
  size_t avail_in = insize;
  const uint8_t* next_in = &in[0];
  BrotliDecoderResult code;

  while (1) {
    size_t out_size;
    size_t avail_out = kBufferSize;
    uint8_t* next_out = temp_buffer;
    code = BrotliDecoderDecompressStream(
        s, &avail_in, &next_in, &avail_out, &next_out, nullptr);
    out_size = next_out - temp_buffer;
    out->resize(out->size() + out_size);
    if (out->size() > max_output_size) return false;
    memcpy(&(*out)[0] + out->size() - out_size, temp_buffer, out_size);
    if (code != BROTLI_DECODER_RESULT_NEEDS_MORE_OUTPUT) break;
  }
  if (code != BROTLI_DECODER_RESULT_SUCCESS) return false;

  free(temp_buffer);
  BrotliDecoderDestroyInstance(s);
  return true;
}

bool BrotliCompress(int quality, const std::vector<uint8_t>& in,
    std::vector<uint8_t>* out) {
  BrotliEncoderState* enc =
    BrotliEncoderCreateInstance(nullptr, nullptr, nullptr);
  if (!enc) return false;

  BrotliEncoderSetParameter(enc, BROTLI_PARAM_QUALITY, quality);
  BrotliEncoderSetParameter(enc, BROTLI_PARAM_LGWIN, 24);
  BrotliEncoderSetParameter(enc, BROTLI_PARAM_LGBLOCK, 0);

  const size_t kBufferSize = 128 * 1024;
  uint8_t* temp_buffer = reinterpret_cast<uint8_t*>(malloc(kBufferSize));

  size_t insize = in.size();
  size_t avail_in = insize;
  const uint8_t* next_in = &in[0];

  size_t total_out = 0;

  while (1) {
    size_t out_size;
    size_t avail_out = kBufferSize;
    uint8_t* next_out = temp_buffer;
    if (!BrotliEncoderCompressStream(
        enc, BROTLI_OPERATION_FINISH,
        &avail_in, &next_in, &avail_out, &next_out, &total_out)) {
      return false;
    }
    out_size = next_out - temp_buffer;
    out->resize(out->size() + out_size);
    memcpy(&(*out)[0] + out->size() - out_size, temp_buffer, out_size);
    if (BrotliEncoderIsFinished(enc)) break;
  }

  return true;
}


std::vector<uint8_t> DeltaEncode(
    const std::vector<uint8_t>& in, size_t stride) {
  std::vector<uint8_t> result(in.size());
  for (size_t j = 0; j < stride; j++) {
    if (j < in.size()) {
      result[j] = in[j];
    }
    for (size_t i = j + stride; i < in.size(); i += stride) {
      result[i] = in[i] - in[i - stride];
    }
  }
  return result;
}

std::vector<uint8_t> DeltaDecode(const std::vector<uint8_t>& in, int stride) {
  std::vector<uint8_t> result(in.size());
  for (size_t j = 0; j < stride; j++) {
    if (j < in.size()) {
      result[j] = in[j];
    }
    for (size_t i = j + stride; i < in.size(); i += stride) {
      result[i] = in[i] + result[i - stride];
    }
  }
  return result;
}

bool EncodeAlpha(const std::vector<uint8_t>& data, size_t stride,
    const CompressParams& params, size_t* bytepos, PaddedBytes* compressed) {
  std::vector<uint8_t> delta = DeltaEncode(data, stride);
  std::vector<uint8_t> brotli;
  int quality = params.fast_mode ? 9 : 11;
  if (!BrotliCompress(quality, delta, &brotli)) return false;
  compressed->resize(*bytepos + 1 + brotli.size());
  compressed->data()[*bytepos] = stride;
  memcpy(compressed->data() + *bytepos + 1, brotli.data(), brotli.size());
  *bytepos += brotli.size() + 1;
  return true;
}

bool DecodeAlpha(size_t stride, size_t num_pixels,
    const DecompressParams& params, size_t bytepos,
    const PaddedBytes& compressed, std::vector<uint8_t>* result) {
  if (bytepos + 1 >= compressed.size()) return false;
  size_t cstride = compressed.data()[bytepos++];
  std::vector<uint8_t> data(compressed.data() + bytepos,
      compressed.data() + compressed.size());
  std::vector<uint8_t> delta;
  if (!BrotliDecompress(data, num_pixels * cstride, &delta)) return false;
  if (delta.size() != num_pixels * cstride) return false;
  data = DeltaDecode(delta, cstride);
  if (stride == cstride) {
    result->swap(data);
  } else if (stride == 1 && cstride == 2) {
    if (data.size() & 1) return false;
    result->resize(data.size() / 2);
    for (size_t i = 0; i < result->size(); i++) {
      (*result)[i] = data[i * 2 + 1];
    }
  } else if (stride == 2 && cstride == 1) {
    result->resize(data.size() * 2);
    for (size_t i = 0; i < data.size(); i++) {
      (*result)[i * 2 + 0] = (*result)[i * 2 + 1] = data[i];
    }
  } else {
    return false;
  }

  return true;
}
}  // namespace

bool AlphaToPik(const CompressParams& params,
    const ImageB& plane, size_t* bytepos, PaddedBytes* compressed) {
  std::vector<uint8_t> data(plane.xsize() * plane.ysize());
  for (size_t y = 0; y < plane.ysize(); ++y) {
    auto row = plane.Row(y);
    for (size_t x = 0; x < plane.xsize(); ++x) {
      data[y * plane.xsize() + x] = row[x];
    }
  }
  return EncodeAlpha(data, 1, params, bytepos, compressed);
}

bool AlphaToPik(const CompressParams& params,
    const ImageF& plane, size_t* bytepos, PaddedBytes* compressed) {
  std::vector<uint8_t> data(plane.xsize() * plane.ysize() * 2);
  for (size_t y = 0; y < plane.ysize(); ++y) {
    auto row = plane.Row(y);
    for (size_t x = 0; x < plane.xsize(); ++x) {
      uint16_t v = static_cast<uint16_t>(std::round(row[x] * 257.0f));
      data[2 * y * plane.xsize() + 2 * x + 0] = v & 255;
      data[2 * y * plane.xsize() + 2 * x + 1] = (v >> 8) & 255;
    }
  }
  return EncodeAlpha(data, 2, params, bytepos, compressed);
}

bool AlphaToPik(const CompressParams& params,
    const ImageU& plane, size_t* bytepos, PaddedBytes* compressed) {
  std::vector<uint8_t> data(plane.xsize() * plane.ysize() * 2);
  for (size_t y = 0; y < plane.ysize(); ++y) {
    auto row = plane.Row(y);
    for (size_t x = 0; x < plane.xsize(); ++x) {
      data[2 * y * plane.xsize() + 2 * x + 0] = row[x] & 255;
      data[2 * y * plane.xsize() + 2 * x + 1] = (row[x] >> 8) & 255;
    }
  }
  return EncodeAlpha(data, 2, params, bytepos, compressed);
}

bool PikToAlpha(const DecompressParams& params,
    size_t bytepos, const PaddedBytes& compressed, ImageB* plane) {
  std::vector<uint8_t> data;
  if (!DecodeAlpha(1, plane->xsize() * plane->ysize(),
      params, bytepos, compressed, &data)) {
    return false;
  }

  for (size_t y = 0; y < plane->ysize(); ++y) {
    auto row = plane->Row(y);
    for (size_t x = 0; x < plane->xsize(); ++x) {
      row[x] = data[y * plane->xsize() + x];
    }
  }
  return true;
}

bool PikToAlpha(const DecompressParams& params,
    size_t bytepos, const PaddedBytes& compressed, ImageF* plane) {
  std::vector<uint8_t> data;
  if (!DecodeAlpha(2, plane->xsize() * plane->ysize(),
      params, bytepos, compressed, &data)) {
    return false;
  }

  for (size_t y = 0; y < plane->ysize(); ++y) {
    auto row = plane->Row(y);
    for (size_t x = 0; x < plane->xsize(); ++x) {
      uint16_t v = data[2 * y * plane->xsize() + 2 * x] +
          (data[2 * y * plane->xsize() + 2 * x + 1] << 8);
      row[x] = v / 257.0f;
    }
  }
  return true;
}

bool PikToAlpha(const DecompressParams& params,
    size_t bytepos, const PaddedBytes& compressed, ImageU* plane) {
  std::vector<uint8_t> data;
  if (!DecodeAlpha(2, plane->xsize() * plane->ysize(),
      params, bytepos, compressed, &data)) {
    return false;
  }

  for (size_t y = 0; y < plane->ysize(); ++y) {
    auto row = plane->Row(y);
    for (size_t x = 0; x < plane->xsize(); ++x) {
      uint16_t v = data[2 * y * plane->xsize() + 2 * x] +
          (data[2 * y * plane->xsize() + 2 * x + 1] << 8);
      row[x] = v;
    }
  }
  return true;
}

}  // namespace pik
