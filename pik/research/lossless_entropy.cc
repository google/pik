// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/lossless_entropy.h"

#include <vector>

#include "pik/base/padded_bytes.h"
#include "pik/common.h"

#define PIK_ENTROPY_SUPPORT_FSE 1
#define PIK_ENTROPY_SUPPORT_RANS 1
#define PIK_ENTROPY_SUPPORT_BROTLI 1

#if PIK_ENTROPY_SUPPORT_FSE
// clang-format off
#include "fse_wrapper.h"  // fse_wrapper.h must be included first.
#include "FiniteStateEntropy/lib/fse.h"
// clang-format on
#endif  // SUPPORT_FSE

#if PIK_ENTROPY_SUPPORT_RANS
#include "pik/entropy_coder.h"
#endif  // SUPPORT_RANS

#if PIK_ENTROPY_SUPPORT_BROTLI
#include "pik/brotli.h"
#endif  // PIK_ENTROPY_SUPPORT_BROTLI

namespace pik {

bool EncodeVarInt(uint64_t value, size_t output_size, size_t* output_pos,
                  uint8_t* output) {
  // While more than 7 bits of data are left,
  // store 7 bits and set the next byte flag
  while (value > 127) {
    if (*output_pos > output_size) return false;
    // |128: Set the next byte flag
    output[(*output_pos)++] = ((uint8_t)(value & 127)) | 128;
    // Remove the seven bits we just wrote
    value >>= 7;
  }
  if (*output_pos > output_size) return false;
  output[(*output_pos)++] = ((uint8_t)value) & 127;
  return true;
}

void EncodeVarInt(uint64_t value, PaddedBytes* data) {
  size_t pos = data->size();
  data->resize(data->size() + 9);
  PIK_CHECK(EncodeVarInt(value, data->size(), &pos, data->data()));
  data->resize(pos);
}

size_t EncodeVarInt(uint64_t value, uint8_t* output) {
  size_t pos = 0;
  EncodeVarInt(value, 9, &pos, output);
  return pos;
}

uint64_t DecodeVarInt(const uint8_t* input, size_t inputSize, size_t* pos) {
  size_t i;
  uint64_t ret = 0;
  for (i = 0; *pos + i < inputSize && i < 10; ++i) {
    ret |= uint64_t(input[*pos + i] & 127) << uint64_t(7 * i);
    // If the next-byte flag is not set, stop
    if ((input[*pos + i] & 128) == 0) break;
  }
  // TODO: Return a decoding error if i == 10.
  *pos += i + 1;
  return ret;
}

bool IsRLECompressible(const uint8_t* data, size_t size) {
  if (size < 4) return false;
  uint8_t first = data[0];
  for (size_t i = 1; i < size; i++) {
    if (data[i] != first) return false;
  }
  return true;
}

#if PIK_ENTROPY_SUPPORT_FSE

// Output size can have special meaning, in each case you must encode the
// data differently yourself and EntropyDecode will not be able to decode it.
// If 0, then compression was not able to reduce size and you should output
// uncompressed.
// If 1, then the input data has exactly one byte repeated size times, and
// you must RLE compress it (encode the amount of times the one value repeats)
bool MaybeEntropyEncodeFse(const uint8_t* data, size_t size,
                           size_t out_capacity, uint8_t* out,
                           size_t* out_size) {
  size_t cs = FSE_compress2(out, out_capacity, data, size, 255,
                            /*FSE_MAX_TABLELOG=*/12);
  if (FSE_isError(cs)) {
    return PIK_FAILURE("FSE enc error: %s", FSE_getErrorName(cs));
  }
  *out_size = cs;
  return true;
}

// Does not know or return the compressed size, must be known from external
// source.
bool MaybeEntropyDecodeFse(const uint8_t* data, size_t size,
                           size_t out_capacity, uint8_t* out,
                           size_t* out_size) {
  size_t ds = FSE_decompress(out, out_capacity, data, size);
  if (FSE_isError(ds)) {
    return PIK_FAILURE("FSE dec error: %s", FSE_getErrorName(ds));
  }
  *out_size = ds;
  return true;
}
#endif  // PIK_ENTROPY_SUPPORT_FSE

#if PIK_ENTROPY_SUPPORT_RANS

// Entropy encode with pik ANS
bool EntropyEncodePikANS(const uint8_t* data, size_t size,
                         std::vector<uint8_t>* result) {
  static const int kAlphabetSize = 256;
  static const int kContext = 0;

  std::vector<ANSHistBin> histogram(kAlphabetSize, 0);
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

  pos += EncodeVarInt(size, storage + pos);
  std::vector<ANSEncodingData> encoding_codes(1);
  BitWriter writer;
  BitWriter::Allotment allotment(&writer, cost_bound * kBitsPerByte);
  encoding_codes[0].BuildAndStore(histogram, &writer);

  std::vector<uint8_t> dummy_context_map;
  dummy_context_map.push_back(0);  // only 1 histogram
  ANSSymbolWriter ans_writer(encoding_codes, dummy_context_map, &writer);
  for (size_t i = 0; i < size; i++) {
    ans_writer.VisitSymbol(data[i], kContext);
  }
  ans_writer.FlushToBitStream();
  writer.ZeroPadToByte();
  Span<const uint8_t> span = writer.GetSpan();
  result->insert(result->end(), span.data(), span.data() + span.size());
  pos += span.size();  // bytes_written;
  ReclaimAndCharge(&writer, &allotment, 0, nullptr);

  return true;
}

// Entropy decode with pik ANS
bool EntropyDecodePikANS(const uint8_t* data, size_t size,
                         size_t max_output_size,
                         std::vector<uint8_t>* result) {
  static const int kContext = 0;
  size_t pos = 0;
  size_t num_symbols = DecodeVarInt(data, size, &pos);
  if (pos >= size) {
    return PIK_FAILURE("lossless pik ANS decode failed");
  }
  if (num_symbols > max_output_size) {
    // Avoid large allocations, we never expect this many symbols for
    // the limited group sizes.
    return PIK_FAILURE("lossless pik ANS decode too large %zu", num_symbols);
  }

  BitReader br(Span<const uint8_t>(data + pos, size - pos));
  ANSCode codes;
  if (!DecodeANSCodes(1, 256, &br, &codes)) {
    return PIK_FAILURE("lossless pik ANS decode failed");
  }

  result->resize(num_symbols);
  ANSSymbolReader reader(&codes);
  for (size_t i = 0; i < num_symbols; i++) {
    br.FillBitBuffer();
    int read_symbol = reader.ReadSymbol(kContext, &br);
    (*result)[i] = read_symbol;
  }
  if (!reader.CheckANSFinalState()) {
    return PIK_FAILURE("lossless pik ANS decode final state failed");
  }

  return true;
}

// TODO(lode): avoid the copying between std::vector and data.
// Entropy encode with pik ANS
bool MaybeEntropyEncodePikANS(const uint8_t* data, size_t size,
                              size_t out_capacity, uint8_t* out,
                              size_t* out_size) {
  if (IsRLECompressible(data, size)) {
    *out_size = 1;  // Indicate the codec should use RLE instead,
    return true;
  }
  std::vector<uint8_t> result;
  if (!EntropyEncodePikANS(data, size, &result)) {
    return PIK_FAILURE("lossless entropy encoding failed");
  }
  if (result.size() > size) {
    *out_size = 0;  // Indicate the codec should use uncompressed mode instead.
    return true;
  }
  if (result.size() > out_capacity) {
    return PIK_FAILURE("lossless entropy encoding out of capacity");
  }
  memcpy(out, result.data(), result.size());
  *out_size = result.size();
  return true;
}

// Entropy decode with pik ANS
bool MaybeEntropyDecodePikANS(const uint8_t* data, size_t size,
                              size_t out_capacity, uint8_t* out,
                              size_t* out_size) {
  std::vector<uint8_t> result;
  if (!EntropyDecodePikANS(data, size, out_capacity, &result)) {
    return PIK_FAILURE("lossless entropy decoding failed");
  }
  if (result.size() > out_capacity) {
    return PIK_FAILURE("lossless entropy encoding out of capacity");
  }
  memcpy(out, result.data(), result.size());
  *out_size = result.size();
  return true;
}

#endif  // PIK_ENTROPY_SUPPORT_RANS

#if PIK_ENTROPY_SUPPORT_BROTLI

bool MaybeEntropyEncodeBrotli(const uint8_t* data, size_t size,
                              size_t out_capacity, uint8_t* out,
                              size_t* out_size) {
  *out_size = 0;
  PIK_RETURN_IF_ERROR(BrotliCompress(11, data, size, out, out_size));
  if (*out_size > out_capacity) {
    return PIK_FAILURE("MaybeEntropyEncode exceeded buffer");
  }
  return true;
}

bool MaybeEntropyDecodeBrotli(const uint8_t* data, size_t size,
                              size_t out_capacity, uint8_t* out,
                              size_t* out_size) {
  size_t bytes_read = 0;
  PaddedBytes padded_out;

  PIK_RETURN_IF_ERROR(BrotliDecompress(Span<const uint8_t>(data, size),
      out_capacity, &bytes_read, &padded_out));
  *out_size = padded_out.size();
  memcpy(out, padded_out.data(), padded_out.size());
  return true;
}

#endif  // PIK_ENTROPY_SUPPORT_FSE

bool MaybeEntropyEncode(LosslessEntropyCodec codec, const uint8_t* data,
                        size_t size, size_t out_capacity, uint8_t* out,
                        size_t* out_size) {
  if (codec == LosslessEntropyCodec::FSE) {
#if PIK_ENTROPY_SUPPORT_FSE
    return MaybeEntropyEncodeFse(data, size, out_capacity, out, out_size);
#else
    return PIK_FAILURE("Codec FSE not supported");
#endif
  } else if (codec == LosslessEntropyCodec::RANS) {
#if PIK_ENTROPY_SUPPORT_RANS
    return MaybeEntropyEncodePikANS(data, size, out_capacity, out, out_size);
#else
    return PIK_FAILURE("Codec RANS not supported");
#endif
  } else if (codec == LosslessEntropyCodec::BROTLI) {
#if PIK_ENTROPY_SUPPORT_BROTLI
    return MaybeEntropyEncodeBrotli(data, size, out_capacity, out, out_size);
#else
    return PIK_FAILURE("Codec BROTLI not supported");
#endif
  } else {
    return PIK_FAILURE("unknown entropy codec");
  }
}

bool MaybeEntropyDecode(LosslessEntropyCodec codec, const uint8_t* data,
                        size_t size, size_t out_capacity, uint8_t* out,
                        size_t* out_size) {
  if (codec == LosslessEntropyCodec::FSE) {
#if PIK_ENTROPY_SUPPORT_FSE
    return MaybeEntropyDecodeFse(data, size, out_capacity, out, out_size);
#else
    return PIK_FAILURE("Codec FSE not supported");
#endif
  } else if (codec == LosslessEntropyCodec::RANS) {
#if PIK_ENTROPY_SUPPORT_RANS
    return MaybeEntropyDecodePikANS(data, size, out_capacity, out, out_size);
#else
    return PIK_FAILURE("Codec RANS not supported");
#endif
  } else if (codec == LosslessEntropyCodec::BROTLI) {
#if PIK_ENTROPY_SUPPORT_BROTLI
    return MaybeEntropyDecodeBrotli(data, size, out_capacity, out, out_size);
#else
    return PIK_FAILURE("Codec BROTLI not supported");
#endif
  } else {
    return PIK_FAILURE("unknown entropy codec");
  }
}

bool CompressWithEntropyCode(LosslessEntropyCodec codec, size_t* pos,
                             size_t src_size, const uint8_t* src,
                             size_t dst_capacity, uint8_t* dst) {
  if (src_size == 0) {
    *pos += EncodeVarInt(0, dst + *pos);
    return true;
  }
  // Ensure large enough for brotli and FSE
  std::vector<uint8_t> temp(src_size * 2 + 1024);
  size_t cs;
  if (IsRLECompressible(src, src_size)) {
    cs = 1;  // use RLE encoding instead
  } else {
    if (!MaybeEntropyEncode(codec, src, src_size, temp.size(), temp.data(),
                            &cs)) {
      return PIK_FAILURE("lossless entropy encode failed");
    }
  }
  if (cs >= src_size) cs = 0;  // EntropyCode worse than original, use memcpy.

  if (*pos + dst_capacity < 9) {
    return PIK_FAILURE("lossless entropy encode: dst too small");
  }
  size_t cs_with_mode = cs <= 1 ? (src_size - 1) * 3 + 1 + cs : cs * 3;
  *pos += EncodeVarInt(cs_with_mode, dst + *pos);

  if (cs == 1) {
    if (*pos + 1 >= dst_capacity) {
      return PIK_FAILURE("lossless entropy encode: dst too small");
    }
    dst[(*pos)++] = *src;
  } else if (cs == 0) {
    if (*pos + src_size >= dst_capacity) {
      return PIK_FAILURE("lossless entropy encode: dst too small");
    }
    // Use memmove because src and dst are allowed to overlap
    memmove(dst + *pos, src, src_size);
    *pos += src_size;
  } else {
    if (*pos + cs >= dst_capacity) {
      return PIK_FAILURE("lossless entropy encode: dst too small");
    }
    memcpy(dst + *pos, temp.data(), cs);
    *pos += cs;
  }
  return true;
}

bool DecompressWithEntropyCode(LosslessEntropyCodec codec, uint8_t* dst,
                               size_t dst_capacity, const uint8_t* src,
                               size_t src_capacity, size_t* ds, size_t* pos) {
  size_t cs = DecodeVarInt(src, src_capacity, pos);
  if (cs == 0) {
    *ds = 0;
    return true;
  }
  size_t mode = cs % 3;
  cs /= 3;
  if (mode == 2) {
    if (*pos >= src_capacity) return PIK_FAILURE("entropy decode failed");
    if (cs + 1 > dst_capacity) return PIK_FAILURE("entropy decode failed");
    memset(dst, src[(*pos)++], ++cs);
    *ds = cs;
  } else if (mode == 1) {
    if (*pos + cs + 1 > src_capacity)
      return PIK_FAILURE("entropy decode failed");
    if (cs + 1 > dst_capacity) return PIK_FAILURE("entropy decode failed");
    memcpy(dst, &src[*pos], ++cs);
    *pos += cs;
    *ds = cs;
  } else {
    if (*pos + cs > src_capacity) return PIK_FAILURE("entropy decode failed");
    if (!MaybeEntropyDecode(codec, &src[*pos], cs, dst_capacity, dst, ds)) {
      return PIK_FAILURE("entropy decode failed");
    }
    *pos += cs;
  }

  return true;
}

bool CompressWithEntropyCode(LosslessEntropyCodec codec, size_t* pos,
                             const size_t* src_size, const uint8_t* const* src,
                             size_t num_src, size_t dst_capacity,
                             uint8_t* dst) {
  // Brotli does its own clustering so concatenate all
  bool use_concatenation = codec == LosslessEntropyCodec::BROTLI;

  if (use_concatenation) {
    size_t total_size = 0;
    for (size_t i = 0; i < num_src; i++) {
      total_size += src_size[i];
    }
    size_t total_capacity = total_size + 9 * num_src;
    std::vector<uint8_t> all(total_capacity);
    size_t all_pos = 0;
    for (size_t i = 0; i < num_src; i++) {
      if (!EncodeVarInt(src_size[i], total_capacity, &all_pos, all.data())) {
        return false;
      }
      memcpy(all.data() + all_pos, src[i], src_size[i]);
      all_pos += src_size[i];
    }
    all.resize(all_pos);
    if (!CompressWithEntropyCode(codec, pos, all.size(), all.data(),
                                 dst_capacity, dst)) {
      return false;
    }
  } else {
    for (size_t i = 0; i < num_src; i++) {
      if (!CompressWithEntropyCode(codec, pos, src_size[i], src[i],
                                   dst_capacity, dst)) {
        return false;
      }

      // If there are two length zero streams in a row, indicate how many more
      // zero streams follow and skip them.
      if (i >= 1 && src_size[i] == 0 && src_size[i - 1] == 0) {
        size_t num = 0;
        while (num + i + 1 < num_src && src_size[num + i + 1] == 0) {
          num++;
        }
        if (!EncodeVarInt(num, dst_capacity, pos, dst)) {
          return PIK_FAILURE("dst too small");
        }
        i += num;
      }
    }
  }
  return true;
}

bool DecompressWithEntropyCode(LosslessEntropyCodec codec, size_t src_capacity,
                               const uint8_t* src, size_t max_decompressed_size,
                               size_t num_dst, std::vector<uint8_t>* dst,
                               size_t* pos) {
  // Brotli does its own clustering so concatenate all
  bool use_concatenation = codec == LosslessEntropyCodec::BROTLI;

  if (use_concatenation) {
    std::vector<uint8_t> all(max_decompressed_size + 9 * num_dst);
    size_t ds;
    if (!DecompressWithEntropyCode(codec, all.data(), all.size(), src,
                                   src_capacity, &ds, pos)) {
      return false;
    }
    all.resize(ds);
    size_t allpos = 0;
    for (size_t i = 0; i < num_dst; i++) {
      if (allpos >= all.size()) return PIK_FAILURE("out of bounds");
      uint64_t size = DecodeVarInt(all.data(), all.size(), &allpos);
      if (allpos + size > all.size()) return PIK_FAILURE("out of bounds");
      dst[i].resize(size);
      memcpy(dst[i].data(), all.data() + allpos, size);
      allpos += size;
    }
  } else {
    std::vector<uint8_t> buffer(max_decompressed_size);
    size_t ds;
    for (size_t i = 0; i < num_dst; i++) {
      if (!DecompressWithEntropyCode(codec, buffer.data(),
                                     max_decompressed_size, src, src_capacity,
                                     &ds, pos)) {
        return false;
      }

      // If there are two length zero streams in a row, read amount of zero
      // streams that follow and skip them.
      dst[i].assign(buffer.data(), buffer.data() + ds);
      if (i >= 1 && dst[i].empty() && dst[i - 1].empty()) {
        if (*pos >= src_capacity) return PIK_FAILURE("src out of bounds");
        size_t num = DecodeVarInt(src, src_capacity, pos);
        if (i + num >= num_dst) return PIK_FAILURE("too many streams");
        for (size_t j = 0; j < num; j++) {
          dst[++i].clear();
        }
      }
    }
  }
  return true;
}

}  // namespace pik
