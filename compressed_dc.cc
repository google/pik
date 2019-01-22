// Copyright 2019 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "compressed_dc.h"

#include "common.h"
#include "compressed_image_fwd.h"
#include "data_parallel.h"
#include "entropy_coder.h"
#include "gradient_map.h"
#include "image.h"
#include "lossless16.h"
#include "lossless8.h"
#include "padded_bytes.h"
#include "pik_info.h"
#include "size_coder.h"

namespace pik {
namespace {

// If grayscale, only the second channel (y) is encoded.
bool Image3SCompress(const Image3S& img, const Rect& rect, bool grayscale,
                     PaddedBytes* bytes) {
  std::array<int16_t, 3> min;
  std::array<int16_t, 3> max;
  Image3MinMax(img, rect, &min, &max);
  bool fit8 = true;  // If all values fit in 8-bit, use the 8-bit codec.
  for (int c = 0; c < 3; c++) {
    if (grayscale && c != 1) continue;
    bytes->push_back(min[c] & 255);
    bytes->push_back(min[c] >> 8);
    if (max[c] - min[c] >= 256) fit8 = false;
  }
  bytes->push_back(fit8);

  if (fit8) {
    if (grayscale) {
      ImageB image(rect.xsize(), rect.ysize());
      for (size_t y = 0; y < rect.ysize(); ++y) {
        const int16_t* PIK_RESTRICT row_in = rect.ConstPlaneRow(img, 1, y);
        uint8_t* PIK_RESTRICT row_out = image.Row(y);
        for (size_t x = 0; x < img.xsize(); ++x) {
          row_out[x] = static_cast<uint8_t>(row_in[x] - min[1]);
        }
      }
      return Grayscale8bit_compress(image, bytes);
    } else {
      Image3B image(rect.xsize(), rect.ysize());
      for (int c = 0; c < 3; ++c) {
        for (size_t y = 0; y < rect.ysize(); ++y) {
          const int16_t* PIK_RESTRICT row_in = rect.ConstPlaneRow(img, c, y);
          uint8_t* PIK_RESTRICT row_out = image.PlaneRow(c, y);
          for (size_t x = 0; x < img.xsize(); ++x) {
            row_out[x] = static_cast<uint8_t>(row_in[x] - min[c]);
          }
        }
      }
      return Colorful8bit_compress(image, bytes);
    }
  } else {
    if (grayscale) {
      ImageU image(rect.xsize(), rect.ysize());
      for (size_t y = 0; y < rect.ysize(); ++y) {
        const int16_t* PIK_RESTRICT row_in = rect.ConstPlaneRow(img, 1, y);
        uint16_t* PIK_RESTRICT row_out = image.Row(y);
        for (size_t x = 0; x < img.xsize(); ++x) {
          row_out[x] = static_cast<uint16_t>(row_in[x] - min[1]);
        }
      }
      return Grayscale16bit_compress(image, bytes);
    } else {
      Image3U image(rect.xsize(), rect.ysize());
      for (int c = 0; c < 3; ++c) {
        for (size_t y = 0; y < rect.ysize(); ++y) {
          const int16_t* PIK_RESTRICT row_in = rect.ConstPlaneRow(img, c, y);
          uint16_t* PIK_RESTRICT row_out = image.PlaneRow(c, y);
          for (size_t x = 0; x < img.xsize(); ++x) {
            row_out[x] = static_cast<uint16_t>(row_in[x] - min[c]);
          }
        }
      }
      return Colorful16bit_compress(image, bytes);
    }
  }
}

// If grayscale, only the second channel (y) is decoded.
bool Image3SDecompress(const PaddedBytes& bytes, bool grayscale, size_t* pos,
                       Image3S* result) {
  if (bytes.size() < *pos + 12) return PIK_FAILURE("Could not decode range");
  std::array<int16_t, 3> min;
  for (int c = 0; c < 3; c++) {
    if (grayscale && c != 1) continue;
    min[c] = static_cast<int16_t>(bytes[*pos] + (bytes[*pos + 1] << 8));
    *pos += 2;
  }
  bool fit8 = bytes[(*pos)++];

  if (fit8) {
    if (grayscale) {
      ImageB image;
      if (!Grayscale8bit_decompress(bytes, pos, &image)) {
        return PIK_FAILURE("Failed to decode DC");
      }
      *result = Image3S(image.xsize(), image.ysize());
      for (size_t y = 0; y < result->ysize(); ++y) {
        const uint8_t* PIK_RESTRICT row_in = image.Row(y);
        int16_t* PIK_RESTRICT row_out0 = result->PlaneRow(0, y);
        int16_t* PIK_RESTRICT row_out1 = result->PlaneRow(1, y);
        int16_t* PIK_RESTRICT row_out2 = result->PlaneRow(2, y);
        std::fill(row_out0, row_out0 + image.xsize(), 0);
        for (size_t x = 0; x < image.xsize(); ++x) {
          row_out1[x] = static_cast<int16_t>(row_in[x]) + min[1];
        }
        std::fill(row_out2, row_out2 + image.xsize(), 0);
      }
    } else {
      Image3B image;
      if (!Colorful8bit_decompress(bytes, pos, &image)) {
        return PIK_FAILURE("Failed to decode DC");
      }
      *result = Image3S(image.xsize(), image.ysize());
      for (int c = 0; c < 3; ++c) {
        for (size_t y = 0; y < result->ysize(); ++y) {
          const uint8_t* PIK_RESTRICT row_in = image.PlaneRow(c, y);
          int16_t* PIK_RESTRICT row_out = result->PlaneRow(c, y);
          for (size_t x = 0; x < image.xsize(); ++x) {
            row_out[x] = static_cast<int16_t>(row_in[x]) + min[c];
          }
        }
      }
    }
  } else {
    if (grayscale) {
      ImageU image;
      if (!Grayscale16bit_decompress(bytes, pos, &image)) {
        return PIK_FAILURE("Failed to decode DC");
      }
      *result = Image3S(image.xsize(), image.ysize());
      for (size_t y = 0; y < result->ysize(); ++y) {
        const uint16_t* PIK_RESTRICT row_in = image.Row(y);
        int16_t* PIK_RESTRICT row_out0 = result->PlaneRow(0, y);
        int16_t* PIK_RESTRICT row_out1 = result->PlaneRow(1, y);
        int16_t* PIK_RESTRICT row_out2 = result->PlaneRow(2, y);
        std::fill(row_out0, row_out0 + image.xsize(), 0);
        for (size_t x = 0; x < image.xsize(); ++x) {
          row_out1[x] = static_cast<int16_t>(row_in[x]) + min[1];
        }
        std::fill(row_out2, row_out2 + image.xsize(), 0);
      }
    } else {
      Image3U image;
      if (!Colorful16bit_decompress(bytes, pos, &image)) {
        return PIK_FAILURE("Failed to decode DC");
      }
      *result = Image3S(image.xsize(), image.ysize());
      for (int c = 0; c < 3; ++c) {
        for (size_t y = 0; y < result->ysize(); ++y) {
          const uint16_t* PIK_RESTRICT row_in = image.PlaneRow(c, y);
          int16_t* PIK_RESTRICT row_out = result->PlaneRow(c, y);
          for (size_t x = 0; x < image.xsize(); ++x) {
            row_out[x] = static_cast<int16_t>(row_in[x]) + min[c];
          }
        }
      }
    }
  }

  return true;
}

//
// Dequantizes and inverse color-transforms the provided quantized DC, to the
// window `rect` within the entire output image `enc_cache->dc`.
SIMD_ATTR void DequantDC(const Image3S& img_dc16, const Rect& rect,
                         const float* mul_dc, const float ytox_dc,
                         const float ytob_dc,
                         PassDecCache* PIK_RESTRICT pass_dec_cache) {
  PIK_ASSERT(SameSize(img_dc16, rect));
  const size_t xsize = rect.xsize();
  const size_t ysize = rect.ysize();

  using D = SIMD_FULL(float);
  constexpr D d;
  constexpr SIMD_PART(int16_t, D::N) d16;
  constexpr SIMD_PART(int32_t, D::N) d32;

  const auto dequant_y = set1(d, mul_dc[1]);

  for (size_t by = 0; by < ysize; ++by) {
    const int16_t* PIK_RESTRICT row_y16 = img_dc16.ConstPlaneRow(1, by);
    float* PIK_RESTRICT row_y = rect.PlaneRow(&pass_dec_cache->dc, 1, by);

    for (size_t bx = 0; bx < xsize; bx += d.N) {
      const auto quantized_y16 = load(d16, row_y16 + bx);
      const auto quantized_y = convert_to(d, convert_to(d32, quantized_y16));
      const auto dequantized_y = quantized_y * dequant_y;
      store(dequantized_y, d, row_y + bx);
    }
  }

  for (int c = 0; c < 3; c += 2) {  // === for c in {0, 2}
    const auto y_mul = set1(d, (c == 0) ? ytox_dc : ytob_dc);
    const auto xb_mul = set1(d, mul_dc[c]);
    for (size_t by = 0; by < ysize; ++by) {
      const int16_t* PIK_RESTRICT row_xb16 = img_dc16.ConstPlaneRow(c, by);
      const float* PIK_RESTRICT row_y =
          rect.ConstPlaneRow(pass_dec_cache->dc, 1, by);
      float* PIK_RESTRICT row_xb = rect.PlaneRow(&pass_dec_cache->dc, c, by);

      for (size_t bx = 0; bx < xsize; bx += d.N) {
        const auto quantized_xb16 = load(d16, row_xb16 + bx);
        const auto quantized_xb =
            convert_to(d, convert_to(d32, quantized_xb16));

        const auto out_y = load(d, row_y + bx);
        const auto dequant_xb = quantized_xb * xb_mul;
        const auto out_xb = mul_add(y_mul, out_y, dequant_xb);
        store(out_xb, d, row_xb + bx);
      }
    }
  }
}

// `rect`: block units
std::string CompressDCGroup(const Image3S& dc, const Rect& rect,
                            bool use_new_dc, bool grayscale,
                            PikImageSizeInfo* dc_info) {
  std::string dc_code;
  if (use_new_dc) {
    PaddedBytes enc_dc;
    Image3SCompress(dc, rect, grayscale, &enc_dc);
    dc_code.assign(enc_dc.data(), enc_dc.data() + enc_dc.size());
  } else {
    Image3S tmp_dc_residuals(rect.xsize(), rect.ysize());
    ShrinkDC(rect, dc, &tmp_dc_residuals);
    dc_code =
        EncodeImageData(Rect(tmp_dc_residuals), tmp_dc_residuals, dc_info);
  }
  return dc_code;
}

// `rect`: block units.
Status DecodeDCGroup(BitReader* reader, const PaddedBytes& compressed,
                     const Rect& rect, bool use_new_dc, bool grayscale,
                     const float* mul_dc, const float ytox_dc,
                     const float ytob_dc, PassDecCache* pass_dec_cache) {
  Image3S quantized_dc(rect.xsize(), rect.ysize());
  if (use_new_dc) {
    size_t dc_pos = reader->Position();
    if (!Image3SDecompress(compressed, grayscale, &dc_pos, &quantized_dc)) {
      return PIK_FAILURE("Failed to decode DC");
    }
  } else {
    ImageS dc_y(rect.xsize(), rect.ysize());
    ImageS dc_xz_residuals(rect.xsize() * 2, rect.ysize());
    ImageS dc_xz_expanded(rect.xsize() * 2, rect.ysize());
    if (!DecodeImage(reader, Rect(quantized_dc), &quantized_dc)) {
      return PIK_FAILURE("Failed to decode DC image");
    }

    ExpandDC(Rect(quantized_dc), &quantized_dc, &dc_y, &dc_xz_residuals,
             &dc_xz_expanded);
  }
  PIK_RETURN_IF_ERROR(reader->JumpToByteBoundary());
  DequantDC(quantized_dc, rect, mul_dc, ytox_dc, ytob_dc, pass_dec_cache);
  return true;
}

// TODO(veluca): is this the right constant?
using DCGroupSizeCoder = SizeCoderT<0x150F0E0C>;

}  // namespace

PaddedBytes EncodeDC(const Quantizer& quantizer,
                     const PassEncCache& pass_enc_cache, ThreadPool* pool,
                     PikImageSizeInfo* dc_info) {
  PaddedBytes out;

  const size_t xsize_blocks = pass_enc_cache.dc.xsize();
  const size_t ysize_blocks = pass_enc_cache.dc.ysize();
  const size_t xsize_groups =
      DivCeil(pass_enc_cache.dc.xsize(), kDcGroupDimInBlocks);
  const size_t ysize_groups =
      DivCeil(pass_enc_cache.dc.ysize(), kDcGroupDimInBlocks);

  const size_t num_groups = xsize_groups * ysize_groups;

  std::vector<std::unique_ptr<PikImageSizeInfo>> size_info(num_groups);
  for (size_t group_index = 0; group_index < num_groups; ++group_index) {
    if (dc_info != nullptr) {
      size_info[group_index] = make_unique<PikImageSizeInfo>();
    }
  }

  std::vector<PaddedBytes> group_codes(num_groups);
  const auto process_group = [&](const int group_index, const int thread) {
    size_t group_pos = 0;
    const size_t gx = group_index % xsize_groups;
    const size_t gy = group_index / xsize_groups;
    const Rect rect(gx * kDcGroupDimInBlocks, gy * kDcGroupDimInBlocks,
                    kDcGroupDimInBlocks, kDcGroupDimInBlocks, xsize_blocks,
                    ysize_blocks);
    std::string group_code = CompressDCGroup(
        pass_enc_cache.dc, rect, pass_enc_cache.use_new_dc,
        pass_enc_cache.grayscale_opt, size_info[group_index].get());
    group_codes[group_index].resize(group_code.size());
    Append(group_code, &group_codes[group_index], &group_pos);
  };
  RunOnPool(pool, 0, num_groups, process_group, "EncodeDC");

  for (size_t group_index = 0; group_index < num_groups; ++group_index) {
    if (dc_info != nullptr) {
      dc_info->Assimilate(*size_info[group_index]);
    }
  }

  // Build TOC.
  PaddedBytes group_toc(DCGroupSizeCoder::MaxSize(num_groups));
  size_t group_toc_pos = 0;
  uint8_t* group_toc_storage = group_toc.data();
  size_t total_groups_size = 0;
  for (size_t group_index = 0; group_index < num_groups; ++group_index) {
    size_t group_size = group_codes[group_index].size();
    DCGroupSizeCoder::Encode(group_size, &group_toc_pos, group_toc_storage);
    total_groups_size += group_size;
  }
  WriteZeroesToByteBoundary(&group_toc_pos, group_toc_storage);
  group_toc.resize(group_toc_pos / kBitsPerByte);

  PaddedBytes serialized_gradient_map;
  if (pass_enc_cache.use_gradient) {
    SerializeGradientMap(pass_enc_cache.gradient, Rect(pass_enc_cache.dc),
                         quantizer, &serialized_gradient_map);
  }

  // Push output.
  size_t pos = 0;
  out.reserve(group_toc.size() + total_groups_size +
              serialized_gradient_map.size());
  out.append(group_toc);
  pos += group_toc.size() * kBitsPerByte;
  for (size_t group_index = 0; group_index < num_groups; ++group_index) {
    const PaddedBytes& group_code = group_codes[group_index];
    out.append(group_code);
    pos += group_code.size() * kBitsPerByte;
  }
  out.append(serialized_gradient_map);

  return out;
}

Status DecodeDC(BitReader* reader, const PaddedBytes& compressed,
                const PassHeader& pass_header, size_t xsize_blocks,
                size_t ysize_blocks, const Quantizer& quantizer,
                const ColorCorrelationMap& cmap, ThreadPool* pool,
                PassDecCache* pass_dec_cache) {
  const float* dequant_matrices = quantizer.DequantMatrix(0, kQuantKindDCT8);
  float mul_dc[3];
  for (int c = 0; c < 3; ++c) {
    mul_dc[c] = dequant_matrices[DequantMatrixOffset(0, kQuantKindDCT8, c) *
                                 kBlockDim * kBlockDim] *
                quantizer.inv_quant_dc();
  }

  pass_dec_cache->dc = Image3F(xsize_blocks, ysize_blocks);

  // Precompute DC inverse color transform.
  float ytox_dc = ColorCorrelationMap::YtoX(1.0f, cmap.ytox_dc);
  float ytob_dc = ColorCorrelationMap::YtoB(1.0f, cmap.ytob_dc);

  const size_t xsize_groups = DivCeil(xsize_blocks, kDcGroupDimInBlocks);
  const size_t ysize_groups = DivCeil(ysize_blocks, kDcGroupDimInBlocks);
  const size_t num_groups = xsize_groups * ysize_groups;

  // Read TOC.
  std::vector<size_t> group_offsets;
  {
    group_offsets.reserve(num_groups + 1);
    group_offsets.push_back(0);
    for (size_t group_index = 0; group_index < num_groups; ++group_index) {
      const uint32_t size = DCGroupSizeCoder::Decode(reader);
      group_offsets.push_back(group_offsets.back() + size);
    }
    PIK_RETURN_IF_ERROR(reader->JumpToByteBoundary());
  }

  // Pretend all groups are read.
  size_t group_codes_begin = reader->Position();
  reader->SkipBits(group_offsets.back() * kBitsPerByte);
  if (reader->Position() > compressed.size()) {
    return PIK_FAILURE("Group code extends after stream end");
  }

  // Decode groups.
  std::atomic<int> num_errors{0};
  const auto process_group = [&](const int group_index, const int thread) {
    size_t group_code_offset = group_offsets[group_index];
    size_t group_reader_limit = group_offsets[group_index + 1];
    // TODO(user): this looks ugly; we should get rid of PaddedBytes parameter
    //               once it is wrapped into BitReader; otherwise it is easy to
    //               screw the things up.
    BitReader group_reader(compressed.data(),
                           group_codes_begin + group_reader_limit);
    group_reader.SkipBits((group_codes_begin + group_code_offset) *
                          kBitsPerByte);
    const size_t gx = group_index % xsize_groups;
    const size_t gy = group_index / xsize_groups;
    const Rect rect(gx * kDcGroupDimInBlocks, gy * kDcGroupDimInBlocks,
                    kDcGroupDimInBlocks, kDcGroupDimInBlocks, xsize_blocks,
                    ysize_blocks);
    if (!DecodeDCGroup(&group_reader, compressed, rect,
                       pass_dec_cache->use_new_dc, pass_dec_cache->grayscale,
                       mul_dc, ytox_dc, ytob_dc, pass_dec_cache)) {
      num_errors.fetch_add(1);
      return;
    }
  };
  RunOnPool(pool, 0, num_groups, process_group, "DecodeDC");

  PIK_RETURN_IF_ERROR(num_errors.load(std::memory_order_relaxed) == 0);

  if (pass_header.flags & PassHeader::kGradientMap) {
    size_t byte_pos = reader->Position();
    PIK_RETURN_IF_ERROR(DeserializeGradientMap(
        xsize_blocks, ysize_blocks,
        pass_header.flags & PassHeader::kGrayscaleOpt, quantizer, compressed,
        &byte_pos, &pass_dec_cache->gradient));
    reader->SkipBits((byte_pos - reader->Position()) * 8);
    ApplyGradientMap(pass_dec_cache->gradient, quantizer, &pass_dec_cache->dc);
  }
  return true;
}

void InitializeDecCache(const PassDecCache& pass_dec_cache, const Rect& rect,
                        DecCache* dec_cache) {
  const size_t full_xsize_blocks = pass_dec_cache.dc.xsize();
  const size_t full_ysize_blocks = pass_dec_cache.dc.ysize();
  const size_t x0_blocks = rect.x0() / kBlockDim;
  const size_t y0_blocks = rect.y0() / kBlockDim;
  const size_t xsize_blocks = rect.xsize() / kBlockDim;
  const size_t ysize_blocks = rect.ysize() / kBlockDim;

  // TODO(veluca): avoid this copy.
  dec_cache->dc = Image3F(xsize_blocks + 2, ysize_blocks + 2);
  for (size_t c = 0; c < 3; c++) {
    for (size_t y = 0; y < ysize_blocks + 2; y++) {
      const size_t y_src = SourceCoord(y + y0_blocks, full_ysize_blocks);
      const float* row_src = pass_dec_cache.dc.ConstPlaneRow(c, y_src);
      float* row_dc = dec_cache->dc.PlaneRow(c, y);
      for (size_t x = 0; x < xsize_blocks + 2; x++) {
        const size_t x_src = SourceCoord(x + x0_blocks, full_xsize_blocks);
        row_dc[x] = row_src[x_src];
      }
    }
  }
}

}  // namespace pik
