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

#include "pik.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "adaptive_quantization.h"
#include "arch_specific.h"
#include "bit_buffer.h"
#include "bits.h"
#include "byte_order.h"
#include "butteraugli_comparator.h"
#include "compiler_specific.h"
#include "compressed_image.h"
#include "header.h"
#include "image_io.h"
#include "opsin_image.h"
#include "opsin_inverse.h"
#include "pik_alpha.h"
#include "quantizer.h"

// If true, prints the quantization maps at each iteration.
bool FLAGS_dump_quant_state = false;

namespace pik {
namespace {

void EncodeU32(const uint32_t val, uint8_t* dest) {
#if PIK_BYTE_ORDER_LITTLE
  memcpy(dest, &val, sizeof(val));
#else
  dest[0] = (val >>  0) & 0xff;
  dest[1] = (val >>  8) & 0xff;
  dest[2] = (val >> 16) & 0xff;
  dest[3] = (val >> 24) & 0xff;
#endif
}

uint32_t DecodeU32(const uint8_t* src) {
  uint32_t val;
#if PIK_BYTE_ORDER_LITTLE
  memcpy(&val, src, sizeof(val));
#else
  val = src[3];
  val <<= 8;
  val += src[2];
  val <<= 8;
  val += src[1];
  val <<= 8;
  val += src[0];
#endif
  return val;
}

inline int Clamp(int minval, int maxval, int val) {
  return std::min(maxval, std::max(minval, val));
}

ImageF TileDistMap(const butteraugli::ImageF& distmap, int tile_size) {
  const int tile_xsize = (distmap.xsize() + tile_size - 1) / tile_size;
  const int tile_ysize = (distmap.ysize() + tile_size - 1) / tile_size;
  ImageF tile_distmap(tile_xsize, tile_ysize);
  for (int tile_y = 0; tile_y < tile_ysize; ++tile_y) {
    for (int tile_x = 0; tile_x < tile_xsize; ++tile_x) {
      int x_max = std::min<int>(distmap.xsize(), tile_size * (tile_x + 1));
      int y_max = std::min<int>(distmap.ysize(), tile_size * (tile_y + 1));
      float max_dist = 0.0;
      for (int y = tile_size * tile_y; y < y_max; ++y) {
        const float* const PIK_RESTRICT row = distmap.Row(y);
        for (int x = tile_size * tile_x; x < x_max; ++x) {
          max_dist = std::max(max_dist, row[x]);
        }
      }
      tile_distmap.Row(tile_y)[tile_x] = max_dist;
    }
  }
  return tile_distmap;
}

ImageF DistToPeakMap(const ImageF& field, float peak_min,
                     int local_radius, float peak_weight) {
  ImageF result(field.xsize(), field.ysize(), -1.0f);
  for (int y0 = 0; y0 < field.ysize(); ++y0) {
    for (int x0 = 0; x0 < field.xsize(); ++x0) {
      int x_min = std::max(0, x0 - local_radius);
      int y_min = std::max(0, y0 - local_radius);
      int x_max = std::min<int>(field.xsize(), x0 + 1 + local_radius);
      int y_max = std::min<int>(field.ysize(), y0 + 1 + local_radius);
      float local_max = peak_min;
      for (int y = y_min; y < y_max; ++y) {
        for (int x = x_min; x < x_max; ++x) {
          local_max = std::max(local_max, field.Row(y)[x]);
        }
      }
      if (field.Row(y0)[x0] >
          (1.0f - peak_weight) * peak_min + peak_weight * local_max) {
        for (int y = y_min; y < y_max; ++y) {
          for (int x = x_min; x < x_max; ++x) {
            float dist = std::max(std::abs(y - y0), std::abs(x - x0));
            float cur_dist = result.Row(y)[x];
            if (cur_dist < 0.0 || cur_dist > dist) {
              result.Row(y)[x] = dist;
            }
          }
        }
      }
    }
  }
  return result;
}

bool AdjustQuantVal(float* const PIK_RESTRICT q,
                    const float d, const float factor,
                    const float quant_max) {
  if (*q >= 0.999f * quant_max) return false;
  const float inv_q = 1.0f / *q;
  const float adj_inv_q = inv_q - factor / (d + 1.0f);
  *q = 1.0f / std::max(1.0f / quant_max, adj_inv_q);
  return true;
}

void DumpHeatmap(const PikInfo* info, const std::string& label,
                 const std::vector<float>& vals, size_t xsize, size_t ysize,
                 float good_threshold, float bad_threshold) {
  std::vector<uint8_t> heatmap(3 * xsize * ysize);
  butteraugli::CreateHeatMapImage(vals, good_threshold, bad_threshold, xsize,
                                  ysize, &heatmap);
  char pathname[200];
  snprintf(pathname, sizeof(pathname), "%s%s%05d.png",
           info->debug_prefix.c_str(), label.c_str(),
           info->num_butteraugli_iters);
  WriteImage(ImageFormatPNG(),
             Image3FromInterleaved(&heatmap[0], xsize, ysize, 3 * xsize),
             pathname);
}

void DumpHeatmaps(const PikInfo* info,
                  size_t xsize, size_t ysize, int qres,
                  float ba_target,
                  const ImageF& quant_field,
                  const ImageF& tile_heatmap) {
  if (info->debug_prefix.empty()) return;
  std::vector<float> qmap(xsize * ysize);
  std::vector<float> dmap(xsize * ysize);
  for (int y = 0; y < quant_field.ysize(); ++y) {
    auto row_q = quant_field.Row(y);
    auto row_d = tile_heatmap.Row(y);
    for (int x = 0; x < quant_field.xsize(); ++x) {
      for (int dy = 0; dy < qres; ++dy) {
        for (int dx = 0; dx < qres; ++dx) {
          int px = qres * x + dx;
          int py = qres * y + dy;
          if (px < xsize && py < ysize) {
            qmap[py * xsize + px] = 1.0f / row_q[x];  // never zero
            dmap[py * xsize + px] = row_d[x];
          }
        }
      }
    }
  }
  DumpHeatmap(info, "quant_heatmap", qmap, xsize, ysize,
              4.0f * ba_target, 6.0f * ba_target);
  DumpHeatmap(info, "tile_heatmap", dmap, xsize, ysize,
              ba_target, 1.5f * ba_target);
}

void FindBestQuantization(const Image3F& opsin_orig,
                          const Image3F& opsin,
                          float butteraugli_target,
                          int max_butteraugli_iters,
                          int ytob,
                          Quantizer* quantizer,
                          PikInfo* aux_out) {
  ButteraugliComparator comparator(opsin_orig);
  const float kInitialQuantDC = 1.0625f / butteraugli_target;
  const float kInitialQuantAC = 0.5625f / butteraugli_target;
  const int block_xsize = opsin.xsize() / 8;
  const int block_ysize = opsin.ysize() / 8;
  ImageF quant_field(block_xsize, block_ysize, kInitialQuantAC);
  ImageF tile_distmap;
  static const int kMaxOuterIters = 3;
  int outer_iter = 0;
  int butteraugli_iter = 0;
  float quant_max = 4.0f;
  for (;;) {
    if (FLAGS_dump_quant_state) {
      printf("\nQuantization field:\n");
      for (int y = 0; y < quant_field.ysize(); ++y) {
        for (int x = 0; x < quant_field.xsize(); ++x) {
          printf(" %.5f", quant_field.Row(y)[x]);
        }
        printf("\n");
      }
      printf("max_butteraugli_iters = %d\n", max_butteraugli_iters);
    }
    if (quantizer->SetQuantField(kInitialQuantDC, quant_field)) {
      if (butteraugli_iter >= max_butteraugli_iters) {
        break;
      }
      QuantizedCoeffs qcoeffs = ComputeCoefficients(opsin, *quantizer);
      Image3F recon = ReconOpsinImage(qcoeffs, *quantizer);
      YToBTransform(ytob / 128.0f, &recon);
      Image3B srgb;
      CenteredOpsinToSrgb(recon, &srgb);
      comparator.Compare(srgb);
      tile_distmap = TileDistMap(comparator.distmap(), 8);
      ++butteraugli_iter;
      if (aux_out) {
        DumpHeatmaps(aux_out, opsin_orig.xsize(), opsin_orig.ysize(),
                     8, butteraugli_target, quant_field, tile_distmap);
        if (!aux_out->debug_prefix.empty()) {
          char pathname[200];
          snprintf(pathname, 200, "%s%s%05d.png", aux_out->debug_prefix.c_str(),
                   "rgb_out", aux_out->num_butteraugli_iters);
          WriteImage(ImageFormatPNG(), srgb, pathname);
        }
        ++aux_out->num_butteraugli_iters;
      }
      if (FLAGS_dump_quant_state) {
        printf("\nButteraugli iter: %d\n", butteraugli_iter);
        printf("Butteraugli distance: %f\n", comparator.distance());
        printf("quant_max: %f\n", quant_max);
        quantizer->DumpQuantizationMap();
      }
    }
    bool changed = false;
    while (!changed && comparator.distance() > butteraugli_target) {
      for (int radius = 1; radius <= 4 && !changed; ++radius) {
        ImageF dist_to_peak_map = DistToPeakMap(
            tile_distmap, butteraugli_target, radius, 0.65);
        for (int y = 0; y < quant_field.ysize(); ++y) {
          float* const PIK_RESTRICT row_q = quant_field.Row(y);
          const float* const PIK_RESTRICT row_dist = dist_to_peak_map.Row(y);
          for (int x = 0; x < quant_field.xsize(); ++x) {
            if (row_dist[x] >= 0.0f) {
              static const float kAdjSpeed[kMaxOuterIters] =
                  { 0.1, 0.05, 0.025 };
              const float factor =
                  kAdjSpeed[outer_iter] * tile_distmap.Row(y)[x];
              if (AdjustQuantVal(&row_q[x], row_dist[x], factor, quant_max)) {
                changed = true;
              }
            }
          }
        }
      }
      if (quant_max >= 8.0f) break;
      if (!changed) quant_max += 0.5f;
    }
    if (!changed) {
      if (++outer_iter == kMaxOuterIters) break;
      static const float kQuantScale[kMaxOuterIters] = { 0.0, 0.8, 0.9 };
      for (int y = 0; y < quant_field.ysize(); ++y) {
        for (int x = 0; x < quant_field.xsize(); ++x) {
          quant_field.Row(y)[x] *= kQuantScale[outer_iter];
        }
      }
    }
  }
}

struct EvalGlobalYToB {
  size_t operator()(int ytob) const {
    Image3F copy = CopyImage3(opsin);
    YToBTransform(-ytob / 128.0f, &copy);
    QuantizedCoeffs qcoeffs = ComputeCoefficients(copy, quantizer);
    return EncodeToBitstream(qcoeffs, quantizer, ytob, true, nullptr).size();
  }
  const Image3F& opsin;
  const Quantizer& quantizer;
};

template <class Eval>
int Optimize(const Eval& eval, int minval, int maxval,
             int best_val, size_t* best_objval) {
  int start = minval;
  int end = maxval;
  for (int resolution = 16; resolution >= 1; resolution /= 4) {
    for (int val = start; val <= end; val += resolution) {
      size_t objval = eval(val);
      if (objval < *best_objval) {
        best_val = val;
        *best_objval = objval;
      }
    }
    start = std::max(minval, best_val - resolution + 1);
    end = std::min(maxval, best_val + resolution - 1);
  }
  return best_val;
}

int FindBestYToBCorrelation(const Image3F& opsin, const Quantizer& quantizer) {
  static const int kStartYToB = 120;
  EvalGlobalYToB eval_global{opsin, quantizer};
  size_t best_size = eval_global(kStartYToB);
  return Optimize(eval_global, 0, 255, kStartYToB, &best_size);
}

bool ScaleQuantizationMap(const float quant_dc,
                          const ImageF& quant_field_ac,
                          float scale,
                          Quantizer* quantizer) {
  float scale_dc = 0.8 * scale + 0.2;
  bool changed = quantizer->SetQuantField(
      scale_dc * quant_dc, ScaleImage(scale, quant_field_ac));
  if (FLAGS_dump_quant_state) {
    printf("\nScaling quantization map with scale %f\n", scale);
    quantizer->DumpQuantizationMap();
  }
  return changed;
}

void ScaleToTargetSize(const Image3F& opsin, size_t target_size,
                       int ytob,
                       Quantizer* quantizer,
                       PikInfo* aux_out) {
  float quant_dc;
  ImageF quant_ac;
  quantizer->GetQuantField(&quant_dc, &quant_ac);
  float scale_bad = 1.0;
  float scale_good = 1.0;
  bool found_candidate = false;
  std::string candidate;
  for (int i = 0; i < 10; ++i) {
    ScaleQuantizationMap(quant_dc, quant_ac, scale_good, quantizer);
    QuantizedCoeffs qcoeffs = ComputeCoefficients(opsin, *quantizer);
    candidate = EncodeToBitstream(qcoeffs, *quantizer, ytob, false, aux_out);
    if (candidate.size() <= target_size) {
      found_candidate = true;
      break;
    }
    scale_bad = scale_good;
    scale_good *= 0.5;
  }
  if (!found_candidate) {
    // We could not make the compressed size small enough
    return;
  }
  if (scale_good == 1.0) {
    // We dont want to go below butteraugli distance 1.0
    return;
  }
  for (int i = 0; i < 16; ++i) {
    float scale = 0.5 * (scale_bad + scale_good);
    if (!ScaleQuantizationMap(quant_dc, quant_ac, scale, quantizer)) {
      break;
    }
    QuantizedCoeffs qcoeffs = ComputeCoefficients(opsin, *quantizer);
    candidate = EncodeToBitstream(qcoeffs, *quantizer, ytob, false, aux_out);
    if (candidate.size() <= target_size) {
      scale_good = scale;
    } else {
      scale_bad = scale;
    }
  }
  ScaleQuantizationMap(quant_dc, quant_ac, scale_good, quantizer);
}




}  // namespace


template<typename T>
bool AlphaToPik(const CompressParams& params, const MetaImage<T>& image,
                 PaddedBytes* compressed, PikInfo* aux_out) {
  if (!image.HasAlpha()) {
    return PIK_FAILURE("Must have alpha if alpha_channel set");
  }
  size_t bytepos = compressed->size();
  if (!AlphaToPik(params, image.GetAlpha(), &bytepos, compressed)) {
    return false;
  }
  return true;
}

template<typename T>
bool AlphaToPik(const CompressParams& params, const Image3<T>& image,
                 PaddedBytes* compressed, PikInfo* aux_out) {
  return PIK_FAILURE("Alpha not supported for Image3");
}

template<typename T>
Image3F OpsinDynamicsImage(const MetaImage<T>& image) {
  return OpsinDynamicsImage(image.GetColor());
}

template<typename Image>
bool PixelsToPikT(const CompressParams& params, const Image& image,
                  PaddedBytes* compressed, PikInfo* aux_out) {
  if (image.xsize() == 0 || image.ysize() == 0) {
    return PIK_FAILURE("Empty image");
  }
  if (!OpsinToPik(params, OpsinDynamicsImage(image), compressed, aux_out)) {
    return false;
  }
  if (params.alpha_channel) {
    if (!AlphaToPik(params, image, compressed, aux_out)) {
      return false;
    }
  }
  return true;
}

bool PixelsToPik(const CompressParams& params, const Image3B& image,
                 PaddedBytes* compressed, PikInfo* aux_out) {
  return PixelsToPikT(params, image, compressed, aux_out);
}

bool PixelsToPik(const CompressParams& params, const Image3F& image,
                 PaddedBytes* compressed, PikInfo* aux_out) {
  return PixelsToPikT(params, image, compressed, aux_out);
}

bool PixelsToPik(const CompressParams& params, const MetaImageB& image,
                 PaddedBytes* compressed, PikInfo* aux_out) {
  return PixelsToPikT(params, image, compressed, aux_out);
}

bool PixelsToPik(const CompressParams& params, const MetaImageF& image,
                 PaddedBytes* compressed, PikInfo* aux_out) {
  return PixelsToPikT(params, image, compressed, aux_out);
}

bool OpsinToPik(const CompressParams& params, const Image3F& opsin_orig,
                PaddedBytes* compressed, PikInfo* aux_out) {
  if (opsin_orig.xsize() == 0 || opsin_orig.ysize() == 0) {
    return PIK_FAILURE("Empty image");
  }
  const size_t xsize = opsin_orig.xsize();
  const size_t ysize = opsin_orig.ysize();
  const size_t block_xsize = (xsize + 7) / 8;
  const size_t block_ysize = (ysize + 7) / 8;
  Image3F opsin = AlignImage(opsin_orig, 8);
  CenterOpsinValues(&opsin);
  Quantizer quantizer(block_xsize, block_ysize);
  quantizer.SetQuant(1.0f);
  int ytob = 120;
  if (params.butteraugli_distance >= 0.0 || params.target_bitrate > 0.0) {
    ytob = FindBestYToBCorrelation(opsin, quantizer);
  }
  YToBTransform(-ytob / 128.0f, &opsin);
  if (params.butteraugli_distance >= 0.0) {
    FindBestQuantization(opsin_orig, opsin, params.butteraugli_distance,
                         params.max_butteraugli_iters, ytob,
                         &quantizer, aux_out);
  } else if (params.target_bitrate > 0.0) {
    FindBestQuantization(opsin_orig, opsin, 1.0, params.max_butteraugli_iters,
                         ytob, &quantizer, aux_out);
    size_t target_size = xsize * ysize * params.target_bitrate / 8.0;
    ScaleToTargetSize(opsin, target_size, ytob, &quantizer, aux_out);
  } else if (params.uniform_quant > 0.0) {
    quantizer.SetQuant(params.uniform_quant);
  } else if (params.fast_mode) {
    const float kQuantDC = 0.76953163840390082;
    const float kQuantAC = 1.52005680264295;
    ImageF qf = AdaptiveQuantizationMap(opsin_orig.plane(1), 8);
    quantizer.SetQuantField(kQuantDC, ScaleImage(kQuantAC, qf));
  }
  QuantizedCoeffs qcoeffs = ComputeCoefficients(opsin, quantizer);
  std::string compressed_data = EncodeToBitstream(
      qcoeffs, quantizer, ytob, params.fast_mode, aux_out);

  Header header;
  header.xsize = xsize;
  header.ysize = ysize;
  if (params.alpha_channel) {
    header.flags |= Header::kAlpha;
  }
  compressed->resize(MaxCompressedHeaderSize() + compressed_data.size());
  uint8_t* header_end = StoreHeader(header, compressed->data());
  if (header_end == nullptr) return false;
  const size_t header_size = header_end - compressed->data();
  compressed->resize(header_size + compressed_data.size());  // no copy!
  memcpy(compressed->data() + header_size, compressed_data.data(),
         compressed_data.size());
  return true;
}


template <typename T>
bool PikToPixelsT(const DecompressParams& params, const PaddedBytes& compressed,
                  MetaImage<T>* image, PikInfo* aux_out) {
  if (compressed.size() == 0) {
    return PIK_FAILURE("Empty input.");
  }
  Image3<T> planes;
  const uint8_t* const compressed_end = compressed.data() + compressed.size();

  Header header;
  const uint8_t* header_end = LoadHeader(compressed.data(), &header);
  if (header_end == nullptr) return false;
  if (header_end > compressed_end) {
    return PIK_FAILURE("Truncated header.");
  }
  size_t byte_pos = header_end - compressed.data();

  if (header.flags & Header::kWebPLossless) {
    return PIK_FAILURE("Invalid format code");
  } else {  // Pik
    if (header.xsize == 0 || header.ysize == 0) {
      return PIK_FAILURE("Empty image.");
    }
    static const uint32_t kMaxWidth = (1 << 25) - 1;
    if (header.xsize > kMaxWidth) {
      return PIK_FAILURE("Image too wide.");
    }
    uint64_t num_pixels = static_cast<uint64_t>(header.xsize) * header.ysize;
    if (num_pixels > params.max_num_pixels) {
      return PIK_FAILURE("Image too big.");
    }
    int block_xsize = (header.xsize + 7) / 8;
    int block_ysize = (header.ysize + 7) / 8;
    Quantizer quantizer(block_xsize, block_ysize);
    QuantizedCoeffs qcoeffs;
    int ytob;
    size_t bytes_read;
    if (!DecodeFromBitstream(header_end, compressed.size() - byte_pos,
                             header.xsize, header.ysize,
                             &ytob, &quantizer, &qcoeffs, &bytes_read)) {
      return PIK_FAILURE("Pik decoding failed.");
    }
    byte_pos += bytes_read;
    Image3F opsin = ReconOpsinImage(qcoeffs, quantizer);
    YToBTransform(ytob / 128.0f, &opsin);
    CenteredOpsinToSrgb(opsin, &planes);
    planes.ShrinkTo(header.xsize, header.ysize);
    image->SetColor(std::move(planes));

    if (header.flags & Header::kAlpha) {
      image->AddAlpha();
      size_t bytes_read;
      if (!PikToAlpha(params, byte_pos, compressed, &bytes_read,
                      &image->GetAlpha())) {
        return false;
      }
      byte_pos += bytes_read;
    }
  }
  if (params.check_decompressed_size && byte_pos != compressed.size()) {
    return PIK_FAILURE("Pik compressed data size mismatch.");
  }
  if (aux_out != nullptr) {
    aux_out->decoded_size = byte_pos;
  }
  return true;
}

bool PikToPixels(const DecompressParams& params, const PaddedBytes& compressed,
                 MetaImageB* image, PikInfo* aux_out) {
  return PikToPixelsT(params, compressed, image, aux_out);
}

bool PikToPixels(const DecompressParams& params, const PaddedBytes& compressed,
                 MetaImageU* image, PikInfo* aux_out) {
  return PikToPixelsT(params, compressed, image, aux_out);
}

bool PikToPixels(const DecompressParams& params, const PaddedBytes& compressed,
                 MetaImageF* image, PikInfo* aux_out) {
  return PikToPixelsT(params, compressed, image, aux_out);
}

template<typename T>
bool PikToPixelsT(const DecompressParams& params, const PaddedBytes& compressed,
                 Image3<T>* image, PikInfo* aux_out) {
  MetaImage<T> temp;
  if (!PikToPixelsT(params, compressed, &temp, aux_out)) {
    return false;
  }
  if (temp.HasAlpha()) {
    return PIK_FAILURE("Unable to output alpha channel");
  }
  *image = std::move(temp.GetColor());
  return true;
}

bool PikToPixels(const DecompressParams& params,
                 const PaddedBytes& compressed,
                 Image3B* image, PikInfo* aux_out) {
  return PikToPixelsT(params, compressed, image, aux_out);
}
bool PikToPixels(const DecompressParams& params,
                 const PaddedBytes& compressed,
                 Image3U* image, PikInfo* aux_out) {
  return PikToPixelsT(params, compressed, image, aux_out);
}
bool PikToPixels(const DecompressParams& params,
                 const PaddedBytes& compressed,
                 Image3F* image, PikInfo* aux_out) {
  return PikToPixelsT(params, compressed, image, aux_out);
}

}  // namespace pik
