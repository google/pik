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
#include "common.h"
#include "compiler_specific.h"
#include "compressed_image.h"
#include "dct_util.h"
#include "fast_log.h"
#include "header.h"
#include "image_io.h"
#include "noise.h"
#include "opsin_codec.h"
#include "opsin_image.h"
#include "opsin_inverse.h"
#include "pik_alpha.h"
#include "quantizer.h"

bool FLAGS_log_search_state = false;
// If true, prints the quantization maps at each iteration.
bool FLAGS_dump_quant_state = false;

namespace pik {
namespace {

const float kYToBScale = 128.0f;

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

template <class Image>
size_t TargetSize(const CompressParams& params, const Image& img) {
  if (params.target_size > 0) {
    return params.target_size;
  }
  if (params.target_bitrate > 0.0) {
    return 0.5 + params.target_bitrate * img.xsize() * img.ysize() / 8;
  }
  return 0;
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
                          const CompressParams& cparams,
                          float butteraugli_target,
                          const Image<int>& ytob_map,
                          const int ytob_dc,
                          Quantizer* quantizer,
                          PikInfo* aux_out) {
  ButteraugliComparator comparator(opsin_orig);
  const float kInitialQuantDC = 0.8f / butteraugli_target;
  const float kInitialQuantAC = 0.5625f / butteraugli_target;
  const int block_xsize = opsin.xsize() / 8;
  const int block_ysize = opsin.ysize() / 8;
  ImageF quant_field(block_xsize, block_ysize, kInitialQuantAC);
  ImageF best_quant_field = CopyImage(quant_field);
  float best_butteraugli = 1000.0f;
  ImageF tile_distmap;
  static const int kMaxOuterIters = 2;
  int outer_iter = 0;
  int butteraugli_iter = 0;
  int search_radius = 0;
  float quant_ceil = 3.0f;
  int num_stalling_iters = 0;
  CompressParams search_params = cparams;
  for (;;) {
    if (FLAGS_dump_quant_state) {
      printf("\nQuantization field:\n");
      for (int y = 0; y < quant_field.ysize(); ++y) {
        for (int x = 0; x < quant_field.xsize(); ++x) {
          printf(" %.5f", quant_field.Row(y)[x]);
        }
        printf("\n");
      }
    }
    float qmin, qmax;
    ImageMinMax(quant_field, &qmin, &qmax);
    if (quantizer->SetQuantField(kInitialQuantDC, quant_field)) {
      QuantizedCoeffs qcoeffs = ComputeCoefficients(
          search_params, opsin, *quantizer);
      Image3F recon = ReconOpsinImage(qcoeffs, *quantizer);
      YToBTransform(ytob_map, ytob_dc, 1.0f / kYToBScale, &recon);
      Image3B srgb;
      CenteredOpsinToSrgb(recon, &srgb);
      comparator.Compare(srgb);
      bool best_quant_updated = false;
      if (comparator.distance() <= best_butteraugli) {
        best_quant_field = CopyImage(quant_field);
        best_butteraugli = std::max(comparator.distance(), butteraugli_target);
        best_quant_updated = true;
        num_stalling_iters = 0;
      } else if (outer_iter == 0) {
        ++num_stalling_iters;
      }
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
      if (FLAGS_log_search_state) {
        float minval, maxval;
        ImageMinMax(quant_field, &minval, &maxval);
        printf("\nButteraugli iter: %d/%d%s\n", butteraugli_iter,
               cparams.max_butteraugli_iters,
               best_quant_updated ? " (*)" : "");
        printf("Butteraugli distance: %f\n", comparator.distance());
        printf("quant range: %f ... %f  DC quant: %f\n", minval, maxval,
               kInitialQuantDC);
        printf("quant_ceil: %f\n", quant_ceil);
        printf("search_radius: %d\n", search_radius);
        printf("outer iters: %d\n", outer_iter);
        if (FLAGS_dump_quant_state) {
          quantizer->DumpQuantizationMap();
        }
      }
      if (butteraugli_iter >= cparams.max_butteraugli_iters) {
        break;
      }
    }
    bool changed = false;
    while (!changed && comparator.distance() > butteraugli_target) {
      for (int radius = 0; radius <= search_radius && !changed; ++radius) {
        ImageF dist_to_peak_map = DistToPeakMap(
            tile_distmap, butteraugli_target, radius, 0.0);
        for (int y = 0; y < quant_field.ysize(); ++y) {
          float* const PIK_RESTRICT row_q = quant_field.Row(y);
          const float* const PIK_RESTRICT row_dist = dist_to_peak_map.Row(y);
          for (int x = 0; x < quant_field.xsize(); ++x) {
            if (row_dist[x] >= 0.0f) {
              static const float kAdjSpeed[kMaxOuterIters] = { 0.1, 0.05 };
              const float factor =
                  kAdjSpeed[outer_iter] * tile_distmap.Row(y)[x];
              if (AdjustQuantVal(&row_q[x], row_dist[x], factor, quant_ceil)) {
                changed = true;
              }
            }
          }
        }
      }
      if (!changed || num_stalling_iters >= 3) {
        // Try to extend the search parameters.
        if (search_radius < 4) {
          if (qmax < 0.99f * quant_ceil ||
              quant_ceil >= 3.0f + search_radius) {
            ++search_radius;
          } else {
            quant_ceil += 0.5f;
          }
        } else if (quant_ceil < 8.0f) {
          quant_ceil += 0.5f;
        } else {
          break;
        }
      }
    }
    if (!changed) {
      if (++outer_iter == kMaxOuterIters) break;
      static const float kQuantScale[kMaxOuterIters] = { 0.0, 0.8 };
      for (int y = 0; y < quant_field.ysize(); ++y) {
        for (int x = 0; x < quant_field.xsize(); ++x) {
          quant_field.Row(y)[x] *= kQuantScale[outer_iter];
        }
      }
      num_stalling_iters = 0;
    }
  }
  quantizer->SetQuantField(kInitialQuantDC, best_quant_field);
}

template <typename T>
inline size_t IndexOfMaximum(const T* array, const size_t len) {
  PIK_ASSERT(len > 0);
  T maxval = array[0];
  size_t maxidx = 0;
  for (size_t i = 1; i < len; ++i) {
    if (array[i] > maxval) {
      maxval = array[i];
      maxidx = i;
    }
  }
  return maxidx;
}

void FindBestYToBCorrelation(const Image3F& opsin,
                             const Quantizer& quantizer,
                             Image<int>* ytob_map,
                             int* ytob_dc) {
  const float kZeroThresh = kYToBScale * 0.7f;
  const Image3F& qscales = quantizer.scale();
  Image3F dct = TransposedScaledDCT(opsin);
  size_t num_zeros[256] = { 0 };
  for (size_t y = 0; y < dct.ysize(); ++y) {
    const float* const PIK_RESTRICT row_y = dct.ConstPlaneRow(1, y);
    const float* const PIK_RESTRICT row_b = dct.ConstPlaneRow(2, y);
    const float* const PIK_RESTRICT row_q = qscales.ConstPlaneRow(2, y);
    for (size_t x = 0; x < dct.xsize(); ++x) {
      if (x % 64 == 0) continue;
      const float scaled_b = kYToBScale * row_b[x] * row_q[x];
      const float scaled_y = row_y[x] * row_q[x];
      for (int ytob = 0; ytob < 256; ++ytob) {
        if (std::abs(scaled_b - ytob * scaled_y) < kZeroThresh) {
          ++num_zeros[ytob];
        }
      }
    }
  }
  *ytob_dc = IndexOfMaximum(num_zeros, 256);
  for (int tile_y = 0; tile_y < ytob_map->ysize(); ++tile_y) {
    for (int tile_x = 0; tile_x < ytob_map->xsize(); ++tile_x) {
      const int y0 = tile_y * kTileInBlocks;
      const int x0 = tile_x * kTileInBlocks * 64;
      const int y1 = std::min<int>(y0 + kTileInBlocks, dct.ysize());
      const int x1 = std::min<int>(x0 + kTileInBlocks * 64, dct.xsize());
      size_t num_zeros[256] = { 0 };
      for (size_t y = y0; y < y1; ++y) {
        const float* const PIK_RESTRICT row_y = dct.ConstPlaneRow(1, y);
        const float* const PIK_RESTRICT row_b = dct.ConstPlaneRow(2, y);
        const float* const PIK_RESTRICT row_q = qscales.ConstPlaneRow(2, y);
        for (size_t x = x0; x < x1; ++x) {
          if (x % 64 == 0) continue;
          const float scaled_b = kYToBScale * row_b[x] * row_q[x];
          const float scaled_y = row_y[x] * row_q[x];
          for (int ytob = 0; ytob < 256; ++ytob) {
            if (std::abs(scaled_b - ytob * scaled_y) < kZeroThresh) {
              ++num_zeros[ytob];
            }
          }
        }
      }
      int best_ytob = IndexOfMaximum(num_zeros, 256);
      // Revert to the global factor used for dc if the number of zeros is
      // not much different.
      if (num_zeros[best_ytob] - num_zeros[*ytob_dc] <= 10) {
        best_ytob = *ytob_dc;
      }
      ytob_map->Row(tile_y)[tile_x] = best_ytob;
    }
  }
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

void ScaleToTargetSize(const Image3F& opsin,
                       const CompressParams& cparams,
                       const NoiseParams& noise_params,
                       size_t target_size,
                       const Image<int>& ytob_map,
                       const int ytob_dc,
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
    QuantizedCoeffs qcoeffs = ComputeCoefficients(cparams, opsin, *quantizer);
    candidate = EncodeToBitstream(qcoeffs, *quantizer, noise_params, ytob_map,
                                  ytob_dc, false, nullptr);
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
    QuantizedCoeffs qcoeffs = ComputeCoefficients(cparams, opsin, *quantizer);
    candidate = EncodeToBitstream(qcoeffs, *quantizer, noise_params, ytob_map,
                                  ytob_dc, false, nullptr);
    if (candidate.size() <= target_size) {
      scale_good = scale;
    } else {
      scale_bad = scale;
    }
  }
  ScaleQuantizationMap(quant_dc, quant_ac, scale_good, quantizer);
}

void CompressToTargetSize(const Image3F& opsin_orig,
                          const Image3F& opsin,
                          const CompressParams& cparams,
                          const NoiseParams& noise_params,
                          size_t target_size,
                          const Image<int>& ytob_map,
                          const int ytob_dc,
                          Quantizer* quantizer,
                          PikInfo* aux_out) {
  float dist_bad = 1.0;
  float dist_good = 1.0;
  bool found_candidate = false;
  std::string candidate;
  for (int i = 0; i < 5; ++i) {
    FindBestQuantization(opsin_orig, opsin, cparams, dist_good,
                         ytob_map, ytob_dc, quantizer, aux_out);
    QuantizedCoeffs qcoeffs = ComputeCoefficients(cparams, opsin, *quantizer);
    candidate = EncodeToBitstream(qcoeffs, *quantizer, noise_params, ytob_map,
                                  ytob_dc, false, nullptr);
    if (candidate.size() <= target_size) {
      found_candidate = true;
      break;
    }
    dist_bad = dist_good;
    dist_good *= 2.0;
  }
  if (!found_candidate) {
    // We could not make the compressed size small enough, return the smallest
    // we have managed.
    return;
  }
  if (dist_good == 1.0) {
    // We dont want to go below butteraugli distance 1.0, return the 1.0 image
    // even if it's smaller.
    return;
  }
  // TODO(user) Optimize this iteration so that we use the quantizer
  // corresponding to dist_bad in FindBestQuantization.
  Quantizer quantizer_good(opsin.xsize() / 8, opsin.ysize() / 8);
  quantizer_good.Copy(*quantizer);
  const float kIntervalLenThresh = 0.05f;
  while (dist_good - dist_bad > kIntervalLenThresh) {
    float dist = 0.5f * (dist_bad + dist_good);
    FindBestQuantization(opsin_orig, opsin, cparams, dist,
                         ytob_map, ytob_dc, quantizer, aux_out);
    QuantizedCoeffs qcoeffs = ComputeCoefficients(cparams, opsin, *quantizer);
    candidate = EncodeToBitstream(qcoeffs, *quantizer, noise_params, ytob_map,
                                  ytob_dc, false, nullptr);
    if (candidate.size() <= target_size) {
      dist_good = dist;
      quantizer_good.Copy(*quantizer);
    } else {
      dist_bad = dist;
    }
  }
  quantizer->Copy(quantizer_good);
}




}  // namespace


template<typename T>
MetaImageF OpsinDynamicsMetaImage(const Image3<T>& image) {
  Image3F opsin = OpsinDynamicsImage(image);
  MetaImageF out;
  out.SetColor(std::move(opsin));
  return out;
}

template<typename T>
MetaImageF OpsinDynamicsMetaImage(const MetaImage<T>& image) {
  MetaImageF out = OpsinDynamicsMetaImage(image.GetColor());
  out.CopyAlpha(image);
  return out;
}

template<typename Image>
bool PixelsToPikT(const CompressParams& params_in, const Image& image,
                  PaddedBytes* compressed, PikInfo* aux_out) {
  if (image.xsize() == 0 || image.ysize() == 0) {
    return PIK_FAILURE("Empty image");
  }
  MetaImageF opsin = OpsinDynamicsMetaImage(image);
  Header header;
  header.xsize = image.xsize();
  header.ysize = image.ysize();
  if (opsin.HasAlpha()) {
    header.flags |= Header::kAlpha;
  }

  compressed->resize(MaxCompressedHeaderSize());
  uint8_t* header_end = StoreHeader(header, compressed->data());
  if (header_end == nullptr) return false;
  const size_t header_size = header_end - compressed->data();
  compressed->resize(header_size);  // no copy!
  if (aux_out) {
    aux_out->layers[kLayerHeader].total_size += header_size;
  }

  if (opsin.HasAlpha()) {
    size_t bytepos = compressed->size();
    if (!AlphaToPik(params_in, opsin.GetAlpha(), opsin.AlphaBitDepth(),
                    &bytepos, compressed)) {
      return false;
    }
    if (aux_out) {
      aux_out->layers[kLayerAlpha].total_size +=
          compressed->size() - header_size;
    }
  }

  CompressParams params = params_in;
  size_t target_size = TargetSize(params, image);
  size_t opsin_target_size =
      (compressed->size() < target_size ? target_size - compressed->size() : 1);
  if (params.target_size > 0 || params.target_bitrate > 0.0) {
    params.target_size = opsin_target_size;
  }
  if (!OpsinToPik(params, opsin, compressed, aux_out)) {
    return false;
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

// Blends the median, and the middle value, by t. (0 -> median, 1 -> middle).
float Denoise(float* begin, float* end, float t) {
  const size_t n = end - begin;
  const float src = begin[n / 2];
  std::nth_element(begin, begin + n / 2, end);
  const float med = begin[n / 2];
  return t * src + (1.0f - t) * med;
}

// Applies a median filter on a disk neigbourhood of radius 6 to reduce
// noise.
Image3F ReduceNoise(const Image3F& opsin_orig) {
  ImageF variance(opsin_orig.xsize(), opsin_orig.ysize());
  Image3F opsin(opsin_orig.xsize(), opsin_orig.ysize());
  // Offsets can be generated as follows:
  // <<END | ghci
  // let points = [(x, y) | y <- [-6..6], x <- [-6..6], x^2 + y^2 <= 6^2]
  // fmap fst points
  // fmap snd points
  // END
  const int kOffX[] = {
      0,  -3, -2, -1, 0,  1,  2,  3,  -4, -3, -2, -1, 0,  1,  2,  3,  4,
      -5, -4, -3, -2, -1, 0,  1,  2,  3,  4,  5,  -5, -4, -3, -2, -1, 0,
      1,  2,  3,  4,  5,  -5, -4, -3, -2, -1, 0,  1,  2,  3,  4,  5,  -6,
      -5, -4, -3, -2, -1, 0,  1,  2,  3,  4,  5,  6,  -5, -4, -3, -2, -1,
      0,  1,  2,  3,  4,  5,  -5, -4, -3, -2, -1, 0,  1,  2,  3,  4,  5,
      -5, -4, -3, -2, -1, 0,  1,  2,  3,  4,  5,  -4, -3, -2, -1, 0,  1,
      2,  3,  4,  -3, -2, -1, 0,  1,  2,  3,  0};
  const int kOffY[] = {
      -6, -5, -5, -5, -5, -5, -5, -5, -4, -4, -4, -4, -4, -4, -4, -4, -4,
      -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -2, -2, -2, -2, -2, -2,
      -2, -2, -2, -2, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,
      1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
      3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  4,  4,  4,  4,  4,  4,
      4,  4,  4,  5,  5,  5,  5,  5,  5,  5,  6};

  const size_t kSamples = sizeof(kOffY) / sizeof(int);
  const float kInvN = 1.0f / static_cast<float>(kSamples);
  float values[3][kSamples] = {{0.0f}, {0.0f}, {0.0f}};

  // First determine the variance of the Y channel of a neighborhood. This is
  // used to determine whether pixels are part of a smooth region (low variance)
  // where noise reduction can be applied safely.
  for (int y = 0; y < opsin_orig.ysize(); y++) {
    for (int x = 0; x < opsin_orig.xsize(); x++) {
      float sum = 0.0f;
      float sum2 = 0.0f;
      for (size_t k = 0; k < kSamples; k++) {
        size_t xo = Clamp(0, opsin_orig.xsize() - 1, x + kOffX[k]);
        size_t yo = Clamp(0, opsin_orig.ysize() - 1, y + kOffY[k]);
        float val = opsin_orig.Row(yo)[1][xo];
        sum += val;
        sum2 += val * val;
      }
      variance.Row(y)[x] = sum2 * kInvN - sum * sum * kInvN * kInvN;
    }
  }

  for (int y = 0; y < opsin_orig.ysize(); y++) {
    for (int x = 0; x < opsin_orig.xsize(); x++) {
      float var = 0.0f;
      for (size_t k = 0; k < kSamples; k++) {
        size_t xo = Clamp(0, opsin_orig.xsize() - 1, x + kOffX[k]);
        size_t yo = Clamp(0, opsin_orig.ysize() - 1, y + kOffY[k]);
        for (int ch = 0; ch < 3; ch++) {
          values[ch][k] = opsin_orig.Row(yo)[ch][xo];
        }
        var += variance.Row(yo)[xo];
      }
      // The 1e3 here scales the variance to the range where [0, 1] can
      // meaningfully distinguish between smooth areas and areas with features.
      // Increase this factor denoise more conservatively, increase to smooth
      // more.
      const float t = std::min(1.0f, var * kInvN * 1e3f);
      for (int ch = 0; ch < 3; ch++) {
        opsin.Row(y)[ch][x] =
            Denoise(std::begin(values[ch]), std::end(values[ch]), t);
      }
    }
  }

  return opsin;
}

bool OpsinToPik(const CompressParams& params, const MetaImageF& opsin_orig,
                PaddedBytes* compressed, PikInfo* aux_out) {
  if (opsin_orig.xsize() == 0 || opsin_orig.ysize() == 0) {
    return PIK_FAILURE("Empty image");
  }
  const size_t xsize = opsin_orig.xsize();
  const size_t ysize = opsin_orig.ysize();
  const size_t block_xsize = DivCeil(xsize, 8);
  const size_t block_ysize = DivCeil(ysize, 8);
  const size_t tile_xsize = DivCeil(xsize, kTileSize);
  const size_t tile_ysize = DivCeil(ysize, kTileSize);
  Image3F opsin = AlignImage(opsin_orig.GetColor(), 8);
  CenterOpsinValues(&opsin);
  NoiseParams noise_params;
  if (params.apply_noise) {
    const float kPatchPercent = 0.2;
    GetNoiseParameter(opsin, kPatchPercent, &noise_params);
  }

  Quantizer quantizer(block_xsize, block_ysize);
  quantizer.SetQuant(1.0f);
  int ytob_dc = 120;
  Image<int> ytob_map(tile_xsize, tile_ysize, 120);
  if (params.butteraugli_distance >= 0.0 || params.target_bitrate > 0.0 ||
      params.target_size > 0) {
    FindBestYToBCorrelation(opsin, quantizer, &ytob_map, &ytob_dc);
  }
  YToBTransform(ytob_map, ytob_dc, -1.0f / kYToBScale, &opsin);
  if (params.butteraugli_distance >= 0.0) {
    FindBestQuantization(opsin_orig.GetColor(), opsin, params,
                         params.butteraugli_distance, ytob_map, ytob_dc,
                         &quantizer, aux_out);
  } else if (params.target_size > 0 || params.target_bitrate > 0.0) {
    size_t target_size = TargetSize(params, opsin);
    if (params.target_size_search_fast_mode) {
      FindBestQuantization(opsin_orig.GetColor(), opsin, params, 1.0,
                           ytob_map, ytob_dc, &quantizer, aux_out);
      ScaleToTargetSize(opsin, params, noise_params, target_size, ytob_map,
                        ytob_dc, &quantizer, aux_out);
    } else {
      CompressToTargetSize(opsin_orig.GetColor(), opsin, params, noise_params,
                           target_size, ytob_map, ytob_dc, &quantizer, aux_out);
    }
  } else if (params.uniform_quant > 0.0) {
    quantizer.SetQuant(params.uniform_quant);
  } else if (params.fast_mode) {
    const float kQuantDC = 0.76953163840390082;
    const float kQuantAC = 1.52005680264295;
    ImageF qf = AdaptiveQuantizationMap(opsin_orig.GetColor().plane(1), 8);
    quantizer.SetQuantField(kQuantDC, ScaleImage(kQuantAC, qf));
  }

  if (params.denoise) {
    opsin = ReduceNoise(opsin);
  }

  QuantizedCoeffs qcoeffs = ComputeCoefficients(params, opsin, quantizer);
  std::string compressed_data = EncodeToBitstream(
      qcoeffs, quantizer, noise_params, ytob_map, ytob_dc,
      params.fast_mode, aux_out);

  size_t old_size = compressed->size();
  compressed->resize(compressed->size() + compressed_data.size());
  memcpy(compressed->data() + old_size, compressed_data.data(),
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
  PIK_ASSERT(byte_pos <= compressed.size());

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

    ImageU alpha(header.xsize, header.ysize);
    int alpha_bit_depth = 0;
    if (header.flags & Header::kAlpha) {
      size_t bytes_read;
      if (!PikToAlpha(params, byte_pos, compressed, &bytes_read,
                      &alpha_bit_depth, &alpha)) {
        return false;
      }
      byte_pos += bytes_read;
      PIK_ASSERT(byte_pos <= compressed.size());
    }

    int block_xsize = (header.xsize + 7) / 8;
    int block_ysize = (header.ysize + 7) / 8;
    Quantizer quantizer(block_xsize, block_ysize);
    QuantizedCoeffs qcoeffs;
    NoiseParams noise_params;
    Image<int> ytob_map;
    int ytob_dc;
    size_t bytes_read;
    if (!DecodeFromBitstream(compressed.data() + byte_pos,
                             compressed.size() - byte_pos,
                             header.xsize, header.ysize,
                             &ytob_map, &ytob_dc,
                             &noise_params, &quantizer, &qcoeffs,
                             &bytes_read)) {
      return PIK_FAILURE("Pik decoding failed.");
    }
    byte_pos += bytes_read;
    PIK_ASSERT(byte_pos <= compressed.size());
    Image3F opsin = ReconOpsinImage(qcoeffs, quantizer);
    YToBTransform(ytob_map, ytob_dc, 1.0f / kYToBScale, &opsin);
    AddNoise(noise_params, &opsin);
    CenteredOpsinToSrgb(opsin, &planes);
    planes.ShrinkTo(header.xsize, header.ysize);
    image->SetColor(std::move(planes));
    if (alpha_bit_depth > 0) {
      image->SetAlpha(std::move(alpha), alpha_bit_depth);
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
