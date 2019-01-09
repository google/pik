// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik_pass.h"

#include <limits.h>  // PATH_MAX
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include "noise.h"

#include "Eigen/Dense"

#undef PROFILER_ENABLED
#define PROFILER_ENABLED 1
#include "ac_strategy.h"
#include "adaptive_quantization.h"
#include "alpha.h"
#include "arch_specific.h"
#include "bits.h"
#include "byte_order.h"
#include "color_correlation.h"
#include "color_encoding.h"
#include "common.h"
#include "compiler_specific.h"
#include "convolve.h"
#include "dct.h"
#include "dct_util.h"
#include "entropy_coder.h"
#include "external_image.h"
#include "fast_log.h"
#include "gaborish.h"
#include "gamma_correct.h"
#include "headers.h"
#include "image.h"
#include "image_io.h"
#include "lossless16.h"
#include "lossless8.h"
#include "opsin_image.h"
#include "opsin_inverse.h"
#include "profiler.h"
#include "resize.h"
#include "simd/targets.h"
#include "size_coder.h"

namespace pik {
namespace {

// For encoder.
uint32_t PassFlagsFromParams(const CompressParams& cparams,
                             const CodecInOut* io) {
  uint32_t flags = 0;

  const float dist = cparams.butteraugli_distance;

  // We don't add noise at low butteraugli distances because the original
  // noise is stored within the compressed image and adding noise makes things
  // worse.
  if (ApplyOverride(cparams.noise, dist >= kMinButteraugliForNoise)) {
    flags |= PassHeader::kNoise;
  }

  // TODO(user): broken for now, to be fixed.
  /*if (ApplyOverride(cparams.gradient, dist >= kMinButteraugliForGradient)) {
    flags |= PassHeader::kGradientMap;
  }*/

  if (io->IsGray()) {
    flags |= PassHeader::kGrayscaleOpt;
  }

  return flags;
}

void OverrideFlag(const Override o, const uint32_t flag,
                  uint32_t* PIK_RESTRICT flags) {
  if (o == Override::kOn) {
    *flags |= flag;
  } else if (o == Override::kOff) {
    *flags &= ~flag;
  }
}

void OverridePassFlags(const DecompressParams& dparams,
                       PassHeader* PIK_RESTRICT pass_header) {
  // Do not forcibly enable features in intermediate passes.
  if (!pass_header->is_last) return;

  OverrideFlag(dparams.noise, PassHeader::kNoise, &pass_header->flags);
  OverrideFlag(dparams.gradient, PassHeader::kGradientMap, &pass_header->flags);

  if (dparams.adaptive_reconstruction == Override::kOff) {
    pass_header->have_adaptive_reconstruction = false;
  } else if (dparams.adaptive_reconstruction == Override::kOn) {
    pass_header->have_adaptive_reconstruction = true;
    pass_header->epf_params = dparams.epf_params;
  }

  if (dparams.override_gaborish) {
    pass_header->gaborish = dparams.gaborish;
  }
}

void OverrideGroupFlags(const DecompressParams& dparams,
                        const PassHeader* PIK_RESTRICT pass_header,
                        GroupHeader* PIK_RESTRICT header) {
  // Do not forcibly enable features in intermediate passes.
  if (!pass_header->is_last) return;
}

// Specializes a 8-bit and 16-bit of rounding from floating point to lossless.
template <typename T>
T RoundForLossless(float in);

template <>
uint8_t RoundForLossless(float in) {
  return static_cast<uint8_t>(in + 0.5f);
}

template <>
uint16_t RoundForLossless(float in) {
  return static_cast<uint16_t>(in * 257.0f + 0.5f);
}

// Specializes a 8-bit and 16-bit lossless diff for previous pass.
template <typename T>
T DiffForLossless(float in, float prev);

template <>
uint8_t DiffForLossless(float in, float prev) {
  uint8_t diff = static_cast<int>(RoundForLossless<uint8_t>(in)) -
                 static_cast<int>(RoundForLossless<uint8_t>(prev));
  if (diff > 127)
    diff = (255 - diff) * 2 + 1;
  else
    diff = diff * 2;
  return diff;
}

template <>
uint16_t DiffForLossless(float in, float prev) {
  uint32_t diff = 0xFFFF & (static_cast<int>(RoundForLossless<uint16_t>(in)) -
                            static_cast<int>(RoundForLossless<uint16_t>(prev)));
  if (diff > 32767)
    diff = (65535 - diff) * 2 + 1;
  else
    diff = diff * 2;
  return diff;
}

// Handles one channel c for converting ImageF or Image3F to lossless 8-bit or
// lossless 16-bit, and optionally handles previous pass delta.
template <typename T>
void LosslessChannelPass(const int c, const CodecInOut* io, const Rect& rect,
                         const Image3F& previous_pass, Image<T>* channel_out) {
  size_t xsize = rect.xsize();
  size_t ysize = rect.ysize();
  if (previous_pass.xsize() == 0) {
    for (size_t y = 0; y < ysize; ++y) {
      const float* const PIK_RESTRICT row_in =
          rect.ConstPlaneRow(io->color(), c, y);
      T* const PIK_RESTRICT row_out = channel_out->Row(y);
      for (size_t x = 0; x < xsize; ++x) {
        row_out[x] = RoundForLossless<T>(row_in[x]);
      }
    }
  } else {
    for (size_t y = 0; y < ysize; ++y) {
      const float* const PIK_RESTRICT row_in =
          rect.ConstPlaneRow(io->color(), c, y);
      T* const PIK_RESTRICT row_out = channel_out->Row(y);
      const float* const PIK_RESTRICT row_prev = previous_pass.PlaneRow(0, y);
      for (size_t x = 0; x < xsize; ++x) {
        row_out[x] = DiffForLossless<T>(row_in[x], row_prev[x]);
      }
    }
  }
}

}  // namespace

Status PixelsToPikLosslessFrame(CompressParams cparams,
                                const PassHeader& pass_header,
                                const CodecInOut* io, const Rect& rect,
                                const Image3F& previous_pass,
                                PaddedBytes* compressed, size_t& pos,
                                PikInfo* aux_out) {
  size_t xsize = rect.xsize();
  size_t ysize = rect.ysize();
  if (pass_header.encoding == ImageEncoding::kLosslessGray16) {
    ImageU channel(xsize, ysize);
    LosslessChannelPass(0, io, rect, previous_pass, &channel);
    compressed->resize(pos / 8);
    if (!Grayscale16bit_compress(channel, compressed)) {
      return PIK_FAILURE("Lossless compression failed");
    }
  } else if (pass_header.encoding == ImageEncoding::kLosslessColor16) {
    Image3U image(xsize, ysize);
    LosslessChannelPass(0, io, rect, previous_pass, image.MutablePlane(0));
    LosslessChannelPass(1, io, rect, previous_pass, image.MutablePlane(1));
    LosslessChannelPass(2, io, rect, previous_pass, image.MutablePlane(2));
    compressed->resize(pos / 8);
    if (!Colorful16bit_compress(image, compressed)) {
      return PIK_FAILURE("Lossless compression failed");
    }
  } else if (pass_header.encoding == ImageEncoding::kLosslessGray8) {
    ImageB channel(xsize, ysize);
    LosslessChannelPass(0, io, rect, previous_pass, &channel);
    compressed->resize(pos / 8);
    if (!Grayscale8bit_compress(channel, compressed)) {
      return PIK_FAILURE("Lossless compression failed");
    }
  } else if (pass_header.encoding == ImageEncoding::kLosslessColor8) {
    Image3B image(xsize, ysize);
    LosslessChannelPass(0, io, rect, previous_pass, image.MutablePlane(0));
    LosslessChannelPass(1, io, rect, previous_pass, image.MutablePlane(1));
    LosslessChannelPass(2, io, rect, previous_pass, image.MutablePlane(2));
    compressed->resize(pos / 8);
    if (!Colorful8bit_compress(image, compressed)) {
      return PIK_FAILURE("Lossless compression failed");
    }
  } else {
    return PIK_FAILURE("Unkonwn lossless encoding");
  }
  pos = compressed->size() * 8;
  return true;
}

Status PikPassHeuristics(CompressParams cparams, const PassHeader& pass_header,
                         const Image3F& opsin_orig, const Image3F& opsin,
                         const NoiseParams& noise_params,
                         MultipassManager* multipass_manager,
                         GroupHeader* template_group_header,
                         ColorCorrelationMap* full_cmap,
                         std::shared_ptr<Quantizer>* full_quantizer,
                         AcStrategyImage* full_ac_strategy, PikInfo* aux_out) {
  size_t target_size = cparams.TargetSize(Rect(opsin_orig));
  // TODO(user): This should take *template_group_header size, and size of
  // other passes into account.
  size_t opsin_target_size = target_size;
  if (cparams.target_size > 0 || cparams.target_bitrate > 0.0) {
    cparams.target_size = opsin_target_size;
  } else if (cparams.butteraugli_distance < 0) {
    return PIK_FAILURE("Expected non-negative distance");
  }

  template_group_header->nonserialized_have_alpha = pass_header.has_alpha;

  if (cparams.lossless_mode) {
    return true;
  }

  constexpr size_t N = kBlockDim;
  PROFILER_ZONE("enc OpsinToPik uninstrumented");
  const size_t xsize = opsin_orig.xsize();
  const size_t ysize = opsin_orig.ysize();
  const size_t xsize_blocks = DivCeil(xsize, N);
  const size_t ysize_blocks = DivCeil(ysize, N);
  multipass_manager->GetColorCorrelationMap(opsin, &*full_cmap);
  ImageF quant_field = InitialQuantField(
      cparams.butteraugli_distance, cparams.GetIntensityMultiplier(),
      opsin_orig, cparams, /*pool=*/nullptr, 1.0);
  EncCache cache;
  cache.use_new_dc = cparams.use_new_dc;
  cache.saliency_threshold = cparams.saliency_threshold;
  cache.saliency_debug_skip_nonsalient = cparams.saliency_debug_skip_nonsalient;

  InitializeEncCache(pass_header, *template_group_header, opsin, Rect(opsin),
                     &cache);
  multipass_manager->GetAcStrategy(
      cparams.butteraugli_distance, &quant_field, cache.src,
      /*pool=*/nullptr, &cache.ac_strategy, aux_out);
  *full_ac_strategy = cache.ac_strategy.Copy();

  *full_quantizer = multipass_manager->GetQuantizer(
      cparams, xsize_blocks, ysize_blocks, opsin_orig, opsin, noise_params,
      pass_header, *template_group_header, *full_cmap, cache.ac_strategy,
      quant_field,
      /*pool=*/nullptr, aux_out);
  return true;
}

Status PixelsToPikGroup(CompressParams cparams, const PassHeader& pass_header,
                        GroupHeader header, const AcStrategyImage& ac_strategy,
                        const Quantizer& full_quantizer,
                        const ColorCorrelationMap& full_cmap,
                        const CodecInOut* io, const Image3F& opsin_in,
                        const NoiseParams& noise_params,
                        PaddedBytes* compressed, size_t& pos, PikInfo* aux_out,
                        MultipassHandler* multipass_handler) {
  const Rect& rect = multipass_handler->GroupRect();

  // In progressive mode, encoders may choose to send alpha in any pass, the
  // decoder shouldn't care in which pass it comes. Since at the moment it is
  // unlikely that previews are used for alpha-blending/compositing, we choose
  // to encode alpha only in the last pass. This might change in the future.
  if (pass_header.has_alpha) {
    PROFILER_ZONE("enc alpha");
    PIK_RETURN_IF_ERROR(EncodeAlpha(cparams, io->alpha(), rect, io->AlphaBits(),
                                    &header.alpha));
  }
  header.nonserialized_have_alpha = pass_header.has_alpha;

  size_t extension_bits, total_bits;
  PIK_RETURN_IF_ERROR(CanEncode(header, &extension_bits, &total_bits));
  compressed->resize(DivCeil(pos + total_bits, kBitsPerByte));
  PIK_RETURN_IF_ERROR(
      WriteGroupHeader(header, extension_bits, &pos, compressed->data()));
  WriteZeroesToByteBoundary(&pos, compressed->data());
  if (aux_out != nullptr) {
    aux_out->layers[kLayerHeader].total_size +=
        DivCeil(total_bits, kBitsPerByte);
  }

  if (cparams.lossless_mode) {
    Image3F previous_pass;
    PIK_RETURN_IF_ERROR(multipass_handler->GetPreviousPass(
        io->dec_c_original, /*pool=*/nullptr, &previous_pass));
    return PixelsToPikLosslessFrame(cparams, pass_header, io, rect,
                                    previous_pass, compressed, pos, aux_out);
  }

  Rect group_in_color_tiles(
      multipass_handler->BlockGroupRect().x0() / kColorTileDimInBlocks,
      multipass_handler->BlockGroupRect().y0() / kColorTileDimInBlocks,
      DivCeil(multipass_handler->BlockGroupRect().xsize(),
              kColorTileDimInBlocks),
      DivCeil(multipass_handler->BlockGroupRect().ysize(),
              kColorTileDimInBlocks));
  ColorCorrelationMap cmap = full_cmap.Copy(group_in_color_tiles);
  EncCache cache;
  cache.use_new_dc = cparams.use_new_dc;
  cache.saliency_threshold = cparams.saliency_threshold;
  cache.saliency_debug_skip_nonsalient = cparams.saliency_debug_skip_nonsalient;

  InitializeEncCache(pass_header, header, opsin_in,
                     multipass_handler->PaddedGroupRect(), &cache);
  cache.ac_strategy = ac_strategy.Copy(multipass_handler->BlockGroupRect());

  Quantizer quantizer =
      full_quantizer.Copy(multipass_handler->BlockGroupRect());

  ComputeCoefficients(quantizer, cmap, /*pool=*/nullptr, &cache,
                      multipass_handler->Manager(), aux_out);

  multipass_handler->Manager()->StripInfo(&cache);

  PaddedBytes compressed_data =
      EncodeToBitstream(cache, Rect(cache.src), quantizer, noise_params, cmap,
                        cparams.fast_mode, multipass_handler, aux_out);

  compressed->append(compressed_data);
  pos += compressed_data.size() * kBitsPerByte;

  return true;
}

// Max observed: 1.1M on RGB noise with d0.1.
// 512*512*4*2 = 2M should be enough for 16-bit RGBA images.
using GroupSizeCoder = SizeCoderT<0x150F0E0C>;

Status PixelsToPikPass(CompressParams cparams, const PassParams& pass_params,
                       const CodecInOut* io, ThreadPool* pool,
                       PaddedBytes* compressed, size_t& pos, PikInfo* aux_out,
                       MultipassManager* multipass_manager) {
  PassHeader pass_header;
  pass_header.have_adaptive_reconstruction = false;
  if (cparams.lossless_mode) {
    bool gray = io->IsGray();
    pass_header.encoding = io->original_bits_per_sample() > 8
                               ? (gray ? ImageEncoding::kLosslessGray16
                                       : ImageEncoding::kLosslessColor16)
                               : (gray ? ImageEncoding::kLosslessGray8
                                       : ImageEncoding::kLosslessColor8);
  }

  pass_header.is_last = pass_params.is_last;
  pass_header.frame = pass_params.frame_info;
  pass_header.has_alpha = io->HasAlpha() && pass_params.is_last;

  if (pass_header.encoding == ImageEncoding::kPasses) {
    pass_header.resampling_factor2 = cparams.resampling_factor2;
    pass_header.flags = PassFlagsFromParams(cparams, io);
    pass_header.predict_hf = cparams.predict_hf;
    pass_header.predict_lf = cparams.predict_lf;
    pass_header.gaborish = cparams.gaborish;

    if (io->xsize() <= 8 || io->ysize() <= 8) {
      // Broken, disable.
      pass_header.gaborish = GaborishStrength::kOff;
    }

    if (ApplyOverride(cparams.adaptive_reconstruction,
                      cparams.butteraugli_distance >=
                          kMinButteraugliForAdaptiveReconstruction)) {
      pass_header.have_adaptive_reconstruction = true;
      pass_header.epf_params = cparams.epf_params;
      if (pass_header.epf_params.enable_adaptive &&
          pass_header.epf_params.lut == 0) {
        if (cparams.butteraugli_distance <= 1.0f) {
          pass_header.epf_params.lut = 1;
        } else if (cparams.butteraugli_distance <= 2.0f) {
          pass_header.epf_params.lut = 2;
        } else if (cparams.butteraugli_distance <= 3.0f) {
          pass_header.epf_params.lut = 3;
        } else {
          pass_header.epf_params.lut = 0;
        }
      }
    }
  }

  multipass_manager->StartPass(pass_header);

  // TODO(user): delay writing the header until we know the TOC and the total
  // pass size.
  size_t extension_bits, total_bits;
  PIK_RETURN_IF_ERROR(CanEncode(pass_header, &extension_bits, &total_bits));
  compressed->resize(DivCeil(pos + total_bits, kBitsPerByte));
  PIK_RETURN_IF_ERROR(
      WritePassHeader(pass_header, extension_bits, &pos, compressed->data()));
  WriteZeroesToByteBoundary(&pos, compressed->data());
  if (aux_out != nullptr) {
    aux_out->layers[kLayerHeader].total_size +=
        DivCeil(total_bits, kBitsPerByte);
  }

  const size_t xsize_groups = DivCeil(io->xsize(), kGroupWidth);
  const size_t ysize_groups = DivCeil(io->ysize(), kGroupHeight);
  const size_t num_groups = xsize_groups * ysize_groups;

  std::vector<std::unique_ptr<PikInfo>> aux_outs(num_groups);
  std::vector<MultipassHandler*> handlers(num_groups);
  for (size_t group_index = 0; group_index < num_groups; ++group_index) {
    const size_t gx = group_index % xsize_groups;
    const size_t gy = group_index / xsize_groups;
    const Rect rect(gx * kGroupWidth, gy * kGroupHeight, kGroupWidth,
                    kGroupHeight, io->xsize(), io->ysize());
    handlers[group_index] =
        multipass_manager->GetGroupHandler(group_index, rect);
    if (aux_out != nullptr) {
      aux_outs[group_index] = make_unique<PikInfo>(*aux_out);
    }
  }

  GroupHeader template_group_header;
  ColorCorrelationMap full_cmap(io->xsize(), io->ysize());
  std::shared_ptr<Quantizer> full_quantizer;
  AcStrategyImage full_ac_strategy;
  Image3F opsin_orig, opsin;
  NoiseParams noise_params;

  if (pass_header.encoding == ImageEncoding::kPasses) {
    opsin_orig = OpsinDynamicsImage(io, Rect(io->color()));
    if (pass_header.resampling_factor2 != 2) {
      opsin_orig = DownsampleImage(opsin_orig, pass_header.resampling_factor2);
    }

    constexpr size_t N = kBlockDim;
    PROFILER_ZONE("enc OpsinToPik uninstrumented");
    const size_t xsize = opsin_orig.xsize();
    const size_t ysize = opsin_orig.ysize();
    if (xsize == 0 || ysize == 0) return PIK_FAILURE("Empty image");
    opsin = PadImageToMultiple(opsin_orig, N);

    if (pass_header.flags & PassHeader::kNoise) {
      PROFILER_ZONE("enc GetNoiseParam");
      // Don't start at zero amplitude since adding noise is expensive -- it
      // significantly slows down decoding, and this is unlikely to completely
      // go away even with advanced optimizations. After the
      // kNoiseModelingRampUpDistanceRange we have reached the full level, i.e.
      // noise is no longer represented by the compressed image, so we can add
      // full noise by the noise modeling itself.
      static const double kNoiseModelingRampUpDistanceRange = 0.6;
      static const double kNoiseLevelAtStartOfRampUp = 0.25;
      // TODO(user) test and properly select quality_coef with smooth filter
      float quality_coef = 1.0f;
      const double rampup =
          (cparams.butteraugli_distance - kMinButteraugliForNoise) /
          kNoiseModelingRampUpDistanceRange;
      if (rampup < 1.0) {
        quality_coef = kNoiseLevelAtStartOfRampUp +
                       (1.0 - kNoiseLevelAtStartOfRampUp) * rampup;
      }
      GetNoiseParameter(opsin, &noise_params, quality_coef);
    }
    if (pass_header.gaborish != GaborishStrength::kOff) {
      opsin = SlowGaborishInverse(opsin, 0.92718927264540152);
    }

    multipass_manager->DecorrelateOpsin(&opsin);

    PIK_RETURN_IF_ERROR(
        PikPassHeuristics(cparams, pass_header, opsin_orig, opsin, noise_params,
                          multipass_manager, &template_group_header, &full_cmap,
                          &full_quantizer, &full_ac_strategy, aux_out));
  }

  // Compress groups.
  std::vector<PaddedBytes> group_codes(num_groups);
  std::atomic<int> num_errors{0};
  const auto process_group = [&](const int group_index, const int thread) {
    PaddedBytes* group_code = &group_codes[group_index];
    size_t group_pos = 0;
    if (!PixelsToPikGroup(cparams, pass_header, template_group_header,
                          full_ac_strategy, *full_quantizer, full_cmap, io,
                          opsin, noise_params, group_code, group_pos,
                          aux_outs[group_index].get(), handlers[group_index])) {
      num_errors.fetch_add(1, std::memory_order_relaxed);
      return;
    }
  };
  RunOnPool(pool, 0, num_groups, process_group, "PixelsToPikPass");

  if (aux_out != nullptr) {
    for (size_t group_index = 0; group_index < num_groups; ++group_index) {
      aux_out->Assimilate(*aux_outs[group_index]);
    }
  }

  PIK_RETURN_IF_ERROR(num_errors.load(std::memory_order_relaxed) == 0);

  // Build TOC.
  PaddedBytes group_toc(GroupSizeCoder::MaxSize(num_groups));
  size_t group_toc_pos = 0;
  uint8_t* group_toc_storage = group_toc.data();
  size_t total_groups_size = 0;
  for (size_t group_index = 0; group_index < num_groups; ++group_index) {
    size_t group_size = group_codes[group_index].size();
    GroupSizeCoder::Encode(group_size, &group_toc_pos, group_toc_storage);
    total_groups_size += group_size;
  }
  WriteZeroesToByteBoundary(&group_toc_pos, group_toc_storage);
  group_toc.resize(group_toc_pos / kBitsPerByte);

  // Push output.
  PIK_ASSERT(pos % kBitsPerByte == 0);
  compressed->reserve(DivCeil(pos, kBitsPerByte) + group_toc.size() +
                      total_groups_size);
  compressed->append(group_toc);
  pos += group_toc.size() * kBitsPerByte;
  for (size_t group_index = 0; group_index < num_groups; ++group_index) {
    const PaddedBytes& group_code = group_codes[group_index];
    compressed->append(group_code);
    pos += group_code.size() * kBitsPerByte;
  }

  io->enc_size = compressed->size();
  return true;
}

Status ValidateImageDimensions(const FileHeader& container,
                               const DecompressParams& dparams) {
  const size_t xsize = container.xsize();
  const size_t ysize = container.ysize();
  if (xsize == 0 || ysize == 0) {
    return PIK_FAILURE("Empty image.");
  }

  static const size_t kMaxWidth = (1 << 25) - 1;
  if (xsize > kMaxWidth) {
    return PIK_FAILURE("Image too wide.");
  }

  const size_t num_pixels = xsize * ysize;
  if (num_pixels > dparams.max_num_pixels) {
    return PIK_FAILURE("Image too big.");
  }

  return true;
}

namespace {

// Specializes a 8-bit and 16-bit of converting to float from lossless.
float ToFloatForLossless(uint8_t in) { return static_cast<float>(in); }

float ToFloatForLossless(uint16_t in) { return in * (1.0f / 257); }

// Specializes a 8-bit and 16-bit undo of lossless diff.
float UndiffForLossless(uint8_t in, float prev) {
  uint16_t diff;
  if (in % 2 == 0)
    diff = in / 2;
  else
    diff = 255 - (in / 2);
  uint8_t out = diff + static_cast<int>(RoundForLossless<uint8_t>(prev));
  return ToFloatForLossless(out);
}

float UndiffForLossless(uint16_t in, float prev) {
  uint16_t diff;
  if (in % 2 == 0)
    diff = in / 2;
  else
    diff = 65535 - (in / 2);
  uint16_t out = diff + static_cast<int>(RoundForLossless<uint16_t>(prev));
  return ToFloatForLossless(out);
}

// Handles converting lossless 8-bit or lossless 16-bit, to Image3F, with
// option to give 3x same channel at input for grayscale, and optionally handles
// previous pass delta.
template <typename T>
void LosslessChannelDecodePass(int num_channels, const Image<T>** in,
                               const Rect& rect, const Image3F& previous_pass,
                               Image3F* color) {
  size_t xsize = rect.xsize();
  size_t ysize = rect.ysize();

  for (int c = 0; c < num_channels; c++) {
    if (previous_pass.xsize() == 0) {
      for (size_t y = 0; y < ysize; ++y) {
        const T* const PIK_RESTRICT row_in = in[c]->Row(y);
        float* const PIK_RESTRICT row_out = rect.PlaneRow(color, c, y);
        for (size_t x = 0; x < xsize; ++x) {
          row_out[x] = ToFloatForLossless(row_in[x]);
        }
      }
    } else {
      for (size_t y = 0; y < ysize; ++y) {
        const T* const PIK_RESTRICT row_in = in[c]->Row(y);
        float* const PIK_RESTRICT row_out = rect.PlaneRow(color, c, y);
        const float* const PIK_RESTRICT row_prev =
            previous_pass.ConstPlaneRow(c, y);
        for (size_t x = 0; x < xsize; ++x) {
          row_out[x] = UndiffForLossless(row_in[x], row_prev[x]);
        }
      }
    }
  }

  // Grayscale, copy the channel tot he other two output channels
  if (num_channels == 1) {
    for (size_t y = 0; y < ysize; ++y) {
      const float* const PIK_RESTRICT row_0 = rect.PlaneRow(color, 0, y);
      float* const PIK_RESTRICT row_1 = rect.PlaneRow(color, 1, y);
      float* const PIK_RESTRICT row_2 = rect.PlaneRow(color, 2, y);
      for (size_t x = 0; x < xsize; ++x) {
        row_1[x] = row_2[x] = row_0[x];
      }
    }
  }
}
}  // namespace

Status PikLosslessFrameToPixels(const PaddedBytes& compressed,
                                const PassHeader& pass_header, size_t* position,
                                Image3F* color, const Rect& rect,
                                const Image3F& previous_pass) {
  PROFILER_FUNC;
  if (pass_header.encoding == ImageEncoding::kLosslessGray8) {
    ImageB image;
    if (!Grayscale8bit_decompress(compressed, position, &image)) {
      return PIK_FAILURE("Lossless decompression failed");
    }
    if (!SameSize(image, rect)) {
      return PIK_FAILURE("Lossless decompression yielded wrong dimensions.");
    }
    const ImageB* array[1] = {&image};
    LosslessChannelDecodePass(1, array, rect, previous_pass, color);
  } else if (pass_header.encoding == ImageEncoding::kLosslessGray16) {
    ImageU image;
    if (!Grayscale16bit_decompress(compressed, position, &image)) {
      return PIK_FAILURE("Lossless decompression failed");
    }
    if (!SameSize(image, rect)) {
      return PIK_FAILURE("Lossless decompression yielded wrong dimensions.");
    }
    const ImageU* array[1] = {&image};
    LosslessChannelDecodePass(1, array, rect, previous_pass, color);
  } else if (pass_header.encoding == ImageEncoding::kLosslessColor8) {
    Image3B image;
    if (!Colorful8bit_decompress(compressed, position, &image)) {
      return PIK_FAILURE("Lossless decompression failed");
    }
    if (!SameSize(image, rect)) {
      return PIK_FAILURE("Lossless decompression yielded wrong dimensions.");
    }
    const ImageB* array[3] = {&image.Plane(0), &image.Plane(1),
                              &image.Plane(2)};
    LosslessChannelDecodePass(3, array, rect, previous_pass, color);
  } else if (pass_header.encoding == ImageEncoding::kLosslessColor16) {
    Image3U image;
    if (!Colorful16bit_decompress(compressed, position, &image)) {
      return PIK_FAILURE("Lossless decompression failed");
    }
    if (!SameSize(image, rect)) {
      return PIK_FAILURE("Lossless decompression yielded wrong dimensions.");
    }
    const ImageU* array[3] = {&image.Plane(0), &image.Plane(1),
                              &image.Plane(2)};
    LosslessChannelDecodePass(3, array, rect, previous_pass, color);
  } else {
    return PIK_FAILURE("Unknown lossless encoding");
  }
  return true;
}

// Fit a projective transform which takes:
// (in[0], in[1]) -> (out[0], out[1])
// (in[2], in[3]) -> (out[2], out[3])
// (in[4], in[5]) -> (out[4], out[5])
// (in[6], in[7]) -> (out[6], out[7]).
// The output is to be used as the projective matrix as:
// [ out_0 out_1 out_2 ]
// [ out_3 out_4 out_5 ]
// [ out_6 out_7 1     ]
// To solve, look at:
// a0*x0 + a1*y0 + a2 - c0*x0*X0 - c1*y0*X0 = X0
// b0*x0 + b1*y0 + b2 - c0*x0*Y0 - c1*y0*Y0 = Y0
// a0*x1 + a1*y1 + a2 - c0*x1*X1 - c1*y1*X1 = X1
// b0*x1 + b1*y1 + b2 - c0*x1*Y1 - c1*y1*Y1 = Y1
// a0*x2 + a1*y2 + a2 - c0*x2*X2 - c1*y2*X2 = X2
// b0*x2 + b1*y2 + b2 - c0*x2*Y2 - c1*y2*Y2 = Y2
// a0*x3 + a1*y3 + a2 - c0*x3*X3 - c1*y3*X3 = X3
// b0*x3 + b1*y3 + b2 - c0*x3*Y3 - c1*y3*Y3 = Y3
// Code looks magical, but outputs match skimage _geometric.py.
// TODO(user): Get rid of the Eigen dep.

std::vector<double> FitMatrixToCorners(const std::vector<int>& in_corners,
                                       const std::vector<int>& out_corners) {
  PIK_ASSERT(in_corners.size() == 8);
  PIK_ASSERT(out_corners.size() == 8);

  std::vector<double> out_params(8);
  Eigen::Matrix<double, 8, 8> mat;

  mat << in_corners[0], in_corners[1], 1, 0, 0, 0,
      -in_corners[0] * out_corners[0], -in_corners[1] * out_corners[0], 0, 0, 0,
      in_corners[0], in_corners[1], 1, -in_corners[0] * out_corners[1],
      -in_corners[1] * out_corners[1], in_corners[2], in_corners[3], 1, 0, 0, 0,
      -in_corners[2] * out_corners[2], -in_corners[3] * out_corners[2], 0, 0, 0,
      in_corners[2], in_corners[3], 1, -in_corners[2] * out_corners[3],
      -in_corners[3] * out_corners[3], in_corners[4], in_corners[5], 1, 0, 0, 0,
      -in_corners[4] * out_corners[4], -in_corners[5] * out_corners[4], 0, 0, 0,
      in_corners[4], in_corners[5], 1, -in_corners[4] * out_corners[5],
      -in_corners[5] * out_corners[5], in_corners[6], in_corners[7], 1, 0, 0, 0,
      -in_corners[6] * out_corners[6], -in_corners[7] * out_corners[6], 0, 0, 0,
      in_corners[6], in_corners[7], 1, -in_corners[6] * out_corners[7],
      -in_corners[7] * out_corners[7];
  Eigen::Matrix<double, 8, 1> rhs;
  rhs << out_corners[0], out_corners[1], out_corners[2], out_corners[3],
      out_corners[4], out_corners[5], out_corners[6], out_corners[7];
  Eigen::Matrix<double, 8, 1> transform_coeffs =
      mat.colPivHouseholderQr().solve(rhs);

  for (size_t i = 0; i < 8; i++) {
    out_params[i] = transform_coeffs(i);
  }

  return out_params;
}

// Take coords and apply a projective transform to them. Slow and simple.
// This can be accelerated by using a better matmul, but performance is not a
// significant priority for this code.

std::pair<double, double> CoordinateTransform(double x, double y,
                                              const double* transform) {
  double out_z = transform[6] * x + transform[7] * y + 1;
  double out_x = transform[0] * x + transform[1] * y + transform[2];
  double out_y = transform[3] * x + transform[4] * y + transform[5];

  return {out_x / out_z, out_y / out_z};
}

// Find nearest, according to L1, pixel to (t_x, t_y) for which we have data.
// Returns (-1, -1) if no data is available.
// TODO(user): When this needs to actually run in reasonable time,
// use something locality aware.
std::pair<int, int> FindNearestAvailablePixel(
    int t_x, int t_y, const bool have_input_data[kTileDim][kTileDim]) {
  int best_dist = std::numeric_limits<int>::max();
  std::pair<int, int> best_pixel = {-1, -1};

  for (int y = 0; y < kTileDim; y++) {
    for (int x = 0; x < kTileDim; x++) {
      if (!have_input_data[x][y]) continue;

      int dist = std::abs(y - t_y) + std::abs(x - t_x);
      if (dist < best_dist) {
        best_dist = dist;
        best_pixel = std::make_pair(x, y);
      }
    }
  }

  return best_pixel;
}

// Does bilinear interpolation at (x, y) given values at:
// (floor(x), floor(y)),
// (ceil(x), floor(y)),
// (floor(x), ceil(y)),
// (ceil(x), ceil(y))

double BilinearInterpolate(double delta_x, double delta_y, double ff_value,
                           double cf_value, double fc_value, double cc_value) {
  return ff_value * (1 - delta_x) * (1 - delta_y) +
         cf_value * delta_x * (1 - delta_y) +
         fc_value * (1 - delta_x) * delta_y + cc_value * delta_x * delta_y;
}

void ReverseProjectiveTransformOnTile(
    const ProjectiveTransformParams& transform_params, const Rect& tile_rect,
    Image3F* PIK_RESTRICT transformed_output) {
  std::vector<int> corner_coords(8);
  for (size_t i = 0; i < 8; i++) {
    corner_coords[i] = transform_params.corner_coords[i] - 127;
  }

  std::vector<double> forward_params = FitMatrixToCorners(
      {0, 0, 0, kTileDim, kTileDim, kTileDim, kTileDim, 0}, corner_coords);

  // Determine which datapoints are available.
  bool have_input_data[kTileDim][kTileDim];

  for (size_t y = 0; y < kTileDim; y++) {
    for (size_t x = 0; x < kTileDim; x++) {
      have_input_data[y][x] = false;
    }
  }

  constexpr int kFourNeighbourhood[5][2] = {
      {0, 0}, {-1, 0}, {1, 0}, {0, -1}, {0, 1}};

  for (size_t y = 0; y < kTileDim; y++) {
    for (size_t x = 0; x < kTileDim; x++) {
      std::pair<double, double> forward_coords =
          CoordinateTransform(x, y, forward_params.data());

      double floor_x = std::floor(forward_coords.first);
      double floor_y = std::floor(forward_coords.second);

      for (size_t n = 0; n < 5; n++) {
        int t_x = static_cast<int>(floor_x + kFourNeighbourhood[n][0]);
        int t_y = static_cast<int>(floor_y + kFourNeighbourhood[n][1]);

        if (t_x >= 0 && t_x < kTileDim && t_y >= 0 && t_y < kTileDim) {
          have_input_data[t_x][t_y] = true;
        }
      }
    }
  }

  // Do reverse transform. This uses bilinear when we have all four relevant
  // pixels and nearest neighbor when we don't.
  // TODO(user): Simplify this logic and use bicubic when dependencies
  // are satisfied.
  Image3F out_tile(kTileDim, kTileDim);

  for (size_t y = 0; y < kTileDim; y++) {
    float* PIK_RESTRICT row_r = out_tile.PlaneRow(0, y);
    float* PIK_RESTRICT row_g = out_tile.PlaneRow(1, y);
    float* PIK_RESTRICT row_b = out_tile.PlaneRow(2, y);
    for (size_t x = 0; x < kTileDim; x++) {
      // Not a bug here, we're looking for where (x, y) ends up after the
      // transform so we can copy from there.
      std::pair<double, double> reverse_coords =
          CoordinateTransform(x, y, forward_params.data());

      int floor_x = std::floor(reverse_coords.first);
      int floor_y = std::floor(reverse_coords.second);
      int ceil_x = std::ceil(reverse_coords.first);
      int ceil_y = std::ceil(reverse_coords.second);

      bool have_all_pixels =
          floor_x >= 0 && floor_x < kTileDim && floor_y >= 0 &&
          floor_y < kTileDim && ceil_x >= 0 && ceil_x < kTileDim &&
          ceil_y >= 0 && ceil_y < kTileDim &&
          have_input_data[floor_x][floor_y] &&
          have_input_data[floor_x][ceil_y] &&
          have_input_data[ceil_x][floor_y] && have_input_data[ceil_x][ceil_y];
      bool should_interpolate = (floor_x != ceil_x) && (floor_y != ceil_y);

      if (have_all_pixels && should_interpolate) {
        double delta_x = reverse_coords.first - floor_x;
        double delta_y = reverse_coords.second - floor_y;

        for (size_t p = 0; p < 3; p++) {
          out_tile.PlaneRow(p, y)[x] = BilinearInterpolate(
              delta_x, delta_y,
              tile_rect.PlaneRow(transformed_output, p, floor_y)[floor_x],
              tile_rect.PlaneRow(transformed_output, p, floor_y)[ceil_x],
              tile_rect.PlaneRow(transformed_output, p, ceil_y)[floor_x],
              tile_rect.PlaneRow(transformed_output, p, ceil_y)[ceil_x]);
        }
      } else {
        double s_x = std::round(reverse_coords.first);
        double s_y = std::round(reverse_coords.second);

        std::pair<int, int> nearest_pixel_with_data = FindNearestAvailablePixel(
            static_cast<int>(s_x), static_cast<int>(s_y), have_input_data);

        PIK_ASSERT(nearest_pixel_with_data.first >= 0 &&
                   nearest_pixel_with_data.second >= 0);
        PIK_ASSERT(nearest_pixel_with_data.first < kTileDim &&
                   nearest_pixel_with_data.second < kTileDim);
        int source_x = nearest_pixel_with_data.first;
        int source_y = nearest_pixel_with_data.second;

        row_r[x] =
            tile_rect.PlaneRow(transformed_output, 0, source_y)[source_x];
        row_g[x] =
            tile_rect.PlaneRow(transformed_output, 1, source_y)[source_x];
        row_b[x] =
            tile_rect.PlaneRow(transformed_output, 2, source_y)[source_x];
      }
    }
  }

  for (size_t y = 0; y < kTileDim; y++) {
    const float* PIK_RESTRICT row_r_in = out_tile.ConstPlaneRow(0, y);
    const float* PIK_RESTRICT row_g_in = out_tile.ConstPlaneRow(1, y);
    const float* PIK_RESTRICT row_b_in = out_tile.ConstPlaneRow(2, y);

    float* PIK_RESTRICT row_r_out =
        tile_rect.PlaneRow(transformed_output, 0, y);
    float* PIK_RESTRICT row_g_out =
        tile_rect.PlaneRow(transformed_output, 1, y);
    float* PIK_RESTRICT row_b_out =
        tile_rect.PlaneRow(transformed_output, 2, y);
    for (size_t x = 0; x < kTileDim; x++) {
      row_r_out[x] = row_r_in[x];
      row_g_out[x] = row_g_in[x];
      row_b_out[x] = row_b_in[x];
    }
  }
}

void ReverseProjectiveTransform(const GroupHeader& group_header,
                                const Rect& group_rect,
                                Image3F* PIK_RESTRICT color_output) {
  constexpr size_t kGroupWidthInTiles = kGroupWidthInBlocks / kTileDimInBlocks;
  constexpr size_t kGroupHeightInTiles =
      kGroupHeightInBlocks / kTileDimInBlocks;

  for (size_t y = 0; y < kGroupHeightInTiles; y++) {
    for (size_t x = 0; x < kGroupWidthInTiles; x++) {
      if (group_header.tile_headers[y * kGroupWidthInTiles + x]
              .have_projective_transform) {
        Rect tile_rect(group_rect.x0() + x * kTileDim,
                       group_rect.y0() + y * kTileDim, kTileDim, kTileDim,
                       kGroupWidth, kGroupHeight);
        ReverseProjectiveTransformOnTile(
            group_header.tile_headers[y * kGroupWidthInTiles + x]
                .projective_transform_params,
            tile_rect, color_output);
      }
    }
  }
}

Status PikGroupToPixels(const DecompressParams& dparams,
                        const FileHeader& container,
                        const PassHeader* pass_header,
                        const PaddedBytes& compressed, BitReader* reader,
                        Image3F* PIK_RESTRICT opsin_output,
                        ImageU* alpha_output, CodecContext* context,
                        PikInfo* aux_out, PassDecCache* pass_dec_cache,
                        MultipassHandler* multipass_handler,
                        const ColorEncoding& original_color_encoding) {
  PROFILER_FUNC;
  const Rect& padded_rect = multipass_handler->PaddedGroupRect();
  const Rect& rect = multipass_handler->GroupRect();
  GroupHeader header;
  header.nonserialized_have_alpha = pass_header->has_alpha;
  PIK_RETURN_IF_ERROR(ReadGroupHeader(reader, &header));
  PIK_RETURN_IF_ERROR(reader->JumpToByteBoundary());
  OverrideGroupFlags(dparams, pass_header, &header);

  if (pass_header->has_alpha) {
    // TODO(user): do not fail here based on the metadata
    // original_bytes_per_alpha, it should be allowed to use an efficient
    // encoding in pik which differs from what the original had (or
    // alternatively if they must be the same, there should not be two fields)
    if (header.alpha.bytes_per_alpha !=
        container.metadata.transcoded.original_bytes_per_alpha) {
      return PIK_FAILURE("Nonuniform alpha bitdepth is not supported yet.");
    }
    if (container.metadata.transcoded.original_bytes_per_alpha == 0) {
      return PIK_FAILURE("Header claims to contain alpha but the depth is 0.");
    }
    PIK_RETURN_IF_ERROR(DecodeAlpha(dparams, header.alpha, alpha_output, rect));
  }

  if (pass_header->encoding == ImageEncoding::kLosslessGray8 ||
      pass_header->encoding == ImageEncoding::kLosslessGray16 ||
      pass_header->encoding == ImageEncoding::kLosslessColor8 ||
      pass_header->encoding == ImageEncoding::kLosslessColor16) {
    size_t pos = reader->Position();
    size_t before_pos = pos;
    Image3F previous_pass;
    PIK_RETURN_IF_ERROR(multipass_handler->GetPreviousPass(
        original_color_encoding, /*pool=*/nullptr, &previous_pass));
    auto result = PikLosslessFrameToPixels(compressed, *pass_header, &pos,
                                           opsin_output, rect, previous_pass);
    reader->SkipBits((pos - before_pos) << 3);
    // Byte-wise; no need to jump to boundary.
    return result;
  }

  const size_t resampling_factor2 = pass_header->resampling_factor2;
  if (resampling_factor2 != 2 && resampling_factor2 != 3 &&
      resampling_factor2 != 4 && resampling_factor2 != 8) {
    return PIK_FAILURE("Pik decoding failed: invalid resampling factor");
  }

  ImageSize opsin_size = DownsampledImageSize(
      ImageSize::Make(padded_rect.xsize(), padded_rect.ysize()),
      resampling_factor2);
  const size_t xsize_blocks = DivCeil<size_t>(opsin_size.xsize, kBlockDim);
  const size_t ysize_blocks = DivCeil<size_t>(opsin_size.ysize, kBlockDim);

  Quantizer quantizer(kBlockDim, 0, 0, 0);
  NoiseParams noise_params;
  ColorCorrelationMap cmap(opsin_size.xsize, opsin_size.ysize);
  DecCache dec_cache;
  dec_cache.use_new_dc = dparams.use_new_dc;

  {
    PROFILER_ZONE("dec_bitstr");
    if (!DecodeFromBitstream(*pass_header, header, compressed, reader,
                             padded_rect, multipass_handler, xsize_blocks,
                             ysize_blocks, &cmap, &noise_params, &quantizer,
                             &dec_cache, pass_dec_cache)) {
      return PIK_FAILURE("Pik decoding failed.");
    }
    if (!reader->JumpToByteBoundary()) {
      return PIK_FAILURE("Pik bitstream is corrupted.");
    }
  }

  multipass_handler->SaveAcStrategy(pass_dec_cache->ac_strategy);
  multipass_handler->SaveQuantField(pass_dec_cache->raw_quant_field);

  // DequantImage is not invoked, because coefficients are eagerly dequantized
  // in DecodeFromBitstream.
  // TODO(user): avoid copy by passing opsin_output and having ReconOpsinImage
  // fill it (assuming no resampling).
  Image3F opsin = ReconOpsinImage(*pass_header, header, quantizer,
                                  multipass_handler->BlockGroupRect(),
                                  &dec_cache, pass_dec_cache, aux_out);

  if (pass_header->flags & PassHeader::kNoise) {
    PROFILER_ZONE("add_noise");
    AddNoise(noise_params, &opsin);
  }

  if (resampling_factor2 != 2) {
    PROFILER_ZONE("UpsampleImage");
    opsin = UpsampleImage(opsin, padded_rect.xsize(), padded_rect.ysize(),
                          resampling_factor2);
  }

  for (size_t c = 0; c < 3; c++) {
    for (size_t y = 0; y < padded_rect.ysize(); y++) {
      const float* PIK_RESTRICT row = opsin.ConstPlaneRow(c, y);
      float* PIK_RESTRICT output_row = padded_rect.PlaneRow(opsin_output, c, y);
      for (size_t x = 0; x < padded_rect.xsize(); x++) {
        output_row[x] = row[x];
      }
    }
  }

  multipass_handler->SetDecoderQuantizer(std::move(quantizer));
  if (pass_header->is_last) {
    ReverseProjectiveTransform(header, rect, opsin_output);
  }
  return true;
}

Status PikPassToPixels(const DecompressParams& dparams,
                       const PaddedBytes& compressed,
                       const FileHeader& container, ThreadPool* pool,
                       BitReader* reader, CodecInOut* io, PikInfo* aux_out,
                       MultipassManager* multipass_handler) {
  PROFILER_ZONE("PikPassToPixels uninstrumented");
  PIK_RETURN_IF_ERROR(ValidateImageDimensions(container, dparams));

  io->metadata = container.metadata;

  // Used when writing the output file unless DecoderHints overrides it.
  io->SetOriginalBitsPerSample(
      container.metadata.transcoded.original_bit_depth);
  io->dec_c_original = container.metadata.transcoded.original_color_encoding;
  if (io->dec_c_original.icc.empty()) {
    // Removed by MaybeRemoveProfile; fail unless we successfully restore it.
    PIK_RETURN_IF_ERROR(
        ColorManagement::SetProfileFromFields(&io->dec_c_original));
  }

  const size_t xsize = container.xsize();
  const size_t ysize = container.ysize();
  size_t padded_xsize = DivCeil(xsize, kBlockDim) * kBlockDim;
  size_t padded_ysize = DivCeil(ysize, kBlockDim) * kBlockDim;

  PassHeader header;
  PIK_RETURN_IF_ERROR(ReadPassHeader(reader, &header));

  PIK_RETURN_IF_ERROR(reader->JumpToByteBoundary());

  // TODO(user): add kProgressive.
  if (header.encoding != ImageEncoding::kPasses &&
      header.encoding != ImageEncoding::kLosslessGray8 &&
      header.encoding != ImageEncoding::kLosslessGray16 &&
      header.encoding != ImageEncoding::kLosslessColor8 &&
      header.encoding != ImageEncoding::kLosslessColor16) {
    return PIK_FAILURE("Unsupported bitstream");
  }

  multipass_handler->StartPass(header);

  OverridePassFlags(dparams, &header);

  ImageU alpha;
  if (header.has_alpha) {
    alpha = ImageU(xsize, ysize);
  }

  const size_t xsize_groups = DivCeil(xsize, kGroupWidth);
  const size_t ysize_groups = DivCeil(ysize, kGroupHeight);
  const size_t num_groups = xsize_groups * ysize_groups;

  std::vector<PikInfo> aux_outs;
  if (aux_out != nullptr) {
    aux_outs.resize(num_groups, *aux_out);
  }
  std::vector<MultipassHandler*> handlers(num_groups);
  {
    PROFILER_ZONE("Get handlers");
    for (size_t group_index = 0; group_index < num_groups; ++group_index) {
      const size_t gx = group_index % xsize_groups;
      const size_t gy = group_index / xsize_groups;
      const size_t x = gx * kGroupWidth;
      const size_t y = gy * kGroupHeight;
      Rect rect(x, y, kGroupWidth, kGroupHeight, xsize, ysize);
      handlers[group_index] =
          multipass_handler->GetGroupHandler(group_index, rect);
    }
  }

  // Read TOC.
  std::vector<size_t> group_offsets;
  {
    PROFILER_ZONE("Read TOC");
    group_offsets.reserve(num_groups + 1);
    group_offsets.push_back(0);
    for (size_t group_index = 0; group_index < num_groups; ++group_index) {
      const uint32_t size = GroupSizeCoder::Decode(reader);
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

  PassDecCache pass_dec_cache;
  pass_dec_cache.ac_strategy =
      AcStrategyImage(padded_xsize / kBlockDim, padded_ysize / kBlockDim);
  pass_dec_cache.raw_quant_field =
      ImageI(padded_xsize / kBlockDim, padded_ysize / kBlockDim);
  pass_dec_cache.biases =
      Image3F(padded_xsize * kBlockDim, padded_ysize / kBlockDim);

  Image3F opsin(padded_xsize, padded_ysize);

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

    PikInfo* my_aux_out = aux_out ? &aux_outs[group_index] : nullptr;
    if (!PikGroupToPixels(dparams, container, &header, compressed,
                          &group_reader, &opsin, &alpha, io->Context(),
                          my_aux_out, &pass_dec_cache, handlers[group_index],
                          io->dec_c_original)) {
      num_errors.fetch_add(1);
      return;
    }
  };
  RunOnPool(pool, 0, num_groups, process_group, "PikPassToPixels");

  if (aux_out != nullptr) {
    for (size_t group_index = 0; group_index < num_groups; ++group_index) {
      aux_out->Assimilate(aux_outs[group_index]);
    }
  }

  PIK_RETURN_IF_ERROR(num_errors.load(std::memory_order_relaxed) == 0);

  if (header.encoding == ImageEncoding::kPasses) {
    // TODO(user): here we "steal" the quantizer of the first group. This
    // assumes that all group quantizer share the global scale and DC, which is
    // true as the encoder produces these kind of images. The format should
    // disallow other situations.
    Quantizer quantizer = handlers[0]->TakeDecoderQuantizer();

    multipass_handler->RestoreOpsin(&opsin);
    multipass_handler->UpdateBiases(&pass_dec_cache.biases);
    multipass_handler->StoreBiases(pass_dec_cache.biases);
    multipass_handler->SetDecodedPass(opsin);

    opsin = FinalizePassDecoding(std::move(opsin), header, quantizer,
                                 &pass_dec_cache, aux_out);

    Image3F color(padded_xsize, padded_ysize);
    OpsinToLinear(opsin, Rect(opsin), &color);

    if (header.flags & PassHeader::kGrayscaleOpt) {
      PROFILER_ZONE("Grayscale opt");
      // Force all channels to gray
      for (size_t y = 0; y < color.ysize(); ++y) {
        float* PIK_RESTRICT row_r = color.PlaneRow(0, y);
        float* PIK_RESTRICT row_g = color.PlaneRow(1, y);
        float* PIK_RESTRICT row_b = color.PlaneRow(2, y);
        for (size_t x = 0; x < color.xsize(); x++) {
          float gray = row_r[x] * 0.299 + row_g[x] * 0.587 + row_b[x] * 0.114;
          row_r[x] = row_g[x] = row_b[x] = gray;
        }
      }
    }
    const ColorEncoding& c =
        io->Context()->c_linear_srgb[io->dec_c_original.IsGray()];
    io->SetFromImage(std::move(color), c);
  } else if (header.encoding == ImageEncoding::kLosslessGray8 ||
             header.encoding == ImageEncoding::kLosslessGray16 ||
             header.encoding == ImageEncoding::kLosslessColor8 ||
             header.encoding == ImageEncoding::kLosslessColor16) {
    io->SetFromImage(std::move(opsin), io->dec_c_original);
    multipass_handler->SetDecodedPass(io);
  } else {
    return PIK_FAILURE("Unsupported image encoding");
  }

  if (header.has_alpha) {
    io->SetAlpha(std::move(alpha),
                 8 * container.metadata.transcoded.original_bytes_per_alpha);
  }

  io->ShrinkTo(xsize, ysize);

  return true;
}

}  // namespace pik
