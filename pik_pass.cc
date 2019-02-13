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

#undef PROFILER_ENABLED
#define PROFILER_ENABLED 1
#include "ac_strategy.h"
#include "adaptive_quantization.h"
#include "alpha.h"
#include "ar_control_field.h"
#include "arch_specific.h"
#include "bit_reader.h"
#include "bits.h"
#include "byte_order.h"
#include "color_correlation.h"
#include "color_encoding.h"
#include "common.h"
#include "compiler_specific.h"
#include "compressed_dc.h"
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
#include "lossless16.h"
#include "lossless8.h"
#include "multipass_handler.h"
#include "noise.h"
#include "opsin_image.h"
#include "opsin_inverse.h"
#include "padded_bytes.h"
#include "pik_params.h"
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

  if (ApplyOverride(cparams.gradient, dist >= kMinButteraugliForGradient)) {
    flags |= PassHeader::kGradientMap;
  }

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
  OverrideFlag(dparams.noise, PassHeader::kNoise, &pass_header->flags);
  OverrideFlag(dparams.gradient, PassHeader::kGradientMap, &pass_header->flags);

  if (dparams.adaptive_reconstruction == Override::kOff) {
    pass_header->have_adaptive_reconstruction = false;
  } else if (dparams.adaptive_reconstruction == Override::kOn) {
    pass_header->have_adaptive_reconstruction = true;
  }
  pass_header->epf_params.use_sharpened = ApplyOverride(
      dparams.epf_use_sharpened, pass_header->epf_params.use_sharpened);
  if (dparams.epf_sigma > 0) {
    pass_header->epf_params.enable_adaptive = false;
    pass_header->epf_params.sigma = dparams.epf_sigma;
  }

  if (dparams.gaborish != -1) {
    pass_header->gaborish = GaborishStrength(dparams.gaborish);
  }
}

void OverrideGroupFlags(const DecompressParams& dparams,
                        const PassHeader* PIK_RESTRICT pass_header,
                        GroupHeader* PIK_RESTRICT header) {}

// Specializes a 8-bit and 16-bit of rounding from floating point to lossless.
template <typename T>
T RoundForLossless(float in);

template <>
uint8_t RoundForLossless(float in) {
  // NOTE: if in was originally an 8 or 16 bit value, we don't need to round
  // because such values are exactly representable as floats. Rounding is only
  // needed when forcing inexact values back to integers.
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

Status PixelsToPikLosslessFrame(CompressParams cparams,
                                const PassHeader& pass_header,
                                const CodecInOut* io, const Rect& rect,
                                const Image3F& previous_pass,
                                PaddedBytes* compressed, size_t& pos,
                                PikInfo* aux_out) {
  PIK_ASSERT(pos % kBitsPerByte == 0);
  size_t xsize = rect.xsize();
  size_t ysize = rect.ysize();
  if (pass_header.lossless_grayscale) {
    if (pass_header.lossless_16_bits) {
      ImageU channel(xsize, ysize);
      LosslessChannelPass(0, io, rect, previous_pass, &channel);
      compressed->resize(pos / kBitsPerByte);
      if (!Grayscale16bit_compress(channel, compressed)) {
        return PIK_FAILURE("Lossless compression failed");
      }
    } else {
      ImageB channel(xsize, ysize);
      LosslessChannelPass(0, io, rect, previous_pass, &channel);
      compressed->resize(pos / kBitsPerByte);
      if (!Grayscale8bit_compress(channel, compressed)) {
        return PIK_FAILURE("Lossless compression failed");
      }
    }
  } else {
    if (pass_header.lossless_16_bits) {
      Image3U image(xsize, ysize);
      LosslessChannelPass(0, io, rect, previous_pass,
                          const_cast<ImageU*>(&image.Plane(0)));
      LosslessChannelPass(1, io, rect, previous_pass,
                          const_cast<ImageU*>(&image.Plane(1)));
      LosslessChannelPass(2, io, rect, previous_pass,
                          const_cast<ImageU*>(&image.Plane(2)));
      compressed->resize(pos / kBitsPerByte);
      if (!Colorful16bit_compress(image, compressed)) {
        return PIK_FAILURE("Lossless compression failed");
      }
    } else {
      Image3B image(xsize, ysize);
      LosslessChannelPass(0, io, rect, previous_pass,
                          const_cast<ImageB*>(&image.Plane(0)));
      LosslessChannelPass(1, io, rect, previous_pass,
                          const_cast<ImageB*>(&image.Plane(1)));
      LosslessChannelPass(2, io, rect, previous_pass,
                          const_cast<ImageB*>(&image.Plane(2)));
      compressed->resize(pos / kBitsPerByte);
      if (!Colorful8bit_compress(image, compressed)) {
        return PIK_FAILURE("Lossless compression failed");
      }
    }
  }
  pos = compressed->size() * kBitsPerByte;
  return true;
}

// Returns the target size based on whether bitrate or direct targetsize is
// given.
size_t TargetSize(const CompressParams& cparams, const Rect& rect) {
  if (cparams.target_size > 0) {
    return cparams.target_size;
  }
  if (cparams.target_bitrate > 0.0) {
    return 0.5 + cparams.target_bitrate * rect.xsize() * rect.ysize() / 8;
  }
  return 0;
}

Status PikPassHeuristics(
    CompressParams cparams, const PassHeader& pass_header,
    const Image3F& opsin_orig, const Image3F& opsin, DequantMatrices* dequant,
    MultipassManager* multipass_manager, GroupHeader* template_group_header,
    ColorCorrelationMap* full_cmap, std::shared_ptr<Quantizer>* full_quantizer,
    AcStrategyImage* full_ac_strategy, ImageB* full_ar_sigma_lut_ids,
    BlockDictionary* block_dictionary, PikInfo* aux_out) {
  size_t target_size = TargetSize(cparams, Rect(opsin_orig));
  // TODO(robryk): This should take *template_group_header size, and size of
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

  *dequant = multipass_manager->GetDequantMatrices(cparams.butteraugli_distance,
                                                   opsin);

  *block_dictionary = multipass_manager->GetBlockDictionary(
      cparams.butteraugli_distance, opsin);

  Image3F opsin_with_removed_blocks = CopyImage(opsin);
  block_dictionary->SubtractFrom(&opsin_with_removed_blocks);

  multipass_manager->GetColorCorrelationMap(opsin_with_removed_blocks, dequant,
                                            &*full_cmap);
  ImageF quant_field = InitialQuantField(
      cparams.butteraugli_distance, cparams.GetIntensityMultiplier(),
      opsin_orig, cparams, /*pool=*/nullptr, 1.0);

  multipass_manager->GetAcStrategy(cparams.butteraugli_distance, &quant_field,
                                   dequant, opsin_with_removed_blocks,
                                   /*pool=*/nullptr, full_ac_strategy, aux_out);

  // TODO(veluca): investigate if this should be included in multipass_manager.
  FindBestArControlField(cparams.butteraugli_distance,
                         opsin_with_removed_blocks, *full_ac_strategy,
                         quant_field, pass_header.gaborish,
                         full_ar_sigma_lut_ids);

  *full_quantizer = multipass_manager->GetQuantizer(
      cparams, xsize_blocks, ysize_blocks, opsin_orig, opsin, pass_header,
      *template_group_header, *full_cmap, *block_dictionary, *full_ac_strategy,
      *full_ar_sigma_lut_ids, dequant, quant_field,
      /*pool=*/nullptr, aux_out);
  return true;
}

Status PixelsToPikGroup(CompressParams cparams, const PassHeader& pass_header,
                        GroupHeader header, const AcStrategyImage& ac_strategy,
                        const Quantizer* full_quantizer,
                        const ColorCorrelationMap& full_cmap,
                        const CodecInOut* io, const Image3F& opsin_in,
                        const NoiseParams& noise_params,
                        std::vector<PaddedBytes>* compressed, size_t& pos,
                        const PassEncCache& pass_enc_cache, PikInfo* aux_out,
                        MultipassHandler* multipass_handler) {
  const Rect& rect = multipass_handler->GroupRect();
  const Rect& padded_rect = multipass_handler->PaddedGroupRect();
  const Rect area_to_encode =
      Rect(0, 0, padded_rect.xsize(), padded_rect.ysize());

  if (pass_header.has_alpha) {
    PROFILER_ZONE("enc alpha");
    PIK_RETURN_IF_ERROR(EncodeAlpha(cparams, io->alpha(), rect, io->AlphaBits(),
                                    &header.alpha));
  }
  header.nonserialized_have_alpha = pass_header.has_alpha;

  size_t extension_bits, total_bits;
  PIK_RETURN_IF_ERROR(CanEncode(header, &extension_bits, &total_bits));
  (*compressed)[0].resize(DivCeil(pos + total_bits, kBitsPerByte));
  PIK_RETURN_IF_ERROR(
      WriteGroupHeader(header, extension_bits, &pos, (*compressed)[0].data()));
  WriteZeroesToByteBoundary(&pos, (*compressed)[0].data());
  if (aux_out != nullptr) {
    aux_out->layers[kLayerHeader].total_size +=
        DivCeil(total_bits, kBitsPerByte);
  }

  if (cparams.lossless_mode) {
    // Done; we'll encode the entire image in one shot later.
    return true;
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
  cache.saliency_threshold = cparams.saliency_threshold;
  cache.saliency_debug_skip_nonsalient = cparams.saliency_debug_skip_nonsalient;

  InitializeEncCache(pass_header, header, pass_enc_cache,
                     multipass_handler->PaddedGroupRect(), &cache);
  cache.ac_strategy = ac_strategy.Copy(multipass_handler->BlockGroupRect());

  Quantizer quantizer =
      full_quantizer->Copy(multipass_handler->BlockGroupRect());

  ComputeCoefficients(quantizer, full_cmap, group_in_color_tiles,
                      /*pool=*/nullptr, &cache, aux_out);

  std::vector<Image3S> ac_split = multipass_handler->SplitACCoefficients(
      std::move(cache.ac), cache.ac_strategy);

  PIK_ASSERT(ac_split.size() == multipass_handler->Manager()->GetNumPasses());

  for (size_t i = 0; i < ac_split.size(); i++) {
    cache.ac = std::move(ac_split[i]);
    PaddedBytes compressed_data =
        EncodeToBitstream(cache, area_to_encode, quantizer, noise_params,
                          cparams.fast_mode, multipass_handler, aux_out);

    (*compressed)[i].append(compressed_data);
    pos += compressed_data.size() * kBitsPerByte;
  }

  return true;
}

// Max observed: 1.1M on RGB noise with d0.1.
// 512*512*4*2 = 2M should be enough for 16-bit RGBA images.
using GroupSizeCoder = SizeCoderT<0x150F0E0C>;

}  // namespace

Status PixelsToPikPass(CompressParams cparams, const PassParams& pass_params,
                       const CodecInOut* io, ThreadPool* pool,
                       PaddedBytes* compressed, size_t& pos, PikInfo* aux_out,
                       MultipassManager* multipass_manager) {
  PassHeader pass_header;
  pass_header.num_passes = multipass_manager->GetNumPasses();
  pass_header.have_adaptive_reconstruction = false;
  if (cparams.lossless_mode) {
    pass_header.encoding = ImageEncoding::kLossless;
    pass_header.lossless_16_bits = io->original_bits_per_sample() > 8;
    pass_header.lossless_grayscale = io->IsGray();
  }

  pass_header.frame = pass_params.frame_info;
  pass_header.has_alpha = io->HasAlpha();

  if (pass_header.encoding == ImageEncoding::kPasses) {
    pass_header.flags = PassFlagsFromParams(cparams, io);
    pass_header.predict_hf = cparams.predict_hf;
    pass_header.predict_lf = cparams.predict_lf;
    pass_header.gaborish = GaborishStrength(cparams.gaborish);

    if (ApplyOverride(cparams.adaptive_reconstruction,
                      cparams.butteraugli_distance >=
                          kMinButteraugliForAdaptiveReconstruction)) {
      pass_header.have_adaptive_reconstruction = true;
      pass_header.epf_params.use_sharpened = ApplyOverride(
          cparams.epf_use_sharpened, pass_header.epf_params.use_sharpened);
      if (cparams.epf_sigma > 0) {
        pass_header.epf_params.enable_adaptive = false;
        pass_header.epf_params.sigma = cparams.epf_sigma;
      }
    }
  }

  multipass_manager->StartPass(pass_header);

  // TODO(veluca): delay writing the header until we know the total pass size.
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

  const size_t xsize_groups = DivCeil(io->xsize(), kGroupDim);
  const size_t ysize_groups = DivCeil(io->ysize(), kGroupDim);
  const size_t num_groups = xsize_groups * ysize_groups;

  std::vector<std::unique_ptr<PikInfo>> aux_outs(num_groups);
  std::vector<MultipassHandler*> handlers(num_groups);
  for (size_t group_index = 0; group_index < num_groups; ++group_index) {
    const size_t gx = group_index % xsize_groups;
    const size_t gy = group_index / xsize_groups;
    const Rect rect(gx * kGroupDim, gy * kGroupDim, kGroupDim, kGroupDim,
                    io->xsize(), io->ysize());
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
  ImageB ar_sigma_lut_ids;
  Image3F opsin_orig, opsin;
  NoiseParams noise_params;
  BlockDictionary block_dictionary;
  PassEncCache pass_enc_cache;

  if (pass_header.encoding == ImageEncoding::kPasses) {
    opsin_orig = OpsinDynamicsImage(io, Rect(io->color()));
    if (aux_out != nullptr) {
      PIK_RETURN_IF_ERROR(
          aux_out->InspectImage3F("pik_pass:OpsinDynamicsImage", opsin_orig));
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
      opsin = GaborishInverse(opsin, 0.92718927264540152);
    }

    multipass_manager->DecorrelateOpsin(&opsin);

    PIK_RETURN_IF_ERROR(PikPassHeuristics(
        cparams, pass_header, opsin_orig, opsin, &pass_enc_cache.matrices,
        multipass_manager, &template_group_header, &full_cmap, &full_quantizer,
        &full_ac_strategy, &ar_sigma_lut_ids, &block_dictionary, aux_out));

    // Initialize pass_enc_cache and encode DC.
    InitializePassEncCache(pass_header, opsin, full_ac_strategy,
                           *full_quantizer, full_cmap, block_dictionary, pool,
                           &pass_enc_cache, aux_out);
    pass_enc_cache.use_new_dc = cparams.use_new_dc;

    PikImageSizeInfo* matrices_info =
        aux_out != nullptr ? &aux_out->layers[kLayerDequantTables] : nullptr;

    std::string dequant_code = pass_enc_cache.matrices.Encode(matrices_info);
    compressed->append(dequant_code);
    pos += dequant_code.size() * 8;

    PaddedBytes pass_global_code;
    size_t byte_pos = 0;

    // Encode quantizer DC and global scale.
    PikImageSizeInfo* quant_info =
        aux_out ? &aux_out->layers[kLayerQuant] : nullptr;
    std::string quant_code = full_quantizer->Encode(quant_info);

    // Encode cmap. TODO(veluca): consider encoding DC part of cmap only here,
    // and AC in (super)groups.
    PikImageSizeInfo* cmap_info =
        aux_out ? &aux_out->layers[kLayerCmap] : nullptr;
    std::string cmap_code =
        EncodeColorMap(full_cmap.ytob_map, Rect(full_cmap.ytob_map),
                       full_cmap.ytob_dc, cmap_info) +
        EncodeColorMap(full_cmap.ytox_map, Rect(full_cmap.ytox_map),
                       full_cmap.ytox_dc, cmap_info);

    pass_global_code.resize(quant_code.size() + cmap_code.size());
    Append(quant_code, &pass_global_code, &byte_pos);
    Append(cmap_code, &pass_global_code, &byte_pos);

    PikImageSizeInfo* dc_info =
        aux_out != nullptr ? &aux_out->layers[kLayerDC] : nullptr;
    PikImageSizeInfo* cfields_info =
        aux_out != nullptr ? &aux_out->layers[kLayerControlFields] : nullptr;

    pass_global_code.append(EncodeDC(*full_quantizer, pass_enc_cache,
                                     full_ac_strategy, pool, multipass_manager,
                                     dc_info, cfields_info));
    compressed->append(pass_global_code);
    pos += pass_global_code.size() * 8;
    PikImageSizeInfo* dictionary_info =
        aux_out ? &aux_out->layers[kLayerDictionary] : nullptr;
    std::string dictionary_code = block_dictionary.Encode(dictionary_info);
    compressed->append(dictionary_code);
    pos += dictionary_code.size() * 8;
  }

  // Compress groups: one per combination of group and pass. Outer loop lists
  // passes, inner lists groups. Group headers are only encoded in the groups of
  // the first pass.
  std::vector<std::vector<PaddedBytes>> group_codes(num_groups);
  std::atomic<int> num_errors{0};
  const auto process_group = [&](const int group_index, const int thread) {
    std::vector<PaddedBytes>* group_code = &group_codes[group_index];
    size_t group_pos = 0;
    group_code->resize(multipass_manager->GetNumPasses());
    if (!PixelsToPikGroup(cparams, pass_header, template_group_header,
                          full_ac_strategy, full_quantizer.get(), full_cmap, io,
                          opsin, noise_params, group_code, group_pos,
                          pass_enc_cache, aux_outs[group_index].get(),
                          handlers[group_index])) {
      num_errors.fetch_add(1, std::memory_order_relaxed);
      return;
    }
  };
  RunOnPool(pool, 0, num_groups, process_group, "PixelsToPikPass");

  for (size_t i = 0; i < num_groups; i++) {
    PIK_ASSERT(group_codes[i].size() == multipass_manager->GetNumPasses());
  }

  if (aux_out != nullptr) {
    for (size_t group_index = 0; group_index < num_groups; ++group_index) {
      aux_out->Assimilate(*aux_outs[group_index]);
    }
  }

  PIK_RETURN_IF_ERROR(num_errors.load(std::memory_order_relaxed) == 0);

  // Build TOCs.

  for (size_t i = 0; i < multipass_manager->GetNumPasses(); i++) {
    size_t group_toc_pos = 0;
    PaddedBytes group_toc(PaddedBytes(GroupSizeCoder::MaxSize(num_groups)));
    uint8_t* group_toc_storage = group_toc.data();
    size_t total_groups_size = 0;
    for (size_t group_index = 0; group_index < num_groups; ++group_index) {
      size_t group_size = group_codes[group_index][i].size();
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

    // Only do lossless encoding in the first pass, if there is more than one.
    if (pass_header.encoding == ImageEncoding::kLossless && i == 0) {
      // Encode entire image at once to avoid per-group overhead. Must come
      // BEFORE the encoded groups because the decoder assumes that the last
      // group coincides with the end of the bitstream.
      const Rect rect(io->color());

      Image3F previous_pass;
      PIK_RETURN_IF_ERROR(multipass_manager->GetPreviousPass(
          io->dec_c_original, pool, &previous_pass));
      PIK_RETURN_IF_ERROR(PixelsToPikLosslessFrame(cparams, pass_header, io,
                                                   rect, previous_pass,
                                                   compressed, pos, aux_out));
    }

    for (size_t group_index = 0; group_index < num_groups; ++group_index) {
      const PaddedBytes& group_code = group_codes[group_index][i];
      compressed->append(group_code);
      pos += group_code.size() * kBitsPerByte;
    }
  }

  io->enc_size = compressed->size();
  return true;
}

namespace {

Status ValidateImageDimensions(const FileHeader& file_header,
                               const DecompressParams& dparams) {
  const size_t xsize = file_header.xsize();
  const size_t ysize = file_header.ysize();
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

  // Grayscale, copy the channel to the other two output channels
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

Status PikLosslessFrameToPixels(const PaddedBytes& compressed,
                                const PassHeader& pass_header, size_t* position,
                                Image3F* color, const Rect& rect,
                                const Image3F& previous_pass) {
  PROFILER_FUNC;
  if (pass_header.lossless_grayscale) {
    if (pass_header.lossless_16_bits) {
      ImageU image;
      if (!Grayscale16bit_decompress(compressed, position, &image)) {
        return PIK_FAILURE("Lossless decompression failed");
      }
      if (!SameSize(image, rect)) {
        return PIK_FAILURE("Lossless decompression yielded wrong dimensions.");
      }
      const ImageU* array[1] = {&image};
      LosslessChannelDecodePass(1, array, rect, previous_pass, color);
    } else {
      ImageB image;
      if (!Grayscale8bit_decompress(compressed, position, &image)) {
        return PIK_FAILURE("Lossless decompression failed");
      }
      if (!SameSize(image, rect)) {
        return PIK_FAILURE("Lossless decompression yielded wrong dimensions.");
      }
      const ImageB* array[1] = {&image};
      LosslessChannelDecodePass(1, array, rect, previous_pass, color);
    }
  } else {
    if (pass_header.lossless_16_bits) {
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
    }
  }
  return true;
}

// `reader` is a vector of readers (one per pass). Group headers are only
// present in the first pass, thus the group header of this group is read from
// `reader[0]`.
Status PikGroupToPixels(
    const DecompressParams& dparams, const FileHeader& file_header,
    const PassHeader* pass_header, const PaddedBytes& compressed,
    const Quantizer& quantizer, const ColorCorrelationMap& full_cmap,
    std::vector<BitReader>* reader, Image3F* PIK_RESTRICT opsin_output,
    ImageU* alpha_output, const CodecContext* context, PikInfo* aux_out,
    PassDecCache* PIK_RESTRICT pass_dec_cache,
    GroupDecCache* PIK_RESTRICT group_dec_cache,
    MultipassHandler* multipass_handler,
    const ColorEncoding& original_color_encoding, size_t downsample) {
  PROFILER_FUNC;
  const Rect& padded_rect = multipass_handler->PaddedGroupRect();
  const Rect& rect = multipass_handler->GroupRect();
  GroupHeader header;
  header.nonserialized_have_alpha = pass_header->has_alpha;
  PIK_RETURN_IF_ERROR(ReadGroupHeader(&(*reader)[0], &header));
  PIK_RETURN_IF_ERROR((*reader)[0].JumpToByteBoundary());
  OverrideGroupFlags(dparams, pass_header, &header);

  if (pass_header->has_alpha) {
    // TODO(lode): do not fail here based on the metadata
    // original_bytes_per_alpha, it should be allowed to use an efficient
    // encoding in pik which differs from what the original had (or
    // alternatively if they must be the same, there should not be two fields)
    if (header.alpha.bytes_per_alpha !=
        file_header.metadata.transcoded.original_bytes_per_alpha) {
      return PIK_FAILURE("Nonuniform alpha bitdepth is not supported yet.");
    }
    if (file_header.metadata.transcoded.original_bytes_per_alpha == 0) {
      return PIK_FAILURE("Header claims to contain alpha but the depth is 0.");
    }
    PIK_RETURN_IF_ERROR(DecodeAlpha(dparams, header.alpha, alpha_output, rect));
  }

  if (pass_header->encoding == ImageEncoding::kLossless) {
    // Done; we'll decode the entire image in one shot later.
    return true;
  }

  ImageSize opsin_size =
      ImageSize::Make(padded_rect.xsize(), padded_rect.ysize());
  const size_t xsize_blocks = DivCeil<size_t>(opsin_size.xsize, kBlockDim);
  const size_t ysize_blocks = DivCeil<size_t>(opsin_size.ysize, kBlockDim);

  Rect group_in_color_tiles(
      multipass_handler->BlockGroupRect().x0() / kColorTileDimInBlocks,
      multipass_handler->BlockGroupRect().y0() / kColorTileDimInBlocks,
      DivCeil(multipass_handler->BlockGroupRect().xsize(),
              kColorTileDimInBlocks),
      DivCeil(multipass_handler->BlockGroupRect().ysize(),
              kColorTileDimInBlocks));

  NoiseParams noise_params;

  InitializeDecCache(*pass_dec_cache, padded_rect, group_dec_cache);

  for (size_t i = 0; i < pass_header->num_passes; i++) {
    PROFILER_ZONE("dec_bitstr");
    auto decode = i == 0 ? &DecodeFromBitstream</*first=*/true>
                         : &DecodeFromBitstream</*first=*/false>;
    if (!decode(*pass_header, header, compressed, &(*reader)[i], padded_rect,
                multipass_handler, xsize_blocks, ysize_blocks, full_cmap,
                group_in_color_tiles, &noise_params, quantizer, pass_dec_cache,
                group_dec_cache)) {
      return PIK_FAILURE("Pik decoding failed.");
    }
    if (!(*reader)[i].JumpToByteBoundary()) {
      return PIK_FAILURE("Pik bitstream is corrupted.");
    }
  }

  Rect opsin_rect(padded_rect.x0() / downsample, padded_rect.y0() / downsample,
                  DivCeil(padded_rect.xsize(), downsample),
                  DivCeil(padded_rect.ysize(), downsample));

  // Note: DecodeFromBitstream already performed dequantization.
  ReconOpsinImage(*pass_header, header, quantizer,
                  multipass_handler->BlockGroupRect(), pass_dec_cache,
                  group_dec_cache, opsin_output, opsin_rect, aux_out,
                  downsample);

  return true;
}

}  // namespace

Status PikPassToPixels(const DecompressParams& dparams,
                       const PaddedBytes& compressed,
                       const FileHeader& file_header, ThreadPool* pool,
                       BitReader* reader, CodecInOut* io, PikInfo* aux_out,
                       MultipassManager* multipass_manager, size_t downsample) {
  PROFILER_ZONE("PikPassToPixels uninstrumented");
  PIK_RETURN_IF_ERROR(ValidateImageDimensions(file_header, dparams));

  io->metadata = file_header.metadata;

  // Used when writing the output file unless DecoderHints overrides it.
  io->SetOriginalBitsPerSample(
      file_header.metadata.transcoded.original_bit_depth);
  io->dec_c_original = file_header.metadata.transcoded.original_color_encoding;
  if (io->dec_c_original.icc.empty()) {
    // Removed by MaybeRemoveProfile; fail unless we successfully restore it.
    PIK_RETURN_IF_ERROR(
        ColorManagement::SetProfileFromFields(&io->dec_c_original));
  }

  const size_t xsize = file_header.xsize();
  const size_t ysize = file_header.ysize();
  const size_t padded_xsize = DivCeil(xsize, kBlockDim) * kBlockDim;
  const size_t padded_ysize = DivCeil(ysize, kBlockDim) * kBlockDim;

  PassHeader header;
  PIK_RETURN_IF_ERROR(ReadPassHeader(reader, &header));

  PIK_RETURN_IF_ERROR(reader->JumpToByteBoundary());

  // TODO(veluca): add kProgressive.
  if (header.encoding != ImageEncoding::kPasses &&
      header.encoding != ImageEncoding::kLossless) {
    return PIK_FAILURE("Unsupported bitstream");
  }

  OverridePassFlags(dparams, &header);

  multipass_manager->StartPass(header);

  ImageU alpha;
  if (header.has_alpha) {
    alpha = ImageU(xsize, ysize);
  }

  const size_t xsize_groups = DivCeil(xsize, kGroupDim);
  const size_t ysize_groups = DivCeil(ysize, kGroupDim);
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
      const size_t x = gx * kGroupDim;
      const size_t y = gy * kGroupDim;
      Rect rect(x, y, kGroupDim, kGroupDim, xsize, ysize);
      handlers[group_index] =
          multipass_manager->GetGroupHandler(group_index, rect);
    }
  }

  const size_t xsize_blocks = padded_xsize / kBlockDim;
  const size_t ysize_blocks = padded_ysize / kBlockDim;

  PassDecCache pass_dec_cache;
  pass_dec_cache.use_new_dc = dparams.use_new_dc;
  pass_dec_cache.grayscale = header.flags & PassHeader::kGrayscaleOpt;
  pass_dec_cache.ac_strategy = AcStrategyImage(xsize_blocks, ysize_blocks);
  pass_dec_cache.raw_quant_field = ImageI(xsize_blocks, ysize_blocks);

  // TODO(veluca): Decode the lut image in DC groups decoding.
  pass_dec_cache.sigma_lut_ids = ImageB(xsize_blocks, ysize_blocks);
  ZeroFillImage(&pass_dec_cache.sigma_lut_ids);

  ColorCorrelationMap cmap(xsize, ysize);

  // TODO(veluca): deserialize quantization tables from the bitstream.

  Quantizer quantizer(&pass_dec_cache.matrices, 0, 0);
  BlockDictionary block_dictionary;

  std::vector<GroupDecCache> group_dec_caches(NumThreads(pool));

  if (header.encoding == ImageEncoding::kPasses) {
    PROFILER_ZONE("DecodeColorMap+DC");
    PIK_RETURN_IF_ERROR(pass_dec_cache.matrices.Decode(reader));
    PIK_RETURN_IF_ERROR(quantizer.Decode(reader));
    PIK_RETURN_IF_ERROR(reader->JumpToByteBoundary());
    DecodeColorMap(reader, &cmap.ytob_map, &cmap.ytob_dc);
    DecodeColorMap(reader, &cmap.ytox_map, &cmap.ytox_dc);
    PIK_RETURN_IF_ERROR(DecodeDC(
        reader, compressed, header, xsize_blocks, ysize_blocks, quantizer, cmap,
        pool, multipass_manager, &pass_dec_cache, &group_dec_caches));
    PIK_RETURN_IF_ERROR(
        block_dictionary.Decode(reader, padded_xsize, padded_ysize));
    multipass_manager->SaveAcStrategy(pass_dec_cache.ac_strategy);
    multipass_manager->SaveQuantField(pass_dec_cache.raw_quant_field);
  }

  Image3F opsin(DivCeil(padded_xsize, downsample),
                DivCeil(padded_ysize, downsample));

  // Read TOCs.
  std::vector<std::vector<size_t>> group_offsets(header.num_passes);
  std::vector<size_t> group_codes_begin(header.num_passes);
  for (size_t i = 0; i < header.num_passes; i++) {
    PROFILER_ZONE("Read TOC");
    std::vector<size_t>& group_offsets_pass = group_offsets[i];
    group_offsets_pass.reserve(num_groups + 1);
    group_offsets_pass.push_back(0);
    for (size_t group_index = 0; group_index < num_groups; ++group_index) {
      const uint32_t size = GroupSizeCoder::Decode(reader);
      group_offsets_pass.push_back(group_offsets_pass.back() + size);
    }
    PIK_RETURN_IF_ERROR(reader->JumpToByteBoundary());
    // On first pass, read lossless.
    if (header.encoding == ImageEncoding::kLossless && i == 0) {
      const Rect rect(0, 0, xsize, ysize);

      size_t pos = reader->Position();
      const size_t before_pos = pos;

      Image3F previous_pass;
      PIK_RETURN_IF_ERROR(multipass_manager->GetPreviousPass(
          io->dec_c_original, pool, &previous_pass));
      PIK_RETURN_IF_ERROR(PikLosslessFrameToPixels(
          compressed, header, &pos, &opsin, rect, previous_pass));
      reader->SkipBits((pos - before_pos) * kBitsPerByte);
      // Byte-wise; no need to jump to boundary.
    }
    // Pretend all groups of this pass are read.
    group_codes_begin[i] = reader->Position();
    reader->SkipBits(group_offsets_pass.back() * kBitsPerByte);
    if (reader->Position() > compressed.size()) {
      return PIK_FAILURE("Group code extends after stream end");
    }
  }

  // Decode groups.
  std::atomic<int> num_errors{0};
  const auto process_group = [&](const int group_index, const int thread) {
    std::vector<BitReader> readers;
    for (size_t i = 0; i < header.num_passes; i++) {
      size_t group_code_offset = group_offsets[i][group_index];
      size_t group_reader_limit = group_offsets[i][group_index + 1];
      // TODO(user): this looks ugly; we should get rid of PaddedBytes
      //               parameter once it is wrapped into BitReader; otherwise
      //               it is easy to screw the things up.
      readers.emplace_back(compressed.data(),
                           group_codes_begin[i] + group_reader_limit);
      readers.back().SkipBits((group_codes_begin[i] + group_code_offset) *
                              kBitsPerByte);
    }

    PikInfo* my_aux_out = aux_out ? &aux_outs[group_index] : nullptr;
    if (!PikGroupToPixels(dparams, file_header, &header, compressed, quantizer,
                          cmap, &readers, &opsin, &alpha, io->Context(),
                          my_aux_out, &pass_dec_cache,
                          &group_dec_caches[thread], handlers[group_index],
                          io->dec_c_original, downsample)) {
      num_errors.fetch_add(1);
      return;
    }
  };
  {
    PROFILER_ZONE("PikPassToPixels pool");
    RunOnPool(pool, 0, num_groups, process_group, "PikPassToPixels");
  }

  PIK_RETURN_IF_ERROR(num_errors.load(std::memory_order_relaxed) == 0);

  if (aux_out != nullptr) {
    for (size_t group_index = 0; group_index < num_groups; ++group_index) {
      aux_out->Assimilate(aux_outs[group_index]);
    }
  }

  if (header.encoding == ImageEncoding::kPasses) {
    multipass_manager->RestoreOpsin(&opsin);
    multipass_manager->SetDecodedPass(opsin);

    PIK_RETURN_IF_ERROR(
        FinalizePassDecoding(&opsin, file_header.xsize(), file_header.ysize(),
                             header, NoiseParams(), quantizer, block_dictionary,
                             pool, &pass_dec_cache, aux_out, downsample));
    // From now on, `opsin` is actually linear sRGB.

    if (header.flags & PassHeader::kGrayscaleOpt) {
      PROFILER_ZONE("Grayscale opt");
      // Force all channels to gray
      for (size_t y = 0; y < opsin.ysize(); ++y) {
        float* PIK_RESTRICT row_r = opsin.PlaneRow(0, y);
        float* PIK_RESTRICT row_g = opsin.PlaneRow(1, y);
        float* PIK_RESTRICT row_b = opsin.PlaneRow(2, y);
        for (size_t x = 0; x < opsin.xsize(); x++) {
          float gray = row_r[x] * 0.299 + row_g[x] * 0.587 + row_b[x] * 0.114;
          row_r[x] = row_g[x] = row_b[x] = gray;
        }
      }
    }
    const ColorEncoding& c =
        io->Context()->c_linear_srgb[io->dec_c_original.IsGray()];
    io->SetFromImage(std::move(opsin), c);
  } else if (header.encoding == ImageEncoding::kLossless) {
    io->SetFromImage(std::move(opsin), io->dec_c_original);
    io->ShrinkTo(xsize, ysize);
    multipass_manager->SetDecodedPass(io);
  } else {
    return PIK_FAILURE("Unsupported image encoding");
  }

  if (header.has_alpha) {
    io->SetAlpha(std::move(alpha),
                 8 * file_header.metadata.transcoded.original_bytes_per_alpha);
  }

  io->ShrinkTo(DivCeil(xsize, downsample), DivCeil(ysize, downsample));

  return true;
}

}  // namespace pik
