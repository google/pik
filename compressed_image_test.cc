// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "compressed_image.h"

#include <array>

#include "gtest/gtest.h"
#include "butteraugli_distance.h"
#include "codec.h"
#include "common.h"
#include "compressed_dc.h"
#include "entropy_coder.h"
#include "gaborish.h"
#include "gradient_map.h"
#include "image_test_utils.h"
#include "opsin_image.h"
#include "opsin_inverse.h"
#include "single_image_handler.h"
#include "testdata_path.h"

namespace pik {
namespace {

// Verifies ReconOpsinImage reconstructs with low butteraugli distance.
void RoundTrip(const CompressParams& cparams, const PassHeader& pass_header,
               const GroupHeader& header, const Image3F& opsin,
               Quantizer* quantizer, const ColorCorrelationMap& cmap,
               const Rect& cmap_rect, const CodecInOut* io0, ThreadPool* pool) {
  AcStrategyImage ac_strategy(quantizer->RawQuantField().xsize(),
                              quantizer->RawQuantField().ysize());
  PassEncCache pass_enc_cache;
  BlockDictionary dictionary;
  InitializePassEncCache(pass_header, opsin, ac_strategy, *quantizer, cmap,
                         dictionary, pool, &pass_enc_cache, nullptr);
  EncCache enc_cache;
  InitializeEncCache(pass_header, header, pass_enc_cache, Rect(opsin),
                     &enc_cache);
  enc_cache.ac_strategy = ac_strategy.Copy();
  ComputeCoefficients(*quantizer, cmap, cmap_rect, pool, &enc_cache);

  PassDecCache pass_dec_cache;
  pass_dec_cache.dc = CopyImage(pass_enc_cache.dc_dec);
  pass_dec_cache.gradient = std::move(pass_enc_cache.gradient);
  if (pass_header.flags & PassHeader::kGradientMap) {
    ApplyGradientMap(pass_dec_cache.gradient, *quantizer, &pass_dec_cache.dc);
  }
  // Override quant field with the one seen by decoder.
  pass_dec_cache.raw_quant_field = std::move(enc_cache.quant_field);
  pass_dec_cache.ac_strategy = std::move(enc_cache.ac_strategy);

  GroupDecCache group_dec_cache;

  InitializeDecCache(pass_dec_cache, Rect(opsin), &group_dec_cache);
  DequantImageAC(*quantizer, cmap, cmap_rect, enc_cache.ac, &pass_dec_cache,
                 &group_dec_cache, Rect(opsin));

  Image3F recon(pass_dec_cache.dc.xsize() * kBlockDim,
                pass_dec_cache.dc.ysize() * kBlockDim);
  ReconOpsinImage(pass_header, header, *quantizer, Rect(enc_cache.quant_field),
                  &pass_dec_cache, &group_dec_cache, &recon, Rect(recon));

  PIK_CHECK(FinalizePassDecoding(&recon, io0->xsize(), io0->ysize(),
                                 pass_header, NoiseParams(), *quantizer,
                                 dictionary, pool, &pass_dec_cache));
  CodecInOut io1(io0->Context());
  io1.SetFromImage(std::move(recon), io0->Context()->c_linear_srgb[0]);

  EXPECT_LE(ButteraugliDistance(io0, &io1, cparams.hf_asymmetry,
                                /*distmap=*/nullptr, pool),
            1.2);
}

void RunRGBRoundTrip(float distance, bool fast) {
  CodecContext codec_context;
  ThreadPool pool(4);

  const std::string& pathname =
      GetTestDataPath("wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  constexpr int N = kBlockDim;
  CodecInOut io(&codec_context);
  PIK_CHECK(io.SetFromFile(pathname, &pool));

  Image3F opsin =
      PadImageToMultiple(OpsinDynamicsImage(&io, Rect(io.color())), N);
  opsin = GaborishInverse(opsin, 1.0);

  ColorCorrelationMap cmap(opsin.xsize(), opsin.ysize());
  const Rect cmap_rect(cmap.ytob_map);
  DequantMatrices dequant(/*need_inv_matrices=*/true);
  Quantizer quantizer(&dequant, opsin.xsize() / N, opsin.ysize() / N);
  quantizer.SetQuant(4.0f);

  CompressParams cparams;
  cparams.butteraugli_distance = distance;
  cparams.fast_mode = fast;

  PassHeader pass_header;
  GroupHeader header;
  pass_header.gaborish = GaborishStrength::k1000;

  RoundTrip(cparams, pass_header, header, opsin, &quantizer, cmap, cmap_rect,
            &io, &pool);
}

TEST(CompressedImageTest, RGBRoundTrip_1) { RunRGBRoundTrip(1.0, false); }

TEST(CompressedImageTest, RGBRoundTrip_1_fast) { RunRGBRoundTrip(1.0, true); }

TEST(CompressedImageTest, RGBRoundTrip_2) { RunRGBRoundTrip(2.0, false); }

TEST(CompressedImageTest, RGBRoundTrip_2_fast) { RunRGBRoundTrip(2.0, true); }

}  // namespace
}  // namespace pik
