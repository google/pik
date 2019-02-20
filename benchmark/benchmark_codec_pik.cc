// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.
#include "benchmark/benchmark_codec_pik.h"

#include <sstream>
#include <string>

#include "benchmark/benchmark_file_io.h"
#include "pik/image.h"
#include "pik/pik.h"

namespace pik {

struct PikArgs {
  double quant_bias;

  bool use_ac_strategy;
  bool progressive;

  double intensity_target;
  Override noise;
  Override adaptive_reconstruction;
  Override gradient;

  int gaborish = int(GaborishStrength::k750);

  bool use_new_dc;

  std::string debug_image_dir;
};

static PikArgs* const pikargs = new PikArgs;

Status AddCommandLineOptionsPikCodec(BenchmarkArgs* args) {
  args->AddDouble(&pikargs->quant_bias, "quant_bias",
                  "Bias border pixels during quantization by this ratio.", 0.0);
  args->AddFlag(&pikargs->use_ac_strategy, "use_ac_strategy",
                "If true, AC strategy will be used.", false);
  args->AddFlag(&pikargs->progressive, "progressive",
                "Enable progressive mode.", false);

  args->AddDouble(
      &pikargs->intensity_target, "intensity_target",
      "Intended viewing intensity target in nits, defaults to the value "
      "Butteraugli is tuned for.",
      pik::kDefaultIntensityTarget);

  args->AddOverride(&pikargs->noise, "noise",
                    "Enable(1)/disable(0) noise generation.");
  args->AddOverride(
      &pikargs->adaptive_reconstruction, "adaptive_reconstruction",
      "Enable(1)/disable(0) adaptive reconstruction (deringing).");
  args->AddOverride(
      &pikargs->gradient, "gradient",
      "Enable(1)/disable(0) the extra gradient map (e.g. for skies).");

  args->cmdline.AddOptionValue('\0', "gaborish", "0..7",
                               "chooses deblocking strength (4=normal).",
                               &pikargs->gaborish, &ParseGaborishStrength);

  args->AddFlag(
      &pikargs->use_new_dc, "use_new_dc",
      "Enable new Lossless codec for DC. This flag exists only temporarily "
      "as long as both old and new implementation co-exist, and eventually "
      "only the new implementation should remain.",
      false);

  args->AddString(
      &pikargs->debug_image_dir, "debug_image_dir",
      "If not empty, saves debug images for each "
      "input image and each codec that provides it to this directory.");

  return true;
}

class PikCodec : public ImageCodec {
 public:
  PikCodec(const BenchmarkArgs& args, const CodecContext& codec_context)
      : ImageCodec(args, codec_context) {}
  Status ParseParam(const std::string& param) override {
    const std::string kMaxPassesPrefix = "max_passes=";
    const std::string kDownsamplingPrefix = "downsampling=";

    dparams_.noise = pikargs->noise;
    dparams_.gradient = pikargs->gradient;
    dparams_.adaptive_reconstruction = pikargs->adaptive_reconstruction;

    cparams_.progressive_mode = pikargs->progressive;
    cparams_.use_new_dc = pikargs->use_new_dc;
    dparams_.use_new_dc = pikargs->use_new_dc;

    if (ImageCodec::ParseParam(param)) {
      // Nothing to do.
    } else if (param[0] == 'q') {
      cparams_.uniform_quant = strtof(param.substr(1).c_str(), nullptr);
      hf_asymmetry_ = args_.hf_asymmetry;
    } else if (param.substr(0, kMaxPassesPrefix.size()) == kMaxPassesPrefix) {
      std::istringstream parser(param.substr(kMaxPassesPrefix.size()));
      parser >> dparams_.max_passes;
    } else if (param.substr(0, kDownsamplingPrefix.size()) ==
               kDownsamplingPrefix) {
      std::istringstream parser(param.substr(kDownsamplingPrefix.size()));
      parser >> dparams_.max_downsampling;
    } else if (param == "lossless") {
      cparams_.lossless_mode = true;
    } else if (param == "fast") {
      cparams_.fast_mode = true;
    } else if (param == "guetzli") {
      cparams_.guetzli_mode = true;
    } else {
      return PIK_FAILURE("Unrecognized PIK param");
    }
    return true;
  }

  Status Compress(const std::string& filename, const CodecInOut* io,
                  ThreadPool* pool, PaddedBytes* compressed) override {
    if (!pikargs->debug_image_dir.empty()) {
      cinfo_.debug_prefix =
          JoinPath(pikargs->debug_image_dir, FileBaseName(filename)) +
          ".pik:" + params_ + ".dbg/";
      PIK_RETURN_IF_ERROR(MakeDir(cinfo_.debug_prefix));
    }
    cparams_.butteraugli_distance = butteraugli_target_;
    cparams_.target_bitrate = bitrate_target_;
    cparams_.intensity_target = pikargs->intensity_target;

    cparams_.noise = pikargs->noise;
    cparams_.gradient = pikargs->gradient;
    cparams_.adaptive_reconstruction = pikargs->adaptive_reconstruction;
    cparams_.gaborish = pikargs->gaborish;

    cparams_.quant_border_bias = pikargs->quant_bias;
    cparams_.use_ac_strategy = pikargs->use_ac_strategy;
    cparams_.hf_asymmetry = hf_asymmetry_;

    return PixelsToPik(cparams_, io, compressed, &cinfo_, pool);
  }

  Status Decompress(const std::string& filename, const PaddedBytes& compressed,
                    ThreadPool* pool, CodecInOut* io) override {
    if (!pikargs->debug_image_dir.empty()) {
      dinfo_.debug_prefix =
          JoinPath(pikargs->debug_image_dir, FileBaseName(filename)) +
          ".pik:" + params_ + ".dbg/";
      PIK_RETURN_IF_ERROR(MakeDir(dinfo_.debug_prefix));
    }
    return PikToPixels(dparams_, compressed, io, &dinfo_, pool);
  }

  void GetMoreStats(BenchmarkStats* stats) override {
    PikStats pik_stats;
    pik_stats.num_inputs = 1;
    pik_stats.info = cinfo_;
    pik_stats.info.adaptive_reconstruction_aux =
        dinfo_.adaptive_reconstruction_aux;
    stats->pik_stats.Assimilate(pik_stats);
  }

 protected:
  PikInfo cinfo_;
  PikInfo dinfo_;
  CompressParams cparams_;
  DecompressParams dparams_;
};

ImageCodec* CreateNewPikCodec(const BenchmarkArgs& args,
                              const CodecContext& context) {
  return new PikCodec(args, context);
}

}  // namespace pik
