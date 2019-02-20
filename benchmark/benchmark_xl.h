// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_BENCHMARK_BENCHMARK_H_
#define PIK_BENCHMARK_BENCHMARK_H_

#include <deque>
#include <string>

#include "pik/args.h"
#include "pik/cmdline.h"
#include "pik/image.h"
#include "pik/pik_info.h"
#include "pik/pik_params.h"

namespace pik {

int ParseIntParam(const std::string& param, int lower_bound, int upper_bound);

struct PikStats {
  PikStats() {
    num_inputs = 0;
    info = PikInfo();
  }
  void Assimilate(const PikStats& victim) {
    num_inputs += victim.num_inputs;
    info.Assimilate(victim.info);
  }
  void Print() const { info.Print(num_inputs); }

  size_t num_inputs;
  PikInfo info;
};

// The value of an entry in the table. Depending on the ColumnType, the string,
// size_t or double should be used.
struct ColumnValue {
  std::string s;
  size_t i;
  double f;
};

struct BenchmarkStats {
  BenchmarkStats();

  void Assimilate(const BenchmarkStats& victim);

  float TotalTime() const { return total_time_encode + total_time_decode; }

  std::vector<ColumnValue> ComputeColumns(const std::string& codec_desc,
                                          size_t corpus_size,
                                          size_t num_threads) const;

  std::string PrintLine(const std::string& codec_desc, size_t corpus_size,
                   size_t num_threads) const;

  void PrintMoreStats() const;

  size_t num_inputs;
  size_t total_input_pixels;
  size_t total_uncompressed_input_size;  // [bytes]
  size_t total_compressed_size;
  size_t total_adj_compressed_size;
  double total_time_encode;
  double total_time_decode;
  float max_distance;  // Max butteraugli score
  double
      distance_p_norm;  // sum of 8th powers of all butteraugli distmap pixels.
  double distance_2;    // sum of 2nd powers of all differences between R, G, B.
  std::vector<float> distances;
  PikStats pik_stats;
};

// Command line arguments
struct BenchmarkArgs {
  void AddFlag(bool* field, const char* longName, const char* help,
      bool defaultValue) {
    const char* noName = RememberString_(std::string("no") + longName);
    cmdline.AddOptionFlag('\0', longName, nullptr, field, &SetBooleanTrue);
    cmdline.AddOptionFlag('\0', noName, help, field, &SetBooleanFalse);
    *field = defaultValue;
  }

  void AddOverride(Override* field, const char* longName, const char* help) {
    cmdline.AddOptionValue('\0', longName, "0|1", help, field,
        &ParseOverride);
    *field = Override::kDefault;
  }

  void AddString(std::string* field, const char* longName, const char* help,
      const std::string& defaultValue = "") {
    cmdline.AddOptionValue('\0', longName, "<string>",
                           help, field, &ParseString);
    *field = defaultValue;
  }

  void AddDouble(double* field, const char* longName, const char* help,
      double defaultValue) {
    cmdline.AddOptionValue('\0', longName, "<scalar>",
                           help, field, &ParseDouble);
    *field = defaultValue;
  }

  void AddSigned(int* field, const char* longName, const char* help,
      int defaultValue) {
    cmdline.AddOptionValue('\0', longName, "<integer>",
                           help, field, &ParseSigned);
    *field = defaultValue;
  }

  Status AddCommandLineOptions();

  Status ValidateArgs() {
    return true;
  }

  bool Parse(int argc, const char** argv) {
    return cmdline.Parse(argc, argv);
  }

  void PrintHelp() const {
    cmdline.PrintHelp();
  }

  std::string input;
  std::string codec;
  bool print_details;
  bool print_more_stats;
  bool print_distance_percentiles;
  bool save_compressed;
  bool save_decompressed;

  double mul_output;
  double heatmap_good;
  double heatmap_bad;

  bool write_html_report;

  std::string originals_url;
  std::string output_dir;

  int num_threads;
  int inner_threads;

  std::string sample_tmp_dir;

  int num_samples;
  int sample_dimensions;
  double hf_asymmetry;

  bool profiler;
  double error_pnorm;
  bool show_progress;

  tools::CommandLineParser cmdline;

 private:
  const char* RememberString_(const std::string& text) {
    const char* data = text.c_str();
    std::vector<char> copy(data, data + text.size() + 1);
    string_pool_.push_back(copy);
    return string_pool_.back().data();
  }

  // A memory pool with stable addresses for strings to provide stable
  // const char pointers to cmdline.h for dynamic help/name strings.
  std::deque<std::vector<char>> string_pool_;
};

// Thread-compatible.
class ImageCodec {
 public:
  ImageCodec(const BenchmarkArgs& args, const CodecContext& context)
      : args_(args), codec_context_(context),
        butteraugli_target_(1.0f),
        bitrate_target_(0.0f), hf_asymmetry_(1.0f) {}

  virtual ~ImageCodec() {}

  void set_description(const std::string& desc) { description_ = desc; }
  const std::string& description() const { return description_; }

  float hf_asymmetry() const { return hf_asymmetry_; }

  virtual void ParseParameters(const std::string& parameters);

  virtual Status ParseParam(const std::string& param);

  virtual Status Compress(const std::string& filename, const CodecInOut* io,
                          ThreadPool* pool, PaddedBytes* compressed) = 0;

  virtual Status Decompress(const std::string& filename,
                            const PaddedBytes& compressed, ThreadPool* pool,
                            CodecInOut* io) = 0;

  virtual void GetMoreStats(BenchmarkStats* stats) {}

  virtual Status CanRecompressJpeg() const { return false; }
  virtual Status RecompressJpeg(const std::string& filename,
                                const std::string& data,
                                PaddedBytes* compressed) {
    return false;
  }

 protected:
  const BenchmarkArgs& args_;
  const CodecContext& codec_context_;
  std::string params_;
  std::string description_;
  float butteraugli_target_;
  float bitrate_target_;
  float hf_asymmetry_;
};

using ImageCodecPtr = std::unique_ptr<ImageCodec>;
}  // namespace pik

#endif  // PIK_BENCHMARK_BENCHMARK_H_
