// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.
#include "benchmark/benchmark_xl.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <memory>
#include <random>
#include <vector>

#include "pik/arch_specific.h"
#include "benchmark/benchmark_codec_pik.h"
#include "benchmark/benchmark_codec_png.h"
#include "benchmark/benchmark_file_io.h"
#include "pik/butteraugli/butteraugli.h"
#include "pik/butteraugli_distance.h"
#include "pik/codec.h"
#include "pik/codec_png.h"
#include "pik/data_parallel.h"
#include "pik/image.h"
#include "pik/os_specific.h"
#include "pik/padded_bytes.h"
#include "pik/profiler.h"
#include "pik/status.h"
#include "pik/tsc_timer.h"
#include "pik/yuv_opsin_convert.h"

namespace pik {

BenchmarkArgs* const args = new BenchmarkArgs();

Status BenchmarkArgs::AddCommandLineOptions() {
  AddString(&input, "input", "File or file pattern matching input files.");
  AddString(&codec, "codec",
            "Comma separated list of image codec descriptions to benchmark.",
            "pik");
  AddFlag(&print_details, "print_details",
          "Prints size and distortion for each image. Not safe for "
          "concurrent benchmark runs.",
          false);
  AddFlag(
      &print_more_stats, "print_more_stats",
      "Prints codec-specific stats. Not safe for concurrent benchmark runs.",
      false);
  AddFlag(&print_distance_percentiles, "print_distance_percentiles",
          "Prints distance percentiles for the corpus. Not safe for "
          "concurrent benchmark runs.",
          false);
  AddFlag(&save_compressed, "save_compressed",
          "Saves the compressed files for each input image and each codec.",
          false);
  AddFlag(&save_decompressed, "save_decompressed",
          "Saves the decompressed files as PNG for each input image "
          "and each codec.",
          false);

  AddDouble(&mul_output, "mul_output",
            "If nonzero, multiplies linear sRGB by this and clamps to 255",
            0.0);
  AddDouble(&heatmap_good, "heatmap_good",
            "If greater than zero, use this as the good "
            "threshold for creating heatmap images.",
            0.0);
  AddDouble(&heatmap_bad, "heatmap_bad",
            "If greater than zero, use this as the bad "
            "threshold for creating heatmap images.",
            0.0);

  AddFlag(&write_html_report, "write_html_report",
          "Creates an html report with original and compressed images.", false);

  AddString(&originals_url, "originals_url",
            "Url prefix to serve original images from in the html report.");
  AddString(&output_dir, "output_dir",
            "If not empty, save compressed and decompressed "
            "images here.");

  AddSigned(&num_threads, "num_threads",
            "The number of threads for concurrent benchmarking. Defaults to "
            "1 thread per CPU core (if negative).",
            -1);
  AddSigned(&inner_threads, "inner_threads",
            "The number of extra threads per task. "
            "Defaults to occupy cores (if negative).",
            -1);

  AddString(&sample_tmp_dir, "sample_tmp_dir",
            "Directory to put samples from input images.");

  AddSigned(&num_samples, "num_samples", "How many sample areas to take.", 0);
  AddSigned(&sample_dimensions, "sample_dimensions",
            "How big areas to sample from the input.", 64);

  AddDouble(&error_pnorm, "error_pnorm",
            "p norm for collecting butteraugli values", 7.0);

  AddDouble(&hf_asymmetry, "hf_asymmetry",
            "Multiplier for weighting HF artefacts more than features "
            "being smoothed out. 1.0 means no HF asymmetry. 0.3 is "
            "a good value to start exploring for asymmetry.",
            0.8);
  AddFlag(&profiler, "profiler", "If true, print profiler results.", true);

  AddFlag(&show_progress, "show_progress",
          "Show activity dots per completed file during benchmark.", false);

  if (!AddCommandLineOptionsPikCodec(this)) return false;
  if (!AddCommandLineOptionsPNGCodec(this)) return false;

  return true;
}

CodecContext codec_context;

std::string StringPrintf(const char* format, ...) {
  char buf[2000];
  va_list args;
  va_start(args, format);
  vsnprintf(buf, sizeof(buf), format, args);
  va_end(args);
  return std::string(buf);
}

std::vector<std::string> SplitString(const std::string& s, char c) {
  std::vector<std::string> result;
  size_t pos = 0;
  for (size_t i = 0; i <= s.size(); i++) {
    if (i == s.size() || s[i] == c) {
      result.push_back(s.substr(pos, i - pos));
      pos = i + 1;
    }
  }
  return result;
}

// Computes longest codec name from args->codec, for table alignment.
int ComputeLargestCodecName() {
  std::vector<std::string> methods = SplitString(args->codec, ',');
  size_t max = strlen("Aggregate:");  // Include final row's name
  for (const auto& method : methods) {
    max = std::max(max, method.size());
  }
  return max;
}

// The benchmark result is a table of heterogeneous data, the column type
// specifies its data type.
enum ColumnType {
  // Formatted string
  TYPE_STRING,
  // Positive size
  TYPE_SIZE,
  // Floating point value (double precision) which is interpreted as
  // "not applicable" if <= 0, must be strictly positive to be valid but can be
  // set to 0 or negative to be printed as "---", for example for a speed that
  // is not measured.
  TYPE_POSITIVE_FLOAT,
};

struct ColumnDescriptor {
  // Column name, printed across two lines.
  std::string label[2];
  // Total width to render the values of this column. If t his is a floating
  // point value, make sure this is large enough to contain a space and the
  // point, plus precision digits after the point, plus the max amount of
  // integer digits you expect in front of the point.
  int width;
  // Amount of digits after the point, or 0 if not a floating point value.
  int precision;
  ColumnType type;
};

// To add or change a column to the benchmark ASCII table output, add/change
// an entry here with table header line 1, table header line 2, width of the
// column, precision after the point in case of floating point, and the
// data type. Then add/change the corresponding formula or formatting in
// the function ComputeColumns.
const std::vector<ColumnDescriptor>& GetColumnDescriptors() {
  // clang-format off
  static const std::vector<ColumnDescriptor> result = {
      {{"Compr", "Method"}, ComputeLargestCodecName() + 1, 0, TYPE_STRING},
      {{"Input", "Pixels"},         13,  0, TYPE_SIZE},
      {{"Compr", "Size"},            9,  0, TYPE_SIZE},
      {{"Compr", "BPP"},            17, 11, TYPE_POSITIVE_FLOAT},
      {{"", "#"},                    4,  0, TYPE_STRING},
      {{"Compr", "MB/s"},            8,  3, TYPE_POSITIVE_FLOAT},
      {{"Decomp", "MB/s"},           8,  3, TYPE_POSITIVE_FLOAT},
      {{"Butteraugli", "Distance"}, 13,  8, TYPE_POSITIVE_FLOAT},
      {{"", "Error p norm"},        16, 11, TYPE_POSITIVE_FLOAT},
      {{"", "PSNR"},                 7,  2, TYPE_POSITIVE_FLOAT},
      {{"", "QABPP"},                8,  3, TYPE_POSITIVE_FLOAT},
      {{"", "DCT2"},                 9,  7, TYPE_POSITIVE_FLOAT},
      {{"", "DCT4"},                 9,  7, TYPE_POSITIVE_FLOAT},
      {{"", "DCT16"},                9,  7, TYPE_POSITIVE_FLOAT},
      {{"", "DCT32"},                9,  7, TYPE_POSITIVE_FLOAT},
      {{"", "Entropy"},             12,  3, TYPE_POSITIVE_FLOAT},
      {{"", "BPP*pnorm"},           20, 16, TYPE_POSITIVE_FLOAT},
  };
  // clang-format on

  return result;
}

BenchmarkStats::BenchmarkStats() {
  num_inputs = 0;
  total_input_pixels = 0;
  total_uncompressed_input_size = 0;
  total_compressed_size = 0;
  total_adj_compressed_size = 0;
  total_time_encode = 0.0;
  total_time_decode = 0.0;
  max_distance = -1.0;
  distance_p_norm = 0;
  distance_2 = 0;
  pik_stats = PikStats();
}

void BenchmarkStats::Assimilate(const BenchmarkStats& victim) {
  num_inputs += victim.num_inputs;
  total_input_pixels += victim.total_input_pixels;
  total_uncompressed_input_size += victim.total_uncompressed_input_size;
  total_compressed_size += victim.total_compressed_size;
  total_adj_compressed_size += victim.total_adj_compressed_size;
  total_time_encode += victim.total_time_encode;
  total_time_decode += victim.total_time_decode;
  max_distance = std::max(max_distance, victim.max_distance);
  distance_p_norm += victim.distance_p_norm;
  distance_2 += victim.distance_2;
  distances.insert(distances.end(), victim.distances.begin(),
                   victim.distances.end());
  pik_stats.Assimilate(victim.pik_stats);
}

static float ComputeSpeed(float size_MB, float time_s) {
  if (time_s == 0.0) return 0;
  return size_MB / time_s;
}

static std::string FormatFloat(const ColumnDescriptor& label, double value) {
  std::string result =
      StringPrintf("%*.*f", label.width - 1, label.precision, value);

  // Reduce precision if the value is too wide for the column. However, keep
  // at least one digit to the right of the point, and especially the integer
  // digits.
  if (result.size() >= label.width) {
    size_t point = result.rfind('.');
    if (point != std::string::npos) {
      int end = std::max<int>(point + 2, label.width - 1);
      result = result.substr(0, end);
    }
  }
  return result;
}

static std::string PrintHeader() {
  std::string out;
  const auto& descriptors = GetColumnDescriptors();
  for (int row = 0; row < 2; row++) {
    for (int i = 0; i < descriptors.size(); i++) {
      const std::string& label = descriptors[i].label[row];
      int numspaces = descriptors[i].width - label.size();
      // All except the first one are right-aligned.
      if (i == 0) out += label.c_str();
      out += std::string(numspaces, ' ');
      if (i != 0) out += label.c_str();
    }
    out += '\n';
  }
  for (int i = 0; i < descriptors.size(); i++) {
    out += std::string(descriptors[i].width, '-');
  }
  return out + "\n";
}

std::vector<ColumnValue> BenchmarkStats::ComputeColumns(
    const std::string& codec_desc, size_t corpus_size,
    size_t num_threads) const {
  PIK_CHECK(0 == num_inputs % corpus_size);
  PIK_CHECK(num_inputs > 0);
  size_t num_iters = num_inputs / corpus_size;
  PIK_CHECK(0 == total_input_pixels % num_iters);
  PIK_CHECK(0 == total_uncompressed_input_size % num_iters);
  PIK_CHECK(0 == total_compressed_size % num_iters);
  uint64_t input_pixels = total_input_pixels / num_iters;
  float comp_bpp = total_compressed_size * 8.0 / total_input_pixels;
  float adj_comp_bpp = total_adj_compressed_size * 8.0 / total_input_pixels;
  // NOTE: assumes RGB, no alpha.
  float uncompressed_size_MB = total_uncompressed_input_size * 1E-6f;
  float compression_speed =
      ComputeSpeed(uncompressed_size_MB, total_time_encode);
  float decompression_speed =
      ComputeSpeed(uncompressed_size_MB, total_time_decode);
  // Already weighted, no need to divide by #channels.
  const double rmse = std::sqrt(distance_2 / total_input_pixels);
  double psnr = total_compressed_size == 0
                    ? 0.0
                    : (distance_2 == 0) ? 99.99 : (20 * std::log10(255 / rmse));
  double p_norm =
      std::pow(distance_p_norm / total_input_pixels, 1.0 / args->error_pnorm);
  size_t compressed_size = total_compressed_size / num_iters;

  double bpp_p_norm = p_norm * comp_bpp;

  std::vector<ColumnValue> values(GetColumnDescriptors().size());

  values[0].s = codec_desc;
  values[1].i = input_pixels;
  values[2].i = compressed_size;
  values[3].f = comp_bpp;
  values[4].s = num_threads <= 1
                    ? StringPrintf("%zd", num_iters)
                    : StringPrintf("%zd/%zu", num_iters, num_threads);
  values[5].f = compression_speed;
  values[6].f = decompression_speed;
  values[7].f = max_distance;
  values[8].f = p_norm;
  values[9].f = psnr;
  values[10].f = adj_comp_bpp;
  // The DCT 4x4 is applied to an 8x8 block by having 2x2 DCT4x4s,
  // thus we need to multiply the block count by 8.0 * 8.0 pixels.
  // Same for DCT 2x2.
  values[11].f = pik_stats.info.num_dct2_blocks * 8.0 * 8.0 / input_pixels;
  values[12].f = pik_stats.info.num_dct4_blocks * 8.0 * 8.0 / input_pixels;
  values[13].f = pik_stats.info.num_dct16_blocks * 16.0 * 16.0 / input_pixels;
  values[14].f = pik_stats.info.num_dct32_blocks * 32.0 * 32.0 / input_pixels;
  values[15].f = pik_stats.info.entropy_estimate;
  values[16].f = bpp_p_norm;

  return values;
}

static std::string PrintFormattedEntries(
    const std::vector<ColumnValue>& values) {
  const auto& descriptors = GetColumnDescriptors();

  std::string out;
  for (int i = 0; i < descriptors.size(); i++) {
    std::string value;
    if (descriptors[i].type == TYPE_STRING) {
      value = values[i].s;
    } else if (descriptors[i].type == TYPE_SIZE) {
      value = values[i].i ? StringPrintf("%zd", values[i].i) : "---";
    } else if (descriptors[i].type == TYPE_POSITIVE_FLOAT) {
      value = FormatFloat(descriptors[i], values[i].f);
      value = FormatFloat(descriptors[i], values[i].f);
    }

    int numspaces = descriptors[i].width - value.size();
    if (numspaces < 1) {
      numspaces = 1;
    }
    // All except the first one are right-aligned, the first one is the name,
    // others are numbers with digits matching from the right.
    if (i == 0) out += value.c_str();
    out += std::string(numspaces, ' ');
    if (i != 0) out += value.c_str();
  }
  return out + "\n";
}

std::string BenchmarkStats::PrintLine(const std::string& codec_desc,
    size_t corpus_size, size_t num_threads) const {
  std::vector<ColumnValue> values =
      ComputeColumns(codec_desc, corpus_size, num_threads);
  return PrintFormattedEntries(values);
}

// Given the rows of all printed statistics, print an aggregate row.
static std::string PrintAggregate(
    const std::vector<std::vector<ColumnValue>>& aggregate) {
  const auto& descriptors = GetColumnDescriptors();

  for (size_t i = 0; i < aggregate.size(); i++) {
    // Check when statistics has wrong amount of column entries
    PIK_CHECK(aggregate[i].size() == descriptors.size());
  }

  std::vector<ColumnValue> result(descriptors.size());

  // Statistics for the aggregate row are combined together with different
  // formulas than Assimilate uses for combining the statistics of files.
  for (size_t i = 0; i < descriptors.size(); i++) {
    if (descriptors[i].type == TYPE_STRING) {
      // "---" for the Iters column since this does not have meaning for
      // the aggregate stats.
      result[i].s = i == 0 ? "Aggregate:" : "---";
      continue;
    }

    ColumnType type = descriptors[i].type;

    double logsum = 0;
    size_t numvalid = 0;
    for (size_t j = 0; j < aggregate.size(); j++) {
      double value =
          (type == TYPE_SIZE) ? aggregate[j][i].i : aggregate[j][i].f;
      if (value > 0) {
        numvalid++;
        logsum += std::log2(value);
      }
    }
    double geomean = numvalid ? std::exp2(logsum / numvalid) : 0.0;

    if (type == TYPE_SIZE) {
      result[i].i = static_cast<size_t>(geomean + 0.5);
    } else if (type == TYPE_POSITIVE_FLOAT) {
      result[i].f = geomean;
    } else {
      PIK_ABORT("unknown entry type");
    }
  }

  return PrintFormattedEntries(result);
}

void BenchmarkStats::PrintMoreStats() const {
  if (args->print_more_stats) {
    pik_stats.Print();
  }
  if (args->print_distance_percentiles) {
    std::vector<float> sorted = distances;
    std::sort(sorted.begin(), sorted.end());
    int p50idx = 0.5 * distances.size();
    int p90idx = 0.9 * distances.size();
    printf("50th/90th percentile distance: %.8f  %.8f\n", sorted[p50idx],
           sorted[p90idx]);
  }
}

// Creates an image codec by name, e.g. "pik" to get a new instance of the
// pik codec. Optionally, behind a colon, parameters can be specified,
// then ParseParameters of the codec gets called with the part behind the colon.
ImageCodecPtr CreateImageCodec(const std::string& description);

int ParseIntParam(const std::string& param, int lower_bound, int upper_bound) {
  int val = strtol(param.substr(1).c_str(), nullptr, 10);
  PIK_CHECK(val >= lower_bound && val <= upper_bound);
  return val;
}

void ImageCodec::ParseParameters(const std::string& parameters) {
  params_ = parameters;
  std::vector<std::string> parts = SplitString(parameters, ':');
  for (int i = 0; i < parts.size(); ++i) {
    if (!ParseParam(parts[i])) {
      PIK_ABORT("Invalid parameter %s", parts[i].c_str());
    }
  }
}

Status ImageCodec::ParseParam(const std::string& param) {
  if (param[0] == 'd') {
    const std::string distance_param = param.substr(1);
    char* end;
    const float butteraugli_target = strtof(distance_param.c_str(), &end);
    if (end == distance_param.c_str() ||
        end != distance_param.c_str() + distance_param.size()) {
      return false;
    }
    butteraugli_target_ = butteraugli_target;

    // full hf asymmetry at high distance
    static const double kHighDistance = 2.5;

    // no hf asymmetry at low distance
    static const double kLowDistance = 0.6;

    if (butteraugli_target_ >= kHighDistance) {
      hf_asymmetry_ = args_.hf_asymmetry;
    } else if (butteraugli_target_ >= kLowDistance) {
      float w = (butteraugli_target_ - kLowDistance) /
                (kHighDistance - kLowDistance);
      hf_asymmetry_ = args_.hf_asymmetry * w + 1.0f * (1.0f - w);
    } else {
      hf_asymmetry_ = 1.0f;
    }
    return true;
  } else if (param[0] == 'r') {
    butteraugli_target_ = -1.0;
    hf_asymmetry_ = args_.hf_asymmetry;
    bitrate_target_ = strtof(param.substr(1).c_str(), nullptr);
    return true;
  }
  return false;
}

// Low-overhead "codec" for measuring benchmark overhead.
class NoneCodec : public ImageCodec {
 public:
  NoneCodec(const BenchmarkArgs& args, const CodecContext& codec_context)
      : ImageCodec(args, codec_context) {}
  Status ParseParam(const std::string& param) override { return true; }

  Status Compress(const std::string& filename, const CodecInOut* io,
                  ThreadPool* pool, PaddedBytes* compressed) override {
    PROFILER_ZONE("NoneCompress");
    // Encode image size so we "decompress" something of the same size, as
    // required by butteraugli.
    const uint32_t xsize = io->xsize();
    const uint32_t ysize = io->ysize();
    compressed->resize(8);
    memcpy(compressed->data(), &xsize, 4);
    memcpy(compressed->data() + 4, &ysize, 4);
    return true;
  }

  Status Decompress(const std::string& filename, const PaddedBytes& compressed,
                    ThreadPool* pool, CodecInOut* io) override {
    PROFILER_ZONE("NoneDecompress");
    PIK_ASSERT(compressed.size() == 8);
    uint32_t xsize, ysize;
    memcpy(&xsize, compressed.data(), 4);
    memcpy(&ysize, compressed.data() + 4, 4);
    Image3F image(xsize, ysize);
    ZeroFillImage(&image);
    io->SetFromImage(std::move(image), io->Context()->c_srgb[0]);
    return true;
  }

  void GetMoreStats(BenchmarkStats* stats) override {}
};

SIMD_ATTR double ComputeDistanceP(const ImageF& distmap) {
  PROFILER_FUNC;
  const double p = args->error_pnorm;
  if (std::abs(p - 7.0) < 1E-6) {
    const SIMD_FULL(double) dd;
    const SIMD_PART(float, dd.N) df;
    const SIMD_PART(double, 1) d1;
    auto sums = setzero(dd);
    double sum1 = 0.0;
    for (size_t y = 0; y < distmap.ysize(); ++y) {
      const float* PIK_RESTRICT row = distmap.ConstRow(y);
      size_t x = 0;
      for (; x + df.N <= distmap.xsize(); x += df.N) {
        const auto d1 = convert_to(dd, load(df, row + x));
        const auto d2 = d1 * d1;
        const auto d3 = d2 * d1;
        const auto d4 = d2 * d2;
        sums += d3 * d4;
      }
      for (; x < distmap.xsize(); ++x) {
        const double d1 = row[x];
        const double d2 = d1 * d1;
        const double d3 = d2 * d1;
        const double d4 = d2 * d2;
        sum1 += d3 * d4;
      }
    }
    return sum1 + get_part(d1, ext::sum_of_lanes(sums));
  } else {
    static std::atomic<int> once{0};
    if (once.fetch_add(1, std::memory_order_relaxed) == 0) {
      fprintf(stderr, "WARNING: using slow ComputeDistanceP\n");
    }
    double result = 0;
    for (size_t y = 0; y < distmap.ysize(); ++y) {
      const float* PIK_RESTRICT row = distmap.ConstRow(y);
      for (size_t x = 0; x < distmap.xsize(); ++x) {
        double val = std::pow(row[x], p);
        result += val;
      }
    }
    return result;
  }
}

// TODO(lode): take alpha into account when needed
SIMD_ATTR double ComputeDistance2(const CodecInOut* io1,
                                  const CodecInOut* io2) {
  PROFILER_FUNC;
  const CodecContext* codec_context = io1->Context();
  // Convert to sRGB - closer to perception than linear.
  const Image3F* srgb1 = &io1->color();
  Image3F copy1;
  if (!io1->IsSRGB()) {
    PIK_CHECK(io1->CopyTo(Rect(io1->color()),
                          codec_context->c_srgb[io1->IsGray()], &copy1));
    srgb1 = &copy1;
  }
  const Image3F* srgb2 = &io2->color();
  Image3F copy2;
  if (!io2->IsSRGB()) {
    PIK_CHECK(io2->CopyTo(Rect(io2->color()),
                          codec_context->c_srgb[io2->IsGray()], &copy2));
    srgb2 = &copy2;
  }

  PIK_CHECK(SameSize(*srgb1, *srgb2));

  SIMD_FULL(float) d;
  SIMD_PART(float, 1) d1;

  auto sums = setzero(d);
  double result = 0;
  // Weighted PSNR as in JPEG-XL: chroma counts 1/8 (they compute on YCbCr).
  // Avoid squaring the weight - 1/64 is too extreme.
  const float weights[3] = {1.0f / 8, 6.0f / 8, 1.0f / 8};
  for (int c = 0; c < 3; ++c) {
    const auto weight = set1(d, weights[c]);
    for (size_t y = 0; y < srgb1->ysize(); ++y) {
      const float* PIK_RESTRICT row1 = srgb1->ConstPlaneRow(c, y);
      const float* PIK_RESTRICT row2 = srgb2->ConstPlaneRow(c, y);
      size_t x = 0;
      for (; x + d.N <= srgb1->xsize(); x += d.N) {
        const auto diff = load(d, row1 + x) - load(d, row2 + x);
        sums += diff * diff * weight;
      }
      for (; x < srgb1->xsize(); ++x) {
        const float diff = row1[x] - row2[x];
        result += diff * diff * weights[c];
      }
    }
  }
  const float sum = get_part(d1, ext::sum_of_lanes(sums));
  return sum + result;
}

Status WritePNG(const Image3B& image, const std::string& filename) {
  CodecContext codec_context;
  ThreadPool pool(4);
  std::vector<uint8_t> rgb(image.xsize() * image.ysize() * 3);
  CodecInOut io(&codec_context);
  io.SetFromImage(StaticCastImage3<float>(image), codec_context.c_srgb[0]);
  PaddedBytes compressed;
  PIK_CHECK(EncodeImagePNG(&io, io.c_current(), 8, &pool, &compressed));
  return WriteFile(compressed, filename);
}

Status ReadPNG(const std::string& filename, Image3B* image) {
  CodecContext codec_context;
  CodecInOut io(&codec_context);
  PIK_CHECK(io.SetFromFile(filename));
  *image = StaticCastImage3<uint8_t>(io.color());
  return true;
}

SIMD_ATTR void DoCompress(const std::string& filename, const CodecInOut& io,
                          ImageCodec* codec, PaddedBytes* compressed,
                          BenchmarkStats* s) {
  PROFILER_FUNC;
  const size_t xsize = io.xsize();
  const size_t ysize = io.ysize();
  const size_t input_pixels = xsize * ysize;

  // This function is already a task called from a ThreadPool but we may have
  // fewer tasks than threads (e.g. a few large images), so allow nested
  // parallelism if requested.
  ThreadPool inner_pool(args->inner_threads);

  double start, end;
  std::string ext = FileExtension(filename);
  if (codec->CanRecompressJpeg() && (ext == ".jpg" || ext == ".jpeg")) {
    std::string data_in;
    PIK_CHECK(ReadFile(filename, &data_in));
    start = Now();
    PIK_CHECK(codec->RecompressJpeg(filename, data_in, compressed));
    end = Now();
  } else {
    start = Now();
    PIK_CHECK(codec->Compress(filename, &io, &inner_pool, compressed));
    end = Now();
  }
  s->total_time_encode += end - start;

  // Decompress
  CodecInOut io2(&codec_context);
  io2.dec_c_original = io.dec_c_original;
  io2.SetOriginalBitsPerSample(io.original_bits_per_sample());
  start = Now();
  PIK_CHECK(codec->Decompress(filename, *compressed, &inner_pool, &io2));
  end = Now();
  s->total_time_decode += end - start;

  // By default the benchmark will save the image after roundtrip with the
  // same color encoding as the image before roundtrip. Not all codecs
  // necessarily preserve the amount of channels (1 for gray, 3 for RGB)
  // though, since not all image formats necessarily allow a way to remember
  // what amount of channels you happened to give the the benchmark codec
  // input (say, an RGB-only format) and that is fine since in the end what
  // matters is that the pixels look the same on a 3-channel RGB monitor
  // while using grayscale encoding is an internal compression optimization.
  // If that is the case, output with the current color model instead,
  // because CodecInOut does not automatically convert between 1 or 3
  // channels, and giving a ColorEncoding  with a different amount of
  // channels is not allowed.
  const ColorEncoding& c_desired =
      (io2.dec_c_original.Channels() == io2.c_current().Channels())
          ? io2.dec_c_original
          : io2.c_current();

  // Verify output
  // TODO(robryk): Reenable once we reproduce the same alpha bitdepth.
  // Currently alpha equality is sort-of included in Butteraugli distance.
#if 0
  PIK_CHECK(io.HasAlpha() == io2.HasAlpha());
  if (io.HasAlpha()) {
    PIK_CHECK(SamePixels(io.alpha(), io2.alpha()));
  }
#endif

  ImageF distmap;
  float distance;
  if (SameSize(io.color(), io2.color())) {
    distance = ButteraugliDistance(&io, &io2, codec->hf_asymmetry(), &distmap,
                                   &inner_pool);
    // Ensure pixels in range 0-255
    s->distance_2 += ComputeDistance2(&io, &io2);
  } else {
    // TODO(veluca): re-upsample and compute proper distance.
    distance = 1e+4f;
    distmap = ImageF(1, 1);
    distmap.Row(0)[0] = distance;
    s->distance_2 += distance;
  }
  // Update stats
  s->distance_p_norm += ComputeDistanceP(distmap);
  s->max_distance = std::max(s->max_distance, distance);
  s->distances.push_back(distance);
  s->total_input_pixels += input_pixels;
  const size_t channels = io.c_current().Channels() + io.HasAlpha();
  const size_t bytes_per_channel =
      DivCeil(io.original_bits_per_sample(), kBitsPerByte);
  s->total_uncompressed_input_size +=
      input_pixels * channels * bytes_per_channel;
  s->total_compressed_size += compressed->size();
  s->total_adj_compressed_size += compressed->size() * std::max(1.0f, distance);
  codec->GetMoreStats(s);
  ++s->num_inputs;
  if (args->print_details) {
    printf("   %-23s %10zd   %6.4f                               %10.8f\n",
           FileBaseName(filename).c_str(), compressed->size(),
           compressed->size() * 8.0 / input_pixels, distance);
    fflush(stdout);  // for run_benchmark
  }

  if (args->save_compressed || args->save_decompressed) {
    std::string dir = FileDirName(filename);
    std::string name = FileBaseName(filename);
    std::string outdir =
        args->output_dir.empty() ? dir + "/out" : args->output_dir.c_str();
    std::string codec_name = codec->description();
    // Make compatible for filename
    std::replace(codec_name.begin(), codec_name.end(), ':', '_');
    std::string compressed_fn = outdir + "/" + name + "." + codec_name;
    std::string decompressed_fn = compressed_fn + ".png";
    std::string heatmap_fn = compressed_fn + ".heatmap.png";
    PIK_CHECK(MakeDir(outdir));
    if (args->save_compressed) {
      std::string compressed_str(
          reinterpret_cast<const char*>(compressed->data()),
          compressed->size());
      PIK_CHECK(WriteFile(compressed_str, compressed_fn));
    }
    if (args->save_decompressed) {
      // For verifying HDR: scale output and clamp.
      if (args->mul_output != 0.0) {
        fprintf(stderr, "WARNING: scaling outputs by %f\n", args->mul_output);
        PIK_CHECK(io2.TransformTo(io2.Context()->c_linear_srgb[io2.IsGray()],
            &inner_pool));
        for (int c = 0; c < 3; ++c) {
          for (size_t y = 0; y < io2.ysize(); ++y) {
            float* row = const_cast<float*>(io2.color().ConstPlaneRow(c, y));
            for (size_t x = 0; x < io2.xsize(); ++x) {
              row[x] = std::min(row[x] * float(args->mul_output), 255.0f);
            }
          }
        }
      }

      PIK_CHECK(io2.EncodeToFile(c_desired, io2.original_bits_per_sample(),
          decompressed_fn));
      float good = args->heatmap_good > 0.0f
                       ? args->heatmap_good
                       : butteraugli::ButteraugliFuzzyInverse(1.5);
      float bad = args->heatmap_bad > 0.0f
                      ? args->heatmap_bad
                      : butteraugli::ButteraugliFuzzyInverse(0.5);
      Image3B heatmap = butteraugli::CreateHeatMapImage(distmap, good, bad);
      PIK_CHECK(WritePNG(heatmap, heatmap_fn));
    }
  }
  if (args->show_progress) {
    printf(".");
    fflush(stdout);
  }
}

class Benchmark {
  using StringVec = std::vector<std::string>;

  struct Task {
    ImageCodecPtr codec;
    size_t idx_image;
    size_t idx_method;
    BenchmarkStats stats;
  };

 public:
  static void Run() {
    {
      PROFILER_FUNC;

      const StringVec methods = GetMethods();
      const StringVec fnames = GetFilenames();

      ThreadPool pool(NumThreads(methods.size() * fnames.size()));
      PinThreads(fnames, &pool);

      const std::vector<CodecInOut> loaded_images = LoadImages(fnames, &pool);

      std::vector<Task> tasks = CreateTasks(methods, fnames);
      RunTasks(fnames, loaded_images, &pool, &tasks);
      PrintStats(methods, fnames, tasks);
    }

    // Must have exited profiler zone above before calling.
    if (args->profiler) {
      PROFILER_PRINT_RESULTS();
    }
    CacheAligned::PrintStats();
  }

 private:
  static size_t NumThreads(const int num_tasks) {
    // TODO(janwas): AvailableCPUs() returns hyperthreads; detect #HT per core.
    const int num_cores = AvailableCPUs().size() / 2;
    PIK_CHECK(num_cores != 0);

    int num_threads = args->num_threads;
    // Default to #cores
    if (num_threads < 0) num_threads = num_cores;

    // No point exceeding #cores - we already max out TDP with 1/core.
    num_threads = std::min(num_threads, num_cores);

    // Don't create more threads than there are tasks (pointless/wasteful).
    num_threads = std::min(num_threads, num_tasks);

    // Default: keep cores busy even if not enough tasks.
    if (args->inner_threads < 0) {
      args->inner_threads =
          num_threads == 0 ? num_cores : num_cores / num_threads;
      // Not enough cores to give each thread/task at least two inner_threads -
      // one inner_thread doesn't help, so don't create any.
      if (args->inner_threads < 2) args->inner_threads = 0;
    }

    fprintf(stderr, "%d threads, %d inner threads, %d tasks, %d cores\n",
            num_threads, args->inner_threads, num_tasks, num_cores);
    return num_threads;
  }

  static void PinThreads(const StringVec& fnames, ThreadPool* pool) {
    PROFILER_FUNC;

    // In single-image mode, we're running concurrent benchmarks, so don't pin.
    if (fnames.size() == 1) return;

    // Only try to pin to unavailable CPUs.
    const std::vector<int> cpus = AvailableCPUs();

    RunOnEachThread(pool, [&cpus](const int task, const int thread) {
      // 1.1-1.2x speedup from pinning.
      if (thread < cpus.size()) {
        if (!PinThreadToCPU(cpus[thread])) {
          fprintf(stderr, "WARNING: failed to pin thread %d.\n", thread);
        }
      }
    });
  }

  static StringVec GetMethods() {
    StringVec methods = SplitString(args->codec, ',');
    return methods;
  }

  static StringVec SampleFromInput(const StringVec& fnames,
                                   const std::string& sample_tmp_dir,
                                   int num_samples, int size) {
    PIK_CHECK(!sample_tmp_dir.empty());
    fprintf(stderr, "Creating samples of %dx%d tiles...\n", size, size);
    StringVec fnames_out;
    std::vector<Image3B> images;
    std::vector<size_t> offsets;
    size_t total_num_tiles = 0;
    for (int i = 0; i < fnames.size(); ++i) {
      Image3B img;
      PIK_CHECK(ReadPNG(fnames[i], &img));
      PIK_CHECK(img.xsize() >= size);
      PIK_CHECK(img.ysize() >= size);
      total_num_tiles += (img.xsize() - size + 1) * (img.ysize() - size + 1);
      offsets.push_back(total_num_tiles);
      images.emplace_back(std::move(img));
    }
    PIK_CHECK(MakeDir(sample_tmp_dir));
    std::mt19937_64 rng;
    for (int i = 0; i < num_samples; ++i) {
      int val = std::uniform_int_distribution<>(0, offsets.back())(rng);
      int idx = (std::lower_bound(offsets.begin(), offsets.end(), val) -
                 offsets.begin());
      PIK_CHECK(idx < images.size());
      const Image3B& img = images[idx];
      int x0 = std::uniform_int_distribution<>(0, img.xsize() - size)(rng);
      int y0 = std::uniform_int_distribution<>(0, img.ysize() - size)(rng);
      Image3B sample(size, size);
      for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < size; ++y) {
          const uint8_t* PIK_RESTRICT row_in = img.PlaneRow(c, y0 + y);
          uint8_t* PIK_RESTRICT row_out = sample.PlaneRow(c, y);
          memcpy(row_out, &row_in[x0], size * sizeof(row_out[0]));
        }
      }
      std::string fn_output = StringPrintf(
          "%s/%s.crop_%dx%d+%d+%d.png", sample_tmp_dir.c_str(),
          FileBaseName(fnames[idx]).c_str(), size, size, x0, y0);
      PIK_CHECK(WritePNG(sample, fn_output));
      fnames_out.push_back(fn_output);
    }
    fprintf(stderr, "Created %d sample tiles\n", num_samples);
    return fnames_out;
  }

  static StringVec GetFilenames() {
    StringVec fnames;
    PIK_CHECK(MatchFiles(args->input, &fnames));
    if (fnames.empty()) {
      fprintf(stderr, "No input file matches pattern: '%s'\n",
              args->input.c_str());
      exit(1);
    }

    if (args->num_samples > 0) {
      fnames = SampleFromInput(fnames, args->sample_tmp_dir, args->num_samples,
                               args->sample_dimensions);
    }
    return fnames;
  }

  // (Load only once, not for every codec)
  static std::vector<CodecInOut> LoadImages(const StringVec& fnames,
                                            ThreadPool* pool) {
    PROFILER_FUNC;
    std::vector<CodecInOut> loaded_images;
    loaded_images.reserve(fnames.size());
    for (size_t i = 0; i < fnames.size(); ++i) {
      loaded_images.emplace_back(&codec_context);
    }
    RunOnPool(pool, 0, fnames.size(),
              [&fnames, &loaded_images](const int task, const int thread) {
                Status ok = loaded_images[task].SetFromFile(fnames[task]);
                if (!ok) PIK_ASSERT(false);
              });
    return loaded_images;
  }

  static std::vector<Task> CreateTasks(const StringVec& methods,
                                       const StringVec& fnames) {
    std::vector<Task> tasks;
    tasks.reserve(methods.size() * fnames.size());
    for (size_t idx_image = 0; idx_image < fnames.size(); ++idx_image) {
      for (size_t idx_method = 0; idx_method < methods.size(); ++idx_method) {
        tasks.push_back(Task());
        Task& t = tasks.back();
        t.codec = CreateImageCodec(methods[idx_method]);
        t.idx_image = idx_image;
        t.idx_method = idx_method;
        // t.stats is default-initialized.
      }
    }
    PIK_ASSERT(tasks.size() == tasks.capacity());
    return tasks;
  }

  static void WriteHtmlReport(const std::string& codec_desc,
                              const StringVec& fnames) {
    const std::string toggle_js = R"(
<script type="text/javascript">
  var counter = [];
  function toggle(i) {
    for (index = counter.length; index <= i; index++) {
      counter.push(0);
    }
    var preview = document.getElementById("preview" + i);
    var orig = document.getElementById("orig" + i);
    var hm = document.getElementById("hm" + i);
    if (counter[i] == 0) {
      preview.style.display = 'none';
      orig.style.display = 'block';
      hm.style.display = 'none';
    } else if (counter[i] == 1) {
      preview.style.display = 'block';
      orig.style.display = 'none';
      hm.style.display = 'none';
    } else if (counter[i] == 2) {
      preview.style.display = 'none';
      orig.style.display = 'none';
      hm.style.display = 'block';
    }
    counter[i] = (counter[i] + 1) % 3;
  }
</script>
)";
    std::string out_html = toggle_js;
    std::string outdir;
    out_html += "<body text=\"$fff\" bgcolor=\"#000\">\n";
    std::string codec_name = codec_desc;
    // Make compatible for filename
    std::replace(codec_name.begin(), codec_name.end(), ':', '_');
    for (int i = 0; i < fnames.size(); ++i) {
      std::string dir = FileDirName(fnames[i]);
      std::string name = FileBaseName(fnames[i]);
      std::string name_out = name + "." + codec_name + ".png";
      std::string orig_url = args->originals_url.empty()
                                 ? ("file://" + fnames[i])
                                 : (args->originals_url + "/" + name);
      std::string heatmap_out = name + "." + codec_name + ".heatmap.png";
      outdir = args->output_dir.empty() ? dir + "/out" : args->output_dir;
      out_html += StringPrintf(
          "<div onclick=\"toggle(%d);\" style=\"display:inline-block\">\n"
          "  <img id=\"preview%d\" src=\"%s\" style=\"display:block;\"/>\n"
          "  <img id=\"orig%d\" src=\"%s\" style=\"display:none;\"/>\n"
          "  <img id=\"hm%d\" src=\"%s\" style=\"display:none;\"/>\n"
          "</div>\n",
          i, i, name_out.c_str(), i, orig_url.c_str(), i, heatmap_out.c_str());
    }
    out_html += "</body>\n";
    PIK_CHECK(WriteFile(out_html, outdir + "/index." + codec_name + ".html"));
  }

  static void RunTasks(const StringVec& fnames,
                       const std::vector<CodecInOut>& loaded_images,
                       ThreadPool* pool, std::vector<Task>* tasks) {
    PROFILER_FUNC;
    RunOnPool(
        pool, 0, tasks->size(),
        [&fnames, &loaded_images, tasks](const int i, const int thread) {
          Task& t = (*tasks)[i];
          PaddedBytes compressed;
          DoCompress(fnames[t.idx_image], loaded_images[t.idx_image],
                     t.codec.get(), &compressed, &t.stats);
        },
        "Benchmark tasks");
    if (args->show_progress) printf("\n");
  }

  static void PrintStats(const StringVec& methods, const StringVec& fnames,
                         const std::vector<Task>& tasks) {
    PROFILER_FUNC;

    // Assimilate all tasks with the same idx_method.
    std::vector<BenchmarkStats> per_method(methods.size());
    for (const Task& t : tasks) {
      per_method[t.idx_method].Assimilate(t.stats);
    }

    std::vector<std::vector<ColumnValue>> stats_aggregate;
    // Note: "print" to std::string so that all relevant output from a
    // single-file benchmark is printed together with the filename, even if
    // run concurrently.
    std::string out;
    if (fnames.size() == 1) out += fnames[0] + "\n";
    out += PrintHeader();
    for (size_t idx_method = 0; idx_method < methods.size(); ++idx_method) {
      const std::string& codec_desc = methods[idx_method];
      const BenchmarkStats& stats = per_method[idx_method];
      stats.PrintMoreStats();  // not concurrent
      out += stats.PrintLine(codec_desc, fnames.size(), /*num_threads=*/1);

      if (args->write_html_report) {
        WriteHtmlReport(codec_desc, fnames);
      }

      stats_aggregate.push_back(
          stats.ComputeColumns(codec_desc, fnames.size(), 1));
    }

    out += PrintAggregate(stats_aggregate);

    printf("%s\n\n", out.c_str());
    fflush(stdout);
  }
};

ImageCodecPtr CreateImageCodec(const std::string& description) {
  std::string name = description;
  std::string parameters = "";
  size_t colon = description.find(':');
  if (colon < description.size()) {
    name = description.substr(0, colon);
    parameters = description.substr(colon + 1);
  }
  ImageCodecPtr result;
  if (name == "pik") {
    result.reset(CreateNewPikCodec(*args, codec_context));
  } else if (name == "png") {
    result.reset(CreateNewPNGCodec(*args, codec_context));
  } else if (name == "none") {
    result.reset(new NoneCodec(*args, codec_context));
  } else {
    PIK_ABORT("Unknown image codec: %s", name.c_str());
  }
  result->set_description(description);
  if (!parameters.empty()) result->ParseParameters(parameters);
  return result;
}

int BenchmarkMain(int argc, char** argv) {
  PIK_CHECK(args->AddCommandLineOptions());
  if (!args->Parse(argc, const_cast<const char**>(argv)) ||
      !args->ValidateArgs()) {
    args->PrintHelp();
    return 1;
  }

  Benchmark::Run();
  return 0;
}

}  // namespace pik

int main(int argc, char** argv) { return pik::BenchmarkMain(argc, argv); }
