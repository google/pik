#include <cstdio>
#include <random>

#include "af_stats.h"
#include "noise.h"
#include "opsin_params.h"
#include "optimize.h"
#include "write_bits.h"

namespace pik {

float ClipMinMax(float x, float min_v, float max_v) {
  x = x < min_v ? min_v : x;
  x = x > max_v ? max_v : x;
  return x;
}

std::vector<float> GetRandomVector(const int rnd_size_w, const int rnd_size_h) {
  const int kMinRndV = 0;
  const int kMaxRndV = 100000;
  const int kSizeFilterW = 5;
  const int kSizeFilterShift = static_cast<int>(kSizeFilterW / 2);
  const int kSizeFilterH = 2;
  const int kSizeFilter = kSizeFilterW * kSizeFilterH +
                          (kSizeFilterW - kSizeFilterShift - 1);
  const int rnd_size = rnd_size_w * rnd_size_h;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(kMinRndV, kMaxRndV);
  std::uniform_int_distribution<> rnd_shift(1, kSizeFilter);
  std::vector<float> rnd_vector(rnd_size);

  for (int n = 0; n < rnd_size; ++n) {
     rnd_vector[n] = dis(gen);
  }

  // Add laplacian-like filter
  for (int x = 0; x < rnd_size_w; ++x) {
    for (int y = 0; y < rnd_size_h; ++y) {
      int shift = rnd_shift(gen);
      int pair_pos_x = x + (shift + kSizeFilterShift) %
                       kSizeFilterW - kSizeFilterShift;
      int pair_pos_y = y + (shift + kSizeFilterShift) / kSizeFilterW;
      int pos_in_rnd = x + y * rnd_size_w;
      if (pair_pos_x < 0 || pair_pos_x >= rnd_size_w ||
          pair_pos_y < 0 || pair_pos_y >= rnd_size_h) {
        rnd_vector[pos_in_rnd] -= dis(gen);
      } else {
        int pair_pos_in_rnd = pair_pos_x + pair_pos_y * rnd_size_w;
        rnd_vector[pos_in_rnd] -= rnd_vector[pair_pos_in_rnd];
      }
      rnd_vector[pos_in_rnd] /= (kMaxRndV - kMinRndV);
    }
  }

  return rnd_vector;
}

float GetScoreSumsOfAbsoluteDifferences(const Image3F& opsin, const int x,
                                        const int y, const int block_size) {
  const int small_bl_size = block_size / 2;
  const int kNumSAD =
      (block_size - small_bl_size) * (block_size - small_bl_size);
  // block_size x block_size reference pixels
  int counter = 0;
  const int offset = 2;

  std::vector<float> sad(kNumSAD, 0);
  for (int y_bl = 0; y_bl + small_bl_size < block_size; ++y_bl) {
    for (int x_bl = 0; x_bl + small_bl_size < block_size; ++x_bl) {
      float sad_sum = 0;
      // size of the center patch, we compare all the patches inside window with
      // the center one
      for (int cy = 0; cy < small_bl_size; ++cy) {
        for (int cx = 0; cx < small_bl_size; ++cx) {
          float wnd = 0.5 * (opsin.PlaneRow(1, y + y_bl + cy)[x + x_bl + cx] +
                             opsin.PlaneRow(0, y + y_bl + cy)[x + x_bl + cx]);
          float center =
              0.5 * (opsin.PlaneRow(1, y + offset + cy)[x + offset + cx] +
                     opsin.PlaneRow(0, y + offset + cy)[x + offset + cx]);
          sad_sum += std::abs(center - wnd);
        }
      }
      sad[counter++] = sad_sum;
    }
  }
  const int kSamples = (kNumSAD) / 2;
  // As with ROAD (rank order absolute distance), we keep the smallest half of
  // the values in SAD (we use here the more robust patch SAD instead of
  // absolute single-pixel differences).
  std::sort(sad.begin(), sad.end());
  const float total_sad_sum =
      std::accumulate(sad.begin(), sad.begin() + kSamples, 0.0f);
  return total_sad_sum / kSamples;
}

std::vector<float> GetSADScoresForPatches(const Image3F& opsin,
                                          const int block_s, const int num_bin,
                                          Histogram* sad_histogram) {
  std::vector<float> sad_scores(
      (opsin.ysize() / block_s) * (opsin.xsize() / block_s), 0.);

  int block_index = 0;

  for (int y = 0; y + block_s <= opsin.ysize(); y += block_s) {
    for (int x = 0; x + block_s <= opsin.xsize(); x += block_s) {
      // We assume that we work with Y opsin channel [-0.5, 0.5]
      float sad_sc = GetScoreSumsOfAbsoluteDifferences(opsin, x, y, block_s);
      sad_scores[block_index++] = sad_sc;
      sad_histogram->Increment(sad_sc * num_bin);
    }
  }
  return sad_scores;
}

float GetSADThreshold(const Histogram& histogram, const int num_bin) {
  // Here we assume that the most patches with similar SAD value is a "flat"
  // patches. However, some images might contain regular texture part and
  // generate second strong peak at the histogram
  // TODO(user) handle bimodale and heavy-tailed case
  const int mode = histogram.Mode();
  return static_cast<float>(mode) / Histogram::kBins;
}

float GetValue(const NoiseParams& noise_params, const float x) {
  const float kMaxNoiseLevel = 1.0f;
  const float kMinNoiseLevel = 0.0f;
  return ClipMinMax(
      noise_params.alpha * std::pow(x, noise_params.gamma) + noise_params.beta,
      kMinNoiseLevel, kMaxNoiseLevel);
}

void AddNoiseToRGB(const float rnd_noise_r, const float rnd_noise_g,
                   const float rnd_noise_cor, const float noise_strength_g,
                   float noise_strength_r, float* x_channel, float* y_channel,
                   float* b_channel) {
  const float kRGCorr = 0.9;
  const float kRGNCorr = 1 - kRGCorr;

  // Add noise
  float red_noise = kRGNCorr * rnd_noise_r * noise_strength_r +
                    kRGCorr * rnd_noise_cor * noise_strength_r;
  float green_noise = kRGNCorr * rnd_noise_g * noise_strength_g +
                      kRGCorr * rnd_noise_cor * noise_strength_g;

  *x_channel += red_noise - green_noise;
  *y_channel += red_noise + green_noise;
  *b_channel += 0.9375 * (red_noise + green_noise);

  *x_channel = ClipMinMax(*x_channel, -kXybRange[0], kXybRange[0]);
  *y_channel = ClipMinMax(*y_channel, -kXybRange[1], kXybRange[1]);
  *b_channel = ClipMinMax(*b_channel, -kXybRange[2], kXybRange[2]);
}

void AddNoise(const NoiseParams& noise_params, Image3F* opsin) {
  if (noise_params.alpha == 0 && noise_params.beta == 0 &&
      noise_params.gamma == 0) {
    return;
  }
  std::vector<float> rnd_noise_red =
      GetRandomVector(opsin->xsize(), opsin->ysize());
  std::vector<float> rnd_noise_green =
      GetRandomVector(opsin->xsize(), opsin->ysize());
  std::vector<float> rnd_noise_correlated =
      GetRandomVector(opsin->xsize(), opsin->ysize());
  const float norm_const = 0.6;
  for (int y = 0; y < opsin->ysize(); ++y) {
    for (int x = 0; x < opsin->xsize(); ++x) {
      auto row = opsin->Row(y);
      float noise_strength_g =
          GetValue(noise_params, ClipMinMax(0.5 * (row[1][x] - row[0][x]),
                                            -kXybCenter[1], kXybCenter[1]) +
                                     kXybCenter[1]);
      float noise_strength_r =
          GetValue(noise_params, ClipMinMax(0.5 * (row[1][x] + row[0][x]),
                                            -kXybCenter[1], kXybCenter[1]) +
                                     kXybCenter[1]);
      float addit_rnd_noise_red =
          rnd_noise_red[y * opsin->xsize() + x] * norm_const;
      float addit_rnd_noise_green =
          rnd_noise_green[y * opsin->xsize() + x] * norm_const;
      float addit_rnd_noise_correlated =
          rnd_noise_correlated[y * opsin->xsize() + x] * norm_const;
      AddNoiseToRGB(addit_rnd_noise_red, addit_rnd_noise_green,
                    addit_rnd_noise_correlated, noise_strength_g,
                    noise_strength_r, &row[0][x], &row[1][x], &row[2][x]);
    }
  }
}

// F(alpha, beta, gamma| x,y) = (1-n) * sum_i(y_i - (alpha x_i ^ gamma +
// beta))^2 + n * alpha * gamma.
struct LossFunction {
  explicit LossFunction(const std::vector<NoiseLevel>& nl0) : nl(nl0) {}

  double Compute(const std::vector<double>& w, std::vector<double>* df) const {
    double loss_function = 0;
    const double kEpsilon = 1e-2;
    const double kRegul = 0.00005;
    (*df)[0] = 0;
    (*df)[1] = 0;
    (*df)[2] = 0;
    for (int ind = 0; ind < nl.size(); ++ind) {
      double shifted_intensity = nl[ind].intensity + kXybCenter[1];
      if (shifted_intensity > kEpsilon) {
        double l_f =
            nl[ind].noise_level - (w[0] * pow(shifted_intensity, w[1]) + w[2]);
        (*df)[0] += (1 - kRegul) * 2.0 * l_f * pow(shifted_intensity, w[1]) +
                    kRegul * w[1];
        (*df)[1] += (1 - kRegul) * 2.0 * l_f * w[0] *
                        pow(shifted_intensity, w[1]) * log(shifted_intensity) +
                    kRegul * w[0];
        (*df)[2] += (1 - kRegul) * 2.0 * l_f;
        loss_function += (1 - kRegul) * l_f * l_f + kRegul * w[0] * w[1];
      }
    }
    return loss_function;
  }

  std::vector<NoiseLevel> nl;
};

void AddPointsForExtrapolation(std::vector<NoiseLevel>* noise_level) {
  NoiseLevel nl_min;
  NoiseLevel nl_max;
  nl_min.noise_level = 2;
  nl_max.noise_level = -2;
  for (auto nl : *noise_level) {
    if (nl.noise_level < nl_min.noise_level) {
      nl_min.intensity = nl.intensity;
      nl_min.noise_level = nl.noise_level;
    }
    if (nl.noise_level > nl_max.noise_level) {
      nl_max.intensity = nl.intensity;
      nl_max.noise_level = nl.noise_level;
    }
  }
  nl_max.intensity = -0.5;
  nl_min.intensity = 0.5;
  noise_level->push_back(nl_min);
  noise_level->push_back(nl_max);
}

void GetNoiseParameter(const Image3F& opsin, NoiseParams* noise_params) {
  // The size of a patch in decoder might be different from encoder's patch
  // size.
  // For encoder: the patch size should be big enough to estimate
  //              noise level, but, at the same time, it should be not too big
  //              to be able to estimate intensity value of the patch
  const int block_s = 8;
  const int kNumBin = 256;
  Histogram sad_histogram;
  std::vector<float> sad_scores =
      GetSADScoresForPatches(opsin, block_s, kNumBin, &sad_histogram);
  float sad_threshold =
      ClipMinMax(GetSADThreshold(sad_histogram, kNumBin), 0.0f, 0.15f);
  std::vector<NoiseLevel> nl =
      GetNoiseLevel(opsin, sad_scores, sad_threshold, block_s);

  AddPointsForExtrapolation(&nl);
  OptimizeNoiseParameters(nl, noise_params);
}

const float kNoisePrecision = 1000.;

void EncodeFloatParam(float val, float precision, size_t* storage_ix,
                      uint8_t* storage) {
  WriteBits(1, val >= 0 ? 1 : 0, storage_ix, storage);
  const int absval_quant = static_cast<int>(std::abs(val) * precision + 0.5f);
  PIK_ASSERT(absval_quant < (1 << 16));
  WriteBits(16, absval_quant, storage_ix, storage);
}

void DecodeFloatParam(float precision, float* val, BitReader* br) {
  const int sign = 2 * br->ReadBits(1) - 1;
  const int absval_quant = br->ReadBits(16);
  *val = sign * absval_quant / precision;
}

std::string EncodeNoise(const NoiseParams& noise_params) {
  const size_t kMaxNoiseSize = 16;
  std::string output(kMaxNoiseSize, 0);
  size_t storage_ix = 0;
  uint8_t* storage = reinterpret_cast<uint8_t*>(&output[0]);
  storage[0] = 0;
  const bool have_noise =
      (noise_params.alpha != 0.0f || noise_params.gamma != 0.0f ||
       noise_params.beta != 0.0f);
  WriteBits(1, have_noise, &storage_ix, storage);
  if (have_noise) {
    EncodeFloatParam(noise_params.alpha, kNoisePrecision, &storage_ix, storage);
    EncodeFloatParam(noise_params.gamma, kNoisePrecision, &storage_ix, storage);
    EncodeFloatParam(noise_params.beta, kNoisePrecision, &storage_ix, storage);
  }
  size_t jump_bits = ((storage_ix + 7) & ~7) - storage_ix;
  WriteBits(jump_bits, 0, &storage_ix, storage);
  PIK_ASSERT(storage_ix % 8 == 0);
  size_t output_size = storage_ix >> 3;
  output.resize(output_size);
  return output;
}

bool DecodeNoise(BitReader* br, NoiseParams* noise_params) {
  const bool have_noise = br->ReadBits(1);
  if (have_noise) {
    DecodeFloatParam(kNoisePrecision, &noise_params->alpha, br);
    DecodeFloatParam(kNoisePrecision, &noise_params->gamma, br);
    DecodeFloatParam(kNoisePrecision, &noise_params->beta, br);
  } else {
    noise_params->alpha = noise_params->gamma = noise_params->beta = 0.0f;
  }
  br->JumpToByteBoundary();
  return true;
}

void OptimizeNoiseParameters(const std::vector<NoiseLevel>& noise_level,
                             NoiseParams* noise_params) {
  static const double kPrecision = 1e-8;
  static const int kMaxIter = 1000;

  LossFunction loss_function(noise_level);
  std::vector<double> parameter_vector(3);
  parameter_vector[0] = -0.05;
  parameter_vector[1] = 2.6;
  parameter_vector[2] = 0.025;

  parameter_vector = optimize::OptimizeWithScaledConjugateGradientMethod(
      loss_function, parameter_vector, kPrecision, kMaxIter);

  noise_params->alpha = parameter_vector[0];
  noise_params->gamma = parameter_vector[1];
  noise_params->beta = parameter_vector[2];
}

std::vector<float> GetTextureStrength(const Image3F& opsin, const int block_s) {
  std::vector<float> texture_strength_index((opsin.ysize() / block_s) *
                                            (opsin.xsize() / block_s));
  int block_index = 0;

  for (int y = 0; y + block_s <= opsin.ysize(); y += block_s) {
    for (int x = 0; x + block_s <= opsin.xsize(); x += block_s) {
      float texture_strength = 0;
      for (int y_bl = 0; y_bl < block_s; ++y_bl) {
        for (int x_bl = 0; x_bl + 1 < block_s; ++x_bl) {
          float diff = opsin.PlaneRow(1, y)[x + x_bl + 1] -
                       opsin.PlaneRow(1, y)[x + x_bl];
          texture_strength += diff * diff;
        }
      }
      for (int y_bl = 0; y_bl + 1 < block_s; ++y_bl) {
        for (int x_bl = 0; x_bl < block_s; ++x_bl) {
          float diff = opsin.PlaneRow(1, y + 1)[x + x_bl] -
                       opsin.PlaneRow(1, y)[x + x_bl];
          texture_strength += diff * diff;
        }
      }
      texture_strength_index[block_index] = texture_strength;
      ++block_index;
    }
  }
  return texture_strength_index;
}

float GetThresholdFlatIndices(const std::vector<float>& texture_strength,
                              const int n_patches) {
  std::vector<float> kth_statistic = texture_strength;
  std::stable_sort(kth_statistic.begin(), kth_statistic.end());
  return kth_statistic[n_patches];
}

std::vector<NoiseLevel> GetNoiseLevel(
    const Image3F& opsin, const std::vector<float>& texture_strength,
    const float threshold, const int block_s) {
  std::vector<NoiseLevel> noise_level_per_intensity;

  const int filt_size = 1;
  static const float kLaplFilter[filt_size * 2 + 1][filt_size * 2 + 1] = {
      {-0.25f, -1.0f, -0.25f},
      {-1.0f, 5.0f, -1.0f},
      {-0.25f, -1.0f, -0.25f},
  };

  // The noise model is build based on channel 0.5 * (X+Y) as we notices that it
  // is similar to the model 0.5 * (Y-X)
  int patch_index = 0;

  for (int y = 0; y + block_s <= opsin.ysize(); y += block_s) {
    for (int x = 0; x + block_s <= opsin.xsize(); x += block_s) {
      if (texture_strength[patch_index] <= threshold) {
        // Calculate mean value
        float mean_int = 0;
        for (int y_bl = 0; y_bl < block_s; ++y_bl) {
          for (int x_bl = 0; x_bl < block_s; ++x_bl) {
            mean_int += 0.5 * (opsin.PlaneRow(1, y + y_bl)[x + x_bl] +
                               opsin.PlaneRow(0, y + y_bl)[x + x_bl]);
          }
        }
        mean_int /= block_s * block_s;

        // Calculate Noise level
        float noise_level = 0;
        int count = 0;
        for (int y_bl = 0; y_bl < block_s; ++y_bl) {
          for (int x_bl = 0; x_bl < block_s; ++x_bl) {
            float filtered_value = 0;
            for (int y_f = -1 * filt_size; y_f <= filt_size; ++y_f) {
              if (((y_bl + y_f) < block_s) && ((y_bl + y_f) >= 0)) {
                for (int x_f = -1 * filt_size; x_f <= filt_size; ++x_f) {
                  if ((x_bl + x_f) >= 0 && (x_bl + x_f) < block_s) {
                    filtered_value +=
                        0.5 *
                        (opsin.PlaneRow(1, y + y_bl + y_f)[x + x_bl + x_f] +
                         opsin.PlaneRow(0, y + y_bl + y_f)[x + x_bl + x_f]) *
                        kLaplFilter[y_f + filt_size][x_f + filt_size];
                  } else {
                    filtered_value +=
                        0.5 *
                        (opsin.PlaneRow(1, y + y_bl + y_f)[x + x_bl - x_f] +
                         opsin.PlaneRow(0, y + y_bl + y_f)[x + x_bl - x_f]) *
                        kLaplFilter[y_f + filt_size][x_f + filt_size];
                  }
                }
              } else {
                for (int x_f = -1 * filt_size; x_f <= filt_size; ++x_f) {
                  if ((x_bl + x_f) >= 0 && (x_bl + x_f) < block_s) {
                    filtered_value +=
                        0.5 *
                        (opsin.PlaneRow(1, y + y_bl - y_f)[x + x_bl + x_f] +
                         opsin.PlaneRow(0, y + y_bl - y_f)[x + x_bl + x_f]) *
                        kLaplFilter[y_f + filt_size][x_f + filt_size];
                  } else {
                    filtered_value +=
                        0.5 *
                        (opsin.PlaneRow(1, y + y_bl - y_f)[x + x_bl - x_f] +
                         opsin.PlaneRow(0, y + y_bl - y_f)[x + x_bl - x_f]) *
                        kLaplFilter[y_f + filt_size][x_f + filt_size];
                  }
                }
              }
            }
            noise_level += std::abs(filtered_value);
            ++count;
          }
        }
        noise_level /= count;
        NoiseLevel nl;
        nl.intensity = mean_int;
        nl.noise_level = noise_level;
        noise_level_per_intensity.push_back(nl);
      }
      ++patch_index;
    }
  }
  return noise_level_per_intensity;
}

}  // namespace pik
