// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.
#include "quant_weights.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include "status.h"

namespace pik {

const double* GetQuantWeightsDCT2() {
  static double kQuantWeights[3 * 8 * 8] = {};
  const double w[3 * 6] = {
      // 0
      4550.2400438662207,
      2790.3067417106022,
      1260.6528676551586,
      977.40268487576566,
      303.87618775918281,
      184.8081075700596,
      // 1
      938.04633392441201,
      1068.7966975488603,
      381.83935642524102,
      249.77126290521193,
      79.490566498206576,
      53.610532132277953,
      // 2
      137.60469269062932,
      79.95678364244722,
      18.165359145911914,
      16.621451258020944,
      5.4779381244764922,
      6.6857881312979934,
  };
  for (size_t i = 0; i < sizeof(w) / sizeof *w; i++) {
    PIK_ASSERT(w[i] > 0);
  }
  for (size_t c = 0; c < 3; c++) {
    size_t start = c * 64;
    size_t wstart = c * 6;
    kQuantWeights[start] = 0xBAD;
    kQuantWeights[start + 1] = kQuantWeights[start + 8] = w[wstart];
    kQuantWeights[start + 9] = w[wstart + 1];
    for (size_t y = 0; y < 2; y++) {
      for (size_t x = 0; x < 2; x++) {
        kQuantWeights[start + y * 8 + x + 2] = w[wstart + 2];
        kQuantWeights[start + (y + 2) * 8 + x] = w[wstart + 2];
      }
    }
    for (size_t y = 0; y < 2; y++) {
      for (size_t x = 0; x < 2; x++) {
        kQuantWeights[start + (y + 2) * 8 + x + 2] = w[wstart + 3];
      }
    }
    for (size_t y = 0; y < 4; y++) {
      for (size_t x = 0; x < 4; x++) {
        kQuantWeights[start + y * 8 + x + 4] = w[wstart + 4];
        kQuantWeights[start + (y + 4) * 8 + x] = w[wstart + 4];
      }
    }
    for (size_t y = 0; y < 4; y++) {
      for (size_t x = 0; x < 4; x++) {
        kQuantWeights[start + (y + 4) * 8 + x + 4] = w[wstart + 5];
      }
    }
  }
  return kQuantWeights;
}

const double* GetQuantWeightsIdentity() {
  static double kQuantWeights[3 * 8 * 8] = {};
  for (int i = 0; i < 64; i++) {
    kQuantWeights[i] = 301.30394932821116;
  }
  kQuantWeights[1] = 7081.5558061990732;
  kQuantWeights[8] = kQuantWeights[1];
  kQuantWeights[9] = 4450.0000509661077;
  for (int i = 0; i < 64; i++) {
    kQuantWeights[64 + i] = 86.888251607079184;
  }
  kQuantWeights[64 + 1] = 1468.7565141770958;
  kQuantWeights[64 + 8] = kQuantWeights[64 + 1];
  kQuantWeights[64 + 9] = 1332.1981563633162;
  for (int i = 0; i < 64; i++) {
    kQuantWeights[128 + i] = 2.61335580699784;
  }
  kQuantWeights[128 + 1] = 23.020993061811691;
  kQuantWeights[128 + 8] = kQuantWeights[128 + 1];
  kQuantWeights[128 + 9] = 9.5053232620699593;
  return kQuantWeights;
}

namespace {

// Computes quant weights for a SX*SY-sized transform, using num_bands radial
// bands and num_ebands eccentricity bands. If print_mode is 1, prints the
// resulting matrix; if print_mode is 2, prints the matrix in a format suitable
// for a 3d plot with gnuplot.
template <size_t SX, size_t SY, size_t num_bands, size_t num_ebands,
          size_t print_mode = 0>
const double* GetQuantWeigts(const double distance_bands[num_bands],
                             const double eccentricity_bands[num_ebands]) {
  static double kQuantWeights[SX * SY * 3] = {};
  auto mult = [](double v) {
    if (v > 0) return 1 + v;
    return 1 / (1 - v);
  };

  auto interpolate = [](double pos, double max, double* array, size_t len) {
    double scaled_pos = pos * (len - 1) / max;
    size_t idx = scaled_pos;
    PIK_ASSERT(idx + 1 < len);
    double a = array[idx];
    double b = array[idx + 1];
    return a * pow(b / a, scaled_pos - idx);
  };

  for (size_t c = 0; c < 3; c++) {
    if (print_mode) {
      fprintf(stderr, "Channel %lu\n", c);
    }
    double bands[num_bands] = {distance_bands[c * num_bands]};
    for (size_t i = 1; i < num_bands; i++) {
      bands[i] = bands[i - 1] * mult(distance_bands[c * num_bands + i]);
      PIK_ASSERT(bands[i] > 0);
    }
    double ebands[num_ebands + 1] = {1.0};
    for (size_t i = 1; i <= num_ebands; i++) {
      ebands[i] =
          ebands[i - 1] * mult(eccentricity_bands[c * num_ebands + i - 1]);
      PIK_ASSERT(ebands[i] > 0);
    }
    for (size_t y = 0; y < SY; y++) {
      for (size_t x = 0; x < SX; x++) {
        double dx = 1.0 * x / (SX - 1);
        double dy = 1.0 * y / (SY - 1);
        double distance = std::sqrt(dx * dx + dy * dy);
        double wd =
            interpolate(distance, std::sqrt(2) + 1e-6, bands, num_bands);
        double eccentricity =
            (x == 0 && y == 0) ? 0 : std::abs((double)dx - dy) / distance;
        double we =
            interpolate(eccentricity, 1.0 + 1e-6, ebands, num_ebands + 1);
        double weight = we * wd;

        if (print_mode == 1) {
          fprintf(stderr, "%15.12f, ", weight);
        }
        if (print_mode == 2) {
          fprintf(stderr, "%lu %lu %15.12f\n", x, y, weight);
        }
        kQuantWeights[c * SX * SY + y * SX + x] = weight;
      }
      if (print_mode) fprintf(stderr, "\n");
      if (print_mode == 1) fprintf(stderr, "\n");
    }
    if (print_mode) fprintf(stderr, "\n");
  }
  return kQuantWeights;
}

}  // namespace

const double* GetQuantWeightsDCT32() {
  const double distance_bands[] = {
      0.94750910485230289,
      -1.3979642625613624,
      -0.11147655787296532,
      0.88241659843270581,
      -9.3580880728171838,
      -2.6713397057184132,
      3.3463055123360244,
      1.5876432146495048,
      //
      0.22165267548323492,
      -0.97950274581322783,
      -0.36842931577928406,
      -1.222735274282637,
      -0.19000575164086059,
      -0.36023610911661091,
      -5.2944142704488826,
      -2.597968561590402,
      //
      1.1949141856904151,
      -13.8737359188386,
      -2.661414017692409,
      -7.4696951525219921,
      -18.860033264791454,
      3.1489203841028255,
      -0.1499976751546418,
      3.1377752504563228,
  };

  const double eccentricity_bands[] = {
      -0.32431275763522588,
      0.14892947347409025,
      -0.55627770960663858,
      0.44595641014041676,
      //
      0.072087373012534053,
      0.10886484416213704,
      -0.12565674108338995,
      -0.2682482437559861,
      //
      1.2608741032190469,
      -1.9402333941799501,
      -4.6222271471039944,
      0.11251151675815185,
  };
  constexpr size_t num_bands =
      sizeof(distance_bands) / (sizeof *distance_bands * 3);
  constexpr size_t num_ebands =
      sizeof(eccentricity_bands) / (sizeof *eccentricity_bands * 3);
  return GetQuantWeigts<32, 32, num_bands, num_ebands>(distance_bands,
                                                       eccentricity_bands);
}

const double* GetQuantWeightsDCT16() {
  const double distance_bands[] = {
      2.4213103676363636,
      -1.9661496571849142,
      0.15789683237032592,
      -21.875213745902276,
      -0.57312146615279747,
      0.71563574283814546,
      //
      0.692448552105694,
      -0.51983810999475077,
      -1.3812208666262689,
      -0.47129717191848619,
      -0.4659892465903433,
      0.49714400642991319,
      //
      0.51611293455456542,
      -5.2050261002702385,
      0.85093205981252795,
      -157.81885785236119,
      -4.6550942782414539,
      5.0353879727107831,
  };

  const double eccentricity_bands[] = {
      0.089463427823042674,
      0.25925661835373331,
      -0.21638228954086891,
      //
      0.065273489038446764,
      -0.21577617583309769,
      -0.088954506031061115,
      //
      0.19447329401150901,
      -2.7875463715717927,
      1.0556419025661832,
  };
  constexpr size_t num_bands =
      sizeof(distance_bands) / (sizeof *distance_bands * 3);
  constexpr size_t num_ebands =
      sizeof(eccentricity_bands) / (sizeof *eccentricity_bands * 3);
  return GetQuantWeigts<16, 16, num_bands, num_ebands>(distance_bands,
                                                       eccentricity_bands);
}

const double* GetQuantWeightsDCT8() {
  const double distance_bands[] = {
      8.9391271057027097,
      -1.061576498844176,
      -0.40889955402999378,
      -0.93478700775430934,
      0.74553954779821274,
      -2.3553943958363757,
      //
      1.9937401032859812,
      -0.24317101947310532,
      -0.69913053988110863,
      -0.77134442701221628,
      -0.33608846688684418,
      -0.27271203235543462,
      //
      0.50278210649237409,
      -6.8345267610243674,
      1.7529574859215795,
      -1.9283110790488094,
      0.62124993332784417,
      -4.8269038548388057,
  };

  const double eccentricity_bands[] = {
      0.056994944633443199,
      0.25807550983295885,
      0.078789479873710919,
      //
      -0.015390307111439572,
      -0.037775282774398754,
      0.09485767844955402,
      //
      -1.0397067730495964,
      -0.029185459303206013,
      1.7751115900605332,
  };
  constexpr size_t num_bands =
      sizeof(distance_bands) / (sizeof *distance_bands * 3);
  constexpr size_t num_ebands =
      sizeof(eccentricity_bands) / (sizeof *eccentricity_bands * 3);
  return GetQuantWeigts<8, 8, num_bands, num_ebands>(distance_bands,
                                                     eccentricity_bands);
}

const double* GetQuantWeightsDCT4(double* mul01, double* mul11) {
  const double distance_bands[] = {
    18.965372917648189,
    -0.56383538304134484,
    -7.0973545966358254,
    -4.1667770811237563,
    //
    4.4672573181699757,
    -0.021850755786586087,
    -2.0423267056931866,
    0.056086698064817607,
    //
    3.9327220778337142,
    -22.437465242377993,
    -0.23811905541335746,
    -8.6390739006724662,
  };

  const double eccentricity_bands[] = {
    -0.63097377816480016,
    0.78699078523843091,
    //
    -0.048458745246926782,
    0.29489603761878241,
    //
    -0.63702627091670638,
    -0.026945873546224838,
  };
  mul01[0] = 0.26040825616176849;
  mul01[1] = 0.25265453204973043;
  mul01[2] = 1.4514733256299659;
  mul11[0] = 0.61165425349550706;
  mul11[1] = 0.31532824793319586;
  mul11[2] = 4.4474786589651378;
  constexpr size_t num_bands =
      sizeof(distance_bands) / (sizeof *distance_bands * 3);
  constexpr size_t num_ebands =
      sizeof(eccentricity_bands) / (sizeof *eccentricity_bands * 3);
  return GetQuantWeigts<4, 4, num_bands, num_ebands>(distance_bands,
                                                     eccentricity_bands);
}

}  // namespace pik
