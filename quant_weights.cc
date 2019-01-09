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
      1.2595332457428179,
      -1.407358221562681,
      -0.26214645111800094,
      -0.11783736114426774,
      -2.4767712425299204,
      -1.6466164664560843,
      1.6110585724878885,
      0.7852970261911113,
      //
      0.22591094421515998,
      -0.91614182548373746,
      -0.53690126301256924,
      -0.91361112421096391,
      -0.27803508461821602,
      -0.2298378672564847,
      -1.0836383694961815,
      -2.5965738676373817,
      //
      5.7433154120986986e-05,
      -12.910372564263197,
      -2.5147449257412089,
      -6.6486764161726057,
      -2.0279922199238665,
      4.0818036176137067,
      4.9288300404369201,
      5.6824904097916011,
  };

  const double eccentricity_bands[] = {
      -0.27325753388415697,
      -0.071612958411618327,
      -0.18830522178343412,
      0.37645047902064749,
      //
      0.071337528710177261,
      0.084616727499219946,
      -0.049230497414472441,
      0.097890384530738506,
      //
      -1.9570792348693768,
      -1.9694782085479716,
      -3.7504468889181739,
      0.31450524168066851,
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
      2.4129890698173089,
      -1.7834520255827282,
      0.064215445048806163,
      -19.087723150524248,
      -0.57416033116290721,
      0.75015289457820145,
      //
      0.60179049977047794,
      -0.34917022105071643,
      -1.5730686042302935,
      -0.24207047902255194,
      -0.41922043249172514,
      0.87635984509041842,
      //
      0.14656834396461099,
      -5.0702991139170219,
      1.3313432027292684,
      -11.661300601750384,
      -4.6557354938112878,
      5.034192865635247,
  };

  const double eccentricity_bands[] = {
      0.23722423381474905,
      -0.051367866899926333,
      0.077105438197057868,
      //
      0.20382346591270431,
      -0.24271551910320613,
      0.063375485343062504,
      //
      0.2301926954408183,
      -2.1969226397855444,
      1.7977659953969698,
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
      8.895172794303388,
      -1.0529567126369268,
      -0.3075651670355537,
      -1.0487446824614266,
      0.57935486008094783,
      -3.3289714228537561,
      //
      2.5509616630416296,
      -0.19891399623390349,
      -1.0236753303512214,
      -0.52238344475629062,
      -0.12124179303752852,
      -0.50239682609477632,
      //
      0.58305949076089703,
      -6.7637850194471625,
      1.6051026521297738,
      -1.8514587571808723,
      0.94169186228392576,
      -4.7779717715293852,
  };

  const double eccentricity_bands[] = {
      0.058558182288355631,
      0.0034558346422338126,
      -0.011617976474054686,
      //
      -0.11176382671069321,
      -0.082030527696103056,
      0.27807493717185566,
      //
      -1.1828275861327033,
      -0.10128402549782546,
      1.6775014483488675,
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
      24.714627909941235,
      -0.65018110067351575,
      0.18114122806114086,
      -3.3935433500157233,
      //
      9.4527416441920078,
      0.083738236532008514,
      -1.1018188522869485,
      -1.2448514346208102,
      //
      5.7173316153047704,
      -7.5776040381305547,
      -1.6696643380764691,
      -4.5976313737337122,
  };

  const double eccentricity_bands[] = {
      0.44168515919125861,
      0.59254626138856836,
      //
      -0.14393729206153422,
      0.44582518975263968,
      //
      -1.5556483554586973,
      1.2214624859183765,
  };
  mul01[0] = 0.1885403750310497;
  mul01[1] = 0.25265765123133699;
  mul01[2] = 1.257339497936532;
  mul11[0] = 0.17647209306358702;
  mul11[1] = 0.27463455769449496;
  mul11[2] = 1.0838214107010011;
  constexpr size_t num_bands =
      sizeof(distance_bands) / (sizeof *distance_bands * 3);
  constexpr size_t num_ebands =
      sizeof(eccentricity_bands) / (sizeof *eccentricity_bands * 3);
  return GetQuantWeigts<4, 4, num_bands, num_ebands>(distance_bands,
                                                     eccentricity_bands);
}

}  // namespace pik
