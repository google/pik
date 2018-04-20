#include "af_stats.h"

#include <numeric>

#include "af_solver.h"
#include "arch_specific.h"
#include "simd/simd.h"
#include "status.h"

namespace pik {
namespace {

void CumulativeDistributionFunction(const int* PIK_RESTRICT histogram,
                                    const size_t num_bins,
                                    int* PIK_RESTRICT cdf) {
  using namespace SIMD_NAMESPACE;
#if SIMD_TARGET_VALUE != SIMD_NONE
  const Part<int32_t, 4> d;
  using V = Part<int32_t, 4>::V;
  PIK_CHECK(num_bins % d.N == 0);
  V sum = setzero(d);
  for (size_t idx_bin = 0; idx_bin < num_bins; idx_bin += d.N) {
    const V h3210 = load_unaligned(d, histogram + idx_bin);
    const V h210z = shift_left_bytes<4>(h3210);
    const V h32_21_10_0z = h3210 + h210z;
    const V h10_0z_zz_zz = shift_left_bytes<8>(h32_21_10_0z);
    const V prefix_sum = h32_21_10_0z + h10_0z_zz_zz;
    sum += prefix_sum;
    store(sum, d, cdf + idx_bin);
    // Broadcast total (highest lane) to all lanes.
    sum = broadcast<3>(sum);
  }
#else
  uint64_t sum = 0;
  for (size_t idx_bin = 0; idx_bin < num_bins; ++idx_bin) {
    sum += histogram[idx_bin];
    cdf[idx_bin] = sum;
  }
#endif
}

int IndexWithMaxDensity(const int* cdf, const int min_value,
                        const int half_range) {
  int max_density = 0;
  int index_with_max_density;
  for (int i = min_value; i < min_value + half_range; ++i) {
    const int density = cdf[i + half_range] - cdf[i];
    PIK_CHECK(density >= 0);
    if (density > max_density) {
      max_density = density;
      index_with_max_density = i;
    }
  }

  // All equal (index_with_max_density is uninitialized) => return middle.
  if (max_density == 0) {
    return min_value + half_range;
  }

  return index_with_max_density;
}

int HalfRangeMode(const int* PIK_RESTRICT cdf, const int min_value,
                  const int max_value) {
  const int range = max_value - min_value + 1;
  PIK_CHECK(range >= 0);
  if (range <= 2) {
    return (min_value + max_value + 1) / 2;  // average (with rounding).
  }
  const int half_range = range / 2;
  const int start = IndexWithMaxDensity(cdf, min_value, half_range);
  return HalfRangeMode(cdf, start, start + half_range - 1);
}

template <typename T>
bool IsZero(const T x) {
  return std::abs(x) < 1E-7;
}

}  // namespace

void Histogram::Print() const {
  for (size_t i = 0; i < kBins; ++i) {
    printf("%d\n", bins[i]);
  }
}

// Note: tsc_timer contains an algorithm for mode estimation in sorted data.
int Histogram::Mode() const {
  int cdf[kBins];
  CumulativeDistributionFunction(bins, kBins, cdf);
  return HalfRangeMode(cdf, 0, kBins - 1);
}

double Histogram::Quantile(double q01) const {
  const int64_t total = std::accumulate(bins, bins + kBins, 1LL);
  const int64_t target = static_cast<int64_t>(q01 * total);
  // Until sum >= target:
  int64_t sum = 0;
  size_t i = 0;
  for (; i < kBins; ++i) {
    sum += bins[i];
    // Exact match: assume middle of bin i
    if (sum == target) {
      return i + 0.5;
    }
    if (sum > target) break;
  }

  // Next non-empty bin (in case histogram is sparsely filled)
  size_t next = i + 1;
  while (next < kBins && bins[next] == 0) { ++next; }

  // Linear interpolation according to how far into next we went
  const double excess = target - sum;
  const double weight_next = bins[Index(next)] / excess;
  return ClampX(next * weight_next + i * (1.0 - weight_next));
}

int EstimateGaussianMixture(const Stats& stats, GaussianMixture* mixtures) {
  using C = std::complex<double>;

  const double mu1 = stats.Mu1();
  const double mu2 = stats.Mu2();
  const double mu3 = stats.Mu3();
  const double mu4 = stats.Mu4();

  int num_mixtures = 0;

  // Candidates for t via (46) in "On the dissection of frequency functions".
  const double p = 0.5 * mu4 - 1.5 * mu2 * mu2;
  const double q = 0.5 * mu3 * mu3;
  if (IsZero(p) && IsZero(q)) {
    return 0;  // If both are zero, t = 0 (rejected below).
  }
  C roots[3];
  const int num_roots = Solver::SolveDepressedCubic(p, q, roots);
  for (int i = 0; i < num_roots; ++i) {
    const auto t = roots[i];
    if (IsZero(t)) continue;  // Avoid division by 0.

    // Solve for m, M (the means relative to mu1).
    C means[2];
    Solver::SolveReducedQuadratic(mu3 / t, t, means);
    const double m = means[0].real();
    const double M = means[1].real();

    // Compute weight and variance.
    const double d = M - m;
    if (IsZero(d)) continue;  // Avoid division by 0.
    const double weight = M / d;
    const double variance = t.real() + mu2;
    if (!(0.0 <= weight && weight <= 1.0)) continue;  // Infeasible solution.
    if (variance < 0.0) continue;                     // Infeasible solution.

    // Add solution.
    mixtures[num_mixtures].weight = weight;
    mixtures[num_mixtures].mean0 = m + mu1;
    mixtures[num_mixtures].mean1 = M + mu1;
    mixtures[num_mixtures].var0 = variance;
    mixtures[num_mixtures].var1 = variance;
    // Ensure the convention mean0 < mean.
    if (mixtures[num_mixtures].mean0 > mixtures[num_mixtures].mean1) {
      std::swap(mixtures[num_mixtures].mean0, mixtures[num_mixtures].mean1);
      std::swap(mixtures[num_mixtures].var0, mixtures[num_mixtures].var1);
      mixtures[num_mixtures].weight = 1.0 - mixtures[num_mixtures].weight;
    }

    ++num_mixtures;
  }

  return num_mixtures;
}

}  // namespace pik
