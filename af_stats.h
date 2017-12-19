// For analyzing the range/distribution of scalars.

#ifndef AF_STATS_H_
#define AF_STATS_H_

#include <stddef.h>
#include <stdio.h>
#include <algorithm>
#include <cmath>
#include <complex>

#include "compiler_specific.h"

namespace pik {

// Descriptive statistics of a variable (4 moments).
class Stats {
 public:
  void Notify(const float x) {
    ++n_;

    min_ = std::min(min_, x);
    max_ = std::max(max_, x);

    product_ *= x;

    // Online moments. Reference: https://goo.gl/9ha694
    const double d = x - m1_;
    const double d_div_n = d / n_;
    const double d2n1_div_n = d * (n_ - 1) * d_div_n;
    const long n_poly = n_ * n_ - 3 * n_ + 3;
    m1_ += d_div_n;
    m4_ += d_div_n * (d_div_n * (d2n1_div_n * n_poly + 6.0 * m2_) - 4.0 * m3_);
    m3_ += d_div_n * (d2n1_div_n * (n_ - 2) - 3.0 * m2_);
    m2_ += d2n1_div_n;
  }

  size_t Count() const { return n_; }

  float Min() const { return min_; }
  float Max() const { return max_; }

  double GeometricMean() const {
    return n_ == 0 ? 0.0 : pow(product_, 1.0 / n_);
  }

  double Mean() const { return m1_; }
  // Same as Mu2. Assumes n_ is large.
  double SampleVariance() const { return m2_ / static_cast<int>(n_); }
  // Unbiased estimator for population variance even for smaller n_.
  double Variance() const { return m2_ / static_cast<int>(n_ - 1); }
  double StandardDeviation() const { return sqrt(Variance()); }
  // Near zero for normal distributions; if positive on a unimodal distribution,
  // the right tail is fatter. Assumes n_ is large.
  double SampleSkewness() const { return m3_ * sqrt(n_) / pow(m2_, 1.5); }
  // Corrected for bias (same as Wikipedia and Minitab but not Excel).
  double Skewness() const {
    const double biased = SampleSkewness();
    const double r = (n_ - 1.0) / n_;
    return biased * pow(r, 1.5);
  }
  // Near zero for normal distributions; smaller values indicate fewer/smaller
  // outliers and larger indicates more/larger outliers. Assumes n_ is large.
  double SampleKurtosis() const { return m4_ * n_ / (m2_ * m2_); }
  // Corrected for bias (same as Wikipedia and Minitab but not Excel).
  double Kurtosis() const {
    const double biased = SampleKurtosis();
    const double r = (n_ - 1.0) / n_;
    return biased * r * r;
  }

  // Central moments, useful for "method of moments"-based parameter estimation
  // of a mixture of two Gaussians.
  double Mu1() const { return m1_; }
  double Mu2() const { return m2_ / static_cast<int>(n_); }
  double Mu3() const { return m3_ / static_cast<int>(n_); }
  double Mu4() const { return m4_ / static_cast<int>(n_); }

  void Dump() const {
    printf(
        "Avg %9.6f Min %8.5f Max %8.5f Std %8.5f GeoMean %9.6f "
        "Skew %9.6f Kurt %9.6f\n",
        Mean(), min_, max_, StandardDeviation(), GeometricMean(), Skewness(),
        Kurtosis());
  }

 private:
  size_t n_ = 0;

  float min_ = 1E30f;
  float max_ = -1E30f;

  double product_ = 1.0;

  // Moments
  double m1_ = 0.0;
  double m2_ = 0.0;
  double m3_ = 0.0;
  double m4_ = 0.0;
};

class Histogram {
 public:
  static constexpr int kBins = 256;

  Histogram() { std::fill(bins, bins + kBins, 0); }

  void Increment(const float x) { bins[Index(x)] += 1; }
  int Get(const float x) const { return bins[Index(x)]; }
  int Bin(const size_t bin) const { return bins[bin]; }

  void Print() const;
  int Mode() const;
  double Quantile(double q01) const;
  // Inter-quartile range
  double IQR() const { return Quantile(0.75) - Quantile(0.25); }

 private:
  template <typename T>
  T ClampX(const T x) const {
    return std::min(std::max(T(0), x), T(kBins - 1));
  }
  size_t Index(const float x) const { return ClampX(static_cast<int>(x)); }

  int bins[kBins];
};

struct GaussianMixture {
  double weight;
  double mean0;
  double mean1;
  // variance = sqrt(standard deviation)
  double var0;
  double var1;
};

// Estimates the parameters of a mixture of two Gaussians. Returns number of
// feasible solutions <= 3 written to mixtures.
int EstimateGaussianMixture(const Stats& stats, GaussianMixture* mixtures);

}  // namespace pik

#endif  // AF_STATS_H_
