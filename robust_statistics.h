// Copyright 2017 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef ROBUST_STATISTICS_H_
#define ROBUST_STATISTICS_H_

// Robust statistics: Mode, Median, MedianAbsoluteDeviation.

#include <stddef.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include "arch_specific.h"
#include "compiler_specific.h"
#include "status.h"

namespace pik {

// @return i in [idx_begin, idx_begin + half_count) that minimizes
// sorted[i + half_count] - sorted[i].
template <typename T>
size_t MinRange(const T* const PIK_RESTRICT sorted, const size_t idx_begin,
                const size_t half_count) {
  T min_range = std::numeric_limits<T>::max();
  size_t min_idx = 0;

  for (size_t idx = idx_begin; idx < idx_begin + half_count; ++idx) {
    PIK_ASSERT(sorted[idx] <= sorted[idx + half_count]);
    const T range = sorted[idx + half_count] - sorted[idx];
    if (range < min_range) {
      min_range = range;
      min_idx = idx;
    }
  }

  return min_idx;
}

// Round up for integers
template<class T, typename std::enable_if<
    std::numeric_limits<T>::is_integer>::type* = nullptr>
inline T Half(T x)
{
  return (x + 1) / 2;
}

// Mul is faster than div.
template<class T, typename std::enable_if<
    !std::numeric_limits<T>::is_integer>::type* = nullptr>
inline T Half(T x)
{
  return x * 0.5;
}

// Returns an estimate of the mode by calling MinRange on successively
// halved intervals. "sorted" must be in ascending order. This is the
// Half Sample Mode estimator proposed by Bickel in "On a fast, robust
// estimator of the mode", with complexity O(N log N). The mode is less
// affected by outliers in highly-skewed distributions than the median.
// The averaging operation below assumes "T" is an unsigned integer type.
template <typename T>
T Mode(const T* const PIK_RESTRICT sorted, const size_t num_values) {
  size_t idx_begin = 0;
  size_t half_count = num_values / 2;
  while (half_count > 1) {
    idx_begin = MinRange(sorted, idx_begin, half_count);
    half_count >>= 1;
  }

  const T x = sorted[idx_begin + 0];
  if (half_count == 0) {
    return x;
  }
  PIK_ASSERT(half_count == 1);
  const T average = Half(x + sorted[idx_begin + 1]);
  return average;
}

// Sorts integral values in ascending order. About 3x faster than std::sort for
// input distributions with very few unique values.
template <class T>
void CountingSort(T* begin, T* end) {
  // Unique values and their frequency (similar to flat_map).
  using Unique = std::pair<T, int>;
  std::vector<Unique> unique;
  for (const T* p = begin; p != end; ++p) {
    const T value = *p;
    const auto pos =
        std::find_if(unique.begin(), unique.end(),
                     [value](const Unique& u) { return u.first == value; });
    if (pos == unique.end()) {
      unique.push_back(std::make_pair(*p, 1));
    } else {
      ++pos->second;
    }
  }

  // Sort in ascending order of value (pair.first).
  std::sort(unique.begin(), unique.end());

  // Write that many copies of each unique value to the array.
  T* PIK_RESTRICT p = begin;
  for (const auto& value_count : unique) {
    std::fill(p, p + value_count.second, value_count.first);
    p += value_count.second;
  }
  PIK_ASSERT(p == end);
}

// Returns the median value. Side effect: values <= median will appear before,
// values >= median after the middle index.
// Guarantees average speed O(num_values).
template <typename T>
T Median(T* samples, const size_t num_samples) {
  PIK_ASSERT(num_samples != 0);
  std::nth_element(samples, samples + num_samples / 2, samples + num_samples);
  T result = samples[num_samples / 2];
  // If even size, find largest element in the partially sorted vector to
  // use as second element to average with
  if ((num_samples & 1) == 0) {
    T biggest = *std::max_element(samples, samples + num_samples / 2);
    result = Half(result + biggest);
  }
  return result;
}

template <typename T>
T Median(std::vector<T>* samples) {
  return Median(samples->data(), samples->size());
}

// Returns a robust measure of variability.
template <typename T>
T MedianAbsoluteDeviation(const T* samples, const size_t num_samples,
                          const T median) {
  PIK_ASSERT(num_samples != 0);
  std::vector<T> abs_deviations;
  abs_deviations.reserve(num_samples);
  for (size_t i = 0; i < num_samples; ++i) {
    abs_deviations.push_back(std::abs(samples[i] - median));
  }
  return Median(&abs_deviations);
}

template <typename T>
T MedianAbsoluteDeviation(const std::vector<T>& samples, const T median) {
  return MedianAbsoluteDeviation(samples.data(), samples.size(), median);
}

}  // namespace pik

#endif  // ROBUST_STATISTICS_H_
