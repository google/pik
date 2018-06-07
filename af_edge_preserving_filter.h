#ifndef AF_EDGE_PRESERVING_FILTER_H_
#define AF_EDGE_PRESERVING_FILTER_H_

#include "image.h"

namespace pik {
namespace epf {

// The "sigma" parameter is the half-width at half-maximum, i.e. the SAD value
// for which the weight is 0.5. It is about 1.2 times the standard deviation of
// a normal distribution. Larger values cause more smoothing.

// All sigma values are pre-shifted by this value to increase their resolution.
// This allows adaptive sigma to compute "5.5" (represented as 22) without an
// additional floating-point multiplication.
static constexpr int kSigmaShift = 2;

// This is the smallest value that avoids 16-bit overflow (see kShiftSAD); it
// corresponds to 1/3 of patch pixels having the minimum integer SAD of 1.
static constexpr int kMinSigma = 4 << kSigmaShift;
// Somewhat arbitrary; determines size of a lookup table.
static constexpr int kMaxSigma = 288 << kSigmaShift;  // 24 per patch pixel

// Number of extra pixels on the top/bottom/left/right edges of the "guide" and
// "in" images relative to "out".
static constexpr int kBorder = 6;  // = Quad radius(2) + reference radius(4)

// Unit test. Call via dispatch::ForeachTarget.
struct EdgePreservingFilterTest {
  template <class Target>
  void operator()();
};

// Call any of these via dispatch::Run.

struct AdaptiveFilterParams {
  // For each block, adaptive sigma := sigma_add +
  // sigma_mul * max_quantization_interval (i.e. reciprocal of min(*_quant)).
  // Note: per-block filter enable is too expensive in size and speed.
  int dc_quant;
  const ImageI* ac_quant;  // not owned
  float sigma_mul;
  float sigma_add;
};

// Adaptive smoothing. "sigma" must be in [kMinSigma, kMaxSigma]. Fills each
// pixel of "out", which must be pre-allocated.
struct EdgePreservingFilter {
  // Self-guided filter. If the min/max input value are already known,
  // passing them in avoids recomputing them.
  template <class Target>
  void operator()(ImageF* in_out, float sigma, float min = 0.0f,
                  float max = 0.0f);
  template <class Target>
  void operator()(Image3F* in_out, float sigma, float min = 0.0f,
                  float max = 0.0f);

  // For PIK: adaptive sigma based on quantization intervals.
  template <class Target>
  void operator()(Image3F* in_out, const AdaptiveFilterParams& params,
                  float min = 0.0f, float max = 0.0f);

  // Low-level version with separate guide image: "in" and "guide" must have
  // kBorder extra pixels on each side.
  template <class Target>
  void operator()(const ImageB& guide, const ImageF& in, float sigma,
                  ImageF* PIK_RESTRICT out);
  template <class Target>
  void operator()(const Image3B& guide, const Image3F& in, float sigma,
                  Image3F* PIK_RESTRICT out);
};

// The following are experimental:

// Same as above, but unoptimized version for comparison.
struct EdgePreservingFilterSlow {
  template <class Target>
  void operator()(ImageF* in_out, float sigma, float min = 0.0f,
                  float max = 0.0f);
  template <class Target>
  void operator()(Image3F* in_out, float sigma, float min = 0.0f,
                  float max = 0.0f);

  template <class Target>
  void operator()(const ImageB& guide, const ImageF& in, float sigma,
                  ImageF* PIK_RESTRICT out);
  template <class Target>
  void operator()(const Image3B& guide, const Image3F& in, float sigma,
                  Image3F* PIK_RESTRICT out);
};

}  // namespace epf
}  // namespace pik

#endif  // AF_EDGE_PRESERVING_FILTER_H_
