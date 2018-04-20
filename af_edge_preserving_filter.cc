#include "af_edge_preserving_filter.h"

// Edge-preserving smoothing for denoising and texture/cartoon decomposition.
// Uses 7x8 weighted average based on L1 patch similarity.

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <numeric>  // std::accumulate
#include <mutex>

#define DUMP_SIGMA 0

#include "af_stats.h"
#if DUMP_SIGMA
#include "image_io.h"
#endif
#include "profiler.h"
#include "simd/dispatch.h"
#include "simd_helpers.h"

#ifndef CHECK
#define CHECK(condition)                                            \
  while (!(condition)) {                                            \
    fprintf(stderr, "Check failed at %s:%d\n", __FILE__, __LINE__); \
    abort();                                                        \
  }
#endif

namespace pik {
namespace SIMD_NAMESPACE {
namespace {

using D16 = Full<int16_t>;
using DF = Full<float>;
const D16 d16;
const DF df;
using V16 = D16::V;
using VF = DF::V;

using epf::AdaptiveFilterParams;
using epf::kBorder;
using epf::kMaxSigma;
using epf::kMinSigma;
using epf::kSigmaShift;

//------------------------------------------------------------------------------
// Distance: sum of absolute differences on patches

class Distance {
 public:
  // "Patches" are 3x4 areas with top-left pixel northwest of the reference
  // pixel or its 7x8 neighbors. The 4-pixel width ("quad") is dictated by
  // MPSADBW.
  static constexpr int kPatchArea = 4 * 3;

  static constexpr size_t kNeighbors = 7 * 8;

  // Maximum possible sum of 8-bit differences, used in tests.
  static constexpr int kMaxSAD = kPatchArea * 255;  // = 3060

  static SIMD_INLINE void SumsOfAbsoluteDifferences(
      const uint8_t* SIMD_RESTRICT guide_m4, const size_t guide_stride,
      int16_t* SIMD_RESTRICT sad) {
#if SIMD_TARGET_VALUE == SIMD_NONE
    // 7x8 reference pixels (total search window: 9x11)
    // 56 * 12 * 3 = 2016 ops per pixel, counting abs as one op.
    for (int cy = -3; cy <= 3; ++cy) {
      for (int cx = -3; cx <= 4; ++cx) {
        int sad_sum = 0;
        // 3x4 patch
        for (int iy = -1; iy <= 1; ++iy) {
          const uint8_t* row_ref = guide_m4 + (iy + 4) * guide_stride;
          const uint8_t* row_wnd = guide_m4 + (cy + iy + 4) * guide_stride;
          for (int ix = -1; ix <= 2; ++ix) {
            sad_sum += std::abs(row_ref[ix] - row_wnd[cx + ix]);
          }
        }

        sad[(cy + 3) * 8 + cx + 3] = static_cast<int16_t>(sad_sum);
      }
    }
#elif SIMD_TARGET_VALUE != SIMD_AVX2
    const Part<uint8_t, 16> d8;
    const Part<int16_t, 8> d16;
    const Part<uint32_t, 4> d32;
    const Part<uint64_t, 2> d64;

    // Offset to the leftmost pixel of the search window.
    const int kWindow = -4;  // Starts at row0

    const uint8_t* SIMD_RESTRICT row0 = guide_m4;
    const uint8_t* SIMD_RESTRICT row1 = guide_m4 + 1 * guide_stride;
    const uint8_t* SIMD_RESTRICT row2 = guide_m4 + 2 * guide_stride;
    const uint8_t* SIMD_RESTRICT row3 = guide_m4 + 3 * guide_stride;
    const uint8_t* SIMD_RESTRICT row4 = guide_m4 + 4 * guide_stride;
    const uint8_t* SIMD_RESTRICT row5 = guide_m4 + 5 * guide_stride;
    const uint8_t* SIMD_RESTRICT row6 = guide_m4 + 6 * guide_stride;
    const uint8_t* SIMD_RESTRICT row7 = guide_m4 + 7 * guide_stride;
    const uint8_t* SIMD_RESTRICT row8 = guide_m4 + 8 * guide_stride;

    const uint8_t* ref_pos_t = row3 - 1;

    // "ref" := one four-byte quad from three rows (t/m/b = top/middle/bottom),
    // assembled into 128 bits.
    // Gather would be faster on SKX, but on HSW we reduce port 5 pressure by
    // loading m and b MINUS 4 and 8 bytes to shift those quads upwards.
    // This is safe because we're only shifting m and b => there are valid
    // pixels to load from the previous row. x = don't care/ignored.
    const auto ref_xxT = load_dup128(d8, ref_pos_t);
    const auto ref_xMx = load_dup128(d8, ref_pos_t + guide_stride - 4);
    const auto ref_Bxx = load_dup128(d8, ref_pos_t + 2 * guide_stride - 8);

    // 3 patch rows x 7 window rows (m3 to p3) = 21x 128-bit SAD.
    const auto wnd_p2 = load_unaligned(d8, row6 + kWindow);
    const auto wnd_p3 = load_unaligned(d8, row7 + kWindow);
    const auto wnd_p4 = load_unaligned(d8, row8 + kWindow);

    const auto ref_xMT =
        cast_to(d8, odd_even(cast_to(d32, ref_xMx), cast_to(d32, ref_xxT)));
    const auto ref =
        cast_to(d8, odd_even(cast_to(d64, ref_Bxx), cast_to(d64, ref_xMT)));

    // MPSADBW is 3 uops (p0 + 2p5) and 6 bytes.
    auto sad_6t = ext::mpsadbw<0>(wnd_p2, ref);
    const auto wnd_p0 = load_unaligned(d8, row4 + kWindow);
    const auto wnd_p1 = load_unaligned(d8, row5 + kWindow);

    const auto sad_6m = ext::mpsadbw<1>(wnd_p3, ref);
    const auto wnd_m2 = load_unaligned(d8, row2 + kWindow);

    const auto sad_6b = ext::mpsadbw<2>(wnd_p4, ref);
    // Begin adding together the SAD results from each of the t/m/b rows.
    sad_6t += sad_6m;
    const auto wnd_m1 = load_unaligned(d8, row3 + kWindow);

    auto sad_5t = ext::mpsadbw<0>(wnd_p1, ref);
    auto sad_4m = ext::mpsadbw<1>(wnd_p1, ref);
    sad_6t += sad_6b;
    const auto wnd_m4 = load_unaligned(d8, row0 + kWindow);

    const auto sad_5m = ext::mpsadbw<1>(wnd_p2, ref);
    const auto sad_4b = ext::mpsadbw<2>(wnd_p2, ref);

    const auto sad_5b = ext::mpsadbw<2>(wnd_p3, ref);
    const auto sad_4t = ext::mpsadbw<0>(wnd_p0, ref);

    auto sad_3t = ext::mpsadbw<0>(wnd_m1, ref);
    auto sad_2m = ext::mpsadbw<1>(wnd_m1, ref);
    sad_5t += sad_5m;
    sad_4m += sad_4b;
    const auto wnd_m3 = load_unaligned(d8, row1 + kWindow);

    const auto sad_3m = ext::mpsadbw<1>(wnd_p0, ref);
    const auto sad_2b = ext::mpsadbw<2>(wnd_p0, ref);
    sad_5t += sad_5b;
    sad_4m += sad_4t;

    const auto sad_3b = ext::mpsadbw<2>(wnd_p1, ref);
    const auto sad_2t = ext::mpsadbw<0>(wnd_m2, ref);

    auto sad_1b = ext::mpsadbw<2>(wnd_m1, ref);
    auto sad_0t = ext::mpsadbw<0>(wnd_m4, ref);
    sad_3t += sad_3m;
    sad_2m += sad_2b;

    const auto sad_1t = ext::mpsadbw<0>(wnd_m3, ref);
    const auto sad_0m = ext::mpsadbw<1>(wnd_m3, ref);
    sad_3t += sad_3b;
    sad_2m += sad_2t;

    const auto sad_1m = ext::mpsadbw<1>(wnd_m2, ref);
    const auto sad_0b = ext::mpsadbw<2>(wnd_m2, ref);

    sad_1b += sad_1t;
    sad_0t += sad_0m;
    sad_1b += sad_1m;
    sad_0t += sad_0b;

    store(sad_0t, d16, sad + 0 * d16.N);
    store(sad_1b, d16, sad + 1 * d16.N);
    store(sad_2m, d16, sad + 2 * d16.N);
    store(sad_3t, d16, sad + 3 * d16.N);
    store(sad_4m, d16, sad + 4 * d16.N);
    store(sad_5t, d16, sad + 5 * d16.N);
    store(sad_6t, d16, sad + 6 * d16.N);
#else   // AVX2
    const Full<uint8_t> d8;
    const Full<uint32_t> d32;
    const Full<uint64_t> d64;

    // Leftmost pixel of the search window and reference patch.
    const uint8_t* SIMD_RESTRICT wnd_pos_m4 = guide_m4 - 4;
    const uint8_t* SIMD_RESTRICT ref_pos_m1 = guide_m4 + 3 * guide_stride - 1;
    const size_t gbpr2 = 2 * guide_stride;
    const size_t gbpr4 = 4 * guide_stride;

    // "ref" := one four-byte quad from three rows (t/m/b = top/middle/bottom),
    // assembled into 128 bits, which are duplicated for use by SAD (its
    // arguments select which two quads/rows to use).
    // Gather would be faster on SKX, but on HSW we reduce port 5 pressure by
    // loading m and b MINUS 4 and 8 bytes to shift those quads upwards.
    // This is safe because we're only shifting m and b => there are valid
    // pixels to load from the previous row. x = don't care/ignored.
    const auto ref_xxT = load_dup128(d8, ref_pos_m1);
    const auto ref_xMx = load_dup128(d8, ref_pos_m1 + guide_stride - 4);
    const auto ref_Bxx = load_dup128(d8, ref_pos_m1 + gbpr2 - 8);

    // 3 patch rows x 7 window rows (m3 to p3) = 21x 128-bit SAD = 9 + 3 SAD(),
    // which requires windows to be duplicated into both 128-bit lanes.

    // SAD 10
    const auto ref_xMT =
        cast_to(d8, odd_even(cast_to(d32, ref_xMx), cast_to(d32, ref_xxT)));
    const auto wnd_m3 = load_dup128(d8, wnd_pos_m4 + 1 * guide_stride);
    const auto ref =
        cast_to(d8, odd_even(cast_to(d64, ref_Bxx), cast_to(d64, ref_xMT)));
    const auto wnd_m2 = load_dup128(d8, wnd_pos_m4 + gbpr2);
    auto sad_1t0m = ext::mpsadbw2<0, 1>(wnd_m3, ref);
    const auto wnd_m1 = load_dup128(d8, wnd_pos_m4 + 3 * guide_stride);
    const auto sad_1m0b = ext::mpsadbw2<1, 2>(wnd_m2, ref);
    const auto wnd_m4 = load_dup128(d8, wnd_pos_m4);
    const auto wnd_m1m4 = concat_hi_lo(wnd_m1, wnd_m4);
    const auto sad_1b0t = ext::mpsadbw2<2, 0>(wnd_m1m4, ref);
    sad_1t0m += sad_1m0b;
    sad_1t0m += sad_1b0t;
    store(sad_1t0m, d16, sad + 0 * d16.N);

    // SAD 32
    const auto wnd_p0 = load_dup128(d8, wnd_pos_m4 + gbpr4);
    const auto wnd_p1 = load_dup128(d8, wnd_pos_m4 + 5 * guide_stride);
    auto sad_3t2m = ext::mpsadbw2<0, 1>(wnd_m1, ref);
    const auto wnd_p1m2 = concat_hi_lo(wnd_p1, wnd_m2);
    const auto sad_3m2b = ext::mpsadbw2<1, 2>(wnd_p0, ref);
    const auto sad_3b2t = ext::mpsadbw2<2, 0>(wnd_p1m2, ref);
    sad_3t2m += sad_3m2b;
    sad_3t2m += sad_3b2t;
    store(sad_3t2m, d16, sad + 1 * d16.N);

    // SAD 54
    const auto wnd_p2 = load_dup128(d8, wnd_pos_m4 + 6 * guide_stride);
    const auto wnd_p3 = load_dup128(d8, wnd_pos_m4 + 7 * guide_stride);
    const auto wnd_p3p0 = concat_hi_lo(wnd_p3, wnd_p0);
    auto sad_5t4m = ext::mpsadbw2<0, 1>(wnd_p1, ref);
    const auto sad_5m4b = ext::mpsadbw2<1, 2>(wnd_p2, ref);
    const auto sad_5b4t = ext::mpsadbw2<2, 0>(wnd_p3p0, ref);
    sad_5t4m += sad_5m4b;
    sad_5t4m += sad_5b4t;
    store(sad_5t4m, d16, sad + 2 * d16.N);

    const auto wnd_p4 = load_dup128(d8, wnd_pos_m4 + 8 * guide_stride);
    auto sad_6 = ext::mpsadbw2<0, 0>(wnd_p2, ref);  // t
    const auto sad_6m = ext::mpsadbw2<1, 1>(wnd_p3, ref);
    const auto sad_6b = ext::mpsadbw2<2, 2>(wnd_p4, ref);
    sad_6 += sad_6m;
    sad_6 += sad_6b;
    // Both 128-bit blocks are identical - required by SameBlocks().
    store(sad_6, d16, sad + 3 * d16.N);
#endif  // AVX2
  }
};

//------------------------------------------------------------------------------
// Exponentially decreasing weight functions

// Max such that mul_high(kClampedSAD << kShiftSAD, -32768) + bias=127*128 > 0.
// Also used by WeightExp to match WeightFast behavior at large distances.
// Doubling this maximum requires doubling kMinSigma.
constexpr int16_t kClampedSAD = 507;

// Straightforward but slow: computes e^{-s*x}.
class WeightExp {
 public:
  // W(sigma) = 0.5 = exp(mul_ * sigma) => mul_ = ln(0.5) / sigma.
  void SetSigma(const int sigma) {
    mul_ = (1 << kSigmaShift) * -0.69314717f / sigma;
  }

  void operator()(const V16 sad, VF* SIMD_RESTRICT lo,
                  VF* SIMD_RESTRICT hi) const {
    const auto clamped = min(sad, set1(d16, kClampedSAD));
    SIMD_ALIGN int16_t sad_lanes[d16.N];
    store(clamped, d16, sad_lanes);
    SIMD_ALIGN float weight_lanes[d16.N];
    for (size_t i = 0; i < d16.N; ++i) {
      weight_lanes[i] = expf(sad_lanes[i] * mul_);
    }
    *lo = load(df, weight_lanes);
    *hi = load(df, weight_lanes + df.N);
  }

  // All blocks of "sad" are identical, but this function does not make use
  // of that.
  VF SameBlocks(const V16 sad) const {
    const auto clamped = min(sad, set1(d16, kClampedSAD));
    SIMD_ALIGN int16_t sad_lanes[d16.N];
    store(clamped, d16, sad_lanes);
    // 1 for scalar, otherwise a full f32 vector.
    const size_t N = (d16.N + 1) / 2;
    float weight_lanes[N];
    for (size_t i = 0; i < N; ++i) {
      weight_lanes[i] = expf(sad_lanes[i] * mul_);
    }
    return load(df, weight_lanes);
  }

 private:
  float mul_;
};

static int16_t mul_table[kMaxSigma + 1];

// Fast approximation using the 2^x in the IEEE-754 representation.
class WeightFast {
 public:
  using D32 = Full<int32_t>;

  WeightFast() : bias_(set1(d16, 127 << (23 - 16))) { InitMulTable(); }

  void SetSigma(const int sigma) {
    // CHECK(kMinSigma <= sigma && sigma <= kMaxSigma);
    mul_ = set1(d16, mul_table[sigma]);
  }

  // Fills two f32 vectors from one i16 vector. On AVX2, "lo" are the lower
  // halves of two vectors (avoids crossing blocks).
  SIMD_INLINE void operator()(const V16 sad, VF* SIMD_RESTRICT lo,
                              VF* SIMD_RESTRICT hi) const {
    const auto zero = setzero(d16);

    // Avoid 16-bit overflow; ensures biased_exp >= 0.
    const auto clamped = min(sad, set1(d16, kClampedSAD));

    // Pre-shift to increase the multiplier range.
    const auto prescaled = shift_left<kShiftSAD>(clamped);

    // _Decrease_ to an unbiased exponent and fill in some mantissa bits.
    const auto unbiased_exp = ext::mul_high(prescaled, mul_);

    // Add exponent bias.
    auto biased_exp = unbiased_exp + bias_;

    // Assemble into an IEEE-754 representation with mantissa = zero.
    const auto bits_lo = zip_lo(zero, biased_exp);
    const auto bits_hi = zip_hi(zero, biased_exp);

    // Approximates exp(-s * sad).
    *lo = cast_to(df, bits_lo);
    *hi = cast_to(df, bits_hi);
  }

  // Same as above, but with faster i16x8->i32x8 conversion on AVX2 because all
  // blocks of "sad" are equal.
  SIMD_INLINE VF SameBlocks(const V16 sad) const {
    const auto clamped = min(sad, set1(d16, kClampedSAD));
    const auto prescaled = shift_left<kShiftSAD>(clamped);
    const auto unbiased_exp = ext::mul_high(prescaled, mul_);
    const auto biased_exp = unbiased_exp + bias_;

#if SIMD_TARGET_VALUE == SIMD_AVX2
    // Both blocks of biased_exp are identical, so we can MOVZX + shift into
    // the upper 16 bits using a single-cycle shuffle.
    SIMD_ALIGN constexpr int32_t kHi32From16[8] = {
        0x0100FFFF, 0x0302FFFF, 0x0504FFFF, 0x0706FFFF,
        0x0908FFFF, 0x0B0AFFFF, 0x0D0CFFFF, 0x0F0EFFFF,
    };
    const auto bits = table_lookup_bytes(cast_to(D32(), biased_exp),
                                         load(D32(), kHi32From16));
#else
    const auto bits = zip_lo(setzero(d16), biased_exp);
#endif

    return cast_to(df, bits);
  }

 private:
  // Larger shift = higher precision but narrower range of permissible SAD
  // (limited by 16-bit overflow, see kClampedSAD).
  static constexpr int kShiftSAD = 6;

  // Called once per target.
  void InitMulTable() {
    const int gap = 1 << kSigmaShift;
    // TODO(janwas): not thread-safe
    if (mul_table[0] != 0) return;
    int mul = -32768;
    for (int sigma = kMinSigma; sigma <= kMaxSigma; sigma += gap) {
      float w = 0.0f;
      for (; mul < 0; ++mul) {
        mul_ = set1(d16, mul);
        const auto weight = SameBlocks(set1(d16, sigma >> kSigmaShift));
        w = get_part(Part<float, 1>(), weight);
        if (w > 0.5f) {
          break;
        }
      }
      mul_table[sigma] = mul;
    }

    // Fill in (sigma, sigma + gap) via linear interpolation
    for (int sigma = kMinSigma; sigma < kMaxSigma; sigma += gap) {
      const float mul_step = (mul_table[sigma + gap] - mul_table[sigma]) / gap;
      for (int i = 1; i < gap; ++i) {
        mul_table[sigma + i] = mul_table[sigma] + i * mul_step;
      }
    }
  }

  // Upper 16 bits of the IEEE-754 exponent bias.
  const V16 bias_;  // must initialize before mul_
  V16 mul_;
};

// (Must be in same file to use WeightFast etc.)
class InternalWeightTests {
 public:
  static void Run() {
    TestEndpoints();
    TestWeaklyMonotonicallyDecreasing();
    TestFastMatchesExp();
  }

 private:
  // Returns weight, or aborts.
  static float EnsureWeightEquals(const float expected, const int16_t sad,
                                  const int sigma,
                                  const WeightFast& weight_func,
                                  const float tolerance) {
    const Part<float, 1> df1;

    VF lo, hi;
    weight_func(set1(d16, sad), &lo, &hi);
    const float w0 = get_part(df1, lo);
    const float w1 = get_part(df1, hi);
    CHECK(w0 == w1);

    if (std::abs(w0 - expected) > tolerance) {
      printf("Weight %f too far from %f for sigma %d, sad %d\n", w0, expected,
             sigma, sad);
      abort();
    }
    return w0;
  }

  static void TestEndpoints() {
    WeightFast weight_func;
    // Only test at integral sigma because we can't represent fractional SAD,
    // and weight_{sigma+3}(sad) is too far from 0.5.
    for (int sigma = kMinSigma; sigma <= kMaxSigma; sigma += 1 << kSigmaShift) {
      weight_func.SetSigma(sigma);
      // Zero SAD => max weight 1.0
      EnsureWeightEquals(1.0f, 0, sigma, weight_func, 0.02f);
      // Half-width at half max => 0.5
      EnsureWeightEquals(0.5f, sigma >> kSigmaShift, sigma, weight_func, 0.02f);
    }
  }

  // WeightFast and WeightExp should return similar values.
  static void TestFastMatchesExp() {
    WeightExp func_slow;
    WeightFast func;

    for (int sigma = kMinSigma; sigma <= kMaxSigma; ++sigma) {
      func_slow.SetSigma(sigma);
      func.SetSigma(sigma);

      for (int sad = 0; sad <= Distance::kMaxSAD; ++sad) {
        VF lo_slow, unused;
        func_slow(set1(d16, sad), &lo_slow, &unused);
        const float weight_slow = get_part(Part<float, 1>(), lo_slow);
        // Max tolerance is required for very low sigma (0.75 vs 0.707).
        EnsureWeightEquals(weight_slow, sad, sigma, func, 0.05f);
      }
    }
  }

  // Weight(sad + 1) <= Weight(sad).
  static void TestWeaklyMonotonicallyDecreasing() {
    WeightFast weight_func;
    weight_func.SetSigma(30 << kSigmaShift);  // half width at half max

    const Part<float, 1> df1;

    float last_w = 1.1f;
    for (int sad = kMinSigma; sad <= kMaxSigma; ++sad) {
      VF lo, hi;
      weight_func(set1(d16, sad), &lo, &hi);
      const float w = get_part(df1, lo);
      CHECK(w <= last_w);
      last_w = w;
    }
  }
};

//------------------------------------------------------------------------------

// Need three separate guide, input and output rows for the RGB case.
struct Args3 {
  void SetRow(const size_t y, const Image3B& guide, const Image3F& in,
              Image3F* SIMD_RESTRICT out) {
    for (size_t c = 0; c < 3; ++c) {
      guide_m4[c] = guide.ConstPlaneRow(c, y - 4 + kBorder) + kBorder;
      in_m3[c] = in.ConstPlaneRow(c, y - 3 + kBorder) + kBorder;
      this->out[c] = out->PlaneRow(c, y);
    }
  }

  size_t guide_stride;
  size_t in_stride;

  // "guide_m4" and "in_m3" are 4 and 3 rows above the current pixel.
  const uint8_t* SIMD_RESTRICT guide_m4[3];
  const float* SIMD_RESTRICT in_m3[3];
  float* SIMD_RESTRICT out[3];
};

struct Args1 {
  void SetRow(const size_t y, const ImageB& guide, const ImageF& in,
              ImageF* SIMD_RESTRICT out) {
    guide_m4 = guide.ConstRow(y - 4 + kBorder) + kBorder;
    in_m3 = in.ConstRow(y - 3 + kBorder) + kBorder;
    this->out = out->Row(y);
  }

  size_t guide_stride;
  size_t in_stride;

  // "guide_m4" and "in_m3" are 4 and 3 rows above the current pixel.
  const uint8_t* SIMD_RESTRICT guide_m4;
  const float* SIMD_RESTRICT in_m3;
  float* SIMD_RESTRICT out;
};

// Factory functions enable deduction of Arg* from the image type.
Args3 MakeArgs(const Image3B& guide, const Image3F& in) {
  Args3 args;
  args.guide_stride = guide.plane(0).bytes_per_row();
  args.in_stride = in.plane(0).bytes_per_row();
  for (size_t c = 0; c < 3; ++c) {
    PIK_CHECK(args.guide_stride == guide.plane(c).bytes_per_row());
    PIK_CHECK(args.in_stride == in.plane(c).bytes_per_row());
  }
  return args;
}

Args1 MakeArgs(const ImageB& guide, const ImageF& in) {
  Args1 args;
  args.guide_stride = guide.bytes_per_row();
  args.in_stride = in.bytes_per_row();
  return args;
}

class WeightedSum {
 public:
  static constexpr size_t kNeighbors = Distance::kNeighbors;

  static void Test() { TestHorzSums(); }

  template <class WeightFunc>
  static SIMD_INLINE void Compute(const size_t x, const Args1& args,
                                  const WeightFunc& weight_func) {
    SIMD_ALIGN float weights[kNeighbors];
    ComputeWeights(x, args, weight_func, weights);
    FromWeights(args.in_m3 + x, args.in_stride, weights, args.out + x);
  }

  template <class WeightFunc>
  static SIMD_INLINE void Compute(const size_t x, const Args3& args,
                                  const WeightFunc& weight_func) {
    SIMD_ALIGN float weights[kNeighbors];
    ComputeWeights(x, args, weight_func, weights);
    for (size_t c = 0; c < 3; ++c) {
      FromWeights(args.in_m3[c] + x, args.in_stride, weights, args.out[c] + x);
    }
  }

 private:
  // 2465 ops per pixel (2016 + 56 * (5 + 3) + 1)  -- TODO(janwas): split
  // NOTE: weights may be stored interleaved.
  template <class WeightFunc>
  static SIMD_INLINE void WeightsFromSAD(const int16_t* SIMD_RESTRICT sad,
                                         const WeightFunc& weight_func,
                                         float* SIMD_RESTRICT weights) {
#if SIMD_TARGET_VALUE == SIMD_NONE
    for (size_t i = 0; i < kNeighbors; ++i) {
      const auto sad_v = set1(d16, sad[i]);
      VF lo, unused;
      weight_func(sad_v, &lo, &unused);
      store(lo, df, weights + i);
    }
#elif SIMD_TARGET_VALUE != SIMD_AVX2
    f32x4 w0L, w0H, w1L, w1H, w2L, w2H, w3L, w3H, w4L, w4H, w5L, w5H, w6L, w6H;
    weight_func(load(d16, sad + 0 * d16.N), &w0L, &w0H);
    weight_func(load(d16, sad + 1 * d16.N), &w1L, &w1H);
    weight_func(load(d16, sad + 2 * d16.N), &w2L, &w2H);
    weight_func(load(d16, sad + 3 * d16.N), &w3L, &w3H);
    weight_func(load(d16, sad + 4 * d16.N), &w4L, &w4H);
    weight_func(load(d16, sad + 5 * d16.N), &w5L, &w5H);
    weight_func(load(d16, sad + 6 * d16.N), &w6L, &w6H);
    store(w0L, df, weights + 0 * df.N);
    store(w0H, df, weights + 1 * df.N);
    store(w1L, df, weights + 2 * df.N);
    store(w1H, df, weights + 3 * df.N);
    store(w2L, df, weights + 4 * df.N);
    store(w2H, df, weights + 5 * df.N);
    store(w3L, df, weights + 6 * df.N);
    store(w3H, df, weights + 7 * df.N);
    store(w4L, df, weights + 8 * df.N);
    store(w4H, df, weights + 9 * df.N);
    store(w5L, df, weights + 10 * df.N);
    store(w5H, df, weights + 11 * df.N);
    store(w6L, df, weights + 12 * df.N);
    store(w6H, df, weights + 13 * df.N);
#else  // AVX2
    decltype(setzero(df)) w10L, w10H, w32L, w32H, w54L, w54H, w6;
    weight_func(load(d16, sad + 0 * d16.N), &w10L, &w10H);
    weight_func(load(d16, sad + 1 * d16.N), &w32L, &w32H);
    weight_func(load(d16, sad + 2 * d16.N), &w54L, &w54H);
    w6 = weight_func.SameBlocks(load(d16, sad + 3 * d16.N));
    store(w10L, df, weights + 0 * df.N);
    store(w10H, df, weights + 1 * df.N);
    store(w32L, df, weights + 2 * df.N);
    store(w32H, df, weights + 3 * df.N);
    store(w54L, df, weights + 4 * df.N);
    store(w54H, df, weights + 5 * df.N);
    store(w6, df, weights + 6 * df.N);
#endif
  }

  // Returns weights for 7x8 neighbor pixels
  template <class WeightFunc>
  static SIMD_INLINE void ComputeWeights(const size_t x, const Args1& args,
                                         const WeightFunc& weight_func,
                                         float* SIMD_RESTRICT weights) {
    SIMD_ALIGN int16_t sad[64];
    Distance::SumsOfAbsoluteDifferences(args.guide_m4 + x, args.guide_stride,
                                        sad);
    WeightsFromSAD(sad, weight_func, weights);
  }

  template <class WeightFunc>
  static SIMD_INLINE void ComputeWeights(const size_t x, const Args3& args,
                                         const WeightFunc& weight_func,
                                         float* SIMD_RESTRICT weights) {
    // It's important to include all channels, only computing for X and Y
    // channels misses/weakens some edges.
    SIMD_ALIGN int16_t sad[3][64];
    for (size_t c = 0; c < 3; ++c) {
      Distance::SumsOfAbsoluteDifferences(args.guide_m4[c] + x,
                                          args.guide_stride, &sad[c][0]);
    }

    Full<int16_t> d;
    auto sum_d = set1(d, 0);
    for (size_t i = 0; i < 64; i += d.N) {
      const auto d0 = load(d, &sad[0][i]);
      const auto d1 = load(d, &sad[1][i]);
      const auto d2 = load(d, &sad[2][i]);
      sum_d += d0 + d1 + d2;
      const auto max_d = max(max(d0, d1), d2);
      store(max_d, d, &sad[0][i]);
    }

    SIMD_ALIGN int16_t lanes[d.N];
    store(sum_d, d, lanes);
    int sum = 0;
    for (size_t i = 0; i < d.N; ++i) {
      sum += lanes[i];
    }

    WeightsFromSAD(&sad[0][0], weight_func, weights);

    // TODO(janwas): resume experiment
    // Per channel and pixel (*12)
    // ximpulse = sum > 3 * xsigma * 16 * 12;
    // float wsum = std::accumulate(weights, weights + kNeighbors, 0.0f);
    // ximpulse = wsum < 1.01f;
    // if (ximpulse) {
    //   xdsum += double(sum) / xsigma;
    //   ++ximpulse_count;
    // }
  }

  // Returns sum(num) / sum(den).
  template <class V>
  static SIMD_INLINE Part<float, 1>::V RatioOfHorizontalSums(const V num,
                                                             const V den) {
    const Part<float, 1> d;
    // Faster than concat_lo_lo/hi_hi plus single sum_of_lanes.
    const auto sum_den = any_part(d, ext::sum_of_lanes(den));
    const auto sum_num = any_part(d, ext::sum_of_lanes(num));
    const auto rcp_den = approximate_reciprocal(sum_den);
    return rcp_den * sum_num;
  }

  static SIMD_INLINE void FromWeights(const float* SIMD_RESTRICT in_m3,
                                      const size_t in_stride,
                                      const float* SIMD_RESTRICT weights,
                                      float* SIMD_RESTRICT out) {
#if SIMD_TARGET_VALUE == SIMD_NONE
    float weighted_sum = 0.0f;
    float sum_weights = 0.0f;
    int i = 0;
    for (int cy = -3; cy <= 3; ++cy) {
      const float* SIMD_RESTRICT in_row =
          ByteOffset(in_m3, (cy + 3) * in_stride);
      for (int cx = -3; cx <= 4; ++cx) {
        const float neighbor = in_row[cx];
        const float weight = weights[i++];
        weighted_sum += neighbor * weight;
        sum_weights += weight;
      }
    }

    // Safe because weights[27] == 1.
    *out = weighted_sum / sum_weights;
#elif SIMD_TARGET_VALUE != SIMD_AVX2
    in_m3 -= 3;

    const auto w0L = load(df, weights + 0 * df.N);
    const auto w0H = load(df, weights + 1 * df.N);
    const auto w1L = load(df, weights + 2 * df.N);
    const auto w1H = load(df, weights + 3 * df.N);
    const auto w2L = load(df, weights + 4 * df.N);
    const auto w2H = load(df, weights + 5 * df.N);
    const auto w3L = load(df, weights + 6 * df.N);
    const auto w3H = load(df, weights + 7 * df.N);
    const auto w4L = load(df, weights + 8 * df.N);
    const auto w4H = load(df, weights + 9 * df.N);
    const auto w5L = load(df, weights + 10 * df.N);
    const auto w5H = load(df, weights + 11 * df.N);
    const auto w6L = load(df, weights + 12 * df.N);
    const auto w6H = load(df, weights + 13 * df.N);

    const auto n0L = load_unaligned(df, ByteOffset(in_m3, 0 * in_stride));
    const auto n1L = load_unaligned(df, ByteOffset(in_m3, 1 * in_stride));
    const auto n2L = load_unaligned(df, ByteOffset(in_m3, 2 * in_stride));
    const auto n3L = load_unaligned(df, ByteOffset(in_m3, 3 * in_stride));
    const auto n4L = load_unaligned(df, ByteOffset(in_m3, 4 * in_stride));
    const auto n5L = load_unaligned(df, ByteOffset(in_m3, 5 * in_stride));
    const auto n6L = load_unaligned(df, ByteOffset(in_m3, 6 * in_stride));
    const auto n0H =
        load_unaligned(df, ByteOffset(in_m3, 0 * in_stride) + df.N);
    const auto n1H =
        load_unaligned(df, ByteOffset(in_m3, 1 * in_stride) + df.N);
    const auto n2H =
        load_unaligned(df, ByteOffset(in_m3, 2 * in_stride) + df.N);
    const auto n3H =
        load_unaligned(df, ByteOffset(in_m3, 3 * in_stride) + df.N);
    const auto n4H =
        load_unaligned(df, ByteOffset(in_m3, 4 * in_stride) + df.N);
    const auto n5H =
        load_unaligned(df, ByteOffset(in_m3, 5 * in_stride) + df.N);
    const auto n6H =
        load_unaligned(df, ByteOffset(in_m3, 6 * in_stride) + df.N);

    const auto sum_weights = w0L + w0H + w1L + w1H + w2L + w2H + w3L + w3H +
                             w4L + w4H + w5L + w5H + w6L + w6H;

    auto weighted_sum = n0L * w0L;
    weighted_sum = mul_add(n0H, w0H, weighted_sum);
    weighted_sum = mul_add(n1L, w1L, weighted_sum);
    weighted_sum = mul_add(n1H, w1H, weighted_sum);
    weighted_sum = mul_add(n2L, w2L, weighted_sum);
    weighted_sum = mul_add(n2H, w2H, weighted_sum);
    weighted_sum = mul_add(n3L, w3L, weighted_sum);
    weighted_sum = mul_add(n3H, w3H, weighted_sum);
    weighted_sum = mul_add(n4L, w4L, weighted_sum);
    weighted_sum = mul_add(n4H, w4H, weighted_sum);
    weighted_sum = mul_add(n5L, w5L, weighted_sum);
    weighted_sum = mul_add(n5H, w5H, weighted_sum);
    weighted_sum = mul_add(n6L, w6L, weighted_sum);
    weighted_sum = mul_add(n6H, w6H, weighted_sum);

    store(RatioOfHorizontalSums(weighted_sum, sum_weights), Part<float, 1>(),
          out);
#else  // AVX2
    in_m3 -= 3;
    const size_t kN2 = df.N / 2;

    // Weighted sum 10
    const auto n0 = load_unaligned(df, ByteOffset(in_m3, 0 * in_stride));
    const auto n1L = load_dup128(df, ByteOffset(in_m3, 1 * in_stride));
    const auto n1H = load_dup128(df, ByteOffset(in_m3, 1 * in_stride) + kN2);
    const auto w10L = load(df, weights + 0 * df.N);
    const auto w10H = load(df, weights + 1 * df.N);
    const auto n10L = concat_hi_lo(n1L, n0);
    const auto n10H = concat_hi_hi(n1H, n0);
    const auto sum01 = w10L + w10H;
    const auto mul0 = n10L * w10L;
    const auto mul1 = n10H * w10H;

    // Weighted sum 32
    const auto n2 = load_unaligned(df, ByteOffset(in_m3, 2 * in_stride));
    const auto n3L = load_dup128(df, ByteOffset(in_m3, 3 * in_stride));
    const auto n3H = load_dup128(df, ByteOffset(in_m3, 3 * in_stride) + kN2);
    const auto w32L = load(df, weights + 2 * df.N);
    const auto w32H = load(df, weights + 3 * df.N);
    const auto n32L = concat_hi_lo(n3L, n2);
    const auto n32H = concat_hi_hi(n3H, n2);
    const auto sum23 = w32L + w32H;
    const auto mul02 = mul_add(n32L, w32L, mul0);
    const auto mul13 = mul_add(n32H, w32H, mul1);

    // Weighted sum 54
    const auto n4 = load_unaligned(df, ByteOffset(in_m3, 4 * in_stride));
    const auto n5L = load_dup128(df, ByteOffset(in_m3, 5 * in_stride));
    const auto n5H = load_dup128(df, ByteOffset(in_m3, 5 * in_stride) + kN2);
    const auto w54L = load(df, weights + 4 * df.N);
    const auto w54H = load(df, weights + 5 * df.N);
    const auto n54L = concat_hi_lo(n5L, n4);
    const auto n54H = concat_hi_hi(n5H, n4);
    const auto sum0123 = sum01 + sum23;
    const auto mul024 = mul_add(n54L, w54L, mul02);
    const auto sum45 = w54L + w54H;
    const auto mul135 = mul_add(n54H, w54H, mul13);

    const auto mul012345 = mul024 + mul135;
    const auto sum012345 = sum0123 + sum45;

    const auto n6 = load_unaligned(df, ByteOffset(in_m3, 6 * in_stride));
    const auto w6 = load(df, weights + 6 * df.N);
    const auto weighted_sum = mul_add(n6, w6, mul012345);
    const auto sum_weights = sum012345 + w6;

    store(RatioOfHorizontalSums(weighted_sum, sum_weights), Part<float, 1>(),
          out);
#endif
  }

  static void TestHorzSums() {
    const Part<float, 1> df1;

    SIMD_ALIGN const float in0_lanes[8] = {256.8f, 128.7f, 64.6f, 32.5f,
                                           16.4f,  8.3f,   4.2f,  2.1f};
    SIMD_ALIGN const float in1_lanes[8] = {-0.1f, -1.2f, -2.3f, -3.4f,
                                           -4.5f, -5.6f, -6.7f, -7.8f};
    for (size_t i = 0; i < 8; i += df.N) {
      const auto in0 = load(df, in0_lanes + i);
      const auto in1 = load(df, in1_lanes + i);

      const float expected0 =
          std::accumulate(in0_lanes + i, in0_lanes + i + df.N, 0.0f);
      const float expected1 =
          std::accumulate(in1_lanes + i, in1_lanes + i + df.N, 0.0f);
      const float expected = X64_Reciprocal12(expected1) * expected0;

      const float actual = get_part(df1, RatioOfHorizontalSums(in0, in1));
      CHECK(std::abs(expected - actual) < 2E-2f);
    }
  }
};

// The only difference between fast and slow versions is WeightFunc.
template <class Guide, class Image, class WeightFunc>
void Filter(const Guide& guide, const Image& in, const WeightFunc& weight_func,
            Image* SIMD_RESTRICT out) {
  const size_t xsize = out->xsize();
  const size_t ysize = out->ysize();
  PIK_CHECK(xsize != 0 && ysize != 0);
  // "guide" and "in" have kBorder extra pixels on each side.
  PIK_CHECK(SameSize(guide, in));
  PIK_CHECK(in.xsize() >= xsize + 2 * kBorder);
  PIK_CHECK(in.ysize() >= ysize + 2 * kBorder);

  auto args = MakeArgs(guide, in);

  for (size_t y = 0; y < ysize; ++y) {
    args.SetRow(y, guide, in, out);

    for (size_t x = 0; x < xsize; ++x) {
      WeightedSum::Compute(x, args, weight_func);
    }
  }
}

int Sigma(const AdaptiveFilterParams& params, const int ac_quant) {
  const int min_quant = ac_quant;
  int sigma = 1.0f / (params.sigma_mul * min_quant);
  // int sigma = 1.0f / (params.sigma_mul * min_quant * sqrtf(min_quant));
  return std::min(sigma, epf::kMaxSigma);
}

template <class Guide, class Image>
void FilterAdaptive(const Guide& guide, const Image& in,
                    const AdaptiveFilterParams& params,
                    Image* SIMD_RESTRICT out) {
  const size_t xsize = out->xsize();
  const size_t ysize = out->ysize();
#if PROFILER_ENABLED
  printf("filter %zu pix\n", xsize * ysize);
#endif
  PROFILER_FUNC;
  PIK_CHECK(xsize != 0 && ysize != 0);
  PIK_CHECK((xsize | ysize) % 8 == 0);
  // "guide" and "in" have kBorder extra pixels on each side.
  PIK_CHECK(SameSize(guide, in));
  PIK_CHECK(in.xsize() >= xsize + 2 * kBorder);
  PIK_CHECK(in.ysize() >= ysize + 2 * kBorder);

  auto args = MakeArgs(guide, in);
  const int kSkipThreshold = kMinSigma;
  WeightFast weight_func;

#if DUMP_SIGMA
  ImageB dump(xsize / 8, ysize / 8);
#endif

  for (size_t by = 0; by < ysize; by += 8) {
    const int* PIK_RESTRICT ac_quant_row = params.ac_quant->Row(by / 8);
#if DUMP_SIGMA
    uint8_t* dump_row = dump.Row(by / 8);
#endif

    for (size_t bx = 0; bx < xsize; bx += 8) {
      const int sigma = Sigma(params, ac_quant_row[bx / 8]);
#if DUMP_SIGMA
      dump_row[bx / 8] = sigma;
#endif
      if (sigma < kSkipThreshold) continue;
      weight_func.SetSigma(sigma);

      for (size_t iy = 0; iy < 8; ++iy) {
        args.SetRow(by + iy, guide, in, out);
        for (size_t ix = 0; ix < 8; ++ix) {
          WeightedSum::Compute(bx + ix, args, weight_func);
        }
      }
    }
  }

#if DUMP_SIGMA
  WriteImage(ImageFormatPNG(), dump, "/tmp/out/sigma.png");
#endif
}

class Padding {
 public:
  // Returns a new image with kBorder additional pixels on each side initialized
  // by mirroring.
  template <typename T>
  static Image<T> PadImage(const Image<T>& in) {
    const int64_t ixsize = in.xsize();
    const int64_t iysize = in.ysize();
    Image<T> out(ixsize + 2 * kBorder, iysize + 2 * kBorder);
    int64_t iy = -kBorder;
    for (; iy < 0; ++iy) {
      PadRow<WrapMirror>(in, iy, iysize, &out);
    }
    for (; iy < iysize; ++iy) {
      PadRow<WrapUnchanged>(in, iy, iysize, &out);
    }
    for (; iy < iysize + kBorder; ++iy) {
      PadRow<WrapMirror>(in, iy, iysize, &out);
    }
    return out;
  }

  template <typename T>
  static Image3<T> PadImage(const Image3<T>& in) {
    return Image3<T>(PadImage(in.plane(0)), PadImage(in.plane(1)),
                     PadImage(in.plane(2)));
  }

  static void Test() {
    for (size_t ysize = 8; ysize < 16; ++ysize) {
      for (size_t xsize = 8; xsize < 16; ++xsize) {
        ImageF in(xsize, ysize, 0.0f);
        ImageF padded = PadImage(in);
        Image3F in3(xsize, ysize, 0.0f);
        Image3F padded3 = PadImage(in3);
        EnsureInitialized(padded);
        for (int c = 0; c < 3; ++c) {
          EnsureInitialized(padded3.plane(c));
        }
      }
    }
  }

 private:
  template <class Wrap>
  static void PadRow(const ImageF& in, const int64_t iy, const int64_t ysize,
                     ImageF* SIMD_RESTRICT out) {
    const int64_t ixsize = in.xsize();
    const int64_t clamped_y = Wrap()(iy, ysize);
    const float* const PIK_RESTRICT row_in = in.ConstRow(clamped_y);
    float* const PIK_RESTRICT row_out = out->Row(iy + kBorder) + kBorder;

    // Ensure store alignment (faster than loading aligned)
    constexpr int64_t first_aligned = (kBorder + df.N - 1) & ~(df.N - 1);

    // Left: mirror and vector alignment
    int64_t ix = -kBorder;
    for (; ix < first_aligned - kBorder; ++ix) {
      const int clamped_x = Mirror(ix, ixsize);
      row_out[ix] = row_in[clamped_x];
    }

    // Interior: whole vectors
    for (; ix + df.N <= ixsize; ix += df.N) {
      store(load_unaligned(df, row_in + ix), df, row_out + ix);
    }

    // Right: vector remainder and mirror
    for (; ix < ixsize + kBorder; ++ix) {
      const int clamped_x = Mirror(ix, ixsize);
      row_out[ix] = row_in[clamped_x];
    }
  }

  static void EnsureInitialized(const ImageF& padded) {
    for (size_t y = 0; y < padded.ysize(); ++y) {
      const float* SIMD_RESTRICT row = padded.Row(y);
      for (size_t x = 0; x < padded.xsize(); ++x) {
        if (row[x] != 0.0f) {
          printf("Uninitialized at %zu %zu\n", x, y);
          abort();
        }
      }
    }
  }
};

// Returns a guide image for "in". Required for the SAD hardware acceleration;
// precomputing is faster than converting a 7x8 window for each pixel.
// Providing min/max values avoids an additional scan through the image.
ImageB MakeGuide(const ImageF& in, float min = 0.0f, float max = 0.0f) {
  const size_t xsize = in.xsize();
  const size_t ysize = in.ysize();
  ImageB out(xsize, ysize);

  if (max == 0.0f) {
    PROFILER_ZONE("minmax");
    ImageMinMax(in, &min, &max);
    CHECK(max != 0.0f);
  }
  CHECK(max > min);

  const Part<uint8_t, df.N> d8;
  const auto vmul = set1(df, 255.0f / (max - min));
  const auto vmin = set1(df, min);

  for (size_t y = 0; y < ysize; ++y) {
    const float* SIMD_RESTRICT row_in = in.ConstRow(y);
    uint8_t* SIMD_RESTRICT row_out = out.Row(y);

    for (size_t x = 0; x < in.xsize(); x += df.N) {
      const auto scaled = (load(df, row_in + x) - vmin) * vmul;
      const auto i32 = convert_to(Full<int32_t>(), scaled);
      const auto bytes = u8_from_u32(cast_to(Full<uint32_t>(), i32));
      store(bytes, d8, row_out + x);
    }
  }

  return out;
}

// Same as above for RGB.
Image3B MakeGuide(const Image3F& in, float min = 0.0f, float max = 0.0f) {
  const size_t xsize = in.xsize();
  const size_t ysize = in.ysize();
  Image3B out(xsize, ysize);

  if (max == 0.0f) {
    PROFILER_ZONE("minmax");
    std::array<float, 3> min3, max3;
    Image3MinMax(in, &min3, &max3);
    min = *std::min_element(min3.begin(), min3.end());
    max = *std::max_element(max3.begin(), max3.end());
    CHECK(max != 0.0f);
  }

  CHECK(max > min);
  const Part<uint8_t, df.N> d8;
  const auto vmul = set1(df, 255.0f / (max - min));
  const auto vmin = set1(df, min);

  for (size_t c = 0; c < 3; ++c) {
    for (size_t y = 0; y < in.ysize(); ++y) {
      const float* SIMD_RESTRICT row_in = in.ConstPlaneRow(c, y);
      uint8_t* SIMD_RESTRICT row_out = out.PlaneRow(c, y);

      for (size_t x = 0; x < in.xsize(); x += df.N) {
        const auto scaled = (load(df, row_in + x) - vmin) * vmul;
        const auto i32 = convert_to(Full<int32_t>(), scaled);
        const auto bytes = u8_from_u32(cast_to(Full<uint32_t>(), i32));
        store(bytes, d8, row_out + x);
      }
    }
  }

  return out;
}

// Calls filter after padding "in" and creating a guide from it.
// Image and WeightFunc are deduced.
template <class Image, class WeightFunc>
void PadAndFilter(Image* in_out, const float min, const float max,
                  const WeightFunc& weight_func) {
  PROFILER_FUNC;

  Image padded = Padding::PadImage(*in_out);
  auto guide = MakeGuide(padded, min, max);
  Filter(guide, padded, weight_func, in_out);
}

//-----------------------------------------------------------------------------
// The following are unused in PIK, where we care about decode speed and could
// use side information to estimate the local sigma.

// Partitions intensities into "bands" containing similar numbers of pixels.
template <size_t kNumBands>
class BandPartition {
 public:
  template <class Guide>
  BandPartition(const Guide& guide) {
    const size_t xsize = guide.xsize();
    const size_t ysize = guide.ysize();

    const size_t kNumIntensities = 256;
    int histogram[kNumIntensities] = {0};
    for (size_t y = 0; y < ysize; ++y) {
      const auto guide_row = guide.ConstRow(y);
      for (size_t x = 0; x < xsize; ++x) {
        ++histogram[static_cast<int>(guide_row[x])];
      }
    }

    const size_t votes_per_band = xsize * ysize / kNumBands;
    size_t bin = 0;
    // Do not reset to zero in every band so that excess votes
    // will count towards the next band.
    size_t votes = 0;
    for (size_t band = 0; band < kNumBands; ++band) {
      while (votes < votes_per_band && bin < kNumIntensities) {
        votes += histogram[bin];
        ++bin;
      }
      votes -= votes_per_band;
      band_end_[band] = bin;
      // TODO(janwas): remove
      printf("%zu: %zu\n", band, bin);
    }
  }

  size_t BandFromIntensity(const int intensity) const {
    // TODO(janwas): SIMD broadcast, compare, movemask, clz
    for (size_t band = 0; band < kNumBands; ++band) {
      if (intensity < band_end_[band]) return band;
    }
    CHECK(false);  // not within any range => possibly negative?
    return 0;
  }

 private:
  std::array<int, kNumBands> band_end_;
};

// Modified from edge_detector.cc.
// https://en.wikipedia.org/wiki/Otsu%27s_method
float OtsuThreshold(const Histogram& hist, const int max_bin) {
  int64_t total = 0;
  for (size_t i = 0; i < max_bin; ++i) {
    total += hist.Bin(i);
  }
  const double scale = 1.0 / total;
  double omega0 = 0.0;
  double mu0 = 0.0;
  double mu_total = 0.0;
  for (int i = 0; i < max_bin; ++i) {
    mu_total += i * hist.Bin(i) * scale;
  }
  int best_i = 0;
  double best_sigma = 0.0;
  for (int i = 1; i < max_bin; ++i) {
    const double prob = hist.Bin(i - 1) * scale;
    omega0 += prob;
    mu0 += (i - 1) * prob;
    const double omega1 = 1.0 - omega0;
    const double mu1 = mu_total - mu0;
    const double delta = mu0 / omega0 - mu1 / omega1;
    const double sigma = omega0 * omega1 * delta * delta;
    if (sigma > best_sigma) {
      best_sigma = sigma;
      best_i = i;
    }
  }
  return best_i;
}

class DifferenceDistribution {
 public:
  static float Mul() {
    return 1.0f;  // clamp
  }
  static double Encode(double x) { return std::min(x, 255.0); }
  static double Decode(double x) { return x; }

  void Add(const uint8_t* SIMD_RESTRICT guide_m4, const size_t guide_stride) {
    // The last 8 are duplicates.
    SIMD_ALIGN int16_t sad[Distance::kNeighbors + 8];
    // TODO(janwas): specialized version for smaller 5x5 neighborhood?
    Distance::SumsOfAbsoluteDifferences(guide_m4, guide_stride, sad);
    CHECK(sad[3 * 8 + 3] == 0);

    // Rank Order Absolute Difference: sum of the smallest differences
    // indicates how much a pixel resembles impulse noise. We use a much
    // larger window of patch-based SADs to increase robustness.
    const size_t kSamples = (Distance::kNeighbors - 1) / 2;
    std::sort(sad, sad + Distance::kNeighbors);
    const int sad_sum = std::accumulate(sad, sad + kSamples, 0);
    const float sad_avg = static_cast<float>(sad_sum) / kSamples;
    sad_org_.Increment(sad_avg);
    const float x = Encode(sad_avg);
    sad_histogram_.Increment(x);
    stats_.Notify(x);
  }
  /*
      // We expect a heavy-tailed distribution with many values outside the
      // histogram domain (assigned to the last bin). The Half-Range Mode
      // estimator will ignore the final peak because its mass is much less than
      // around the mode, so the estimator will zoom in on the first half and
      // never see the final peak again. Similarly, a smaller peak around zero
      // is also ignored.

      const int mode = sad_histogram_.Mode();
      int64_t sum = 0;
      for (size_t i = 0; i < Histogram::kBins - 1; ++i) {
        sum += sad_histogram_.Bin(i);
      }
      if (sad_histogram_.Bin(Histogram::kBins - 1) > sum / 2)  return 0;

      return exp(mode / Mul());
    }
  */
  void Print(const char* message = "") const {
    const int mode = sad_histogram_.Mode();
    int64_t sum = 0;
    for (size_t i = 0; i < Histogram::kBins - 1; ++i) {
      sum += sad_histogram_.Bin(i);
    }
    const bool heavy_tail = sad_histogram_.Bin(Histogram::kBins - 1) > sum / 2;

    // TODO(janwas): iterate over IQR candidates, compute probability of
    // observing the histogram.
    const int tail_begin = Decode(mode);
    Stats stats2;
    for (size_t i = 0; i < Histogram::kBins; ++i) {
      if (i > tail_begin) break;
      for (size_t j = 0; j < sad_org_.Bin(i); ++j) {
        stats2.Notify(i);
      }
    }

    const int otsu = OtsuThreshold(sad_org_, mode);

    static std::mutex mutex;
    mutex.lock();

    if (heavy_tail) {
      printf("%s: heavy\n", message);
      mutex.unlock();
      return;
    }
    printf("%s: cutoff %d otsu %d iqr %f skew %f kurt %f heavy %d\n", message,
           tail_begin, otsu, sad_org_.IQR(), stats2.Skewness(),
           stats2.Kurtosis(), heavy_tail);
    GaussianMixture mix[3];
    const int num_mix = EstimateGaussianMixture(stats2, mix);
    for (int i = 0; i < num_mix; ++i) {
      if (mix[i].mean0 < 0.0 && mix[i].weight > 0.01) continue;
      if (mix[i].mean1 < 0.0 && mix[i].weight < 0.99) continue;
      printf("  w %f %f %f (var %f)\n", mix[i].weight, mix[i].mean0,
             mix[i].mean1, mix[i].var0);
    }
    // TODO(janwas): remove
    sad_org_.Print();
    // sad_histogram_.Print();
    mutex.unlock();
  }

 private:
  Histogram sad_histogram_;
  Histogram sad_org_;
  Stats stats_;
};

// Estimates sigma and filters twice.
template <class Image>
void TwoPhase(const Image& in, Image* SIMD_RESTRICT out, float min, float max) {
  Image padded = Padding::PadImage(in);
  const size_t xsize = padded.xsize();
  const size_t ysize = padded.ysize();

  auto guide = MakeGuide(padded, min, max);
  const size_t guide_stride = guide.bytes_per_row();

  constexpr size_t kNumBands = 1;
  const BandPartition<kNumBands> band_partition(guide);

  std::vector<DifferenceDistribution> diff_distributions(kNumBands);

  for (int y = kBorder; y < ysize - kBorder; y += 1) {
    const uint8_t* SIMD_RESTRICT guide_m4 = guide.ConstRow(y - 4);
    const uint8_t* SIMD_RESTRICT guide_row = guide.ConstRow(y);

    for (int x = kBorder; x < xsize - kBorder; x += 1) {
      const size_t band = band_partition.BandFromIntensity(guide_row[x]);
      diff_distributions[band].Add(guide_m4 + x, guide_stride);
    }
  }

  for (size_t band = 0; band < kNumBands; ++band) {
    printf("=================================%zu\n", band);
    diff_distributions[band].Print();
  }

  // TODO(janwas): find better mapping of mode -> filter sigma.

  // const int sigma = std::min(std::max(kMinSigma, mode), kMaxSigma);
  // Filter with estimated parameter
}

template <class Image>
int EstimateNoise(const char* message, const Image& in, float min, float max) {
  const size_t xsize = in.xsize();
  const size_t ysize = in.ysize();

  Image vst(xsize, ysize);
  for (size_t y = 0; y < ysize; ++y) {
    const float* SIMD_RESTRICT row_in = in.ConstRow(y);
    float* SIMD_RESTRICT row_out = vst.Row(y);
    for (size_t x = 0; x < xsize; ++x) {
      row_out[x] = std::sqrt(row_in[x]);
    }
  }

  auto guide = MakeGuide(vst, min, max);
  const size_t guide_stride = guide.bytes_per_row();

  DifferenceDistribution diff_distribution;
  for (int y = kBorder; y < ysize - kBorder; y += 2) {
    const uint8_t* SIMD_RESTRICT guide_m4 = guide.ConstRow(y - 4);

    for (int x = kBorder; x < xsize - kBorder; x += 2) {
      diff_distribution.Add(guide_m4 + x, guide_stride);
    }
  }

  diff_distribution.Print(message);
  return 0;
}

}  // namespace
}  // namespace SIMD_NAMESPACE

namespace epf {
using namespace pik::SIMD_NAMESPACE;

// Self-guided.
template <>
void EdgePreservingFilter::operator()<SIMD_TARGET>(ImageF* in_out, int sigma,
                                                   float min, float max) {
  WeightFast weight_func;
  weight_func.SetSigma(sigma);
  PadAndFilter(in_out, min, max, weight_func);
}

template <>
void EdgePreservingFilter::operator()<SIMD_TARGET>(Image3F* in_out, int sigma,
                                                   float min, float max) {
  WeightFast weight_func;
  weight_func.SetSigma(sigma);
  PadAndFilter(in_out, min, max, weight_func);
}

// PIK adaptive
template <>
void EdgePreservingFilter::operator()<SIMD_TARGET>(
    Image3F* in_out, const AdaptiveFilterParams& params, float min, float max) {
  Image3F padded = Padding::PadImage(*in_out);
  auto guide = MakeGuide(padded, min, max);
  FilterAdaptive(guide, padded, params, in_out);
}

// Separate guide image.
template <>
void EdgePreservingFilter::operator()<SIMD_TARGET>(const ImageB& guide,
                                                   const ImageF& in, int sigma,
                                                   ImageF* SIMD_RESTRICT out) {
  WeightFast weight_func;
  weight_func.SetSigma(sigma);
  Filter(guide, in, weight_func, out);
}

template <>
void EdgePreservingFilter::operator()<SIMD_TARGET>(const Image3B& guide,
                                                   const Image3F& in, int sigma,
                                                   Image3F* SIMD_RESTRICT out) {
  WeightFast weight_func;
  weight_func.SetSigma(sigma);
  Filter(guide, in, weight_func, out);
}

template <>
void EdgePreservingFilterSlow::operator()<SIMD_TARGET>(ImageF* in_out,
                                                       int sigma, float min,
                                                       float max) {
  WeightExp weight_func;
  weight_func.SetSigma(sigma);
  PadAndFilter(in_out, min, max, weight_func);
}

template <>
void EdgePreservingFilterSlow::operator()<SIMD_TARGET>(Image3F* in_out,
                                                       int sigma, float min,
                                                       float max) {
  WeightExp weight_func;
  weight_func.SetSigma(sigma);
  PadAndFilter(in_out, min, max, weight_func);
}

template <>
void EdgePreservingFilterSlow::operator()<SIMD_TARGET>(
    const ImageB& guide, const ImageF& in, int sigma,
    ImageF* SIMD_RESTRICT out) {
  WeightExp weight_func;
  weight_func.SetSigma(sigma);
  Filter(guide, in, weight_func, out);
}

template <>
void EdgePreservingFilterSlow::operator()<SIMD_TARGET>(
    const Image3B& guide, const Image3F& in, int sigma,
    Image3F* SIMD_RESTRICT out) {
  WeightExp weight_func;
  weight_func.SetSigma(sigma);
  Filter(guide, in, weight_func, out);
}

template <>
void EdgePreservingFilterTwoPhase::operator()<SIMD_TARGET>(
    const ImageF& in, ImageF* SIMD_RESTRICT out, const float min,
    const float max) {
  TwoPhase(in, out, min, max);
}

template <>
void TestEstimateNoise::operator()<SIMD_TARGET>(const char* message,
                                                const ImageF& in,
                                                const float min,
                                                const float max) {
  EstimateNoise(message, in, min, max);
}

template <>
void EdgePreservingFilterTest::operator()<SIMD_TARGET>() {
  Padding::Test();
  InternalWeightTests::Run();
  WeightedSum::Test();
  printf("Tests OK: %s\n", vec_name<DF>());
}

}  // namespace epf
}  // namespace pik
