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

#include "dc_predictor.h"

#include <stddef.h>

#include "compiler_specific.h"
#include "simd/simd.h"

namespace pik {
namespace SIMD_NAMESPACE {
namespace {

constexpr size_t kNumPredictors = 8;
#if SIMD_TARGET_VALUE == SIMD_NONE
using DI = Scalar<int16_t>;
// For predictors and costs.
struct VIx8 {
  DI::V lanes[kNumPredictors];
};
// For U, V.
struct VIx2 {
  DI::V lanes[2];
};
PIK_INLINE VIx2 operator+(const VIx2& a, const VIx2& b) {
  VIx2 ret;
  ret.lanes[0] = a.lanes[0] + b.lanes[0];
  ret.lanes[1] = a.lanes[1] + b.lanes[1];
  return ret;
}
PIK_INLINE VIx2 operator-(const VIx2& a, const VIx2& b) {
  VIx2 ret;
  ret.lanes[0] = a.lanes[0] - b.lanes[0];
  ret.lanes[1] = a.lanes[1] - b.lanes[1];
  return ret;
}
#else
using DI = Part<int16_t, kNumPredictors>;
using VIx8 = DI::V;
using VIx2 = Part<int16_t, 2>::V;
#endif

// Not the same as avg, which rounds rather than truncates!
template <class V>
PIK_INLINE V Average(const V v0, const V v1) {
  return shift_right<1>(saturated_add(v0, v1));
}

// Clamps gradient to the min/max of n, w, l.
template <class V>
PIK_INLINE V ClampedGradient(const V n, const V w, const V l) {
  const V grad = saturated_subtract(saturated_add(n, w), l);
  const V vmin = min(n, min(w, l));
  const V vmax = max(n, max(w, l));
  return min(max(vmin, grad), vmax);
}

template <class V>
PIK_INLINE V AbsResidual(const V c, const V pred) {
  return abs(saturated_subtract(c, pred));
}

#if SIMD_TARGET_VALUE == SIMD_NONE

PIK_INLINE size_t IndexOfMinCost(const VIx8& abs_costs) {
  const DI d;
  // Algorithm must exactly match minpos_epu16.
  size_t idx_pred = 0;
  int16_t min_cost = get_part(d, abs_costs.lanes[0]);
  for (size_t i = 0; i < kNumPredictors; ++i) {
    const int16_t cost = get_part(d, abs_costs.lanes[i]);
    if (cost < min_cost) {
      min_cost = cost;
      idx_pred = i;
    }
  }
  return idx_pred;
}

#else

// Returns a shuffle mask for moving lane i to lane 0 (i = argmin abs_costs[i]).
// This is used for selecting the best predictor(s). The shuffle also broadcasts
// the result to all lanes so that callers can use any_part.
PIK_INLINE u8x16 ShuffleForMinCost(const VIx8 abs_costs) {
  using D8 = Part<uint8_t, kNumPredictors * 2>;
  const D8 d8;
  // Replicates index16 returned from minpos into all bytes.
  SIMD_ALIGN const uint8_t kIdx[16] = {2, 2, 2, 2, 2, 2, 2, 2,
                                       2, 2, 2, 2, 2, 2, 2, 2};
  // Offset for the most significant byte in each 16-bit pair.
  SIMD_ALIGN const uint8_t kHighByte[16] = {0, 1, 0, 1, 0, 1, 0, 1,
                                            0, 1, 0, 1, 0, 1, 0, 1};
  const auto bytes_from_idx = load(d8, kIdx);
  const auto high_byte = load(d8, kHighByte);
  // Note: minpos is unsigned; LimitsMin (a large absolute value) will have a
  // higher cost than any other value.
  using DU = Part<uint16_t, kNumPredictors>;
  const auto idx_min = ext::minpos(cast_to(DU(), abs_costs));
  const auto idx_idx = table_lookup_bytes(idx_min, bytes_from_idx);
  const auto byte_idx = idx_idx + idx_idx;  // shift left by 1 => byte index
  return cast_to(d8, byte_idx) + high_byte;
}

#endif

// Sliding window of "causal" (already decoded) pixels, plus simple functions
// to predict the next pixel "c" from its neighbors: l n r
// The single-letter names shorten identifiers.      w c
//
// Predictions are more accurate when the preceding w pixel is available, but
// this interferes with SIMD because subsequent pixels depend on the decoding
// of their predecessor. The encoder can compute residuals in parallel because
// it knows all DC values up front, but its speed is less important. A diagonal
// 'wavefront' order would allow computing multiple predictions efficiently,
// but scattering those to the corresponding pixel positions would be slow.
// Interleaving pixels by the lane count (eight pixels with x mod 8 = 0, etc)
// would work if the two pixels before each prediction are already known, but
// scattering lanes to multiples of 10 would also be slow.
//
// We instead compute the various predictors using SIMD, especially because
// many of them are similar. Horizontal operations are generally inefficient,
// but we take advantage of special hardware support for video codecs (minpos).
//
// The set of 8 predictors was chosen from a set of 16 as the combination that
// minimized a simple model of encoding cost. Their order matters because
// minpos(lanes) returns the lowest i with lanes[i] == min. We again retained
// the permutation with the lowest encoding cost.
class PixelNeighborsY {
 public:
  // Single Y value.
  using PixelD = Part<int16_t, 1>;
  using PixelV = PixelD::V;

  static PIK_INLINE PixelV Load(const DC* PIK_RESTRICT row, const size_t x) {
    return set_part(PixelD(), row[x]);
  }

  static PIK_INLINE void Store(const PixelV dc, DC* PIK_RESTRICT row,
                               const size_t x) {
    row[x] = get_part(PixelD(), dc);
  }

  static PIK_INLINE DI::V Broadcast(const PixelV dc) {
    return broadcast_part<0>(DI(), dc);
  }

  // Loads the neighborhood required for predicting at x = 2. This involves
  // top/middle/bottom rows; if y = 1, row_t == row_m == Row(0).
  PixelNeighborsY(const DC* PIK_RESTRICT row_ym, const DC* PIK_RESTRICT row_yb,
                  const DC* PIK_RESTRICT row_t, const DC* PIK_RESTRICT row_m,
                  const DC* PIK_RESTRICT row_b) {
    const DI d;
    const auto wl = set1(d, row_m[0]);
    const auto ww = set1(d, row_b[0]);
    tl_ = set1(d, row_t[1]);
    tn_ = set1(d, row_t[2]);
    l_ = set1(d, row_m[1]);
    n_ = set1(d, row_m[2]);
    w_ = set1(d, row_b[1]);
    Predict(l_, ww, wl, n_, &pred_w_);
  }

  // Estimates "cost" for each predictor by comparing with known n and w.
  PIK_INLINE void PredictorCosts(const size_t x, const DC* PIK_RESTRICT row_ym,
                                 const DC* PIK_RESTRICT row_yb,
                                 const DC* PIK_RESTRICT row_t,
                                 VIx8* PIK_RESTRICT costs) {
    const auto tr = Broadcast(Load(row_t, x + 1));
    VIx8 pred_n;
    Predict(tn_, l_, tl_, tr, &pred_n);
#if SIMD_TARGET_VALUE == SIMD_NONE
    for (size_t i = 0; i < kNumPredictors; ++i) {
      costs->lanes[i] =
          AbsResidual(n_, pred_n.lanes[i]) + AbsResidual(w_, pred_w_.lanes[i]);
    }
#else
    *costs = AbsResidual(n_, pred_n) + AbsResidual(w_, pred_w_);
#endif
    tl_ = tn_;
    tn_ = tr;
  }

  // Returns predictor for pixel c with min cost and updates pred_w_.
  PIK_INLINE PixelV PredictC(const PixelV r, const VIx8 costs) {
    VIx8 pred_c;
    Predict(n_, w_, l_, Broadcast(r), &pred_c);
    pred_w_ = pred_c;
#if SIMD_TARGET_VALUE == SIMD_NONE
    return pred_c.lanes[IndexOfMinCost(costs)];
#else
    return any_part(PixelD(),
                    table_lookup_bytes(pred_c, ShuffleForMinCost(costs)));
#endif
  }

  PIK_INLINE void Advance(const PixelV r, const PixelV c) {
    l_ = n_;
    n_ = Broadcast(r);
    w_ = Broadcast(c);
  }

 private:
  // All input arguments are broadcasted.
  static PIK_INLINE void Predict(const DI::V n, const DI::V w, const DI::V l,
                                 const DI::V r, VIx8* PIK_RESTRICT pred) {
#if SIMD_TARGET_VALUE == SIMD_NONE
    // Eight predictors for luminance (decreases coded size by ~0.5% vs four)
    pred->lanes[0] = Average(Average(n, w), r);
    pred->lanes[1] = Average(w, n);
    pred->lanes[2] = Average(n, r);
    pred->lanes[3] = Average(w, l);
    pred->lanes[4] = Average(n, l);
    pred->lanes[5] = w;
    pred->lanes[6] = ClampedGradient(n, w, l);
    pred->lanes[7] = n;
#else
    // "x" are invalid/don't care lanes.
    const auto vRN = interleave_lo(n, r);
    const auto v6 = ClampedGradient(n, w, l);
    const auto vLLRN = combine_shift_right_bytes<12>(l, vRN);
    const auto vNWNWNWNW = interleave_lo(w, n);
    const auto vWxxxLLRN = concat_hi_lo(w, vLLRN);
    const auto vAxxx4321 = Average(vNWNWNWNW, vWxxxLLRN);
    const auto vx765xxxx = interleave_lo(vNWNWNWNW, v6);
    const auto vx7654321 = concat_hi_lo(vx765xxxx, vAxxx4321);
    const auto v0xxxxxxx = Average(vAxxx4321, r);
    *pred = combine_shift_right_bytes<14>(vx7654321, v0xxxxxxx);
#endif
  }

  DI::V tl_;
  DI::V tn_;
  DI::V n_;
  DI::V w_;
  DI::V l_;
  // (30% overall speedup by reusing the current prediction as the next pred_w_)
  VIx8 pred_w_;
};

// Providing separate sets of predictors for the luminance and chrominance bands
// reduces the magnitude of residuals, but differentiating between the
// chrominance bands does not.
class PixelNeighborsUV {
 public:
#if SIMD_TARGET_VALUE != SIMD_NONE
  using PixelD = Part<int16_t, 2>;
#endif
  using PixelV = VIx2;

  // U in lane1, V in lane0.
  static PIK_INLINE PixelV Load(const DC* PIK_RESTRICT row, const size_t x) {
#if SIMD_TARGET_VALUE == SIMD_NONE
    PixelV ret;
    ret.lanes[0] = load(DI(), row + 2 * x + 0);  // V
    ret.lanes[1] = load(DI(), row + 2 * x + 1);  // U
    return ret;
#else
    return load(PixelD(), row + 2 * x);
#endif
  }

  static PIK_INLINE void Store(const PixelV uv, DC* PIK_RESTRICT row,
                               const size_t x) {
#if SIMD_TARGET_VALUE == SIMD_NONE
    store(uv.lanes[0], DI(), row + 2 * x + 0);  // V
    store(uv.lanes[1], DI(), row + 2 * x + 1);  // U
#else
    store(uv, PixelD(), row + 2 * x);
#endif
  }

  PixelNeighborsUV(const DC* PIK_RESTRICT row_ym, const DC* PIK_RESTRICT row_yb,
                   const DC* PIK_RESTRICT row_t, const DC* PIK_RESTRICT row_m,
                   const DC* PIK_RESTRICT row_b) {
    const DI d;
    yn_ = set1(d, row_ym[2]);
    yw_ = set1(d, row_yb[1]);
    yl_ = set1(d, row_ym[1]);
    n_ = Load(row_m, 2);
    w_ = Load(row_b, 1);
    l_ = Load(row_m, 1);
  }

  // Estimates "cost" for each predictor by comparing with known c from Y band.
  PIK_INLINE void PredictorCosts(const size_t x, const DC* PIK_RESTRICT row_ym,
                                 const DC* PIK_RESTRICT row_yb,
                                 const DC* PIK_RESTRICT,
                                 VIx8* PIK_RESTRICT costs) {
    const auto yr = set1(DI(), row_ym[x + 1]);
    const auto yc = set1(DI(), row_yb[x]);
    VIx8 pred_y;
    Predict(yn_, yw_, yl_, yr, &pred_y);
#if SIMD_TARGET_VALUE == SIMD_NONE
    for (size_t i = 0; i < kNumPredictors; ++i) {
      costs->lanes[i] = AbsResidual(yc, pred_y.lanes[i]);
    }
#else
    *costs = AbsResidual(yc, pred_y);
#endif
    yl_ = yn_;
    yn_ = yr;
    yw_ = yc;
  }

  // Returns predictor for pixel c with min cost.
  PIK_INLINE PixelV PredictC(const PixelV r, const VIx8& costs) const {
    VIx8 u, v;
    Predict(BroadcastU(n_), BroadcastU(w_), BroadcastU(l_), BroadcastU(r), &u);
    Predict(BroadcastV(n_), BroadcastV(w_), BroadcastV(l_), BroadcastV(r), &v);

#if SIMD_TARGET_VALUE == SIMD_NONE
    const size_t idx_pred = IndexOfMinCost(costs);
    PixelV ret;
    ret.lanes[0] = v.lanes[idx_pred];
    ret.lanes[1] = u.lanes[idx_pred];
    return ret;
#else
    const auto shuffle = ShuffleForMinCost(costs);
    const auto best_u = table_lookup_bytes(u, shuffle);
    const auto best_v = table_lookup_bytes(v, shuffle);
    return any_part(PixelD(), interleave_lo(best_v, best_u));
#endif
  }

  PIK_INLINE void Advance(const PixelV r, const PixelV c) {
    l_ = n_;
    n_ = r;
    w_ = c;
  }

 private:
  static PIK_INLINE DI::V BroadcastU(const PixelV uv) {
#if SIMD_TARGET_VALUE == SIMD_NONE
    return uv.lanes[1];
#else
    return broadcast_part<1>(DI(), uv);
#endif
  }
  static PIK_INLINE DI::V BroadcastV(const PixelV uv) {
#if SIMD_TARGET_VALUE == SIMD_NONE
    return uv.lanes[0];
#else
    return broadcast_part<0>(DI(), uv);
#endif
  }

  // All arguments are broadcasted.
  static PIK_INLINE void Predict(const DI::V n, const DI::V w, const DI::V l,
                                 const DI::V r, VIx8* PIK_RESTRICT pred) {
#if SIMD_TARGET_VALUE == SIMD_NONE
    // Eight predictors for chrominance:
    pred->lanes[0] = ClampedGradient(n, w, l);
    pred->lanes[1] = Average(n, w);
    pred->lanes[2] = n;
    pred->lanes[3] = Average(n, r);
    pred->lanes[4] = w;
    pred->lanes[5] = Average(w, l);
    pred->lanes[6] = r;
    pred->lanes[7] = Average(Average(w, r), n);
#else
    // "x" lanes are unused.
    const auto v0 = ClampedGradient(n, w, l);
    const auto vRN = interleave_lo(n, r);
    const auto vW0 = interleave_lo(v0, w);
    const auto vLNN = combine_shift_right_bytes<12>(l, n);
    const auto vWRWR = interleave_lo(r, w);
    const auto vLNNW = combine_shift_right_bytes<14>(vLNN, w);
    const auto vRWN0 = interleave_lo(vW0, vRN);
    const auto v531A = Average(vLNNW, vWRWR);
    const auto v6543210x = interleave_lo(v531A, vRWN0);
    const auto v7 = Average(v531A, n);
    *pred = combine_shift_right_bytes<2>(v7, v6543210x);
#endif
  }

  DI::V yn_;
  DI::V yw_;
  DI::V yl_;
  PixelV n_;
  PixelV w_;
  PixelV l_;
};

// Computes residuals of a fixed predictor (the preceding pixel W).
// Useful for Row(0) because no preceding row is required.
template <class N>
struct FixedW {
  static PIK_INLINE void Shrink(const size_t xsize, const DC* PIK_RESTRICT dc,
                                DC* PIK_RESTRICT residuals) {
    N::Store(N::Load(dc, 0), residuals, 0);
    for (size_t x = 1; x < xsize; ++x) {
      N::Store(N::Load(dc, x) - N::Load(dc, x - 1), residuals, x);
    }
  }

  static PIK_INLINE void Expand(const size_t xsize,
                                const DC* PIK_RESTRICT residuals,
                                DC* PIK_RESTRICT dc) {
    N::Store(N::Load(residuals, 0), dc, 0);
    for (size_t x = 1; x < xsize; ++x) {
      N::Store(N::Load(dc, x - 1) + N::Load(residuals, x), dc, x);
    }
  }
};

// Predicts x = 0 with n, x = 1 with w; this decreases the overall abs
// residuals by 6% vs FixedW, which stores the first coefficient directly.
template <class N>
struct LeftBorder2 {
  static PIK_INLINE void Shrink(const size_t xsize,
                                const DC* PIK_RESTRICT row_m,
                                const DC* PIK_RESTRICT row_b,
                                DC* PIK_RESTRICT residuals) {
    N::Store(N::Load(row_b, 0) - N::Load(row_m, 0), residuals, 0);
    if (xsize >= 2) {
      // TODO(user): Clamped gradient should be slightly better here.
      N::Store(N::Load(row_b, 1) - N::Load(row_b, 0), residuals, 1);
    }
  }

  static PIK_INLINE void Expand(const size_t xsize,
                                const DC* PIK_RESTRICT residuals,
                                const DC* PIK_RESTRICT row_m,
                                DC* PIK_RESTRICT row_b) {
    N::Store(N::Load(row_m, 0) + N::Load(residuals, 0), row_b, 0);
    if (xsize >= 2) {
      N::Store(N::Load(row_b, 0) + N::Load(residuals, 1), row_b, 1);
    }
  }
};

// Predicts the final x with w, necessary because PixelNeighbors* require "r".
template <class N>
struct RightBorder1 {
  static PIK_INLINE void Shrink(const size_t xsize, const DC* PIK_RESTRICT dc,
                                DC* PIK_RESTRICT residuals) {
    // TODO(user): Clamped gradient should be slightly better here.
    if (xsize >= 2) {
      const auto res = N::Load(dc, xsize - 1) - N::Load(dc, xsize - 2);
      N::Store(res, residuals, xsize - 1);
    }
  }

  static PIK_INLINE void Expand(const size_t xsize,
                                const DC* PIK_RESTRICT residuals,
                                DC* PIK_RESTRICT dc) {
    if (xsize >= 2) {
      const auto uv = N::Load(dc, xsize - 2) + N::Load(residuals, xsize - 1);
      N::Store(uv, dc, xsize - 1);
    }
  }
};

// Selects predictor based upon its error at the prior n and w pixels.
// Requires two preceding rows (t, m) and the current row b. The row_y*
// pointers are unused and may be null if N = PixelNeighborsY.
template <class N>
class Adaptive {
  using PixelV = typename N::PixelV;

 public:
  static void Shrink(const size_t xsize, const DC* PIK_RESTRICT row_ym,
                     const DC* PIK_RESTRICT row_yb,
                     const DC* PIK_RESTRICT row_t, const DC* PIK_RESTRICT row_m,
                     const DC* PIK_RESTRICT row_b, DC* PIK_RESTRICT residuals) {
    LeftBorder2<N>::Shrink(xsize, row_m, row_b, residuals);

    ForeachPrediction(xsize, row_ym, row_yb, row_t, row_m, row_b,
                      [row_b, residuals](const size_t x, const PixelV pred) {
                        const auto c = N::Load(row_b, x);
                        N::Store(c - pred, residuals, x);
                        return c;
                      });

    RightBorder1<N>::Shrink(xsize, row_b, residuals);
  }

  static void Expand(const size_t xsize, const DC* PIK_RESTRICT row_ym,
                     const DC* PIK_RESTRICT row_yb,
                     const DC* PIK_RESTRICT residuals,
                     const DC* PIK_RESTRICT row_t, const DC* PIK_RESTRICT row_m,
                     DC* PIK_RESTRICT row_b) {
    LeftBorder2<N>::Expand(xsize, residuals, row_m, row_b);

    ForeachPrediction(xsize, row_ym, row_yb, row_t, row_m, row_b,
                      [row_b, residuals](const size_t x, const PixelV pred) {
                        const auto c = pred + N::Load(residuals, x);
                        N::Store(c, row_b, x);
                        return c;
                      });

    RightBorder1<N>::Expand(xsize, residuals, row_b);
  }

 private:
  // "Func" returns the current pixel, dc[x].
  template <class Func>
  static PIK_INLINE void ForeachPrediction(const size_t xsize,
                                           const DC* PIK_RESTRICT row_ym,
                                           const DC* PIK_RESTRICT row_yb,
                                           const DC* PIK_RESTRICT row_t,
                                           const DC* PIK_RESTRICT row_m,
                                           const DC* PIK_RESTRICT row_b,
                                           const Func& func) {
    if (xsize < 2) {
      return;  // Avoid out of bounds reads.
    }
    N neighbors(row_ym, row_yb, row_t, row_m, row_b);
    // PixelNeighborsY uses w at x - 1 => two pixel margin.
    for (size_t x = 2; x < xsize - 1; ++x) {
      const auto r = N::Load(row_m, x + 1);
      VIx8 costs;
      neighbors.PredictorCosts(x, row_ym, row_yb, row_t, &costs);
      const auto pred_c = neighbors.PredictC(r, costs);
      const auto c = func(x, pred_c);
      neighbors.Advance(r, c);
    }
  }
};

void ShrinkY(const ImageS& dc, ImageS* PIK_RESTRICT residuals) {
  const size_t xsize = dc.xsize();
  const size_t ysize = dc.ysize();

  FixedW<PixelNeighborsY>::Shrink(xsize, dc.Row(0), residuals->Row(0));

  if (ysize >= 2) {
    // Only one previous row, so row_t == row_m.
    Adaptive<PixelNeighborsY>::Shrink(xsize, nullptr, nullptr, dc.Row(0),
                                      dc.Row(0), dc.Row(1), residuals->Row(1));
  }

  for (size_t y = 2; y < ysize; ++y) {
    Adaptive<PixelNeighborsY>::Shrink(xsize, nullptr, nullptr, dc.Row(y - 2),
                                      dc.Row(y - 1), dc.Row(y),
                                      residuals->Row(y));
  }
}

void ShrinkUV(const ImageS& dc_y, const ImageS& dc,
              ImageS* PIK_RESTRICT residuals) {
  const size_t xsize = dc.xsize() / 2;
  const size_t ysize = dc.ysize();

  FixedW<PixelNeighborsUV>::Shrink(xsize, dc.Row(0), residuals->Row(0));

  if (ysize >= 2) {
    // Only one previous row, so row_t == row_m.
    Adaptive<PixelNeighborsUV>::Shrink(xsize, dc_y.Row(0), dc_y.Row(1),
                                       dc.Row(0), dc.Row(0), dc.Row(1),
                                       residuals->Row(1));
  }

  for (size_t y = 2; y < ysize; ++y) {
    Adaptive<PixelNeighborsUV>::Shrink(xsize, dc_y.Row(y - 1), dc_y.Row(y),
                                       dc.Row(y - 2), dc.Row(y - 1), dc.Row(y),
                                       residuals->Row(y));
  }
}

void ExpandY(const ImageS& residuals, ImageS* PIK_RESTRICT dc) {
  const size_t xsize = dc->xsize();
  const size_t ysize = dc->ysize();

  FixedW<PixelNeighborsY>::Expand(xsize, residuals.Row(0), dc->Row(0));

  if (ysize >= 2) {
    Adaptive<PixelNeighborsY>::Expand(xsize, nullptr, nullptr, residuals.Row(1),
                                      dc->Row(0), dc->Row(0), dc->Row(1));
  }

  for (size_t y = 2; y < ysize; ++y) {
    Adaptive<PixelNeighborsY>::Expand(xsize, nullptr, nullptr, residuals.Row(y),
                                      dc->Row(y - 2), dc->Row(y - 1),
                                      dc->Row(y));
  }
}

void ExpandUV(const ImageS& dc_y, const ImageS& residuals,
              ImageS* PIK_RESTRICT dc) {
  const size_t xsize = dc->xsize() / 2;
  const size_t ysize = dc->ysize();

  FixedW<PixelNeighborsUV>::Expand(xsize, residuals.Row(0), dc->Row(0));

  if (ysize >= 2) {
    Adaptive<PixelNeighborsUV>::Expand(xsize, dc_y.Row(0), dc_y.Row(1),
                                       residuals.Row(1), dc->Row(0), dc->Row(0),
                                       dc->Row(1));
  }

  for (size_t y = 2; y < ysize; ++y) {
    Adaptive<PixelNeighborsUV>::Expand(xsize, dc_y.Row(y - 1), dc_y.Row(y),
                                       residuals.Row(y), dc->Row(y - 2),
                                       dc->Row(y - 1), dc->Row(y));
  }
}

}  // namespace
}  // namespace SIMD_NAMESPACE

void ShrinkY(const ImageS& dc, ImageS* PIK_RESTRICT residuals) {
  SIMD_NAMESPACE::ShrinkY(dc, residuals);
}

void ShrinkUV(const ImageS& dc_y, const ImageS& dc,
              ImageS* PIK_RESTRICT residuals) {
  SIMD_NAMESPACE::ShrinkUV(dc_y, dc, residuals);
}

void ExpandY(const ImageS& residuals, ImageS* PIK_RESTRICT dc) {
  SIMD_NAMESPACE::ExpandY(residuals, dc);
}

void ExpandUV(const ImageS& dc_y, const ImageS& residuals,
              ImageS* PIK_RESTRICT dc) {
  SIMD_NAMESPACE::ExpandUV(dc_y, residuals, dc);
}

}  // namespace pik
