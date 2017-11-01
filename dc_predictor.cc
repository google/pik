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
using DI = Part<int16_t, kNumPredictors, SIMD_TARGET>;
using DU = Part<uint16_t, kNumPredictors, SIMD_TARGET>;
using D8 = Part<uint8_t, kNumPredictors * 2, SIMD_TARGET>;

// Not the same as avg, which rounds rather than truncates!
template <class V>
PIK_INLINE V Average(const V v0, const V v1) {
  return shift_right<1>(add_sat(v0, v1));
}

// Clamps gradient to the min/max of n, w, l.
template <class V>
PIK_INLINE V ClampedGradient(const V n, const V w, const V l) {
  const V grad = sub_sat(add_sat(n, w), l);
  const V vmin = min(n, min(w, l));
  const V vmax = max(n, max(w, l));
  return min(max(vmin, grad), vmax);
}

template <class V>
PIK_INLINE V AbsResidual(const V c, const V pred) {
  return abs(sub_sat(c, pred));
}

// Returns a shuffle mask for moving lane i to lane 0 (i = argmin abs_costs[i]).
// This is used for selecting the best predictor(s).
PIK_INLINE u8x16 ShuffleForMinCost(const DI::V abs_costs) {
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
  const auto idx_min = ext::minpos(cast_to(DU(), abs_costs));
  const auto idx_idx = shuffle_bytes(idx_min, bytes_from_idx);
  const auto byte_idx = idx_idx + idx_idx;  // shift left by 1 => byte index
  return cast_to(d8, byte_idx) + high_byte;
}

// Return value is broadcasted into all lanes => caller can use any_part.
PIK_INLINE DI::V SelectMinCost(const DI::V pred, const D8::V shuffle) {
  return shuffle_bytes(pred, shuffle);
}

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
  using PixelD = Part<int16_t, 1, SIMD_TARGET>;

  static PIK_INLINE PixelD::V LoadPixel(const DC* const PIK_RESTRICT row,
                                        const size_t x) {
    return set_part(PixelD(), row[x]);
  }

  static PIK_INLINE void StorePixel(const PixelD::V dc,
                                    DC* const PIK_RESTRICT row,
                                    const size_t x) {
    row[x] = get_part(PixelD(), dc);
  }

  static PIK_INLINE DI::V Broadcast(const PixelD::V dc) {
    return broadcast_part<0>(DI(), dc);
  }

  // Loads the neighborhood required for predicting at x = 2. This involves
  // top/middle/bottom rows; if y = 1, row_t == row_m == Row(0).
  PixelNeighborsY(const DC* const PIK_RESTRICT row_ym,
                  const DC* const PIK_RESTRICT row_yb,
                  const DC* const PIK_RESTRICT row_t,
                  const DC* const PIK_RESTRICT row_m,
                  const DC* const PIK_RESTRICT row_b) {
    const DI d;
    const auto wl = set1(d, row_m[0]);
    const auto ww = set1(d, row_b[0]);
    tl_ = set1(d, row_t[1]);
    tn_ = set1(d, row_t[2]);
    l_ = set1(d, row_m[1]);
    n_ = set1(d, row_m[2]);
    w_ = set1(d, row_b[1]);
    pred_w_ = Predict(l_, ww, wl, n_);
  }

  // Estimates "cost" for each predictor by comparing with known n and w.
  PIK_INLINE DI::V PredictorCosts(const size_t x,
                                  const DC* const PIK_RESTRICT row_ym,
                                  const DC* const PIK_RESTRICT row_yb,
                                  const DC* const PIK_RESTRICT row_t) {
    const auto tr = Broadcast(LoadPixel(row_t, x + 1));
    const auto costs =
        AbsResidual(n_, Predict(tn_, l_, tl_, tr)) + AbsResidual(w_, pred_w_);
    tl_ = tn_;
    tn_ = tr;
    return costs;
  }

  // Returns predictor for pixel c with min cost and updates pred_w_.
  PIK_INLINE PixelD::V PredictC(const PixelD::V r, const DI::V costs) {
    const auto pred_c = Predict(n_, w_, l_, Broadcast(r));
    pred_w_ = pred_c;
    const auto shuffle = ShuffleForMinCost(costs);
    return any_part(PixelD(), SelectMinCost(pred_c, shuffle));
  }

  PIK_INLINE void Advance(const PixelD::V r, const PixelD::V c) {
    l_ = n_;
    n_ = Broadcast(r);
    w_ = Broadcast(c);
  }

 private:
  // All arguments are broadcasted.
  static PIK_INLINE DI::V Predict(const DI::V n, const DI::V w, const DI::V l,
                                  const DI::V r) {
    // "x" are invalid/don't care lanes.
    const auto vRN = interleave_lo(n, r);
    const auto v6 = ClampedGradient(n, w, l);
    const auto vLLRN = extract_concat_bytes<12>(l, vRN);
    const auto vNWNWNWNW = interleave_lo(w, n);
    const auto vWxxxLLRN = concat_hi_lo(w, vLLRN);
    const auto vAxxx4321 = Average(vNWNWNWNW, vWxxxLLRN);
    const auto vx765xxxx = interleave_lo(vNWNWNWNW, v6);
    const auto vx7654321 = concat_hi_lo(vx765xxxx, vAxxx4321);
    const auto v0xxxxxxx = Average(vAxxx4321, r);
    // Eight predictors for luminance (decreases coded size by ~0.5% vs four)
    // 0: Average(Average(n, w), r);
    // 1: Average(w, n);
    // 2: Average(n, r);
    // 3: Average(w, l);
    // 4: Average(n, l);
    // 5: w;
    // 6: PredClampedGrad(n, w, l);
    // 7: n;
    return extract_concat_bytes<14>(vx7654321, v0xxxxxxx);
  }

  DI::V tl_;
  DI::V tn_;
  DI::V n_;
  DI::V w_;
  DI::V l_;
  // (30% overall speedup by reusing the current prediction as the next pred_w_)
  DI::V pred_w_;
};

// Providing separate sets of predictors for the luminance and chrominance bands
// reduces the magnitude of residuals, but differentiating between the
// chrominance bands does not.
class PixelNeighborsUV {
 public:
  // UV (U in higher lane, V loaded first).
  using PixelD = Part<int16_t, 2, SIMD_TARGET>;

  static PIK_INLINE PixelD::V LoadPixel(const DC* const PIK_RESTRICT row,
                                        const size_t x) {
    return load(PixelD(), row + 2 * x);
  }

  static PIK_INLINE void StorePixel(const PixelD::V uv,
                                    DC* const PIK_RESTRICT row,
                                    const size_t x) {
    store(uv, PixelD(), row + 2 * x);
  }

  PixelNeighborsUV(const DC* const PIK_RESTRICT row_ym,
                   const DC* const PIK_RESTRICT row_yb,
                   const DC* const PIK_RESTRICT row_t,
                   const DC* const PIK_RESTRICT row_m,
                   const DC* const PIK_RESTRICT row_b) {
    const DI d;
    yn_ = set1(d, row_ym[2]);
    yw_ = set1(d, row_yb[1]);
    yl_ = set1(d, row_ym[1]);
    n_ = LoadPixel(row_m, 2);
    w_ = LoadPixel(row_b, 1);
    l_ = LoadPixel(row_m, 1);
  }

  // Estimates "cost" for each predictor by comparing with known c from Y band.
  PIK_INLINE DI::V PredictorCosts(const size_t x,
                                  const DC* const PIK_RESTRICT row_ym,
                                  const DC* const PIK_RESTRICT row_yb,
                                  const DC* const PIK_RESTRICT) {
    const auto yr = set1(DI(), row_ym[x + 1]);
    const auto yc = set1(DI(), row_yb[x]);
    const auto costs = AbsResidual(yc, Predict(yn_, yw_, yl_, yr));
    yl_ = yn_;
    yn_ = yr;
    yw_ = yc;
    return costs;
  }

  // Returns predictor for pixel c with min cost.
  PIK_INLINE PixelD::V PredictC(const PixelD::V r, const DI::V costs) const {
    const DI::V pred_u =
        Predict(BroadcastU(n_), BroadcastU(w_), BroadcastU(l_), BroadcastU(r));
    const DI::V pred_v =
        Predict(BroadcastV(n_), BroadcastV(w_), BroadcastV(l_), BroadcastV(r));
    const auto shuffle = ShuffleForMinCost(costs);
    const auto best_u = SelectMinCost(pred_u, shuffle);
    const auto best_v = SelectMinCost(pred_v, shuffle);
    return any_part(PixelD(), interleave_lo(best_v, best_u));
  }

  PIK_INLINE void Advance(const PixelD::V r, const PixelD::V c) {
    l_ = n_;
    n_ = r;
    w_ = c;
  }

 private:
  static PIK_INLINE DI::V BroadcastU(const PixelD::V uv) {
    return broadcast_part<1>(DI(), uv);
  }
  static PIK_INLINE DI::V BroadcastV(const PixelD::V uv) {
    return broadcast_part<0>(DI(), uv);
  }

  // All arguments are broadcasted.
  static PIK_INLINE DI::V Predict(const DI::V n, const DI::V w, const DI::V l,
                                  const DI::V r) {
    // "x" lanes are unused.
    const auto v0 = ClampedGradient(n, w, l);
    const auto vRN = interleave_lo(n, r);
    const auto vW0 = interleave_lo(v0, w);
    const auto vLNN = extract_concat_bytes<12>(l, n);
    const auto vWRWR = interleave_lo(r, w);
    const auto vLNNW = extract_concat_bytes<14>(vLNN, w);
    const auto vRWN0 = interleave_lo(vW0, vRN);
    const auto v531A = Average(vLNNW, vWRWR);
    const auto v6543210x = interleave_lo(v531A, vRWN0);
    const auto v7 = Average(v531A, n);
    // Eight predictors for chrominance:
    // 0: ClampedGrad(n, w, l);
    // 1: Average2(n, w);
    // 2: n;
    // 3: Average2(n, r);
    // 4: w;
    // 5: Average2(w, l);
    // 6: r;
    // 7: Average2(Average2(w, r), n);
    return extract_concat_bytes<2>(v7, v6543210x);
  }

  DI::V yn_;
  DI::V yw_;
  DI::V yl_;
  PixelD::V n_;
  PixelD::V w_;
  PixelD::V l_;
};

// Computes residuals of a fixed predictor (the preceding pixel W).
// Useful for Row(0) because no preceding row is required.
template <class N>
struct FixedW {
  static PIK_INLINE void Shrink(const size_t xsize,
                                const DC* const PIK_RESTRICT dc,
                                DC* const PIK_RESTRICT residuals) {
    N::StorePixel(N::LoadPixel(dc, 0), residuals, 0);
    for (size_t x = 1; x < xsize; ++x) {
      N::StorePixel(N::LoadPixel(dc, x) - N::LoadPixel(dc, x - 1), residuals,
                    x);
    }
  }

  static PIK_INLINE void Expand(const size_t xsize,
                                const DC* const PIK_RESTRICT residuals,
                                DC* const PIK_RESTRICT dc) {
    N::StorePixel(N::LoadPixel(residuals, 0), dc, 0);
    for (size_t x = 1; x < xsize; ++x) {
      N::StorePixel(N::LoadPixel(dc, x - 1) + N::LoadPixel(residuals, x), dc,
                    x);
    }
  }
};

// Predicts x = 0 with n, x = 1 with w; this decreases the overall abs
// residuals by 6% vs FixedW, which stores the first coefficient directly.
template <class N>
struct LeftBorder2 {
  static PIK_INLINE void Shrink(const size_t xsize,
                                const DC* const PIK_RESTRICT row_m,
                                const DC* const PIK_RESTRICT row_b,
                                DC* const PIK_RESTRICT residuals) {
    N::StorePixel(N::LoadPixel(row_b, 0) - N::LoadPixel(row_m, 0), residuals,
                  0);
    if (xsize >= 2) {
      // TODO(user): Clamped gradient should be slightly better here.
      N::StorePixel(N::LoadPixel(row_b, 1) - N::LoadPixel(row_b, 0), residuals,
                    1);
    }
  }

  static PIK_INLINE void Expand(const size_t xsize,
                                const DC* const PIK_RESTRICT residuals,
                                const DC* const PIK_RESTRICT row_m,
                                DC* const PIK_RESTRICT row_b) {
    N::StorePixel(N::LoadPixel(row_m, 0) + N::LoadPixel(residuals, 0), row_b,
                  0);
    if (xsize >= 2) {
      N::StorePixel(N::LoadPixel(row_b, 0) + N::LoadPixel(residuals, 1), row_b,
                    1);
    }
  }
};

// Predicts the final x with w, necessary because PixelNeighbors* require "r".
template <class N>
struct RightBorder1 {
  static PIK_INLINE void Shrink(const size_t xsize,
                                const DC* const PIK_RESTRICT dc,
                                DC* const PIK_RESTRICT residuals) {
    // TODO(user): Clamped gradient should be slightly better here.
    if (xsize >= 2) {
      const auto res =
          N::LoadPixel(dc, xsize - 1) - N::LoadPixel(dc, xsize - 2);
      N::StorePixel(res, residuals, xsize - 1);
    }
  }

  static PIK_INLINE void Expand(const size_t xsize,
                                const DC* const PIK_RESTRICT residuals,
                                DC* const PIK_RESTRICT dc) {
    if (xsize >= 2) {
      const auto uv =
          N::LoadPixel(dc, xsize - 2) + N::LoadPixel(residuals, xsize - 1);
      N::StorePixel(uv, dc, xsize - 1);
    }
  }
};

// Selects predictor based upon its error at the prior n and w pixels.
// Requires two preceding rows (t, m) and the current row b. The row_y*
// pointers are unused and may be null if N = PixelNeighborsY.
template <class N>
class Adaptive {
  using PixelV = typename N::PixelD::V;

 public:
  static void Shrink(const size_t xsize, const DC* const PIK_RESTRICT row_ym,
                     const DC* const PIK_RESTRICT row_yb,
                     const DC* const PIK_RESTRICT row_t,
                     const DC* const PIK_RESTRICT row_m,
                     const DC* const PIK_RESTRICT row_b,
                     DC* const PIK_RESTRICT residuals) {
    LeftBorder2<N>::Shrink(xsize, row_m, row_b, residuals);

    ForeachPrediction(xsize, row_ym, row_yb, row_t, row_m, row_b,
                      [row_b, residuals](const size_t x, const PixelV pred) {
                        const auto c = N::LoadPixel(row_b, x);
                        N::StorePixel(c - pred, residuals, x);
                        return c;
                      });

    RightBorder1<N>::Shrink(xsize, row_b, residuals);
  }

  static void Expand(const size_t xsize, const DC* const PIK_RESTRICT row_ym,
                     const DC* const PIK_RESTRICT row_yb,
                     const DC* const PIK_RESTRICT residuals,
                     const DC* const PIK_RESTRICT row_t,
                     const DC* const PIK_RESTRICT row_m,
                     DC* const PIK_RESTRICT row_b) {
    LeftBorder2<N>::Expand(xsize, residuals, row_m, row_b);

    ForeachPrediction(xsize, row_ym, row_yb, row_t, row_m, row_b,
                      [row_b, residuals](const size_t x, const PixelV pred) {
                        const auto c = pred + N::LoadPixel(residuals, x);
                        N::StorePixel(c, row_b, x);
                        return c;
                      });

    RightBorder1<N>::Expand(xsize, residuals, row_b);
  }

 private:
  // "Func" returns the current pixel, dc[x].
  template <class Func>
  static PIK_INLINE void ForeachPrediction(const size_t xsize,
                                           const DC* const PIK_RESTRICT row_ym,
                                           const DC* const PIK_RESTRICT row_yb,
                                           const DC* const PIK_RESTRICT row_t,
                                           const DC* const PIK_RESTRICT row_m,
                                           const DC* const PIK_RESTRICT row_b,
                                           const Func& func) {
    if (xsize < 2) {
      return;  // Avoid out of bounds reads.
    }
    N neighbors(row_ym, row_yb, row_t, row_m, row_b);
    // PixelNeighborsY uses w at x - 1 => two pixel margin.
    for (size_t x = 2; x < xsize - 1; ++x) {
      const auto r = N::LoadPixel(row_m, x + 1);
      const auto costs = neighbors.PredictorCosts(x, row_ym, row_yb, row_t);
      const auto pred_c = neighbors.PredictC(r, costs);
      const auto c = func(x, pred_c);
      neighbors.Advance(r, c);
    }
  }
};


void ShrinkY(const Image<DC>& dc, Image<DC>* const PIK_RESTRICT residuals) {
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

void ShrinkUV(const Image<DC>& dc_y, const Image<DC>& dc,
              Image<DC>* const PIK_RESTRICT residuals) {
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

void ExpandY(const Image<DC>& residuals, Image<DC>* const PIK_RESTRICT dc) {
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

void ExpandUV(const Image<DC>& dc_y, const Image<DC>& residuals,
              Image<DC>* const PIK_RESTRICT dc) {
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

void ShrinkY(const Image<DC>& dc, Image<DC>* const PIK_RESTRICT residuals) {
  SIMD_NAMESPACE::ShrinkY(dc, residuals);
}

void ShrinkUV(const Image<DC>& dc_y, const Image<DC>& dc,
              Image<DC>* const PIK_RESTRICT residuals) {
  SIMD_NAMESPACE::ShrinkUV(dc_y, dc, residuals);
}

void ExpandY(const Image<DC>& residuals, Image<DC>* const PIK_RESTRICT dc) {
  SIMD_NAMESPACE::ExpandY(residuals, dc);
}

void ExpandUV(const Image<DC>& dc_y, const Image<DC>& residuals,
              Image<DC>* const PIK_RESTRICT dc) {
  SIMD_NAMESPACE::ExpandUV(dc_y, residuals, dc);
}

}  // namespace pik
