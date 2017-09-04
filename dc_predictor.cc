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

#include "vector128.h"
#include "vector256.h"

namespace pik {
namespace PIK_TARGET_NAME {
namespace {

template <class V>
static PIK_INLINE V Average(const V& v0, const V& v1) {
  return (v0 + v1) >> 1;
}

// Clamps gradient to the min/max of n, w, l.
template <class V>
static PIK_INLINE V ClampedGradient(const V& n, const V& w, const V& l) {
  const V grad = n + w - l;
  const V min = Min(n, Min(w, l));
  const V max = Max(n, Max(w, l));
  return Min(Max(min, grad), max);
}

static PIK_INLINE V8x32I AbsResidual(const V8x32I& c, const V8x32I& pred) {
  return V8x32I(_mm256_abs_epi32(c - pred));
}

static PIK_INLINE V8x16U Costs16(const V8x32I& costs) {
  // Saturate to 16-bit for minpos; due to 128-bit interleaving, only the lower
  // 64 bits of each half are valid.
  const V16x16U costs7654_3210(_mm256_packus_epi32(costs, costs));
  const V8x16U costs3210(_mm256_extracti128_si256(costs7654_3210, 0));
  const V8x16U costs7654(_mm256_extracti128_si256(costs7654_3210, 1));
  return V8x16U(_mm_unpacklo_epi64(costs3210, costs7654));
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
  using V = V8x32I;

 public:
  // LoadT/StoreT/compute single Y values.
  using T = V4x32I;
  static PIK_INLINE T LoadT(const DC* const PIK_RESTRICT row, const size_t x) {
    return T(_mm_cvtsi32_si128(row[x]));
  }

  static PIK_INLINE void StoreT(const T& dc, DC* const PIK_RESTRICT row,
                                const size_t x) {
    row[x] = _mm_cvtsi128_si32(dc);
  }

  static PIK_INLINE V Broadcast(const T& dc) {
    return V(_mm256_broadcastd_epi32(dc));
  }

  // Loads the neighborhood required for predicting at x = 2. This involves
  // top/middle/bottom rows; if y = 1, row_t == row_m == Row(0).
  PixelNeighborsY(const DC* const PIK_RESTRICT row_ym,
                  const DC* const PIK_RESTRICT row_yb,
                  const DC* const PIK_RESTRICT row_t,
                  const DC* const PIK_RESTRICT row_m,
                  const DC* const PIK_RESTRICT row_b) {
    const V wl(row_m[0]);
    const V ww(row_b[0]);
    tl_ = V(row_t[1]);
    tn_ = V(row_t[2]);
    l_ = V(row_m[1]);
    n_ = V(row_m[2]);
    w_ = V(row_b[1]);
    pred_w_ = Predict(l_, ww, wl, n_);
  }

  // Estimates "cost" for each predictor by comparing with known n and w.
  PIK_INLINE V PredictorCosts(const size_t x,
                              const DC* const PIK_RESTRICT row_ym,
                              const DC* const PIK_RESTRICT row_yb,
                              const DC* const PIK_RESTRICT row_t) {
    const V tr(Broadcast(LoadT(row_t, x + 1)));
    const V costs =
        AbsResidual(n_, Predict(tn_, l_, tl_, tr)) + AbsResidual(w_, pred_w_);
    tl_ = tn_;
    tn_ = tr;
    return costs;
  }

  // Returns predictor for pixel c with min cost and updates pred_w_.
  PIK_INLINE T PredictC(const T& r, const V& costs) {
    const V8x16U idx_min(_mm_minpos_epu16(Costs16(costs)));
    const V8x32U index = V8x32U(_mm256_broadcastd_epi32(idx_min)) >> 16;

    const V pred_c = Predict(n_, w_, l_, Broadcast(r));
    pred_w_ = pred_c;

    const V best(_mm256_permutevar8x32_epi32(pred_c, index));
    return T(_mm256_extracti128_si256(best, 0));
  }

  PIK_INLINE void Advance(const T& r, const T& c) {
    l_ = n_;
    n_ = Broadcast(r);
    w_ = Broadcast(c);
  }

 private:
  // Eight predictors for luminance (decreases coded size by ~0.5% vs four)
  // 0: Average(w, n);
  // 1: Average(Average(w, r), n);
  // 2: Average(n, r);
  // 3: Average(w, l);
  // 4: Average(l, n);
  // 5: w;
  // 6: PredClampedGrad(n, w, l);
  // 7: n;
  // All arguments are broadcasted.
  static PIK_INLINE V Predict(const V& n, const V& w, const V& l, const V& r) {
    const V rnrnrnrn(_mm256_unpacklo_epi32(n, r));
    // "x" are invalid/don't care lanes.
    const V xxxnwrwn(_mm256_unpacklo_epi32(rnrnrnrn, w));
    const V p76xxxxxx(_mm256_unpacklo_epi32(ClampedGradient(n, w, l), n));
    const V xxxnwrww(_mm256_blend_epi32(xxxnwrwn, w, 0x01));
    const V p765xxxxx(_mm256_blend_epi32(p76xxxxxx, w, 0x20));
    const V xxxllnrn(_mm256_blend_epi32(rnrnrnrn, l, 0x18));
    // The first five predictors are averages; "a" needs another Average.
    const V pxxx432a0 = Average(xxxllnrn, xxxnwrww);
    const V pxxxxxx1x = Average(pxxx432a0, n);  // = A(A(w, r), n)
    const V p765432x0(_mm256_blend_epi32(pxxx432a0, p765xxxxx, 0xE0));
    return V(_mm256_blend_epi32(p765432x0, pxxxxxx1x, 0x02));
  }

  V tl_;
  V tn_;
  V n_;
  V w_;
  V l_;
  // (30% overall speedup by reusing the current prediction as the next pred_w_)
  V pred_w_;
};

// Providing separate sets of predictors for the luminance and chrominance bands
// reduces the magnitude of residuals, but differentiating between the
// chrominance bands does not.
class PixelNeighborsUV {
  using V = V8x32I;

 public:
  // LoadT/StoreT/compute pairs of U, V.
  using T = V4x32I;

  // Returns 00UV.
  static PIK_INLINE T LoadT(const DC* const PIK_RESTRICT row, const size_t x) {
    return T(_mm_loadl_epi64(
        reinterpret_cast<const __m128i * PIK_RESTRICT>(row + 2 * x)));
  }

  static PIK_INLINE void StoreT(const T& uv, DC* const PIK_RESTRICT row,
                                const size_t x) {
    _mm_storel_epi64(reinterpret_cast<__m128i * PIK_RESTRICT>(row + 2 * x), uv);
  }

  PixelNeighborsUV(const DC* const PIK_RESTRICT row_ym,
                   const DC* const PIK_RESTRICT row_yb,
                   const DC* const PIK_RESTRICT row_t,
                   const DC* const PIK_RESTRICT row_m,
                   const DC* const PIK_RESTRICT row_b) {
    yn_ = V(row_ym[2]);
    yw_ = V(row_yb[1]);
    yl_ = V(row_ym[1]);
    n_ = LoadT(row_m, 2);
    w_ = LoadT(row_b, 1);
    l_ = LoadT(row_m, 1);
  }

  // Estimates "cost" for each predictor by comparing with known c from Y band.
  PIK_INLINE V PredictorCosts(const size_t x,
                              const DC* const PIK_RESTRICT row_ym,
                              const DC* const PIK_RESTRICT row_yb,
                              const DC* const PIK_RESTRICT) {
    const V yr(row_ym[x + 1]);
    const V yc(row_yb[x]);
    const V costs = AbsResidual(yc, Predict(yn_, yw_, yl_, yr));
    yl_ = yn_;
    yn_ = yr;
    yw_ = yc;
    return costs;
  }

  // Returns predictor for pixel c with min cost.
  PIK_INLINE T PredictC(const T& r, const V& costs) const {
    const V8x16U idx_min(_mm_minpos_epu16(Costs16(costs)));
    const V8x32U index = V8x32U(_mm256_broadcastd_epi32(idx_min)) >> 16;

    const V predictors_u =
        Predict(BroadcastU(n_), BroadcastU(w_), BroadcastU(l_), BroadcastU(r));
    const V predictors_v =
        Predict(BroadcastV(n_), BroadcastV(w_), BroadcastV(l_), BroadcastV(r));
    // permutevar is faster than Store + load_ss.
    const V best_u(_mm256_permutevar8x32_epi32(predictors_u, index));
    const V best_v(_mm256_permutevar8x32_epi32(predictors_v, index));
    const T best_u128(_mm256_extracti128_si256(best_u, 0));
    const T best_v128(_mm256_extracti128_si256(best_v, 0));
    return T(_mm_unpacklo_epi32(best_v128, best_u128));
  }

  PIK_INLINE void Advance(const T& r, const T& c) {
    l_ = n_;
    n_ = r;
    w_ = c;
  }

 private:
  static PIK_INLINE V BroadcastU(const T& uv) {
    const T u(_mm_srli_si128(uv, sizeof(DC)));
    return V(_mm256_broadcastd_epi32(u));
  }

  static PIK_INLINE V BroadcastV(const T& uv) {
    return V(_mm256_broadcastd_epi32(uv));
  }

  // Eight predictors for chrominance:
  // 0: ClampedGrad(n, w, l);
  // 1: n;
  // 2: Average2(n, w);
  // 3: Average2(Average2(w, r), n);
  // 4: w;
  // 5: Average2(n, r);
  // 6: Average2(w, l);
  // 7: r;
  // All arguments are broadcasted.
  static PIK_INLINE V Predict(const V& n, const V& w, const V& l, const V& r) {
    const V xxxxxxx0 = ClampedGradient(n, w, l);
    // "x" lanes are unused.
    const V xxxxxx10(_mm256_unpacklo_epi32(xxxxxxx0, n));
    const V rwrwrwrw(_mm256_unpacklo_epi32(w, r));
    const V xlrxrwxx(_mm256_blend_epi32(rwrwrwrw, l, 0x40));
    const V xwnxwnxx(_mm256_blend_epi32(w, n, 0x24));
    // "a" requires further averaging.
    const V x65xa2xx = Average(xlrxrwxx, xwnxwnxx);
    const V x65xa210(_mm256_blend_epi32(x65xa2xx, xxxxxx10, 0x03));
    const V xxxx3xxx = Average(x65xa210, n);
    const V x65x3210(_mm256_blend_epi32(x65xa210, xxxx3xxx, 0x08));
    return V(_mm256_blend_epi32(x65x3210, rwrwrwrw, 0x90));
  }

  V yn_;
  V yw_;
  V yl_;
  T n_;
  T w_;
  T l_;
};

// Computes residuals of a fixed predictor (the preceding pixel W).
// Useful for Row(0) because no preceding row is required.
template <class N>
struct FixedW {
  static PIK_INLINE void Shrink(const size_t xsize,
                                const DC* const PIK_RESTRICT dc,
                                DC* const PIK_RESTRICT residuals) {
    N::StoreT(N::LoadT(dc, 0), residuals, 0);
    for (size_t x = 1; x < xsize; ++x) {
      N::StoreT(N::LoadT(dc, x) - N::LoadT(dc, x - 1), residuals, x);
    }
  }

  static PIK_INLINE void Expand(const size_t xsize,
                                const DC* const PIK_RESTRICT residuals,
                                DC* const PIK_RESTRICT dc) {
    N::StoreT(N::LoadT(residuals, 0), dc, 0);
    for (size_t x = 1; x < xsize; ++x) {
      N::StoreT(N::LoadT(dc, x - 1) + N::LoadT(residuals, x), dc, x);
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
    N::StoreT(N::LoadT(row_b, 0) - N::LoadT(row_m, 0), residuals, 0);
    if (xsize >= 2) {
      // TODO(user): Clamped gradient should be slightly better here.
      N::StoreT(N::LoadT(row_b, 1) - N::LoadT(row_b, 0), residuals, 1);
    }
  }

  static PIK_INLINE void Expand(const size_t xsize,
                                const DC* const PIK_RESTRICT residuals,
                                const DC* const PIK_RESTRICT row_m,
                                DC* const PIK_RESTRICT row_b) {
    N::StoreT(N::LoadT(row_m, 0) + N::LoadT(residuals, 0), row_b, 0);
    if (xsize >= 2) {
      N::StoreT(N::LoadT(row_b, 0) + N::LoadT(residuals, 1), row_b, 1);
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
      const auto res = N::LoadT(dc, xsize - 1) - N::LoadT(dc, xsize - 2);
      N::StoreT(res, residuals, xsize - 1);
    }
  }

  static PIK_INLINE void Expand(const size_t xsize,
                                const DC* const PIK_RESTRICT residuals,
                                DC* const PIK_RESTRICT dc) {
    if (xsize >= 2) {
      const auto uv = N::LoadT(dc, xsize - 2) + N::LoadT(residuals, xsize - 1);
      N::StoreT(uv, dc, xsize - 1);
    }
  }
};

// Selects predictor based upon its error at the prior n and w pixels.
// Requires two preceding rows (t, m) and the current row b. The row_y*
// pointers are unused and may be null if N = PixelNeighborsY.
template <class N>
class Adaptive {
  using T = typename N::T;

 public:
  static void Shrink(const size_t xsize, const DC* const PIK_RESTRICT row_ym,
                     const DC* const PIK_RESTRICT row_yb,
                     const DC* const PIK_RESTRICT row_t,
                     const DC* const PIK_RESTRICT row_m,
                     const DC* const PIK_RESTRICT row_b,
                     DC* const PIK_RESTRICT residuals) {
    LeftBorder2<N>::Shrink(xsize, row_m, row_b, residuals);

    ForeachPrediction(xsize, row_ym, row_yb, row_t, row_m, row_b,
                      [row_b, residuals](const size_t x, const T& pred) {
                        const T c = N::LoadT(row_b, x);
                        N::StoreT(c - pred, residuals, x);
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
                      [row_b, residuals](const size_t x, const T& pred) {
                        const T c = pred + N::LoadT(residuals, x);
                        N::StoreT(c, row_b, x);
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
      const T r = N::LoadT(row_m, x + 1);
      const V8x32I costs = neighbors.PredictorCosts(x, row_ym, row_yb, row_t);
      const T pred_c = neighbors.PredictC(r, costs);
      const T c = func(x, pred_c);
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
}  // namespace PIK_TARGET_NAME

// WARNING: the current implementation requires AVX2 and doesn't yet support
// dynamic dispatching.

void ShrinkY(const Image<DC>& dc, Image<DC>* const PIK_RESTRICT residuals) {
  PIK_TARGET_NAME::ShrinkY(dc, residuals);
}

void ShrinkUV(const Image<DC>& dc_y, const Image<DC>& dc,
              Image<DC>* const PIK_RESTRICT residuals) {
  PIK_TARGET_NAME::ShrinkUV(dc_y, dc, residuals);
}

void ExpandY(const Image<DC>& residuals, Image<DC>* const PIK_RESTRICT dc) {
  PIK_TARGET_NAME::ExpandY(residuals, dc);
}

void ExpandUV(const Image<DC>& dc_y, const Image<DC>& residuals,
              Image<DC>* const PIK_RESTRICT dc) {
  PIK_TARGET_NAME::ExpandUV(dc_y, residuals, dc);
}

}  // namespace pik

