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

#ifndef DC_PREDICTOR_SLOW_H_
#define DC_PREDICTOR_SLOW_H_

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <algorithm>

#include "compiler_specific.h"
#include "image.h"
#include "status.h"

namespace pik {

template <typename V>
class Predictors {
 public:
  static const size_t kNum = 8;
  struct Y {
    Y() {}

    PIK_INLINE void operator()(const V* const PIK_RESTRICT pos,
                               const intptr_t neg_col_stride,
                               const intptr_t neg_row_stride,
                               V pred[kNum]) const {
      const V w = pos[neg_col_stride];
      const V n = pos[neg_row_stride];
      const V l = pos[neg_row_stride + neg_col_stride];
      const V r = pos[neg_row_stride - neg_col_stride];
      pred[0] = Average(Average(n, w), r);
      pred[1] = Average(w, n);
      pred[2] = Average(n, r);
      pred[3] = Average(w, l);
      pred[4] = Average(l, n);
      pred[5] = w;
      pred[6] = ClampedGradient(n, w, l);
      pred[7] = n;
    }
  };

  struct UV {
    UV() {}

    PIK_INLINE void operator()(const V* const PIK_RESTRICT pos,
                               const intptr_t neg_col_stride,
                               const intptr_t neg_row_stride,
                               V pred[kNum]) const {
      const V w = pos[neg_col_stride];
      const V n = pos[neg_row_stride];
      const V l = pos[neg_row_stride + neg_col_stride];
      const V r = pos[neg_row_stride - neg_col_stride];
      pred[0] = ClampedGradient(n, w, l);
      pred[1] = Average(n, w);
      pred[2] = n;
      pred[3] = Average(n, r);
      pred[4] = w;
      pred[5] = Average(w, l);
      pred[6] = r;
      pred[7] = Average(Average(w, r), n);
    }
  };

 private:
  // Clamps gradient to the min/max of n, w, l.
  static PIK_INLINE V ClampedGradient(const V& n, const V& w, const V& l) {
    const V grad = n + w - l;
    const V min = std::min(n, std::min(w, l));
    const V max = std::max(n, std::max(w, l));
    return std::min(std::max(min, grad), max);
  }

  static PIK_INLINE V Average(const V& v0, const V& v1) {
    return (v0 + v1) >> 1;
  }
};

template <class Predictor, typename V>
V MinCostPrediction(const V* const PIK_RESTRICT dc, size_t x, size_t y,
                    size_t xsize, intptr_t neg_col_stride,
                    intptr_t neg_row_stride) {
  if (y == 0) {
    return x ? dc[neg_col_stride] : 0;
  } else if (x == 0) {
    return dc[neg_row_stride];
  } else if (x == 1 || x + 1 == xsize) {
    return dc[neg_col_stride];
  } else {
    const Predictor predictor;
    V pred[Predictors<V>::kNum];
    predictor(dc, neg_col_stride, neg_row_stride, pred);
    V pred_w[Predictors<V>::kNum];
    predictor(&dc[neg_col_stride], neg_col_stride, neg_row_stride, pred_w);
    const V w = dc[neg_col_stride];
    const V n = dc[neg_row_stride];
    V costs[Predictors<V>::kNum];
    for (int i = 0; i < Predictors<V>::kNum; ++i) {
      costs[i] = std::abs(w - pred_w[i]);
    }
    V pred_n[Predictors<V>::kNum];
    if (y > 1) {
      predictor(&dc[neg_row_stride], neg_col_stride, neg_row_stride, pred_n);
    } else {
      predictor(&dc[neg_row_stride], neg_col_stride, 0, pred_n);
    }
    for (int i = 0; i < Predictors<V>::kNum; ++i) {
      costs[i] += std::abs(n - pred_n[i]);
    }
    const int idx =
        std::min_element(costs, costs + Predictors<V>::kNum) - costs;
    return pred[idx];
  }
}

template <class Predictor, typename V>
V MinCostYPrediction(const V* const PIK_RESTRICT dc, size_t x, size_t y,
                     size_t xsize, intptr_t neg_col_stride,
                     intptr_t neg_row_stride, int uv_predictor) {
  if (y == 0) {
    return x ? dc[neg_col_stride] : 0;
  } else if (x == 0) {
    return dc[neg_row_stride];
  } else if (x == 1 || x + 1 == xsize) {
    return dc[neg_col_stride];
  } else {
    const Predictor predictor;
    V pred[Predictors<V>::kNum];
    predictor(dc, neg_col_stride, neg_row_stride, pred);
    return pred[uv_predictor];
  }
}

template <typename V>
int GetUVPredictor(const V* const PIK_RESTRICT dc_y, size_t x, size_t y,
                   size_t xsize, intptr_t neg_col_stride,
                   intptr_t neg_row_stride) {
  if (y == 0 || x <= 1 || x + 1 == xsize) {
    return 0;
  }
  const typename Predictors<V>::UV predictor;
  V pred_y[Predictors<V>::kNum];
  V costs[Predictors<V>::kNum];
  predictor(dc_y, neg_col_stride, neg_row_stride, pred_y);
  for (int i = 0; i < Predictors<V>::kNum; ++i) {
    costs[i] = std::abs(dc_y[0] - pred_y[i]);
  }
  return std::min_element(costs, costs + Predictors<V>::kNum) - costs;
}

template <typename V>
V MinCostPredict(const V* const PIK_RESTRICT dc, size_t x, size_t y,
                 size_t xsize, intptr_t neg_col_stride, intptr_t neg_row_stride,
                 bool use_uv_predictor, int uv_predictor) {
  return use_uv_predictor
             ? MinCostYPrediction<typename Predictors<V>::UV>(
                   dc, x, y, xsize, neg_col_stride, neg_row_stride,
                   uv_predictor)
             : MinCostPrediction<typename Predictors<V>::Y>(
                   dc, x, y, xsize, neg_col_stride, neg_row_stride);
}

static inline int Quantize(const int dc, const int quant_step) {
  // TODO(janwas): division via table of magic multipliers+fixups
  const int trunc = dc / quant_step;
  const int mod = dc - (trunc * quant_step);
  // If closer to the end of the interval, round up (+1).
  const int rounded = trunc + (mod > quant_step / 2);
  return rounded;
}

// Drop-in replacements for dc_predictor functions
template <typename DC>
void SlowShrinkY(const Image<DC>& dc, Image<DC>* const PIK_RESTRICT residuals) {
  const size_t xsize = dc.xsize();
  const size_t ysize = dc.ysize();
  for (size_t y = 0; y < ysize; y++) {
    const DC* const PIK_RESTRICT row = dc.Row(y);
    DC* const PIK_RESTRICT row_out = residuals->Row(y);
    for (size_t x = 0; x < xsize; x++) {
      DC pred = MinCostPredict(&row[x], x, y, xsize, -1,
                               -dc.bytes_per_row() / sizeof(DC), false, 0);
      row_out[x] = row[x] - pred;
    }
  }
}

template <typename DC>
void SlowShrinkUV(const Image<DC>& dc_y, const Image<DC>& dc,
                  Image<DC>* const PIK_RESTRICT residuals) {
  // WARNING: dc and residuals have twice the xsize.
  const size_t xsize = dc_y.xsize();
  const size_t ysize = dc_y.ysize();
  for (size_t y = 0; y < ysize; y++) {
    const DC* const PIK_RESTRICT row = dc.Row(y);
    const DC* const PIK_RESTRICT row_y = dc_y.Row(y);
    DC* const PIK_RESTRICT row_out = residuals->Row(y);
    for (size_t x = 0; x < xsize; x++) {
      int predictor = GetUVPredictor(&row_y[x], x, y, xsize, -1,
                                     -dc_y.bytes_per_row() / sizeof(DC));
      for (int i = 0; i < 2; i++) {
        DC pred =
            MinCostPredict(&row[2 * x + i], x, y, xsize, -2,
                           -dc.bytes_per_row() / sizeof(DC), true, predictor);
        row_out[2 * x + i] = row[2 * x + i] - pred;
      }
    }
  }
}

template <typename DC>
void SlowQuantizeY(const Image<DC>& dc, const ImageS& quant_map,
                   Image<DC>* const PIK_RESTRICT residuals) {
  PIK_CHECK(SameSize(*residuals, quant_map));
  PIK_CHECK(SameSize(*residuals, dc));
  const size_t xsize = dc.xsize();
  const size_t ysize = dc.ysize();
  Image<DC> qdc(xsize, ysize);
  for (size_t y = 0; y < ysize; y++) {
    const DC* const PIK_RESTRICT row = dc.Row(y);
    const int16_t* const PIK_RESTRICT row_quant = quant_map.Row(y);
    DC* const PIK_RESTRICT row_out = residuals->Row(y);
    DC* const PIK_RESTRICT row_qdc = qdc.Row(y);
    for (size_t x = 0; x < xsize; x++) {
      const DC pred =
          MinCostPredict(&row_qdc[x], x, y, xsize, -1,
                         -qdc.bytes_per_row() / sizeof(DC), false, 0);
      const int res = row[x] - pred;
      row_out[x] = Quantize(res, row_quant[x]);
      row_qdc[x] = pred + row_out[x] * row_quant[x];
      // printf("%2zu %2zu: %d (pred %d)=>%d, q %d => %d\n", x, y, row[x], pred,
      //        res, row_quant[x], row_out[x]);
    }
  }
}

template <typename DC>
void SlowDequantizeY(const Image<DC>& residuals, const ImageS& quant_map,
                     Image<DC>* const PIK_RESTRICT dc) {
  PIK_CHECK(SameSize(residuals, quant_map));
  PIK_CHECK(SameSize(residuals, *dc));
  const size_t xsize = dc->xsize();
  const size_t ysize = dc->ysize();
  for (size_t y = 0; y < ysize; y++) {
    DC* const PIK_RESTRICT row = dc->Row(y);
    const DC* const PIK_RESTRICT row_res = residuals.Row(y);
    const DC* const PIK_RESTRICT row_quant = quant_map.Row(y);
    for (size_t x = 0; x < xsize; x++) {
      const DC pred =
          MinCostPredict(&row[x], x, y, xsize, -1,
                         -dc->bytes_per_row() / sizeof(DC), false, 0);
      const int res = row_res[x] * row_quant[x];
      row[x] = pred + res;
      // printf("%2zu %2zu: %d (pred %d), q %d res %d\n", x, y, row[x], pred,
      //        row_quant[x], res);
    }
  }
}

}  // namespace pik

#endif  // DC_PREDICTOR_SLOW_H_
