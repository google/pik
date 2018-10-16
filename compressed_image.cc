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

#include "compressed_image.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <string>

#define PROFILER_ENABLED 1
#include "brotli.h"
#include "butteraugli_distance.h"
#include "common.h"
#include "compiler_specific.h"
#include "convolve.h"
#include "dc_predictor.h"
#include "dct.h"
#include "dct_util.h"
#include "deconvolve.h"
#include "entropy_coder.h"
#include "fields.h"
#include "gauss_blur.h"
#include "huffman_decode.h"
#include "huffman_encode.h"
#include "opsin_image.h"
#include "opsin_inverse.h"
#include "opsin_params.h"
#include "profiler.h"
#include "resample.h"
#include "simd/simd.h"
#include "status.h"
#include "upscaler.h"

namespace pik {

// Tile is the rectangular grid of blocks that could be resampled before the
// integral transform. Coincidentally it has the same size as color tile.
constexpr size_t kTileDim = 64;

// Factor by which DCT16 blocks should be scaled before quantization.
// TODO(user): this is a hack needed until adaptive quantization is fixed
// to work with different block types.
constexpr float kDct16Scale = 8;
constexpr float kDct16ScaleInv = 1.0 / kDct16Scale;

static_assert(kTileDim % kBlockDim == 0,
              "Tile dim should be divisible by block dim");
constexpr size_t kTileDimInBlocks = kTileDim / kBlockDim;

static_assert(kGroupWidthInBlocks % kTileDimInBlocks == 0,
              "Group dim should be divisible by tile dim");

constexpr float kIdentityAvgParam = 0.25;

// Predict image with gradients in 64x64 blocks (8x8 DC components)
// Gradient map improves the banding of highly quantized sky. Most effective
// for lower qualities.
struct GradientMap {
  GradientMap(size_t xsize_dc, size_t ysize_dc, bool grayscale)
      : xsize_dc_(xsize_dc), ysize_dc_(ysize_dc), grayscale_(grayscale) {
    // numx and numy are amount of blocks in x and y direction, and the
    // amount is such that when there are N * kNumBlocks_ + 1 pixels, there
    // are only N blocks (the one extra pixel can still be part of the last
    // block), once there are N * kNumBlocks_ + 2 pixels, there are N + 1
    // blocks. Note that kNumBlocks_ is in fact the size of 1 block, num blocks
    // refers to amount of DC values (from DCT blocks) this block contains.
    size_t numx = DivCeil(xsize_dc_ - 1, kNumBlocks_);
    size_t numy = DivCeil(ysize_dc_ - 1, kNumBlocks_);

    // Size of the gradient map: one bigger than numx and numy because the
    // blocks have values on all corners ("fenceposts").
    xsize_ = numx + 1;
    ysize_ = numy + 1;

    // Note that gradient_ is much smaller than the DC image, and the DC image
    // in turn already is much smaller than the full original image.
    gradient_ = Image3F(xsize_, ysize_);
  }

  static double Interpolate(double v00, double v01, double v10, double v11,
                            double x, double y) {
    return v00 * (1 - x) * (1 - y) + v10 * x * (1 - y) + v01 * (1 - x) * y +
           v11 * x * y;
  }

  // Finds values for block corner points r that best fit the points p with
  // linear interpolation.
  // p must have xsize * ysize points, bs is the block size, r must have
  // ((xsize + bs - 2) / bs + 1) * ((ysize + bs - 2) / bs + 1) values.
  // Set point values to the value of exclude to not take them into account,
  // and enable guess_initial to let this function guess initial values, false
  // to use user-chosen initial values of r.
  void PlanePieceFit(const float* p, size_t xsize, size_t ysize, int bs,
                     float exclude, bool guess_initial, float* r) const {
    // Total amount of points on the 2D grid.
    size_t size = xsize * ysize;

    // Amount of basis functions.
    size_t numx = ((xsize + bs - 2) / bs + 1);
    size_t numy = ((ysize + bs - 2) / bs + 1);
    size_t m = numx * numy;

    // Point index, used to skip excluded points.
    std::vector<int> p_index(size);
    std::vector<float> included;
    size_t n = 0;
    for (int j = 0; j < size; j++) {
      if (p[j] != exclude) {
        p_index[j] = n;
        included.push_back(p[j]);
        n++;
      } else {
        p_index[j] = -1;
      }
    }

    // matrix F: One basis function per row, one included point per column.
    std::vector<float> f(n * m, 0);
    for (int i = 0; i < m; i++) {
      int bx = i % numx;
      int by = i / numx;
      // i-th basisfunction is bilinear interpolation pyramid with small
      // support around the current peak point.
      int x0 = std::max<int>(0, (bx - 1) * bs);
      int xm = std::min<int>(xsize - 1, bx * bs);
      int x1 = std::min<int>(xsize - 1, (bx + 1) * bs);
      int y0 = std::max<int>(0, (by - 1) * bs);
      int ym = std::min<int>(ysize - 1, by * bs);
      int y1 = std::min<int>(ysize - 1, (by + 1) * bs);
      for (int y = y0; y <= y1; y++) {
        for (int x = x0; x <= x1; x++) {
          int j = y * xsize + x;
          if (p[j] == exclude) continue;
          float vx = 0;
          if (x < xm) {
            vx = (x - x0) * 1.0 / (xm - x0);
          } else if (x == xm) {
            vx = 1.0;
          } else {
            vx = 1.0 - (x - xm) * 1.0 / (x1 - xm);
          }
          float vy = 0;
          if (y < ym) {
            vy = (y - y0) * 1.0 / (ym - y0);
          } else if (y == ym) {
            vy = 1.0;
          } else {
            vy = 1.0 - (y - ym) * 1.0 / (y1 - ym);
          }

          f[i * n + p_index[j]] = vx * vy;
        }
      }
    }

    if (guess_initial) {
      // Simple heuristic: guess points that match block corners, but skip
      // excluded points.
      for (size_t y = 0; y < numy; y++) {
        float prev = 0;
        for (size_t x = 0; x < numx; x++) {
          int px = x * bs;
          int py = y * bs;
          int j = py * xsize + px;
          int i = y * numx + x;
          r[i] = (px < xsize && py < ysize && p[j] != exclude) ? p[j] : prev;
          prev = r[i];
        }
      }
    }

    FEM(f.data(), m, n, included.data(), r);
  }

  // Computes the gradient map for the given image of DC
  // values.
  // The opsin image must be in centered opsin color space
  void ComputeFromSource(const Image3F& opsin, const Quantizer& quantizer,
                         ThreadPool* pool) {
    static const float kMinDiff[3] = {0.3, 0.8, 0.2};

    const auto compute_gradient_channel = [this, &opsin, &quantizer](
                                              const int task,
                                              const int thread) {
      const int c = task;
      if (grayscale_ && c != 1) return;
      static const float kExclude = 999999;
      std::vector<float> points(ysize_dc_ * xsize_dc_, kExclude);

      // A block can be one pixel bigger on edges.
      size_t max_size = (kNumBlocks_ + 1) * (kNumBlocks_ + 1);
      std::vector<unsigned char> done(max_size);
      std::vector<std::pair<size_t, size_t>> best(max_size);
      std::vector<std::pair<size_t, size_t>> current(max_size);
      std::vector<std::pair<size_t, size_t>> stack(max_size);

      // The DC quantization step for this channel.
      const float scale = quantizer.inv_quant_dc() *
                          quantizer.DequantMatrix(c, kQuantKindDCT8)[0];

      for (size_t by = 0; by + 1 < ysize_; by++) {
        for (size_t bx = 0; bx + 1 < xsize_; bx++) {
          size_t x0 = bx * kNumBlocks_;
          size_t x1 = std::min<size_t>(xsize_dc_, x0 + kNumBlocks_);
          size_t y0 = by * kNumBlocks_;
          size_t y1 = std::min<size_t>(ysize_dc_, y0 + kNumBlocks_);
          // Block is one larger than normal if on a right or bottom edge
          // with particular size.
          if (bx + 2 == xsize_ && xsize_dc_ % kNumBlocks_ == 1) x1++;
          if (by + 2 == ysize_ && ysize_dc_ % kNumBlocks_ == 1) y1++;
          size_t dx = x1 - x0;
          size_t dy = y1 - y0;

          // Search for largest smooth cluster in block using floodfills.
          const float maxdiff = scale * kMinDiff[c];
          const float mingroupratio = 0.45;
          const int mingroupsize = static_cast<int>(dx * dy * mingroupratio);

          std::fill(done.begin(), done.end(), 0);
          best.clear();

          for (size_t sy = 0; sy < dy; sy++) {
            for (size_t sx = 0; sx < dx; sx++) {
              if (done[sy * dx + sx]) continue;
              done[sy * dx + sx] = 1;
              size_t current_size = 0;
              size_t stack_size = 0;
              stack[stack_size++] = std::make_pair(sx, sy);
              while (stack_size != 0) {
                auto p = stack[--stack_size];
                current[current_size++] = p;
                int tx = p.first;
                int ty = p.second;
                float v0 = opsin.PlaneRow(c, y0 + ty)[x0 + tx];
                // 4 neighbors: N, E, S, W
                for (int d = 0; d < 4; d++) {
                  int nx = tx + ((d == 1) ? 1 : ((d == 3) ? -1 : 0));
                  int ny = ty + ((d == 2) ? 1 : ((d == 0) ? -1 : 0));
                  if (nx >= dx || nx < 0 || ny >= dy || ny < 0) continue;
                  if (done[ny * dx + nx]) continue;
                  float v1 = opsin.PlaneRow(c, y0 + ny)[x0 + nx];
                  if (fabs(v1 - v0) > maxdiff) continue;
                  done[ny * dx + nx] = 1;
                  stack[stack_size++] = std::make_pair(nx, ny);
                }
              }
              if (current_size > best.size()) {
                best.assign(current.begin(), current.begin() + current_size);
              }
            }
          }
          if (best.size() >= mingroupsize) {
            for (size_t i = 0; i < best.size(); i++) {
              size_t px = x0 + best[i].first;
              size_t py = y0 + best[i].second;
              points[py * xsize_dc_ + px] = opsin.PlaneRow(c, py)[px];
            }
          }
        }
      }
      std::vector<float> coeffs(xsize_ * ysize_);
      PlanePieceFit(points.data(), xsize_dc_, ysize_dc_, kNumBlocks_, kExclude,
                    true, coeffs.data());
      *gradient_.MutablePlane(c) = ImageFromPacked(coeffs, xsize_, ysize_);
    };

    pool->Run(0, 3, compute_gradient_channel);
    AccountForQuantization(quantizer);
  }

  // Applies the stored gradient map in the decoder.
  void Apply(const Quantizer& quantizer, Image3F* opsin, ThreadPool* pool) {
    Image3F upscaled = ComputeGradientImage(pool);
    static const float kScale[3] = {1.0, 1.0, 2.0};

    const auto apply_gradient_channel = [this, &upscaled, &quantizer, opsin](
                                            const int task, const int thread) {
      const int c = task;
      if (grayscale_ && c != 1) return;
      const float step = quantizer.inv_quant_dc() *
                         quantizer.DequantMatrix(c, kQuantKindDCT8)[0] *
                         kScale[c];

      std::vector<char> apply(ysize_dc_ * xsize_dc_, 0);

      for (size_t y = 0; y < ysize_dc_; y++) {
        float* PIK_RESTRICT row_out = opsin->PlaneRow(c, y);
        const float* PIK_RESTRICT row_in = upscaled.ConstPlaneRow(c, y);
        for (size_t x = 0; x < xsize_dc_; x++) {
          float diff = fabs(row_out[x] - row_in[x]);
          if (diff < step) {
            apply[y * xsize_dc_ + x] = 1;
          }
        }
      }

      // Reduce the size of the field where to apply the gradient, to avoid
      // doing it in noisy areas
      for (int i = 0; i < 3; i++) {
        std::vector<char> old = apply;
        for (size_t y = 0; y < ysize_dc_; y++) {
          for (size_t x = 0; x < xsize_dc_; x++) {
            int num = 0;
            num += y > 0 ? (old[(y - 1) * xsize_dc_ + x] == 0) : 0;
            num +=
                (y + 1) < ysize_dc_ ? (old[(y + 1) * xsize_dc_ + x] == 0) : 0;
            num += x > 0 ? (old[y * xsize_dc_ + x - 1] == 0) : 0;
            num += (x + 1) < xsize_dc_ ? (old[y * xsize_dc_ + x + 1] == 0) : 0;
            if (num != 0) apply[y * xsize_dc_ + x] = 0;
          }
        }
      }

      for (size_t y = 0; y < ysize_dc_; y++) {
        float* PIK_RESTRICT row_out = opsin->PlaneRow(c, y);
        const float* PIK_RESTRICT row_in = upscaled.ConstPlaneRow(c, y);
        for (size_t x = 0; x < xsize_dc_; x++) {
          if (apply[y * xsize_dc_ + x]) {
            row_out[x] = row_in[x];
          }
        }
      }
    };
    pool->Run(0, 3, apply_gradient_channel);
  }

  static PaddedBytes CompressBytes(const PaddedBytes& v) {
    PaddedBytes result;
    PIK_CHECK(BrotliCompress(11, v, &result));
    return result;
  }

  static pik::Status DecompressBytes(const uint8_t* in, size_t max_in_size,
                                     size_t expected_result_size, size_t* pos,
                                     PaddedBytes* out) {
    size_t bytes_read = 0;
    size_t max_decoded_size = expected_result_size;
    if (*pos >= max_in_size) {
      return PIK_FAILURE("Gradient: invalid position");
    }
    if (!BrotliDecompress(in + *pos, max_in_size - *pos, max_decoded_size,
                          &bytes_read, out)) {
      return PIK_FAILURE("Gradient: invalid brotli stream");
    }
    *pos += bytes_read;
    return true;
  }

  // Sets gradient map from external source.
  bool SetGradientMap(const Image3F& gradient) {
    if (gradient.xsize() != gradient_.xsize() ||
        gradient.ysize() != gradient_.ysize()) {
      return PIK_FAILURE("Gradient: invalid size");
    }
    gradient_ = CopyImage(gradient);
    return true;
  }

  const Image3F& GetGradientMap() const { return gradient_; }

  PaddedBytes Quantize(const Quantizer& quantizer, const Image3F& image) const {
    PaddedBytes bytes(xsize_ * ysize_ * (grayscale_ ? 1 : 3));
    size_t pos = 0;
    for (int c = 0; c < 3; c++) {
      if (grayscale_ && c != 1) continue;
      const float step = quantizer.inv_quant_dc() *
                         quantizer.DequantMatrix(c, kQuantKindDCT8)[0];

      float range = kXybRange[c] * 2;
      int steps = std::min(std::max(16, (int)(3 * range / step)), 256);
      int zerolevel = steps / 2;
      float mul = steps / range;

      for (size_t y = 0; y < image.ysize(); y++) {
        const auto* row = image.PlaneRow(c, y);
        for (size_t x = 0; x < image.xsize(); x++) {
          int value = std::round(row[x] * mul) + zerolevel;
          value = std::min(std::max(0, value), steps - 1);
          bytes[pos++] = value;
        }
      }
    }
    return bytes;
  }

  Image3F Dequantize(const Quantizer& quantizer,
                     const PaddedBytes& bytes) const {
    Image3F result(xsize_, ysize_);
    size_t pos = 0;
    for (int c = 0; c < 3; c++) {
      if (grayscale_ && c != 1) continue;
      const float step = quantizer.inv_quant_dc() *
                         quantizer.DequantMatrix(c, kQuantKindDCT8)[0];

      float range = kXybRange[c] * 2;
      int steps = std::min(std::max(16, (int)(3 * range / step)), 256);
      int zerolevel = steps / 2;

      float mul = steps / range;
      float inv = 1.0f / mul;
      for (size_t y = 0; y < result.ysize(); y++) {
        auto* row_out = result.PlaneRow(c, y);
        for (size_t x = 0; x < result.xsize(); x++) {
          int byte = bytes[pos++];
          float v = (byte - zerolevel) * inv;
          row_out[x] = v;
        }
      }
    }
    return result;
  }

  void Serialize(const Quantizer& quantizer, PaddedBytes* compressed) const {
    PaddedBytes encoded = Quantize(quantizer, gradient_);
    encoded = CompressBytes(encoded);
    size_t pos = compressed->size();
    compressed->resize(compressed->size() + encoded.size());
    for (size_t i = 0; i < encoded.size(); i++) {
      compressed->data()[pos++] = encoded[i];
    }
  }

  pik::Status Deserialize(const Quantizer& quantizer,
                          const PaddedBytes& compressed, size_t* byte_pos) {
    size_t encoded_size = xsize_ * ysize_ * (grayscale_ ? 1 : 3);
    PaddedBytes encoded;
    PIK_RETURN_IF_ERROR(DecompressBytes(compressed.data(), compressed.size(),
                                        encoded_size, byte_pos, &encoded));
    if (encoded.size() != encoded_size) {
      return PIK_FAILURE("Gradient: invalid size");
    }

    gradient_ = Dequantize(quantizer, encoded);

    return true;  // success
  }

 private:
  // Computes the smooth gradient image from the computed corner points.
  Image3F ComputeGradientImage(ThreadPool* pool) {
    Image3F upscaled(xsize_dc_, ysize_dc_);
    const auto compute_gradient_row = [this, &upscaled](const int task,
                                                        const int thread) {
      for (int c = 0; c < 3; c++) {
        const size_t by = task;
        const auto* row0 = gradient_.PlaneRow(c, by);
        const auto* row1 = gradient_.PlaneRow(c, by + 1);
        for (size_t bx = 0; bx + 1 < xsize_; bx++) {
          float v00 = row0[bx];
          float v01 = row1[bx];
          float v10 = row0[bx + 1];
          float v11 = row1[bx + 1];
          // x1 and y1 are exclusive endpoints and are valid coordinates
          // because there is one more point than amount of blocks.
          size_t x0 = bx * kNumBlocks_;
          size_t x1 = std::min<size_t>(xsize_dc_ - 1, x0 + kNumBlocks_);
          size_t y0 = by * kNumBlocks_;
          size_t y1 = std::min<size_t>(ysize_dc_ - 1, y0 + kNumBlocks_);
          float dx = x1 - x0;
          float dy = y1 - y0;
          for (size_t y = y0; y <= y1; y++) {
            auto* row_out = upscaled.PlaneRow(c, y);
            for (size_t x = x0; x <= x1; x++) {
              row_out[x] =
                  Interpolate(v00, v01, v10, v11, (x - x0) / dx, (y - y0) / dy);
            }
          }
        }
      }
    };
    pool->Run(0, ysize_ - 1, compute_gradient_row);
    return upscaled;
  }

  // Serializes and deserializes the gradient image so it has the values the
  // decoder will see.
  void AccountForQuantization(const Quantizer& quantizer) {
    PaddedBytes bytes = Quantize(quantizer, gradient_);
    gradient_ = Dequantize(quantizer, bytes);
  }

  // The images are in centered opsin color format
  Image3F gradient_;  // corners of the gradient map tiles
  // Size of the superblock, in amount of DCT blocks. So we operate on
  // blocks of kNumBlocks_ * kNumBlocks_ DC components, or 8x8 times as much
  // original image pixels.
  static const size_t kNumBlocks_ = 8;
  // Size of the DC image
  size_t xsize_dc_;
  size_t ysize_dc_;
  // Size of the gradient map (amount of corner points of tiles, one larger than
  // amount of tiles in x and y direction)
  size_t xsize_;
  size_t ysize_;

  bool grayscale_;
};

// This struct allow to remove the X and B channels of centered XYB images, and
// reconstruct them again from only the Y channel, when the image is grayscale.
struct GrayXyb {
  static const constexpr int kM = 16;  // Amount of line pieces.

  GrayXyb() { Compute(); }

  void YToXyb(float y, float* x, float* b) const {
    int i = (int)((y - ysub) * ymul * kM);
    i = std::min(std::max(0, i), kM - 1);
    *x = y * y_to_x_slope[i] + y_to_x_constant[i];
    *b = y * y_to_b_slope[i] + y_to_b_constant[i];
  }

  void RemoveXB(Image3F* image) const {
    for (size_t y = 0; y < image->ysize(); ++y) {
      float* PIK_RESTRICT row_x = image->PlaneRow(0, y);
      float* PIK_RESTRICT row_b = image->PlaneRow(2, y);
      for (size_t x = 0; x < image->xsize(); x++) {
        row_x[x] = 0;
        row_b[x] = 0;
      }
    }
  }

  void RestoreXB(Image3F* image) const {
    for (size_t y = 0; y < image->ysize(); ++y) {
      const float* PIK_RESTRICT row_y = image->PlaneRow(1, y);
      float* PIK_RESTRICT row_x = image->PlaneRow(0, y);
      float* PIK_RESTRICT row_b = image->PlaneRow(2, y);
      for (size_t x = 0; x < image->xsize(); x++) {
        YToXyb(row_y[x], &row_x[x], &row_b[x]);
      }
    }
  }

 private:
  void Compute() {
    static const int kN = 1024;
    std::vector<float> x(kN);
    std::vector<float> y(kN);
    std::vector<float> z(kN);
    for (int i = 0; i < kN; i++) {
      float gray = (float)(256.0f * i / kN);
      LinearToXyb(gray, gray, gray, &x[i], &y[i], &z[i]);
      x[i] -= kXybCenter[0];
      y[i] -= kXybCenter[1];
      z[i] -= kXybCenter[2];
    }

    float min = y[0];
    float max = y[kN - 1];
    int m = 0;
    int border[kM + 1];
    for (int i = 0; i < kN; i++) {
      if (y[i] >= y[0] + (max - min) * m / kM) {
        border[m] = i;
        m++;
      }
    }
    border[kM] = kN;

    ysub = min;
    ymul = 1.0 / (max - min);

    for (int i = 0; i < kM; i++) {
      LinearRegression(y.data() + border[i], x.data() + border[i],
                       border[i + 1] - border[i], &y_to_x_constant[i],
                       &y_to_x_slope[i]);
      LinearRegression(y.data() + border[i], z.data() + border[i],
                       border[i + 1] - border[i], &y_to_b_constant[i],
                       &y_to_b_slope[i]);
    }
  }

  // finds a and b such that y ~= b*x + a
  void LinearRegression(const float* x, const float* y, size_t size, double* a,
                        double* b) {
    double mx = 0, my = 0;    // mean
    double mx2 = 0, my2 = 0;  // second moment
    double mxy = 0;
    for (size_t i = 0; i < size; i++) {
      double inv = 1.0 / (i + 1);

      double dx = x[i] - mx;
      double xn = dx * inv;
      mx += xn;
      mx2 += dx * xn * i;

      double dy = y[i] - my;
      double yn = dy * inv;
      my += yn;
      my2 += dy * yn * i;

      mxy += i * xn * yn - mxy * inv;
    }

    double sx = std::sqrt(mx2 / (size - 1));
    double sy = std::sqrt(my2 / (size - 1));

    double sumxy = mxy * size + my * mx * size;
    double r = (sumxy - size * mx * my) / ((size - 1.0) * sx * sy);

    *b = r * sy / sx;
    *a = my - *b * mx;
  }

  double y_to_x_slope[kM];
  double y_to_x_constant[kM];
  double y_to_b_slope[kM];
  double y_to_b_constant[kM];

  double ysub;
  double ymul;
};

static const std::unique_ptr<GrayXyb> kGrayXyb(new GrayXyb);

// Run |transform| for each block for each color plane.
template <typename TransformBlockFunc>
void TransformBlocks(ThreadPool* pool, size_t xsize_blocks, size_t ysize_blocks,
                     const TransformBlockFunc& transform) {
  const auto transform_row = [&xsize_blocks, &transform](const int task,
                                                         const int thread) {
    const size_t by = task;
    for (int c = 0; c < 3; ++c) {
      for (size_t bx = 0; bx < xsize_blocks; ++bx) {
        transform(c, bx, by);
      }
    }
  };
  pool->Run(0, ysize_blocks, transform_row, "TransformBlocks");
}

void CenterOpsinValues(Image3F* PIK_RESTRICT img) {
  PROFILER_FUNC;
  const size_t xsize = img->xsize();
  const size_t ysize = img->ysize();
  for (int c = 0; c < 3; ++c) {
    for (size_t y = 0; y < ysize; ++y) {
      float* PIK_RESTRICT row = img->PlaneRow(c, y);
      for (size_t x = 0; x < xsize; ++x) {
        row[x] -= kXybCenter[c];
      }
    }
  }
}

// "y_plane" may refer to plane#1 of "coeffs"; it is also organized in the
// block layout (consecutive block coefficient `pixels').
// Class Dequant applies color correlation maps back.
SIMD_ATTR void UnapplyColorCorrelationAC(const ColorCorrelationMap& cmap,
                                         const ImageF& y_plane,
                                         const ImageB& ac_strategy,
                                         Image3F* coeffs) {
  constexpr size_t N = kBlockDim;
  constexpr size_t block_size = N * N;
  const SIMD_FULL(float) d;

  const size_t xsize_blocks = coeffs->xsize() / block_size;
  const size_t ysize_blocks = coeffs->ysize();
  for (size_t y = 0; y < ysize_blocks; ++y) {
    size_t ty = y / kColorTileDimInBlocks;
    const int* PIK_RESTRICT row_ytob = cmap.ytob_map.Row(ty);
    const int* PIK_RESTRICT row_ytox = cmap.ytox_map.Row(ty);

    for (size_t x = 0; x < xsize_blocks; ++x) {
      size_t tx = x / kColorTileDimInBlocks;
      const float* PIK_RESTRICT row_y = y_plane.Row(y) + x * block_size;
      float* PIK_RESTRICT row_x = coeffs->PlaneRow(0, y) + x * block_size;
      float* PIK_RESTRICT row_b = coeffs->PlaneRow(2, y) + x * block_size;
      const auto ytob = set1(d, ColorCorrelationMap::YtoB(1.0f, row_ytob[tx]));
      const auto ytox = set1(d, ColorCorrelationMap::YtoX(1.0f, row_ytox[tx]));
      for (size_t k = 0; k < block_size; k += d.N) {
        const auto in_y = load(d, row_y + k);
        const auto in_b = load(d, row_b + k);
        const auto in_x = load(d, row_x + k);
        const auto out_b = in_b - ytob * in_y;
        const auto out_x = in_x - ytox * in_y;
        store(out_b, d, row_b + k);
        store(out_x, d, row_x + k);
      }
    }
  }
}

SIMD_ATTR void UnapplyColorCorrelationDC(const ColorCorrelationMap& cmap,
                                         const ImageF& y_plane_dc,
                                         Image3F* coeffs_dc) {
  const SIMD_FULL(float) d;
  const size_t xsize_blocks = coeffs_dc->xsize();
  const size_t ysize_blocks = coeffs_dc->ysize();

  const auto ytob = set1(d, ColorCorrelationMap::YtoB(1.0f, cmap.ytob_dc));
  const auto ytox = set1(d, ColorCorrelationMap::YtoX(1.0f, cmap.ytox_dc));

  for (size_t y = 0; y < ysize_blocks; ++y) {
    const float* PIK_RESTRICT row_y = y_plane_dc.Row(y);
    float* PIK_RESTRICT row_x = coeffs_dc->PlaneRow(0, y);
    float* PIK_RESTRICT row_b = coeffs_dc->PlaneRow(2, y);

    for (size_t x = 0; x < xsize_blocks; x += d.N) {
      const auto in_y = load(d, row_y + x);
      const auto in_b = load(d, row_b + x);
      const auto in_x = load(d, row_x + x);
      const auto out_b = in_b - ytob * in_y;
      const auto out_x = in_x - ytox * in_y;
      store(out_b, d, row_b + x);
      store(out_x, d, row_x + x);
    }
  }
}

namespace kernel {

struct Gaborish3_1000 {
  PIK_INLINE const Weights3x3& Weights() const {
    // Unnormalized.
    constexpr float wu0 = 1.0f;
    const float wu1 = static_cast<float>(0.11501538179658321);
    const float wu2 = static_cast<float>(0.089979079587015454);
    const float mul = 1.0 / (wu0 + 4 * (wu1 + wu2));
    const float w0 = wu0 * mul;
    const float w1 = wu1 * mul;
    const float w2 = wu2 * mul;
    static const Weights3x3 weights = {
        {SIMD_REP4(w2)}, {SIMD_REP4(w1)}, {SIMD_REP4(w2)},
        {SIMD_REP4(w1)}, {SIMD_REP4(w0)}, {SIMD_REP4(w1)},
        {SIMD_REP4(w2)}, {SIMD_REP4(w1)}, {SIMD_REP4(w2)}};
    return weights;
  }
};

struct Gaborish3_875 {
  PIK_INLINE const Weights3x3& Weights() const {
    // Unnormalized.
    constexpr float wu0 = 1.0f;
    const float x = 0.875;
    const float wu1 = static_cast<float>(x * 0.11501538179658321);
    const float wu2 = static_cast<float>(x * 0.089979079587015454);
    const float mul = 1.0 / (wu0 + 4 * (wu1 + wu2));
    const float w0 = wu0 * mul;
    const float w1 = wu1 * mul;
    const float w2 = wu2 * mul;
    static const Weights3x3 weights = {
        {SIMD_REP4(w2)}, {SIMD_REP4(w1)}, {SIMD_REP4(w2)},
        {SIMD_REP4(w1)}, {SIMD_REP4(w0)}, {SIMD_REP4(w1)},
        {SIMD_REP4(w2)}, {SIMD_REP4(w1)}, {SIMD_REP4(w2)}};
    return weights;
  }
};

struct Gaborish3_750 {
  PIK_INLINE const Weights3x3& Weights() const {
    // Unnormalized.
    constexpr float wu0 = 1.0f;
    const float x = 0.75;
    const float wu1 = static_cast<float>(x * 0.11501538179658321);
    const float wu2 = static_cast<float>(x * 0.089979079587015454);
    const float mul = 1.0 / (wu0 + 4 * (wu1 + wu2));
    const float w0 = wu0 * mul;
    const float w1 = wu1 * mul;
    const float w2 = wu2 * mul;
    static const Weights3x3 weights = {
        {SIMD_REP4(w2)}, {SIMD_REP4(w1)}, {SIMD_REP4(w2)},
        {SIMD_REP4(w1)}, {SIMD_REP4(w0)}, {SIMD_REP4(w1)},
        {SIMD_REP4(w2)}, {SIMD_REP4(w1)}, {SIMD_REP4(w2)}};
    return weights;
  }
};

struct Gaborish3_500 {
  PIK_INLINE const Weights3x3& Weights() const {
    // Unnormalized.
    constexpr float wu0 = 1.0f;
    const float x = 0.50;
    const float wu1 = static_cast<float>(x * 0.11501538179658321);
    const float wu2 = static_cast<float>(x * 0.089979079587015454);
    const float mul = 1.0 / (wu0 + 4 * (wu1 + wu2));
    const float w0 = wu0 * mul;
    const float w1 = wu1 * mul;
    const float w2 = wu2 * mul;
    static const Weights3x3 weights = {
        {SIMD_REP4(w2)}, {SIMD_REP4(w1)}, {SIMD_REP4(w2)},
        {SIMD_REP4(w1)}, {SIMD_REP4(w0)}, {SIMD_REP4(w1)},
        {SIMD_REP4(w2)}, {SIMD_REP4(w1)}, {SIMD_REP4(w2)}};
    return weights;
  }
};

}  // namespace kernel

void GaborishInverse(Image3F& opsin, double strength) {
  PROFILER_FUNC;
  static const double kGaborish[5] = {
      -0.085173244037473464, -0.038580909941845161, 0.010968069910892428,
      0.0010550132992337335, 0.0010596620544903366,
  };
  const float smooth_weights5[9] = {
      1.0f,
      static_cast<float>(strength * kGaborish[0]),
      static_cast<float>(strength * kGaborish[2]),

      static_cast<float>(strength * kGaborish[0]),
      static_cast<float>(strength * kGaborish[1]),
      static_cast<float>(strength * kGaborish[3]),

      static_cast<float>(strength * kGaborish[2]),
      static_cast<float>(strength * kGaborish[3]),
      static_cast<float>(strength * kGaborish[4]),
  };
  ImageF res[3] = {ImageF(opsin.xsize(), opsin.ysize()),
                   ImageF(opsin.xsize(), opsin.ysize()),
                   ImageF(opsin.xsize(), opsin.ysize())};
  for (int i = 0; i < 3; ++i) {
    slow::SymmetricConvolution<2, WrapClamp>::Run(
        opsin.Plane(i), opsin.xsize(), opsin.ysize(), smooth_weights5, &res[i]);
  }
  Image3F smooth(std::move(res[0]), std::move(res[1]), std::move(res[2]));
  smooth.Swap(opsin);
}

SIMD_ATTR Image3F ConvolveGaborish(Image3F&& in, double strength,
                                   ThreadPool* pool) {
  PROFILER_ZONE("|| gaborish");
  Image3F out(in.xsize(), in.ysize());
  for (int c = 0; c < 3; ++c) {
    if (strength == 1.0) {
      ConvolveT<strategy::Symmetric3>::Run(
          BorderNeverUsed(), ExecutorPool(pool), in.Plane(c),
          kernel::Gaborish3_1000(), out.MutablePlane(c));
    } else if (strength == 0.75) {
      ConvolveT<strategy::Symmetric3>::Run(
          BorderNeverUsed(), ExecutorPool(pool), in.Plane(c),
          kernel::Gaborish3_875(), out.MutablePlane(c));
    } else if (strength == 0.50) {
      ConvolveT<strategy::Symmetric3>::Run(
          BorderNeverUsed(), ExecutorPool(pool), in.Plane(c),
          kernel::Gaborish3_750(), out.MutablePlane(c));
    } else if (strength == 0.25) {
      ConvolveT<strategy::Symmetric3>::Run(
          BorderNeverUsed(), ExecutorPool(pool), in.Plane(c),
          kernel::Gaborish3_500(), out.MutablePlane(c));
    } else {
      return std::move(in);
    }
  }
  out.CheckSizesSame();
  return out;
}

// Avoid including <functional> for such tiny functions.
struct Plus {
  constexpr float operator()(const float first, const float second) const {
    return first + second;
  }
};
struct Minus {
  constexpr float operator()(const float first, const float second) const {
    return first - second;
  }
};

template <size_t N>
std::vector<float> DCfiedGaussianKernel(float sigma) {
  std::vector<float> result(3, 0.0);
  std::vector<float> hires = GaussianKernel<float>(N, sigma);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < hires.size(); j++) {
      result[(i + j) / N] += hires[j] / N;
    }
  }
  return result;
}

// Called from local static ctor.
SIMD_ATTR kernel::Custom<3> MakeUpsampleKernel() {
  Image3F impulse_dc(20, 20);
  FillImage(0.0f, &impulse_dc);
  impulse_dc.PlaneRow(0, 10)[10] = 1.0;
  Image3F probe_expected = UpscalerReconstruct(impulse_dc);
  // We are trying to extract a kernel with a smaller radius. This kernel will
  // be unnormalized. However, we don't mind that: the encoder will compensate
  // when encoding.
  auto kernel6 = kernel::Custom<3>::FromResult(probe_expected.Plane(0));

  ImageF probe_test(probe_expected.xsize(), probe_expected.ysize());
  Upsample<GeneralUpsampler8_6x6>(ExecutorLoop(), impulse_dc.Plane(0), kernel6,
                                  &probe_test);
  VerifyRelativeError(probe_expected.Plane(0), probe_test, 5e-2, 5e-2);

  return kernel6;
}

template <size_t N, class Image>  // ImageF or Image3F
Image BlurUpsampleDC(const Image& original_dc, ThreadPool* pool) {
  const ExecutorPool executor(pool);
  Image out(original_dc.xsize() * N, original_dc.ysize() * N);
  // TODO(user): In the encoder we want only the DC of the result. That could
  // be done more quickly.
  static auto kernel6 = MakeUpsampleKernel();
  Upsample<N>(executor, original_dc, kernel6, &out);
  return out;
}

// Called by local static ctor.
template <size_t N>
std::vector<float> MakeSharpenKernel() {
  // TODO(user): With the new blur, this makes no sense.
  std::vector<float> blur_kernel = DCfiedGaussianKernel<N>(5.5474144946511581);
  constexpr int kSharpenKernelSize = 3;
  std::vector<float> sharpen_kernel(kSharpenKernelSize);
  InvertConvolution(&blur_kernel[0], blur_kernel.size(), &sharpen_kernel[0],
                    sharpen_kernel.size());
  return sharpen_kernel;
}

// Returns true if L1(residual) < max_error.
bool AddResidualAndCompare(const ImageF& dc, const ImageF& blurred_dc,
                           const float max_error,
                           ImageF* PIK_RESTRICT dc_to_encode) {
  PIK_CHECK(SameSize(dc, blurred_dc) && SameSize(dc, *dc_to_encode));
  const size_t xsize = dc.xsize();
  const size_t ysize = dc.ysize();
  bool all_less = true;
  for (size_t y = 0; y < ysize; ++y) {
    const float* PIK_RESTRICT row_dc = dc.ConstRow(y);
    const float* PIK_RESTRICT row_blurred = blurred_dc.ConstRow(y);
    float* PIK_RESTRICT row_out = dc_to_encode->Row(y);
    for (size_t x = 0; x < xsize; ++x) {
      const float diff = row_dc[x] - row_blurred[x];
      all_less &= fabsf(diff) < max_error;
      row_out[x] += diff;
    }
  }
  return all_less;
}

template <size_t N>
Image3F SharpenDC(const Image3F& original_dc, ThreadPool* pool) {
  PROFILER_FUNC;
  constexpr int kMaxIters = 200;
  constexpr float kAcceptableError[3] = {1e-4, 1e-4, 1e-4};

  static const std::vector<float> sharpen_kernel = MakeSharpenKernel<N>();
  Image3F dc_to_encode = Convolve(original_dc, sharpen_kernel);

  // Individually per channel, until error is acceptable:
  for (int c = 0; c < 3; ++c) {
    for (int iter = 0; iter < kMaxIters; iter++) {
      const ImageF up = BlurUpsampleDC<N>(dc_to_encode.Plane(c), pool);
      const ImageF blurred = ScaledDC<N>(up, pool);
      // Change pixels of dc_to_encode but not its size.
      if (AddResidualAndCompare(original_dc.Plane(c), blurred,
                                kAcceptableError[c],
                                dc_to_encode.MutablePlane(c))) {
        break;  // next channel
      }
    }
    dc_to_encode.CheckSizesSame();
  }

  return dc_to_encode;
}

void RecomputeGradient(const Quantizer& quantizer, ThreadPool* pool,
                       EncCache* cache) {
  if (!cache->use_gradient) return;

  GradientMap gradient_map(cache->xsize_blocks, cache->ysize_blocks,
                           cache->grayscale_opt);
  gradient_map.ComputeFromSource(cache->dc_init, quantizer, pool);
  cache->gradient = CopyImage(gradient_map.GetGradientMap());
}

void EnsureEncCachePerThreadInitialized(EncCachePerThread* cache) {
  if (cache->initialized) return;

  cache->diff = ImageF(kTileDim, kTileDim);
  cache->initialized = true;
}

// Toy AC strategy.
void StrategyResampleAll(ImageB* image) {
  const size_t xsize_tiles = DivCeil(image->xsize(), kTileDimInBlocks);
  const size_t ysize_tiles = DivCeil(image->ysize(), kTileDimInBlocks);
  for (size_t ty = 0; ty < ysize_tiles; ++ty) {
    for (size_t tx = 0; tx < xsize_tiles; ++tx) {
      const size_t bx0 = tx * kTileDimInBlocks;
      const size_t by0 = ty * kTileDimInBlocks;
      size_t resample_share_x = kTileDimInBlocks;
      size_t resample_share_y = kTileDimInBlocks;
      const size_t bx1 = std::min(bx0 + kTileDimInBlocks, image->xsize());
      const size_t by1 = std::min(by0 + kTileDimInBlocks, image->ysize());
      const size_t w = bx1 - bx0;
      const size_t h = by1 - by0;
      uint8_t tile_strategy = AcStrategyType::NONE;
      if (w == kTileDimInBlocks) {
        tile_strategy |= AcStrategyType::RESAMPLE_X * 2;
        resample_share_x = (2 * resample_share_x) / 8;
      }
      if (h == kTileDimInBlocks) {
        tile_strategy |= AcStrategyType::RESAMPLE_Y * 2;
        resample_share_y = (2 * resample_share_y) / 8;
      }
      for (size_t by = 0; by < h; ++by) {
        for (size_t bx = 0; bx < w; ++bx) {
          uint8_t strategy = AcStrategyType::DCT;
          if (bx >= resample_share_x) strategy = AcStrategyType::NONE;
          if (by >= resample_share_y) strategy = AcStrategyType::NONE;
          image->Row(by0 + by)[bx0 + bx] = strategy;
        }
      }
      // Per-tile strategy overlay.
      image->Row(by0)[bx0] |= tile_strategy;
    }
  }
}

void SubsampleTile(uint8_t strategy, ImageF* tile) {
  const uint8_t resample_x = (strategy / AcStrategyType::RESAMPLE_X) & 0x7;
  const uint8_t resample_y = (strategy / AcStrategyType::RESAMPLE_Y) & 0x7;
  size_t w = kTileDim;
  size_t h = kTileDim;

  if (resample_y != 0) {
    h = (h * resample_y) / 8;
    // TODO(user): do y-resampling.
  }

  if (resample_x != 0) {
    w = (w * resample_x) / 8;
    // TODO(user): do x-resampling.
  }
}

void UpsampleTile(uint8_t strategy, const Rect tile_rect, ImageF* idct) {
  const uint8_t resample_x = (strategy / AcStrategyType::RESAMPLE_X) & 0x7;
  const uint8_t resample_y = (strategy / AcStrategyType::RESAMPLE_Y) & 0x7;
  size_t w = tile_rect.xsize();
  size_t h = tile_rect.ysize();
  if (resample_x != 0) w = (w * resample_x) / 8;
  if (resample_y != 0) h = (h * resample_y) / 8;

  if (resample_x != 0) {
    // TODO(user): do x-resampling.
    w = tile_rect.xsize();
  }

  if (resample_y != 0) {
    // TODO(user): do y-resampling.
    h = tile_rect.ysize();
  }
}

void SubsampleQuantFieldTile(uint8_t strategy, const Rect tile_rect,
                             ImageI* quant) {
  const uint8_t resample_x = (strategy / AcStrategyType::RESAMPLE_X) & 0x7;
  const uint8_t resample_y = (strategy / AcStrategyType::RESAMPLE_Y) & 0x7;
  size_t w = tile_rect.xsize();
  size_t h = tile_rect.ysize();

  if (resample_y != 0) {
    h = (h * resample_y) / 8;
    // TODO(user): do y-resampling.
  }

  if (resample_x != 0) {
    w = (w * resample_x) / 8;
    // TODO(user): do x-resampling.
  }
}

void SubsampleQuantField(const AcStrategy& ac_strategy, ImageI* quant_field) {
  constexpr size_t N = kBlockDim;
  constexpr size_t tile_dim_blocks = kTileDim / N;
  const size_t xsize_blocks = quant_field->xsize();
  const size_t ysize_blocks = quant_field->ysize();

  for (size_t by = 0; by < ysize_blocks; by += tile_dim_blocks) {
    for (size_t bx = 0; bx < xsize_blocks; bx += tile_dim_blocks) {
      const uint8_t strategy = ac_strategy.layers.ConstRow(by)[bx];
      const Rect rect(bx, by, tile_dim_blocks, tile_dim_blocks, xsize_blocks,
                      ysize_blocks);
      SubsampleQuantFieldTile(strategy, rect, quant_field);
    }
  }
}

// See also AddPredictions_Smooth.
SIMD_ATTR void ComputePredictionResiduals_Smooth(const Quantizer& quantizer,
                                                 ThreadPool* pool,
                                                 EncCache* cache) {
  PROFILER_FUNC;
  constexpr size_t N = kBlockDim;
  PIK_ASSERT(N == quantizer.block_dim());

  // Smooth mode depends only on DC quantization.
  if (cache->last_quant_dc_key == quantizer.QuantDcKey()) {
    return;
  }

  RecomputeGradient(quantizer, pool, cache);

  Image3F dc_dec = QuantizeRoundtripDC(quantizer, cache->dc_sharp);
  Image3F blurred = BlurUpsampleDC<N>(dc_dec, pool);
  PIK_ASSERT(blurred.xsize() % N == 0);
  PIK_ASSERT(blurred.ysize() % N == 0);
  const size_t xsize_tiles = DivCeil(cache->xsize_blocks * N, kTileDim);
  const size_t ysize_tiles = DivCeil(cache->ysize_blocks * N, kTileDim);
  const size_t tile_count = xsize_tiles * ysize_tiles;
  const auto process_tile = [xsize_tiles, &cache, &blurred](const int task,
                                                            const int thread) {
    // Would use "full", but it will require more careful tail processing.
    constexpr size_t N = kBlockDim;
    constexpr size_t block_size = N * N;
    constexpr size_t tile_dim_blocks = kTileDim / N;
    constexpr BlockDesc d;
    EncCachePerThread& thread_cache = cache->buffers[thread];
    EnsureEncCachePerThreadInitialized(&thread_cache);
    ImageF* diff = &thread_cache.diff;
    const size_t tile_stride = diff->PixelsPerRow();
    const size_t tile_idx = task;
    const size_t ty = tile_idx / xsize_tiles;
    const size_t tx = tile_idx % xsize_tiles;
    const size_t by0 = ty * tile_dim_blocks;
    const size_t bx0 = tx * tile_dim_blocks;
    const size_t bx1 = std::min(bx0 + tile_dim_blocks, cache->xsize_blocks);
    const size_t by1 = std::min(by0 + tile_dim_blocks, cache->ysize_blocks);
    const size_t bw = bx1 - bx0;
    const size_t bh = by1 - by0;
    const size_t w = bw * N;
    const size_t h = bh * N;
    const ImageB& strategies = cache->ac_strategy.layers;
    for (size_t c = 0; c < 3; ++c) {
      const ImageF& orig = cache->src->Plane(c);
      const ImageF& pred = blurred.Plane(c);
      for (size_t y = 0; y < h; ++y) {
        const float* PIK_RESTRICT orig_in =
            orig.ConstRow(by0 * N + y) + bx0 * N;
        const float* PIK_RESTRICT pred_in =
            pred.ConstRow(by0 * N + y) + bx0 * N;
        float* PIK_RESTRICT diff_out = diff->Row(y);
        for (size_t x = 0; x < w; x += d.N) {
          store(load(d, orig_in + x) - load(d, pred_in + x), d, diff_out + x);
        }
      }

      SubsampleTile(strategies.ConstRow(by0)[bx0], diff);

      ImageF* output = cache->coeffs.MutablePlane(c);
      for (size_t by = 0; by < bh; ++by) {
        const float* PIK_RESTRICT row_diff = diff->Row(by * N);
        const uint8_t* row_strategy = strategies.ConstRow(by0 + by) + bx0;
        float* PIK_RESTRICT row_out = output->Row(by0 + by) + bx0 * block_size;
        for (size_t bx = 0; bx < bw; ++bx) {
          const uint8_t block_strategy = row_strategy[bx];
#ifdef ADDRESS_SANITIZER
          PIK_ASSERT(AcStrategyType::IsDct(block_strategy));
#endif
          // TODO(user): support other strategies.
          if (AcStrategyType::IsDct(block_strategy)) {
            ComputeTransposedScaledDCT<N>(
                FromLines(row_diff + bx * N, tile_stride),
                ScaleToBlock<N>(row_out + bx * block_size));
          }
        }
      }
    }
  };
  pool->Run(0, tile_count, process_tile, "enc PredSmooth");

  cache->last_quant_dc_key = quantizer.QuantDcKey();
}

namespace kernel {

constexpr float kWeight0 = 0.027630534023046f;
constexpr float kWeight1 = 0.133676439523697f;
constexpr float kWeight2 = 0.035697385668755f;

struct AC11 {
  PIK_INLINE const Weights3x3& Weights() const {
    static constexpr Weights3x3 weights = {
        {SIMD_REP4(kWeight2)},  {SIMD_REP4(0.)}, {SIMD_REP4(-kWeight2)},
        {SIMD_REP4(0.)},        {SIMD_REP4(0.)}, {SIMD_REP4(0.)},
        {SIMD_REP4(-kWeight2)}, {SIMD_REP4(0.)}, {SIMD_REP4(kWeight2)}};
    return weights;
  }
};

struct AC01 {
  PIK_INLINE const Weights3x3& Weights() const {
    static constexpr Weights3x3 weights = {
        {SIMD_REP4(kWeight0)},  {SIMD_REP4(kWeight1)},  {SIMD_REP4(kWeight0)},
        {SIMD_REP4(0.)},        {SIMD_REP4(0.)},        {SIMD_REP4(0.)},
        {SIMD_REP4(-kWeight0)}, {SIMD_REP4(-kWeight1)}, {SIMD_REP4(-kWeight0)}};
    return weights;
  }
};

struct AC10 {
  PIK_INLINE const Weights3x3& Weights() const {
    static constexpr Weights3x3 weights = {
        {SIMD_REP4(kWeight0)}, {SIMD_REP4(0.)}, {SIMD_REP4(-kWeight0)},
        {SIMD_REP4(kWeight1)}, {SIMD_REP4(0.)}, {SIMD_REP4(-kWeight1)},
        {SIMD_REP4(kWeight0)}, {SIMD_REP4(0.)}, {SIMD_REP4(-kWeight0)}};
    return weights;
  }
};

}  // namespace kernel

// NB: only HVD coefficients are updated; DC remains intact.
template <class Operator>  // Plus/Minus
SIMD_ATTR void AdjustHVD_64FromDC(const Image3F& dc,
                                  Image3F* PIK_RESTRICT coeffs) {
  constexpr size_t N = kBlockDim;
  constexpr size_t block_size = N * N;
  const size_t xsize = dc.xsize();
  const size_t ysize = dc.ysize();

  const Operator op;

  Image3F ac01(xsize, ysize);
  Image3F ac10(xsize, ysize);
  Image3F ac11(xsize, ysize);
  // Must avoid ConvolveT for tiny images (PIK_CHECK fails)
  if (xsize < kConvolveMinWidth) {
    using Convolution = slow::General3x3Convolution<1, WrapMirror>;
    Convolution::Run(dc, xsize, ysize, kernel::AC01(), &ac01);
    Convolution::Run(dc, xsize, ysize, kernel::AC10(), &ac10);
    Convolution::Run(dc, xsize, ysize, kernel::AC11(), &ac11);
  } else {
    ConvolveT<strategy::GradY3>::Run(dc, kernel::AC01(), &ac01);
    ConvolveT<strategy::GradX3>::Run(dc, kernel::AC10(), &ac10);
    ConvolveT<strategy::Corner3>::Run(dc, kernel::AC11(), &ac11);
  }
  for (int c = 0; c < 3; ++c) {
    for (size_t by = 0; by < ysize; ++by) {
      const float* PIK_RESTRICT row01 = ac01.ConstPlaneRow(c, by);
      const float* PIK_RESTRICT row10 = ac10.ConstPlaneRow(c, by);
      const float* PIK_RESTRICT row11 = ac11.ConstPlaneRow(c, by);
      float* SIMD_RESTRICT row_out = coeffs->PlaneRow(c, by);
      for (size_t bx = 0; bx < xsize; ++bx) {
        float* PIK_RESTRICT block = row_out + block_size * bx;
        block[1] = op(block[1], row01[bx]);
        block[N] = op(block[N], row10[bx]);
        block[N + 1] = op(block[N + 1], row11[bx]);
      }
    }
  }
}

// Returns pixel-space prediction using same adjustment as above followed by
// GetPixelSpaceImageFrom0HVD. "ac4" is [--, 1, 8, 9].
// Parallelizing doesn't help, even for XX MP images (DC is still small).
SIMD_ATTR Image3F PredictSpatial2x2_AC4(const Image3F& dc, const Image3F& ac4,
                                        const ImageB& ac_strategy) {
  constexpr size_t N = kBlockDim;
  const size_t xsize = dc.xsize();
  const size_t ysize = dc.ysize();

  Image3F ac01(xsize, ysize);
  Image3F ac10(xsize, ysize);
  Image3F ac11(xsize, ysize);
  // Must avoid ConvolveT for tiny images (PIK_CHECK fails)
  if (xsize < kConvolveMinWidth) {
    using Convolution = slow::General3x3Convolution<1, WrapMirror>;
    Convolution::Run(dc, xsize, ysize, kernel::AC01(), &ac01);
    Convolution::Run(dc, xsize, ysize, kernel::AC10(), &ac10);
    Convolution::Run(dc, xsize, ysize, kernel::AC11(), &ac11);
  } else {
    ConvolveT<strategy::GradY3>::Run(dc, kernel::AC01(), &ac01);
    ConvolveT<strategy::GradX3>::Run(dc, kernel::AC10(), &ac10);
    ConvolveT<strategy::Corner3>::Run(dc, kernel::AC11(), &ac11);
  }

  // TODO(user): what is this magic constant?
  const float magic = 0.9061274463528878f;  // sin(7 * pi / 16) * cos(pi / 8)
  const float kScale01 = N * magic * DCTScales<N>()[0] * DCTScales<N>()[1];
  const float kScale11 =
      N * magic * magic * DCTScales<N>()[1] * DCTScales<N>()[1];
  Image3F out2x2(xsize * 2, ysize * 2);
  for (int c = 0; c < 3; ++c) {
    for (size_t by = 0; by < ysize; ++by) {
      const float* PIK_RESTRICT row_dc = dc.ConstPlaneRow(c, by);
      const float* PIK_RESTRICT row_ac = ac4.ConstPlaneRow(c, by);
      const float* PIK_RESTRICT row01 = ac01.ConstPlaneRow(c, by);
      const float* PIK_RESTRICT row10 = ac10.ConstPlaneRow(c, by);
      const float* PIK_RESTRICT row11 = ac11.ConstPlaneRow(c, by);
      const uint8_t* PIK_RESTRICT ac_strategy_row = ac_strategy.ConstRow(by);
      float* SIMD_RESTRICT row_out0 = out2x2.PlaneRow(c, by * 2 + 0);
      float* SIMD_RESTRICT row_out1 = out2x2.PlaneRow(c, by * 2 + 1);
      for (size_t bx = 0; bx < xsize; ++bx) {
        const bool ac00_from_dc = AcStrategyType::IsDct(ac_strategy_row[bx]);
        const float* PIK_RESTRICT block_ac = row_ac + bx * 4;
        const float a00 = ac00_from_dc ? row_dc[bx] : block_ac[0];
        const float a01 = (block_ac[2] + row10[bx]) * kScale01;
        const float a10 = (block_ac[1] + row01[bx]) * kScale01;
        const float a11 = (block_ac[3] + row11[bx]) * kScale11;
        row_out0[2 * bx + 0] = a00 + a01 + a10 + a11;
        row_out0[2 * bx + 1] = a00 - a01 + a10 - a11;
        row_out1[2 * bx + 0] = a00 + a01 - a10 - a11;
        row_out1[2 * bx + 1] = a00 - a01 - a10 + a11;
      }
    }
  }

  return out2x2;
}

// Returns pixel-space prediction using same adjustment as above followed by
// GetPixelSpaceImageFrom0HVD. Modifies "ac64" 0HVD.
SIMD_ATTR Image3F PredictSpatial2x2_AC64(const Image3F& dc,
                                         const ImageB& ac_strategy,
                                         ThreadPool* pool, Image3F* ac64) {
  constexpr size_t N = kBlockDim;
  constexpr size_t block_size = N * N;
  const size_t xsize = dc.xsize();
  const size_t ysize = dc.ysize();

  Image3F ac01(xsize, ysize);
  Image3F ac10(xsize, ysize);
  Image3F ac11(xsize, ysize);
  // Must avoid ConvolveT for tiny images (PIK_CHECK fails)
  if (xsize < kConvolveMinWidth) {
    using Convolution = slow::General3x3Convolution<1, WrapMirror>;
    Convolution::Run(dc, xsize, ysize, kernel::AC01(), &ac01);
    Convolution::Run(dc, xsize, ysize, kernel::AC10(), &ac10);
    Convolution::Run(dc, xsize, ysize, kernel::AC11(), &ac11);
  } else {
    const BorderNeverUsed border;
    // Parallel doesn't help here for moderate-sized images.
    const ExecutorLoop executor;
    ConvolveT<strategy::GradY3>::Run(border, executor, dc, kernel::AC01(),
                                     &ac01);
    ConvolveT<strategy::GradX3>::Run(border, executor, dc, kernel::AC10(),
                                     &ac10);
    ConvolveT<strategy::Corner3>::Run(border, executor, dc, kernel::AC11(),
                                      &ac11);
  }

  // TODO(user): what is this magic constant?
  const float magic = 0.9061274463528878f;  // sin(7 * pi / 16) * cos(pi / 8)
  const float kScale01 = N * magic * DCTScales<N>()[0] * DCTScales<N>()[1];
  const float kScale11 =
      N * magic * magic * DCTScales<N>()[1] * DCTScales<N>()[1];
  Image3F out2x2(xsize * 2, ysize * 2);
  const auto process_row = [&](const int task, const int thread) {
    const size_t by = task;
    for (size_t c = 0; c < 3; ++c) {
      const float* PIK_RESTRICT row_dc = dc.ConstPlaneRow(c, by);
      float* PIK_RESTRICT row_ac64 = ac64->PlaneRow(c, by);
      const float* PIK_RESTRICT row01 = ac01.ConstPlaneRow(c, by);
      const float* PIK_RESTRICT row10 = ac10.ConstPlaneRow(c, by);
      const float* PIK_RESTRICT row11 = ac11.ConstPlaneRow(c, by);
      const uint8_t* PIK_RESTRICT ac_strategy_row = ac_strategy.ConstRow(by);
      float* SIMD_RESTRICT row_out0 = out2x2.PlaneRow(c, by * 2 + 0);
      float* SIMD_RESTRICT row_out1 = out2x2.PlaneRow(c, by * 2 + 1);
      for (size_t bx = 0; bx < xsize; ++bx) {
        const bool ac00_from_dc = AcStrategyType::IsDct(ac_strategy_row[bx]);
        float* PIK_RESTRICT block_ac = row_ac64 + bx * block_size;
        block_ac[0] = ac00_from_dc ? row_dc[bx] : block_ac[0];
        block_ac[1] += row01[bx];
        block_ac[N] += row10[bx];
        block_ac[N + 1] += row11[bx];

        const float a00 = block_ac[0];
        const float a01 = block_ac[N] * kScale01;
        const float a10 = block_ac[1] * kScale01;
        const float a11 = block_ac[N + 1] * kScale11;
        row_out0[2 * bx + 0] = a00 + a01 + a10 + a11;
        row_out0[2 * bx + 1] = a00 - a01 + a10 - a11;
        row_out1[2 * bx + 0] = a00 + a01 - a10 - a11;
        row_out1[2 * bx + 1] = a00 - a01 - a10 + a11;
      }
    }
  };
  pool->Run(0, ysize, process_row);

  return out2x2;
}

// See also AddPredictions.
SIMD_ATTR void ComputePredictionResiduals(const Quantizer& quantizer,
                                          ThreadPool* pool, EncCache* cache) {
  if (cache->last_quant_dc_key != quantizer.QuantDcKey()) {
    RecomputeGradient(quantizer, pool, cache);
    cache->dc_dec = QuantizeRoundtripDC(quantizer, cache->dc_init);

    // Do not overwrite coeffs_init itself.
    cache->coeffs_dec = CopyImage(cache->coeffs_init);
    AdjustHVD_64FromDC<Minus>(cache->dc_dec, &cache->coeffs_dec);
    cache->last_quant_dc_key = quantizer.QuantDcKey();
  }
  // TODO(user): store/process HVD separately; cache ac_hvd_rounded.
  Image3F ac_hvd_rounded = QuantizeRoundtripExtractHVD(
      quantizer, cache->coeffs_dec, cache->ac_strategy.layers);
  for (size_t c = 0; c < cache->coeffs_dec.kNumPlanes; c++) {
    for (size_t by = 0; by < cache->dc_dec.ysize(); ++by) {
      const float* PIK_RESTRICT row_in = cache->coeffs_dec.ConstPlaneRow(c, by);
      float* PIK_RESTRICT row_out = ac_hvd_rounded.PlaneRow(c, by);
      for (size_t bx = 0; bx < cache->dc_dec.xsize(); ++bx) {
        row_out[bx * 4] = row_in[bx * kBlockDim * kBlockDim];
      }
    }
  }
  Image3F pred2x2 = PredictSpatial2x2_AC4(cache->dc_dec, ac_hvd_rounded,
                                          cache->ac_strategy.layers);
  cache->coeffs = CopyImage(cache->coeffs_dec);
  UpSample4x4BlurDCT(pred2x2, 1.5f, -0.0f, pool, &cache->coeffs);
  const auto subtract_dc = [&](size_t c, size_t bx, size_t by) {
    if (AcStrategyType::IsIdentity(
            cache->ac_strategy.layers.ConstRow(by)[bx])) {
      cache->coeffs.PlaneRow(c, by)[bx * kBlockDim * kBlockDim] -=
          cache->dc_dec.ConstPlaneRow(c, by)[bx];
    }
  };
  if (cache->ac_strategy.use_ac_strategy)
    TransformBlocks(pool, cache->dc.xsize(), cache->dc.ysize(), subtract_dc);
}

SIMD_ATTR void ComputeInitialCoefficients(const Header& header,
                                          const Image3F& opsin,
                                          ThreadPool* pool, EncCache* cache) {
  constexpr size_t N = kBlockDim;
  constexpr size_t block_size = N * N;
  PIK_ASSERT(!cache->initialized);

  cache->xsize_blocks = opsin.xsize() / N;
  cache->ysize_blocks = opsin.ysize() / N;
  cache->is_smooth = header.flags & Header::kSmoothDCPred;
  cache->use_gradient = header.flags & Header::kGradientMap;
  cache->grayscale_opt = header.flags & Header::kGrayscaleOpt;
  cache->ac_strategy.use_ac_strategy = header.flags & Header::kUseAcStrategy;

  if (cache->is_smooth) {
    cache->src = &opsin;
    cache->dc_init = ScaledDC<N>(opsin, pool);
    cache->coeffs =
        Image3F(cache->xsize_blocks * block_size, cache->ysize_blocks);
    cache->dc_sharp = SharpenDC<N>(cache->dc_init, pool);
    size_t max_threads = std::max<size_t>(1, pool->NumThreads());
    if (cache->buffers.size() < max_threads) {
      cache->buffers.resize(max_threads);
    }
  } else {
    cache->src = &opsin;
    cache->coeffs_init = TransposedScaledDCT<N>(opsin, pool);
    cache->dc_init = DCImage<N>(cache->coeffs_init);
  }

  cache->initialized = true;
}

SIMD_ATTR void ComputeCoefficients(const Quantizer& quantizer,
                                   const ColorCorrelationMap& cmap,
                                   ThreadPool* pool, EncCache* cache,
                                   const PikInfo* aux_out) {
  constexpr size_t N = kBlockDim;
  constexpr size_t block_size = N * N;
  const size_t xsize_blocks = cache->xsize_blocks;
  const size_t ysize_blocks = cache->ysize_blocks;
  PIK_ASSERT(cache->initialized);

  const auto process_block = [cache](int c, size_t bx, size_t by) {
    if (AcStrategyType::IsIdentity(
            cache->ac_strategy.layers.ConstRow(by)[bx])) {
      // First row is encoded as increments.
      float* PIK_RESTRICT first_out =
          cache->coeffs_init.PlaneRow(c, by) + block_size * bx;
      const float* PIK_RESTRICT first_in =
          cache->src->ConstPlaneRow(c, by * N) + N * bx;
      // First pixel is special - encoded as it is.
      first_out[0] = first_in[0];
      float running_avg[N] = {first_in[0]};
      for (size_t ix = 1; ix < N; ++ix) {
        first_out[ix] = first_in[ix] - running_avg[ix - 1];
        running_avg[ix] = kIdentityAvgParam * running_avg[ix - 1] +
                          (1.0f - kIdentityAvgParam) * first_in[ix];
      }
      for (size_t iy = 1; iy < N; ++iy) {
        float* PIK_RESTRICT pos_out =
            cache->coeffs_init.PlaneRow(c, by) + block_size * bx + iy * N;
        const float* PIK_RESTRICT pos_in =
            cache->src->ConstPlaneRow(c, by * N + iy) + N * bx;
        // First column is encoded as increments too.
        pos_out[0] = pos_in[0] - running_avg[0];
        running_avg[0] = kIdentityAvgParam * running_avg[0] +
                         (1.0f - kIdentityAvgParam) * pos_in[0];
        // Every other pixel is encoded as the difference between the
        // average of the one on the left and the one above.
        for (size_t ix = 1; ix < N; ++ix) {
          float predicted = (running_avg[ix - 1] + running_avg[ix]) * 0.5f;
          pos_out[ix] = pos_in[ix] - predicted;
          running_avg[ix] = running_avg[ix] * kIdentityAvgParam +
                            (1.0f - kIdentityAvgParam) *
                                (kIdentityAvgParam * running_avg[ix - 1] +
                                 (1.0f - kIdentityAvgParam) * pos_in[ix]);
        }
      }
    } else if (AcStrategyType::IsDct16x16(
                   cache->ac_strategy.layers.ConstRow(by)[bx])) {
      SIMD_ALIGN float output[4 * N * N] = {};
      ComputeTransposedScaledDCT<2 * N>(
          FromLines(cache->src->ConstPlaneRow(c, N * by) + N * bx,
                    cache->src->PixelsPerRow()),
          ScaleToBlock<2 * N>(output));
      // Permute block so that applying zig-zag order to each sub-block
      // will result in the zig-zag order of the whole DCT16 block.
      // TODO(user): also consider putting coefficients in blocks by
      // alternating blocks, i.e. according to pos%4.
      for (size_t k = 0; k < 4 * N * N; k++) {
        size_t zigzag_pos = kNaturalCoeffOrderLut16[k];
        size_t dest_block = zigzag_pos / (N * N);
        size_t block_pos = zigzag_pos - dest_block * N * N;
        size_t block_pos_unzigzag = kNaturalCoeffOrder8[block_pos];
        size_t dest_block_y = by + dest_block / 2;
        size_t dest_block_x = bx + dest_block % 2;
        cache->coeffs_init.PlaneRow(
            c, dest_block_y)[dest_block_x * N * N + block_pos_unzigzag] =
            output[k] * kDct16Scale;
      }
    }
  };
  TransformBlocks(pool, xsize_blocks, ysize_blocks, process_block);

  cache->quant_field = CopyImage(quantizer.RawQuantField());
  ImageI& quant_field = cache->quant_field;

  if (cache->is_smooth) {
    ComputePredictionResiduals_Smooth(quantizer, pool, cache);
  } else {
    ComputePredictionResiduals(quantizer, pool, cache);
  }

  SubsampleQuantField(cache->ac_strategy, &quant_field);

  PROFILER_ZONE("enc cmap+quant");

  constexpr int cY = 1;  // Y color channel.

  // TODO(user): it would be better to find & apply correlation here, when
  // quantization is chosen.

  {
    Image3F coeffs_dc =
        CopyImage(cache->is_smooth ? cache->dc_sharp : cache->dc_init);

    ImageF dec_dc_Y = QuantizeRoundtripDC(quantizer, cY, coeffs_dc.Plane(cY));

    if (cache->grayscale_opt) {
      kGrayXyb->RemoveXB(&coeffs_dc);
    } else {
      UnapplyColorCorrelationDC(cmap, dec_dc_Y, &coeffs_dc);
    }

    cache->dc = QuantizeCoeffsDC(coeffs_dc, quantizer);
  }

  {
    Image3F coeffs_ac = CopyImage(cache->coeffs);

    ImageF dec_ac_Y(xsize_blocks * block_size, ysize_blocks);
    for (size_t by = 0; by < ysize_blocks; ++by) {
      const float* PIK_RESTRICT row_in = coeffs_ac.ConstPlaneRow(cY, by);
      float* PIK_RESTRICT row_out = dec_ac_Y.Row(by);
      for (size_t bx = 0; bx < xsize_blocks; ++bx) {
        const int32_t quant_ac = quant_field.Row(by)[bx];
        quantizer.QuantizeRoundtripBlockAC(
            cY, quant_ac,
            GetQuantKindFromAcStrategy(cache->ac_strategy.layers, bx, by),
            row_in + bx * block_size, row_out + bx * block_size);
      }
    }

    UnapplyColorCorrelationAC(cmap, dec_ac_Y, cache->ac_strategy.layers,
                              &coeffs_ac);

    cache->ac = Image3S(xsize_blocks * block_size, ysize_blocks);
    for (int c = 0; c < 3; ++c) {
      for (size_t by = 0; by < ysize_blocks; ++by) {
        const float* PIK_RESTRICT row_in = coeffs_ac.PlaneRow(c, by);
        int16_t* PIK_RESTRICT row_out = cache->ac.PlaneRow(c, by);
        const int32_t* row_quant = quant_field.ConstRow(by);
        for (size_t bx = 0; bx < xsize_blocks; ++bx) {
          const float* PIK_RESTRICT block_in = &row_in[bx * block_size];
          int16_t* PIK_RESTRICT block_out = &row_out[bx * block_size];
          quantizer.QuantizeBlockAC(
              row_quant[bx],
              GetQuantKindFromAcStrategy(cache->ac_strategy.layers, bx, by), c,
              block_in, block_out);
        }
      }
    }
  }
}

// Computes contexts in [0, kOrderContexts) from "rect_dc" within "dc" and
// writes to "rect_ctx" within "ctx".
SIMD_ATTR void ComputeBlockContextFromDC(const Rect& rect_dc, const Image3S& dc,
                                         const Quantizer& quantizer,
                                         const Rect& strategy_rect,
                                         const ImageB& ac_strategy,
                                         const Rect& rect_ctx,
                                         Image3B* PIK_RESTRICT ctx) {
  PROFILER_FUNC;
  PIK_ASSERT(SameSize(rect_dc, rect_ctx));
  const size_t xsize = rect_dc.xsize();
  const size_t ysize = rect_dc.ysize();

  const float iquant_base = quantizer.inv_quant_dc();
  for (int c = 0; c < 3; ++c) {
    const ImageS& plane_dc = dc.Plane(c);
    ImageB* PIK_RESTRICT plane_ctx = ctx->MutablePlane(c);

    // DC quantization uses DCT8 values.
    const float iquant =
        iquant_base * quantizer.DequantMatrix(c, kQuantKindDCT8)[0];
    const float range = kXybRange[c] / iquant;
    const int64_t kR2Thresh = std::min(10.24f * range * range + 1.0f, 1E18f);

    for (size_t y = 0; y < ysize; ++y) {
      const int16_t* PIK_RESTRICT row_t =
          rect_dc.ConstRow(plane_dc, y == 0 ? y : y - 1);
      const int16_t* PIK_RESTRICT row_m = rect_dc.ConstRow(plane_dc, y);
      const int16_t* PIK_RESTRICT row_b =
          rect_dc.ConstRow(plane_dc, y + 1 == ysize ? y : y + 1);
      const uint8_t* PIK_RESTRICT ac_strategy_last_row =
          strategy_rect.ConstRow(ac_strategy, y == 0 ? y : y - 1);
      const uint8_t* PIK_RESTRICT ac_strategy_row =
          strategy_rect.ConstRow(ac_strategy, y);
      uint8_t* PIK_RESTRICT row_out = rect_ctx.Row(plane_ctx, y);
      for (size_t bx = 0; bx < xsize; ++bx) {
        if (AcStrategyType::IsDct(ac_strategy_row[bx])) {
          if (y == 0 || y + 1 == ysize || bx == 0 || bx + 1 == xsize) {
            row_out[bx] = kFlatOrderContextStart + c;
            continue;
          }
          const int16_t val_tl = row_t[bx - 1];
          const int16_t val_tm = row_t[bx];
          const int16_t val_tr = row_t[bx + 1];
          const int16_t val_ml = row_m[bx - 1];
          const int16_t val_mr = row_m[bx + 1];
          const int16_t val_bl = row_b[bx - 1];
          const int16_t val_bm = row_b[bx];
          const int16_t val_br = row_b[bx + 1];
          const int64_t dx = (3 * (val_tr - val_tl + val_br - val_bl) +
                              10 * (val_mr - val_ml));
          const int64_t dy = (3 * (val_bl - val_tl + val_br - val_tr) +
                              10 * (val_bm - val_tm));
          const int64_t dx2 = dx * dx;
          const int64_t dy2 = dy * dy;
          const int64_t dxdy = std::abs(2 * dx * dy);
          const int64_t r2 = dx2 + dy2;
          const int64_t d2 = dy2 - dx2;
          if (r2 < kR2Thresh) {
            row_out[bx] = kFlatOrderContextStart + c;
          } else if (d2 < -dxdy) {
            row_out[bx] = kDirectionalOrderContextStart;
          } else if (d2 > dxdy) {
            row_out[bx] = kDirectionalOrderContextStart + 2;
          } else {
            row_out[bx] = kDirectionalOrderContextStart + 1;
          }
        } else if (AcStrategyType::IsIdentity(ac_strategy_row[bx])) {
          row_out[bx] = kIdentityOrderContextStart + c;
        } else if (AcStrategyType::IsDct16x16(ac_strategy_row[bx])) {
          row_out[bx] = kDct16OrderContextStart + c;
        } else if (AcStrategyType::IsNone(ac_strategy_row[bx])) {
          if (bx != 0 && AcStrategyType::IsDct16x16(ac_strategy_row[bx - 1])) {
            row_out[bx] = kDct16OrderContextStart + 3 + c;
          } else if (y != 0 &&
                     AcStrategyType::IsDct16x16(ac_strategy_last_row[bx])) {
            row_out[bx] = kDct16OrderContextStart + 6 + c;
          } else {
            PIK_ASSERT(
                bx != 0 && y != 0 &&
                AcStrategyType::IsDct16x16(ac_strategy_last_row[bx - 1]));
            row_out[bx] = kDct16OrderContextStart + 9 + c;
          }
        }
      }
    }
  }
}

std::string EncodeAcStrategy(const ImageB& ac_strategy) {
  const size_t max_out_size = ac_strategy.xsize() * ac_strategy.ysize() + 1024;
  std::string output(max_out_size, 0);
  size_t storage_ix = 0;
  uint8_t* storage = reinterpret_cast<uint8_t*>(&output[0]);
  storage[0] = 0;
  std::vector<uint32_t> histogram(256);
  for (int y = 0; y < ac_strategy.ysize(); ++y) {
    for (int x = 0; x < ac_strategy.xsize(); ++x) {
      ++histogram[ac_strategy.Row(y)[x]];
    }
  }
  std::vector<uint8_t> bit_depths(256);
  std::vector<uint16_t> bit_codes(256);
  BuildAndStoreHuffmanTree(histogram.data(), histogram.size(),
                           bit_depths.data(), bit_codes.data(), &storage_ix,
                           storage);
  for (int y = 0; y < ac_strategy.ysize(); ++y) {
    const uint8_t* PIK_RESTRICT row = ac_strategy.Row(y);
    for (int x = 0; x < ac_strategy.xsize(); ++x) {
      WriteBits(bit_depths[row[x]], bit_codes[row[x]], &storage_ix, storage);
    }
  }
  WriteZeroesToByteBoundary(&storage_ix, storage);
  PIK_ASSERT((storage_ix >> 3) <= output.size());
  output.resize(storage_ix >> 3);
  return output;
}

std::string EncodeColorMap(const ImageI& ac_map, const int dc_val,
                           PikImageSizeInfo* info) {
  const size_t max_out_size = ac_map.xsize() * ac_map.ysize() + 1024;
  std::string output(max_out_size, 0);
  size_t storage_ix = 0;
  uint8_t* storage = reinterpret_cast<uint8_t*>(&output[0]);
  storage[0] = 0;
  std::vector<uint32_t> histogram(256);
  ++histogram[dc_val];
  for (int y = 0; y < ac_map.ysize(); ++y) {
    for (int x = 0; x < ac_map.xsize(); ++x) {
      ++histogram[ac_map.Row(y)[x]];
    }
  }
  std::vector<uint8_t> bit_depths(256);
  std::vector<uint16_t> bit_codes(256);
  BuildAndStoreHuffmanTree(histogram.data(), histogram.size(),
                           bit_depths.data(), bit_codes.data(), &storage_ix,
                           storage);
  const size_t histo_bits = storage_ix;
  WriteBits(bit_depths[dc_val], bit_codes[dc_val], &storage_ix, storage);
  for (int y = 0; y < ac_map.ysize(); ++y) {
    const int* PIK_RESTRICT row = ac_map.Row(y);
    for (int x = 0; x < ac_map.xsize(); ++x) {
      WriteBits(bit_depths[row[x]], bit_codes[row[x]], &storage_ix, storage);
    }
  }
  WriteZeroesToByteBoundary(&storage_ix, storage);
  PIK_ASSERT((storage_ix >> 3) <= output.size());
  output.resize(storage_ix >> 3);
  if (info) {
    info->histogram_size += histo_bits >> 3;
    info->entropy_coded_bits += storage_ix - histo_bits;
    info->total_size += output.size();
  }
  return output;
}

namespace {

template <uint32_t kDistribution>
class SizeCoderT {
 public:
  static size_t MaxSize(const size_t num_sizes) {
    const size_t bits = U32Coder::MaxEncodedBits(kDistribution) * num_sizes;
    return DivCeil(bits, kBitsPerByte) + 8;  // 8 extra bytes for WriteBits.
  }

  static void Encode(const size_t size, size_t* PIK_RESTRICT pos,
                     uint8_t* storage) {
    PIK_CHECK(U32Coder::Store(kDistribution, size, pos, storage));
  }

  static size_t Decode(BitReader* reader) {
    return U32Coder::Load(kDistribution, reader);
  }
};

// Quartiles of distribution observed from benchmark.
using DcGroupSizeCoder = SizeCoderT<0x110C0A09>;  // max observed: 8K
using AcGroupSizeCoder = SizeCoderT<0x150F0E0C>;  // max observed: 142K

template <typename T>
static inline void Append(const T& s, PaddedBytes* out,
                          size_t* PIK_RESTRICT byte_pos) {
  memcpy(out->data() + *byte_pos, s.data(), s.size());
  *byte_pos += s.size();
  PIK_CHECK(*byte_pos <= out->size());
}

}  // namespace

PaddedBytes EncodeToBitstream(const EncCache& cache, const Quantizer& quantizer,
                              const Image3F& gradient,
                              const NoiseParams& noise_params,
                              const ColorCorrelationMap& cmap, bool fast_mode,
                              PikInfo* info) {
  PROFILER_FUNC;
  constexpr size_t N = kBlockDim;
  constexpr size_t block_size = N * N;
  PIK_ASSERT(quantizer.block_dim() == N);
  const size_t xsize_blocks = cache.dc.xsize();
  const size_t ysize_blocks = cache.dc.ysize();
  const size_t xsize_groups = DivCeil(xsize_blocks, kGroupWidthInBlocks);
  const size_t ysize_groups = DivCeil(ysize_blocks, kGroupHeightInBlocks);
  const size_t num_groups = xsize_groups * ysize_groups;
  PikImageSizeInfo* cmap_info = info ? &info->layers[kLayerCmap] : nullptr;
  std::string cmap_code =
      EncodeColorMap(cmap.ytob_map, cmap.ytob_dc, cmap_info) +
      EncodeColorMap(cmap.ytox_map, cmap.ytox_dc, cmap_info);
  PikImageSizeInfo* quant_info = info ? &info->layers[kLayerQuant] : nullptr;
  PikImageSizeInfo* dc_info = info ? &info->layers[kLayerDC] : nullptr;
  PikImageSizeInfo* ac_info = info ? &info->layers[kLayerAC] : nullptr;
  std::string noise_code = EncodeNoise(noise_params);
  std::string quant_code = quantizer.Encode(quant_info);

  PaddedBytes serialized_gradient_map;
  if (gradient.xsize() != 0) {
    GradientMap gradient_map(xsize_blocks, ysize_blocks, cache.grayscale_opt);
    gradient_map.SetGradientMap(gradient);
    gradient_map.Serialize(quantizer, &serialized_gradient_map);
  }

  std::string dc_code = "";
  std::string dc_toc(DcGroupSizeCoder::MaxSize(num_groups), '\0');
  size_t dc_toc_pos = 0;
  uint8_t* dc_toc_storage =
      reinterpret_cast<uint8_t*>(const_cast<char*>(dc_toc.data()));

  // TODO(janwas): per-thread; also pass other tmp args to ShrinkDC
  Image3S tmp_dc_residuals(kGroupWidthInBlocks, kGroupHeightInBlocks);

  // TODO(janwas): per-group once ComputeCoeffOrder is incremental
  Image3B block_ctx(xsize_blocks, ysize_blocks);
  for (size_t y = 0; y < ysize_groups; y++) {
    for (size_t x = 0; x < xsize_groups; x++) {
      const Rect rect(x * kGroupWidthInBlocks, y * kGroupHeightInBlocks,
                      kGroupWidthInBlocks, kGroupHeightInBlocks, xsize_blocks,
                      ysize_blocks);
      const Rect tmp_rect(0, 0, rect.xsize(), rect.ysize());

      ShrinkDC(rect, cache.dc, &tmp_dc_residuals);
      ComputeBlockContextFromDC(rect, cache.dc, quantizer, rect,
                                cache.ac_strategy.layers, rect, &block_ctx);

      // (Need rect to indicate size because border groups may be smaller)
      const std::string& dc_group_code =
          EncodeImageData(tmp_rect, tmp_dc_residuals, dc_info);
      DcGroupSizeCoder::Encode(dc_group_code.size(), &dc_toc_pos,
                               dc_toc_storage);
      dc_code += dc_group_code;
    }
  }
  WriteZeroesToByteBoundary(&dc_toc_pos, dc_toc_storage);
  dc_toc.resize(dc_toc_pos / kBitsPerByte);

  std::string ac_strategy_code = "";
  std::string ac_code = "";
  std::string order_code = "";
  std::string histo_code = "";
  int32_t order[kOrderContexts * block_size];
  std::vector<ANSEncodingData> codes;
  std::vector<uint8_t> context_map;
  std::vector<std::vector<Token>> all_tokens;
  if (fast_mode) {
    ComputeCoeffOrderFast<N>(order);
  } else {
    ComputeCoeffOrder<N>(cache.ac, block_ctx, order);
  }

  if (cache.ac_strategy.use_ac_strategy) {
    // TODO(user): pass size info?
    ac_strategy_code = EncodeAcStrategy(cache.ac_strategy.layers);
  }

  order_code = EncodeCoeffOrders<N>(order, info);
  for (size_t y = 0; y < ysize_groups; y++) {
    for (size_t x = 0; x < xsize_groups; x++) {
      const Rect rect(x * kGroupWidthInBlocks, y * kGroupHeightInBlocks,
                      kGroupWidthInBlocks, kGroupHeightInBlocks, xsize_blocks,
                      ysize_blocks);
      all_tokens.emplace_back(std::vector<Token>());
      TokenizeQuantField(rect, cache.quant_field, cache.ac_strategy.layers,
                         &all_tokens.back());
      // WARNING: TokenizeCoefficients also uses the DC values in qcoeffs.ac!
      TokenizeCoefficients<N>(order, rect, cache.ac, block_ctx,
                              cache.ac_strategy.layers, &all_tokens.back());
    }
  }
  if (fast_mode) {
    histo_code = BuildAndEncodeHistogramsFast<N>(all_tokens, &codes,
                                                 &context_map, ac_info);
  } else {
    histo_code = BuildAndEncodeHistograms(kNumContexts, all_tokens, &codes,
                                          &context_map, ac_info);
  }

  std::string ac_toc(AcGroupSizeCoder::MaxSize(all_tokens.size()), '\0');
  size_t ac_toc_pos = 0;
  uint8_t* ac_toc_storage =
      reinterpret_cast<uint8_t*>(const_cast<char*>(ac_toc.data()));

  for (int i = 0; i < all_tokens.size(); ++i) {
    std::string ac_group_code =
        WriteTokens(all_tokens[i], codes, context_map, ac_info);
    AcGroupSizeCoder::Encode(ac_group_code.size(), &ac_toc_pos, ac_toc_storage);
    ac_code += ac_group_code;
  }
  WriteZeroesToByteBoundary(&ac_toc_pos, ac_toc_storage);
  ac_toc.resize(ac_toc_pos / kBitsPerByte);

  if (info) {
    info->layers[kLayerHeader].total_size +=
        noise_code.size() + dc_toc.size() + ac_toc.size();
  }

  PaddedBytes out(ac_strategy_code.size() + cmap_code.size() +
                  noise_code.size() + quant_code.size() +
                  serialized_gradient_map.size() + dc_toc.size() +
                  dc_code.size() + order_code.size() + histo_code.size() +
                  ac_toc.size() + ac_code.size());
  size_t byte_pos = 0;
  Append(ac_strategy_code, &out, &byte_pos);
  Append(cmap_code, &out, &byte_pos);
  Append(noise_code, &out, &byte_pos);
  Append(quant_code, &out, &byte_pos);
  Append(serialized_gradient_map, &out, &byte_pos);
  Append(dc_toc, &out, &byte_pos);
  Append(dc_code, &out, &byte_pos);
  Append(order_code, &out, &byte_pos);
  Append(histo_code, &out, &byte_pos);
  Append(ac_toc, &out, &byte_pos);
  Append(ac_code, &out, &byte_pos);
  return out;
}

bool DecodeColorMap(BitReader* PIK_RESTRICT br, ImageI* PIK_RESTRICT ac_map,
                    int* PIK_RESTRICT dc_val) {
  HuffmanDecodingData entropy;
  if (!entropy.ReadFromBitStream(br)) {
    return PIK_FAILURE("Invalid histogram data.");
  }
  HuffmanDecoder decoder;
  br->FillBitBuffer();
  *dc_val = decoder.ReadSymbol(entropy, br);
  for (size_t y = 0; y < ac_map->ysize(); ++y) {
    int* PIK_RESTRICT row = ac_map->Row(y);
    for (size_t x = 0; x < ac_map->xsize(); ++x) {
      br->FillBitBuffer();
      row[x] = decoder.ReadSymbol(entropy, br);
    }
  }
  br->JumpToByteBoundary();
  return true;
}

template <class Coder>
std::vector<uint64_t> OffsetsFromSizes(const size_t num_groups,
                                       BitReader* PIK_RESTRICT reader) {
  // = prefix sum of sizes.
  std::vector<uint64_t> offsets;
  offsets.reserve(1 + num_groups);
  offsets.push_back(0);
  for (size_t i = 0; i < num_groups; ++i) {
    const uint32_t size = Coder::Decode(reader);
    offsets.push_back(offsets.back() + size);
  }
  reader->JumpToByteBoundary();
  return offsets;
}

// Returns whether the entire group is accessible - we need bounds checking
// because offsets are derived from untrusted sizes.
bool IsSizeWithinBounds(const uint8_t* groups_begin, const uint8_t* data_end,
                        const uint64_t offset_begin, const uint64_t offset_end,
                        size_t* PIK_RESTRICT size) {
  if (groups_begin + offset_end > data_end) {
    return PIK_FAILURE("Group size exceeds [truncated?] stream length");
  }
  *size = offset_end - offset_begin;
  return true;
}

class Dequant {
 public:
  void Init(const ColorCorrelationMap& cmap, const Quantizer& quantizer) {
    // Precompute DC dequantization multipliers.
    for (int kind = 0; kind < kNumQuantKinds; kind++) {
      for (int c = 0; c < 3; ++c) {
        dequant_matrix_[kind * 3 + c] = quantizer.DequantMatrix(c, kind);
      }
    }

    for (int c = 0; c < 3; ++c) {
      mul_dc_[c] = dequant_matrix_[c][0] * quantizer.inv_quant_dc();
    }

    // Precompute DC inverse color transform.
    ytox_dc_ = ColorCorrelationMap::YtoX(1.0f, cmap.ytox_dc);
    ytob_dc_ = ColorCorrelationMap::YtoB(1.0f, cmap.ytob_dc);

    inv_global_scale_ = quantizer.InvGlobalScale();
  }

  // Dequantizes and inverse color-transforms one group's worth of DC, i.e. the
  // window "rect_dc" within the entire output image "cache->dc".
  SIMD_ATTR void DoDC(const Rect& rect_dc16, const Image3S& img_dc16,
                      const Rect& rect_dc, DecCache* PIK_RESTRICT cache) const {
    PIK_ASSERT(SameSize(rect_dc16, rect_dc));
    const size_t xsize = rect_dc16.xsize();
    const size_t ysize = rect_dc16.ysize();

    using D = SIMD_FULL(float);
    constexpr D d;
    constexpr SIMD_PART(int16_t, D::N) d16;
    constexpr SIMD_PART(int32_t, D::N) d32;

    const auto dequant_y = set1(d, mul_dc_[1]);

    for (size_t by = 0; by < ysize; ++by) {
      const int16_t* PIK_RESTRICT row_y16 =
          rect_dc16.ConstRow(img_dc16.Plane(1), by);
      float* PIK_RESTRICT row_y = rect_dc.Row(cache->dc.MutablePlane(1), by);

      for (size_t bx = 0; bx < xsize; bx += d.N) {
        const auto quantized_y16 = load(d16, row_y16 + bx);
        const auto quantized_y = convert_to(d, convert_to(d32, quantized_y16));
        store(quantized_y * dequant_y, d, row_y + bx);
      }
    }

    for (int c = 0; c < 3; c += 2) {  // === for c in {0, 2}
      const auto y_mul = set1(d, (c == 0) ? ytox_dc_ : ytob_dc_);
      const auto xb_mul = set1(d, mul_dc_[c]);
      for (size_t by = 0; by < ysize; ++by) {
        const int16_t* PIK_RESTRICT row_xb16 =
            rect_dc16.ConstRow(img_dc16.Plane(c), by);
        const float* PIK_RESTRICT row_y =
            rect_dc.ConstRow(cache->dc.Plane(1), by);
        float* PIK_RESTRICT row_xb = rect_dc.Row(cache->dc.MutablePlane(c), by);

        for (size_t bx = 0; bx < xsize; bx += d.N) {
          const auto quantized_xb16 = load(d16, row_xb16 + bx);
          const auto quantized_xb =
              convert_to(d, convert_to(d32, quantized_xb16));
          const auto out_y = load(d, row_y + bx);

          const auto out_xb = mul_add(y_mul, out_y, quantized_xb * xb_mul);
          store(out_xb, d, row_xb + bx);
        }
      }
    }
  }

  // Dequantizes and inverse color-transforms one group, i.e. the window "rect"
  // (in block units) within the entire output image "cache->ac".
  SIMD_ATTR void DoAC(const Rect& rect_ac16, const Image3S& img_ac16,
                      const Rect& rect, const ImageI& img_quant_field,
                      const ImageI& img_ytox, const ImageI& img_ytob,
                      DecCache* PIK_RESTRICT cache) const {
    constexpr size_t N = kBlockDim;
    constexpr size_t block_size = N * N;
    const size_t xsize = rect_ac16.xsize();  // [blocks]
    const size_t ysize = rect_ac16.ysize();
    PIK_ASSERT(img_ac16.xsize() % block_size == 0);
    PIK_ASSERT(xsize <= img_ac16.xsize() / block_size);
    PIK_ASSERT(ysize <= img_ac16.ysize());
    PIK_ASSERT(SameSize(rect_ac16, rect));
    PIK_ASSERT(SameSize(img_ytox, img_ytob));

    using D = SIMD_FULL(float);
    constexpr D d;
    constexpr SIMD_PART(int16_t, D::N) d16;
    constexpr SIMD_PART(int32_t, D::N) d32;

    const size_t x0_cmap = rect.x0() / kColorTileDimInBlocks;
    const size_t y0_cmap = rect.y0() / kColorTileDimInBlocks;
    const size_t x0_dct = rect.x0() * block_size;
    const size_t x0_dct16 = rect_ac16.x0() * block_size;

    for (size_t by = 0; by < ysize; ++by) {
      const int16_t* PIK_RESTRICT row_y16 =
          img_ac16.PlaneRow(1, rect_ac16.y0() + by) + x0_dct16;
      const int* PIK_RESTRICT row_quant_field =
          rect.ConstRow(img_quant_field, by);
      float* PIK_RESTRICT row_y =
          cache->ac.PlaneRow(1, rect.y0() + by) + x0_dct;

      for (size_t bx = 0; bx < xsize; ++bx) {
        size_t kind = GetQuantKindFromAcStrategy(
            cache->ac_strategy.layers, rect.x0() + bx, rect.y0() + by);
        for (size_t k = 0; k < block_size; k += d.N) {
          const size_t x = bx * block_size + k;

          // Y-channel quantization matrix for the given kind.
          const auto dequant = load(d, dequant_matrix_[kind * 3 + 1] + k);
          const auto y_mul = dequant * set1(d, SafeDiv(inv_global_scale_,
                                                       row_quant_field[bx]));

          const auto quantized_y16 = load(d16, row_y16 + x);
          const auto quantized_y =
              convert_to(d, convert_to(d32, quantized_y16));
          store(quantized_y * y_mul, d, row_y + x);
        }
      }
    }

    for (int c = 0; c < 3; c += 2) {  // === for c in {0, 2}
      const ImageI& img_cmap = (c == 0) ? img_ytox : img_ytob;
      for (size_t by = 0; by < ysize; ++by) {
        const size_t ty = by / kColorTileDimInBlocks;
        const int16_t* PIK_RESTRICT row_xb16 =
            img_ac16.PlaneRow(c, rect_ac16.y0() + by) + x0_dct16;
        const int* PIK_RESTRICT row_quant_field =
            rect.ConstRow(img_quant_field, by);
        const int* PIK_RESTRICT row_cmap =
            img_cmap.ConstRow(y0_cmap + ty) + x0_cmap;
        const float* PIK_RESTRICT row_y =
            cache->ac.ConstPlaneRow(1, rect.y0() + by) + x0_dct;
        float* PIK_RESTRICT row_xb =
            cache->ac.PlaneRow(c, rect.y0() + by) + x0_dct;

        for (size_t bx = 0; bx < xsize; ++bx) {
          size_t kind = GetQuantKindFromAcStrategy(
              cache->ac_strategy.layers, rect.x0() + bx, rect.y0() + by);
          const float* dequant_matrix = dequant_matrix_[kind * 3 + c];
          const size_t tx = bx / kColorTileDimInBlocks;
          const int32_t cmap = row_cmap[tx];
          const auto y_mul =
              (c == 0) ? set1(d, ColorCorrelationMap::YtoX(1.0f, cmap))
                       : set1(d, ColorCorrelationMap::YtoB(1.0f, cmap));

          for (size_t k = 0; k < block_size; k += d.N) {
            const size_t x = bx * block_size + k;

            const auto dequant = load(d, dequant_matrix + k);
            const auto xb_mul = dequant * set1(d, SafeDiv(inv_global_scale_,
                                                          row_quant_field[bx]));

            const auto quantized_xb16 = load(d16, row_xb16 + x);
            const auto quantized_xb =
                convert_to(d, convert_to(d32, quantized_xb16));

            const auto out_y = load(d, row_y + x);

            store(mul_add(y_mul, out_y, quantized_xb * xb_mul), d, row_xb + x);
          }
        }
      }
    }
  }

 private:
  static PIK_INLINE float SafeDiv(float num, int32_t div) {
    return div == 0 ? 1E10f : num / div;
  }

  // Precomputed DC dequant/color transform
  float mul_dc_[3];
  float ytox_dc_;
  float ytob_dc_;

  // AC dequant
  const float* PIK_RESTRICT
      dequant_matrix_[3 * kNumQuantKinds];  // [X, Y, B] dequant matrices for
                                            // every quant kind.
  float inv_global_scale_;
};

// Temporary storage; one per thread, for one group.
struct DecoderBuffers {
  template <int N>
  void InitOnce(const bool eager_dequant) {
    constexpr size_t block_size = N * N;
    // This thread already allocated its buffers.
    if (num_nzeroes.xsize() != 0) return;

    // Allocate enough for a whole group - partial groups on the right/bottom
    // border just use a subset. The valid size is passed via Rect.
    const size_t xsize_blocks = kGroupWidthInBlocks;
    const size_t ysize_blocks = kGroupHeightInBlocks;

    block_ctx = Image3B(xsize_blocks, ysize_blocks);

    if (eager_dequant) {
      quantized_dc = Image3S(xsize_blocks, ysize_blocks);
      quantized_ac = Image3S(xsize_blocks * block_size, ysize_blocks);
    }  // else: Decode uses DecCache->quantized_dc/ac.

    dc_y = ImageS(xsize_blocks, ysize_blocks);
    dc_xz_residuals = ImageS(xsize_blocks * 2, ysize_blocks);
    dc_xz_expanded = ImageS(xsize_blocks * 2, ysize_blocks);

    num_nzeroes = Image3I(xsize_blocks, ysize_blocks);
  }

  Image3B block_ctx;

  // Decode (only if eager_dequant)
  Image3S quantized_dc;
  Image3S quantized_ac;

  // ExpandDC
  ImageS dc_y;
  ImageS dc_xz_residuals;
  ImageS dc_xz_expanded;

  // DequantAC
  Image3I num_nzeroes;
};

bool DecodeCoefficientsAndDequantize(
    const size_t xsize_blocks, const size_t ysize_blocks,
    const PaddedBytes& compressed, BitReader* reader, ColorCorrelationMap* cmap,
    ThreadPool* pool, DecCache* cache, Quantizer* quantizer) {
  PROFILER_FUNC;
  constexpr size_t N = kBlockDim;
  constexpr size_t block_size = N * N;

  const size_t xsize_groups = DivCeil(xsize_blocks, kGroupWidthInBlocks);
  const size_t ysize_groups = DivCeil(ysize_blocks, kGroupHeightInBlocks);
  const size_t num_groups = xsize_groups * ysize_groups;

  const uint8_t* const data_end = compressed.data() + compressed.size();

  const std::vector<uint64_t>& dc_group_offsets =
      OffsetsFromSizes<DcGroupSizeCoder>(num_groups, reader);
  const uint8_t* dc_groups_begin = compressed.data() + reader->Position();
  // Skip past what the independent BitReaders will consume.
  reader->SkipBits(dc_group_offsets[num_groups] * kBitsPerByte);

  int coeff_order[kOrderContexts * block_size];
  for (size_t c = 0; c < kOrderContexts; ++c) {
    DecodeCoeffOrder<N>(&coeff_order[c * block_size],
                        /*decode_first=*/c >= kIdentityOrderContextStart,
                        reader);
  }
  reader->JumpToByteBoundary();

  ANSCode code;
  std::vector<uint8_t> context_map;
  // Histogram data size is small and does not require parallelization.
  PIK_RETURN_IF_ERROR(
      DecodeHistograms(reader, kNumContexts, 256, &code, &context_map));
  reader->JumpToByteBoundary();

  const std::vector<uint64_t>& ac_group_offsets =
      OffsetsFromSizes<AcGroupSizeCoder>(num_groups, reader);

  const uint8_t* ac_groups_begin = compressed.data() + reader->Position();
  // Skip past what the independent BitReaders will consume.
  reader->SkipBits(ac_group_offsets[num_groups] * kBitsPerByte);

  // Will be moved into quantizer.
  ImageI ac_quant_field(xsize_blocks, ysize_blocks);

  Dequant dequant;
  if (cache->eager_dequant) {
    dequant.Init(*cmap, *quantizer);
    cache->dc = Image3F(xsize_blocks, ysize_blocks);
    cache->ac = Image3F(xsize_blocks * block_size, ysize_blocks);
  } else {
    cache->quantized_dc = Image3S(xsize_blocks, ysize_blocks);
    cache->quantized_ac = Image3S(xsize_blocks * block_size, ysize_blocks);
  }

  std::vector<DecoderBuffers> decoder_buf(
      std::max<size_t>(1, pool->NumThreads()));

  // For each group: independent/parallel decode
  std::atomic<int> num_errors{0};
  const auto process_group = [&](const int task, const int thread) {
    const size_t group_x = task % xsize_groups;
    const size_t group_y = task / xsize_groups;
    const Rect rect(group_x * kGroupWidthInBlocks,
                    group_y * kGroupHeightInBlocks, kGroupWidthInBlocks,
                    kGroupHeightInBlocks, xsize_blocks, ysize_blocks);
    const Rect tmp_rect(0, 0, rect.xsize(), rect.ysize());
    DecoderBuffers& tmp = decoder_buf[thread];
    tmp.InitOnce<N>(cache->eager_dequant);

    size_t dc_size;
    if (!IsSizeWithinBounds(dc_groups_begin, data_end, dc_group_offsets[task],
                            dc_group_offsets[task + 1], &dc_size)) {
      num_errors.fetch_add(1);
      return;
    }
    BitReader dc_reader(dc_groups_begin + dc_group_offsets[task], dc_size);

    Image3S* quantized_dc =
        cache->eager_dequant ? &tmp.quantized_dc : &cache->quantized_dc;
    const Rect& rect16 = cache->eager_dequant ? tmp_rect : rect;

    if (!DecodeImage(&dc_reader, rect16, quantized_dc)) {
      num_errors.fetch_add(1);
      return;
    }

    ExpandDC(rect16, quantized_dc, &tmp.dc_y, &tmp.dc_xz_residuals,
             &tmp.dc_xz_expanded);

    if (cache->eager_dequant) {
      dequant.DoDC(rect16, *quantized_dc, rect, cache);
    }

    ComputeBlockContextFromDC(rect16, *quantized_dc, *quantizer, rect,
                              cache->ac_strategy.layers, tmp_rect,
                              &tmp.block_ctx);

    size_t ac_size;
    if (!IsSizeWithinBounds(ac_groups_begin, data_end, ac_group_offsets[task],
                            ac_group_offsets[task + 1], &ac_size)) {
      num_errors.fetch_add(1);
      return;
    }
    BitReader ac_reader(ac_groups_begin + ac_group_offsets[task], ac_size);
    Image3S* quantized_ac =
        cache->eager_dequant ? &tmp.quantized_ac : &cache->quantized_ac;
    bool ok =
        DecodeAC<N>(tmp.block_ctx, code, context_map, coeff_order, &ac_reader,
                    rect16, quantized_ac, rect, &ac_quant_field,
                    cache->ac_strategy.layers, &tmp.num_nzeroes);
    if (!ok) {
      num_errors.fetch_add(1);
    }

    if (cache->eager_dequant) {
      dequant.DoAC(rect16, *quantized_ac, rect, ac_quant_field, cmap->ytox_map,
                   cmap->ytob_map, cache);
    }
  };
  pool->Run(0, num_groups, process_group, "DecodeAndDequantize");

  quantizer->SetRawQuantField(std::move(ac_quant_field));
  return num_errors.load(std::memory_order_relaxed) == 0;
}

bool DecodeAcStrategy(BitReader* reader, const Header& header,
                      const size_t xsize_blocks, const size_t ysize_blocks,
                      AcStrategy* ac_strategy) {
  ac_strategy->layers = ImageB(xsize_blocks, ysize_blocks);
  if (header.flags & Header::kUseAcStrategy) {
    HuffmanDecodingData entropy;
    if (!entropy.ReadFromBitStream(reader)) {
      return PIK_FAILURE("Invalid histogram data.");
    }
    HuffmanDecoder decoder;
    reader->FillBitBuffer();
    for (size_t y = 0; y < ac_strategy->layers.ysize(); ++y) {
      uint8_t* PIK_RESTRICT row = ac_strategy->layers.Row(y);
      for (size_t x = 0; x < ac_strategy->layers.xsize(); ++x) {
        reader->FillBitBuffer();
        row[x] = decoder.ReadSymbol(entropy, reader);
      }
    }
    reader->JumpToByteBoundary();
  } else {
    ac_strategy->use_ac_strategy = false;
    FillImage((uint8_t)AcStrategyType::DCT, &ac_strategy->layers);
  }
  return true;
}

bool DecodeFromBitstream(const Header& header, const PaddedBytes& compressed,
                         BitReader* reader, const size_t xsize_blocks,
                         const size_t ysize_blocks, ThreadPool* pool,
                         ColorCorrelationMap* cmap, NoiseParams* noise_params,
                         Quantizer* quantizer, DecCache* cache) {
  PIK_RETURN_IF_ERROR(DecodeAcStrategy(reader, header, xsize_blocks,
                                       ysize_blocks, &cache->ac_strategy));

  DecodeColorMap(reader, &cmap->ytob_map, &cmap->ytob_dc);
  DecodeColorMap(reader, &cmap->ytox_map, &cmap->ytox_dc);

  PIK_RETURN_IF_ERROR(DecodeNoise(reader, noise_params));
  PIK_RETURN_IF_ERROR(quantizer->Decode(reader));

  if (header.flags & Header::kGradientMap) {
    GradientMap gradient_map(xsize_blocks, ysize_blocks,
                             header.flags & Header::kGrayscaleOpt);
    size_t byte_pos = reader->Position();
    PIK_RETURN_IF_ERROR(
        gradient_map.Deserialize(*quantizer, compressed, &byte_pos));
    reader->SkipBits((byte_pos - reader->Position()) * 8);
    cache->gradient = CopyImage(gradient_map.GetGradientMap());
  }

  return DecodeCoefficientsAndDequantize(xsize_blocks, ysize_blocks, compressed,
                                         reader, cmap, pool, cache, quantizer);
}

// Inverse-transform single block.
SIMD_ATTR void InverseIntegralBlockTransform(size_t bx, size_t by,
                                             const ImageF& coeffs,
                                             const ImageB& block_type_map,
                                             ImageF* img) {
  // Fixed 8x8 block size. Might be different from kBlockDim.
  constexpr size_t N = kBlockDim;
  constexpr size_t block_size = N * N;
  const float* PIK_RESTRICT row_in = coeffs.ConstRow(by) + bx * block_size;
  const size_t stride = img->PixelsPerRow();
  float* PIK_RESTRICT row_out = img->Row(by * N) + bx * N;
  uint8_t block_type = block_type_map.ConstRow(by)[bx];
  if (AcStrategyType::IsDct(block_type)) {
    ComputeTransposedScaledIDCT<N>(FromBlock<N>(row_in),
                                   ToLines(row_out, stride));
  } else if (AcStrategyType::IsIdentity(block_type)) {
    // Invert the trasform applied in ComputeCoefficients
    row_out[0] = row_in[0];
    float running_avg[N] = {row_out[0]};
    for (size_t ix = 1; ix < N; ++ix) {
      row_out[ix] = row_in[ix] + running_avg[ix - 1];
      running_avg[ix] = kIdentityAvgParam * running_avg[ix - 1] +
                        (1.0f - kIdentityAvgParam) * row_out[ix];
    }
    for (size_t iy = 1; iy < N; ++iy) {
      float* PIK_RESTRICT pos_out = row_out + iy * stride;
      const float* PIK_RESTRICT pos_in = row_in + iy * N;
      pos_out[0] = pos_in[0] + running_avg[0];
      running_avg[0] = kIdentityAvgParam * running_avg[0] +
                       (1.0f - kIdentityAvgParam) * pos_out[0];
      for (size_t ix = 1; ix < N; ++ix) {
        float predicted = (running_avg[ix - 1] + running_avg[ix]) * 0.5f;
        pos_out[ix] = pos_in[ix] + predicted;
        running_avg[ix] = running_avg[ix] * kIdentityAvgParam +
                          (1.0f - kIdentityAvgParam) *
                              (kIdentityAvgParam * running_avg[ix - 1] +
                               (1.0f - kIdentityAvgParam) * pos_out[ix]);
      }
    }
  } else if (AcStrategyType::IsDct16x16(block_type)) {
    SIMD_ALIGN float input[4 * N * N] = {};
    for (size_t k = 0; k < 4 * N * N; k++) {
      size_t zigzag_pos = kNaturalCoeffOrderLut16[k];
      size_t dest_block = zigzag_pos / (N * N);
      size_t block_pos = zigzag_pos - dest_block * N * N;
      size_t block_pos_unzigzag = kNaturalCoeffOrder8[block_pos];
      size_t dest_block_y = by + dest_block / 2;
      size_t dest_block_x = bx + dest_block % 2;
      input[k] = coeffs.ConstRow(
                     dest_block_y)[dest_block_x * N * N + block_pos_unzigzag] *
                 kDct16ScaleInv;
    }
    ComputeTransposedScaledIDCT<2 * N>(FromBlock<2 * N>(input),
                                       ToLines(row_out, stride));
  }
}

// Adds prepared deltas from "add_spatial" to "out".
static SIMD_ATTR inline void InverseSpatialBlockTransform(
    size_t bx, size_t by, const ImageF& spatial, const ImageB& block_type_map,
    ImageF* img) {
  constexpr size_t N = kBlockDim;
  PIK_CHECK(SameSize(spatial, *img));
  const size_t stride = spatial.PixelsPerRow();
  const float* PIK_RESTRICT row_add = spatial.ConstRow(by * N) + bx * N;
  float* PIK_RESTRICT row_out = img->Row(by * N) + bx * N;
  const SIMD_FULL(float) d;
  uint8_t block_type = block_type_map.ConstRow(by)[bx];
#ifdef ADDRESS_SANITIZER
  PIK_ASSERT(AcStrategyType::IsDct(block_type));
#endif
  // TODO(user): support other AC strategies.
  if (PIK_LIKELY(AcStrategyType::IsDct(block_type))) {
    for (size_t iy = 0; iy < N; ++iy) {
      const float* PIK_RESTRICT pos_add = row_add + stride * iy;
      float* PIK_RESTRICT pos_out = row_out + stride * iy;
      for (size_t ix = 0; ix < N; ix += d.N) {
        const auto pixels = load(d, pos_out + ix);
        const auto add = load(d, pos_add + ix);
        store(pixels + add, d, pos_out + ix);
      }
    }
  }
}

void AddPredictions_Smooth(const Image3F& dc, const ImageB& block_type_map,
                           bool use_ac_strategy, ThreadPool* pool,
                           Image3F* PIK_RESTRICT dcoeffs,
                           Image3F* PIK_RESTRICT idct) {
  PROFILER_FUNC;
  constexpr size_t N = kBlockDim;
  constexpr size_t block_size = N * N;
  constexpr size_t tile_dim_blocks = kTileDim / N;
  PIK_ASSERT(dcoeffs->xsize() % block_size == 0);
  const size_t xsize_blocks = dcoeffs->xsize() / block_size;
  const size_t ysize_blocks = dcoeffs->ysize();
  *idct = Image3F(xsize_blocks * N, ysize_blocks * N);

  const auto inverse_integral_transform = [&idct, &dcoeffs, &block_type_map](
                                              int c, size_t bx, size_t by) {
    dcoeffs->PlaneRow(c, by)[bx * block_size] = 0.0f;
    InverseIntegralBlockTransform(bx, by, dcoeffs->Plane(c), block_type_map,
                                  idct->MutablePlane(c));
  };
  TransformBlocks(pool, xsize_blocks, ysize_blocks, inverse_integral_transform);

  // TODO(janwas): skip if none will be upsampled (global flag)
  for (size_t by = 0; by < ysize_blocks; by += tile_dim_blocks) {
    for (size_t bx = 0; bx < xsize_blocks; bx += tile_dim_blocks) {
      const uint8_t strategy = block_type_map.ConstRow(by)[bx];
      const Rect rect(bx * N, by * N, kTileDim, kTileDim, xsize_blocks * N,
                      ysize_blocks * N);
      for (int c = 0; c < 3; ++c) {
        UpsampleTile(strategy, rect, idct->MutablePlane(c));
      }
    }
  }

  const Image3F upsampled_dc = BlurUpsampleDC<N>(dc, pool);

  TransformBlocks(
      pool, xsize_blocks, ysize_blocks,
      [&idct, &upsampled_dc, &block_type_map](int c, size_t bx, size_t by) {
        InverseSpatialBlockTransform(bx, by, upsampled_dc.Plane(c),
                                     block_type_map, idct->MutablePlane(c));
      });
}

void AddPredictions(const Image3F& dc, const ImageB& block_type_map,
                    bool use_ac_strategy, ThreadPool* pool,
                    Image3F* PIK_RESTRICT dcoeffs, Image3F* PIK_RESTRICT idct) {
  PROFILER_FUNC;
  constexpr size_t N = kBlockDim;
  constexpr size_t block_size = N * N;
  PIK_ASSERT(dcoeffs->xsize() % block_size == 0);
  const size_t xsize_blocks = dcoeffs->xsize() / block_size;
  const size_t ysize_blocks = dcoeffs->ysize();
  *idct = Image3F(xsize_blocks * N, ysize_blocks * N);

  // TODO(user): make sure, that "tile resampling" is off.

  const auto add_dc = [&](size_t c, size_t bx, size_t by) {
    if (AcStrategyType::IsIdentity(block_type_map.ConstRow(by)[bx])) {
      dcoeffs->PlaneRow(c, by)[bx * kBlockDim * kBlockDim] +=
          dc.ConstPlaneRow(c, by)[bx];
    }
  };
  if (use_ac_strategy)
    TransformBlocks(pool, xsize_blocks, ysize_blocks, add_dc);
  // Sets dcoeffs.0 from DC (for DCT and DCT16 blocks) and updates HVD.
  const Image3F pred2x2 =
      PredictSpatial2x2_AC64(dc, block_type_map, pool, dcoeffs);
  // Updates dcoeffs _except_ 0HVD.
  UpSample4x4BlurDCT(pred2x2, 1.5f, 0.0f, pool, dcoeffs);

  TransformBlocks(
      pool, xsize_blocks, ysize_blocks,
      [&idct, &dcoeffs, &block_type_map](int c, size_t bx, size_t by) {
        InverseIntegralBlockTransform(bx, by, dcoeffs->Plane(c), block_type_map,
                                      idct->MutablePlane(c));
      });
}

ImageF IntensityAcEstimate(const ImageF& image, float multiplier,
                           ThreadPool* pool) {
  constexpr size_t N = kBlockDim;
  std::vector<float> blur = DCfiedGaussianKernel<N>(5.5);
  ImageF retval = Convolve(image, blur);
  for (size_t y = 0; y < retval.ysize(); y++) {
    float* PIK_RESTRICT retval_row = retval.Row(y);
    const float* PIK_RESTRICT image_row = image.ConstRow(y);
    for (size_t x = 0; x < retval.xsize(); ++x) {
      retval_row[x] = multiplier * (image_row[x] - retval_row[x]);
    }
  }
  return retval;
}

void DequantImage(const Quantizer& quantizer, const ColorCorrelationMap& cmap,
                  ThreadPool* pool, DecCache* cache) {
  PROFILER_ZONE("dequant");
  constexpr size_t N = kBlockDim;
  constexpr size_t block_size = N * N;

  // Just-in-time dequantization should be performed only in encoder loop.
  PIK_ASSERT(!cache->eager_dequant);

  // Caller must have allocated/filled quantized_dc/ac.
  PIK_CHECK(SameSize(cache->quantized_dc, quantizer.RawQuantField()));

  const size_t xsize_blocks = quantizer.RawQuantField().xsize();
  const size_t ysize_blocks = quantizer.RawQuantField().ysize();
  const size_t xsize_groups = DivCeil(xsize_blocks, kGroupWidthInBlocks);
  const size_t ysize_groups = DivCeil(ysize_blocks, kGroupHeightInBlocks);

  Dequant dequant;
  dequant.Init(cmap, quantizer);
  cache->dc = Image3F(xsize_blocks, ysize_blocks);
  cache->ac = Image3F(xsize_blocks * block_size, ysize_blocks);

  std::vector<DecoderBuffers> decoder_buf(
      std::max<size_t>(1, pool->NumThreads()));

  const size_t num_groups = xsize_groups * ysize_groups;
  const auto dequant_group = [&](const int task, const int thread) {
    DecoderBuffers& tmp = decoder_buf[thread];
    tmp.InitOnce<N>(cache->eager_dequant);

    const size_t group_x = task % xsize_groups;
    const size_t group_y = task / xsize_groups;
    const Rect rect(group_x * kGroupWidthInBlocks,
                    group_y * kGroupHeightInBlocks, kGroupWidthInBlocks,
                    kGroupHeightInBlocks, xsize_blocks, ysize_blocks);

    dequant.DoDC(rect, cache->quantized_dc, rect, cache);
    dequant.DoAC(rect, cache->quantized_ac, rect, quantizer.RawQuantField(),
                 cmap.ytox_map, cmap.ytob_map, cache);
  };
  pool->Run(0, num_groups, dequant_group, "DequantImage");
}

Image3F ReconOpsinImage(const Header& header, const Quantizer& quantizer,
                        ThreadPool* pool, DecCache* cache, PikInfo* pik_info) {
  PROFILER_ZONE("recon");
  constexpr size_t N = kBlockDim;
  const size_t xsize_blocks = cache->dc.xsize();
  const size_t ysize_blocks = cache->dc.ysize();

  if (header.flags & Header::kGradientMap) {
    GradientMap map(xsize_blocks, ysize_blocks,
                    header.flags & Header::kGrayscaleOpt);
    map.SetGradientMap(cache->gradient);
    map.Apply(quantizer, &cache->dc, pool);
  }

  if (header.flags & Header::kGrayscaleOpt) {
    kGrayXyb->RestoreXB(&cache->dc);
  }

  Image3F idct(xsize_blocks * N, ysize_blocks * N);

  // AddPredictions* do not use the (invalid) DC component of cache->ac.
  // Does IDCT already internally to save work.
  if (header.flags & Header::kSmoothDCPred) {
    AddPredictions_Smooth(cache->dc, cache->ac_strategy.layers,
                          cache->ac_strategy.use_ac_strategy, pool, &cache->ac,
                          &idct);
  } else {
    AddPredictions(cache->dc, cache->ac_strategy.layers,
                   cache->ac_strategy.use_ac_strategy, pool, &cache->ac, &idct);
  }

  if (header.flags & Header::kGaborishTransformMask) {
    double strength = (header.flags & Header::kGaborishTransformMask) * 0.25;
    idct = ConvolveGaborish(std::move(idct), strength, pool);
  }

  return idct;
}

}  // namespace pik
