/*
 * Copyright 2016 Google Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "guetzli/output_image.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <cmath>

#include "gamma_correct.h"
#include "guetzli/color_transform.h"
#include "guetzli/dct_double.h"
#include "guetzli/idct.h"
#include "guetzli/quantize.h"

namespace pik {
namespace guetzli {

OutputImageComponent::OutputImageComponent(int w, int h)
    : width_(w), height_(h) {
  Reset(1, 1);
}

void OutputImageComponent::Reset(int factor_x, int factor_y) {
  factor_x_ = factor_x;
  factor_y_ = factor_y;
  width_in_blocks_ = (width_ + 8 * factor_x_ - 1) / (8 * factor_x_);
  height_in_blocks_ = (height_ + 8 * factor_y_ - 1) / (8 * factor_y_);
  num_blocks_ = width_in_blocks_ * height_in_blocks_;
  coeffs_ = std::vector<coeff_t>(num_blocks_ * kDCTBlockSize);
  pixels_ = std::vector<uint16_t>(width_ * height_, 128 << 4);
  for (int i = 0; i < kDCTBlockSize; ++i) quant_[i] = 1;
}

bool OutputImageComponent::IsAllZero() const {
  int numcoeffs = num_blocks_ * kDCTBlockSize;
  for (int i = 0; i < numcoeffs; ++i) {
    if (coeffs_[i] != 0) return false;
  }
  return true;
}

void OutputImageComponent::GetCoeffBlock(int block_x, int block_y,
                                         coeff_t block[kDCTBlockSize]) const {
  assert(block_x < width_in_blocks_);
  assert(block_y < height_in_blocks_);
  int offset = (block_y * width_in_blocks_ + block_x) * kDCTBlockSize;
  memcpy(block, &coeffs_[offset], kDCTBlockSize * sizeof(coeffs_[0]));
}

void OutputImageComponent::ToPixels(int xmin, int ymin, int xsize, int ysize,
                                    uint8_t* out, int stride) const {
  assert(xmin >= 0);
  assert(ymin >= 0);
  assert(xmin < width_);
  assert(ymin < height_);
  const int yend1 = ymin + ysize;
  const int yend0 = std::min(yend1, height_);
  int y = ymin;
  for (; y < yend0; ++y) {
    const int xend1 = xmin + xsize;
    const int xend0 = std::min(xend1, width_);
    int x = xmin;
    int px = y * width_ + xmin;
    for (; x < xend0; ++x, ++px, out += stride) {
      *out = static_cast<uint8_t>((pixels_[px] + 8 - (x & 1)) >> 4);
    }
    const int offset = -stride;
    for (; x < xend1; ++x) {
      *out = out[offset];
      out += stride;
    }
  }
  for (; y < yend1; ++y) {
    const int offset = -stride * xsize;
    for (int x = 0; x < xsize; ++x) {
      *out = out[offset];
      out += stride;
    }
  }
}

void OutputImageComponent::ToFloatPixels(float* out, int stride) const {
  assert(factor_x_ == 1);
  assert(factor_y_ == 1);
  for (int block_y = 0; block_y < height_in_blocks_; ++block_y) {
    for (int block_x = 0; block_x < width_in_blocks_; ++block_x) {
      coeff_t block[kDCTBlockSize];
      GetCoeffBlock(block_x, block_y, block);
      double blockd[kDCTBlockSize];
      for (int k = 0; k < kDCTBlockSize; ++k) {
        blockd[k] = block[k];
      }
      ComputeBlockIDCTDouble(blockd);
      for (int iy = 0; iy < 8; ++iy) {
        for (int ix = 0; ix < 8; ++ix) {
          int y = block_y * 8 + iy;
          int x = block_x * 8 + ix;
          if (y >= height_ || x >= width_) continue;
          out[(y * width_ + x) * stride] = blockd[8 * iy + ix] + 128.0;
        }
      }
    }
  }
}

void OutputImageComponent::SetCoeffBlock(int block_x, int block_y,
                                         const coeff_t block[kDCTBlockSize]) {
  assert(block_x < width_in_blocks_);
  assert(block_y < height_in_blocks_);
  int offset = (block_y * width_in_blocks_ + block_x) * kDCTBlockSize;
  memcpy(&coeffs_[offset], block, kDCTBlockSize * sizeof(coeffs_[0]));
  uint8_t idct[kDCTBlockSize];
  ComputeBlockIDCT(&coeffs_[offset], idct);
  UpdatePixelsForBlock(block_x, block_y, idct);
}

void OutputImageComponent::UpdatePixelsForBlock(
    int block_x, int block_y, const uint8_t idct[kDCTBlockSize]) {
  if (factor_x_ == 1 && factor_y_ == 1) {
    for (int iy = 0; iy < 8; ++iy) {
      for (int ix = 0; ix < 8; ++ix) {
        int x = 8 * block_x + ix;
        int y = 8 * block_y + iy;
        if (x >= width_ || y >= height_) continue;
        int p = y * width_ + x;
        pixels_[p] = idct[8 * iy + ix] << 4;
      }
    }
  } else if (factor_x_ == 2 && factor_y_ == 2) {
    // Fill in the 10x10 pixel area in the subsampled image that will be the
    // basis of the upsampling. This area is enough to hold the 3x3 kernel of
    // the fancy upsampler around each pixel.
    static const int kSubsampledEdgeSize = 10;
    uint16_t subsampled[kSubsampledEdgeSize * kSubsampledEdgeSize];
    for (int j = 0; j < kSubsampledEdgeSize; ++j) {
      // The order we fill in the rows is:
      //   8 rows intersecting the block, row below, row above
      const int y0 = block_y * 16 + (j < 9 ? j * 2 : -2);
      for (int i = 0; i < kSubsampledEdgeSize; ++i) {
        // The order we fill in each row is:
        //   8 pixels within the block, left edge, right edge
        const int ix =
            ((j < 9 ? (j + 1) * kSubsampledEdgeSize : 0) + (i < 9 ? i + 1 : 0));
        const int x0 = block_x * 16 + (i < 9 ? i * 2 : -2);
        if (x0 < 0) {
          subsampled[ix] = subsampled[ix + 1];
        } else if (y0 < 0) {
          subsampled[ix] = subsampled[ix + kSubsampledEdgeSize];
        } else if (x0 >= width_) {
          subsampled[ix] = subsampled[ix - 1];
        } else if (y0 >= height_) {
          subsampled[ix] = subsampled[ix - kSubsampledEdgeSize];
        } else if (i < 8 && j < 8) {
          subsampled[ix] = idct[j * 8 + i] << 4;
        } else {
          // Reconstruct the subsampled pixels around the edge of the current
          // block by computing the inverse of the fancy upsampler.
          const int y1 = std::max(y0 - 1, 0);
          const int x1 = std::max(x0 - 1, 0);
          subsampled[ix] =
              (pixels_[y0 * width_ + x0] * 9 + pixels_[y1 * width_ + x1] +
               pixels_[y0 * width_ + x1] * -3 +
               pixels_[y1 * width_ + x0] * -3) >>
              2;
        }
      }
    }

    // Determine area to update.
    int xmin = std::max(block_x * 16 - 1, 0);
    int xmax = std::min(block_x * 16 + 16, width_ - 1);
    int ymin = std::max(block_y * 16 - 1, 0);
    int ymax = std::min(block_y * 16 + 16, height_ - 1);

    // Apply the fancy upsampler on the subsampled block.
    for (int y = ymin; y <= ymax; ++y) {
      const int y0 = ((y & ~1) / 2 - block_y * 8 + 1) * kSubsampledEdgeSize;
      const int dy = ((y & 1) * 2 - 1) * kSubsampledEdgeSize;
      uint16_t* rowptr = &pixels_[y * width_];
      for (int x = xmin; x <= xmax; ++x) {
        const int x0 = (x & ~1) / 2 - block_x * 8 + 1;
        const int dx = (x & 1) * 2 - 1;
        const int ix = x0 + y0;
        rowptr[x] = (subsampled[ix] * 9 + subsampled[ix + dy] * 3 +
                     subsampled[ix + dx] * 3 + subsampled[ix + dx + dy]) >>
                    4;
      }
    }
  } else {
    printf("Sampling ratio not supported: factor_x = %d factor_y = %d\n",
           factor_x_, factor_y_);
    exit(1);
  }
}

void OutputImageComponent::CopyFromJpegComponent(const JPEGComponent& comp,
                                                 int factor_x, int factor_y,
                                                 const int* quant) {
  Reset(factor_x, factor_y);
  assert(width_in_blocks_ <= comp.width_in_blocks);
  assert(height_in_blocks_ <= comp.height_in_blocks);
  const size_t src_row_size = comp.width_in_blocks * kDCTBlockSize;
  for (int block_y = 0; block_y < height_in_blocks_; ++block_y) {
    const coeff_t* src_coeffs = &comp.coeffs[block_y * src_row_size];
    for (int block_x = 0; block_x < width_in_blocks_; ++block_x) {
      coeff_t block[kDCTBlockSize];
      for (int i = 0; i < kDCTBlockSize; ++i) {
        block[i] = src_coeffs[i] * quant[i];
      }
      SetCoeffBlock(block_x, block_y, block);
      src_coeffs += kDCTBlockSize;
    }
  }
  memcpy(quant_, quant, sizeof(quant_));
}

OutputImage::OutputImage(int w, int h)
    : width_(w), height_(h), components_(3, OutputImageComponent(w, h)) {}

void OutputImage::CopyFromJpegData(const JPEGData& jpg) {
  for (int i = 0; i < jpg.components.size(); ++i) {
    const JPEGComponent& comp = jpg.components[i];
    assert(jpg.max_h_samp_factor % comp.h_samp_factor == 0);
    assert(jpg.max_v_samp_factor % comp.v_samp_factor == 0);
    int factor_x = jpg.max_h_samp_factor / comp.h_samp_factor;
    int factor_y = jpg.max_v_samp_factor / comp.v_samp_factor;
    assert(comp.quant_idx < jpg.quant.size());
    components_[i].CopyFromJpegComponent(comp, factor_x, factor_y,
                                         &jpg.quant[comp.quant_idx].values[0]);
  }
}

std::vector<uint8_t> OutputImage::ToSRGB() const {
  std::vector<uint8_t> rgb(width_ * height_ * 3);
  for (int c = 0; c < 3; ++c) {
    components_[c].ToPixels(0, 0, width_, height_, &rgb[c], 3);
  }
  for (int p = 0; p < rgb.size(); p += 3) {
    ColorTransformYCbCrToRGB(&rgb[p]);
  }
  return rgb;
}

std::string OutputImage::FrameTypeStr() const {
  char buf[128];
  int len = snprintf(buf, sizeof(buf), "f%d%d%d%d%d%d", component(0).factor_x(),
                     component(0).factor_y(), component(1).factor_x(),
                     component(1).factor_y(), component(2).factor_x(),
                     component(2).factor_y());
  return std::string(buf, len);
}

}  // namespace guetzli
}  // namespace pik
