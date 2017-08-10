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

#ifndef COMPRESSED_IMAGE_H_
#define COMPRESSED_IMAGE_H_

#include <stddef.h>
#include <stdint.h>
#include <memory>
#include <string>
#include <utility>

#include "image.h"
#include "opsin_codec.h"
#include "pik_info.h"
#include "quantizer.h"

namespace pik {

static const int kYToBRes = 48;

// Represents both the quantized and transformed original version of an image.
// This class is used in both the encoder and decoder.
class CompressedImage {
 public:
  // Creates a compressed image from an opsin-dynamics image original.
  // The compressed image is in an undefined state until Quantize() is called.
  static CompressedImage FromOpsinImage(const Image3F& opsin, PikInfo* info);

  // Creates a compressed image from the bitstream.
  static CompressedImage Decode(int xsize, int ysize, const std::string& data,
                                PikInfo* info);

  int xsize() const { return xsize_; }
  int ysize() const { return ysize_; }
  int quant_tile_size() const;

  Quantizer& quantizer() { return quantizer_; }
  const Quantizer& quantizer() const { return quantizer_; }

  void QuantizeBlock(int block_x, int block_y);
  void Quantize();

  AdaptiveQuantParams adaptive_quant_params() const {
    AdaptiveQuantParams p;
    p.initial_quant_val_dc = 1.0625;
    p.initial_quant_val_ac = 0.5625;
    return p;
  }

  // Returns the SRGB image based on the quantization values and the quantized
  // coefficients.
  Image3B ToSRGB() const;

  const Image3W& coeffs() const { return dct_coeffs_; }

  // Returns a lossless encoding of the quantized coefficients.
  std::string Encode() const;
  std::string EncodeFast() const;

  // Getters and setters for adaptive Y-to-blue correlation.
  float YToBDC() const { return ytob_dc_ / 128.0; }
  float YToBAC(int tx, int ty) const { return ytob_ac_.Row(ty)[tx] / 128.0; }
  void SetYToBDC(int ytob) { ytob_dc_ = ytob; }
  void SetYToBAC(int tx, int ty, int val) { ytob_ac_.Row(ty)[tx] = val; }

 private:
  CompressedImage(int xsize, int ysize, PikInfo* info);

  void UpdateBlock(const int block_x, const int block_y,
                   float* const PIK_RESTRICT block) const;
  void UpdateSRGB(const float* const PIK_RESTRICT block,
                  int block_x, int block_y,
                  Image3B* const PIK_RESTRICT srgb) const;

  const int xsize_;
  const int ysize_;
  const int block_xsize_;
  const int block_ysize_;
  const int quant_xsize_;
  const int quant_ysize_;
  const int num_blocks_;
  Quantizer quantizer_;
  Image3W dct_coeffs_;
  // Transformed version of the original image, only present if the image
  // was constructed with FromOpsinImage().
  std::unique_ptr<Image3F> opsin_image_;
  int ytob_dc_;
  Image<int> ytob_ac_;
  // Not owned, used to report additional statistics to the callers of
  // PixelsToPik() and PikToPixels().
  PikInfo* pik_info_;
};

}  // namespace pik

#endif  // COMPRESSED_IMAGE_H_
