// Copyright 2018 Google Inc. All Rights Reserved.
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

#ifndef EXTERNAL_IMAGE_H_
#define EXTERNAL_IMAGE_H_

// Interleaved image for color transforms and Codec.

#include <stddef.h>
#include <stdint.h>

#include "codec.h"
#include "status.h"

namespace pik {

// Packed (no row padding), interleaved (RGBRGB) u8/u16/f32.
class ExternalImage {
 public:
  // Copies from existing interleaved image. Called by decoders. "big_endian"
  // only matters for bits_per_sample > 8. "end" is the STL-style end of "bytes"
  // for range checks, or null if unknown.
  ExternalImage(size_t xsize, size_t ysize, const ColorEncoding& c_current,
                bool has_alpha, size_t bits_per_sample, bool big_endian,
                const uint8_t* bytes, const uint8_t* end);

  // Converts pixels from c_current to c_desired. Called by encoders and
  // CodecInOut::CopyTo. alpha is nullptr iff !has_alpha.
  // If temp_intervals != null, fills them such that CopyTo can rescale to that
  // range. Otherwise, clamps temp to [0, 1].
  ExternalImage(CodecContext* codec_context, const Image3F& color,
                const ColorEncoding& c_current, const ColorEncoding& c_desired,
                bool has_alpha, const ImageU* alpha, size_t bits_per_sample,
                bool big_endian, CodecIntervals* temp_intervals);

  // Indicates whether the ctor succeeded; if not, do not use this instance.
  Status IsHealthy() const { return is_healthy_; }

  // Sets "io" to a newly allocated copy with c_current color space.
  // Uses temp_intervals for rescaling if not null.
  Status CopyTo(const CodecIntervals* temp_intervals, CodecInOut* io) const;

  // Packed, interleaved pixels, for passing to encoders.
  const PaddedBytes& Bytes() const { return bytes_; }

  size_t xsize() const { return xsize_; }
  size_t ysize() const { return ysize_; }
  const ColorEncoding& c_current() const { return c_current_; }
  bool IsGray() const { return c_current_.IsGray(); }
  bool HasAlpha() const { return channels_ == 2 || channels_ == 4; }
  size_t BitsPerSample() const { return bits_per_sample_; }
  bool BigEndian() const { return big_endian_; }

  uint8_t* Row(size_t y) { return bytes_.data() + y * row_size_; }
  const uint8_t* ConstRow(size_t y) const {
    return bytes_.data() + y * row_size_;
  }

 private:
  ExternalImage(size_t xsize, size_t ysize, const ColorEncoding& c_current,
                bool has_alpha, size_t bits_per_sample, bool big_endian);

  size_t xsize_;
  size_t ysize_;
  ColorEncoding c_current_;
  size_t channels_;
  size_t bits_per_sample_;
  bool big_endian_;
  size_t row_size_;
  PaddedBytes bytes_;
  bool is_healthy_;
};

}  // namespace pik

#endif  // EXTERNAL_IMAGE_H_
