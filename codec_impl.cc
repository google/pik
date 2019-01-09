// Copyright 2018 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "codec.h"

#include <algorithm>

#undef PROFILER_ENABLED
#define PROFILER_ENABLED 1
#include "byte_order.h"
#include "codec_png.h"
#include "codec_pnm.h"
#include "common.h"
#include "epf.h"
#include "external_image.h"
#include "file_io.h"
#include "profiler.h"
#include "simd/targets.h"

namespace pik {
namespace {

// Any valid encoding is larger (ensures codecs can read the first few bytes)
constexpr size_t kMinBytes = 9;

// Returns RGB/Gray pair of ColorEncoding (indexed by IsGray()).
std::array<ColorEncoding, 2> MakeC2(const Primaries pr,
                                    const TransferFunction tf) {
  std::array<ColorEncoding, 2> c2;
  c2[0].color_space = ColorSpace::kRGB;
  c2[0].white_point = WhitePoint::kD65;
  c2[0].primaries = pr;
  c2[0].transfer_function = tf;
  PIK_CHECK(ColorManagement::SetProfileFromFields(&c2[0]));

  // Same as above, but gray.
  c2[1] = c2[0];
  c2[1].color_space = ColorSpace::kGray;
  PIK_CHECK(ColorManagement::SetProfileFromFields(&c2[1]));
  return c2;
}

template <typename T>
Status FromSRGB(const size_t xsize, const size_t ysize, const bool is_gray,
                const bool has_alpha, const bool big_endian, const T* pixels,
                const T* end, ThreadPool* pool, CodecInOut* io) {
  const ColorEncoding& c = io->Context()->c_srgb[is_gray];
  const size_t bits_per_sample = sizeof(T) * kBitsPerByte;
  const uint8_t* bytes = reinterpret_cast<const uint8_t*>(pixels);
  const uint8_t* bytes_end = reinterpret_cast<const uint8_t*>(end);
  const ExternalImage external(xsize, ysize, c, has_alpha, bits_per_sample,
                               big_endian, bytes, bytes_end);
  const CodecIntervals* temp_intervals = nullptr;  // Don't know min/max.
  return external.CopyTo(temp_intervals, pool, io);
}

// Copies interleaved external color; skips any alpha. Caller ensures
// bits_per_sample matches T, and byte order=native.
template <typename T>
void AllocateAndFill(const ExternalImage& external, Image3<T>* out) {
  PIK_ASSERT(external.IsHealthy());  // Callers must check beforehand.

  // Here we just copy bytes for simplicity; for conversion/byte swapping, use
  // ExternalImage::CopyTo instead.
  PIK_CHECK(external.BitsPerSample() == sizeof(T) * kBitsPerByte);
  PIK_CHECK(external.BigEndian() == !IsLittleEndian());

  const size_t xsize = external.xsize();
  const size_t ysize = external.ysize();
  *out = Image3<T>(xsize, ysize);
  if (external.IsGray()) {
    if (external.HasAlpha()) {
      for (size_t y = 0; y < ysize; ++y) {
        const T* PIK_RESTRICT row =
            reinterpret_cast<const T*>(external.ConstRow(y));
        T* PIK_RESTRICT row0 = out->PlaneRow(0, y);
        T* PIK_RESTRICT row1 = out->PlaneRow(1, y);
        T* PIK_RESTRICT row2 = out->PlaneRow(2, y);
        for (size_t x = 0; x < xsize; ++x) {
          row0[x] = row[2 * x + 0];
          row1[x] = row[2 * x + 0];
          row2[x] = row[2 * x + 0];
        }
      }
    } else {
      for (size_t y = 0; y < ysize; ++y) {
        const T* PIK_RESTRICT row =
            reinterpret_cast<const T*>(external.ConstRow(y));
        T* PIK_RESTRICT row0 = out->PlaneRow(0, y);
        T* PIK_RESTRICT row1 = out->PlaneRow(1, y);
        T* PIK_RESTRICT row2 = out->PlaneRow(2, y);
        for (size_t x = 0; x < xsize; ++x) {
          row0[x] = row[x];
          row1[x] = row[x];
          row2[x] = row[x];
        }
      }
    }
  } else {
    if (external.HasAlpha()) {
      for (size_t y = 0; y < ysize; ++y) {
        const T* PIK_RESTRICT row =
            reinterpret_cast<const T*>(external.ConstRow(y));
        T* PIK_RESTRICT row0 = out->PlaneRow(0, y);
        T* PIK_RESTRICT row1 = out->PlaneRow(1, y);
        T* PIK_RESTRICT row2 = out->PlaneRow(2, y);

        for (size_t x = 0; x < xsize; ++x) {
          row0[x] = row[4 * x + 0];
          row1[x] = row[4 * x + 1];
          row2[x] = row[4 * x + 2];
        }
      }
    } else {
      for (size_t y = 0; y < ysize; ++y) {
        const T* PIK_RESTRICT row =
            reinterpret_cast<const T*>(external.ConstRow(y));
        T* PIK_RESTRICT row0 = out->PlaneRow(0, y);
        T* PIK_RESTRICT row1 = out->PlaneRow(1, y);
        T* PIK_RESTRICT row2 = out->PlaneRow(2, y);
        for (size_t x = 0; x < xsize; ++x) {
          row0[x] = row[3 * x + 0];
          row1[x] = row[3 * x + 1];
          row2[x] = row[3 * x + 2];
        }
      }
    }
  }
}

// Copies io:rect, converts, and copies into out.
template <typename T>
Status CopyToT(const CodecInOut* io, const Rect& rect,
               const ColorEncoding& c_desired, ThreadPool* pool,
               Image3<T>* out) {
  PROFILER_FUNC;
  // Changing IsGray is probably a bug.
  PIK_CHECK(io->IsGray() == c_desired.IsGray());

  const ImageU* alpha = io->HasAlpha() ? &io->alpha() : nullptr;
  const size_t bits_per_sample = sizeof(T) * kBitsPerByte;
  const bool big_endian = !IsLittleEndian();
  CodecIntervals* temp_intervals = nullptr;  // Don't need min/max.
  const ExternalImage external(pool, io->color(), rect, io->c_current(),
                               c_desired, io->HasAlpha(), alpha,
                               bits_per_sample, big_endian, temp_intervals);
  PIK_RETURN_IF_ERROR(external.IsHealthy());
  AllocateAndFill(external, out);
  return true;
}

}  // namespace

std::vector<Codec> Values(Codec) { return {Codec::kPNG, Codec::kPNM}; }

std::string ExtensionFromCodec(Codec codec) {
  switch (codec) {
    case Codec::kPNG:
      return ".png";
    case Codec::kPNM:
      return ".pfm";
    case Codec::kUnknown:
      return std::string();
  }
  PIK_ASSERT(false);
  return std::string();
}

Codec CodecFromExtension(const std::string& extension) {
  if (extension == ".png") return Codec::kPNG;

  if (extension == ".pgm") return Codec::kPNM;
  if (extension == ".ppm") return Codec::kPNM;
  if (extension == ".pfm") return Codec::kPNM;

  return Codec::kUnknown;
}

CodecContext::CodecContext(size_t ignored)
    : c_srgb(MakeC2(Primaries::kSRGB, TransferFunction::kSRGB)),
      c_linear_srgb(MakeC2(Primaries::kSRGB, TransferFunction::kLinear)) {
  // For all supported targets:
  TargetBitfield supported;
  do {
    const Target target = supported.Best();
    Dispatch(target, InitEdgePreservingFilter());
    supported.Clear(target);
  } while (supported.Any());
}

void CodecInOut::SetFromImage(Image3F&& color, const ColorEncoding& c_current) {
  c_current_ = c_current;
  color_ = std::move(color);
}

Status CodecInOut::SetFromSRGB(size_t xsize, size_t ysize, bool is_gray,
                               bool has_alpha, const uint8_t* pixels,
                               const uint8_t* end, ThreadPool* pool) {
  const bool big_endian = false;
  return FromSRGB(xsize, ysize, is_gray, has_alpha, big_endian, pixels, end,
                  pool, this);
}

Status CodecInOut::SetFromSRGB(size_t xsize, size_t ysize, bool is_gray,
                               bool has_alpha, bool big_endian,
                               const uint16_t* pixels, const uint16_t* end,
                               ThreadPool* pool) {
  return FromSRGB(xsize, ysize, is_gray, has_alpha, big_endian, pixels, end,
                  pool, this);
}

Status CodecInOut::SetFromBytes(const PaddedBytes& bytes, ThreadPool* pool) {
  if (bytes.size() < kMinBytes) return PIK_FAILURE("Too few bytes");

  if (!DecodeImagePNG(bytes, pool, this) &&
      !DecodeImagePNM(bytes, pool, this)) {
    return PIK_FAILURE("Codecs failed to decode");
  }

  PIK_CHECK(!c_current().icc.empty());  // Must have gotten ICC profile
  PIK_CHECK(!dec_c_original.icc.empty());
  return true;
}

Status CodecInOut::SetFromFile(const std::string& pathname, ThreadPool* pool) {
  PaddedBytes encoded;
  return ReadFile(pathname, &encoded) && SetFromBytes(encoded, pool);
}

Status CodecInOut::TransformTo(const ColorEncoding& c_desired,
                               ThreadPool* pool) {
  PROFILER_FUNC;
  // Changing IsGray is probably a bug.
  PIK_CHECK(IsGray() == c_desired.IsGray());

  const ImageU* alpha = HasAlpha() ? &alpha_ : nullptr;
  const bool big_endian = !IsLittleEndian();
  CodecIntervals temp_intervals;
  const ExternalImage external(pool, color_, Rect(color_), c_current_,
                               c_desired, HasAlpha(), alpha, 32, big_endian,
                               &temp_intervals);
  return external.IsHealthy() && external.CopyTo(&temp_intervals, pool, this);
}

Status CodecInOut::CopyTo(const Rect& rect, const ColorEncoding& c_desired,
                          Image3B* out, ThreadPool* pool) const {
  return CopyToT(this, rect, c_desired, pool, out);
}
Status CodecInOut::CopyTo(const Rect& rect, const ColorEncoding& c_desired,
                          Image3U* out, ThreadPool* pool) const {
  return CopyToT(this, rect, c_desired, pool, out);
}
Status CodecInOut::CopyTo(const Rect& rect, const ColorEncoding& c_desired,
                          Image3F* out, ThreadPool* pool) const {
  return CopyToT(this, rect, c_desired, pool, out);
}

Status CodecInOut::CopyToSRGB(const Rect& rect, Image3B* out,
                              ThreadPool* pool) const {
  return CopyTo(rect, context_->c_srgb[IsGray()], out, pool);
}

Status CodecInOut::Encode(const Codec codec, const ColorEncoding& c_desired,
                          size_t bits_per_sample, PaddedBytes* bytes,
                          ThreadPool* pool) const {
  PIK_CHECK(!c_current().icc.empty());
  PIK_CHECK(!c_desired.icc.empty());

  switch (codec) {
    case Codec::kPNG:
      return EncodeImagePNG(this, c_desired, bits_per_sample, pool, bytes);
    case Codec::kPNM:
      return EncodeImagePNM(this, c_desired, bits_per_sample, pool, bytes);
    case Codec::kUnknown:
      return PIK_FAILURE("Cannot encode using Codec::kUnknown");
  }

  return PIK_FAILURE("Invalid codec");
}

Status CodecInOut::EncodeToFile(const ColorEncoding& c_desired,
                                size_t bits_per_sample,
                                const std::string& pathname,
                                ThreadPool* pool) const {
  const Codec codec = CodecFromExtension(Extension(pathname));

  PaddedBytes encoded;
  return Encode(codec, c_desired, bits_per_sample, &encoded, pool) &&
         WriteFile(encoded, pathname);
}

}  // namespace pik
