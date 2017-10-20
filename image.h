// Copyright 2016 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Disclaimer: This is not an official Google product.

#ifndef IMAGE_H_
#define IMAGE_H_

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "arch_specific.h"  // kVectorSize
#include "cache_aligned.h"
#include "compiler_specific.h"
#include "status.h"

namespace pik {

// Single channel, contiguous (cache-aligned) rows separated by padding.
// T must be POD.
//
// Rationale: vectorization benefits from aligned operands - unaligned loads and
// especially stores are expensive when the address crosses cache line
// boundaries. Introducing padding after each row ensures the start of a row is
// aligned, and that row loops can process entire vectors (writes to the padding
// are allowed and ignored).
//
// We prefer a planar representation, where channels are stored as separate
// 2D arrays, because that simplifies vectorization (repeating the same
// operation on multiple adjacent components) without the complexity of a
// hybrid layout (8 R, 8 G, 8 B, ...). In particular, clients can easily iterate
// over all components in a row and Image requires no knowledge of the pixel
// format beyond the component type "T". The downside is that we duplicate the
// xsize/ysize members for each channel.
//
// This image layout could also be achieved with a vector and a row accessor
// function, but a class wrapper with support for "deleter" allows wrapping
// existing memory allocated by clients without copying the pixels. It also
// provides convenient accessors for xsize/ysize, which shortens function
// argument lists. Supports move-construction so it can be stored in containers.
template <typename ComponentType>
class Image {
 public:
  using T = ComponentType;
  static constexpr size_t kNumPlanes = 1;

  Image()
      : xsize_(0),
        ysize_(0),
        bytes_per_row_(0),
        bytes_(nullptr, Ignore) {}

  Image(const size_t xsize, const size_t ysize)
      : xsize_(xsize),
        ysize_(ysize),
        bytes_per_row_(BytesPerRow(xsize)),
        bytes_(AllocateArray(bytes_per_row_ * ysize)) {
    for (size_t y = 0; y < ysize_; ++y) {
      T* const PIK_RESTRICT row = Row(y);
      // Avoid use of uninitialized values msan error in padding in WriteImage
      memset(row + xsize_, 0, bytes_per_row_ - sizeof(T) * xsize_);
    }
  }

  Image(const size_t xsize, const size_t ysize, T val)
      : xsize_(xsize),
        ysize_(ysize),
        bytes_per_row_(BytesPerRow(xsize)),
        bytes_(AllocateArray(bytes_per_row_ * ysize)) {
    for (size_t y = 0; y < ysize_; ++y) {
      T* const PIK_RESTRICT row = Row(y);
      for (size_t x = 0; x < xsize_; ++x) {
        row[x] = val;
      }
      // Avoid use of uninitialized values msan error in padding in WriteImage
      memset(row + xsize_, 0, bytes_per_row_ - sizeof(T) * xsize_);
    }
  }

  Image(const size_t xsize, const size_t ysize,
        uint8_t* const PIK_RESTRICT bytes, const size_t bytes_per_row)
      : xsize_(xsize),
        ysize_(ysize),
        bytes_per_row_(bytes_per_row),
        bytes_(bytes, Ignore) {
    PIK_ASSERT(bytes_per_row >= xsize * sizeof(T));
  }

  // Takes ownership.
  Image(const size_t xsize, const size_t ysize, CacheAlignedUniquePtr&& bytes,
        const size_t bytes_per_row)
      : xsize_(xsize),
        ysize_(ysize),
        bytes_per_row_(bytes_per_row),
        bytes_(std::move(bytes)) {
    PIK_ASSERT(bytes_per_row >= xsize * sizeof(T));
  }

  // Move constructor (required for returning Image from function)
  Image(Image&& other)
      : xsize_(other.xsize_),
        ysize_(other.ysize_),
        bytes_per_row_(other.bytes_per_row_),
        bytes_(std::move(other.bytes_)) {}

  // Move assignment (required for std::vector)
  Image& operator=(Image&& other) {
    xsize_ = other.xsize_;
    ysize_ = other.ysize_;
    bytes_per_row_ = other.bytes_per_row_;
    bytes_ = std::move(other.bytes_);
    return *this;
  }

  void Swap(Image& other) {
    std::swap(xsize_, other.xsize_);
    std::swap(ysize_, other.ysize_);
    std::swap(bytes_per_row_, other.bytes_per_row_);
    std::swap(bytes_, other.bytes_);
  }

  // Useful for pre-allocating image with some padding for alignment purposes
  // and later reporting the actual valid dimensions.
  void ShrinkTo(const size_t xsize, const size_t ysize) {
    PIK_ASSERT(xsize <= xsize_);
    PIK_ASSERT(ysize <= ysize_);
    xsize_ = xsize;
    ysize_ = ysize;
  }

  // How many pixels.
  PIK_INLINE size_t xsize() const { return xsize_; }
  PIK_INLINE size_t ysize() const { return ysize_; }

  // Returns pointer to the start of a row, with at least xsize (rounded up to
  // the number of vector lanes) accessible values.
  PIK_INLINE T* PIK_RESTRICT Row(const size_t y) {
    PIK_ASSERT(y < ysize_);
    void* row = bytes_.get() + y * bytes_per_row_;
    return static_cast<T*>(PIK_ASSUME_ALIGNED(row, 64));
  }

  // Returns pointer to const (see above).
  PIK_INLINE const T* PIK_RESTRICT Row(const size_t y) const {
    PIK_ASSERT(y < ysize_);
    void* row = bytes_.get() + y * bytes_per_row_;
    return static_cast<T*>(PIK_ASSUME_ALIGNED(row, 64));
  }

  // Returns pointer to const (see above), even if called on a non-const Image.
  PIK_INLINE const T* PIK_RESTRICT ConstRow(const size_t y) const  {
    return Row(y);
  }

  // Raw access to byte contents, for interfacing with other libraries.
  // Unsigned char instead of char to avoid surprises (sign extension).
  PIK_INLINE uint8_t* PIK_RESTRICT bytes() {
    void* p = bytes_.get();
    return static_cast<uint8_t * PIK_RESTRICT>(PIK_ASSUME_ALIGNED(p, 64));
  }
  PIK_INLINE const uint8_t* PIK_RESTRICT bytes() const {
    void* p = bytes_.get();
    return static_cast<const uint8_t * PIK_RESTRICT>(PIK_ASSUME_ALIGNED(p, 64));
  }

  // Returns cache-aligned row stride, being careful to avoid 2K aliasing.
  static size_t BytesPerRow(const size_t xsize) {
    // lowpass reads one extra AVX-2 vector on the right margin.
    const size_t row_size = xsize * sizeof(T) + kVectorSize;
    const size_t align = CacheAligned::kCacheLineSize;
    size_t bytes_per_row = (row_size + align - 1) & ~(align - 1);
    // During the lengthy window before writes are committed to memory, CPUs
    // guard against read after write hazards by checking the address, but
    // only the lower 11 bits. We avoid a false dependency between writes to
    // consecutive rows by ensuring their sizes are not multiples of 2 KiB.
    if (bytes_per_row % 2048 == 0) {
      bytes_per_row += align;
    }
    return bytes_per_row;
  }

  PIK_INLINE size_t bytes_per_row() const { return bytes_per_row_; }

  // Returns number of pixels (some of which are padding) per row. Useful for
  // computing other rows via pointer arithmetic.
  PIK_INLINE intptr_t PixelsPerRow() const {
    static_assert(CacheAligned::kCacheLineSize % sizeof(T) == 0,
                  "Padding must be divisible by the pixel size.");
    return static_cast<intptr_t>(bytes_per_row_ / sizeof(T));
  }

 private:
  // Deleter used when bytes are not owned.
  static void Ignore(uint8_t* ptr) {}

  // (Members are non-const to enable assignment during move-assignment.)
  uint32_t xsize_;  // original intended pixels, not including any padding.
  uint32_t ysize_;
  size_t bytes_per_row_;  // [bytes] including padding.
  CacheAlignedUniquePtr bytes_;
};

using ImageB = Image<uint8_t>;
// TODO(janwas): rename to ImageS (short/signed)
using ImageW = Image<int16_t>;  // signed integer or half-float
using ImageU = Image<uint16_t>;
using ImageF = Image<float>;

template <typename T>
Image<T> CopyImage(const Image<T>& image) {
  const size_t xsize = image.xsize();
  const size_t ysize = image.ysize();
  Image<T> copy(xsize, ysize);
  for (size_t y = 0; y < ysize; ++y) {
    const T* const PIK_RESTRICT row = image.Row(y);
    T* const PIK_RESTRICT row_copy = copy.Row(y);
    memcpy(row_copy, row, xsize * sizeof(T));
  }
  return copy;
}

template <typename T>
bool SamePixels(const Image<T>& image1, const Image<T>& image2) {
  const size_t xsize = image1.xsize();
  const size_t ysize = image1.ysize();
  PIK_CHECK(xsize == image2.xsize());
  PIK_CHECK(ysize == image2.ysize());
  for (size_t y = 0; y < ysize; ++y) {
    const T* const PIK_RESTRICT row1 = image1.Row(y);
    const T* const PIK_RESTRICT row2 = image2.Row(y);
    for (size_t x = 0; x < xsize; ++x) {
      if (row1[x] != row2[x]) {
        return false;
      }
    }
  }
  return true;
}

template <class ImageIn, class ImageOut>
void Subtract(const ImageIn& image1, const ImageIn& image2, ImageOut* out) {
  using T = typename ImageIn::T;
  const size_t xsize = image1.xsize();
  const size_t ysize = image1.ysize();
  PIK_CHECK(xsize == image2.xsize());
  PIK_CHECK(ysize == image2.ysize());

  for (size_t y = 0; y < ysize; ++y) {
    const T* const PIK_RESTRICT row1 = image1.Row(y);
    const T* const PIK_RESTRICT row2 = image2.Row(y);
    T* const PIK_RESTRICT row_out = out->Row(y);
    for (size_t x = 0; x < xsize; ++x) {
      row_out[x] = row1[x] - row2[x];
    }
  }
}

// In-place.
template <typename Tin, typename Tout>
void SubtractFrom(const Image<Tin>& what, Image<Tout>* to) {
  const size_t xsize = what.xsize();
  const size_t ysize = what.ysize();
  for (size_t y = 0; y < ysize; ++y) {
    const Tin* PIK_RESTRICT row_what = what.ConstRow(y);
    Tout* PIK_RESTRICT row_to = to->Row(y);
    for (size_t x = 0; x < xsize; ++x) {
      row_to[x] -= row_what[x];
    }
  }
}

// In-place.
template <typename Tin, typename Tout>
void AddTo(const Image<Tin>& what, Image<Tout>* to) {
  const size_t xsize = what.xsize();
  const size_t ysize = what.ysize();
  for (size_t y = 0; y < ysize; ++y) {
    const Tin* PIK_RESTRICT row_what = what.ConstRow(y);
    Tout* PIK_RESTRICT row_to = to->Row(y);
    for (size_t x = 0; x < xsize; ++x) {
      row_to[x] += row_what[x];
    }
  }
}

// Returns linear combination of two grayscale images.
template <typename T>
Image<T> LinComb(const T lambda1, const Image<T>& image1,
                 const T lambda2, const Image<T>& image2) {
  const size_t xsize = image1.xsize();
  const size_t ysize = image1.ysize();
  PIK_CHECK(xsize == image2.xsize());
  PIK_CHECK(ysize == image2.ysize());
  Image<T> out(xsize, ysize);
  for (size_t y = 0; y < ysize; ++y) {
    const T* const PIK_RESTRICT row1 = image1.Row(y);
    const T* const PIK_RESTRICT row2 = image2.Row(y);
    T* const PIK_RESTRICT row_out = out.Row(y);
    for (size_t x = 0; x < xsize; ++x) {
      row_out[x] = lambda1 * row1[x] + lambda2 * row2[x];
    }
  }
  return out;
}

// Returns a pixel-by-pixel multiplication of image by lambda.
template <typename T>
Image<T> ScaleImage(const T lambda, const Image<T>& image) {
  Image<T> out(image.xsize(), image.ysize());
  for (size_t y = 0; y < image.ysize(); ++y) {
    const T* const PIK_RESTRICT row = image.Row(y);
    T* const PIK_RESTRICT row_out = out.Row(y);
    for (size_t x = 0; x < image.xsize(); ++x) {
      row_out[x] = lambda * row[x];
    }
  }
  return out;
}

template <typename T>
Image<T> ZeroPadImage(const Image<T>& in, int padx0, int pady0, int padx1,
                      int pady1) {
  Image<T> out(in.xsize() + padx0 + padx1, in.ysize() + pady0 + pady1, T());
  for (int y = 0; y < in.ysize(); ++y) {
    memcpy(out.Row(y + pady0) + padx0, in.Row(y), in.xsize() * sizeof(T));
  }
  return out;
}

template <typename T>
Image<T> Product(const Image<T>& a, const Image<T>& b) {
  Image<T> c(a.xsize(), a.ysize());
  for (size_t y = 0; y < a.ysize(); ++y) {
    const T* const PIK_RESTRICT row_a = a.Row(y);
    const T* const PIK_RESTRICT row_b = b.Row(y);
    T* const PIK_RESTRICT row_c = c.Row(y);
    for (size_t x = 0; x < a.xsize(); ++x) {
      row_c[x] = row_a[x] * row_b[x];
    }
  }
  return c;
}

float DotProduct(const ImageF& a, const ImageF& b);

template <typename T>
Image<T> TorusShift(const Image<T>& img, size_t shift_x, size_t shift_y) {
  Image<T> out(img.xsize(), img.ysize());
  for (size_t y = 0; y < img.ysize(); ++y) {
    const T* const PIK_RESTRICT row_in = img.Row((y + shift_y) % img.ysize());
    T* const PIK_RESTRICT row_out = out.Row(y);
    for (size_t x = 0; x < img.xsize(); ++x) {
      row_out[x] = row_in[(x + shift_x) % img.xsize()];
    }
  }
  return out;
}


// Ensures each row has RoundUp(xsize, N) + N valid values (N = NumLanes<V>()).
// Generates missing values via mirroring with replication.
template <typename T>
void PadImage(Image<T>* image) {
  const size_t xsize = image->xsize();
  const size_t kNumLanes = kVectorSize / sizeof(T);
  const size_t remainder = xsize % kNumLanes;
  const size_t extra = (remainder == 0) ? 0 : (kNumLanes - remainder);
  const size_t max_mirror = std::min(kNumLanes + extra, xsize - 1);

  for (size_t y = 0; y < image->ysize(); ++y) {
    T* const PIK_RESTRICT row = image->Row(y);

    size_t i = 0;
    for (; i < max_mirror; ++i) {
      // The mirror is outside the last column => replicate once.
      row[xsize + i] = row[xsize - 1 - i];
    }
    for (; i < kNumLanes + extra; ++i) {
      row[xsize + i] = 0;
    }
  }
}

// Returns whether the first row is padded (safety check).
template <typename T>
bool VerifyPadding(const Image<T>& image) {
  const size_t xsize = image.xsize();
  const size_t kNumLanes = kVectorSize / sizeof(T);
  const size_t remainder = xsize % kNumLanes;
  const size_t extra = (remainder == 0) ? 0 : (kNumLanes - remainder);
  const size_t max_mirror = std::min(kNumLanes + extra, xsize - 1);

  const T* const PIK_RESTRICT row = image.Row(0);
  size_t i = 0;
  for (; i < max_mirror; ++i) {
    // The mirror is outside the last column => replicate once.
    if (fabsf(row[xsize + i] - row[xsize - 1 - i]) > 1E-3f) {
      return false;
    }
  }
  for (; i < kNumLanes + extra; ++i) {
    if (row[xsize + i] != 0) {
      return false;
    }
  }

  return true;
}

// Sets "thickness" pixels on each border to "value". This is faster than
// initializing the entire image and overwriting valid/interior pixels.
template <typename T>
void SetBorder(const size_t thickness, const T value, Image<T>* image) {
  const size_t xsize = image->xsize();
  const size_t ysize = image->ysize();
  PIK_ASSERT(2 * thickness < xsize && 2 * thickness < ysize);
  // Top
  for (size_t y = 0; y < thickness; ++y) {
    T* const PIK_RESTRICT row = image->Row(y);
    std::fill(row, row + xsize, value);
  }

  // Bottom
  for (size_t y = ysize - thickness; y < ysize; ++y) {
    T* const PIK_RESTRICT row = image->Row(y);
    std::fill(row, row + xsize, value);
  }

  // Left/right
  for (size_t y = thickness; y < ysize - thickness; ++y) {
    T* const PIK_RESTRICT row = image->Row(y);
    std::fill(row, row + thickness, value);
    std::fill(row + xsize - thickness, row + xsize, value);
  }
}

// Shrinks the image (without reallocating nor copying pixels) such that its
// size is a multiple of x/y_multiple.
template <class Image>
void CropToMultipleOf(const size_t x_multiple, const size_t y_multiple,
                      Image* image) {
  const size_t xsize = image->xsize();
  const size_t ysize = image->ysize();
  const size_t x_excess = xsize % x_multiple;
  const size_t y_excess = ysize % y_multiple;
  if (x_excess != 0 || y_excess != 0) {
    image->ShrinkTo(xsize - x_excess, ysize - y_excess);
  }
}

// Computes the minimum and maximum pixel value.
template <typename T>
void ImageMinMax(const Image<T>& image, T* const PIK_RESTRICT min,
                 T* const PIK_RESTRICT max) {
  *min = std::numeric_limits<T>::max();
  *max = std::numeric_limits<T>::min();
  for (size_t y = 0; y < image.ysize(); ++y) {
    const T* const PIK_RESTRICT row = image.Row(y);
    for (size_t x = 0; x < image.xsize(); ++x) {
      *min = std::min(*min, row[x]);
      *max = std::max(*max, row[x]);
    }
  }
}

// Copies pixels, scaling their value relative to the "from" min/max by
// "to_range". Example: U8 [0, 255] := [0.0, 1.0], to_range = 1.0 =>
// outputs [0.0, 1.0].
template <typename FromType, typename ToType>
void ImageConvert(const Image<FromType>& from, const float to_range,
                  Image<ToType>* const PIK_RESTRICT to) {
  PIK_ASSERT(from.xsize() == to->xsize());
  PIK_ASSERT(from.ysize() == to->ysize());
  FromType min_from, max_from;
  ImageMinMax(from, &min_from, &max_from);
  const float scale = to_range / (max_from - min_from);
  for (size_t y = 0; y < from.ysize(); ++y) {
    const FromType* const PIK_RESTRICT row_from = from.Row(y);
    ToType* const PIK_RESTRICT row_to = to->Row(y);
    for (size_t x = 0; x < from.xsize(); ++x) {
      row_to[x] = static_cast<ToType>((row_from[x] - min_from) * scale);
    }
  }
}

// FromType and ToType are the pixel types. For float to byte, consider using
// Float255ToByteImage instead as it rounds the values correctly.
template <typename FromType, typename ToType>
Image<ToType> StaticCastImage(const Image<FromType>& from) {
  Image<ToType> to(from.xsize(), from.ysize());
  for (size_t y = 0; y < from.ysize(); ++y) {
    const FromType* const PIK_RESTRICT row_from = from.Row(y);
    ToType* const PIK_RESTRICT row_to = to.Row(y);
    for (size_t x = 0; x < from.xsize(); ++x) {
      row_to[x] = static_cast<ToType>(row_from[x]);
    }
  }
  return to;
}

ImageB Float255ToByteImage(const ImageF& from);

template <typename T>
std::vector<T> PackedFromImage(const Image<T>& image) {
  const size_t xsize = image.xsize();
  const size_t ysize = image.ysize();
  std::vector<T> packed(xsize * ysize);
  for (size_t y = 0; y < ysize; ++y) {
    memcpy(&packed[y * xsize], image.Row(y), xsize * sizeof(T));
  }
  return packed;
}

template <typename T>
Image<T> ImageFromPacked(const std::vector<T>& packed, const size_t xsize,
                         const size_t ysize) {
  Image<T> out(xsize, ysize);
  for (size_t y = 0; y < ysize; ++y) {
    T* const PIK_RESTRICT row = out.Row(y);
    const T* const PIK_RESTRICT packed_row = &packed[y * xsize];
    memcpy(row, packed_row, xsize * sizeof(T));
  }
  return out;
}

ImageB ImageFromPacked(const uint8_t* packed, const size_t xsize,
                       const size_t ysize, const size_t bytes_per_row);

// A bundle of 3 images of same size.
template <typename ComponentType>
class Image3 {
 public:
  using T = ComponentType;
  using Plane = Image<T>;
  static constexpr size_t kNumPlanes = 3;

  Image3() : planes_{Plane(), Plane(), Plane()} {}

  Image3(const size_t xsize, const size_t ysize)
      : planes_{Plane(xsize, ysize), Plane(xsize, ysize), Plane(xsize, ysize)} {
  }

  Image3(const size_t xsize, const size_t ysize, const T val)
      : planes_{Plane(xsize, ysize, val),
                Plane(xsize, ysize, val),
                Plane(xsize, ysize, val)} {
  }

  Image3(Image3&& other) {
    for (int i = 0; i < kNumPlanes; i++) {
      planes_[i] = std::move(other.planes_[i]);
    }
  }

  Image3(Plane&& plane0, Plane&& plane1, Plane&& plane2) {
    PIK_CHECK(plane0.xsize() == plane1.xsize());
    PIK_CHECK(plane0.ysize() == plane1.ysize());
    PIK_CHECK(plane0.xsize() == plane2.xsize());
    PIK_CHECK(plane0.ysize() == plane2.ysize());
    planes_[0] = std::move(plane0);
    planes_[1] = std::move(plane1);
    planes_[2] = std::move(plane2);
  }

  // Used for reassembling after Deconstruct.
  Image3(std::array<Plane, kNumPlanes>& planes)
      : Image3(std::move(planes[0]), std::move(planes[1]),
               std::move(planes[2])) {}

  Image3& operator=(Image3&& other) {
    for (int i = 0; i < kNumPlanes; i++) {
      planes_[i] = std::move(other.planes_[i]);
    }
    return *this;
  }

  PIK_INLINE std::array<Plane, kNumPlanes> Deconstruct() {
    return std::array<Plane, kNumPlanes>{
        std::move(planes_[0]), std::move(planes_[1]), std::move(planes_[2])};
  }

  // (Image<>::Row takes care of bounds checking)

  // Returns array of row pointers; usage: Row(y)[idx_plane][x] = val.
  PIK_INLINE std::array<T * PIK_RESTRICT, kNumPlanes> Row(const size_t y) {
    return {{planes_[0].Row(y), planes_[1].Row(y), planes_[2].Row(y)}};
  }

  // Returns array of const row pointers; usage: val = Row(y)[idx_plane][x].
  PIK_INLINE std::array<const T * PIK_RESTRICT, kNumPlanes> Row(
      const size_t y) const {
    return {{planes_[0].Row(y), planes_[1].Row(y), planes_[2].Row(y)}};
  }

  // Returns const row pointers, even if called from a non-const Image3.
  PIK_INLINE std::array<const T * PIK_RESTRICT, kNumPlanes> ConstRow(
      const size_t y) const {
    return Row(y);
  }

  // Returns row pointer; usage: PlaneRow(idx_plane, y)[x] = val.
  PIK_INLINE T* PIK_RESTRICT PlaneRow(const int c, const size_t y) {
    return planes_[c].Row(y);
  }

  // Returns const row pointer; usage: val = PlaneRow(idx_plane, y)[x].
  PIK_INLINE const T* PIK_RESTRICT PlaneRow(const int c, const size_t y) const {
    return planes_[c].Row(y);
  }

  // Returns const row pointer, even if called from a non-const Image3.
  PIK_INLINE const T* PIK_RESTRICT ConstPlaneRow(const int c,
                                                 const size_t y) const {
    return PlaneRow(c, y);
  }

  // NOTE: we deliberately avoid non-const plane accessors - callers could use
  // them to change image size etc., leading to hard-to-find bugs.
  PIK_INLINE const Plane& plane(const size_t idx) const { return planes_[idx]; }

  void ShrinkTo(const size_t xsize, const size_t ysize) {
    for (Plane& plane : planes_) { plane.ShrinkTo(xsize, ysize); }
  }

  // Sizes of all three images are guaranteed to be equal.
  PIK_INLINE size_t xsize() const { return planes_[0].xsize(); }
  PIK_INLINE size_t ysize() const { return planes_[0].ysize(); }

 private:
  Plane planes_[kNumPlanes];
};

using Image3B = Image3<uint8_t>;
// TODO(janwas): rename to ImageS (short/signed)
using Image3W = Image3<int16_t>;
using Image3U = Image3<uint16_t>;
using Image3F = Image3<float>;
using Image3D = Image3<double>;

// Image data for formats: Image3 for color, optional Image for alpha channel.
template <typename ComponentType>
class MetaImage {
 public:
  using T = ComponentType;
  using Plane = Image<T>;

  const Image3<T>& GetColor() const {
    return color_;
  }

  Image3<T>& GetColor() {
    return color_;
  }

  void SetColor(Image3<T>&& color) {
    if (has_alpha_) {
      PIK_CHECK(color.xsize() == alpha_.xsize());
      PIK_CHECK(color.ysize() == alpha_.ysize());
    }
    color_ = std::move(color);
  }

  PIK_INLINE size_t xsize() const { return color_.xsize(); }
  PIK_INLINE size_t ysize() const { return color_.ysize(); }

  void AddAlpha() {
    PIK_CHECK(!has_alpha_);
    has_alpha_ = true;
    alpha_ = Plane(color_.xsize(), color_.ysize(), GetOpaqueValue());
  }

  void SetAlpha(Image<T>&& alpha) {
    PIK_CHECK(alpha.xsize() == color_.xsize());
    PIK_CHECK(alpha.ysize() == color_.ysize());
    has_alpha_ = true;
    alpha_ = std::move(alpha);
  }

  bool HasAlpha() const {
    return has_alpha_;
  }

  Image<T>& GetAlpha() {
    return alpha_;
  }

  const Image<T>& GetAlpha() const {
    return alpha_;
  }

  static T GetOpaqueValue() { return GetOpaqueValue(T(0)); }

 private:
  static uint8_t GetOpaqueValue(uint8_t type) { return 255; }
  static uint16_t GetOpaqueValue(uint16_t type) { return 65535; }
  static int16_t GetOpaqueValue(int16_t type) { return -1; }
  static float GetOpaqueValue(float type) { return 255.0f; }
  static double GetOpaqueValue(double type) { return 255.0; }

  Image3<T> color_;
  bool has_alpha_ = false;
  Image<T> alpha_;
};

using MetaImageB = MetaImage<uint8_t>;
// TODO(janwas): rename to MetaImageS (short/signed)
using MetaImageW = MetaImage<int16_t>;
using MetaImageU = MetaImage<uint16_t>;
using MetaImageF = MetaImage<float>;
using MetaImageD = MetaImage<double>;

template <typename T>
Image3<T> CopyImage3(const Image3<T>& image3) {
  return Image3<T>(CopyImage(image3.plane(0)), CopyImage(image3.plane(1)),
                   CopyImage(image3.plane(2)));
}

template <typename T>
bool SamePixels(const Image3<T>& image1, const Image3<T>& image2) {
  const size_t xsize = image1.xsize();
  const size_t ysize = image1.ysize();
  PIK_CHECK(xsize == image2.xsize());
  PIK_CHECK(ysize == image2.ysize());
  for (size_t y = 0; y < ysize; ++y) {
    auto rows1 = image1.Row(y);
    auto rows2 = image2.Row(y);
    for (size_t x = 0; x < xsize; ++x) {
      if (rows1[0][x] != rows2[0][x] || rows1[1][x] != rows2[1][x] ||
          rows1[2][x] != rows2[2][x]) {
        return false;
      }
    }
  }
  return true;
}

// Sets "thickness" pixels on each border to "value". This is faster than
// initializing the entire image and overwriting valid/interior pixels.
template <typename T>
void SetBorder(const size_t thickness, const T value, Image3<T>* image) {
  const size_t xsize = image->xsize();
  const size_t ysize = image->ysize();
  PIK_ASSERT(2 * thickness < xsize && 2 * thickness < ysize);
  // Top
  for (size_t y = 0; y < thickness; ++y) {
    auto row = image->Row(y);
    for (size_t c = 0; c < 3; ++c) {
      std::fill(row[c], row[c] + xsize, value);
    }
  }

  // Bottom
  for (size_t y = ysize - thickness; y < ysize; ++y) {
    auto row = image->Row(y);
    for (size_t c = 0; c < 3; ++c) {
      std::fill(row[c], row[c] + xsize, value);
    }
  }

  // Left/right
  for (size_t y = thickness; y < ysize - thickness; ++y) {
    auto row = image->Row(y);
    for (size_t c = 0; c < 3; ++c) {
      std::fill(row[c], row[c] + thickness, value);
      std::fill(row[c] + xsize - thickness, row[c] + xsize, value);
    }
  }
}

// Computes independent minimum and maximum values for each plane.
template <typename T>
void Image3MinMax(const Image3<T>& image, std::array<T, 3>* min,
                  std::array<T, 3>* max) {
  for (int c = 0; c < 3; ++c) {
    (*min)[c] = std::numeric_limits<T>::max();
    (*max)[c] = std::numeric_limits<T>::min();
  }
  for (size_t y = 0; y < image.ysize(); ++y) {
    const auto rows = image.ConstRow(y);
    for (size_t x = 0; x < image.xsize(); ++x) {
      for (int c = 0; c < 3; ++c) {
        (*min)[c] = std::min((*min)[c], rows[c][x]);
        (*max)[c] = std::max((*max)[c], rows[c][x]);
      }
    }
  }
}

template <typename FromType, typename ToType>
void Image3Convert(const Image3<FromType>& from, const float to_range,
                   Image3<ToType>* const PIK_RESTRICT to) {
  PIK_ASSERT(from.xsize() == to->xsize());
  PIK_ASSERT(from.ysize() == to->ysize());
  std::array<FromType, 3> min_from, max_from;
  Image3MinMax(from, &min_from, &max_from);
  float scales[3];
  for (int c = 0; c < 3; ++c) {
    scales[c] = to_range / (max_from[c] - min_from[c]);
  }
  float scale = std::min(scales[0], std::min(scales[1], scales[2]));
  for (size_t y = 0; y < from.ysize(); ++y) {
    const auto from_rows = from.ConstRow(y);
    auto to_rows = to->Row(y);
    for (size_t x = 0; x < from.xsize(); ++x) {
      for (int c = 0; c < 3; ++c) {
        const float to = (from_rows[c][x] - min_from[c]) * scale;
        to_rows[c][x] = static_cast<ToType>(to);
      }
    }
  }
}

// FromType and ToType are the pixel types. For float to byte, consider using
// Float255ToByteImage instead as it rounds the values correctly.
template <typename FromType, typename ToType>
Image3<ToType> StaticCastImage3(const Image3<FromType>& from) {
  Image3<ToType> to(from.xsize(), from.ysize());
  for (size_t y = 0; y < from.ysize(); ++y) {
    const auto from_rows = from.ConstRow(y);
    auto to_rows = to.Row(y);
    for (size_t x = 0; x < from.xsize(); ++x) {
      to_rows[0][x] = static_cast<ToType>(from_rows[0][x]);
      to_rows[1][x] = static_cast<ToType>(from_rows[1][x]);
      to_rows[2][x] = static_cast<ToType>(from_rows[2][x]);
    }
  }
  return to;
}

// Clamps input components to [0, 255] and casts to uint8_t.
Image3B Float255ToByteImage3(const Image3F& from);

template <typename Tin, typename Tout>
void SubtractFrom(const Image3<Tin>& what, Image3<Tout>* to) {
  const size_t xsize = what.xsize();
  const size_t ysize = what.ysize();
  for (size_t y = 0; y < ysize; ++y) {
    for (int c = 0; c < 3; ++c) {
      const Tin* PIK_RESTRICT row_what = what.ConstPlaneRow(c, y);
      Tout* PIK_RESTRICT row_to = to->PlaneRow(c, y);
      for (size_t x = 0; x < xsize; ++x) {
        row_to[x] -= row_what[x];
      }
    }
  }
}

template <typename Tin, typename Tout>
void AddTo(const Image3<Tin>& what, Image3<Tout>* to) {
  const size_t xsize = what.xsize();
  const size_t ysize = what.ysize();
  for (size_t y = 0; y < ysize; ++y) {
    for (int c = 0; c < 3; ++c) {
      const Tin* PIK_RESTRICT row_what = what.ConstPlaneRow(c, y);
      Tout* PIK_RESTRICT row_to = to->PlaneRow(c, y);
      for (size_t x = 0; x < xsize; ++x) {
        row_to[x] += row_what[x];
      }
    }
  }
}

template <typename T>
Image3<T> LinComb(const T lambda1, const Image3<T>& image1,
                  const T lambda2, const Image3<T>& image2) {
  Image<T> plane0 = LinComb(lambda1, image1.plane(0), lambda2, image2.plane(0));
  Image<T> plane1 = LinComb(lambda1, image1.plane(1), lambda2, image2.plane(1));
  Image<T> plane2 = LinComb(lambda1, image1.plane(2), lambda2, image2.plane(2));
  return Image3<T>(std::move(plane0), std::move(plane1), std::move(plane2));
}

template <typename T>
Image3<T> ScaleImage3(const T lambda, const Image3<T>& image) {
  return Image3<T>(ScaleImage(lambda, image.plane(0)),
                   ScaleImage(lambda, image.plane(1)),
                   ScaleImage(lambda, image.plane(2)));
}

// Assigns generator(x, y, c) to each pixel (x, y).
template <class Generator, typename T>
void FillImage(const Generator& generator, Image3<T>* image) {
  for (size_t y = 0; y < image->ysize(); ++y) {
    auto rows = image->Row(y);
    for (size_t x = 0; x < image->xsize(); ++x) {
      rows[0][x] = generator(x, y, 0);
      rows[1][x] = generator(x, y, 1);
      rows[2][x] = generator(x, y, 2);
    }
  }
}

template <typename T>
std::vector<T> InterleavedFromImage3(const Image3<T>& image3) {
  const size_t xsize = image3.xsize();
  const size_t ysize = image3.ysize();
  std::vector<T> interleaved(xsize * ysize * 3);
  for (size_t y = 0; y < ysize; ++y) {
    auto row = image3.Row(y);
    T* const PIK_RESTRICT interleaved_row = &interleaved[y * xsize * 3];
    for (size_t x = 0; x < xsize; ++x) {
      interleaved_row[3 * x + 0] = row[0][x];
      interleaved_row[3 * x + 1] = row[1][x];
      interleaved_row[3 * x + 2] = row[2][x];
    }
  }
  return interleaved;
}

template <typename T>
Image3<T> Image3FromInterleaved(const T* const interleaved, const size_t xsize,
                                const size_t ysize,
                                const size_t bytes_per_row) {
  PIK_ASSERT(bytes_per_row >= 3 * xsize * sizeof(T));
  Image3<T> image3(xsize, ysize);
  const uint8_t* bytes = reinterpret_cast<const uint8_t*>(interleaved);
  for (size_t y = 0; y < ysize; ++y) {
    auto out_row = image3.Row(y);
    const auto interleaved_row =
        reinterpret_cast<const T*>(bytes + y * bytes_per_row);
    for (size_t x = 0; x < xsize; ++x) {
      out_row[0][x] = interleaved_row[3 * x + 0];
      out_row[1][x] = interleaved_row[3 * x + 1];
      out_row[2][x] = interleaved_row[3 * x + 2];
    }
  }
  return image3;
}

template <typename T>
std::vector<std::vector<T>> Packed3FromImage3(const Image3<T>& planes) {
  std::vector<std::vector<T>> result(
      3, std::vector<T>(planes.xsize() * planes.ysize()));
  for (size_t y = 0; y < planes.ysize(); y++) {
    auto row = planes.Row(y);
    for (size_t x = 0; x < planes.xsize(); x++) {
      for (int i = 0; i < 3; i++) {
        result[i][y * planes.xsize() + x] = row[i][x];
      }
    }
  }
  return result;
}

template <typename T>
Image3<T> Image3FromPacked3(const std::vector<std::vector<T>>& packed,
                            const size_t xsize, const size_t ysize) {
  Image3<T> out(xsize, ysize);
  for (size_t y = 0; y < ysize; ++y) {
    auto row = out.Row(y);
    for (size_t x = 0; x < xsize; ++x) {
      for (int c = 0; c < 3; ++c) {
        row[c][x] = packed[c][y * xsize + x];
      }
    }
  }
  return out;
}

// Rounds size up to multiples of xres and yres by replicating the last pixel.
template <typename T>
Image3<T> ExpandAndCopyBorders(const Image3<T>& img, const size_t xres,
                               const size_t yres) {
  const size_t xsize = xres * ((img.xsize() + xres - 1) / xres);
  const size_t ysize = yres * ((img.ysize() + yres - 1) / yres);
  Image3<T> out(xsize, ysize);
  for (size_t y = 0; y < ysize; ++y) {
    auto row_in = img.Row(std::min(img.ysize() - 1, y));
    auto row_out = out.Row(y);
    for (int c = 0; c < 3; ++c) {
      size_t x = 0;
      for (; x < img.xsize(); ++x) {
        row_out[c][x] = row_in[c][x];
      }
      for (; x < xsize; ++x) {
        row_out[c][x] = row_in[c][img.xsize() - 1];
      }
    }
  }
  return out;
}

float Average(const ImageF& img);

template <typename T>
void AddScalar(T v, Image<T>* img) {
  const size_t xsize = img->xsize();
  const size_t ysize = img->ysize();
  for (size_t y = 0; y < ysize; ++y) {
    auto row = img->Row(y);
    for (size_t x = 0; x < xsize; ++x) {
      row[x] += v;
    }
  }
}

template <typename T>
void AddScalar(T v0, T v1, T v2, Image3<T>* img) {
  const size_t xsize = img->xsize();
  const size_t ysize = img->ysize();
  for (size_t y = 0; y < ysize; ++y) {
    auto row = img->Row(y);
    for (size_t x = 0; x < xsize; ++x) {
      row[0][x] += v0;
      row[1][x] += v1;
      row[2][x] += v2;
    }
  }
}

template<typename T, typename Fun>
void Apply(Fun f, Image<T>* image) {
  const size_t xsize = image->xsize();
  const size_t ysize = image->ysize();

  for(size_t y = 0; y < ysize; y++) {
    auto row = image->Row(y);
    for(size_t x = 0; x < xsize; x++) {
      f(&row[x]);
    }
  }
}

template <typename T>
void PrintImageStats(const std::string& desc, const Image<T>& img) {
  T mn, mx;
  ImageMinMax(img, &mn, &mx);
  fprintf(stderr, "Image %s: min=%lf, max=%lf\n", desc.c_str(), double(mn),
          double(mx));
}

template <typename T>
void PrintImageStats(const std::string& desc, const Image3<T>& img) {
  for (int c = 0; c < 3; ++c) {
    T mn, mx;
    ImageMinMax(img.plane(c), &mn, &mx);
    fprintf(stderr, "Image %s, plane %d: min=%lf, max=%lf\n",
            desc.c_str(), c, double(mn), double(mx));
  }
}

template <typename T>
void PrintImageStats(const std::string& desc,
                     const Image<std::complex<T>>& img) {
  T r_mn, r_mx, i_mn, i_mx;
  ImageMinMax(Real(img), &r_mn, &r_mx);
  ImageMinMax(Imag(img), &i_mn, &i_mx);
  fprintf(stderr,
          "Image %s: min(Re)=%lf, min(Im)=%lf, max(Re)=%lf, max(Im)=%lf\n",
          desc.c_str(), double(r_mn), double(i_mn), double(r_mx), double(i_mx));
}
#define PRINT_IMAGE_STATS_S(x) #x
#define PRINT_IMAGE_STATS_SS(x) PRINT_IMAGE_STATS_S(x)
#define PRINT_IMAGE_STATS(img)                                                 \
  ::pik::PrintImageStats(#img "@" __FILE__ ":" PRINT_IMAGE_STATS_SS(__LINE__), \
                         img)

}  // namespace pik

#endif  // IMAGE_H_
