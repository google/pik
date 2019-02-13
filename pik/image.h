// Copyright 2016 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_IMAGE_H_
#define PIK_IMAGE_H_

// SIMD/multicore-friendly planar image representation with row accessors.

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "pik/cache_aligned.h"
#include "pik/compiler_specific.h"
#include "pik/profiler.h"
#include "pik/status.h"

namespace pik {

// Each row address is a multiple of this - enables aligned loads.
static constexpr size_t kImageAlign = CacheAligned::kAlignment;
static_assert(kImageAlign >= kMaxVectorSize, "Insufficient alignment");

// Returns distance [bytes] between the start of two consecutive rows, a
// multiple of kAlign but NOT CacheAligned::kAlias - see below.
//
// Differing "kAlign" make sense for:
// - Image: 128 to avoid false sharing/RFOs between multiple threads processing
//   rows independently;
// - TileFlow: no cache line alignment needed because buffers are per-thread;
//   just need kMaxVectorSize=16..64 for SIMD.
//
// "valid_bytes" is xsize * sizeof(T).
template <size_t kAlign>
static inline size_t BytesPerRow(const size_t valid_bytes) {
  static_assert((kAlign & (kAlign - 1)) == 0, "kAlign should be power of two");

  // Extra two vectors allow *writing* a partial or full vector on the right AND
  // left border (for convolve.h) without disturbing the next/previous row.
  const size_t row_size = valid_bytes + 2 * kMaxVectorSize;

  // Round up.
  size_t bytes_per_row = (row_size + kAlign - 1) & ~(kAlign - 1);

  // During the lengthy window before writes are committed to memory, CPUs
  // guard against read after write hazards by checking the address, but
  // only the lower 11 bits. We avoid a false dependency between writes to
  // consecutive rows by ensuring their sizes are not multiples of 2 KiB.
  // Avoid2K prevents the same problem for the planes of an Image3.
  if (bytes_per_row % CacheAligned::kAlias == 0) {
    bytes_per_row += kImageAlign;
  }

  return bytes_per_row;
}

// Factored out of Image<> to avoid dependency on profiler.h and <atomic>.
CacheAlignedUniquePtr AllocateImageBytes(size_t size, size_t xsize,
                                         size_t ysize);

// Single channel, aligned rows separated by padding. T must be POD.
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
// format beyond the component type "T".
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

  Image() : xsize_(0), ysize_(0), bytes_per_row_(0), bytes_(nullptr) {}

  Image(const size_t xsize, const size_t ysize)
      : xsize_(xsize),
        ysize_(ysize),
        bytes_per_row_(BytesPerRow<kImageAlign>(xsize * sizeof(T))),
        bytes_(nullptr) {
    PIK_ASSERT(bytes_per_row_ % kImageAlign == 0);
    // xsize and/or ysize can legitimately be zero, in which case we don't
    // want to allocate.
    if (xsize != 0 && ysize != 0) {
      bytes_ = AllocateImageBytes(bytes_per_row_ * ysize + kMaxVectorSize,
                                  xsize, ysize);
    }

#ifdef MEMORY_SANITIZER
    // Only in MSAN builds: ensure full vectors are initialized.
    const size_t partial = (xsize_ * sizeof(T)) % kMaxVectorSize;
    const size_t remainder = (partial == 0) ? 0 : (kMaxVectorSize - partial);
    for (size_t y = 0; y < ysize_; ++y) {
      memset(Row(y) + xsize_, 0, remainder);
    }
#endif
  }

  // Copy construction/assignment is forbidden to avoid inadvertent copies,
  // which can be very expensive. Use CopyImageTo() instead.
  Image(const Image& other) = delete;
  Image& operator=(const Image& other) = delete;

  // Move constructor (required for returning Image from function)
  Image(Image&& other) = default;

  // Move assignment (required for std::vector)
  Image& operator=(Image&& other) = default;

  void Swap(Image& other) {
    std::swap(xsize_, other.xsize_);
    std::swap(ysize_, other.ysize_);
    std::swap(bytes_per_row_, other.bytes_per_row_);
    std::swap(bytes_, other.bytes_);
  }

  // Useful for pre-allocating image with some padding for alignment purposes
  // and later reporting the actual valid dimensions. Caller is responsible
  // for ensuring xsize/ysize are <= the original dimensions.
  void ShrinkTo(const size_t xsize, const size_t ysize) {
    xsize_ = static_cast<uint32_t>(xsize);
    ysize_ = static_cast<uint32_t>(ysize);
    // NOTE: we can't recompute bytes_per_row for more compact storage and
    // better locality because that would invalidate the image contents.
  }

  // How many pixels.
  PIK_INLINE size_t xsize() const { return xsize_; }
  PIK_INLINE size_t ysize() const { return ysize_; }

  // Returns pointer to the start of a row, with at least xsize (rounded up to
  // kImageAlign bytes) accessible values.
  PIK_INLINE T* PIK_RESTRICT Row(const size_t y) {
    RowBoundsCheck(y);
    void* row = bytes_.get() + y * bytes_per_row_;
    return static_cast<T*>(PIK_ASSUME_ALIGNED(row, 64));
  }

  // Returns pointer to non-const - required for writing to individual planes
  // of an Image3.
  PIK_INLINE T* PIK_RESTRICT MutableRow(const size_t y) const {
    RowBoundsCheck(y);
    void* row = bytes_.get() + y * bytes_per_row_;
    return static_cast<T*>(PIK_ASSUME_ALIGNED(row, 64));
  }

  // Returns pointer to const (see above).
  PIK_INLINE const T* PIK_RESTRICT Row(const size_t y) const {
    RowBoundsCheck(y);
    const void* row = bytes_.get() + y * bytes_per_row_;
    return static_cast<const T*>(PIK_ASSUME_ALIGNED(row, 64));
  }

  // Returns pointer to const (see above), even if called on a non-const Image.
  PIK_INLINE const T* PIK_RESTRICT ConstRow(const size_t y) const {
    return Row(y);
  }

  // Raw access to byte contents, for interfacing with other libraries.
  // Unsigned char instead of char to avoid surprises (sign extension).
  PIK_INLINE uint8_t* PIK_RESTRICT bytes() {
    void* p = bytes_.get();
    return static_cast<uint8_t * PIK_RESTRICT>(PIK_ASSUME_ALIGNED(p, 64));
  }
  PIK_INLINE const uint8_t* PIK_RESTRICT bytes() const {
    const void* p = bytes_.get();
    return static_cast<const uint8_t * PIK_RESTRICT>(PIK_ASSUME_ALIGNED(p, 64));
  }

  // NOTE: do not use this for copying rows - the valid xsize may be much less.
  PIK_INLINE size_t bytes_per_row() const { return bytes_per_row_; }

  // Returns number of pixels (some of which are padding) per row. Useful for
  // computing other rows via pointer arithmetic. WARNING: this must
  // NOT be used to determine xsize. NOTE: this is less efficient than
  // ByteOffset(row, bytes_per_row).
  PIK_INLINE intptr_t PixelsPerRow() const {
    static_assert(kImageAlign % sizeof(T) == 0,
                  "Padding must be divisible by the pixel size.");
    return static_cast<intptr_t>(bytes_per_row_ / sizeof(T));
  }

 private:
  PIK_INLINE void RowBoundsCheck(const size_t y) const {
#if defined(ADDRESS_SANITIZER) || defined(MEMORY_SANITIZER)
    if (y >= ysize_) {
      Abort(__FILE__, __LINE__, "Row(%zu) >= %zu\n", y, ysize_);
    }
#endif
  }

  // (Members are non-const to enable assignment during move-assignment.)
  uint32_t xsize_;  // In valid pixels, not including any padding.
  uint32_t ysize_;
  size_t bytes_per_row_;  // Includes padding.
  CacheAlignedUniquePtr bytes_;
};

using ImageB = Image<uint8_t>;
using ImageS = Image<int16_t>;  // signed integer or half-float
using ImageU = Image<uint16_t>;
using ImageI = Image<int32_t>;
using ImageF = Image<float>;
using ImageD = Image<double>;

// We omit unnecessary fields and choose smaller representations to reduce L1
// cache pollution.
#pragma pack(push, 1)

// Size of an image in pixels. POD.
struct ImageSize {
  static ImageSize Make(const size_t xsize, const size_t ysize) {
    ImageSize ret;
    ret.xsize = static_cast<uint32_t>(xsize);
    ret.ysize = static_cast<uint32_t>(ysize);
    return ret;
  }

  bool operator==(const ImageSize& other) const {
    return xsize == other.xsize && ysize == other.ysize;
  }

  uint32_t xsize;
  uint32_t ysize;
};

#pragma pack(pop)

template <typename T>
void CopyImageTo(const Image<T>& from, Image<T>* PIK_RESTRICT to) {
  PROFILER_ZONE("CopyImage1");
  PIK_ASSERT(SameSize(from, *to));
  for (size_t y = 0; y < from.ysize(); ++y) {
    const T* PIK_RESTRICT row_from = from.ConstRow(y);
    T* PIK_RESTRICT row_to = to->Row(y);
    memcpy(row_to, row_from, from.xsize() * sizeof(T));
  }
}

// DEPRECATED - prefer to preallocate result.
template <typename T>
Image<T> CopyImage(const Image<T>& from) {
  Image<T> to(from.xsize(), from.ysize());
  CopyImageTo(from, &to);
  return to;
}

// Also works for Image3 and mixed argument types.
template <class Image1, class Image2>
bool SameSize(const Image1& image1, const Image2& image2) {
  return image1.xsize() == image2.xsize() && image1.ysize() == image2.ysize();
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

// Use for floating-point images with fairly large numbers; tolerates small
// absolute errors and/or small relative errors. Returns max_relative.
template <typename T>
double VerifyRelativeError(const Image<T>& expected, const Image<T>& actual,
                           const double threshold_l1,
                           const double threshold_relative,
                           const size_t border = 0, const size_t c = 0) {
  PIK_CHECK(SameSize(expected, actual));
  // Max over current scanline to give a better idea whether there are
  // systematic errors or just one outlier. Invalid if negative.
  double max_l1 = -1;
  double max_relative = -1;
  for (size_t y = border; y < expected.ysize() - border; ++y) {
    const T* const PIK_RESTRICT row_expected = expected.Row(y);
    const T* const PIK_RESTRICT row_actual = actual.Row(y);
    bool any_bad = false;
    for (size_t x = border; x < expected.xsize() - border; ++x) {
      const double l1 = std::abs(row_expected[x] - row_actual[x]);

      // Cannot compute relative, only check/update L1.
      if (row_expected[x] < 1E-10) {
        if (l1 > threshold_l1) {
          any_bad = true;
          max_l1 = std::max(max_l1, l1);
        }
      } else {
        const double relative = l1 / std::abs(double(row_expected[x]));
        if (l1 > threshold_l1 && relative > threshold_relative) {
          // Fails both tolerances => will exit below, update max_*.
          any_bad = true;
          max_l1 = std::max(max_l1, l1);
          max_relative = std::max(max_relative, relative);
        }
      }
    }

    if (any_bad) {
      // Never had a valid relative value, don't print it.
      if (max_relative < 0) {
        printf("c=%zu: max +/- %E exceeds +/- %.2E\n", c, max_l1, threshold_l1);
      } else {
        printf("c=%zu: max +/- %E, x %E exceeds +/- %.2E, x %.2E\n", c, max_l1,
               max_relative, threshold_l1, threshold_relative);
      }
      // Find first failing x for further debugging.
      for (size_t x = border; x < expected.xsize() - border; ++x) {
        const double l1 = std::abs(row_expected[x] - row_actual[x]);

        bool bad = l1 > threshold_l1;
        if (row_expected[x] > 1E-10) {
          const double relative = l1 / std::abs(double(row_expected[x]));
          bad &= relative > threshold_relative;
        }
        if (bad) {
          printf("%zu, %zu (%zu x %zu) expected %f actual %f\n", x, y,
                 expected.xsize(), expected.ysize(),
                 static_cast<double>(row_expected[x]),
                 static_cast<double>(row_actual[x]));
          fflush(stdout);
          exit(1);
        }
      }

      PIK_CHECK(false);  // if any_bad, we should have exited.
    }
  }

  return (max_relative < 0) ? 0.0 : max_relative;
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
Image<T> LinComb(const T lambda1, const Image<T>& image1, const T lambda2,
                 const Image<T>& image2) {
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
void FillImage(const T value, Image<T>* image) {
  for (size_t y = 0; y < image->ysize(); ++y) {
    T* const PIK_RESTRICT row = image->Row(y);
    for (size_t x = 0; x < image->xsize(); ++x) {
      row[x] = value;
    }
  }
}

template <typename T>
void ZeroFillImage(Image<T>* image) {
  for (size_t y = 0; y < image->ysize(); ++y) {
    T* const PIK_RESTRICT row = image->Row(y);
    memset(row, 0, image->xsize() * sizeof(T));
  }
}

// Generator for independent, uniformly distributed integers [0, max].
template <typename T, typename Random>
class GeneratorRandom {
 public:
  GeneratorRandom(Random* rng, const T max) : rng_(*rng), dist_(0, max) {}

  GeneratorRandom(Random* rng, const T min, const T max)
      : rng_(*rng), dist_(min, max) {}

  T operator()(const size_t x, const size_t y, const int c) const {
    return dist_(rng_);
  }

 private:
  Random& rng_;
  mutable std::uniform_int_distribution<> dist_;
};

template <typename Random>
class GeneratorRandom<float, Random> {
 public:
  GeneratorRandom(Random* rng, const float max)
      : rng_(*rng), dist_(0.0f, max) {}

  GeneratorRandom(Random* rng, const float min, const float max)
      : rng_(*rng), dist_(min, max) {}

  float operator()(const size_t x, const size_t y, const int c) const {
    return dist_(rng_);
  }

 private:
  Random& rng_;
  mutable std::uniform_real_distribution<float> dist_;
};

template <typename Random>
class GeneratorRandom<double, Random> {
 public:
  GeneratorRandom(Random* rng, const double max)
      : rng_(*rng), dist_(0.0, max) {}

  GeneratorRandom(Random* rng, const double min, const double max)
      : rng_(*rng), dist_(min, max) {}

  double operator()(const size_t x, const size_t y, const int c) const {
    return dist_(rng_);
  }

 private:
  Random& rng_;
  mutable std::uniform_real_distribution<> dist_;
};

// Assigns generator(x, y, 0) to each pixel (x, y).
template <class Generator, class Image>
void GenerateImage(const Generator& generator, Image* image) {
  using T = typename Image::T;
  for (size_t y = 0; y < image->ysize(); ++y) {
    T* const PIK_RESTRICT row = image->Row(y);
    for (size_t x = 0; x < image->xsize(); ++x) {
      row[x] = generator(x, y, 0);
    }
  }
}

// Returns an image with the specified dimensions and random pixels.
template <template <typename> class Image, typename T>
void RandomFillImage(Image<T>* image,
                     const T max = std::numeric_limits<T>::max()) {
  std::mt19937_64 rng(129);
  const GeneratorRandom<T, std::mt19937_64> generator(&rng, max);
  GenerateImage(generator, image);
}

// Returns an image with the specified dimensions and random pixels.
template <template <typename> class Image, typename T>
void RandomFillImage(Image<T>* image, const T min, const T max,
                     const int seed) {
  std::mt19937_64 rng(seed);
  const GeneratorRandom<T, std::mt19937_64> generator(&rng, min, max);
  GenerateImage(generator, image);
}

// Mirrors out of bounds coordinates and returns valid coordinates unchanged.
// We assume the radius (distance outside the image) is small compared to the
// image size, otherwise this might not terminate.
// The mirror is outside the last column (border pixel is also replicated).
static inline int64_t Mirror(int64_t x, const int64_t xsize) {
  // TODO(janwas): replace with branchless version
  while (x < 0 || x >= xsize) {
    if (x < 0) {
      x = -x - 1;
    } else {
      x = 2 * xsize - 1 - x;
    }
  }
  return x;
}

// Wrap modes for ensuring X/Y coordinates are in the valid range [0, size):

// Mirrors (repeating the edge pixel once). Useful for convolutions.
struct WrapMirror {
  PIK_INLINE int64_t operator()(const int64_t coord, const int64_t size) const {
    return Mirror(coord, size);
  }
};

// Repeats the edge pixel.
struct WrapClamp {
  PIK_INLINE int64_t operator()(const int64_t coord, const int64_t size) const {
    return std::min(std::max<int64_t>(0, coord), size - 1);
  }
};

// Returns the same coordinate: required for TFNode with Border(), or useful
// when we know "coord" is already valid (e.g. interior of an image).
struct WrapUnchanged {
  PIK_INLINE int64_t operator()(const int64_t coord, const int64_t size) const {
    return coord;
  }
};

// Similar to Wrap* but for row pointers (reduces Row() multiplications).

class WrapRowMirror {
 public:
  template <class ImageOrView>
  WrapRowMirror(const ImageOrView& image, const size_t ysize)
      : first_row_(image.ConstRow(0)), last_row_(image.ConstRow(ysize - 1)) {}

  const float* const PIK_RESTRICT
  operator()(const float* const PIK_RESTRICT row, const int64_t stride) const {
    if (row < first_row_) {
      const int64_t num_before = first_row_ - row;
      // Mirrored; one row before => row 0, two before = row 1, ...
      return first_row_ + num_before - stride;
    }
    if (row > last_row_) {
      const int64_t num_after = row - last_row_;
      // Mirrored; one row after => last row, two after = last - 1, ...
      return last_row_ - num_after + stride;
    }
    return row;
  }

 private:
  const float* const PIK_RESTRICT first_row_;
  const float* const PIK_RESTRICT last_row_;
};

struct WrapRowUnchanged {
  PIK_INLINE const float* const PIK_RESTRICT
  operator()(const float* const PIK_RESTRICT row, const int64_t stride) const {
    return row;
  }
};

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

// Computes the minimum and maximum pixel value.
template <typename T>
void ImageMinMax(const Image<T>& image, T* const PIK_RESTRICT min,
                 T* const PIK_RESTRICT max) {
  *min = std::numeric_limits<T>::max();
  *max = std::numeric_limits<T>::lowest();
  for (size_t y = 0; y < image.ysize(); ++y) {
    const T* const PIK_RESTRICT row = image.Row(y);
    for (size_t x = 0; x < image.xsize(); ++x) {
      *min = std::min(*min, row[x]);
      *max = std::max(*max, row[x]);
    }
  }
}

// Computes the average pixel value.
template <typename T>
double ImageAverage(const Image<T>& image) {
  double result = 0;
  size_t n = 0;
  for (size_t y = 0; y < image.ysize(); ++y) {
    const T* const PIK_RESTRICT row = image.Row(y);
    for (size_t x = 0; x < image.xsize(); ++x) {
      // Numerically stable method.
      double v = row[x];
      double delta = v - result;
      n++;
      result += delta / n;
    }
  }
  return result;
}

// Copies pixels, scaling their value relative to the "from" min/max by
// "to_range". Example: U8 [0, 255] := [0.0, 1.0], to_range = 1.0 =>
// outputs [0.0, 1.0].
template <typename FromType, typename ToType>
void ImageConvert(const Image<FromType>& from, const float to_range,
                  Image<ToType>* const PIK_RESTRICT to) {
  PIK_ASSERT(SameSize(from, *to));
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

// FromType and ToType are the pixel types.
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

template <typename T>
class Image3;

// Rectangular region in image(s). Factoring this out of Image instead of
// shifting the pointer by x0/y0 allows this to apply to multiple images with
// different resolutions (e.g. color transform and quantization field).
// Can compare using SameSize(rect1, rect2).
class Rect {
 public:
  // Most windows are xsize_max * ysize_max, except those on the borders where
  // begin + size_max > end.
  constexpr Rect(size_t xbegin, size_t ybegin, size_t xsize_max,
                 size_t ysize_max, size_t xend, size_t yend)
      : x0_(xbegin),
        y0_(ybegin),
        xsize_(ClampedSize(xbegin, xsize_max, xend)),
        ysize_(ClampedSize(ybegin, ysize_max, yend)) {}

  // Construct with origin and known size (typically from another Rect).
  constexpr Rect(size_t xbegin, size_t ybegin, size_t xsize, size_t ysize)
      : x0_(xbegin), y0_(ybegin), xsize_(xsize), ysize_(ysize) {}

  // Construct a rect that covers a whole image
  template <typename T>
  explicit Rect(const Image3<T>& image)
      : Rect(0, 0, image.xsize(), image.ysize()) {}
  template <typename T>
  explicit Rect(const Image<T>& image)
      : Rect(0, 0, image.xsize(), image.ysize()) {}

  Rect(const Rect&) = default;
  Rect& operator=(const Rect&) = default;

  Rect Subrect(size_t xbegin, size_t ybegin, size_t xsize_max,
               size_t ysize_max) {
    return Rect(x0_ + xbegin, y0_ + ybegin, xsize_max, ysize_max, x0_ + xsize_,
                y0_ + ysize_);
  }

  template <typename T>
  T* Row(Image<T>* image, size_t y) const {
    return image->Row(y + y0_) + x0_;
  }

  template <typename T>
  T* PlaneRow(Image3<T>* image, const int c, size_t y) const {
    return image->PlaneRow(c, y + y0_) + x0_;
  }

  template <typename T>
  const T* ConstRow(const Image<T>& image, size_t y) const {
    return image.ConstRow(y + y0_) + x0_;
  }

  template <typename T>
  const T* ConstPlaneRow(const Image3<T>& image, const int c, size_t y) const {
    return image.ConstPlaneRow(c, y + y0_) + x0_;
  }

  // Returns true if this Rect fully resides in the given image. ImageT could be
  // Image<T> or Image3<T>; however if ImageT is Rect, results are nonsensical.
  template <class ImageT>
  bool IsInside(const ImageT& image) const {
    return (x0_ + xsize_ <= image.xsize()) && (y0_ + ysize_ <= image.ysize());
  }

  size_t x0() const { return x0_; }
  size_t y0() const { return y0_; }
  size_t xsize() const { return xsize_; }
  size_t ysize() const { return ysize_; }

 private:
  // Returns size_max, or whatever is left in [begin, end).
  static constexpr size_t ClampedSize(size_t begin, size_t size_max,
                                      size_t end) {
    return (begin + size_max <= end) ? size_max : end - begin;
  }

  size_t x0_;
  size_t y0_;

  size_t xsize_;
  size_t ysize_;
};

// Copies `from:rect` to `to`.
template <typename T>
void CopyImageTo(const Rect& rect, const Image<T>& from,
                 Image<T>* PIK_RESTRICT to) {
  PROFILER_ZONE("CopyImageR");
  PIK_ASSERT(SameSize(rect, *to));
  for (size_t y = 0; y < rect.ysize(); ++y) {
    const T* PIK_RESTRICT row_from = rect.ConstRow(from, y);
    T* PIK_RESTRICT row_to = to->Row(y);
    memcpy(row_to, row_from, rect.xsize() * sizeof(T));
  }
}

// DEPRECATED - Returns a copy of the "image" pixels that lie in "rect".
template <typename T>
Image<T> CopyImage(const Rect& rect, const Image<T>& image) {
  Image<T> copy(rect.xsize(), rect.ysize());
  CopyImageTo(rect, image, &copy);
  return copy;
}

// Currently, we abuse Image to either refer to an image that owns its storage
// or one that doesn't. In similar vein, we abuse Image* function parameters to
// either mean "assign to me" or "fill the provided image with data".
// Hopefully, the "assign to me" meaning will go away and most images in the Pik
// codebase will not be backed by own storage. When this happens we can redesign
// Image to be a non-storage-holding view class and introduce BackedImage in
// those places that actually need it.

// NOTE: we can't use Image as a view because invariants are violated
// (alignment and the presence of padding before/after each "row").

// A bundle of 3 same-sized images. Typically constructed by moving from three
// rvalue references to Image. To overwrite an existing Image3 using
// single-channel producers, we also need access to Image*. Constructing
// temporary non-owning Image pointing to one plane of an existing Image3 risks
// dangling references, especially if the wrapper is moved. Therefore, we
// store an array of Image (which are compact enough that size is not a concern)
// and provide a Plane+MutableRow accessors.
template <typename ComponentType>
class Image3 {
 public:
  using T = ComponentType;
  using PlaneT = Image<T>;
  static constexpr size_t kNumPlanes = 3;

  Image3() : planes_{PlaneT(), PlaneT(), PlaneT()} {}

  Image3(const size_t xsize, const size_t ysize)
      : planes_{PlaneT(xsize, ysize), PlaneT(xsize, ysize),
                PlaneT(xsize, ysize)} {}

  Image3(Image3&& other) {
    for (int i = 0; i < kNumPlanes; i++) {
      planes_[i] = std::move(other.planes_[i]);
    }
  }

  Image3(PlaneT&& plane0, PlaneT&& plane1, PlaneT&& plane2) {
    PIK_CHECK(SameSize(plane0, plane1));
    PIK_CHECK(SameSize(plane0, plane2));
    planes_[0] = std::move(plane0);
    planes_[1] = std::move(plane1);
    planes_[2] = std::move(plane2);
  }

  // Copy construction/assignment is forbidden to avoid inadvertent copies,
  // which can be very expensive. Use CopyImageTo instead.
  Image3(const Image3& other) = delete;
  Image3& operator=(const Image3& other) = delete;

  Image3& operator=(Image3&& other) {
    for (int i = 0; i < kNumPlanes; i++) {
      planes_[i] = std::move(other.planes_[i]);
    }
    return *this;
  }

  // Returns row pointer; usage: PlaneRow(idx_plane, y)[x] = val.
  PIK_INLINE T* PIK_RESTRICT PlaneRow(const size_t c, const size_t y) {
    // Custom implementation instead of calling planes_[c].Row ensures only a
    // single multiplication is needed for PlaneRow(0..2, y).
    PlaneRowBoundsCheck(c, y);
    const size_t row_offset = y * planes_[0].bytes_per_row();
    void* row = planes_[c].bytes() + row_offset;
    return static_cast<T*>(PIK_ASSUME_ALIGNED(row, 64));
  }

  // Returns const row pointer; usage: val = PlaneRow(idx_plane, y)[x].
  PIK_INLINE const T* PIK_RESTRICT PlaneRow(const size_t c,
                                            const size_t y) const {
    PlaneRowBoundsCheck(c, y);
    const size_t row_offset = y * planes_[0].bytes_per_row();
    const void* row = planes_[c].bytes() + row_offset;
    return static_cast<const T*>(PIK_ASSUME_ALIGNED(row, 64));
  }

  // Returns const row pointer, even if called from a non-const Image3.
  PIK_INLINE const T* PIK_RESTRICT ConstPlaneRow(const size_t c,
                                                 const size_t y) const {
    return PlaneRow(c, y);
  }

  PIK_INLINE const PlaneT& Plane(size_t idx) const { return planes_[idx]; }

  void Swap(Image3& other) {
    for (int c = 0; c < 3; ++c) {
      other.planes_[c].Swap(planes_[c]);
    }
  }

  void ShrinkTo(const size_t xsize, const size_t ysize) {
    for (PlaneT& plane : planes_) {
      plane.ShrinkTo(xsize, ysize);
    }
  }

  // Sizes of all three images are guaranteed to be equal.
  PIK_INLINE size_t xsize() const { return planes_[0].xsize(); }
  PIK_INLINE size_t ysize() const { return planes_[0].ysize(); }
  // Returns offset [bytes] from one row to the next row of the same plane.
  // WARNING: this must NOT be used to determine xsize, nor for copying rows -
  // the valid xsize may be much less.
  PIK_INLINE size_t bytes_per_row() const { return planes_[0].bytes_per_row(); }
  // Returns number of pixels (some of which are padding) per row. Useful for
  // computing other rows via pointer arithmetic. WARNING: this must NOT be used
  // to determine xsize. NOTE: this is less efficient than
  // ByteOffset(row, bytes_per_row).
  PIK_INLINE intptr_t PixelsPerRow() const { return planes_[0].PixelsPerRow(); }

 private:
  PIK_INLINE void PlaneRowBoundsCheck(const size_t c, const size_t y) const {
#if defined(ADDRESS_SANITIZER) || defined(MEMORY_SANITIZER)
    if (c >= kNumPlanes || y >= ysize()) {
      Abort(__FILE__, __LINE__, "PlaneRow(%zu, %zu) >= %zu\n", c, y, ysize());
    }
#endif
  }

 private:
  PlaneT planes_[kNumPlanes];
};

using Image3B = Image3<uint8_t>;
using Image3S = Image3<int16_t>;
using Image3U = Image3<uint16_t>;
using Image3I = Image3<int32_t>;
using Image3F = Image3<float>;
using Image3D = Image3<double>;

template <typename T>
void CopyImageTo(const Image3<T>& from, Image3<T>* PIK_RESTRICT to) {
  PROFILER_ZONE("CopyImage3");
  PIK_ASSERT(SameSize(from, *to));

  for (size_t c = 0; c < from.kNumPlanes; ++c) {
    for (size_t y = 0; y < from.ysize(); ++y) {
      const T* PIK_RESTRICT row_from = from.ConstPlaneRow(c, y);
      T* PIK_RESTRICT row_to = to->PlaneRow(c, y);
      memcpy(row_to, row_from, from.xsize() * sizeof(T));
    }
  }
}

// DEPRECATED - prefer to preallocate result.
template <typename T>
Image3<T> CopyImage(const Image3<T>& from) {
  Image3<T> copy(from.xsize(), from.ysize());
  CopyImageTo(from, &copy);
  return copy;
}

// DEPRECATED - prefer to preallocate result.
template <typename T>
Image3<T> CopyImage(const Rect& rect, const Image3<T>& from) {
  Image3<T> to(rect.xsize(), rect.ysize());
  CopyImageTo(rect, from.Plane(0), const_cast<ImageF*>(&to.Plane(0)));
  CopyImageTo(rect, from.Plane(1), const_cast<ImageF*>(&to.Plane(1)));
  CopyImageTo(rect, from.Plane(2), const_cast<ImageF*>(&to.Plane(2)));
  return to;
}

template <typename T>
bool SamePixels(const Image3<T>& image1, const Image3<T>& image2) {
  PIK_CHECK(SameSize(image1, image2));
  const size_t xsize = image1.xsize();
  const size_t ysize = image1.ysize();
  for (int c = 0; c < 3; ++c) {
    for (size_t y = 0; y < ysize; ++y) {
      const T* PIK_RESTRICT row1 = image1.PlaneRow(c, y);
      const T* PIK_RESTRICT row2 = image2.PlaneRow(c, y);
      for (size_t x = 0; x < xsize; ++x) {
        if (row1[x] != row2[x]) {
          return false;
        }
      }
    }
  }
  return true;
}

template <typename T>
double VerifyRelativeError(const Image3<T>& expected, const Image3<T>& actual,
                           const float threshold_l1,
                           const float threshold_relative,
                           const size_t border = 0) {
  double max_relative = 0.0;
  for (int c = 0; c < 3; ++c) {
    const double rel =
        VerifyRelativeError(expected.Plane(c), actual.Plane(c), threshold_l1,
                            threshold_relative, border, c);
    max_relative = std::max(max_relative, rel);
  }
  return max_relative;
}

// Sets "thickness" pixels on each border to "value". This is faster than
// initializing the entire image and overwriting valid/interior pixels.
template <typename T>
void SetBorder(const size_t thickness, const T value, Image3<T>* image) {
  const size_t xsize = image->xsize();
  const size_t ysize = image->ysize();
  PIK_ASSERT(2 * thickness < xsize && 2 * thickness < ysize);
  // Top
  for (size_t c = 0; c < 3; ++c) {
    for (size_t y = 0; y < thickness; ++y) {
      T* PIK_RESTRICT row = image->PlaneRow(c, y);
      std::fill(row, row + xsize, value);
    }

    // Bottom
    for (size_t y = ysize - thickness; y < ysize; ++y) {
      T* PIK_RESTRICT row = image->PlaneRow(c, y);
      std::fill(row, row + xsize, value);
    }

    // Left/right
    for (size_t y = thickness; y < ysize - thickness; ++y) {
      T* PIK_RESTRICT row = image->PlaneRow(c, y);
      std::fill(row, row + thickness, value);
      std::fill(row + xsize - thickness, row + xsize, value);
    }
  }
}

// Computes independent minimum and maximum values for each plane.
template <typename T>
void Image3MinMax(const Image3<T>& image, const Rect& rect,
                  std::array<T, 3>* out_min, std::array<T, 3>* out_max) {
  for (int c = 0; c < 3; ++c) {
    T min = std::numeric_limits<T>::max();
    T max = std::numeric_limits<T>::min();
    for (size_t y = 0; y < rect.ysize(); ++y) {
      const T* PIK_RESTRICT row = rect.ConstPlaneRow(image, c, y);
      for (size_t x = 0; x < rect.xsize(); ++x) {
        min = std::min(min, row[x]);
        max = std::max(max, row[x]);
      }
    }
    (*out_min)[c] = min;
    (*out_max)[c] = max;
  }
}

// Computes independent minimum and maximum values for each plane.
template <typename T>
void Image3MinMax(const Image3<T>& image, std::array<T, 3>* out_min,
                  std::array<T, 3>* out_max) {
  Image3MinMax(image, Rect(image), out_min, out_max);
}

template <typename T>
void Image3Max(const Image3<T>& image, std::array<T, 3>* out_max) {
  for (int c = 0; c < 3; ++c) {
    T max = std::numeric_limits<T>::min();
    for (size_t y = 0; y < image.ysize(); ++y) {
      const T* PIK_RESTRICT row = image.ConstPlaneRow(c, y);
      for (size_t x = 0; x < image.xsize(); ++x) {
        max = std::max(max, row[x]);
      }
    }
    (*out_max)[c] = max;
  }
}

template <typename FromType, typename ToType>
void Image3Convert(const Image3<FromType>& from, const float to_range,
                   Image3<ToType>* const PIK_RESTRICT to) {
  PIK_ASSERT(SameSize(from, *to));
  std::array<FromType, 3> min_from, max_from;
  Image3MinMax(from, &min_from, &max_from);
  float scales[3];
  for (int c = 0; c < 3; ++c) {
    scales[c] = to_range / (max_from[c] - min_from[c]);
  }
  float scale = std::min(scales[0], std::min(scales[1], scales[2]));
  for (int c = 0; c < 3; ++c) {
    for (size_t y = 0; y < from.ysize(); ++y) {
      const FromType* PIK_RESTRICT row_from = from.ConstPlaneRow(c, y);
      ToType* PIK_RESTRICT row_to = to->PlaneRow(c, y);
      for (size_t x = 0; x < from.xsize(); ++x) {
        const float to = (row_from[x] - min_from[c]) * scale;
        row_to[x] = static_cast<ToType>(to);
      }
    }
  }
}

// FromType and ToType are the pixel types.
template <typename FromType, typename ToType>
Image3<ToType> StaticCastImage3(const Image3<FromType>& from) {
  Image3<ToType> to(from.xsize(), from.ysize());
  for (int c = 0; c < 3; ++c) {
    for (size_t y = 0; y < from.ysize(); ++y) {
      const FromType* PIK_RESTRICT row_from = from.ConstPlaneRow(c, y);
      ToType* PIK_RESTRICT row_to = to.PlaneRow(c, y);
      for (size_t x = 0; x < from.xsize(); ++x) {
        row_to[x] = static_cast<ToType>(row_from[x]);
      }
    }
  }
  return to;
}

template <typename Tin, typename Tout>
void Subtract(const Image3<Tin>& image1, const Image3<Tin>& image2,
              Image3<Tout>* out) {
  const size_t xsize = image1.xsize();
  const size_t ysize = image1.ysize();
  PIK_CHECK(xsize == image2.xsize());
  PIK_CHECK(ysize == image2.ysize());

  for (int c = 0; c < 3; ++c) {
    for (size_t y = 0; y < ysize; ++y) {
      const Tin* const PIK_RESTRICT row1 = image1.ConstPlaneRow(c, y);
      const Tin* const PIK_RESTRICT row2 = image2.ConstPlaneRow(c, y);
      Tout* const PIK_RESTRICT row_out = out->PlaneRow(c, y);
      for (size_t x = 0; x < xsize; ++x) {
        row_out[x] = row1[x] - row2[x];
      }
    }
  }
}

template <typename Tin, typename Tout>
void SubtractFrom(const Image3<Tin>& what, Image3<Tout>* to) {
  const size_t xsize = what.xsize();
  const size_t ysize = what.ysize();
  for (int c = 0; c < 3; ++c) {
    for (size_t y = 0; y < ysize; ++y) {
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
  for (int c = 0; c < 3; ++c) {
    for (size_t y = 0; y < ysize; ++y) {
      const Tin* PIK_RESTRICT row_what = what.ConstPlaneRow(c, y);
      Tout* PIK_RESTRICT row_to = to->PlaneRow(c, y);
      for (size_t x = 0; x < xsize; ++x) {
        row_to[x] += row_what[x];
      }
    }
  }
}

template <typename T>
Image3<T> ScaleImage(const T lambda, const Image3<T>& image) {
  Image3<T> out(image.xsize(), image.ysize());
  for (size_t c = 0; c < 3; ++c) {
    for (size_t y = 0; y < image.ysize(); ++y) {
      const T* PIK_RESTRICT row = image.ConstPlaneRow(c, y);
      T* PIK_RESTRICT row_out = out.PlaneRow(c, y);
      for (size_t x = 0; x < image.xsize(); ++x) {
        row_out[x] = lambda * row[x];
      }
    }
  }
  return out;
}

// Initializes all planes to the same "value".
template <typename T>
void FillImage(const T value, Image3<T>* image) {
  for (int c = 0; c < 3; ++c) {
    for (size_t y = 0; y < image->ysize(); ++y) {
      T* PIK_RESTRICT row = image->PlaneRow(c, y);
      for (size_t x = 0; x < image->xsize(); ++x) {
        row[x] = value;
      }
    }
  }
}

template <typename T>
void ZeroFillImage(Image3<T>* image) {
  for (int c = 0; c < 3; ++c) {
    for (size_t y = 0; y < image->ysize(); ++y) {
      T* PIK_RESTRICT row = image->PlaneRow(c, y);
      memset(row, 0, image->xsize() * sizeof(T));
    }
  }
}

// Assigns generator(x, y, c) to each pixel (x, y).
template <class Generator, typename T>
void GenerateImage(const Generator& generator, Image3<T>* image) {
  for (int c = 0; c < 3; ++c) {
    for (size_t y = 0; y < image->ysize(); ++y) {
      T* PIK_RESTRICT row = image->PlaneRow(c, y);
      for (size_t x = 0; x < image->xsize(); ++x) {
        row[x] = generator(x, y, c);
      }
    }
  }
}

template <typename T>
std::vector<T> InterleavedFromImage3(const Image3<T>& image3) {
  const size_t xsize = image3.xsize();
  const size_t ysize = image3.ysize();
  std::vector<T> interleaved(xsize * ysize * 3);
  for (size_t y = 0; y < ysize; ++y) {
    const T* PIK_RESTRICT row0 = image3.ConstPlaneRow(0, y);
    const T* PIK_RESTRICT row1 = image3.ConstPlaneRow(1, y);
    const T* PIK_RESTRICT row2 = image3.ConstPlaneRow(2, y);
    T* const PIK_RESTRICT row_interleaved = &interleaved[y * xsize * 3];
    for (size_t x = 0; x < xsize; ++x) {
      row_interleaved[3 * x + 0] = row0[x];
      row_interleaved[3 * x + 1] = row1[x];
      row_interleaved[3 * x + 2] = row2[x];
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
  for (int c = 0; c < 3; ++c) {
    for (size_t y = 0; y < ysize; ++y) {
      T* PIK_RESTRICT row_out = image3.PlaneRow(c, y);
      const T* row_interleaved =
          reinterpret_cast<const T*>(bytes + y * bytes_per_row);
      for (size_t x = 0; x < xsize; ++x) {
        row_out[x] = row_interleaved[3 * x + c];
      }
    }
  }
  return image3;
}

// First, image is padded horizontally, with the rightmost value.
// Next, image is padded vertically, by repeating the last line.
Image3F PadImageToMultiple(const Image3F& in, const size_t N);

}  // namespace pik

#endif  // PIK_IMAGE_H_
