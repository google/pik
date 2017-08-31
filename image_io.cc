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

#include "image_io.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <array>
#include <memory>
#include <utility>
#include <vector>

#include "cache_aligned.h"
#include "compiler_specific.h"
#include "gamma_correct.h"
#include "yuv_convert.h"


extern "C" {
#include "jpeglib.h"
#include "png.h"
}

namespace pik {

// RAII, ensures files are closed even when returning early.
class FileWrapper {
 public:
  FileWrapper(const std::string& pathname, const char* mode)
      : file_(fopen(pathname.c_str(), mode)) {}

  ~FileWrapper() {
    if (file_ != nullptr) {
      const int err = fclose(file_);
      PIK_CHECK(err == 0);
    }
  }

  operator FILE*() const { return file_; }

 private:
  FILE* const file_;
};


bool ReadImage(ImageFormatPNM, const std::string& pathname, ImageB* image) {
  FileWrapper f(pathname, "rb");
  if (f == nullptr) {
    return PIK_FAILURE("File open");
  }

  int mode;
  size_t xsize, ysize;
  const int num_fields =
      fscanf(f, "P%d\n%zu %zu\n255\n", &mode, &xsize, &ysize);
  if (num_fields != 3) {
    return PIK_FAILURE("Read header");
  }
  if (mode != 5) {
    return PIK_FAILURE("Not grayscale");
  }

  *image = ImageB(xsize, ysize);
  size_t bytes_read = 0;
      for (size_t y = 0; y < ysize; ++y) {
        bytes_read += fread(image->Row(y), 1, xsize, f);
      }
      PIK_CHECK(bytes_read == xsize * ysize);
      return true;
}

bool ReadImage(ImageFormatPNM, const std::string& pathname, Image3B* image) {
  FileWrapper f(pathname, "rb");
  if (f == nullptr) {
    return PIK_FAILURE("File open");
  }

  int mode;
  size_t xsize, ysize;
  const int num_fields =
      fscanf(f, "P%d\n%zu %zu\n255\n", &mode, &xsize, &ysize);
  if (num_fields != 3) {
    return PIK_FAILURE("Read header");
  }
  if (mode != 6) {
    return PIK_FAILURE("Not RGB");
  }

  const size_t bytes_per_row = xsize * 3;
  std::vector<uint8_t> interleaved(ysize * bytes_per_row);
  const size_t bytes_read = fread(interleaved.data(), 1, interleaved.size(), f);
  PIK_CHECK(bytes_read == interleaved.size());
  *image =
      Image3FromInterleaved(interleaved.data(), xsize, ysize, bytes_per_row);
  return true;
}

// PGM
bool WriteImage(ImageFormatPNM, const ImageB& image,
                const std::string& pathname) {
  FileWrapper f(pathname, "wb");
  if (f == nullptr) {
    return PIK_FAILURE("File open");
  }

  const int ret =
      fprintf(f, "P5\n%zu %zu\n255\n", image.xsize(), image.ysize());
  PIK_CHECK(ret > 0);

  const std::vector<uint8_t>& packed = PackedFromImage(image);
  const size_t bytes_written = fwrite(packed.data(), 1, packed.size(), f);
  PIK_CHECK(bytes_written == packed.size());
  return true;
}

// PPM
bool WriteImage(ImageFormatPNM, const Image3B& image3,
                const std::string& pathname) {
  FileWrapper f(pathname, "wb");
  if (f == nullptr) {
    return PIK_FAILURE("File open");
  }

  const int ret =
      fprintf(f, "P6\n%zu %zu\n255\n", image3.xsize(), image3.ysize());
  PIK_CHECK(ret > 0);

  const std::vector<uint8_t>& interleaved = InterleavedFromImage3(image3);
  const size_t bytes_written =
      fwrite(interleaved.data(), 1, interleaved.size(), f);
  PIK_CHECK(bytes_written == interleaved.size());
  return true;
}

// Y4M
class Y4MReader {
 public:
  explicit Y4MReader(FILE* f) : f_(f), xsize_(0), ysize_(0) {}

  bool ReadHeader() {
    if (!ReadLine()) {
      return false;
    }
    if (memcmp(line_, "YUV4MPEG2 ", 10)) {
      return false;
    }
    int tag_start;
    int tag_end;
    for (tag_start = 10;;tag_start = tag_end + 1) {
      tag_end = tag_start;
      while (line_[tag_end] != ' ' &&
             line_[tag_end] != '\n' &&
             line_[tag_end] != '\0') {
        ++tag_end;
      }
      if (tag_start < tag_end) {
        // Process tag
        switch (line_[tag_start]) {
          case 'W':
            if (sscanf(&line_[tag_start + 1], "%zd", &xsize_) != 1) {
              return false;
            }
            break;
          case 'H':
            if (sscanf(&line_[tag_start + 1], "%zd", &ysize_) != 1) {
              return false;
            }
            break;
          case 'C':
            if (memcmp(&line_[tag_start], "C444", tag_end - tag_start)) {
              return false;
            }
            break;
          default:
            ; // ignore other tags
        }
      }
      if (line_[tag_end] == '\n' ||
          line_[tag_end] == '\0') {
        break;
      }
    }
    return (xsize_ > 0 && ysize_ > 0);
  }

  void ReadFrame(Image3B* yuv) {
    PIK_CHECK(ReadLine());
    PIK_CHECK(!memcmp(line_, "FRAME", 5));
    *yuv = Image3B(xsize_, ysize_);
    for (int c = 0; c < 3; ++c) {
      for (int y = 0; y < ysize_; ++y) {
        const size_t bytes_read = fread(yuv->Row(y)[c], 1, xsize_, f_);
        PIK_CHECK(bytes_read == xsize_);
      }
    }
  }

 private:
  bool ReadLine() {
    int pos = 0;
    for (; pos < 79; ++pos) {
      int n = fread(&line_[pos], 1, 1, f_);
      if (n != 1) {
        return false;
      }
      if (line_[pos] == '\n') break;
    }
    line_[pos] = '\0';
    return true;
  }

  FILE* f_;
  size_t xsize_;
  size_t ysize_;
  char line_[80];
};

bool ReadImage(ImageFormatY4M, const std::string& pathname, Image3B* image) {
  FileWrapper f(pathname, "rb");
  if (f == nullptr) {
    return PIK_FAILURE("File open");
  }
  Y4MReader reader(f);
  if (!reader.ReadHeader()) {
    return false;
  }
  reader.ReadFrame(image);
  return true;
}

bool WriteImage(ImageFormatY4M, const Image3B& image3,
                const std::string& pathname) {
  FileWrapper f(pathname, "wb");
  if (f == nullptr) {
    return PIK_FAILURE("File open");
  }

  const int ret =
      fprintf(f, "YUV4MPEG2 W%zd H%zd F24:1 Ip A0:0 C444\nFRAME\n",
              image3.xsize(), image3.ysize());
  PIK_CHECK(ret > 0);

  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < image3.ysize(); ++y) {
      const size_t bytes_written =
          fwrite(image3.Row(y)[c], 1, image3.xsize(), f);
      PIK_CHECK(bytes_written == image3.xsize());
    }
  }
  return true;
}

class PngReader {
 public:
  PngReader(const std::string& pathname) : file_(pathname, "rb") {
    png_ = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr,
                                  nullptr);
    if (png_ != nullptr) {
      info_ = png_create_info_struct(png_);
    }
  }

  ~PngReader() { png_destroy_read_struct(&png_, &info_, nullptr); }

  bool ReadHeader(size_t* PIK_RESTRICT xsize, size_t* PIK_RESTRICT ysize,
                  size_t* PIK_RESTRICT num_planes,
                  size_t* PIK_RESTRICT bit_depth) {
    if (png_ == nullptr || info_ == nullptr) {
      return PIK_FAILURE("PNG");
    }

#if PIK_PORTABLE_IO
    if (file_ == nullptr) {
      return PIK_FAILURE("File open");
    }
#endif

    if (setjmp(png_jmpbuf(png_)) != 0) {
      return PIK_FAILURE("PNG");
    }

#if PIK_PORTABLE_IO
    png_init_io(png_, file_);
#endif

    // Convert 1..4 samples to bytes. Resolve palette to RGB(A).
    const unsigned int transforms =
        PNG_TRANSFORM_PACKING | PNG_TRANSFORM_EXPAND;

    png_read_png(png_, info_, transforms, nullptr);

    *xsize = png_get_image_width(png_, info_);
    *ysize = png_get_image_height(png_, info_);
    *bit_depth = png_get_bit_depth(png_, info_);
    *num_planes = png_get_channels(png_, info_);

    return true;
  }

  png_bytep* Rows() const { return png_get_rows(png_, info_); }

 private:
#if PIK_PORTABLE_IO
  FileWrapper file_;
#endif
  png_structp png_ = nullptr;
  png_infop info_ = nullptr;
};

// Reads either a T=uint8_t value or a T=uint16_t value from a stream of 8-bit
// values. In the latter case, an 8-bit value is converted to a 16-bit value by
// replicating the 8-bit value in both the higher and the lower byte.
template <typename T>
T PIK_INLINE ReadFromU8(const uint8_t* const p, const int bias) {
  if (sizeof(T) == 1) {
    return p[0];
  } else {
    return (p[0] << 8) + p[0] - bias;
  }
}

// Reads either a T=uint8_t value or a T=uint16_t value from a stream of 16-bit
// big-endian uint16_t.
template <typename T>
T PIK_INLINE ReadFromU16(const uint8_t* const p, const int bias) {
  if (sizeof(T) == 1) {
    return p[0];
  } else {
    return (p[0] << 8) + p[1] - bias;
  }
}

// bias is 0x8000 when T=int16_t to convert the smallest unsigned value into
// the smallest signed value.
template <typename T>
bool ReadPNGImage(const std::string& pathname, const int bias,
                  Image<T>* image) {
  PngReader reader(pathname);
  size_t xsize, ysize, num_planes, bit_depth;
  if (!reader.ReadHeader(&xsize, &ysize, &num_planes, &bit_depth)) {
    return false;
  }
  if (num_planes != 1) {
    return PIK_FAILURE("Wrong #planes");
  }
  if (bit_depth != 8 && bit_depth != 16) {
    return PIK_FAILURE("Wrong bit-depth");
  }
  *image = Image<T>(xsize, ysize);
  png_bytep* const interleaved_rows = reader.Rows();
  const size_t stride = bit_depth / 8;

  for (size_t y = 0; y < ysize; ++y) {
    const uint8_t* const PIK_RESTRICT interleaved_row = interleaved_rows[y];
    auto row = image->Row(y);
    if (stride == 1) {
      for (size_t x = 0; x < xsize; ++x) {
        row[x] = ReadFromU8<T>(&interleaved_row[x], bias);
      }
    } else {
      for (size_t x = 0; x < xsize; ++x) {
        row[x] = ReadFromU16<T>(&interleaved_row[stride * x], bias);
      }
    }
  }

  return true;
}

// Adds alpha channel to the output image only if a non-opaque pixel is present.
template <typename T>
bool ReadPNGMetaImage(const std::string& pathname, const int bias,
                      MetaImage<T>* image) {
  PngReader reader(pathname);
  size_t xsize, ysize, num_planes, bit_depth;
  if (!reader.ReadHeader(&xsize, &ysize, &num_planes, &bit_depth)) {
    return false;
  }
  if (num_planes < 1 || num_planes > 4) {
    return PIK_FAILURE("Wrong #planes");
  }
  if (bit_depth != 8 && bit_depth != 16) {
    return PIK_FAILURE("Wrong bit-depth");
  }

  image->SetColor(Image3<T>(xsize, ysize));
  png_bytep* const interleaved_rows = reader.Rows();
  const size_t stride = bit_depth / 8;

  // Expand gray -> RGB
  if (num_planes == 1) {
    for (size_t y = 0; y < ysize; ++y) {
      const uint8_t* const PIK_RESTRICT interleaved_row = interleaved_rows[y];
      auto rows = image->GetColor().Row(y);
      if (stride == 1) {
        for (size_t x = 0; x < xsize; ++x) {
          rows[0][x] = rows[1][x] = rows[2][x] =
              ReadFromU8<T>(&interleaved_row[x], bias);
        }
      } else {
        for (size_t x = 0; x < xsize; ++x) {
          rows[0][x] = rows[1][x] = rows[2][x] =
              ReadFromU16<T>(&interleaved_row[stride * x], bias);
        }
      }
    }
  } else if (num_planes == 2) {
    uint16_t alpha_masked = 65535;
    for (size_t y = 0; y < ysize; ++y) {
      const uint8_t* const PIK_RESTRICT interleaved_row = interleaved_rows[y];
      auto rows = image->GetColor().Row(y);
      if (stride == 1) {
        for (size_t x = 0; x < xsize; ++x) {
          rows[0][x] = rows[1][x] = rows[2][x] =
              ReadFromU8<T>(&interleaved_row[2 * x + 0], bias);
          alpha_masked &= interleaved_row[2 * x + 1];
        }
      } else {
        for (size_t x = 0; x < xsize; ++x) {
          rows[0][x] = rows[1][x] = rows[2][x] =
              ReadFromU16<T>(&interleaved_row[stride * (2 * x + 0)], bias);
          alpha_masked &= interleaved_row[2 * x + 1];
        }
      }
    }
    if (alpha_masked != (stride == 1 ? 255 : 65535)) {
      image->AddAlpha();
      for (size_t y = 0; y < ysize; ++y) {
        const uint8_t* const PIK_RESTRICT interleaved_row = interleaved_rows[y];
        auto rows = image->GetAlpha().Row(y);
        if (stride == 1) {
          for (size_t x = 0; x < xsize; ++x) {
            rows[x] = ReadFromU8<T>(&interleaved_row[2 * x + 1], bias);
          }
        } else {
          for (size_t x = 0; x < xsize; ++x) {
            rows[x] =
                ReadFromU16<T>(&interleaved_row[stride * (2 * x + 1)], bias);
          }
        }
      }
    }
  } else if (num_planes == 3) {
    for (size_t y = 0; y < ysize; ++y) {
      const uint8_t* const PIK_RESTRICT interleaved_row = interleaved_rows[y];
      auto rows = image->GetColor().Row(y);
      if (stride == 1) {
        for (size_t x = 0; x < xsize; ++x) {
          rows[0][x] = ReadFromU8<T>(&interleaved_row[3 * x + 0], bias);
          rows[1][x] = ReadFromU8<T>(&interleaved_row[3 * x + 1], bias);
          rows[2][x] = ReadFromU8<T>(&interleaved_row[3 * x + 2], bias);
        }
      } else {
        for (size_t x = 0; x < xsize; ++x) {
          rows[0][x] =
              ReadFromU16<T>(&interleaved_row[stride * (3 * x + 0)], bias);
          rows[1][x] =
              ReadFromU16<T>(&interleaved_row[stride * (3 * x + 1)], bias);
          rows[2][x] =
              ReadFromU16<T>(&interleaved_row[stride * (3 * x + 2)], bias);
        }
      }
    }
  } else /* if (num_planes == 4) */ {
    uint16_t alpha_masked = 65535;
    for (size_t y = 0; y < ysize; ++y) {
      const uint8_t* const PIK_RESTRICT interleaved_row = interleaved_rows[y];
      auto rows = image->GetColor().Row(y);
      if (stride == 1) {
        for (size_t x = 0; x < xsize; ++x) {
          rows[0][x] = ReadFromU8<T>(&interleaved_row[4 * x + 0], bias);
          rows[1][x] = ReadFromU8<T>(&interleaved_row[4 * x + 1], bias);
          rows[2][x] = ReadFromU8<T>(&interleaved_row[4 * x + 2], bias);
          alpha_masked &= interleaved_row[4 * x + 3];
        }
      } else {
        for (size_t x = 0; x < xsize; ++x) {
          rows[0][x] =
              ReadFromU16<T>(&interleaved_row[stride * (4 * x + 0)], bias);
          rows[1][x] =
              ReadFromU16<T>(&interleaved_row[stride * (4 * x + 1)], bias);
          rows[2][x] =
              ReadFromU16<T>(&interleaved_row[stride * (4 * x + 2)], bias);
          alpha_masked &=
              ReadFromU16<T>(&interleaved_row[stride * (4 * x + 3)], bias);
        }
      }
    }
    if (alpha_masked != (stride == 1 ? 255 : 65535)) {
      image->AddAlpha();
      for (size_t y = 0; y < ysize; ++y) {
        const uint8_t* const PIK_RESTRICT interleaved_row = interleaved_rows[y];
        auto rows = image->GetAlpha().Row(y);
        if (stride == 1) {
          for (size_t x = 0; x < xsize; ++x) {
            rows[x] = ReadFromU8<T>(&interleaved_row[4 * x + 3], bias);
          }
        } else {
          for (size_t x = 0; x < xsize; ++x) {
            rows[x] =
                ReadFromU16<T>(&interleaved_row[stride * (4 * x + 3)], bias);
          }
        }
      }
    }
  }

  return true;
}

template <typename T>
bool ReadPNGImage3(const std::string& pathname, const int bias,
                   Image3<T>* image) {
  MetaImage<T> meta;
  if (!ReadPNGMetaImage(pathname, bias, &meta)) {
    return false;
  }
  if (meta.HasAlpha()) {
    return PIK_FAILURE("Translucent PNG not supported");
  }
  *image = std::move(meta.GetColor());
  return true;
}

bool ReadImage(ImageFormatPNG, const std::string& pathname, ImageB* image) {
  return ReadPNGImage(pathname, 0, image);
}

bool ReadImage(ImageFormatPNG, const std::string& pathname, ImageW* image) {
  return ReadPNGImage(pathname, 0x8000, image);
}

bool ReadImage(ImageFormatPNG, const std::string& pathname, ImageU* image) {
  return ReadPNGImage(pathname, 0, image);
}

bool ReadImage(ImageFormatPNG, const std::string& pathname, Image3B* image) {
  return ReadPNGImage3(pathname, 0, image);
}

bool ReadImage(ImageFormatPNG, const std::string& pathname, Image3W* image) {
  return ReadPNGImage3(pathname, 0x8000, image);
}

bool ReadImage(ImageFormatPNG, const std::string& pathname, Image3U* image) {
  return ReadPNGImage3(pathname, 0, image);
}

bool ReadImage(ImageFormatPNG, const std::string& pathname, MetaImageB* image) {
  return ReadPNGMetaImage(pathname, 0, image);
}

bool ReadImage(ImageFormatPNG, const std::string& pathname, MetaImageU* image) {
  return ReadPNGMetaImage(pathname, 0, image);
}

// Allocates an internal buffer for 16-bit pixels in WriteHeader => not
// thread-safe, and cannot reuse for multiple images with different sizes.
class PngWriter {
 public:
  PngWriter(const std::string& pathname) : file_(pathname, "wb") {
    png_ = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr,
                                   nullptr);
    if (png_ != nullptr) {
      info_ = png_create_info_struct(png_);
    }
  }

  ~PngWriter() {
    png_destroy_write_struct(&png_, &info_);
  }

  template <class Image>
  bool WriteHeader(const Image& image) {
    using T = typename Image::T;

    xsize_ = image.xsize();
    if (png_ == nullptr || info_ == nullptr) {
      return PIK_FAILURE("PNG");
    }

#if PIK_PORTABLE_IO
    if (file_ == nullptr) {
      return PIK_FAILURE("File open");
    }
#endif

    if (setjmp(png_jmpbuf(png_)) != 0) {
      return PIK_FAILURE("PNG");
    }

    static const size_t num_planes = GetNumColorPlanes(image);
    if (sizeof(T) != 1 || num_planes != 1 || HasAlpha(image)) {
      row_buffer_.resize((num_planes + (HasAlpha(image) ? 1 : 0))
          * xsize_ * sizeof(T));
    }

#if PIK_PORTABLE_IO
    png_init_io(png_, file_);
#endif

    switch (num_planes) {
      case 1:
        color_type_ = PNG_COLOR_TYPE_GRAY;
        break;
      case 3:
        color_type_ =
            HasAlpha(image) ? PNG_COLOR_TYPE_RGBA : PNG_COLOR_TYPE_RGB;
        break;
      default:
        return PIK_FAILURE("Wrong #planes");
    }

    png_set_IHDR(png_, info_, xsize_, image.ysize(), sizeof(T) * 8, color_type_,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
                 PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png_, info_);
    return true;
  }

  PIK_INLINE void WriteRow(const ImageB& image, const size_t y) {
    // PNG is not const-correct and hopefully won't actually modify row.
    uint8_t* const PIK_RESTRICT row = const_cast<uint8_t*>(image.Row(y));
    png_write_row(png_, row);
  }

  PIK_INLINE void WriteRow(const Image3B& image, const size_t y) {
    const auto rows = image.ConstRow(y);
    for (size_t x = 0; x < xsize_; ++x) {
      row_buffer_[3 * x + 0] = rows[0][x];
      row_buffer_[3 * x + 1] = rows[1][x];
      row_buffer_[3 * x + 2] = rows[2][x];
    }
    png_write_row(png_, row_buffer_.data());
  }

  PIK_INLINE void WriteRow(const MetaImageB& image, const size_t y) {
    const auto rows = image.GetColor().ConstRow(y);
    if (color_type_ == PNG_COLOR_TYPE_RGBA) {
      for (size_t x = 0; x < xsize_; ++x) {
        row_buffer_[4 * x + 0] = rows[0][x];
        row_buffer_[4 * x + 1] = rows[1][x];
        row_buffer_[4 * x + 2] = rows[2][x];
        row_buffer_[4 * x + 3] = image.GetAlpha().Row(y)[x];
      }
    } else {
      for (size_t x = 0; x < xsize_; ++x) {
        row_buffer_[3 * x + 0] = rows[0][x];
        row_buffer_[3 * x + 1] = rows[1][x];
        row_buffer_[3 * x + 2] = rows[2][x];
      }
    }
    png_write_row(png_, row_buffer_.data());
  }

  void WriteRow(const ImageW& image, const size_t y) {
    const auto row = image.ConstRow(y);
    uint8_t* PIK_RESTRICT bytes = row_buffer_.data();
    for (size_t x = 0; x < xsize_; ++x) {
      StoreUnsignedBigEndian(row[x], bytes + 2 * x);
    }
    png_write_row(png_, row_buffer_.data());
  }

  void WriteRow(const Image3W& image, const size_t y) {
    const auto row = image.ConstRow(y);
    uint8_t* PIK_RESTRICT bytes = row_buffer_.data();
    for (size_t x = 0; x < xsize_; ++x) {
      StoreUnsignedBigEndian(row[0][x], bytes + 6 * x + 0);
      StoreUnsignedBigEndian(row[1][x], bytes + 6 * x + 2);
      StoreUnsignedBigEndian(row[2][x], bytes + 6 * x + 4);
    }
    png_write_row(png_, row_buffer_.data());
  }

  void WriteRow(const MetaImageW& image, const size_t y) {
    const auto row = image.GetColor().ConstRow(y);
    uint8_t* PIK_RESTRICT bytes = row_buffer_.data();
    if (color_type_ == PNG_COLOR_TYPE_RGBA) {
      for (size_t x = 0; x < xsize_; ++x) {
        StoreUnsignedBigEndian(row[0][x], bytes + 8 * x + 0);
        StoreUnsignedBigEndian(row[1][x], bytes + 8 * x + 2);
        StoreUnsignedBigEndian(row[2][x], bytes + 8 * x + 4);
        StoreUnsignedBigEndian(image.GetAlpha().Row(y)[x],
                               bytes + 8 * x + 6);
      }
    } else {
      for (size_t x = 0; x < xsize_; ++x) {
        StoreUnsignedBigEndian(row[0][x], bytes + 6 * x + 0);
        StoreUnsignedBigEndian(row[1][x], bytes + 6 * x + 2);
        StoreUnsignedBigEndian(row[2][x], bytes + 6 * x + 4);
      }
    }
    png_write_row(png_, row_buffer_.data());
  }

  void WriteRow(const ImageU& image, const size_t y) {
    const auto row = image.ConstRow(y);
    uint8_t* PIK_RESTRICT bytes = row_buffer_.data();
    for (size_t x = 0; x < xsize_; ++x) {
      StoreUnsignedBigEndian(row[x], bytes + 2 * x);
    }
    png_write_row(png_, row_buffer_.data());
  }

  void WriteRow(const Image3U& image, const size_t y) {
    const auto row = image.ConstRow(y);
    uint8_t* PIK_RESTRICT bytes = row_buffer_.data();
    for (size_t x = 0; x < xsize_; ++x) {
      StoreUnsignedBigEndian(row[0][x], bytes + 6 * x + 0);
      StoreUnsignedBigEndian(row[1][x], bytes + 6 * x + 2);
      StoreUnsignedBigEndian(row[2][x], bytes + 6 * x + 4);
    }
    png_write_row(png_, row_buffer_.data());
  }

  void WriteRow(const MetaImageU& image, const size_t y) {
    const auto row = image.GetColor().ConstRow(y);
    const int stride = color_type_ == PNG_COLOR_TYPE_RGBA ? 8 : 6;
    uint8_t* PIK_RESTRICT bytes = row_buffer_.data();
    for (size_t x = 0; x < xsize_; ++x) {
      StoreUnsignedBigEndian(row[0][x], bytes + stride * x + 0);
      StoreUnsignedBigEndian(row[1][x], bytes + stride * x + 2);
      StoreUnsignedBigEndian(row[2][x], bytes + stride * x + 4);
      if (color_type_ == PNG_COLOR_TYPE_RGBA) {
        StoreUnsignedBigEndian(
            image.GetAlpha().Row(y)[x], bytes + stride * x + 6);
      }
    }
    png_write_row(png_, row_buffer_.data());
  }

  // Only call if WriteHeader succeeded.
  void WriteEnd() { png_write_end(png_, nullptr); }

 private:
  void StoreUnsignedBigEndian(const int16_t value, uint8_t* bytes) {
    const unsigned unsigned_value = value + 0x8000;  // [0, 0x10000)
    bytes[0] = unsigned_value >> 8;
    bytes[1] = unsigned_value & 0xFF;
  }

  void StoreUnsignedBigEndian(const uint16_t unsigned_value, uint8_t* bytes) {
    bytes[0] = unsigned_value >> 8;
    bytes[1] = unsigned_value & 0xFF;
  }

  template<typename T>
  static bool HasAlpha(const MetaImage<T>& image) {
    return image.HasAlpha();
  }

  template<typename T>
  static bool HasAlpha(const Image<T>& image) {
    return false;
  }

  template<typename T>
  static bool HasAlpha(const Image3<T>& image) {
    return false;
  }

  // Excludes the alpha channel in the count.
  template<typename T>
  static size_t GetNumColorPlanes(const MetaImage<T>& image) {
    return Image3<T>::kNumPlanes;
  }

  template<typename T>
  static size_t GetNumColorPlanes(const Image<T>& image) {
    return Image<T>::kNumPlanes;
  }

  template<typename T>
  static size_t GetNumColorPlanes(const Image3<T>& image) {
    return Image3<T>::kNumPlanes;
  }

#if PIK_PORTABLE_IO
  FileWrapper file_;
#endif
  size_t xsize_;
  png_structp png_ = nullptr;
  png_infop info_ = nullptr;
  int color_type_ = 0;

  // Buffer for byte-swapping 16-bit pixels, allocated in WriteHeader.
  std::vector<uint8_t> row_buffer_;
};

template <class Image>
bool WritePNGImage(const Image& image, const std::string& pathname) {
  PngWriter png(pathname);
  if (!png.WriteHeader(image)) {
    return false;
  }
  for (size_t y = 0; y < image.ysize(); ++y) {
    png.WriteRow(image, y);
  }
  png.WriteEnd();
  return true;
}

bool WriteImage(ImageFormatPNG, const ImageB& image,
                const std::string& pathname) {
  return WritePNGImage(image, pathname);
}

bool WriteImage(ImageFormatPNG, const ImageW& image,
                const std::string& pathname) {
  return WritePNGImage(image, pathname);
}

bool WriteImage(ImageFormatPNG, const ImageU& image,
                const std::string& pathname) {
  return WritePNGImage(image, pathname);
}

bool WriteImage(ImageFormatPNG, const Image3B& image3,
                const std::string& pathname) {
  return WritePNGImage(image3, pathname);
}

bool WriteImage(ImageFormatPNG, const Image3W& image3,
                const std::string& pathname) {
  return WritePNGImage(image3, pathname);
}

bool WriteImage(ImageFormatPNG, const Image3U& image3,
                const std::string& pathname) {
  return WritePNGImage(image3, pathname);
}

bool WriteImage(ImageFormatPNG, const MetaImageB& image,
                const std::string& pathname) {
  return WritePNGImage(image, pathname);
}

bool WriteImage(ImageFormatPNG, const MetaImageU& image,
                const std::string& pathname) {
  return WritePNGImage(image, pathname);
}

void jpeg_catch_error(j_common_ptr cinfo) {
  (*cinfo->err->output_message)(cinfo);
  jmp_buf* jpeg_jmpbuf = (jmp_buf*)cinfo->client_data;
  jpeg_destroy(cinfo);
  longjmp(*jpeg_jmpbuf, 1);
}

// Can be either a filename or a memory buffer.
struct JpegInput {
  explicit JpegInput(const std::string& fn) : filename(fn) {}
  JpegInput(const uint8_t* buf, size_t len) : inbuffer(buf), insize(len) {}
  std::string filename;
  const uint8_t* inbuffer = nullptr;
  size_t insize = 0;
};

bool ReadJpegImage(const JpegInput& input, Image3B* rgb) {
  std::unique_ptr<FileWrapper> input_file;
  if (!input.filename.empty()) {
    input_file.reset(new FileWrapper(input.filename, "rb"));
    if (input_file.get() == nullptr) {
      return PIK_FAILURE("File open");
    }
  } else if (input.inbuffer == nullptr) {
    return PIK_FAILURE("Invalid JpegInput.");
  }

  jpeg_decompress_struct cinfo;
  jpeg_error_mgr jerr;
  cinfo.err = jpeg_std_error(&jerr);
  cinfo.do_fancy_upsampling = true;

  jmp_buf jpeg_jmpbuf;
  cinfo.client_data = &jpeg_jmpbuf;
  jerr.error_exit = jpeg_catch_error;
  if (setjmp(jpeg_jmpbuf)) {
    return false;
  }

  jpeg_create_decompress(&cinfo);

  if (input_file.get() != nullptr) {
    jpeg_stdio_src(&cinfo, *input_file);
  } else {
    // Libjpeg versions before 9b used a non-const buffer here.
    jpeg_mem_src(&cinfo, const_cast<uint8_t*>(input.inbuffer), input.insize);
  }

  jpeg_read_header(&cinfo, TRUE);
  jpeg_start_decompress(&cinfo);

  const size_t xsize = cinfo.output_width;
  const size_t ysize = cinfo.output_height;
  const size_t row_stride = xsize * cinfo.output_components;
  const size_t buf_size = row_stride * ysize;
  CacheAlignedUniquePtr buffer = AllocateArray(buf_size);
  uint8_t* const imagep = buffer.get();
  JSAMPLE* output_line = static_cast<JSAMPLE*>(imagep);

  *rgb = Image3B(xsize, ysize);

  switch (cinfo.out_color_space) {
    case JCS_GRAYSCALE:
      while (cinfo.output_scanline < cinfo.output_height) {
        jpeg_read_scanlines(&cinfo, &output_line, 1);
        output_line += row_stride;
      }

      for (int y = 0; y < ysize; ++y) {
        const uint8_t* const PIK_RESTRICT row = &imagep[y * row_stride];
        auto rows = rgb->Row(y);

        for (int x = 0; x < xsize; x++) {
          const uint8_t gray = row[x];
          rows[0][x] = rows[1][x] = rows[2][x] = gray;
        }
      }
      break;

    case JCS_RGB:
      while (cinfo.output_scanline < cinfo.output_height) {
        jpeg_read_scanlines(&cinfo, &output_line, 1);
        output_line += row_stride;
      }

      for (int y = 0; y < ysize; ++y) {
        const uint8_t* const PIK_RESTRICT row = &imagep[y * row_stride];
        auto rows = rgb->Row(y);
        for (int x = 0; x < xsize; x++) {
          rows[0][x] = row[3 * x + 0];
          rows[1][x] = row[3 * x + 1];
          rows[2][x] = row[3 * x + 2];
        }
      }
      break;

    default:
      return PIK_FAILURE("Unsupported color space");
  }

  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);
  return true;
}

bool ReadImage(ImageFormatJPG, const std::string& pathname, Image3B* rgb) {
  JpegInput input(pathname);
  return ReadJpegImage(input, rgb);
}

bool ReadImage(ImageFormatJPG, const uint8_t* buf, size_t size, Image3B* rgb) {
  JpegInput input(buf, size);
  return ReadJpegImage(input, rgb);
}

// Planes

struct PlanesHeader {
  static char CharFromType(uint8_t) { return 'B'; }
  static char CharFromType(int16_t) { return 'W'; }
  static char CharFromType(uint16_t) { return 'U'; }
  static char CharFromType(float) { return 'F'; }

  size_t ComponentSize() const {
    switch (type) {
      case 'B':
        return sizeof(uint8_t);
      case 'W':
        return sizeof(int16_t);
      case 'U':
        return sizeof(uint16_t);
      case 'F':
        return sizeof(float);
      default:
        return 0;
    }
  }

  size_t xsize;
  size_t ysize;
  size_t bytes_per_row;
  size_t num_planes;
  char type;  // see CharFromType
};

// Encodes integers as human-readable text. The input/output buffers must
// include kMaxChars bytes of padding per call to Encode/Decode.
class FieldCoder {
 public:
  // Also hard-coded into the format specifier below.
  static constexpr size_t kMaxChars = 30;  // includes null terminator

  static bool Encode(const size_t field, char** pos) {
    char buf[kMaxChars];
    const int bytes_written = snprintf(buf, sizeof(buf), "%zu ", field);
    if (bytes_written <= 0) {
      return PIK_FAILURE("Encoding failed");
    }
    if (bytes_written > kMaxChars) {
      return PIK_FAILURE("Length exceeded");
    }
    memcpy(*pos, buf, bytes_written + 1);  // includes null terminator
    *pos += bytes_written;
    return true;
  }

  static bool Decode(char** pos, size_t* field) {
    int bytes_read = 0;
    const int num_FieldCoder = sscanf(*pos, "%30zu %n", field, &bytes_read);
    if (num_FieldCoder != 1) {
      return PIK_FAILURE("Decoding failed");
    }
    PIK_CHECK(bytes_read > 0);
    if (bytes_read > kMaxChars) {
      return PIK_FAILURE("Length exceeded");
    }
    *pos += bytes_read;
    return true;
  }
};

// Reads/writes headers from/to file. Pre/postcondition: PlanesHeader passes a
// basic sanity check.
class HeaderIO {
  static constexpr size_t kSize = 64;
  static constexpr size_t kPaddedSize = kSize + 4 * FieldCoder::kMaxChars;

 public:
  static bool Read(const FileWrapper& f, PlanesHeader* header) {
    if (f == nullptr) {
      return PIK_FAILURE("File open");
    }

    char storage[kPaddedSize] = {0};
    const size_t bytes_read = fread(storage, 1, kSize, f);
    if (bytes_read != kSize) {
      return PIK_FAILURE("Read header");
    }

    if (memcmp(storage, Signature(), 4) != 0) {
      return PIK_FAILURE("Signature mismatch");
    }
    header->type = storage[4];
    char* pos = storage + 5;
    if (!FieldCoder::Decode(&pos, &header->xsize) ||
        !FieldCoder::Decode(&pos, &header->ysize) ||
        !FieldCoder::Decode(&pos, &header->num_planes) ||
        !FieldCoder::Decode(&pos, &header->bytes_per_row)) {
      return false;
    }
    if (pos > storage + kSize) {
      return PIK_FAILURE("Header size exceeded");
    }
    if (!SanityCheck(*header)) {
      return false;
    }
    return true;
  }

  static bool Write(const PlanesHeader& header, FileWrapper* f) {
    PIK_CHECK(SanityCheck(header));

    if (f == nullptr) {
      return PIK_FAILURE("File open");
    }

    char storage[kPaddedSize] = {0};
    memcpy(storage, Signature(), 4);
    storage[4] = header.type;
    char* pos = storage + 5;
    if (!FieldCoder::Encode(header.xsize, &pos) ||
        !FieldCoder::Encode(header.ysize, &pos) ||
        !FieldCoder::Encode(header.num_planes, &pos) ||
        !FieldCoder::Encode(header.bytes_per_row, &pos)) {
      return false;
    }
    if (pos > storage + kSize) {
      return PIK_FAILURE("Header size exceeded");
    }

    const size_t bytes_written = fwrite(storage, 1, kSize, *f);
    if (bytes_written != kSize) {
      return PIK_FAILURE("Write header");
    }
    return true;
  }

 private:
  static const char* Signature() {
    // 4 byte signature. Newline makes it obvious if the files are opened in
    // text mode; a non-ASCII character detects 8-bit transmission errors.
    return "PL\xA5\n";
  }

  // Returns false if "header" is definitely invalid.
  static bool SanityCheck(const PlanesHeader& header) {
    if (header.xsize == 0 || header.ysize == 0 || header.num_planes == 0) {
      return PIK_FAILURE("Zero dimension");
    }
    const size_t component_size = header.ComponentSize();
    if (component_size == 0) {
      return PIK_FAILURE("Invalid type");
    }
    if (header.bytes_per_row < header.xsize * component_size) {
      return PIK_FAILURE("Insufficient row size");
    }
    return true;
  }
};

namespace {
// Loads header from file and returns one or more 2D arrays.
CacheAlignedUniquePtr LoadPlanes(const std::string& pathname,
                                 PlanesHeader* header) {
  CacheAlignedUniquePtr null(nullptr, CacheAligned::Free);

  FileWrapper f(pathname, "rb");
  if (!HeaderIO::Read(f, header)) {
    return null;
  }

  const size_t size =
      header->bytes_per_row * header->ysize * header->num_planes;
  CacheAlignedUniquePtr planes = AllocateArray(size);

  const size_t bytes_read = fread(planes.get(), 1, size, f);
  if (bytes_read != size) {
    PIK_NOTIFY_ERROR("Read planes");
    return null;
  }
  return planes;
}

// Stores "header" and "planes" (of any type) to file.
bool StorePlanes(const PlanesHeader& header,
                 const std::vector<const uint8_t*>& planes,
                 const std::string& pathname) {
  PIK_CHECK(header.num_planes == planes.size());

  FileWrapper f(pathname, "wb");
  if (!HeaderIO::Write(header, &f)) {
    return false;
  }

  const size_t plane_size = header.ysize * header.bytes_per_row;
  for (const uint8_t* plane : planes) {
    const size_t bytes_written = fwrite(plane, 1, plane_size, f);
    if (bytes_written != plane_size) {
      return PIK_FAILURE("Write planes");
    }
  }

  return true;
}
}  // namespace

template <typename T>
bool ReadImage(ImageFormatPlanes, const std::string& pathname,
               Image<T>* image) {
  PlanesHeader header;
  CacheAlignedUniquePtr storage = LoadPlanes(pathname, &header);
  if (storage == nullptr) {
    return false;
  }

  if (header.type != PlanesHeader::CharFromType(T())) {
    return PIK_FAILURE("Type mismatch");
  }

  // Takes ownership.
  *image = Image<T>(header.xsize, header.ysize, std::move(storage),
                    header.bytes_per_row);
  return true;
}

template <typename T>
bool ReadImage(ImageFormatPlanes, const std::string& pathname,
               Image3<T>* image) {
  PlanesHeader header;
  CacheAlignedUniquePtr storage = LoadPlanes(pathname, &header);
  if (storage == nullptr) {
    return false;
  }

  if (header.type != PlanesHeader::CharFromType(T())) {
    return PIK_FAILURE("Type mismatch");
  }

  // All but the first plane refer to the same storage.
  uint8_t* planes = storage.get();
  const size_t plane_size = header.ysize * header.bytes_per_row;

  // The first plane takes ownership.
  Image<T> plane0(header.xsize, header.ysize, std::move(storage),
                  header.bytes_per_row);
  Image<T> plane1(header.xsize, header.ysize, planes + 1 * plane_size,
                  header.bytes_per_row);
  Image<T> plane2(header.xsize, header.ysize, planes + 2 * plane_size,
                  header.bytes_per_row);

  *image = Image3<T>(std::move(plane0), std::move(plane1), std::move(plane2));
  return true;
}

template <typename T>
bool WriteImage(ImageFormatPlanes, const Image<T>& image,
                const std::string& pathname) {
  PlanesHeader header;
  header.xsize = image.xsize();
  header.ysize = image.ysize();
  header.num_planes = 1;
  header.bytes_per_row = image.bytes_per_row();
  header.type = PlanesHeader::CharFromType(T());

  const std::vector<const uint8_t*> plane_ptrs{image.bytes()};
  return StorePlanes(header, plane_ptrs, pathname);
}

template <typename T>
bool WriteImage(ImageFormatPlanes, const Image3<T>& image,
                const std::string& pathname) {
  PlanesHeader header;
  header.xsize = image.xsize();
  header.ysize = image.ysize();
  header.num_planes = 3;
  header.bytes_per_row = image.plane(0).bytes_per_row();
  header.type = PlanesHeader::CharFromType(T());

  const std::vector<const uint8_t*> plane_ptrs{
      image.plane(0).bytes(), image.plane(1).bytes(), image.plane(2).bytes()};
  return StorePlanes(header, plane_ptrs, pathname);
}

namespace {

// Case-specific (filename may be UTF-8)
bool EndsWith(const char* name, const char* suffix) {
  const size_t name_len = strlen(name);
  const size_t suffix_len = strlen(suffix);
  if (name_len < suffix_len) return false;
  return memcmp(name + name_len - suffix_len, suffix, suffix_len) == 0;
}

}  // namespace

// ImageFormat* cannot return the extension because some formats have several.
bool ImageFormatPNM::IsExtension(const char* filename) {
  return EndsWith(filename, "pgm") || EndsWith(filename, "ppm");
}

bool ImageFormatPNG::IsExtension(const char* filename) {
  return EndsWith(filename, "png");
}

bool ImageFormatY4M::IsExtension(const char* filename) {
  return EndsWith(filename, "y4m");
}

bool ImageFormatJPG::IsExtension(const char* filename) {
  return EndsWith(filename, "jpg") || EndsWith(filename, "jpeg");
}

bool ImageFormatPlanes::IsExtension(const char* filename) {
  return EndsWith(filename, "planes");
}

// Returns true when the visitor returns true.
template <class Visitor>
bool VisitFormats(Visitor* visitor) {
  if ((*visitor)(ImageFormatPNM())) return true;
  if ((*visitor)(ImageFormatPNG())) return true;
  if ((*visitor)(ImageFormatY4M())) return true;
  if ((*visitor)(ImageFormatJPG())) return true;
  if ((*visitor)(ImageFormatPlanes())) return true;
  return false;
}

// Returns true if the given file was loaded and converted to linear RGB.
// Called via VisitFormats. To avoid opening and loading the file multiple
// times, we first attempt to detect the format via file extension.
class LinearLoader {
 public:
  LinearLoader(const std::string& pathname, MetaImageF* linear_rgb)
      : pathname_(pathname), linear_rgb_(linear_rgb) {}

  void DisregardExtensions() { check_extension_ = false; }

  template <class Format>
  bool operator()(const Format format) {
    if (check_extension_ && !Format::IsExtension(pathname_.c_str())) {
      return false;
    }

    typename Format::NativeImage3 image;
    if (!ReadImage(format, pathname_, &image)) {
      return false;
    }

    ConvertToLinearRGB(format, std::move(image));
    return true;
  }

 private:
  // Common case: from sRGB bytes
  template <class Format>
  void ConvertToLinearRGB(Format, const Image3B& bytes) {
    linear_rgb_->SetColor(LinearFromSrgb(bytes));
  }

  // From YUV bytes
  void ConvertToLinearRGB(ImageFormatY4M, const Image3B& bytes) {
    linear_rgb_->SetColor(RGBLinearImageFromYUVRec709(bytes));
  }

  // From 16-bit sRGB
  template <class Format>
  void ConvertToLinearRGB(Format, const Image3U& srgb) {
    linear_rgb_->SetColor(Image3F(srgb.xsize(), srgb.ysize()));
    for (size_t y = 0; y < srgb.ysize(); ++y) {
      auto row_rgb = srgb.Row(y);
      auto row_lin = linear_rgb_->GetColor().Row(y);
      for (size_t x = 0; x < srgb.xsize(); ++x) {
        for (int c = 0; c < 3; ++c) {
          // Dividing the 16 value by 257 scales it to the [0.0, 255.0]
          // interval. If the PNG was 8-bit, this has the same effect as
          // casting the original 8-bit value to a float.
          row_lin[c][x] = Srgb8ToLinearDirect(row_rgb[c][x] / 257.0f);
        }
      }
    }
  }

  // From 16-bit signed
  template <class Format>
  void ConvertToLinearRGB(Format format, const MetaImageU& srgb) {
    ConvertToLinearRGB(format, srgb.GetColor());
    if (srgb.HasAlpha()) {
      linear_rgb_->AddAlpha();
      for (size_t y = 0; y < srgb.ysize(); ++y) {
        const uint16_t* const PIK_RESTRICT row_from = srgb.GetAlpha().Row(y);
        float* const PIK_RESTRICT row_to = linear_rgb_->GetAlpha().Row(y);
        for (size_t x = 0; x < srgb.xsize(); ++x) {
          row_to[x] = (row_from[x] / 257.0f);
        }
      }
    }
  }

  // From 16-bit signed
  template <class Format>
  void ConvertToLinearRGB(Format, const Image3W& srgb) {
    linear_rgb_->SetColor(Image3F(srgb.xsize(), srgb.ysize()));
    for (size_t y = 0; y < srgb.ysize(); ++y) {
      auto row_rgb = srgb.Row(y);
      auto row_lin = linear_rgb_->GetColor().Row(y);
      for (size_t x = 0; x < srgb.xsize(); ++x) {
        for (int c = 0; c < 3; ++c) {
          const int unsigned_value = row_rgb[c][x] + 0x8000;  // [0, 0x10000)
          row_lin[c][x] = Srgb8ToLinearDirect(unsigned_value / 257.0);
        }
      }
    }
  }

  // From 16-bit signed
  template <class Format>
  void ConvertToLinearRGB(Format format, const MetaImageW& srgb) {
    ConvertToLinearRGB(format, srgb.GetColor());
    if (srgb.HasAlpha()) {
      linear_rgb_->AddAlpha();
      for (size_t y = 0; y < srgb.ysize(); ++y) {
        const int16_t* const PIK_RESTRICT row_from = srgb.GetAlpha().Row(y);
        float* const PIK_RESTRICT row_to = linear_rgb_->GetAlpha().Row(y);
        for (size_t x = 0; x < srgb.xsize(); ++x) {
          row_to[x] = (static_cast<uint16_t>(row_from[x] / 257.0f));
        }
      }
    }
  }

  // From linear float (zero-copy)
  template <class Format>
  void ConvertToLinearRGB(Format, Image3F&& linear) {
    linear_rgb_->SetColor(std::move(linear));
  }

  // From linear float (zero-copy)
  template <class Format>
  void ConvertToLinearRGB(Format, MetaImageF&& linear) {
    *linear_rgb_ = std::move(linear);
  }

  const std::string pathname_;
  MetaImageF* linear_rgb_;
  bool check_extension_ = true;
};

MetaImageF ReadMetaImageLinear(const std::string& pathname) {
  MetaImageF linear_rgb;
  LinearLoader loader(pathname, &linear_rgb);

  // First round: only attempt to load if extension matches.
  if (VisitFormats(&loader)) {
    return linear_rgb;
  }

  // Let each format load, regardless of extension.
  loader.DisregardExtensions();
  if (VisitFormats(&loader)) {
    return linear_rgb;
  }

  PIK_NOTIFY_ERROR("Unsupported file format");
  return linear_rgb;
}

Image3F ReadImage3Linear(const std::string& pathname) {
  MetaImageF meta = ReadMetaImageLinear(pathname);
  if (meta.HasAlpha()) {
    PIK_NOTIFY_ERROR("Alpha channel not supported");
  }
  return std::move(meta.GetColor());
}

template bool ReadImage<uint8_t>(ImageFormatPlanes, const std::string&,
                                 ImageB*);
template bool ReadImage<int16_t>(ImageFormatPlanes, const std::string&,
                                 ImageW*);
template bool ReadImage<uint16_t>(ImageFormatPlanes, const std::string&,
                                  ImageU*);
template bool ReadImage<float>(ImageFormatPlanes, const std::string&, ImageF*);

template bool ReadImage<uint8_t>(ImageFormatPlanes, const std::string&,
                                 Image3B*);
template bool ReadImage<int16_t>(ImageFormatPlanes, const std::string&,
                                 Image3W*);
template bool ReadImage<uint16_t>(ImageFormatPlanes, const std::string&,
                                  Image3U*);
template bool ReadImage<float>(ImageFormatPlanes, const std::string&, Image3F*);

template bool WriteImage<uint8_t>(ImageFormatPlanes, const ImageB&,
                                  const std::string&);
template bool WriteImage<int16_t>(ImageFormatPlanes, const ImageW&,
                                  const std::string&);
template bool WriteImage<uint16_t>(ImageFormatPlanes, const ImageU&,
                                   const std::string&);
template bool WriteImage<float>(ImageFormatPlanes, const ImageF&,
                                const std::string&);

template bool WriteImage<uint8_t>(ImageFormatPlanes, const Image3B&,
                                  const std::string&);
template bool WriteImage<int16_t>(ImageFormatPlanes, const Image3W&,
                                  const std::string&);
template bool WriteImage<uint16_t>(ImageFormatPlanes, const Image3U&,
                                   const std::string&);
template bool WriteImage<float>(ImageFormatPlanes, const Image3F&,
                                const std::string&);

}  // namespace pik
