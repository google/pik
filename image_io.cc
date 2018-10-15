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
#include <algorithm>
#include <array>
#include <memory>
#include <utility>
#include <vector>

#include "third_party/lodepng/lodepng.h"
#include "bits.h"
#include "byte_order.h"
#include "cache_aligned.h"
#include "common.h"
#include "compiler_specific.h"
#include "file_io.h"
#include "gamma_correct.h"
#include "yuv_convert.h"

#define ENABLE_JPEG 0

extern "C" {
#if ENABLE_JPEG
#include "jpeglib.h"
#endif
}

#ifdef MEMORY_SANITIZER
#include <sanitizer/msan_interface.h>
#endif

namespace pik {
namespace {

static size_t PlanesFromPNGType(const LodePNGColorType type) {
  switch (type) {
    case LCT_GREY:
      return 1;
    case LCT_GREY_ALPHA:
      return 2;
    case LCT_RGB:
      return 3;
    case LCT_RGBA:
    case LCT_PALETTE:
      return 4;
    default:
      PIK_NOTIFY_ERROR("Unknown color mode");
      return 1;
  }
}

// Case-specific (filename may be UTF-8)
bool EndsWith(const char* name, const char* suffix) {
  const size_t name_len = strlen(name);
  const size_t suffix_len = strlen(suffix);
  if (name_len < suffix_len) return false;
  return memcmp(name + name_len - suffix_len, suffix, suffix_len) == 0;
}

}  // namespace

bool ReadImage(ImageFormatPNM, const std::string& pathname, ImageB* image) {
  FileWrapper f(pathname, "rb");
  if (f == nullptr) {
    return PIK_FAILURE("File open");
  }

  int mode;
  size_t xsize, ysize;
  const int num_fields = fscanf(f, "P%d\n%zu %zu\n255", &mode, &xsize, &ysize);
  // TODO(janwas): avoid fscanf because it can consume multiple whitespace
  // chars; also allow comments.
  char c = getc(f);
  if (num_fields != 3 || !isspace(c)) {
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
  const int num_fields = fscanf(f, "P%d\n%zu %zu\n255", &mode, &xsize, &ysize);
  // TODO(janwas): avoid fscanf because it can consume multiple whitespace
  // chars; also allow comments.
  char c = getc(f);
  if (num_fields != 3 || !isspace(c)) {
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

  int bit_depth() const { return bit_depth_; }

  Status ReadHeader() {
    PIK_RETURN_IF_ERROR(ReadLine());
    if (memcmp(line_, "YUV4MPEG2 ", 10)) {
      return PIK_FAILURE("Invalid Y4M signature");
    }
    int tag_start;
    int tag_end;
    for (tag_start = 10;; tag_start = tag_end + 1) {
      tag_end = tag_start;
      while (line_[tag_end] != ' ' && line_[tag_end] != '\n' &&
             line_[tag_end] != '\0') {
        ++tag_end;
      }
      if (tag_start < tag_end) {
        const int tag_len = tag_end - tag_start;
        // Process tag
        switch (line_[tag_start]) {
          case 'W':
            if (sscanf(&line_[tag_start + 1], "%zd", &xsize_) != 1) {
              return PIK_FAILURE("Invalid width");
            }
            break;
          case 'H':
            if (sscanf(&line_[tag_start + 1], "%zd", &ysize_) != 1) {
              return PIK_FAILURE("Invalid height");
            }
            break;
          case 'C':
            if (tag_len == 4 && !memcmp(&line_[tag_start], "C444", tag_len)) {
              bit_depth_ = 8;
              chroma_subsample_ = false;
            } else if (tag_len == 7 &&
                       !memcmp(&line_[tag_start], "C444p10", tag_len)) {
              bit_depth_ = 10;
              chroma_subsample_ = false;
            } else if (tag_len == 7 &&
                       !memcmp(&line_[tag_start], "C444p12", tag_len)) {
              bit_depth_ = 12;
              chroma_subsample_ = false;
            } else if ((tag_len == 4 &&
                        !memcmp(&line_[tag_start], "C420", tag_len)) ||
                       (tag_len == 8 &&
                        !memcmp(&line_[tag_start], "C420jpeg", tag_len))) {
              bit_depth_ = 8;
              chroma_subsample_ = true;
            } else if (tag_len == 7 &&
                       !memcmp(&line_[tag_start], "C420p10", tag_len)) {
              bit_depth_ = 10;
              chroma_subsample_ = true;
            } else if (tag_len == 7 &&
                       !memcmp(&line_[tag_start], "C420p12", tag_len)) {
              bit_depth_ = 12;
              chroma_subsample_ = true;
            } else {
              return PIK_FAILURE("Unsupported chroma subsampling type");
            }
            break;
          default:;  // ignore other tags
        }
      }
      if (line_[tag_end] == '\n' || line_[tag_end] == '\0') {
        break;
      }
    }
    return (xsize_ > 0 && ysize_ > 0);
  }

  Status ReadFrame(Image3B* yuv) {
    PIK_RETURN_IF_ERROR(ReadLine());
    if (memcmp(line_, "FRAME", 5)) {
      return PIK_FAILURE("Invalid frame header");
    }
    if (bit_depth_ != 8) {
      return PIK_FAILURE("Invalid bit-depth");
    }
    if (chroma_subsample_) {
      return PIK_FAILURE("420 is only supported for Image3U frames.");
    }
    *yuv = Image3B(xsize_, ysize_);
    for (int c = 0; c < 3; ++c) {
      for (int y = 0; y < ysize_; ++y) {
        uint8_t* PIK_RESTRICT row_yuv = yuv->PlaneRow(c, y);
        const size_t bytes_read = fread(row_yuv, 1, xsize_, f_);
        if (bytes_read != xsize_) {
          return PIK_FAILURE("Unexpected end of file");
        }
      }
    }
    return true;
  }

  Status ReadFrame(Image3U* yuv) {
    PIK_RETURN_IF_ERROR(ReadLine());
    if (memcmp(line_, "FRAME", 5)) {
      return PIK_FAILURE("Invalid frame header");
    }
    std::array<ImageU, 3> planes;
    int byte_depth = (bit_depth_ + 7) / 8;
    int limit = (1 << bit_depth_) - 1;
    PIK_ASSERT(byte_depth == 1 || byte_depth == 2);
    for (int c = 0; c < 3; ++c) {
      int pxsize = (c == 0 || !chroma_subsample_) ? xsize_ : (xsize_ + 1) / 2;
      int pysize = (c == 0 || !chroma_subsample_) ? ysize_ : (ysize_ + 1) / 2;
      planes[c] = ImageU(pxsize, pysize);
      for (int y = 0; y < pysize; ++y) {
        uint16_t* const PIK_RESTRICT row = planes[c].Row(y);
        for (int x = 0; x < pxsize; ++x) {
          if (byte_depth == 1) {
            uint8_t val;
            if (fread(&val, sizeof(val), 1, f_) != 1) return false;
            row[x] = val;
          } else {
            uint16_t val;
            if (fread(&val, sizeof(val), 1, f_) != 1) return false;
            if (val > limit) {
              return PIK_FAILURE("Value greater than indicated by bit-depth");
            }
            row[x] = val;
          }
        }
      }
    }
    if (chroma_subsample_) {
      *yuv = SuperSampleChroma(planes[0], planes[1], planes[2], bit_depth_);
    } else {
      *yuv = Image3U(planes);
    }
    return true;
  }

 private:
  bool ReadLine() {
    int pos = 0;
    for (; pos < 79; ++pos) {
      int n = fread(&line_[pos], 1, 1, f_);
      if (n != 1) {
        return PIK_FAILURE("Unexpected end of file");
      }
      if (line_[pos] == '\n') break;
    }
    line_[pos] = '\0';
    return true;
  }

  FILE* f_;
  size_t xsize_;
  size_t ysize_;
  int bit_depth_ = 8;
  bool chroma_subsample_ = false;
  char line_[80];
};

bool ReadImage(ImageFormatY4M, const std::string& pathname, Image3B* image) {
  FileWrapper f(pathname, "rb");
  if (f == nullptr) {
    return PIK_FAILURE("File open");
  }
  Y4MReader reader(f);
  PIK_RETURN_IF_ERROR(reader.ReadHeader());
  return reader.ReadFrame(image);
}

bool ReadImage(ImageFormatY4M, const std::string& pathname, Image3U* image,
               int* bit_depth) {
  FileWrapper f(pathname, "rb");
  if (f == nullptr) {
    return PIK_FAILURE("File open");
  }
  Y4MReader reader(f);
  PIK_RETURN_IF_ERROR(reader.ReadHeader());
  *bit_depth = reader.bit_depth();
  return reader.ReadFrame(image);
}

bool WriteImage(ImageFormatY4M format, const ImageB& image,
                const std::string& pathname) {
  return PIK_FAILURE("Unsupported");
}

bool WriteImage(ImageFormatY4M format, const Image3B& image3,
                const std::string& pathname) {
  FileWrapper f(pathname, "wb");
  if (f == nullptr) {
    return PIK_FAILURE("File open");
  }

  const int ret = fprintf(f, "YUV4MPEG2 W%zd H%zd F24:1 Ip A0:0 C444\nFRAME\n",
                          image3.xsize(), image3.ysize());
  PIK_CHECK(ret > 0);

  for (int c = 0; c < 3; ++c) {
    for (size_t y = 0; y < image3.ysize(); ++y) {
      const uint8_t* PIK_RESTRICT row = image3.ConstPlaneRow(c, y);
      const size_t bytes_written = fwrite(row, 1, image3.xsize(), f);
      PIK_CHECK(bytes_written == image3.xsize());
    }
  }
  return true;
}

bool WriteImage(ImageFormatY4M format, const Image3U& image3,
                const std::string& pathname) {
  const int bit_depth = format.bit_depth;
  PIK_CHECK(bit_depth == 8 || bit_depth == 10 || bit_depth == 12);

  FileWrapper f(pathname, "wb");
  if (f == nullptr) {
    return PIK_FAILURE("File open");
  }

  const int ret = fprintf(
      f, "YUV4MPEG2 W%zd H%zd F24:1 Ip A0:0 C%s%s\nFRAME\n", image3.xsize(),
      image3.ysize(), format.chroma_subsample ? "420" : "444",
      bit_depth == 8 ? "" : bit_depth == 10 ? "p10" : "p12");
  PIK_CHECK(ret > 0);

  std::array<ImageU, 3> subplanes;
  if (format.chroma_subsample) {
    SubSampleChroma(image3, bit_depth, &subplanes[0], &subplanes[1],
                    &subplanes[2]);
  }

  int byte_depth = (bit_depth + 7) / 8;
  int limit = (1 << bit_depth) - 1;
  for (int c = 0; c < 3; ++c) {
    const ImageU& plane =
        format.chroma_subsample ? subplanes[c] : image3.Plane(c);
    for (int y = 0; y < plane.ysize(); ++y) {
      for (int x = 0; x < plane.xsize(); ++x) {
        const uint16_t val = plane.Row(y)[x];
        PIK_CHECK(val <= limit);
        if (byte_depth == 1) {
          const uint8_t v = val;
          PIK_CHECK(fwrite(&v, sizeof(v), 1, f) == 1);
        } else {
          PIK_CHECK(fwrite(&val, sizeof(val), 1, f) == 1);
        }
      }
    }
  }
  return true;
}

namespace {

static LodePNGColorType PNGTypeFromNumPlanes(size_t num_planes) {
  switch (num_planes) {
    case 1:
      return LCT_GREY;
    case 2:
      return LCT_GREY_ALPHA;
    case 3:
      return LCT_RGB;
    case 4:
      return LCT_RGBA;
    default:
      PIK_NOTIFY_ERROR("Invalid num_planes");
      return LCT_GREY;
  }
}

class PngReader {
 public:
  PngReader(const std::string& pathname) {
    if (lodepng::load_file(file_, pathname) != 0) {
      PIK_NOTIFY_ERROR("Failed to read PNG");
    }
  }

  bool ReadHeader(size_t* PIK_RESTRICT xsize, size_t* PIK_RESTRICT ysize,
                  size_t* PIK_RESTRICT num_planes,
                  size_t* PIK_RESTRICT bit_depth) {
    unsigned w, h;
    LodePNGState state;
    lodepng_state_init(&state);
    const unsigned err =
        lodepng_inspect(&w, &h, &state, file_.data(), file_.size());
    if (err != 0) return PIK_FAILURE("Failed to inspect PNG");
    *xsize = w;
    *ysize = h;

    const LodePNGColorMode& color_mode = state.info_png.color;
    *num_planes = PlanesFromPNGType(color_mode.colortype);
    // Only trust 16, otherwise ask for 8 (e.g. 4-bit will be converted).
    *bit_depth = color_mode.bitdepth == 16 ? 16 : 8;
    return true;
  }

  // Arguments are what we want from LodePNG, typically the same as returned by
  // ReadHeader to minimize conversions (we have to convert from interleaved to
  // planar anyway and can convert types at the same time).
  std::vector<uint8_t> Read(size_t num_planes, size_t bit_depth) const {
    unsigned w, h;
    std::vector<uint8_t> image;
    const LodePNGColorType color_type = PNGTypeFromNumPlanes(num_planes);
    if (lodepng::decode(image, w, h, file_, color_type, bit_depth) != 0) {
      PIK_NOTIFY_ERROR("Failed to decode PNG");
    }
    return image;
  }

 private:
  std::vector<unsigned char> file_;
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
Status ReadPNGImage(const std::string& pathname, const int bias,
                    Image<T>* image) {
  PngReader reader(pathname);
  size_t xsize, ysize, num_planes, bit_depth;
  PIK_RETURN_IF_ERROR(
      reader.ReadHeader(&xsize, &ysize, &num_planes, &bit_depth));
  if (num_planes != 1) {
    return PIK_FAILURE("Wrong #planes");
  }
  if (bit_depth != 8 && bit_depth != 16) {
    return PIK_FAILURE("Unsupported bit-depth");
  }
  *image = Image<T>(xsize, ysize);
  const std::vector<uint8_t>& raw = reader.Read(num_planes, bit_depth);
  const size_t stride = bit_depth / kBitsPerByte;
  const size_t bytes_per_row = xsize * stride;

  for (size_t y = 0; y < ysize; ++y) {
    const uint8_t* PIK_RESTRICT row_in = raw.data() + y * bytes_per_row;
    T* PIK_RESTRICT row = image->Row(y);
    if (stride == 1) {
      for (size_t x = 0; x < xsize; ++x) {
        row[x] = ReadFromU8<T>(&row_in[x], bias);
      }
    } else {
      for (size_t x = 0; x < xsize; ++x) {
        row[x] = ReadFromU16<T>(&row_in[stride * x], bias);
      }
    }
  }

  return true;
}

}  // namespace

// Adds alpha channel to the output image only if a non-opaque pixel is present.
template <typename T>
Status ReadPNGMetaImage(const std::string& pathname, const int bias,
                        MetaImage<T>* image) {
  PngReader reader(pathname);
  size_t xsize, ysize, num_planes, bit_depth;
  PIK_RETURN_IF_ERROR(
      reader.ReadHeader(&xsize, &ysize, &num_planes, &bit_depth));
  if (num_planes < 1 || num_planes > 4) {
    return PIK_FAILURE("Wrong #planes");
  }
  if (bit_depth != 8 && bit_depth != 16) {
    return PIK_FAILURE("Wrong bit-depth");
  }

  image->SetColor(Image3<T>(xsize, ysize));
  const std::vector<uint8_t>& raw = reader.Read(num_planes, bit_depth);
  const size_t stride = bit_depth / kBitsPerByte;
  const size_t bytes_per_row = xsize * num_planes * stride;

  // Expand gray -> RGB
  if (num_planes == 1) {
    for (size_t y = 0; y < ysize; ++y) {
      const uint8_t* PIK_RESTRICT row_in = raw.data() + y * bytes_per_row;
      T* PIK_RESTRICT row0 = image->GetColor().PlaneRow(0, y);
      T* PIK_RESTRICT row1 = image->GetColor().PlaneRow(1, y);
      T* PIK_RESTRICT row2 = image->GetColor().PlaneRow(2, y);
      if (stride == 1) {
        for (size_t x = 0; x < xsize; ++x) {
          row0[x] = row1[x] = row2[x] = ReadFromU8<T>(&row_in[x], bias);
        }
      } else {
        for (size_t x = 0; x < xsize; ++x) {
          row0[x] = row1[x] = row2[x] =
              ReadFromU16<T>(&row_in[stride * x], bias);
        }
      }
    }
  } else if (num_planes == 2) {
    uint16_t alpha_masked = 65535;
    for (size_t y = 0; y < ysize; ++y) {
      const uint8_t* PIK_RESTRICT row_in = raw.data() + y * bytes_per_row;
      T* PIK_RESTRICT row0 = image->GetColor().PlaneRow(0, y);
      T* PIK_RESTRICT row1 = image->GetColor().PlaneRow(1, y);
      T* PIK_RESTRICT row2 = image->GetColor().PlaneRow(2, y);
      if (stride == 1) {
        for (size_t x = 0; x < xsize; ++x) {
          row0[x] = row1[x] = row2[x] = ReadFromU8<T>(&row_in[2 * x + 0], bias);
          alpha_masked &= row_in[2 * x + 1];
        }
      } else {
        for (size_t x = 0; x < xsize; ++x) {
          row0[x] = row1[x] = row2[x] =
              ReadFromU16<T>(&row_in[stride * (2 * x + 0)], bias);
          alpha_masked &= row_in[stride * (2 * x + 1)];
        }
      }
    }
    if (alpha_masked != (stride == 1 ? 255 : 65535)) {
      image->AddAlpha(stride * 8);
      for (size_t y = 0; y < ysize; ++y) {
        const uint8_t* PIK_RESTRICT row_in = raw.data() + y * bytes_per_row;
        uint16_t* PIK_RESTRICT row = image->GetAlpha().Row(y);
        if (stride == 1) {
          for (size_t x = 0; x < xsize; ++x) {
            row[x] = row_in[2 * x + 1];
          }
        } else {
          for (size_t x = 0; x < xsize; ++x) {
            row[x] = ReadFromU16<uint16_t>(&row_in[stride * (2 * x + 1)], 0);
          }
        }
      }
    }
  } else if (num_planes == 3) {
    for (size_t y = 0; y < ysize; ++y) {
      const uint8_t* PIK_RESTRICT row_in = raw.data() + y * bytes_per_row;
      T* PIK_RESTRICT row0 = image->GetColor().PlaneRow(0, y);
      T* PIK_RESTRICT row1 = image->GetColor().PlaneRow(1, y);
      T* PIK_RESTRICT row2 = image->GetColor().PlaneRow(2, y);
      if (stride == 1) {
        for (size_t x = 0; x < xsize; ++x) {
          row0[x] = ReadFromU8<T>(&row_in[3 * x + 0], bias);
          row1[x] = ReadFromU8<T>(&row_in[3 * x + 1], bias);
          row2[x] = ReadFromU8<T>(&row_in[3 * x + 2], bias);
        }
      } else {
        for (size_t x = 0; x < xsize; ++x) {
          row0[x] = ReadFromU16<T>(&row_in[stride * (3 * x + 0)], bias);
          row1[x] = ReadFromU16<T>(&row_in[stride * (3 * x + 1)], bias);
          row2[x] = ReadFromU16<T>(&row_in[stride * (3 * x + 2)], bias);
        }
      }
    }
  } else /* if (num_planes == 4) */ {
    uint16_t alpha_masked = 65535;
    for (size_t y = 0; y < ysize; ++y) {
      const uint8_t* PIK_RESTRICT row_in = raw.data() + y * bytes_per_row;
      T* PIK_RESTRICT row0 = image->GetColor().PlaneRow(0, y);
      T* PIK_RESTRICT row1 = image->GetColor().PlaneRow(1, y);
      T* PIK_RESTRICT row2 = image->GetColor().PlaneRow(2, y);
      if (stride == 1) {
        for (size_t x = 0; x < xsize; ++x) {
          row0[x] = ReadFromU8<T>(&row_in[4 * x + 0], bias);
          row1[x] = ReadFromU8<T>(&row_in[4 * x + 1], bias);
          row2[x] = ReadFromU8<T>(&row_in[4 * x + 2], bias);
          alpha_masked &= row_in[4 * x + 3];
        }
      } else {
        for (size_t x = 0; x < xsize; ++x) {
          row0[x] = ReadFromU16<T>(&row_in[stride * (4 * x + 0)], bias);
          row1[x] = ReadFromU16<T>(&row_in[stride * (4 * x + 1)], bias);
          row2[x] = ReadFromU16<T>(&row_in[stride * (4 * x + 2)], bias);
          alpha_masked &= ReadFromU16<T>(&row_in[stride * (4 * x + 3)], bias);
        }
      }
    }
    if (alpha_masked != (stride == 1 ? 255 : 65535)) {
      image->AddAlpha(stride * 8);
      for (size_t y = 0; y < ysize; ++y) {
        const uint8_t* PIK_RESTRICT row_in = raw.data() + y * bytes_per_row;
        uint16_t* PIK_RESTRICT row = image->GetAlpha().Row(y);
        if (stride == 1) {
          for (size_t x = 0; x < xsize; ++x) {
            row[x] = row_in[4 * x + 3];
          }
        } else {
          for (size_t x = 0; x < xsize; ++x) {
            row[x] = ReadFromU16<uint16_t>(&row_in[stride * (4 * x + 3)], 0);
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
  PIK_RETURN_IF_ERROR(ReadPNGMetaImage(pathname, bias, &meta));
  if (meta.HasAlpha()) {
    return PIK_FAILURE("Translucent PNG not supported");
  }
  *image = std::move(meta.GetColor());
  return true;
}

bool ReadImage(ImageFormatPNG, const std::string& pathname, ImageB* image) {
  return ReadPNGImage(pathname, 0, image);
}

bool ReadImage(ImageFormatPNG, const std::string& pathname, ImageS* image) {
  return ReadPNGImage(pathname, 0x8000, image);
}

bool ReadImage(ImageFormatPNG, const std::string& pathname, ImageU* image) {
  return ReadPNGImage(pathname, 0, image);
}

bool ReadImage(ImageFormatPNG, const std::string& pathname, Image3B* image) {
  return ReadPNGImage3(pathname, 0, image);
}

bool ReadImage(ImageFormatPNG, const std::string& pathname, Image3S* image) {
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

namespace {

// Allocates an internal buffer for 16-bit pixels in WriteHeader => not
// thread-safe, and cannot reuse for multiple images with different sizes.
class PngWriter {
  template <typename T>
  static size_t NumPlanes(const MetaImage<T>& image) {
    return image.HasAlpha() ? 4 : 3;
  }

  template <typename T>
  static size_t NumPlanes(const Image<T>& image) {
    return 1;
  }

  template <typename T>
  static size_t NumPlanes(const Image3<T>& image) {
    return 3;
  }

 public:
  template <class Image>
  explicit PngWriter(const Image& image) {
    using T = typename Image::T;

    xsize_ = image.xsize();
    ysize_ = image.ysize();
    num_planes_ = NumPlanes(image);
    bit_depth_ = sizeof(T) * kBitsPerByte;
    bytes_per_row_ = xsize_ * num_planes_ * sizeof(T);
    raw_.resize(ysize_ * bytes_per_row_);
    pos_ = raw_.data();
  }

  PIK_INLINE void WriteRow(const ImageB& image, const size_t y) {
    memcpy(pos_, image.Row(y), bytes_per_row_);
  }

  PIK_INLINE void WriteRow(const Image3B& image, const size_t y) {
    const uint8_t* PIK_RESTRICT row0 = image.ConstPlaneRow(0, y);
    const uint8_t* PIK_RESTRICT row1 = image.ConstPlaneRow(1, y);
    const uint8_t* PIK_RESTRICT row2 = image.ConstPlaneRow(2, y);
    for (size_t x = 0; x < xsize_; ++x) {
      pos_[3 * x + 0] = row0[x];
      pos_[3 * x + 1] = row1[x];
      pos_[3 * x + 2] = row2[x];
    }
  }

  PIK_INLINE void WriteRow(const MetaImageB& image, const size_t y) {
    const uint8_t* PIK_RESTRICT row0 = image.GetColor().ConstPlaneRow(0, y);
    const uint8_t* PIK_RESTRICT row1 = image.GetColor().ConstPlaneRow(1, y);
    const uint8_t* PIK_RESTRICT row2 = image.GetColor().ConstPlaneRow(2, y);
    if (num_planes_ == 4) {
      const uint16_t* PIK_RESTRICT row_alpha = image.GetAlpha().Row(y);
      for (size_t x = 0; x < xsize_; ++x) {
        pos_[4 * x + 0] = row0[x];
        pos_[4 * x + 1] = row1[x];
        pos_[4 * x + 2] = row2[x];
        pos_[4 * x + 3] = row_alpha[x] & 255;
      }
    } else {
      for (size_t x = 0; x < xsize_; ++x) {
        pos_[3 * x + 0] = row0[x];
        pos_[3 * x + 1] = row1[x];
        pos_[3 * x + 2] = row2[x];
      }
    }
  }

  void WriteRow(const ImageS& image, const size_t y) {
    const int16_t* PIK_RESTRICT row = image.ConstRow(y);
    for (size_t x = 0; x < xsize_; ++x) {
      StoreUnsignedBigEndian(row[x], pos_ + 2 * x);
    }
  }

  void WriteRow(const Image3S& image, const size_t y) {
    const int16_t* PIK_RESTRICT row0 = image.ConstPlaneRow(0, y);
    const int16_t* PIK_RESTRICT row1 = image.ConstPlaneRow(1, y);
    const int16_t* PIK_RESTRICT row2 = image.ConstPlaneRow(2, y);
    for (size_t x = 0; x < xsize_; ++x) {
      StoreUnsignedBigEndian(row0[x], pos_ + 6 * x + 0);
      StoreUnsignedBigEndian(row1[x], pos_ + 6 * x + 2);
      StoreUnsignedBigEndian(row2[x], pos_ + 6 * x + 4);
    }
  }

  void WriteRow(const MetaImageS& image, const size_t y) {
    const int16_t* PIK_RESTRICT row0 = image.GetColor().ConstPlaneRow(0, y);
    const int16_t* PIK_RESTRICT row1 = image.GetColor().ConstPlaneRow(1, y);
    const int16_t* PIK_RESTRICT row2 = image.GetColor().ConstPlaneRow(2, y);
    if (num_planes_ == 4) {
      const uint16_t* PIK_RESTRICT row_alpha = image.GetAlpha().Row(y);
      for (size_t x = 0; x < xsize_; ++x) {
        StoreUnsignedBigEndian(row0[x], pos_ + 8 * x + 0);
        StoreUnsignedBigEndian(row1[x], pos_ + 8 * x + 2);
        StoreUnsignedBigEndian(row2[x], pos_ + 8 * x + 4);
        StoreUnsignedBigEndian(row_alpha[x], pos_ + 8 * x + 6);
      }
    } else {
      for (size_t x = 0; x < xsize_; ++x) {
        StoreUnsignedBigEndian(row0[x], pos_ + 6 * x + 0);
        StoreUnsignedBigEndian(row1[x], pos_ + 6 * x + 2);
        StoreUnsignedBigEndian(row2[x], pos_ + 6 * x + 4);
      }
    }
  }

  void WriteRow(const ImageU& image, const size_t y) {
    const uint16_t* PIK_RESTRICT row = image.ConstRow(y);
    for (size_t x = 0; x < xsize_; ++x) {
      StoreUnsignedBigEndian(row[x], pos_ + 2 * x);
    }
  }

  void WriteRow(const Image3U& image, const size_t y) {
    const uint16_t* PIK_RESTRICT row0 = image.ConstPlaneRow(0, y);
    const uint16_t* PIK_RESTRICT row1 = image.ConstPlaneRow(1, y);
    const uint16_t* PIK_RESTRICT row2 = image.ConstPlaneRow(2, y);
    for (size_t x = 0; x < xsize_; ++x) {
      StoreUnsignedBigEndian(row0[x], pos_ + 6 * x + 0);
      StoreUnsignedBigEndian(row1[x], pos_ + 6 * x + 2);
      StoreUnsignedBigEndian(row2[x], pos_ + 6 * x + 4);
    }
  }

  void WriteRow(const MetaImageU& image, const size_t y) {
    const uint16_t* PIK_RESTRICT row0 = image.GetColor().ConstPlaneRow(0, y);
    const uint16_t* PIK_RESTRICT row1 = image.GetColor().ConstPlaneRow(1, y);
    const uint16_t* PIK_RESTRICT row2 = image.GetColor().ConstPlaneRow(2, y);
    if (num_planes_ == 4) {
      const uint16_t* PIK_RESTRICT row_alpha = image.GetAlpha().Row(y);
      for (size_t x = 0; x < xsize_; ++x) {
        StoreUnsignedBigEndian(row0[x], pos_ + 8 * x + 0);
        StoreUnsignedBigEndian(row1[x], pos_ + 8 * x + 2);
        StoreUnsignedBigEndian(row2[x], pos_ + 8 * x + 4);
        StoreUnsignedBigEndian(row_alpha[x], pos_ + 8 * x + 6);
      }
    } else {
      for (size_t x = 0; x < xsize_; ++x) {
        StoreUnsignedBigEndian(row0[x], pos_ + 6 * x + 0);
        StoreUnsignedBigEndian(row1[x], pos_ + 6 * x + 2);
        StoreUnsignedBigEndian(row2[x], pos_ + 6 * x + 4);
      }
    }
  }

  void NextRow() { pos_ += bytes_per_row_; }

  bool WriteEnd(const std::string& pathname) {
    PIK_CHECK(pos_ == raw_.data() + raw_.size());
    if (lodepng::encode(pathname, raw_.data(), xsize_, ysize_,
                        PNGTypeFromNumPlanes(num_planes_), bit_depth_) != 0) {
      return PIK_FAILURE("Failed to encode/write PNG");
    }
    return true;
  }

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

  size_t xsize_;
  size_t ysize_;
  size_t num_planes_;
  size_t bit_depth_;
  size_t bytes_per_row_;

  std::vector<uint8_t> raw_;
  uint8_t* pos_;
};

template <class Image>
bool WritePNGImage(const Image& image, const std::string& pathname) {
  PngWriter png(image);
  for (size_t y = 0; y < image.ysize(); ++y) {
    png.WriteRow(image, y);
    png.NextRow();
  }
  return png.WriteEnd(pathname);
}

}  // namespace

bool WriteImage(ImageFormatPNG, const ImageB& image,
                const std::string& pathname) {
  return WritePNGImage(image, pathname);
}

bool WriteImage(ImageFormatPNG, const ImageS& image,
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

bool WriteImage(ImageFormatPNG, const Image3S& image3,
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

#if ENABLE_JPEG

static void jpeg_catch_error(j_common_ptr cinfo) {
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
    FILE* f = *input_file.get();
    if (f == nullptr) {
      return PIK_FAILURE("File open");
    }
  } else if (input.inbuffer == nullptr) {
    return PIK_FAILURE("Invalid JpegInput.");
  }

  jpeg_decompress_struct cinfo;
  jpeg_error_mgr jerr;
  cinfo.err = jpeg_std_error(&jerr);
  cinfo.do_fancy_upsampling = TRUE;

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
        const uint8_t* PIK_RESTRICT row = &imagep[y * row_stride];
        uint8_t* PIK_RESTRICT row0 = rgb->PlaneRow(0, y);
        uint8_t* PIK_RESTRICT row1 = rgb->PlaneRow(1, y);
        uint8_t* PIK_RESTRICT row2 = rgb->PlaneRow(2, y);

        for (int x = 0; x < xsize; x++) {
          const uint8_t gray = row[x];
          row0[x] = row1[x] = row2[x] = gray;
        }
      }
      break;

    case JCS_RGB:
      while (cinfo.output_scanline < cinfo.output_height) {
        jpeg_read_scanlines(&cinfo, &output_line, 1);
        output_line += row_stride;
      }

      for (size_t y = 0; y < ysize; ++y) {
        const uint8_t* PIK_RESTRICT row = &imagep[y * row_stride];
        uint8_t* PIK_RESTRICT row0 = rgb->PlaneRow(0, y);
        uint8_t* PIK_RESTRICT row1 = rgb->PlaneRow(1, y);
        uint8_t* PIK_RESTRICT row2 = rgb->PlaneRow(2, y);
        for (size_t x = 0; x < xsize; x++) {
          row0[x] = row[3 * x + 0];
          row1[x] = row[3 * x + 1];
          row2[x] = row[3 * x + 2];
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

#endif  // #if ENABLE_JPEG

bool ReadImage(ImageFormatJPG, const std::string& pathname, Image3B* rgb) {
#if ENABLE_JPEG
  JpegInput input(pathname);
  return ReadJpegImage(input, rgb);
#else
  return PIK_FAILURE("Support for reading JPEG is disabled");
#endif
}

bool ReadImage(ImageFormatJPG, const uint8_t* buf, size_t size, Image3B* rgb) {
#if ENABLE_JPEG
  JpegInput input(buf, size);
  return ReadJpegImage(input, rgb);
#else
  return PIK_FAILURE("Support for reading JPEG is disabled");
#endif
}

bool WriteImage(ImageFormatJPG, const ImageB&, const std::string&) {
  return PIK_FAILURE("Writing JPEG is not supported");
}
bool WriteImage(ImageFormatJPG, const Image3B&, const std::string&) {
  return PIK_FAILURE("Writing JPEG is not supported");
}

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

}  // namespace pik
