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

#ifndef IMAGE_IO_H_
#define IMAGE_IO_H_

// Read/write Image or Image3. DEPRECATED, use CodecInOut instead.

#include <memory>
#include <string>
#include <utility>  // std::move
#include <vector>

#include "arch_specific.h"
#include "cache_aligned.h"
#include "image.h"
#include "status.h"

namespace pik {

// PGM for 1 plane, PPM for 3 planes.
struct ImageFormatPNM {
  static const char* Name() { return "PNM"; }
  static bool IsExtension(const char* filename);
  using NativeImage3 = Image3B;
};

// Gray, RGB, RGBA.
struct ImageFormatPNG {
  static const char* Name() { return "PNG"; }
  static bool IsExtension(const char* filename);
  using NativeImage3 = MetaImageU;
};

struct ImageFormatY4M {
  ImageFormatY4M() {}
  ImageFormatY4M(int bits, bool subsample)
      : bit_depth(bits), chroma_subsample(subsample) {}
  static const char* Name() { return "Y4M"; }
  static bool IsExtension(const char* filename);
  using NativeImage3 = Image3B;
  const int bit_depth = 8;
  const bool chroma_subsample = false;
};

// Loads RGB (possibly expanded from gray).
struct ImageFormatJPG {
  static const char* Name() { return "JPG"; }
  static bool IsExtension(const char* filename);
  using NativeImage3 = Image3B;
};

// Wrappers

// Writes after linear rescaling to 0-255.
template <class Format>
void WriteScaledToB(Format format, const ImageF& image,
                    const std::string& pathname) {
  ImageB bytes(image.xsize(), image.ysize());
  ImageConvert(image, 255.0f, &bytes);
  WriteImage(format, bytes, pathname);
}

template <class Format>
void WriteScaledToB(Format format, const Image3F& image,
                    const std::string& pathname) {
  Image3B bytes(image.xsize(), image.ysize());
  Image3Convert(image, 255.0f, &bytes);
  WriteImage(format, bytes, pathname);
}

// PNM

bool ReadImage(ImageFormatPNM, const std::string&, ImageB*);
bool ReadImage(ImageFormatPNM, const std::string&, Image3B*);

bool WriteImage(ImageFormatPNM, const ImageB&, const std::string&);
bool WriteImage(ImageFormatPNM, const Image3B&, const std::string&);

// PNG

// The format specifies unsigned pixels; when converting from/to int16_t, we
// add/subtract 0x8000 to maintain the relative ordering of values.

bool ReadImage(ImageFormatPNG, const std::string&, ImageB*);
bool ReadImage(ImageFormatPNG, const std::string&, ImageS*);
bool ReadImage(ImageFormatPNG, const std::string&, ImageU*);
bool ReadImage(ImageFormatPNG, const std::string&, Image3B*);
bool ReadImage(ImageFormatPNG, const std::string&, Image3S*);
bool ReadImage(ImageFormatPNG, const std::string&, Image3U*);
bool ReadImage(ImageFormatPNG, const std::string&, MetaImageB*);
bool ReadImage(ImageFormatPNG, const std::string&, MetaImageU*);

bool WriteImage(ImageFormatPNG, const ImageB&, const std::string&);
bool WriteImage(ImageFormatPNG, const ImageS&, const std::string&);
bool WriteImage(ImageFormatPNG, const ImageU&, const std::string&);
bool WriteImage(ImageFormatPNG, const Image3B&, const std::string&);
bool WriteImage(ImageFormatPNG, const Image3S&, const std::string&);
bool WriteImage(ImageFormatPNG, const Image3U&, const std::string&);
bool WriteImage(ImageFormatPNG, const MetaImageB&, const std::string&);
bool WriteImage(ImageFormatPNG, const MetaImageU&, const std::string&);

// Y4M

bool ReadImage(ImageFormatY4M, const std::string&, Image3B*);
bool ReadImage(ImageFormatY4M, const std::string& pathname, Image3U* image,
               int* bit_depth);

// Unsupported (will return false) but required by WriteLinear.
bool WriteImage(ImageFormatY4M, const ImageB&, const std::string&);

bool WriteImage(ImageFormatY4M, const Image3B&, const std::string&);
bool WriteImage(ImageFormatY4M, const Image3U&, const std::string&);

// JPEG

bool ReadImage(ImageFormatJPG, const std::string&, Image3B*);

bool ReadImage(ImageFormatJPG, const uint8_t* buf, size_t size, Image3B*);

// Unsupported (will return false) but required by WriteLinear.
bool WriteImage(ImageFormatJPG, const ImageB&, const std::string&);
bool WriteImage(ImageFormatJPG, const Image3B&, const std::string&);

}  // namespace pik

#endif  // IMAGE_IO_H_
