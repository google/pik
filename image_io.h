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

// Read/write Image or Image3.

#include <string>
#include <utility>  // std::move
#include <vector>

#include "arch_specific.h"
#include "image.h"
#include "status.h"

#define PIK_PORTABLE_IO 1

namespace pik {

// Formats
// IsExtension() = true if the filename appears to be an image of this format.
// kPortableOnly = false if the format supports internal file APIs.
// NativeImage3 is what to load from file before converting to linear RGB.

// PGM for 1 plane, PPM for 3 planes.
struct ImageFormatPNM {
  static const char* Name() { return "PNM"; }
  static bool IsExtension(const char* filename);
  static constexpr bool kPortableOnly = true;
  using NativeImage3 = Image3B;
};

// Gray, RGB, RGBA.
struct ImageFormatPNG {
  static const char* Name() { return "PNG"; }
  static bool IsExtension(const char* filename);
  static constexpr bool kPortableOnly = false;
  using NativeImage3 = Image3U;
};

struct ImageFormatY4M {
  static const char* Name() { return "Y4M"; }
  static bool IsExtension(const char* filename);
  static constexpr bool kPortableOnly = true;
  using NativeImage3 = Image3B;
};

// Loads RGB (possibly expanded from gray).
struct ImageFormatJPG {
  static const char* Name() { return "JPG"; }
  static bool IsExtension(const char* filename);
  static constexpr bool kPortableOnly = true;
  using NativeImage3 = Image3B;
};

struct ImageFormatPlanes {
  static const char* Name() { return "Planes"; }
  static bool IsExtension(const char* filename);
  static constexpr bool kPortableOnly = true;
  using NativeImage3 = Image3F;
};

// Wrappers

// Generic image reader with type auto-detection, the output is linear sRGB.
// NOTE: For PNGs, we assume sRGB color space. 16-bit PNGs are also supported.
Image3F ReadImage3Linear(const std::string& pathname);

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
bool ReadImage(ImageFormatPNG, const std::string&, ImageW*);
bool ReadImage(ImageFormatPNG, const std::string&, ImageU*);
bool ReadImage(ImageFormatPNG, const std::string&, Image3B*);
bool ReadImage(ImageFormatPNG, const std::string&, Image3W*);
bool ReadImage(ImageFormatPNG, const std::string&, Image3U*);

bool WriteImage(ImageFormatPNG, const ImageB&, const std::string&);
bool WriteImage(ImageFormatPNG, const ImageW&, const std::string&);
bool WriteImage(ImageFormatPNG, const ImageU&, const std::string&);
bool WriteImage(ImageFormatPNG, const Image3B&, const std::string&);
bool WriteImage(ImageFormatPNG, const Image3W&, const std::string&);
bool WriteImage(ImageFormatPNG, const Image3U&, const std::string&);

// Y4M

bool ReadImage(ImageFormatY4M, const std::string&, Image3B*);

bool WriteImage(ImageFormatY4M, const Image3B&, const std::string&);

// JPEG

bool ReadImage(ImageFormatJPG, const std::string&, Image3B*);

// Planes - text header, raw (linear) 2D array[s] with padding

template <typename T>
bool ReadImage(ImageFormatPlanes, const std::string&, Image<T>*);
template <typename T>
bool ReadImage(ImageFormatPlanes, const std::string&, Image3<T>*);

template <typename T>
bool WriteImage(ImageFormatPlanes, const Image<T>&, const std::string&);
template <typename T>
bool WriteImage(ImageFormatPlanes, const Image3<T>&, const std::string&);

}  // namespace pik

#endif  // IMAGE_IO_H_
