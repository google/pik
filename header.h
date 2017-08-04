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

#ifndef HEADER_H_
#define HEADER_H_

// Container file format with backward and forward-compatible extension
// capability and compression of integer fields.

#include <stddef.h>
#include <stdint.h>
#include <memory>

#include "bit_buffer.h"
#include "compiler_specific.h"
#include "status.h"
#include "types.h"

namespace pik {

#pragma pack(push, 1)

// U32Coder selector_bits for a uniform distribution over the 32-bit range.
constexpr uint32_t kU32Selectors = 0x20181008;

// Header applicable to all image types. Only minimally extensible via flags;
// use Sections to add additional functionality in future.
struct Header {
  // Loosely associated bit flags that together describe the pixel format and
  // compression mode.
  enum Flags {
    // The last plane is alpha and compressed without loss. None of the
    // other components are premultiplied.
    kAlpha = 1,


    // Any non-alpha plane(s) are compressed without loss.
    kWebPLossless = 4,

    // A palette precedes the image data (indices, possibly more than 8 bits).
    kPalette = 8,

  };

  // For loading/storing fields from/to the compressed stream. Accepts Bytes or
  // uint32_t preceded by U32Coder's selector_bits.
  template <class Visitor>
  void VisitFields(Visitor* const PIK_RESTRICT visitor) {
    // Almost all camera images are less than 8K * 8K. We also allow the
    // full 32-bit range for completeness.
    (*visitor)(0x200E0B09, &xsize);
    (*visitor)(0x200E0B09, &ysize);
    // 2-bit encodings for 1 and 3 common cases; a dozen components for
    // remote-sensing data, or thousands for hyperspectral images.
    (*visitor)(0x00100004, &num_components);
    (*visitor)(kU32Selectors, &flags);

    // Do not add other fields - only sections can be added.
  }

  uint32_t xsize = 0;
  uint32_t ysize = 0;
  uint32_t num_components = 0;
  uint32_t flags = 0;
  // TODO(janwas): hash?
};

// Returns an upper bound on the number of bytes needed to store a Header.
size_t MaxCompressedHeaderSize();

bool LoadHeader(BitSource* const PIK_RESTRICT source,
                Header* const PIK_RESTRICT header);

bool StoreHeader(const Header& header, BitSink* const PIK_RESTRICT sink);

// A "section" is optional metadata that is only read from/written to the
// compressed stream if needed. Adding sections is the only way to extend
// this container format. Old code is forwards compatible because unknown
// sections can be skipped. Old files are backwards compatible because they
// indicate future sections are not present using a bit field in the header.
//
// To ensure interoperability, there will be no opaque fields nor proprietary
// section definitions. To introduce a new section and thereby advance the
// "version number" of this file format, add a unique_ptr member to Header and
// append a call to "visitor" in Header::VisitSections.

struct ICC {
  template <class Visitor>
  void VisitFields(Visitor* const PIK_RESTRICT visitor) {
    (*visitor)(&profile);
  }

  Bytes profile;
};

struct EXIF {
  template <class Visitor>
  void VisitFields(Visitor* const PIK_RESTRICT visitor) {
    (*visitor)(&metadata);
  }

  Bytes metadata;
};

struct Sections {
  template <class Visitor>
  void VisitSections(Visitor* const PIK_RESTRICT visitor) {
    // Sections are read/written in this order, so do not rearrange.
    (*visitor)(&icc);
    (*visitor)(&exif);
    // Add new sections before this comment.
  }

  // Valid/present if non-null.
  std::unique_ptr<ICC> icc;
  std::unique_ptr<EXIF> exif;
};

// Returns an upper bound on the number of bytes needed to store "sections".
size_t MaxCompressedSectionsSize(const Sections& sections);

const uint8_t* const PIK_RESTRICT
LoadSections(BitSource* const PIK_RESTRICT source,
             Sections* const PIK_RESTRICT sections);

// "max_bytes" is the return value of MaxCompressedSectionsSize.
uint8_t* const PIK_RESTRICT StoreSections(const Sections& sections,
                                          const size_t max_bytes,
                                          BitSink* const PIK_RESTRICT sink);

#pragma pack(pop)

}  // namespace pik

#endif  // HEADER_H_
