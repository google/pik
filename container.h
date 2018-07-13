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

#ifndef CONTAINER_H_
#define CONTAINER_H_

// Container file format with backward and forward-compatible extension
// capability and compressed integer fields.

#include <stddef.h>
#include <stdint.h>
#include <memory>
#include <vector>

#include "bit_reader.h"
#include "compiler_specific.h"

namespace pik {

#pragma pack(push, 1)

struct Container {
  // \n causes files opened in text mode to be rejected, and \xCC detects
  // 7-bit transfers (it is also an uppercase I with accent in ISO-8859-1).
  static constexpr uint32_t kSignature = 0x0A4BCC50;

  uint32_t signature = kSignature;
  uint32_t flags = 0;       // For future extensions.
  uint32_t num_images = 1;  // 0 = unbounded stream (skip size / checksum)

  // Useful for computing checksum below; skipped in unbounded stream mode.
  uint32_t payload_size0 = 0;  // least significant
  uint32_t payload_size1 = 0;

  // 0 for none, or 1 (HighwayHash64 & 0xFFFFFFFFu), 2 (HighwayHash64),
  // 4 (HighwayHash128) of the next payload_size bytes (after padding).
  // Useful for archiving; skipped in unbounded stream mode.
  uint32_t num_checksum = 0;
  uint32_t checksum[4] = {0};  // least significant first

  // Add new fields (only visited if a new flag is set) before this comment.

  // Payload (including sections) begins after padding to byte boundary!
};

// For loading/storing fields from/to the compressed stream.
template <class Visitor>
void VisitFields(Visitor* PIK_RESTRICT visitor,
                 Container* PIK_RESTRICT container) {
  visitor->U32(~32u + 32, &container->signature);
  visitor->U32(0x20100880, &container->flags);
  visitor->U32(0x20100681, &container->num_images);

  if (container->num_images != 0) {
    // Full 32-bit (size can approach or exceed 4 GiB), but most fit in 16 MiB,
    // many in 1 MiB; zero for upper 32 bits.
    visitor->U32(0x20181410, &container->payload_size0);
    visitor->U32(0x20181080, &container->payload_size1);

    visitor->U32(0x84828180u, &container->num_checksum);
    for (uint32_t i = 0; i < container->num_checksum; ++i) {
      // Send raw 32 to allow updating the checksum in-place.
      visitor->U32(~32u + 32, &container->checksum[i]);
    }
  }
}

// Returns whether the struct can be encoded (i.e. all fields have a valid
// representation). If so, "*encoded_bits" is the exact number of bits required.
bool CanEncode(const Container& container, size_t* PIK_RESTRICT encoded_bits);

bool LoadContainer(BitReader* PIK_RESTRICT reader,
                   Container* PIK_RESTRICT container);

bool StoreContainer(const Container& container, size_t* PIK_RESTRICT pos,
                    uint8_t* storage);

// Optional container-level metadata with backward and forward compatibility.
// (Per-image sections are in header.h.)

struct Metadata {
  std::vector<uint8_t> icc_profile;
  std::vector<uint8_t> exif;
  std::vector<uint8_t> xmp;
};

template <class Visitor>
void VisitFields(Visitor* PIK_RESTRICT visitor,
                 Metadata* PIK_RESTRICT metadata) {
  visitor->Bytes(&metadata->icc_profile);
  visitor->Bytes(&metadata->exif);
  visitor->Bytes(&metadata->xmp);
}

// Must be present if Container.num_images > 1.
struct Multiframe {
  enum {
    kStereo = 8,
    kCubeMap,
    kAnimation,
    // Reserve 0..7 for extensions.
  };

  uint32_t type;
};

template <class Visitor>
void VisitFields(Visitor* PIK_RESTRICT visitor,
                 Multiframe* PIK_RESTRICT multiframe) {
  visitor->U32(0x038A8988, &multiframe->type);
  if (multiframe->type == Multiframe::kAnimation) {
    // TODO(janwas): animation fields
  }
}

struct ContainerSections {
  // Number of known sections at the time the bitstream was frozen. No need to
  // encode size if idx_section < kNumKnown because the decoder already knows
  // how to to read them. Do not change this after freezing!
  static constexpr size_t kNumKnown = 2;

  // Valid/present if non-null.
  std::unique_ptr<Metadata> metadata;
  std::unique_ptr<Multiframe> multiframe;

  // Add new members before this comment.
};

template <class Visitor>
void VisitSections(Visitor* PIK_RESTRICT visitor,
                   ContainerSections* PIK_RESTRICT sections) {
  // Sections are read/written in this order, so do not rearrange.
  (*visitor)(&sections->metadata);
  (*visitor)(&sections->multiframe);

  // Add new section visitor before this comment.
}

// Returns whether "sections" can be encoded (i.e. all fields have a valid
// representation). If so, "*encoded_bits" is the exact number of bits required.
bool CanEncode(const ContainerSections& sections,
               size_t* PIK_RESTRICT encoded_bits);

bool LoadSections(BitReader* reader, ContainerSections* PIK_RESTRICT sections);

bool StoreSections(const ContainerSections& sections, size_t* PIK_RESTRICT pos,
                   uint8_t* storage);

#pragma pack(pop)

}  // namespace pik

#endif  // CONTAINER_H_
