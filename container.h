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
#include "color_encoding.h"
#include "compiler_specific.h"
#include "field_encodings.h"
#include "padded_bytes.h"
#include "pik_params.h"
#include "status.h"

namespace pik {

struct Container {
  // \n causes files opened in text mode to be rejected, and \xCC detects
  // 7-bit transfers (it is also an uppercase I with accent in ISO-8859-1).
  static constexpr uint32_t kSignature = 0x0A4BCC50;

  uint32_t signature = kSignature;

  // If set, skips all subsequent fields (leaves them default-initialized).
  // NOTE: keep in sync with the conditional assignment in pik.cc (must update
  // it after adding a new field here).
  uint32_t simple = 0;

  uint32_t flags = 0;       // For future extensions.
  uint32_t num_images = 1;  // 0 = unbounded stream (skip size / checksum)

  // Of the original input file (i.e. io->dec_c_original in the encoder).
  ColorEncoding original_color_encoding;

  // From CodecInOut.dec_bit_depth, see description there.
  uint32_t original_bit_depth = 8;

  uint32_t target_nits_div50 = kDefaultIntensityTarget / 50;

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
Status VisitFields(Visitor* PIK_RESTRICT visitor,
                   Container* PIK_RESTRICT container) {
  visitor->U32(kU32RawBits + 32, &container->signature);
  visitor->U32(kU32RawBits + 1, &container->simple);
  if (container->simple) return true;

  visitor->U32(0x20100880, &container->flags);
  visitor->U32(0x20100681, &container->num_images);

  PIK_RETURN_IF_ERROR(
      VisitFields(visitor, &container->original_color_encoding));
  visitor->U32(0x05A09088, &container->original_bit_depth);
  // 100, 250, 4000 are common; don't anticipate more than 10,000.
  visitor->U32(0x08D08582, &container->target_nits_div50);

  if (container->num_images != 0) {
    // Full 32-bit (size can approach or exceed 4 GiB), but most fit in 16 MiB,
    // many in 1 MiB; zero for upper 32 bits.
    visitor->U32(0x20181410, &container->payload_size0);
    visitor->U32(0x20181080, &container->payload_size1);

    visitor->U32(0x84828180u, &container->num_checksum);
    for (uint32_t i = 0; i < container->num_checksum; ++i) {
      // Send raw 32 to allow updating the checksum in-place.
      visitor->U32(kU32RawBits + 32, &container->checksum[i]);
    }
  }

  return true;
}

// Returns whether the struct can be encoded (i.e. all fields have a valid
// representation). If so, "*encoded_bits" is the exact number of bits required.
Status CanEncode(const Container& container, size_t* PIK_RESTRICT encoded_bits);

Status LoadContainer(BitReader* PIK_RESTRICT reader,
                     Container* PIK_RESTRICT container);

Status StoreContainer(const Container& container, size_t* PIK_RESTRICT pos,
                      uint8_t* storage);

// Optional container-level metadata with backward and forward compatibility.
// (Per-image sections are in header.h.)

// Passed to/from image_codecs; also optionally stored in a Container section.
struct Metadata {
  bool HasAny() const { return !exif.empty() || !iptc.empty() || !xmp.empty(); }

  PaddedBytes exif;
  PaddedBytes iptc;
  PaddedBytes xmp;
};

template <class Visitor>
Status VisitFields(Visitor* PIK_RESTRICT visitor,
                   Metadata* PIK_RESTRICT metadata) {
  visitor->Bytes(Bytes::kBrotli, &metadata->exif);
  visitor->Bytes(Bytes::kBrotli, &metadata->iptc);
  visitor->Bytes(Bytes::kBrotli, &metadata->xmp);
  return true;
}

// Must be present if Container.num_images > 1.
struct Multiframe {
  enum class Type {
    kAnimation = 0,  // 1 image
    kStereo,         // 2 images
    kCubeMap,        // 6 images
    // Future extensions: [3, 10]
  } type;

  uint32_t num_loops;  // 0 means to repeat infinitely.

  uint32_t background_x;  // Background color first component
  uint32_t background_y;  // Background color second component
  uint32_t background_b;  // Background color third component
  uint32_t background_a;  // Background color alpha component

  // Ticks as rational number in seconds per tick
  uint32_t ticks_numerator;
  uint32_t ticks_denominator;  // Must be at least 1

  struct Rect {
    // if 0, use next encoded image in the container, else use a previous
    // rectangle with index current_index - source.
    uint32_t source;

    // offset and size of this rectangle in the source
    uint32_t x;
    uint32_t y;
    uint32_t xsize;  // if 0, size extended to max size of source
    uint32_t ysize;  // if 0, size extended to max size of source

    uint32_t delay;  // Delay in ticks

    // Blend mode to apply this rectangle to the image
    enum class BlendMode : uint32_t {
      kBlit = 0,  // Replace
      kAlpha,     // Alpha-blend
     // Future extensions: [2, 10]
    } blend_mode;

    // What data to apply the new rectangle to
    enum class DisposeMode : uint32_t {
      kKeep = 0,  // Keep previous pixels
      kReset,     // Reset to background
     // Future extensions: [2, 10]
    } dispose_mode;
  };

  std::vector<Rect> rectangles;
};

template <class Visitor>
Status VisitFields(Visitor* PIK_RESTRICT visitor,
                   Multiframe::Rect* PIK_RESTRICT rect) {
  visitor->U32(0x20100380, &rect->source);
  visitor->U32(0x20140980, &rect->x);
  visitor->U32(0x20140980, &rect->y);
  visitor->U32(0x20140980, &rect->xsize);
  visitor->U32(0x20140980, &rect->ysize);
  visitor->U32(0x20108180, &rect->delay);
  visitor->Enum(kU32Direct3Plus8, &rect->blend_mode);
  visitor->Enum(kU32Direct3Plus8, &rect->dispose_mode);
  return true;
}

template <class Visitor>
Status VisitFields(Visitor* PIK_RESTRICT visitor,
                   Multiframe* PIK_RESTRICT multiframe) {
  visitor->Enum(kU32Direct3Plus8, &multiframe->type);

  visitor->U32(0x20100380, &multiframe->num_loops);

  // TODO(user): improve how color values from the PIK color model are used
  // and encoded as background colors here.
  visitor->U32(0x200D0B09, &multiframe->background_x);
  visitor->U32(0x200D0B09, &multiframe->background_y);
  visitor->U32(0x200D0B09, &multiframe->background_b);
  visitor->U32(0x200D0B09, &multiframe->background_a);

  visitor->U32(0x20140981, &multiframe->ticks_numerator);
  visitor->U32(0x20140981, &multiframe->ticks_denominator);

  uint32_t num_rectangles = multiframe->rectangles.size();
  visitor->U32(0x20181004, &num_rectangles);
  visitor->EnsureContainerSize(num_rectangles, &multiframe->rectangles);

  for (uint32_t i = 0; i < multiframe->rectangles.size(); ++i) {
    PIK_RETURN_IF_ERROR(VisitFields(visitor, &multiframe->rectangles[i]));
  }
  return true;
}

struct ContainerSections {
  // Number of known sections at the time the bitstream was frozen. No need to
  // encode size if idx_section < kNumKnown because the decoder already knows
  // how to to read them. Do not change this after freezing!
  static constexpr size_t kNumKnown = 2;

  // Valid/present if non-null.
  std::unique_ptr<Metadata> metadata;      // if null: sRGB.
  std::unique_ptr<Multiframe> multiframe;  // if null: one main image

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
Status CanEncode(const ContainerSections& sections,
                 size_t* PIK_RESTRICT encoded_bits);

Status LoadSections(BitReader* reader,
                    ContainerSections* PIK_RESTRICT sections);

Status StoreSections(const ContainerSections& sections,
                     size_t* PIK_RESTRICT pos, uint8_t* storage);

}  // namespace pik

#endif  // CONTAINER_H_
