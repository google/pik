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

#include "container.h"

#include "common.h"
#include "fields.h"

namespace pik {

Status CanEncode(const Container& container,
                 size_t* PIK_RESTRICT encoded_bits) {
  PIK_RETURN_IF_ERROR(CanEncodeFields(container, encoded_bits));
  *encoded_bits = DivCeil(*encoded_bits, kBitsPerByte) * kBitsPerByte;
  return true;
}

Status LoadContainer(BitReader* PIK_RESTRICT reader,
                     Container* PIK_RESTRICT container) {
  PIK_RETURN_IF_ERROR(LoadFields(reader, container));
  if (container->signature != Container::kSignature) {
    return PIK_FAILURE("Container signature mismatch");
  }
  // Byte alignment for computing checksum of payload (including sections).
  reader->JumpToByteBoundary();
  return true;
}

Status StoreContainer(const Container& container, size_t* PIK_RESTRICT pos,
                      uint8_t* storage) {
  PIK_RETURN_IF_ERROR(StoreFields(container, pos, storage));
  WriteZeroesToByteBoundary(pos, storage);
  return true;
}

Status CanEncode(const ContainerSections& sections,
                 size_t* PIK_RESTRICT encoded_bits) {
  return CanEncodeSectionsT(sections, encoded_bits);
}

Status LoadSections(BitReader* reader,
                    ContainerSections* PIK_RESTRICT sections) {
  return LoadSectionsT(reader, sections);
}

Status StoreSections(const ContainerSections& sections,
                     size_t* PIK_RESTRICT pos, uint8_t* storage) {
  return StoreSectionsT(sections, pos, storage);
}

}  // namespace pik
