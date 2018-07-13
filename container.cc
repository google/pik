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
#include "status.h"

namespace pik {

bool CanEncode(const Container& container, size_t* PIK_RESTRICT encoded_bits) {
  if (!CanEncodeFields(container, encoded_bits)) return false;
  *encoded_bits = DivCeil(*encoded_bits, kBitsPerByte) * kBitsPerByte;
  return true;
}

bool LoadContainer(BitReader* PIK_RESTRICT reader,
                   Container* PIK_RESTRICT container) {
  LoadFields(reader, container);
  if (container->signature != Container::kSignature) {
    return PIK_FAILURE("Container signature mismatch");
  }
  // Byte alignment for computing checksum of payload (including sections).
  reader->JumpToByteBoundary();
  return true;
}

bool StoreContainer(const Container& container, size_t* PIK_RESTRICT pos,
                    uint8_t* storage) {
  if (!StoreFields(container, pos, storage)) return false;
  WriteZeroesToByteBoundary(pos, storage);
  return true;
}

bool CanEncodeSections(const ContainerSections& sections,
                       size_t* PIK_RESTRICT encoded_bits) {
  return CanEncodeSectionsT(sections, encoded_bits);
}

bool LoadSections(BitReader* reader, ContainerSections* PIK_RESTRICT sections) {
  return LoadSectionsT(reader, sections);
}

template <class T>
bool StoreSections(const ContainerSections& sections, size_t* PIK_RESTRICT pos,
                   uint8_t* storage) {
  return StoreSectionsT(sections, pos, storage);
}

}  // namespace pik
