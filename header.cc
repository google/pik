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

#include "header.h"

#include <stdint.h>
#include <string.h>
#include <algorithm>

#include "byte_order.h"
#include "fields.h"

namespace pik {
namespace {

// T provides a non-const VisitFields (allows ReadFieldsVisitor to load fields),
// so we need to cast const T to non-const for visitors that don't actually need
// non-const (e.g. WriteFieldsVisitor).
template <class T, class Visitor>
void VisitFieldsConst(const T& t, Visitor* visitor) {
  // Note: only called for Visitor that don't actually change T.
  const_cast<T*>(&t)->VisitFields(visitor);
}

// Stores and verifies the 4-byte file signature.
class Magic {
 public:
  static constexpr size_t EncodedBits() { return 32; }

  static bool Verify(BitReader* PIK_RESTRICT reader) {
    reader->FillBitBuffer();
    // (FillBitBuffer ensures we can read up to 32 bits over several calls)
    for (int i = 0; i < 4; ++i) {
      if (reader->PeekFixedBits<8>() != String()[i]) {
        return PIK_FAILURE("Wrong magic bytes");
      }
      reader->Advance(8);
    }
    return true;
  }

  static void Store(size_t* PIK_RESTRICT pos, uint8_t* storage) {
#if PIK_BYTE_ORDER_LITTLE
    uint32_t buf;
    memcpy(&buf, String(), 4);
    WriteBits(32, buf, pos, storage);
#else
    for (int i = 0; i < 4; ++i) {
      WriteBits(8, String()[i], pos, storage);
    }
#endif
  }

 private:
  static const unsigned char* String() {
    // \n causes files opened in text mode to be rejected, and \xCC detects
    // 7-bit transfers (it is also an uppercase I with accent in ISO-8859-1).
    return reinterpret_cast<const unsigned char*>("P\xCCK\n");
  }
};

}  // namespace

bool CanEncode(const Header& header, size_t* PIK_RESTRICT encoded_bits) {
  const size_t magic_bits = Magic::EncodedBits();

  CanEncodeFieldsVisitor visitor;
  VisitFieldsConst(header, &visitor);
  const bool ok = visitor.OK();
  *encoded_bits = ok ? magic_bits + visitor.EncodedBits() : 0;
  return ok;
}

bool LoadHeader(BitReader* reader, Header* PIK_RESTRICT header) {
  if (!Magic::Verify(reader)) return false;

  ReadFieldsVisitor visitor(reader);
  header->VisitFields(&visitor);
  return true;
}

bool StoreHeader(const Header& header, size_t* pos, uint8_t* storage) {
  Magic::Store(pos, storage);

  WriteFieldsVisitor visitor(pos, storage);
  VisitFieldsConst(header, &visitor);
  return visitor.OK();
}

}  // namespace pik
