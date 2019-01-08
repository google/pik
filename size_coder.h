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

#ifndef SIZE_CODER_H_
#define SIZE_CODER_H_

#include <cstddef>
#include <cstdint>
#include "fields.h"
#include "status.h"

namespace pik {

template <uint32_t kDistribution>
class SizeCoderT {
 public:
  static size_t MaxSize(const size_t num_sizes) {
    const size_t bits = U32Coder::MaxEncodedBits(kDistribution) * num_sizes;
    return DivCeil(bits, kBitsPerByte);
  }

  static void Encode(const size_t size, size_t* PIK_RESTRICT pos,
                     uint8_t* storage) {
    PIK_CHECK(U32Coder::Write(kDistribution, size, pos, storage));
  }

  static size_t Decode(BitReader* reader) {
    return U32Coder::Read(kDistribution, reader);
  }
};

}  // namespace pik

#endif  // SIZE_CODER_H_
