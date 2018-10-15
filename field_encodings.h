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

#ifndef FIELD_ENCODINGS_H_
#define FIELD_ENCODINGS_H_

// Constants needed to encode/decode fields; avoids including the full fields.h.

#include <stdint.h>

namespace pik {

// kU32RawBits + x => send x raw bits. This value is convenient because x <= 32
// and ~32u + 32 == ~0u, which ensures RawBits can never exceed 32 and also
// allows the values to be sign-extended from an 8-bit immediate.
static constexpr uint32_t kU32RawBits = ~32u;

// Four direct values [0, 4).
static constexpr uint32_t kU32Direct0To3 = 0x83828180u;

// Three direct values 0, 1, 2 or 2 extra bits for [3, 6].
static constexpr uint32_t kU32Direct3Plus4 = 0x51828180u;

// Three direct values 0, 1, 2 or 3 extra bits for [3, 10].
static constexpr uint32_t kU32Direct3Plus8 = 0x52828180u;

// Four direct values 2, 3, 4 or 8.
static constexpr uint32_t kU32Direct2348 = 0x88848382u;

enum Bytes {
  // Values are determined by kU32Direct3Plus8.
  kNone = 0,  // Not present, don't write size
  kRaw,
  kBrotli  // Only if smaller, otherwise kRaw.
  // Future extensions: [3, 10].
};

}  // namespace pik

#endif  // FIELD_ENCODINGS_H_
