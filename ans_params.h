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

// Common parameters that are needed for both the ANS entropy encoding and
// decoding methods.

#ifndef ANS_PARAMS_H_
#define ANS_PARAMS_H_

#include <stdint.h>
#include <cstdlib>

namespace pik {

static const int kANSBufferSize = 1 << 16;

#define ANS_LOG_TAB_SIZE 10
#define ANS_TAB_SIZE (1 << ANS_LOG_TAB_SIZE)
#define ANS_TAB_MASK (ANS_TAB_SIZE - 1)
#define ANS_SIGNATURE 0x13  // Initial state, used as CRC.

}  // namespace pik

#endif  // ANS_PARAMS_H_
