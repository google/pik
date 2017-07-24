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

#include "lehmer_code.h"

#include <vector>

namespace pik {

int FindIndexAndRemove(int val, int* s, int len) {
  int idx = 0;
  for (int i = 0; i < len; ++i) {
    if (s[i] == val) {
      s[i] = -1;
      break;
    } else if (s[i] != -1) {
      ++idx;
    }
  }
  return idx;
}

void ComputeLehmerCode(const int* sigma, const int len, int* code) {
  std::vector<int> stdorder(len);
  for (int i = 0; i < len; ++i) {
    stdorder[i] = i;
  }
  for (int i = 0; i < len; ++i) {
    code[i] = FindIndexAndRemove(sigma[i], &stdorder[0], len);
  }
}

int FindValueAndRemove(int idx, int* s, int len) {
  int pos = 0;
  int val = 0;
  for (int i = 0; i < len; ++i) {
    if (s[i] == -1) continue;
    if (pos == idx) {
      val = s[i];
      s[i] = -1;
      break;
    }
    ++pos;
  }
  return val;
}

void DecodeLehmerCode(const int* code, int len, int* sigma) {
  std::vector<int> stdorder(len);
  for (int i = 0; i < len; ++i) {
    stdorder[i] = i;
  }
  for (int i = 0; i < len; ++i) {
    sigma[i] = FindValueAndRemove(code[i], &stdorder[0], len);
  }
}

}  // namespace pik
