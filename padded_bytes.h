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

#ifndef PADDED_BYTES_H_
#define PADDED_BYTES_H_

#include <stddef.h>
#include <stdint.h>
#include <memory>

#include "cache_aligned.h"

namespace pik {

// Subset of std::vector; allows WriteBits to write 64 bits at a time without
// bounds checking.
class PaddedBytes {
 public:
  // Required for output params.
  PaddedBytes() : size_(0), padded_size_(0), data_(nullptr, DummyDeleter) {}

  PaddedBytes(size_t size)
      : size_(size),
        padded_size_(PaddedSize(size)),
        data_(AllocateArray(padded_size_)) {
    memset(data_.get() + size_, 0, padded_size_ - size_);
  }

  // Reallocates and copies if PaddedSize(size) > padded_size_. The common case
  // of resizing to the final size after preallocating the upper bound is cheap.
  void resize(size_t size);

  const uint8_t* data() const { return data_.get(); }
  uint8_t* data() { return data_.get(); }
  size_t size() const { return size_; }

  size_t padded_size() const { return padded_size_; }

 private:
  static void DummyDeleter(uint8_t*) {}
  static size_t PaddedSize(size_t size);

  size_t size_;
  size_t padded_size_;
  CacheAlignedUniquePtr data_;
};

}  // namespace pik

#endif  // PADDED_BYTES_H_
