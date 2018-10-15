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
#include <string.h>
#include <memory>

#include "cache_aligned.h"
#include "status.h"

namespace pik {

// Provides a subset of the std::vector interface with some differences:
// - allows WriteBits to write 64 bits at a time without bounds checking;
// - ONLY zero-initializes the first byte (required by WriteBits);
// - ensures cache-line alignment.
class PaddedBytes {
 public:
  // Required for output params.
  PaddedBytes() {}

  explicit PaddedBytes(size_t size) : size_(size) {
    if (size != 0) IncreaseCapacityTo(size);
  }

  PaddedBytes(const PaddedBytes& other) : size_(other.size_) {
    if (size_ != 0) IncreaseCapacityTo(size_);
    if (data() != nullptr) memcpy(data(), other.data(), size_);
  }
  PaddedBytes& operator=(const PaddedBytes& other) {
    // Self-assignment is safe.
    resize(other.size());
    if (data() != nullptr) memmove(data(), other.data(), size_);
    return *this;
  }

  PaddedBytes(PaddedBytes&& other) = default;
  PaddedBytes& operator=(PaddedBytes&& other) = default;

  void swap(PaddedBytes& other) {
    std::swap(size_, other.size_);
    std::swap(capacity_, other.capacity_);
    std::swap(data_, other.data_);
  }

  void reserve(size_t capacity) {
    if (capacity > capacity_) IncreaseCapacityTo(capacity);
  }
  // NOTE: unlike vector, this does not initialize the new data!
  // However, we guarantee that write_bits can safely append after
  // the resize, as we zero-initialize the first new byte of data.
  void resize(size_t size) {
    if (size > capacity_) IncreaseCapacityTo(size);
    size_ = (data() == nullptr) ? 0 : size;
  }
  // Amortized constant complexity due to exponential growth.
  void push_back(uint8_t x) {
    if (size_ == capacity_) {
      IncreaseCapacityTo(std::max<size_t>(3 * capacity_ / 2, 64));
      if (data() == nullptr) return;
    }

    data_[size_++] = x;
  }

  size_t size() const { return size_; }
  size_t capacity() const { return capacity_; }

  uint8_t* data() { return data_.get(); }
  const uint8_t* data() const { return data_.get(); }

  // std::vector operations implemented in terms of the public interface above.

  void clear() { resize(0); }
  bool empty() const { return size() == 0; }

  void assign(std::initializer_list<uint8_t> il) {
    resize(il.size());
    memcpy(data(), il.begin(), il.size());
  }

  uint8_t* begin() { return data(); }
  const uint8_t* begin() const { return data(); }
  uint8_t* end() { return begin() + size(); }
  const uint8_t* end() const { return begin() + size(); }

  uint8_t& operator[](const size_t i) {
    PIK_ASSERT(i < size());
    return data()[i];
  }
  const uint8_t& operator[](const size_t i) const {
    PIK_ASSERT(i < size());
    return data()[i];
  }

  uint8_t& back() {
    PIK_ASSERT(size() != 0);
    return data()[size() - 1];
  }
  const uint8_t& back() const {
    PIK_ASSERT(size() != 0);
    return data()[size() - 1];
  }

 private:
  // Copies existing data to newly allocated "data_". If allocation fails,
  // data() == nullptr and size_ = capacity_ = 0.
  void IncreaseCapacityTo(size_t capacity);

  size_t size_ = 0;
  size_t capacity_ = 0;
  CacheAlignedUniquePtr data_;
};

}  // namespace pik

#endif  // PADDED_BYTES_H_
