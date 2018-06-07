#include "padded_bytes.h"

#include <string.h>
#include <algorithm>
#include <memory>

#include "header.h"

namespace pik {

size_t PaddedBytes::PaddedSize(const size_t size) {
  // Allow writing entire 64-bit words.
  const size_t rounded_up = (size + 7) & ~7;
  // Avoid bounds checks in LoadHeader.
  const size_t safe_header = std::max(rounded_up, MaxCompressedHeaderSize());
  return safe_header;
}

void PaddedBytes::resize(const size_t size) {
  const size_t new_padded_size = PaddedSize(size);

  // Shrinking, no copy needed.
  if (new_padded_size <= padded_size_) {
    size_ = size;
    return;
  }

  CacheAlignedUniquePtr new_data = AllocateArray(new_padded_size);
  memcpy(new_data.get(), data_.get(), size_);  // old size
  memset(new_data.get() + size_, 0, new_padded_size - size_);

  size_ = size;  // update after copying!
  padded_size_ = new_padded_size;
  std::swap(new_data, data_);
}

}  // namespace pik
