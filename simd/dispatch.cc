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

#include "simd/dispatch.h"

#include <stdint.h>
#include <atomic>

#if SIMD_ARCH_X86
#include <xmmintrin.h>
#ifndef _MSC_VER
#include <cpuid.h>
#endif
#endif

namespace pik {
namespace dispatch {

namespace {

bool IsBitSet(const uint32_t reg, const int index) {
  return (reg & (1U << index)) != 0;
}

#if SIMD_ARCH_X86

// Calls CPUID instruction with eax=level and ecx=count and returns the result
// in abcd array where abcd = {eax, ebx, ecx, edx} (hence the name abcd).
void Cpuid(const uint32_t level, const uint32_t count,
           uint32_t* SIMD_RESTRICT abcd) {
#ifdef _MSC_VER
  int regs[4];
  __cpuidex(regs, level, count);
  for (int i = 0; i < 4; ++i) {
    abcd[i] = regs[i];
  }
#else
  uint32_t a, b, c, d;
  __cpuid_count(level, count, a, b, c, d);
  abcd[0] = a;
  abcd[1] = b;
  abcd[2] = c;
  abcd[3] = d;
#endif
}

// Returns the lower 32 bits of extended control register 0.
// Requires CPU support for "OSXSAVE" (see below).
uint32_t ReadXCR0() {
#ifdef _MSC_VER
  return static_cast<uint32_t>(_xgetbv(0));
#else
  uint32_t xcr0, xcr0_high;
  const uint32_t index = 0;
  asm volatile(".byte 0x0F, 0x01, 0xD0"
               : "=a"(xcr0), "=d"(xcr0_high)
               : "c"(index));
  return xcr0;
#endif
}

#endif  // SIMD_ARCH_X86

// Not function-local => no compiler-generated locking.
std::atomic<int> supported_{-1};  // Not yet initialized

// Bits indicating which instruction set extensions are supported.
enum {
  kSSE = 1 << 0,
  kSSE2 = 1 << 1,
  kSSE3 = 1 << 2,
  kSSSE3 = 1 << 3,
  kSSE41 = 1 << 4,
  kSSE42 = 1 << 5,
  kAVX = 1 << 6,
  kAVX2 = 1 << 7,
  kFMA = 1 << 8,
  kLZCNT = 1 << 9,
  kBMI = 1 << 10,
  kBMI2 = 1 << 11,

  kGroupAVX2 = kAVX | kAVX2 | kFMA | kLZCNT | kBMI | kBMI2,
  kGroupSSE4 = kSSE | kSSE2 | kSSE3 | kSSSE3 | kSSE41 | kSSE42
};

}  // namespace

int SupportedTargets() {
  int supported = supported_.load(std::memory_order_acquire);
  // Already initialized, return that.
  if (SIMD_LIKELY(supported != -1)) {
    return supported;
  }

  supported = SIMD_NONE;

#if SIMD_ARCH_X86
  uint32_t flags = 0;
  uint32_t abcd[4];

  Cpuid(0, 0, abcd);
  const uint32_t max_level = abcd[0];

  // Standard feature flags
  Cpuid(1, 0, abcd);
  flags |= IsBitSet(abcd[3], 25) ? kSSE : 0;
  flags |= IsBitSet(abcd[3], 26) ? kSSE2 : 0;
  flags |= IsBitSet(abcd[2], 0) ? kSSE3 : 0;
  flags |= IsBitSet(abcd[2], 9) ? kSSSE3 : 0;
  flags |= IsBitSet(abcd[2], 19) ? kSSE41 : 0;
  flags |= IsBitSet(abcd[2], 20) ? kSSE42 : 0;
  flags |= IsBitSet(abcd[2], 12) ? kFMA : 0;
  flags |= IsBitSet(abcd[2], 28) ? kAVX : 0;
  const bool has_osxsave = IsBitSet(abcd[2], 27);

  // Extended feature flags
  Cpuid(0x80000001U, 0, abcd);
  flags |= IsBitSet(abcd[2], 5) ? kLZCNT : 0;

  // Extended features
  if (max_level >= 7) {
    Cpuid(7, 0, abcd);
    flags |= IsBitSet(abcd[1], 3) ? kBMI : 0;
    flags |= IsBitSet(abcd[1], 5) ? kAVX2 : 0;
    flags |= IsBitSet(abcd[1], 8) ? kBMI2 : 0;
  }

  // Verify OS support for XSAVE, without which XMM/YMM registers are not
  // preserved across context switches and are not safe to use.
  if (has_osxsave) {
    const uint32_t xcr0 = ReadXCR0();
    // XMM
    if (!IsBitSet(xcr0, 1)) {
      flags &= ~(kSSE | kSSE2 | kSSE3 | kSSSE3 | kSSE41 | kSSE42 | kAVX |
                 kAVX2 | kFMA);
    }
    // YMM
    if (!IsBitSet(xcr0, 2)) {
      flags &= ~(kAVX | kAVX2);
    }
  }

  // Set target bit(s) if all their group's flags are all set.
  if ((flags & kGroupAVX2) == kGroupAVX2) {
    supported |= SIMD_AVX2;
  }
  if ((flags & kGroupSSE4) == kGroupSSE4) {
    supported |= SIMD_SSE4;
  }
#elif SIMD_ARCH_ARM
  supported |= SIMD_ARM;
#endif

  supported_.store(supported, std::memory_order_release);
  return supported;
}

}  // namespace dispatch
}  // namespace pik
