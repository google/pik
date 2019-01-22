// Copyright 2018 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "epf.h"

// Edge-preserving smoothing: 7x8 weighted average based on L1 patch similarity.

#include <float.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <numeric>  // std::accumulate

#ifndef EPF_DUMP_SIGMA
#define EPF_DUMP_SIGMA 0
#endif
#ifndef EPF_ENABLE_STATS
#define EPF_ENABLE_STATS 0
#endif

#undef PROFILER_ENABLED
#define PROFILER_ENABLED 1
#include "ac_strategy.h"
#include "common.h"
#include "descriptive_statistics.h"
#include "fields.h"
#include "profiler.h"
#include "simd/simd.h"
#include "status.h"
#if EPF_DUMP_SIGMA
#include "image_io.h"
#endif

#if 1
#define EPF_ASSERT(condition)                           \
  while (!(condition)) {                                \
    printf("EPF assert failed at line %d\n", __LINE__); \
    exit(1);                                            \
  }

#else
#define EPF_ASSERT(condition)
#endif

namespace pik {

EpfParams::EpfParams() { Bundle::Init(this); }

}  // namespace pik

// Must include "normally" so the build system understands the dependency.
#include "epf_target.cc"

#define SIMD_ATTR_IMPL "epf_target.cc"
#include "simd/foreach_target.h"
