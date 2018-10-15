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

#include "af_edge_preserving_filter.h"

// Edge-preserving smoothing: 7x8 weighted average based on L1 patch similarity.

#include <float.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <numeric>  // std::accumulate

#define DUMP_SIGMA 0

#undef PROFILER_ENABLED
#define PROFILER_ENABLED 1
#include "af_stats.h"
#include "common.h"
#include "data_parallel.h"
#include "profiler.h"
#include "simd/simd.h"
#include "status.h"
#if DUMP_SIGMA
#include "image_io.h"
#endif

#if 0
#define EPF_ASSERT(condition)                           \
  while (!(condition)) {                                \
    printf("EPF assert failed at line %d\n", __LINE__); \
    exit(1);                                            \
  }

#else
#define EPF_ASSERT(condition)
#endif

// Must include "normally" so the build system understands the dependency.
#include "af_epf.cctest"

#define SIMD_ATTR_IMPL "af_epf.cctest"
#include "simd/foreach_target.h"
