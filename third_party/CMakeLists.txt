# Copyright 2019 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

# lodepng
include(lodepng.cmake)

# brotli
if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/brotli/c/include/brotli/decode.h")
  message(FATAL_ERROR "Brotli is required to build pik, run \"git submodule update --init\" to get it, then run cmake again")
endif ()
add_subdirectory(brotli EXCLUDE_FROM_ALL)

# lcms2
include(lcms2.cmake)

# FiniteStateEntropy
include(fse.cmake)

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(benchmark EXCLUDE_FROM_ALL)

add_subdirectory(gflags EXCLUDE_FROM_ALL)

add_subdirectory(googletest EXCLUDE_FROM_ALL)
