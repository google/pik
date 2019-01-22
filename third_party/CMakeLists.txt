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
add_subdirectory(brotli)

# lcms2
include(lcms2.cmake)

# FiniteStateEntropy
include(fse.cmake)
