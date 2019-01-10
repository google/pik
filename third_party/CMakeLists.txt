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
set(LCMS2_INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/lcms" PARENT_SCOPE)
include(ExternalProject)
ExternalProject_add(
  LCMS2
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lcms"
  PREFIX "${CMAKE_CURRENT_BINARY_DIR}/lcms"
  CONFIGURE_COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/lcms/configure" --prefix=<INSTALL_DIR>
  BUILD_COMMAND make
  BUILD_IN_SOURCE 1
)
