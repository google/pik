# Copyright 2019 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

cmake_minimum_required(VERSION 3.1)

project(PIK C CXX)

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

set(THREADS_PREFER_PTHREAD_FLAG YES)
find_package(Threads REQUIRED)

add_subdirectory(third_party)

# The pikcommon library definition.
include(pikcommon.cmake)

add_executable(cpik cpik_main.cc cpik.cc cmdline.cc)
add_executable(dpik dpik_main.cc dpik.cc cmdline.cc)
add_executable(butteraugli_main butteraugli_main.cc)

set(BINARIES cpik dpik butteraugli_main)
foreach (BINARY IN LISTS BINARIES)
  target_link_libraries("${BINARY}" pikcommon)
endforeach ()
install(TARGETS ${BINARIES} RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")

add_subdirectory(comparison_tool/viewer)
