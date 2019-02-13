# Copyright 2019 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

cmake_minimum_required(VERSION 3.1)

# Set compilers to default to Clang is not set.
if ("$ENV{CC}" STREQUAL "")
  set(ENV{CC} clang)
endif()
if ("$ENV{CXX}" STREQUAL "")
  set(ENV{CXX} clang++)
endif()

project(PIK C CXX)

# Require clang for building.
if(NOT "${CMAKE_C_COMPILER_ID}" MATCHES "Clang" OR
   NOT "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  message(FATAL_ERROR "${CMAKE_CXX_COMPILER_ID} compiler is not supported.\n"
    "Use clang instead:\n  CC=clang CXX=clang++ cmake ..")
endif()

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

set(THREADS_PREFER_PTHREAD_FLAG YES)
find_package(Threads REQUIRED)

add_subdirectory(third_party)

# The pikcommon library definition.
include(pik/pikcommon.cmake)

add_executable(cpik pik/cpik_main.cc pik/cpik.cc pik/cmdline.cc)
add_executable(dpik pik/dpik_main.cc pik/dpik.cc pik/cmdline.cc)
add_executable(decode_and_encode pik/decode_and_encode.cc)
add_executable(butteraugli_main pik/butteraugli_main.cc)

set(BINARIES cpik dpik decode_and_encode butteraugli_main)
foreach (BINARY IN LISTS BINARIES)
  target_link_libraries("${BINARY}" pikcommon)
endforeach ()
install(TARGETS ${BINARIES} RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")

add_executable(multi_threading_benchmark pik/multi_threading_benchmark.cc)
target_link_libraries(multi_threading_benchmark
  benchmark
  gflags::gflags
  pikcommon
)

add_subdirectory(comparison_tool/viewer)

enable_testing()
cmake_policy(SET CMP0057 NEW)  # https://gitlab.kitware.com/cmake/cmake/issues/18198
include(GoogleTest)
set(TESTS
  ac_predictions_test
  adaptive_reconstruction_test
  ans_encode_test
  ans_test
  approx_cube_root_test
  bit_reader_test
  bits_test
  brotli_test
  byte_order_test
  codec_impl_test
  color_encoding_test
  color_management_test
  compressed_image_test
  convolve_test
  data_parallel_test
  dc_predictor_test
  dct_test
  dct_util_test
  deconvolve_test
  descriptive_statistics_test
  entropy_coder_test
  epf_test
  external_image_test
  fields_test
  gaborish_test
  gamma_correct_test
  gradient_test
  headers_test
  image_test
  linalg_test
  lossless8_test
  lossless16_test
  opsin_image_test
  opsin_inverse_test
  optimize_test
  padded_bytes_test
  quantizer_test
  rational_polynomial_test
  resample_test
  robust_statistics_test
  yuv_convert_test
  yuv_opsin_convert_test
)
foreach (TEST IN LISTS TESTS)
  add_executable("${TEST}" "pik/${TEST}.cc")
  target_compile_definitions("${TEST}" PRIVATE -DTEST_DATA_PATH="${CMAKE_CURRENT_SOURCE_DIR}/third_party/testdata")
  target_link_libraries("${TEST}" pikcommon gmock gtest gtest_main)
  gtest_add_tests(TARGET "${TEST}")
endforeach ()

target_link_libraries(opsin_image_test gflags::gflags)
