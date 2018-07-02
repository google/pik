cmake_minimum_required(VERSION 3.1)

project(PIK CXX)

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

set(THREADS_PREFER_PTHREAD_FLAG YES)
find_package(Threads REQUIRED)

if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/brotli/c/include/brotli/decode.h")
  message(FATAL_ERROR "Brotli is required to build pik, run \"git submodule update --init\" to get it, then run cmake again")
endif ()
set(BROTLI_BUNDLED_MODE ON)
add_subdirectory(third_party/brotli)

find_package(PNG REQUIRED)
find_package(JPEG REQUIRED)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

add_library(af_edge_preserving_filter_none STATIC af_edge_preserving_filter.cc)

add_library(pikcommon STATIC
  simd/dispatch.cc
  simd/dispatch.h
  adaptive_quantization.cc
  adaptive_quantization.h
  af_edge_preserving_filter.cc
  af_edge_preserving_filter.h
  af_solver.h
  af_stats.cc
  af_stats.h
  alpha_blend.cc
  alpha_blend.h
  ans_decode.cc
  ans_decode.h
  ans_encode.cc
  ans_encode.h
  ans_params.h
  approx_cube_root.h
  arch_specific.cc
  arch_specific.h
  args.h
  bit_reader.h
  bits.h
  butteraugli/butteraugli.cc
  butteraugli/butteraugli.h
  butteraugli_comparator.cc
  butteraugli_comparator.h
  butteraugli_distance.cc
  butteraugli_distance.h
  byte_order.h
  cache_aligned.h
  cluster.h
  common.h
  compiler_specific.h
  compressed_image.cc
  compressed_image.h
  context.h
  context_map_decode.cc
  context_map_decode.h
  context_map_encode.cc
  context_map_encode.h
  convolve.h
  data_parallel.h
  dc_predictor.cc
  dc_predictor.h
  dc_predictor_slow.h
  dct.cc
  dct.h
  dct_util.cc
  dct_util.h
  deconvolve.cc
  deconvolve.h
  fast_log.h
  fields.h
  gamma_correct.cc
  gamma_correct.h
  gamma_correct_poly.h
  gauss_blur.cc
  gauss_blur.h
  header.cc
  header.h
  huffman_decode.cc
  huffman_decode.h
  huffman_encode.cc
  huffman_encode.h
  image.cc
  image.h
  image_io.cc
  image_io.h
  lehmer_code.cc
  lehmer_code.h
  linalg.cc
  linalg.h
  noise.cc
  noise.h
  entropy_coder.cc
  entropy_coder.h
  opsin_image.cc
  opsin_image.h
  opsin_inverse.cc
  opsin_inverse.h
  opsin_params.cc
  opsin_params.h
  optimize.h
  os_specific.cc
  os_specific.h
  padded_bytes.cc
  padded_bytes.h
  pik_alpha.cc
  pik_alpha.h
  pik.cc
  pik.h
  pik_info.cc
  pik_info.h
  pik_params.h
  prevent_elision.h
  profiler.h
  quantizer.cc
  quantizer.h
  rational_polynomial.h
  resample.h
  robust_statistics.h
  sections.cc
  sections.h
  simd_helpers.h
  status.h
  tile_flow.cc
  tile_flow.h
  tsc_timer.h
  upscaler.cc
  upscaler.h
  write_bits.h
  xorshift128plus.h
  yuv_convert.cc
  yuv_convert.h
)

target_compile_definitions(pikcommon PRIVATE -DSIMD_ENABLE=4)
target_compile_options(pikcommon PRIVATE -msse4.2 -maes)

target_include_directories(pikcommon PRIVATE "${JPEG_INCLUDE_DIR}")

target_link_libraries(pikcommon PRIVATE
  af_edge_preserving_filter_none
  brotlicommon-static
  brotlienc-static
  brotlidec-static
  PNG::PNG
  "${JPEG_LIBRARIES}"
  Threads::Threads
)

set(BINARIES cpik dpik butteraugli_main png2y4m y4m2png)
foreach (BINARY IN LISTS BINARIES)
  add_executable("${BINARY}" "${BINARY}.cc")
  target_link_libraries("${BINARY}" pikcommon)
endforeach ()
install(TARGETS ${BINARIES} RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
