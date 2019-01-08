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
add_subdirectory(third_party/brotli)

set(LCMS2_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/lcms/install")
include(ExternalProject)
ExternalProject_add(
  LCMS2
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/lcms"
  PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/third_party/lcms"
  CONFIGURE_COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/third_party/lcms/configure" --prefix=<INSTALL_DIR>
  INSTALL_DIR "${LCMS2_INSTALL_DIR}"
  BUILD_COMMAND make
  BUILD_IN_SOURCE 1
)

find_package(Eigen3 REQUIRED)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

add_library(pikcommon STATIC
  simd/targets.cc
  simd/targets.h
  third_party/lodepng/lodepng.cc
  third_party/lodepng/lodepng.h
  ac_predictions.cc
  ac_predictions.h
  ac_strategy.cc
  ac_strategy.h
  adaptive_quantization.cc
  adaptive_quantization.h
  adaptive_reconstruction.cc
  adaptive_reconstruction_fwd.h
  adaptive_reconstruction.h
  alpha.cc
  alpha.h
  ans_common.cc
  ans_common.h
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
  block.h
  brotli.cc
  brotli.h
  butteraugli/butteraugli.cc
  butteraugli/butteraugli.h
  butteraugli_comparator.cc
  butteraugli_comparator.h
  butteraugli_distance.cc
  butteraugli_distance.h
  byte_order.h
  cache_aligned.h
  cluster.h
  codec.h
  codec_impl.cc
  codec_png.cc
  codec_png.h
  codec_pnm.cc
  codec_pnm.h
  color_correlation.cc
  color_correlation.h
  color_encoding.cc
  color_encoding.h
  color_management.cc
  color_management.h
  common.h
  compiler_specific.h
  compressed_image.cc
  compressed_image_fwd.h
  compressed_image.h
  context_map_decode.cc
  context_map_decode.h
  context_map_encode.cc
  context_map_encode.h
  convolve.h
  data_parallel.cc
  data_parallel.h
  dc_predictor.cc
  dc_predictor.h
  dct.cc
  dct.h
  dct_simd_4.h
  dct_simd_8.h
  dct_simd_any.h
  dct_util.cc
  dct_util.h
  decode_and_encode.cc
  deconvolve.cc
  deconvolve.h
  descriptive_statistics.cc
  descriptive_statistics.h
  entropy_coder.cc
  entropy_coder.h
  entropy_source.h
  epf.cc
  epf.h
  epf_stats.h
  epf_target.cc
  external_image.cc
  external_image.h
  fast_log.h
  field_encodings.h
  fields.h
  file_io.h
  gaborish.cc
  gaborish.h
  gamma_correct.h
  gauss_blur.cc
  gauss_blur.h
  gradient_map.cc
  gradient_map.h
  headers.cc
  headers.h
  huffman_decode.cc
  huffman_decode.h
  huffman_encode.cc
  huffman_encode.h
  image.cc
  image.h
  image_io.cc
  image_io.h
  jpeg_quant_tables.cc
  jpeg_quant_tables.h
  lehmer_code.cc
  lehmer_code.h
  linalg.cc
  linalg.h
  lossless16.cc
  lossless16.h
  lossless8.cc
  lossless8.h
  metadata.cc
  metadata.h
  multipass_handler.h
  noise.cc
  noise.h
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
  pik.cc
  pik.h
  pik_info.cc
  pik_info.h
  pik_multipass.cc
  pik_multipass.h
  pik_params.h
  pik_pass.cc
  pik_pass.h
  profiler.h
  quantizer.cc
  quantizer.h
  quant_weights.cc
  quant_weights.h
  rational_polynomial.h
  resample.h
  resize.h
  saliency_map.cc
  saliency_map.h
  single_image_handler.cc
  single_image_handler.h
  size_coder.h
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

target_compile_options(pikcommon PUBLIC -mavx2)
add_dependencies(pikcommon LCMS2)

target_link_libraries(pikcommon PRIVATE
  brotlicommon-static
  brotlienc-static
  brotlidec-static
  Eigen3::Eigen
  "${LCMS2_INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}lcms2${CMAKE_STATIC_LIBRARY_SUFFIX}"
  Threads::Threads
  "${CMAKE_DL_LIBS}"
)

set(BINARIES cpik dpik butteraugli_main)
foreach (BINARY IN LISTS BINARIES)
  add_executable("${BINARY}" "${BINARY}.cc")
  target_link_libraries("${BINARY}" pikcommon)
endforeach ()
install(TARGETS ${BINARIES} RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")

add_subdirectory(comparison_tool/viewer)
