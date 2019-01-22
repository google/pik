# Copyright 2019 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

add_library(fse STATIC
  FiniteStateEntropy/lib/bitstream.h
  FiniteStateEntropy/lib/entropy_common.c
  FiniteStateEntropy/lib/error_private.h
  FiniteStateEntropy/lib/error_public.h
  FiniteStateEntropy/lib/fse.h
  FiniteStateEntropy/lib/fse_compress.c
  FiniteStateEntropy/lib/fse_decompress.c
  FiniteStateEntropy/lib/huf.h
  FiniteStateEntropy/lib/mem.h
  fse_error_wrapper.h
  fse_wrapper.h
)
target_include_directories(fse
    INTERFACE "${CMAKE_CURRENT_LIST_DIR}/")

# Force every .c file to include the wrapper.
target_compile_options(fse
    PRIVATE -include "${CMAKE_CURRENT_LIST_DIR}/fse_wrapper.h" -Wall -Werror)

# fse_decompress.c uses a "#define FSE_isError" so we can't use that define in
# our wrapper for that specific file. This removes the #define FSE_isError from
# our wrapper only on that file.
set_source_files_properties(FiniteStateEntropy/lib/fse_decompress.c PROPERTIES
    COMPILE_FLAGS "-include ${CMAKE_CURRENT_LIST_DIR}/fse_error_wrapper.h")
