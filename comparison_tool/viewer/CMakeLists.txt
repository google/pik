# Copyright 2019 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

find_package(Qt5 COMPONENTS Concurrent Widgets X11Extras)
if (NOT Qt5_FOUND)
  message(WARNING "Qt5 was not found. The comparison tool will not be built.")
  return()
endif ()

find_package(ECM NO_MODULE)
if (NOT ECM_FOUND)
  message(WARNING "extra-cmake-modules were not found. The comparison tool will not be built.")
  return()
endif ()
set(CMAKE_MODULE_PATH ${ECM_FIND_MODULE_DIR})

find_package(XCB COMPONENTS XCB)
if (NOT XCB_FOUND)
  message(WARNING "XCB was not found. The comparison tool will not be built.")
  return()
endif ()

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)

add_library(x11_icc STATIC
  ../../x11/icc.cc
  ../../x11/icc.h
)
target_link_libraries(x11_icc PUBLIC pikcommon XCB::XCB)

add_library(image_loading STATIC
  image_loading.cc
  image_loading.h
)
target_link_libraries(image_loading PUBLIC Qt5::Widgets pikcommon)

add_executable(compare_codecs WIN32
  codec_comparison_window.cc
  codec_comparison_window.h
  codec_comparison_window.ui
  compare_codecs.cc
  split_image_renderer.cc
  split_image_renderer.h
  split_image_view.cc
  split_image_view.h
  split_image_view.ui
)
target_link_libraries(compare_codecs
  image_loading
  Qt5::Concurrent
  Qt5::Widgets
  Qt5::X11Extras
  x11_icc
)

add_executable(compare_images WIN32
  compare_images.cc
  split_image_renderer.cc
  split_image_renderer.h
  split_image_view.cc
  split_image_view.h
  split_image_view.ui
)
target_link_libraries(compare_images
  image_loading
  Qt5::Widgets
  Qt5::X11Extras
  x11_icc
)
