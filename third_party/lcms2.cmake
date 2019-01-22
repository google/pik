# Copyright 2019 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

add_library(lcms2 STATIC
  lcms/src/cmsalpha.c
  lcms/src/cmscam02.c
  lcms/src/cmscgats.c
  lcms/src/cmscnvrt.c
  lcms/src/cmserr.c
  lcms/src/cmsgamma.c
  lcms/src/cmsgmt.c
  lcms/src/cmshalf.c
  lcms/src/cmsintrp.c
  lcms/src/cmsio0.c
  lcms/src/cmsio1.c
  lcms/src/cmslut.c
  lcms/src/cmsmd5.c
  lcms/src/cmsmtrx.c
  lcms/src/cmsnamed.c
  lcms/src/cmsopt.c
  lcms/src/cmspack.c
  lcms/src/cmspcs.c
  lcms/src/cmsplugin.c
  lcms/src/cmsps2.c
  lcms/src/cmssamp.c
  lcms/src/cmssm.c
  lcms/src/cmstypes.c
  lcms/src/cmsvirt.c
  lcms/src/cmswtpnt.c
  lcms/src/cmsxform.c
  lcms/src/lcms2_internal.h
)
target_include_directories(lcms2
    PUBLIC "${CMAKE_CURRENT_LIST_DIR}/lcms/include")
