// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "opsin_inverse.h"
#include "codec.h"
#include "image_test_utils.h"
#include "opsin_image.h"

#include "gtest/gtest.h"

namespace pik {
namespace {

TEST(OpsinInverseTest, LinearInverseInverts) {
  Image3F linear(128, 128);
  RandomFillImage(&linear, 255.0f);

  CodecContext codec_context;
  CodecInOut io(&codec_context);
  io.SetFromImage(CopyImage(linear), codec_context.c_linear_srgb[0]);
  Image3F opsin = OpsinDynamicsImage(&io, Rect(io.color()));

  OpsinToLinear(&opsin, /*pool=*/nullptr);

  VerifyRelativeError(linear, opsin, 3E-3, 2E-4);
}

}  // namespace
}  // namespace pik
