// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "gaborish.h"

#include "gtest/gtest.h"

namespace pik {
namespace {

void TestRoundTrip(const Image3F& in, double max_l1) {
  for (GaborishStrength strength :
       {GaborishStrength::k1000, GaborishStrength::k875,
        GaborishStrength::k750, GaborishStrength::k500}) {
    Image3F fwd;
    ASSERT_TRUE((strength != GaborishStrength::kOff) ==
        ConvolveGaborish(in, strength, /*pool=*/nullptr, &fwd));
    const Image3F rev = GaborishInverse(fwd, 0.92718927264540152);
    VerifyRelativeError(in, rev, max_l1, 1E-4);
  }
}

TEST(GaborishTest, TestZero) {
  Image3F in(20, 20);
  ZeroFillImage(&in);
  TestRoundTrip(in, 0.0);
}

// Disabled: large difference.
#if 0
TEST(GaborishTest, TestDirac) {
  Image3F in(20, 20);
  ZeroFillImage(&in);
  in.PlaneRow(1, 10)[10] = 10.0f;
  TestRoundTrip(in, 0.26);
}
#endif

TEST(GaborishTest, TestFlat) {
  Image3F in(20, 20);
  FillImage(1.0f, &in);
  TestRoundTrip(in, 1E-5);
}

}  // namespace
}  // namespace pik
