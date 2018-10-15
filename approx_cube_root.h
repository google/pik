// Copyright 2017 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef APPROX_CUBE_ROOT_H_
#define APPROX_CUBE_ROOT_H_

#include <string.h>

#include "compiler_specific.h"

namespace pik {

PIK_INLINE float CubeRootInitialGuess(float y) {
  int ix;
  memcpy(&ix, &y, sizeof(ix));
  // At this point, ix is the integer value corresponding to the binary
  // representation of the floating point value y. Inspired by the well-known
  // floating-point recipe for 1/sqrt(y), which takes an initial guess in the
  // form of <magic constant> - ix / 2, our initial guess has the form
  // <magic constant> + ix / 3. Since we know the set of all floating
  // point values that will be the input of the cube root function in pik (see
  // LinearToXyb() in opsin_image.cc), we can search for the magic constant that
  // gives the minimum worst-case error. The chosen value here is optimal among
  // the magic constants whose 8 least significant bits are zero.
  ix = 0x2a50f200 + ix / 3;
  float x;
  memcpy(&x, &ix, sizeof(x));
  return x;
}

PIK_INLINE float CubeRootNewtonStep(float y, float xn) {
  constexpr float kOneThird = 1.0f / 3.0f;
  // f(x) = x^3 - y
  // x_{n+1} = x_n - f(x_n) / f'(x_n) =
  //         = x_n - (x_n^3 - y) / (3 * x_n^2) =
  //         = 2/3 * x_n + 1/3 * y / x_n^2
  return kOneThird * (2.0f * xn + y / (xn * xn));
}

// Returns an approximation of the cube root of y,
// with an accuracy of about 1e-6 for 0 <= y <= 1.
PIK_INLINE float ApproxCubeRoot(float y) {
  const float x0 = CubeRootInitialGuess(y);
  const float x1 = CubeRootNewtonStep(y, x0);
  const float x2 = CubeRootNewtonStep(y, x1);
  return x2;
}

}  // namespace pik

#endif  // APPROX_CUBE_ROOT_H_
