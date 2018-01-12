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

#ifndef DCT_H_
#define DCT_H_

namespace pik {

// Computes the in-place 8x8 DCT of block.
// Requires that block is 32-bytes aligned.
//
// The DCT used is a scaled variant of DCT-II, which is orthonormal:
//
// G(u,v) =
//     (1/4)alpha(u)alpha(v)*
//     sum_{x,y}(g(x,y)*cos((2x+1)uM_PI/16)*cos((2y+1)vM_PI/16))
//
// where alpha(u) is 1/sqrt(2) if u = 0 and 1 otherwise, g_{x,y} is the pixel
// value at coordiantes (x,y) and G(u,v) is the DCT coefficient at spatial
// frequency (u,v).
void ComputeBlockDCTFloat(float block[64]);

// Computes the in-place 8x8 inverse DCT of block.
// Requires that block is 32-bytes aligned.
void ComputeBlockIDCTFloat(float block[64]);

// Final scaling factors of outputs/inputs in the Arai, Agui, and Nakajima
// algorithm computing the DCT/IDCT.
// The algorithm is described in the book JPEG: Still Image Data Compression
// Standard, section 4.3.5.
static const float kIDCTScales[8] = {
    0.3535533906f, 0.4903926402f, 0.4619397663f, 0.4157348062f,
    0.3535533906f, 0.2777851165f, 0.1913417162f, 0.0975451610f};

static const float kRecipIDCTScales[8] = {
    1.0 / 0.3535533906f, 1.0 / 0.4903926402f, 1.0 / 0.4619397663f,
    1.0 / 0.4157348062f, 1.0 / 0.3535533906f, 1.0 / 0.2777851165f,
    1.0 / 0.1913417162f, 1.0 / 0.0975451610f};

// Same as ComputeBlockDCTFloat(), but the output is further transformed with
// the following:
//   block'[8 * ky + kx] =
//     block[8 * kx + ky] * 64.0 * kIDCTScales[kx] * kIDCTScales[ky]
// Requires that block is 32-bytes aligned.
void ComputeTransposedScaledBlockDCTFloat(float block[64]);

// Same as ComputeBlockIDCTFloat(), but the input is first transformed with
// the following:
//   block'[8 * ky + kx] =
//     block[8 * kx + ky] / (kIDCTScales[kx] * kIDCTScales[ky])
// Requires that block is 32-bytes aligned.
void ComputeTransposedScaledBlockIDCTFloat(float block[64]);

void TransposeBlock(float block[64]);

}  // namespace pik

#endif  // DCT_H_
