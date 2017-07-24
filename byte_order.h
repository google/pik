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

#ifndef BYTE_ORDER_H_
#define BYTE_ORDER_H_

#include "compiler_specific.h"

#if PIK_COMPILER_MSVC
#include <intrin.h>  // _byteswap_*
#else
#include <x86intrin.h>
#endif

#if (defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__))
#define PIK_BYTE_ORDER_LITTLE 1
#else
// This means that we don't know that the byte order is little endian, in
// this case we use endian-neutral code that works for both little- and
// big-endian.
#define PIK_BYTE_ORDER_LITTLE 0
#endif

#if PIK_COMPILER_MSVC
#define PIK_BSWAP32(x) _byteswap_ulong(x)
#define PIK_BSWAP64(x) _byteswap_uint64(x)
#else
#define PIK_BSWAP32(x) __builtin_bswap32(x)
#define PIK_BSWAP64(x) __builtin_bswap64(x)
#endif

#endif  // BYTE_ORDER_H_
