// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef X11_ICC_H_
#define X11_ICC_H_

#include <xcb/xcb.h>

#include "padded_bytes.h"

namespace pik {

// Should be cached if possible.
PaddedBytes GetMonitorIccProfile(xcb_connection_t* connection,
                                 int screen_number = 0);

}  // namespace pik

#endif  // X11_ICC_H_
