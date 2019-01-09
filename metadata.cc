// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "metadata.h"

#include "fields.h"

namespace pik {

Transcoded::Transcoded() { Bundle::Init(this); }
Metadata::Metadata() { Bundle::Init(this); }

}  // namespace pik
