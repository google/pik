// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_COMPARISON_TOOL_VIEWER_IMAGE_LOADING_H_
#define PIK_COMPARISON_TOOL_VIEWER_IMAGE_LOADING_H_

#include <QImage>
#include <QString>

#include "pik/padded_bytes.h"

namespace pik {

// Converts the loaded image to the given display profile, or sRGB if not
// specified. Thread-hostile.
QImage loadImage(const QString& filename,
                 PaddedBytes targetIccProfile = PaddedBytes(),
                 const QString& sourceColorSpaceHint = QString());

}  // namespace pik

#endif  // PIK_COMPARISON_TOOL_VIEWER_IMAGE_LOADING_H_
