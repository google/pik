#ifndef COMPARISON_TOOL_VIEWER_IMAGE_LOADING_H_
#define COMPARISON_TOOL_VIEWER_IMAGE_LOADING_H_

#include <QImage>
#include <QString>

#include "../../padded_bytes.h"

namespace pik {

// Converts the loaded image to the given display profile, or sRGB if not
// specified. Thread-hostile.
QImage loadImage(const QString& filename,
                 PaddedBytes targetIccProfile = PaddedBytes());

}  // namespace pik

#endif  // COMPARISON_TOOL_VIEWER_IMAGE_LOADING_H_
