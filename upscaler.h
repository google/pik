#ifndef UPSCALER_H_
#define UPSCALER_H_

#include "image.h"

namespace pik {

Image3F UpscalerReconstruct(const Image3F& dc);

Image3F Blur(const Image3F& image, float sigma);

}  // namespace pik

#endif  // UPSCALER_H_
