// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "comparison_tool/viewer/image_loading.h"

#include <QRgb>
#include <QThread>

#include "pik/codec.h"

namespace pik {

QImage loadImage(const QString& filename, PaddedBytes targetIccProfile,
                 const QString& sourceColorSpaceHint) {
  static pik::CodecContext codecContext;
  static pik::ThreadPool pool(QThread::idealThreadCount());

  pik::CodecInOut decoder(&codecContext);
  if (!sourceColorSpaceHint.isEmpty()) {
    decoder.dec_hints.Add("color_space", sourceColorSpaceHint.toStdString());
  }
  if (!decoder.SetFromFile(filename.toStdString(), &pool)) {
    return QImage();
  }

  pik::ColorEncoding targetColorSpace;
  if (!ColorManagement::SetFromProfile(std::move(targetIccProfile),
                                       &targetColorSpace)) {
    targetColorSpace = codecContext.c_srgb[decoder.IsGray()];
  }
  pik::Image3B converted;
  if (!decoder.CopyTo(Rect(decoder.color()), targetColorSpace, &converted,
                      &pool)) {
    return QImage();
  }

  QImage image(converted.xsize(), converted.ysize(), QImage::Format_ARGB32);

  if (decoder.HasAlpha()) {
    const int alphaRightShiftAmount = static_cast<int>(decoder.AlphaBits()) - 8;
    for (int y = 0; y < image.height(); ++y) {
      QRgb* const row = reinterpret_cast<QRgb*>(image.scanLine(y));
      const uint16_t* const alphaRow = decoder.alpha().ConstRow(y);
      const uint8_t* const redRow = converted.ConstPlaneRow(0, y);
      const uint8_t* const greenRow = converted.ConstPlaneRow(1, y);
      const uint8_t* const blueRow = converted.ConstPlaneRow(2, y);
      for (int x = 0; x < image.width(); ++x) {
        row[x] = qRgba(redRow[x], greenRow[x], blueRow[x],
                       alphaRow[x] >> alphaRightShiftAmount);
      }
    }
  } else {
    for (int y = 0; y < image.height(); ++y) {
      QRgb* const row = reinterpret_cast<QRgb*>(image.scanLine(y));
      const uint8_t* const redRow = converted.ConstPlaneRow(0, y);
      const uint8_t* const greenRow = converted.ConstPlaneRow(1, y);
      const uint8_t* const blueRow = converted.ConstPlaneRow(2, y);
      for (int x = 0; x < image.width(); ++x) {
        row[x] = qRgb(redRow[x], greenRow[x], blueRow[x]);
      }
    }
  }

  return image;
}

}  // namespace pik
