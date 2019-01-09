// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef COMPARISON_TOOL_VIEWER_SPLIT_IMAGE_VIEW_H_
#define COMPARISON_TOOL_VIEWER_SPLIT_IMAGE_VIEW_H_

#include <QWidget>

#include "ui_split_image_view.h"

namespace pik {

class SplitImageView : public QWidget {
  Q_OBJECT

 public:
  explicit SplitImageView(QWidget* parent = nullptr);
  ~SplitImageView() override = default;

  void setLeftImage(QImage image);
  void setRightImage(QImage image);
  void setMiddleImage(QImage image);

 signals:
  void renderingModeChanged(SplitImageRenderingMode newMode);

 private:
  Ui::SplitImageView ui_;
};

}  // namespace pik

#endif  // COMPARISON_TOOL_VIEWER_SPLIT_IMAGE_VIEW_H_
