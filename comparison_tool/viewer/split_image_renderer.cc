#include "split_image_renderer.h"

#include <algorithm>
#include <cmath>
#include <utility>

#include <QEvent>
#include <QGuiApplication>
#include <QPainter>
#include <QPalette>
#include <QPen>
#include <QPoint>
#include <QRect>

namespace pik {

SplitImageRenderer::SplitImageRenderer(QWidget* const parent)
    : QWidget(parent) {
  setBackgroundRole(QPalette::Shadow);
  setAutoFillBackground(true);
  setMouseTracking(true);
  setFocusPolicy(Qt::WheelFocus);
}

void SplitImageRenderer::setLeftImage(QImage image) {
  leftImage_ = std::move(image);
  updateMinimumSize();
  update();
}
void SplitImageRenderer::setRightImage(QImage image) {
  rightImage_ = std::move(image);
  updateMinimumSize();
  update();
}
void SplitImageRenderer::setMiddleImage(QImage image) {
  middleImage_ = std::move(image);
  updateMinimumSize();
  update();
}

void SplitImageRenderer::setMiddleWidthPercent(const int percent) {
  middleWidthPercent_ = percent;
  update();
}

void SplitImageRenderer::setZoomLevel(double scale) {
  scale_ = scale;
  updateMinimumSize();
  update();
}

void SplitImageRenderer::keyPressEvent(QKeyEvent* const event) {
  switch (event->key()) {
    case Qt::Key_Left:
      mode_ = SplitImageRenderingMode::LEFT;
      emit renderingModeChanged(mode_);
      break;

    case Qt::Key_Right:
      mode_ = SplitImageRenderingMode::RIGHT;
      emit renderingModeChanged(mode_);
      break;

    case Qt::Key_Up:
    case Qt::Key_Down:
      mode_ = SplitImageRenderingMode::MIDDLE;
      emit renderingModeChanged(mode_);
      break;

    case Qt::Key_ZoomIn:
      emit zoomLevelIncreaseRequested();
      break;
    case Qt::Key_ZoomOut:
      emit zoomLevelDecreaseRequested();
      break;

    default:
      QWidget::keyPressEvent(event);
      break;
  }
  update();
}

void SplitImageRenderer::mouseMoveEvent(QMouseEvent* const event) {
  mode_ = SplitImageRenderingMode::SPLIT;
  emit renderingModeChanged(mode_);
  middleX_ = event->pos().x();
  update();
}

void SplitImageRenderer::wheelEvent(QWheelEvent* event) {
  if (QGuiApplication::keyboardModifiers().testFlag(Qt::ControlModifier)) {
    if (event->angleDelta().y() > 0) {
      emit zoomLevelIncreaseRequested();
      return;
    } else if (event->angleDelta().y() < 0) {
      emit zoomLevelDecreaseRequested();
      return;
    }
  }

  event->ignore();
}

void SplitImageRenderer::paintEvent(QPaintEvent* const event) {
  QRectF drawingArea(0., 0., minimumWidth(), minimumHeight());

  QPainter painter(this);
  painter.translate(QRectF(rect()).center() - drawingArea.center());
  painter.scale(scale_, scale_);
  if (scale_ < 1.) {
    painter.setRenderHint(QPainter::SmoothPixmapTransform);
  }
  const double transformedMiddleX =
      painter.worldTransform().inverted().map(QPointF(middleX_, 0.)).x();

  switch (mode_) {
    case SplitImageRenderingMode::LEFT:
      painter.drawImage(QPointF(0., 0.), leftImage_);
      return;
    case SplitImageRenderingMode::RIGHT:
      painter.drawImage(QPointF(0., 0.), rightImage_);
      return;
    case SplitImageRenderingMode::MIDDLE:
      painter.drawImage(QPointF(0., 0.), middleImage_);
      return;

    default:
      break;
  }

  const qreal middleWidth =
      std::min<qreal>((minimumWidth() / scale_) * middleWidthPercent_ / 100.,
                      middleImage_.width());

  QRectF middleRect = middleImage_.rect();
  middleRect.setWidth(middleWidth);
  middleRect.moveCenter(QPointF(transformedMiddleX, middleRect.center().y()));
  middleRect.setLeft(std::round(middleRect.left()));
  middleRect.setRight(std::round(middleRect.right()));

  QRectF leftRect = leftImage_.rect();
  leftRect.setRight(middleRect.left());

  QRectF rightRect = rightImage_.rect();
  rightRect.setLeft(middleRect.right());

  painter.drawImage(leftRect, leftImage_, leftRect);
  painter.drawImage(rightRect, rightImage_, rightRect);
  painter.drawImage(middleRect, middleImage_, middleRect);

  QPen middlePen;
  middlePen.setStyle(Qt::DotLine);
  painter.setPen(middlePen);
  painter.drawLine(leftRect.topRight(), leftRect.bottomRight());
  painter.drawLine(rightRect.topLeft(), rightRect.bottomLeft());
}

void SplitImageRenderer::updateMinimumSize() {
  const int imagesWidth = std::max(
      std::max(leftImage_.width(), rightImage_.width()), middleImage_.width());
  const int imagesHeight =
      std::max(std::max(leftImage_.height(), rightImage_.height()),
               middleImage_.height());
  setMinimumSize(scale_ * QSize(imagesWidth, imagesHeight));
}

}  // namespace pik
