#ifndef COMPARISON_TOOL_VIEWER_SPLIT_IMAGE_RENDERER_H_
#define COMPARISON_TOOL_VIEWER_SPLIT_IMAGE_RENDERER_H_

#include <QImage>
#include <QKeyEvent>
#include <QMouseEvent>
#include <QPaintEvent>
#include <QWheelEvent>
#include <QWidget>

namespace pik {

enum class SplitImageRenderingMode {
  // The default mode when using the mouse: one (partial) image is shown on each
  // side of the cursor, with a vertical band of the middle image if applicable.
  SPLIT,
  // Only show the left image (accessed by pressing the left arrow key when the
  // renderer has focus).
  LEFT,
  // Only show the right image (accessed by pressing the right arrow key).
  RIGHT,
  // Only show the middle image (accessed by pressing the up or down arrow key).
  MIDDLE,
};

class SplitImageRenderer : public QWidget {
  Q_OBJECT

 public:
  explicit SplitImageRenderer(QWidget* parent = nullptr);
  ~SplitImageRenderer() override = default;

  QSize sizeHint() const override { return minimumSize(); }

  void setLeftImage(QImage image);
  void setRightImage(QImage image);
  void setMiddleImage(QImage image);

 public slots:
  void setMiddleWidthPercent(int percent);
  void setZoomLevel(double scale);

 signals:
  void zoomLevelIncreaseRequested();
  void zoomLevelDecreaseRequested();

  void renderingModeChanged(SplitImageRenderingMode newMode);

 protected:
  void keyPressEvent(QKeyEvent* event) override;
  void mouseMoveEvent(QMouseEvent* event) override;
  void wheelEvent(QWheelEvent* event) override;
  void paintEvent(QPaintEvent* event) override;

 private:
  void updateMinimumSize();

  QImage leftImage_, rightImage_, middleImage_;
  SplitImageRenderingMode mode_ = SplitImageRenderingMode::SPLIT;
  int middleX_ = 0;
  int middleWidthPercent_ = 10;
  double scale_ = 1.;
};

}  // namespace pik

#endif  // COMPARISON_TOOL_VIEWER_SPLIT_IMAGE_RENDERER_H_
