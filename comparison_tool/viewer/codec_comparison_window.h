// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_COMPARISON_TOOL_VIEWER_CODEC_COMPARISON_WINDOW_H_
#define PIK_COMPARISON_TOOL_VIEWER_CODEC_COMPARISON_WINDOW_H_

#include <QDir>
#include <QMainWindow>
#include <QMap>
#include <QSet>
#include <QString>

#include "pik/padded_bytes.h"
#include "comparison_tool/viewer/ui_codec_comparison_window.h"

namespace pik {

class CodecComparisonWindow : public QMainWindow {
  Q_OBJECT

 public:
  explicit CodecComparisonWindow(QWidget* parent = nullptr);
  explicit CodecComparisonWindow(const QString& directory,
                                 QWidget* parent = nullptr);
  ~CodecComparisonWindow() override = default;

 private slots:
  void handleImageSetSelection(const QString& imageSetName);
  void handleImageSelection(const QString& imageName);

 private:
  struct ComparableImage {
    // Absolute path to the decoded PNG (or an image that Qt can read).
    QString decodedImagePath;
    // Size of the encoded image (*not* the PNG).
    qint64 byteSize = 0;
  };
  // Keys are compression levels.
  using Codec = QMap<QString, ComparableImage>;
  // Keys are codec names.
  using Codecs = QMap<QString, Codec>;
  // Keys are image names (relative to the image set directory).
  using ImageSet = QMap<QString, Codecs>;
  // Keys are paths to image sets (relative to the base directory chosen by the
  // user).
  using ImageSets = QMap<QString, ImageSet>;

  enum class Side { LEFT, RIGHT };

  QString pathToOriginalImage(const QString& imageSet,
                              const QString& imageName) const;
  ComparableImage currentlySelectedImage(Side side) const;

  void handleCodecChange(Side side);
  void updateSideImage(Side side);
  void matchSize(Side side);

  void loadDirectory(const QString& directory);
  // Recursive, called by loadDirectory.
  void browseDirectory(const QDir& directory, int depth = 0);

  Ui::CodecComparisonWindow ui_;

  QDir baseDirectory_;
  ImageSets imageSets_;
  QSet<QString> visited_;

  const PaddedBytes monitorIccProfile_;
};

}  // namespace pik

#endif  // PIK_COMPARISON_TOOL_VIEWER_CODEC_COMPARISON_WINDOW_H_
