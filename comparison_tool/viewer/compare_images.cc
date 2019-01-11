// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include <cstdlib>

#include <QApplication>
#include <QCommandLineOption>
#include <QCommandLineParser>
#include <QFlags>
#include <QImage>
#include <QMessageBox>
#include <QStringList>
#include <QX11Info>

#include "../../x11/icc.h"
#include "image_loading.h"
#include "split_image_view.h"

namespace {

void displayLoadingError(const QString& path) {
  QMessageBox message;
  message.setIcon(QMessageBox::Critical);
  message.setWindowTitle(
      QCoreApplication::translate("SplitImageView", "Error"));
  message.setText(QCoreApplication::translate("SplitImageView",
                                              "Could not load image \"%1\".")
                      .arg(path));
  message.exec();
}

}  // namespace

int main(int argc, char** argv) {
  QApplication application(argc, argv);

  QCommandLineParser parser;
  parser.setApplicationDescription(
      QCoreApplication::translate("compare_images", "Image comparison tool"));
  parser.addHelpOption();
  parser.addPositionalArgument(
      "left-image",
      QCoreApplication::translate("compare_images",
                                  "The image to display on the left."),
      "<left-image>");
  parser.addPositionalArgument(
      "right-image",
      QCoreApplication::translate("compare_images",
                                  "The image to display on the right."),
      "<right-image>");
  parser.addPositionalArgument(
      "middle-image",
      QCoreApplication::translate(
          "compare_images", "The image to display in the middle (optional)."),
      "[<middle-image>]");

  QCommandLineOption colorSpaceOption(
      {"color-space", "c"},
      QCoreApplication::translate(
          "compare_images",
          "The color space to use for untagged images (typically PNM)."),
      QCoreApplication::translate("compare_images", "color-space"));
  parser.addOption(colorSpaceOption);

  parser.process(application);

  const QString colorSpaceHint = parser.value(colorSpaceOption);

  QStringList arguments = parser.positionalArguments();
  if (arguments.size() < 2 || arguments.size() > 3) {
    parser.showHelp(EXIT_FAILURE);
  }

  pik::SplitImageView view;

  const pik::PaddedBytes monitorIccProfile =
      pik::GetMonitorIccProfile(QX11Info::connection(), QX11Info::appScreen());

  const QString leftImagePath = arguments.takeFirst();
  QImage leftImage =
      pik::loadImage(leftImagePath, monitorIccProfile, colorSpaceHint);
  if (leftImage.isNull()) {
    displayLoadingError(leftImagePath);
    return EXIT_FAILURE;
  }
  view.setLeftImage(std::move(leftImage));

  const QString rightImagePath = arguments.takeFirst();
  QImage rightImage =
      pik::loadImage(rightImagePath, monitorIccProfile, colorSpaceHint);
  if (rightImage.isNull()) {
    displayLoadingError(rightImagePath);
    return EXIT_FAILURE;
  }
  view.setRightImage(std::move(rightImage));

  if (!arguments.empty()) {
    const QString middleImagePath = arguments.takeFirst();
    QImage middleImage =
        pik::loadImage(middleImagePath, monitorIccProfile, colorSpaceHint);
    if (middleImage.isNull()) {
      displayLoadingError(middleImagePath);
      return EXIT_FAILURE;
    }
    view.setMiddleImage(std::move(middleImage));
  }

  view.setWindowFlags(view.windowFlags() | Qt::Window);
  view.setWindowState(Qt::WindowMaximized);
  view.show();

  return application.exec();
}
