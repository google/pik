#include <cstdlib>

#include <QApplication>
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

  QStringList arguments = application.arguments();
  arguments.removeFirst();  // program name
  if (arguments.size() < 2) {
    QMessageBox message;
    message.setIcon(QMessageBox::Information);
    message.setWindowTitle(
        QCoreApplication::translate("SplitImageView", "Usage"));
    message.setText(QCoreApplication::translate(
        "SplitImageView", "Please pass at least two images to this tool."));
    message.setInformativeText(
        QCoreApplication::translate("SplitImageView",
                                    "A third image can optionally be passed. "
                                    "It will be displayed in the middle."));
    message.exec();
    return EXIT_FAILURE;
  }

  pik::SplitImageView view;

  const pik::PaddedBytes monitorIccProfile =
      pik::GetMonitorIccProfile(QX11Info::connection(), QX11Info::appScreen());

  const QString leftImagePath = arguments.takeFirst();
  QImage leftImage = pik::loadImage(leftImagePath, monitorIccProfile);
  if (leftImage.isNull()) {
    displayLoadingError(leftImagePath);
    return EXIT_FAILURE;
  }
  view.setLeftImage(std::move(leftImage));

  const QString rightImagePath = arguments.takeFirst();
  QImage rightImage = pik::loadImage(rightImagePath, monitorIccProfile);
  if (rightImage.isNull()) {
    displayLoadingError(rightImagePath);
    return EXIT_FAILURE;
  }
  view.setRightImage(std::move(rightImage));

  if (!arguments.empty()) {
    const QString middleImagePath = arguments.takeFirst();
    QImage middleImage = pik::loadImage(middleImagePath, monitorIccProfile);
    if (middleImage.isNull()) {
      displayLoadingError(middleImagePath);
      return EXIT_FAILURE;
    }
    view.setMiddleImage(std::move(middleImage));
  }

  view.setWindowFlags(view.windowFlags() | Qt::Window);
  view.show();

  return application.exec();
}
