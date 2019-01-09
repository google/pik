// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include <cstdlib>

#include <QApplication>
#include <QMessageBox>
#include <QString>
#include <QStringList>

#include "codec_comparison_window.h"

namespace {

template <typename Window, typename... Arguments>
void displayNewWindow(Arguments&&... arguments) {
  Window* const window = new Window(arguments...);
  window->setAttribute(Qt::WA_DeleteOnClose);
  window->show();
}

}  // namespace

int main(int argc, char** argv) {
  QApplication application(argc, argv);

  QStringList arguments = application.arguments();
  arguments.removeFirst();  // program name

  if (arguments.empty()) {
    QMessageBox message;
    message.setIcon(QMessageBox::Information);
    message.setWindowTitle(
        QCoreApplication::translate("CodecComparisonWindow", "Usage"));
    message.setText(QCoreApplication::translate(
        "CodecComparisonWindow", "Please specify a directory to use."));
    message.setDetailedText(QCoreApplication::translate(
        "CodecComparisonWindow",
        "That directory should contain images in the following layout:\n"
        "- .../<image name>/original.png (optional)\n"
        "- .../<image_name>/<codec_name>/<compression_level>.<ext>\n"
        "- .../<image_name>/<codec_name>/<compression_level>.png (optional for "
        "formats that Qt can load)\n"
        "With arbitrary nesting allowed before that. (The \"...\" part is "
        "referred to as an \"image set\" by the tool."));
    message.exec();
    return EXIT_FAILURE;
  }

  for (const QString& argument : arguments) {
    displayNewWindow<pik::CodecComparisonWindow>(argument);
  }

  return application.exec();
}
