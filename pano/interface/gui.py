"""파노라마 영상처리 GUI"""
# pylint: disable=wrong-import-position, ungrouped-imports

import os
import sys

from pano.interface.common.init import init_project

init_project(qt=True)

from loguru import logger
from matplotlib_backend_qtquick.backend_qtquickagg import FigureCanvas
from PySide2 import QtCore
from PySide2 import QtGui
from PySide2 import QtQml
from PySide2 import QtSvg  # for matplotlib-backend-qtquick

from pano import utils
from pano.interface.controller import Controller
from pano.interface.controller import RegistrationPlotController

_qt_message = {
    QtCore.QtMsgType.QtDebugMsg: 'DEBUG',
    QtCore.QtMsgType.QtInfoMsg: 'INFO',
    QtCore.QtMsgType.QtWarningMsg: 'WARNING',
    QtCore.QtMsgType.QtCriticalMsg: 'ERROR',
    QtCore.QtMsgType.QtSystemMsg: 'ERROR',
    QtCore.QtMsgType.QtFatalMsg: 'CRITICAL',
}


def _qt_message_handler(mode, context, message):
  level = _qt_message.get(mode, 'INFO')
  logger.log(level, message)


def main(log_level=20):
  utils.set_logger(log_level)

  conf_path = utils.DIR.RESOURCE.joinpath('qtquickcontrols2.conf')
  qml_path = utils.DIR.RESOURCE.joinpath('qml/main.qml')

  for p in [conf_path, qml_path]:
    if not p.exists():
      raise FileNotFoundError(p)

  QtCore.qInstallMessageHandler(_qt_message_handler)
  QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
  QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)
  os.environ['QT_QUICK_CONTROLS_CONF'] = conf_path.as_posix()

  QtQml.qmlRegisterType(FigureCanvas, 'Backend', 1, 0, 'RegistrationCanvas')

  app = QtGui.QGuiApplication(sys.argv)
  engine = QtQml.QQmlApplicationEngine(qml_path.as_posix())

  root_objects = engine.rootObjects()
  if not root_objects:
    logger.critical('Failed to load QML {}', qml_path)
    return

  win: QtGui.QWindow = root_objects[0]

  controller = Controller(win)
  context = engine.rootContext()
  context.setContextProperty('con', controller)

  canvas = win.findChild(FigureCanvas, 'registration_plot')
  rpc = RegistrationPlotController()
  rpc.init(app, canvas)
  controller.rpc = rpc

  sys.exit(app.exec_())


if __name__ == '__main__':
  main(log_level=10)
