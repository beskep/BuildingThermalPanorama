"""파노라마 영상처리 GUI"""
# pylint: disable=wrong-import-position, ungrouped-imports

import os
import sys

from pano.interface.init import init_project

init_project(qt=True)

from loguru import logger
from PySide2 import QtCore
from PySide2 import QtGui
from PySide2 import QtQml

from pano import utils
from pano.interface.controller import Controller


def main(log_level=20):
  utils.set_logger(log_level)

  conf_path = utils.DIR.RESOURCE.joinpath('qtquickcontrols2.conf')
  qml_path = utils.DIR.RESOURCE.joinpath('qml/main.qml')

  for p in [conf_path, qml_path]:
    if not p.exists():
      raise FileNotFoundError(p)

  QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
  os.environ['QT_QUICK_CONTROLS_CONF'] = conf_path.as_posix()

  app = QtGui.QGuiApplication(sys.argv)
  engine = QtQml.QQmlApplicationEngine()
  context = engine.rootContext()

  controller = Controller()
  context.setContextProperty('con', controller)

  engine.load(qml_path.as_posix())
  root_objects = engine.rootObjects()
  if not root_objects:
    logger.error('Failed to load QML ({})', qml_path.as_posix())
    sys.exit()

  win: QtGui.QWindow = root_objects[0]
  controller.set_window(win)

  sys.exit(app.exec_())


if __name__ == '__main__':
  main()
