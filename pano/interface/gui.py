"""파노라마 영상처리 GUI"""
# pylint: disable=wrong-import-position

import os
from pathlib import Path
import sys

from loguru import logger
import matplotlib.pyplot
import PySide2
from PySide2 import QtCore
from PySide2 import QtGui
from PySide2 import QtQml
import skimage.io

from pano import utils
from pano.interface.controller import Controller

plugins_path = Path(PySide2.__file__).parent.joinpath('plugins')
os.environ['QT_PLUGIN_PATH'] = plugins_path.as_posix()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
skimage.io.use_plugin('pil')


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
