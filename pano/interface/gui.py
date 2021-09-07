"""파노라마 영상처리 GUI"""
# pylint: disable=wrong-import-position, ungrouped-imports

from multiprocessing import freeze_support
import os
import sys

import click
from loguru import logger

from pano import utils
from pano.interface.common.init import init_project
from pano.interface.controller import Controller
from pano.interface.controller import RegistrationPlotController
from pano.interface.mbq import FigureCanvas
from pano.interface.mbq import QtCore
from pano.interface.mbq import QtGui
from pano.interface.mbq import QtQml

init_project(qt=True)

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


@click.command()
@click.option('-d', '--debug', is_flag=True)
@click.option('-l', '--loglevel', default=20)
def main(debug=False, loglevel=20):
  freeze_support()

  loglevel = min(loglevel, (10 if debug else 20))
  utils.set_logger(loglevel)

  if loglevel < 10:
    os.environ['QT_DEBUG_PLUGINS'] = '1'

  conf_path = utils.DIR.RESOURCE.joinpath('qtquickcontrols2.conf')
  qml_path = utils.DIR.RESOURCE.joinpath('qml/main.qml')

  for p in [conf_path, qml_path]:
    if not p.exists():
      raise FileNotFoundError(p)

  QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
  QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)
  os.environ['QT_QUICK_CONTROLS_CONF'] = conf_path.as_posix()

  QtQml.qmlRegisterType(FigureCanvas, 'Backend', 1, 0, 'RegistrationCanvas')

  app = QtGui.QGuiApplication(sys.argv)
  engine = QtQml.QQmlApplicationEngine(qml_path.as_posix())

  root_objects = engine.rootObjects()
  if not root_objects:
    logger.critical('Failed to load QML {}', qml_path)
    return -1

  win: QtGui.QWindow = root_objects[0]

  controller = Controller(win, loglevel)
  context = engine.rootContext()
  context.setContextProperty('con', controller)

  canvas = win.findChild(FigureCanvas, 'registration_plot')
  rpc = RegistrationPlotController()
  rpc.init(app, canvas)
  controller.rpc = rpc

  return app.exec_()


if __name__ == '__main__':
  main()
