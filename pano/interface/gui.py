"""파노라마 영상처리 GUI"""

from multiprocessing import freeze_support
import os
import sys

import click
from loguru import logger

from pano import utils
from pano.interface.common.init import init_project
from pano.interface.controller import Controller
from pano.interface.mbq import FigureCanvas
from pano.interface.mbq import QtCore
from pano.interface.mbq import QtGui
from pano.interface.mbq import QtQml
from pano.interface.plot_controller import PlotControllers

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


CONF_PATH = utils.DIR.RESOURCE.joinpath('qtquickcontrols2.conf')
QML_PATH = utils.DIR.RESOURCE.joinpath('qml/main.qml')

for p in [CONF_PATH, QML_PATH]:
  p.stat()


@click.command(context_settings={
    'allow_extra_args': True,
    'ignore_unknown_options': True
})
@click.option('-d', '--debug', is_flag=True)
@click.option('-l', '--loglevel', default=20)
def main(debug=False, loglevel=20):
  freeze_support()

  loglevel = min(loglevel, (10 if debug else 20))
  utils.set_logger(loglevel)

  if loglevel < 10:
    os.environ['QT_DEBUG_PLUGINS'] = '1'

  QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
  QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)
  os.environ['QT_QUICK_CONTROLS_CONF'] = CONF_PATH.as_posix()

  QtQml.qmlRegisterType(FigureCanvas, 'Backend', 1, 0, 'FigureCanvas')

  app = QtGui.QGuiApplication(sys.argv)
  engine = QtQml.QQmlApplicationEngine(QML_PATH.as_posix())

  root_objects = engine.rootObjects()
  if not root_objects:
    logger.critical('Failed to load QML {}', QML_PATH)
    return -1

  win: QtGui.QWindow = root_objects[0]

  controller = Controller(win, loglevel)
  context = engine.rootContext()
  context.setContextProperty('con', controller)

  pcs = {}
  for name, cls in PlotControllers.classes():
    canvas = win.findChild(FigureCanvas, f'{name}_plot')
    pc = cls()
    pc.init(app, canvas)
    pcs[name] = pc

  controller.set_plot_controllers(pcs)

  return app.exec_()


if __name__ == '__main__':
  main()
