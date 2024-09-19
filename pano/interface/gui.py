"""파노라마 영상처리 GUI"""

# pylint: disable=wrong-import-position
# ruff: noqa: E402

import os
import sys
from multiprocessing import freeze_support
from typing import Annotated

from cyclopts import App, Parameter
from loguru import logger

from pano.interface.common.init import init_project

init_project(qt=True)

from pano import utils
from pano.interface.controller.panorama import Controller
from pano.interface.mbq import FigureCanvas, QtCore, QtGui, QtQml
from pano.interface.plot_controller import PlotControllers

_qt_message = {
  QtCore.QtMsgType.QtDebugMsg: 'DEBUG',
  QtCore.QtMsgType.QtInfoMsg: 'INFO',
  QtCore.QtMsgType.QtWarningMsg: 'WARNING',
  QtCore.QtMsgType.QtCriticalMsg: 'ERROR',
  QtCore.QtMsgType.QtSystemMsg: 'ERROR',
  QtCore.QtMsgType.QtFatalMsg: 'CRITICAL',
}


def _qt_message_handler(mode, _context, message):
  level = _qt_message.get(mode, 'INFO')
  logger.log(level, message)


class Files:
  CONF = utils.DIR.QT / 'QtQuickCtrlsPano.conf'
  QML = utils.DIR.QT / 'qml/Panorama.qml'


def _init(loglevel):
  utils.set_logger(loglevel)

  if loglevel < 10:  # noqa: PLR2004
    os.environ['QT_DEBUG_PLUGINS'] = '1'

  if loglevel < 5:  # noqa: PLR2004
    QtCore.qInstallMessageHandler(_qt_message_handler)

  QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
  QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)
  os.environ['QT_QUICK_CONTROLS_CONF'] = str(Files.CONF)


app = App()


@app.default
def main(
  *,
  debug: bool = False,
  loglevel: Annotated[int, Parameter(['--loglevel', '-l'])] = 20,
):
  freeze_support()
  loglevel = min(loglevel, (10 if debug else 20))
  _init(loglevel=loglevel)

  QtQml.qmlRegisterType(FigureCanvas, 'Backend', 1, 0, 'FigureCanvas')

  app = QtGui.QGuiApplication(sys.argv)
  engine = QtQml.QQmlApplicationEngine(str(Files.QML))

  try:
    win: QtGui.QWindow = engine.rootObjects()[0]
  except IndexError:
    logger.critical('Failed to load QML {}', Files.QML)
    return -1

  controller = Controller(win, loglevel)
  context = engine.rootContext()
  context.setContextProperty('con', controller)

  pcs = {}
  for name, cls in PlotControllers.classes():
    canvas = win.findChild(FigureCanvas, f'{name}_plot')
    pc = cls()
    pc.init(app, canvas)
    pcs[name] = pc

  controller.pc = pcs  # type: ignore[assignment]

  return app.exec_()


if __name__ == '__main__':
  app()
