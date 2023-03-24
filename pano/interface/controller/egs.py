import pano.interface.controller.controller as con
from pano.interface.mbq import QtCore
from pano.interface.mbq import QtGui


class Controller(QtCore.QObject):

  def __init__(self, win: QtGui.QWindow, loglevel=20) -> None:
    super().__init__()

    self._win = con.Window(win)
    self._loglevel = loglevel

    # TODO plot controller

  @property
  def win(self) -> con.Window:
    return self._win

  @QtCore.Slot(str)
  def log(self, message: str):
    con.log(message=message)
