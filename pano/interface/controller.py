from pathlib import Path
from typing import Optional

from loguru import logger
from PySide2 import QtCore
from PySide2 import QtGui

from .pano_files import ThermalPanoramaFileManager
from .pano_project import init_directory
from .tree import tree_string


def _log(message: str):
  find = message.find('|')
  if find == -1:
    level = 'DEBUG'
  else:
    level = message[:find].upper()
    message = message[(find + 1):]

  logger.log(level, message)


class _Window:

  def __init__(self, window: QtGui.QWindow) -> None:
    self._window = window

  def update_project_tree(self, text: str):
    self._window.update_project_tree(text)

  def update_image_view(self, paths: list):
    self._window.update_image_view(paths)


class Controller(QtCore.QObject):

  def __init__(self) -> None:
    super().__init__()

    self._win: Optional[_Window] = None
    self._wd: Optional[Path] = None

  @property
  def win(self) -> _Window:
    if self._win is None:
      raise ValueError('Window not set')

    return self._win

  def set_window(self, window: QtGui.QWindow):
    self._win = _Window(window)

  @QtCore.Slot(str)
  def log(self, message: str):
    _log(message=message)

  @QtCore.Slot(str)
  def select_working_dir(self, wd):
    wd = Path(wd).resolve()
    if not wd.exists():
      raise FileNotFoundError(wd)

    self._wd = wd
    self.update_project_tree()
    self.update_image_view()

  @QtCore.Slot()
  def update_project_tree(self):
    if self._wd is None:
      raise ValueError('Working directory not set')

    tree = tree_string(self._wd)
    self.win.update_project_tree(tree)

  def update_image_view(self):
    if self._wd is None:
      raise ValueError('Working directory not set')

    fm = ThermalPanoramaFileManager(self._wd)
    files = fm.raw_files()
    if files:
      self.win.update_image_view(['file:///' + x.as_posix() for x in files])

  @QtCore.Slot()
  def init_directory(self):
    if self._wd is None:
      raise ValueError('Working directory not set')

    init_directory(directory=self._wd)
    self.update_project_tree()
    self.update_image_view()
