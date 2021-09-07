import multiprocessing as mp
from pathlib import Path
from typing import Optional

from loguru import logger

from pano.utils import set_logger

from .common.pano_files import init_directory
from .common.pano_files import ThermalPanoramaFileManager
from .mbq import QtCore
from .mbq import QtGui
from .plot_controller import RegistrationPlotController
from .plot_controller import SegmentationPlotController
from .tree import tree_string


def _log(message: str):
  find = message.find('|')
  if find == -1:
    level = 'DEBUG'
  else:
    level = message[:find].upper()
    message = message[(find + 1):]

  logger.log(level, message)


def _path2url(path):
  return 'file:///' + Path(path).as_posix()


def _url2path(url: str):
  return Path(url.replace('file:///', ''))


def _producer(queue: mp.Queue, directory, command: str, loglevel: int):
  set_logger(loglevel)

  # pylint: disable=import-outside-toplevel
  from .pano_project import ThermalPanorama

  tp = ThermalPanorama(directory=directory)

  try:
    fn = getattr(tp, f'{command}_generator')
  except AttributeError:
    getattr(tp, command)()
    queue.put(1.0)
  else:
    for r in fn():
      queue.put(r)


class _Consumer(QtCore.QThread):
  done = QtCore.Signal()  # TODO command 종료 후 화면 업데이트
  update = QtCore.Signal(float)

  def __init__(self) -> None:
    super().__init__()
    self._queue: Optional[mp.Queue] = None

  @property
  def queue(self):
    return self._queue

  @queue.setter
  def queue(self, value):
    self._queue = value

  def run(self):
    if self.queue is None:
      raise ValueError('queue not set')

    while True:
      if not self.queue.empty():
        r = self.queue.get()
        self.update.emit(r)

        if r >= 1.0:
          self.done.emit()
          break


class _Window:

  def __init__(self, window: QtGui.QWindow) -> None:
    self._window = window

  def pbar(self, value: float):
    return self._window.pbar(value)

  def panel_funtion(self, panel: str, fn: str, *args):
    p = self._window.get_panel(panel)
    if p is None:
      raise ValueError(f'Invalid panel name: {panel}')

    f = getattr(p, fn)

    return f(*args)


class Controller(QtCore.QObject):

  def __init__(self, win: QtGui.QWindow, loglevel=20) -> None:
    super().__init__()

    self._win = _Window(win)
    self._loglevel = loglevel

    self._consumer = _Consumer()
    self._consumer.update.connect(self.pbar)

    self._wd: Optional[Path] = None
    self._fm: Optional[ThermalPanoramaFileManager] = None
    self._rpc: Optional[RegistrationPlotController] = None
    self._spc: Optional[SegmentationPlotController] = None

  @property
  def win(self) -> _Window:
    return self._win

  @win.setter
  def win(self, win: QtGui.QWindow):
    self._win = _Window(win)

  def set_plot_controllers(self, registration, segmentation):
    self._rpc = registration
    self._spc = segmentation

  @QtCore.Slot(str)
  def log(self, message: str):
    _log(message=message)

  @QtCore.Slot(float)
  def pbar(self, r: float):
    self.win.pbar(r)

  @QtCore.Slot(str)
  def prj_select_working_dir(self, wd):
    wd = Path(wd).resolve()
    if not wd.exists():
      raise FileNotFoundError(wd)

    self._wd = wd
    self._fm = ThermalPanoramaFileManager(wd)
    self.rpc.fm = self._fm
    self.spc.fm = self._fm

    self.prj_update_project_tree()
    self.update_image_view()

  @QtCore.Slot()
  def prj_init_directory(self):
    init_directory(directory=self._wd)
    self.prj_update_project_tree()
    self.update_image_view()

  def prj_update_project_tree(self):
    tree = tree_string(self._wd)
    self.win.panel_funtion('project', 'update_project_tree', tree)

  @QtCore.Slot(str)
  def command(self, command: str):
    if self._wd is None:
      logger.warning('Directory not set')
      return

    self.pbar(0.0)

    queue = mp.Queue()
    self._consumer.queue = queue
    self._consumer.start()

    process = mp.Process(name=command,
                         target=_producer,
                         args=(queue, self._wd, command, self._loglevel))
    process.start()

  def update_image_view(self,
                        panels=('project', 'registration', 'segmentation')):
    if self._fm is None:
      logger.warning('Directory not set')
      return

    files = self._fm.raw_files()
    if not files:
      # logger.warning('No raw files')
      return

    # TODO testo .xlsx 파일
    for panel in panels:
      self.win.panel_funtion(panel, 'update_image_view',
                             [_path2url(x) for x in files])

  @QtCore.Slot(str)
  def rgst_plot(self, url):
    assert self._wd is not None
    assert self._rpc is not None
    path = _url2path(url)
    self._rpc.plot(path)

  @QtCore.Slot()
  def rgst_save(self):
    assert self._rpc is not None
    self._rpc.save()

  @QtCore.Slot(str)
  def seg_plot(self, url):
    assert self._wd is not None
    assert self._spc is not None
    path = _url2path(url)
    self._spc.plot(path)
