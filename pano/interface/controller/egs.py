import multiprocessing as mp
from pathlib import Path
from typing import Iterable

from omegaconf import DictConfig

import pano.interface.common.pano_files as pf
import pano.interface.controller.controller as con
from pano.interface.mbq import QtCore
from pano.interface.mbq import QtGui


class Window(con.Window):

  def panel_funtion(self, fn: str, *args):
    p = self._window.get_panel()
    assert p is not None
    f = getattr(p, fn)
    return f(*args)


class Controller(QtCore.QObject):

  def __init__(self, win: QtGui.QWindow, loglevel=20) -> None:
    super().__init__()

    self._win = Window(win)
    self._loglevel = loglevel

    self._consumer = con.Consumer()
    self._consumer.state.connect(self.win.pb_state)
    self._consumer.update.connect(self.win.pb_value)
    self._consumer.fail.connect(self.win.error_popup)

    self._wd: Path | None = None
    self._fm: pf.ThermalPanoramaFileManager | None = None
    self._config: DictConfig | None = None

    # TODO plot controller

  @property
  def win(self) -> Window:
    return self._win

  @property
  def fm(self) -> pf.ThermalPanoramaFileManager:
    if self._fm is None:
      raise ValueError('FileManager not set')
    return self._fm

  @QtCore.Slot(str)
  def log(self, message: str):
    con.log(message=message)

  def command(self, commands: str | Iterable[str], name: str | None = None):
    if self._wd is None:
      self.win.popup('Warning', '경로가 선택되지 않았습니다.')
      self.win.pb_state(False)
      return

    if isinstance(commands, str):
      commands = (commands,)
    else:
      commands = tuple(commands)

    name = '' if name is None else f'{name.strip()} '
    self.win.pb_state(True)

    def done():
      # XXX method로?
      self.win.popup('Success', f'{name}작업 완료')
      self.win.pb_state(False)
      self.win.pb_value(1.0)

      try:
        self._consumer.done.disconnect()
      except TypeError:
        pass

    queue = mp.Queue()
    self._consumer.queue = queue
    self._consumer.done.connect(done)
    self._consumer.start()

    process = mp.Process(name=name,
                         target=con.producer,
                         args=(queue, self._wd, commands, self._loglevel))
    process.start()

  def update_image_view(self):
    try:
      files = self.fm.files(pf.DIR.VIS)
    except FileNotFoundError:
      files = self.fm.files(pf.DIR.RAW)

    uri = [con.path2uri(x) for x in files]
    self.win.panel_funtion('update_image_view', uri)

  @QtCore.Slot(str)
  def select_working_dir(self, path):
    path = con.uri2path(path)
    if not path.exists():
      raise FileNotFoundError(path)
    if not path.is_dir():
      raise NotADirectoryError(path)

    try:
      self._config = pf.init_directory(path)
    except pf.ImageNotFoundError as e:
      self.win.popup('Error', f'{e.args[0]}\n({e.args[1]})')
      return

    self._wd = path
    self._fm = pf.ThermalPanoramaFileManager(path)
    self.update_image_view()

    # TODO controller fm 초기화

  @QtCore.Slot()
  def extract_and_register(self):
    self.command(commands=('extract', 'register'), name='열·실화상 추출 및 정합')
