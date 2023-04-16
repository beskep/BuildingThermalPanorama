from contextlib import suppress
import multiprocessing as mp
from pathlib import Path
from typing import Iterable

from omegaconf import DictConfig

import pano.interface.common.pano_files as pf
import pano.interface.controller.controller as con
from pano.interface.mbq import QtCore
from pano.interface.mbq import QtGui
from pano.interface.plot_controller.egs import AnomalyThresholdNotSetError
from pano.interface.plot_controller.egs import PlotController


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
    self._pc = PlotController()
    self._config: DictConfig | None = None

  @property
  def win(self) -> Window:
    return self._win

  @property
  def fm(self) -> pf.ThermalPanoramaFileManager:
    if self._fm is None:
      raise ValueError('FileManager not set')
    return self._fm

  @property
  def pc(self) -> PlotController:
    return self._pc

  @QtCore.Slot(str)
  def log(self, message: str):
    con.log(message=message)

  def command(self, commands: str | Iterable[str], name: str | None = None):
    if self._wd is None:
      self.win.popup('Warning', '경로가 선택되지 않았습니다.')
      self.win.pb_state(False)
      return

    commands = (commands,) if isinstance(commands, str) else tuple(commands)
    name = '' if name is None else f'{name.strip()} '
    self.win.pb_state(True)

    def done():
      self.win.popup('Success', f'{name}작업 완료')
      self.win.pb_state(False)
      self.win.pb_value(1.0)

      with suppress(TypeError):
        self._consumer.done.disconnect()

      if 'detect' in commands:
        self.pc.update_threshold()

    queue = mp.Queue()
    self._consumer.queue = queue
    self._consumer.done.connect(done)
    self._consumer.start()

    process = mp.Process(
        name=name,
        target=con.producer,
        args=(queue, self._wd, commands, self._loglevel, 'AnomalyDetection'),
    )
    process.start()

  def update_image_view(self):
    files = [con.path2uri(x) for x in self.fm.files(pf.DIR.RAW)]
    self.win.panel_funtion('update_image_view', files)

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
    self._pc.fm = self._fm

    self.update_image_view()

    with suppress(FileNotFoundError):
      self.pc.update_threshold()

  @QtCore.Slot(str, int)
  def plot(self, uri, mode):
    self.pc.reset()
    if not uri:
      return

    path = con.uri2path(uri)
    self.pc.rgst = mode == 1
    mode_ = ('raw', 'registration', 'anomaly')[mode]

    try:
      self.pc.plot(path=path, mode=mode_)
    except FileNotFoundError:
      self.win.popup('Error', '파일을 먼저 추출해주세요.')
    except AnomalyThresholdNotSetError:
      self.win.popup('Error', '이상 영역을 먼저 검출해주세요.')

  @QtCore.Slot(bool, bool)
  def plot_navigation(self, home, zoom):
    self.pc.navigation(home=home, zoom=zoom)

  @QtCore.Slot(str, str)
  def qml_command(self, commands: str, name: str):
    self.command(commands=map(str.strip, commands.split(',')), name=name)
