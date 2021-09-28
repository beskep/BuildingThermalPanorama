import multiprocessing as mp
from pathlib import Path
from typing import Optional

from loguru import logger

from pano.utils import set_logger

from .common.pano_files import init_directory
from .common.pano_files import ThermalPanoramaFileManager
from .mbq import QtCore
from .mbq import QtGui
from .plot_controller import PanoramaPlotController
from .plot_controller import RegistrationPlotController
from .plot_controller import save_manual_correction
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
    fn = getattr(tp, f'{command}_generator', None)
    if fn is not None:
      queue.put(0.0)
      for r in fn():
        queue.put(r)
    else:
      fn = getattr(tp, command)
      queue.put(fn())
  except (ValueError, RuntimeError, IOError) as e:
    logger.exception(e)
    queue.put(f'{type(e).__name__}: {e}')

  queue.put(1.0)


def _save_manual_correction(queue: mp.Queue, wd: str, subdir: str,
                            viewing_angle: float, roll: float, pitch: float,
                            yaw: float):
  try:
    save_manual_correction(wd, subdir, viewing_angle, roll, pitch, yaw)
  except (ValueError, RuntimeError, IOError) as e:
    logger.exception(e)
    queue.put(f'{type(e).__name__}: {e}')

  queue.put(1.0)


class _Consumer(QtCore.QThread):
  update = QtCore.Signal(float)  # 진행률 [0, 1] 업데이트
  done = QtCore.Signal()  # 작업 종료 signal (진행률 >= 1.0)
  fail = QtCore.Signal(str)  # 에러 발생 signal. 에러 메세지 emit

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
        value = self.queue.get()

        if isinstance(value, str):
          self.fail.emit(value)
          break

        if value is None or value >= 1.0:
          self.done.emit()
          self.quit()
          break

        self.update.emit(value)


class _Window:

  def __init__(self, window: QtGui.QWindow) -> None:
    self._window = window

  def pb_value(self, value: float):
    return self._window.pb_value(value)

  def pb_state(self, indeterminate: bool):
    self._window.pb_state(indeterminate)

  def popup(self, title: str, message: str, timeout=2000):
    logger.debug('[Popup] {}: {}', title, message)
    self._window.popup(title, message, timeout)

  def panel_funtion(self, panel: str, fn: str, *args):
    p = self._window.get_panel(panel)
    if p is None:
      raise ValueError(f'Invalid panel name: {panel}')

    f = getattr(p, fn)

    return f(*args)


class Controller(QtCore.QObject):
  _CMD_KR = {
      'extract': '열화상 추출',
      'register': '열화상-실화상 정합',
      'segment': '외피 부위 인식',
      'panorama': '파노라마 생성',
      'correct': '왜곡 보정'
  }

  def __init__(self, win: QtGui.QWindow, loglevel=20) -> None:
    super().__init__()

    self._win = _Window(win)
    self._loglevel = loglevel

    self._consumer = _Consumer()
    self._consumer.update.connect(self._pb_value)
    self._consumer.fail.connect(self._error_popup)

    self._wd: Optional[Path] = None
    self._fm: Optional[ThermalPanoramaFileManager] = None
    self._rpc: Optional[RegistrationPlotController] = None
    self._spc: Optional[SegmentationPlotController] = None
    self._ppc: Optional[PanoramaPlotController] = None

  @property
  def win(self) -> _Window:
    return self._win

  @win.setter
  def win(self, win: QtGui.QWindow):
    self._win = _Window(win)

  def set_plot_controllers(self, registration, segmentation, panorama):
    self._rpc = registration
    self._spc = segmentation
    self._ppc = panorama

  @QtCore.Slot(str)
  def log(self, message: str):
    _log(message=message)

  def _pb_value(self, value: float):
    if value == 0:
      self.win.pb_state(False)
    self.win.pb_value(value)

  def _error_popup(self, message: str):
    self.win.popup('Error', message, timeout=10000)
    self.win.pb_state(False)
    self.win.pb_value(1.0)

  @QtCore.Slot(str)
  def prj_select_working_dir(self, wd):
    wd = Path(wd).resolve()
    if not wd.exists():
      raise FileNotFoundError(wd)

    self._wd = wd
    self._fm = ThermalPanoramaFileManager(wd)
    self._rpc.fm = self._fm
    self._spc.fm = self._fm
    self._ppc.fm = self._fm

    self.prj_update_project_tree()
    self.update_image_view()

  @QtCore.Slot()
  def prj_init_directory(self):
    if self._wd is None:
      return

    init_directory(directory=self._wd)
    self.prj_update_project_tree()
    self.update_image_view()
    self.win.popup('Success', '초기화 완료')

  def prj_update_project_tree(self):
    tree = tree_string(self._wd, width=40)
    self.win.panel_funtion('project', 'update_project_tree', tree)

  @QtCore.Slot(str)
  def command(self, command: str):
    if self._wd is None:
      self.win.popup('Warning', '경로가 선택되지 않았습니다.')
      self.win.pb_state(False)
      return

    self.win.pb_state(True)

    queue = mp.Queue()
    cmd_kr = self._CMD_KR[command]

    def _done():
      if command in ('panorama', 'correct'):
        self._ppc.plot(force=True)
        self.win.panel_funtion('panorama', 'reset')
      elif command == 'register':
        self._rpc.reset()
        self._rpc.reset_matrices()

      self.win.popup('Success', f'{cmd_kr} 완료')
      self.win.pb_state(False)
      self.win.pb_value(1.0)

      try:
        self._consumer.done.disconnect()
      except TypeError:
        pass

    self._consumer.queue = queue
    self._consumer.done.connect(_done)
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

    try:
      self._rpc.plot(path)
    except FileNotFoundError:
      logger.warning('Data not extracted: {}', path)
      self.win.popup('Error', f'파일 {path.name}의 데이터가 추출되지 않았습니다.')

  @QtCore.Slot()
  def rgst_save(self):
    assert self._rpc is not None
    self._rpc.save()

  @QtCore.Slot()
  def rgst_reset(self):
    assert self._rpc is not None
    self._rpc.reset()

  @QtCore.Slot(bool)
  def rgst_set_grid(self, grid):
    self._rpc.set_grid(grid)

  @QtCore.Slot()
  def rgst_home(self):
    self._rpc.home()

  @QtCore.Slot(bool)
  def rgst_zoom(self, value):
    self._rpc.zoom(value)

  @QtCore.Slot(str)
  def seg_plot(self, url):
    assert self._wd is not None
    assert self._spc is not None
    path = _url2path(url)

    try:
      self._spc.plot(path)
    except FileNotFoundError:
      logger.warning('File not found: {}', path)

  @QtCore.Slot()
  def pano_plot(self):
    try:
      self._ppc.plot(force=False)
    except OSError:
      pass

  @QtCore.Slot(float, float, float, int)
  def pano_rotate(self, roll, pitch, yaw, limit):
    self._ppc.project(roll=roll, pitch=pitch, yaw=yaw, limit=limit)

  @QtCore.Slot(bool)
  def pano_set_grid(self, grid):
    self._ppc.set_grid(grid)

  @QtCore.Slot(str)
  def pano_set_viewing_angle(self, value):
    try:
      angle = float(value)
    except ValueError:
      pass
    else:
      self._ppc.viewing_angle = angle
      logger.debug('Vewing angle: {}', angle)

  @QtCore.Slot(float, float, float)
  def pano_save_manual_correction(self, roll, pitch, yaw):
    self.win.pb_state(True)

    def _done():
      self.win.popup('Success', '저장 완료', timeout=1000)
      self.win.pb_state(False)
      self.win.pb_value(1.0)

      try:
        self._consumer.done.disconnect()
      except TypeError:
        pass

    queue = mp.Queue()
    self._consumer.queue = queue
    self._consumer.done.connect(_done)
    self._consumer.start()

    args = (queue, self._wd.as_posix(), self._ppc.subdir,
            self._ppc.viewing_angle, roll, pitch, yaw)
    process = mp.Process(name='save', target=_save_manual_correction, args=args)
    process.start()
