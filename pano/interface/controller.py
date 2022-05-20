import json
import multiprocessing as mp
from pathlib import Path
from typing import Optional

from loguru import logger
import numpy as np
from omegaconf import DictConfig
from omegaconf import OmegaConf

from pano.misc.imageio import ImageIO as IIO
from pano.utils import set_logger

from . import analysis
from .common.config import update_config
from .common.pano_files import DIR
from .common.pano_files import init_directory
from .common.pano_files import SP
from .common.pano_files import ThermalPanoramaFileManager
from .mbq import QtCore
from .mbq import QtGui
from .plot_controller import AnalysisPlotController
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

  try:
    tp = ThermalPanorama(directory=directory)
  except (ValueError, OSError) as e:
    queue.put(f'{type(e).__name__}: {e}')
    return

  try:
    fn = getattr(tp, f'{command}_generator', None)
    if fn is not None:
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
                            viewing_angle: float, angles: tuple,
                            crop_range: Optional[np.ndarray]):
  try:
    save_manual_correction(wd, subdir, viewing_angle, angles, crop_range)
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
    if title.lower() != 'error':
      msg = message.replace('\n', ' ')
      self._window.status_message(f'[{title}] {msg}')

  def panel_funtion(self, panel: str, fn: str, *args):
    p = self._window.get_panel(panel)
    if p is None:
      raise ValueError(f'Invalid panel name: {panel}')

    f = getattr(p, fn)

    return f(*args)

  def update_config(self, config: DictConfig):
    self._window.update_config(json.dumps(OmegaConf.to_object(config)))


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
    self._apc: Optional[AnalysisPlotController] = None
    self._config: Optional[DictConfig] = None

  @property
  def win(self) -> _Window:
    return self._win

  @win.setter
  def win(self, win: QtGui.QWindow):
    self._win = _Window(win)

  def set_plot_controllers(self, controllers):
    self._rpc, self._spc, self._ppc, self._apc = controllers

  @QtCore.Slot(str)
  def log(self, message: str):
    _log(message=message)

  def _pb_value(self, value: float):
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

    for pc in (self._rpc, self._spc, self._ppc, self._apc):
      pc.fm = self._fm

    self._apc.show_point_temperature = lambda x: self.win.panel_funtion(
        'analysis', 'show_point_temperature', x)

    self._config = init_directory(directory=wd)
    self.win.update_config(self._config)

    self.prj_update_project_tree()
    self.update_image_view()

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
        self._ppc.plot(d=(DIR.PANO if command == 'panorama' else DIR.COR),
                       sp=SP.IR)
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

  @QtCore.Slot(str)
  def configure(self, string: str):
    if self._config is None:
      # 프로젝트 폴더 선택 안됨
      return

    assert self._wd is not None

    config = json.loads(string)
    logger.debug(config)

    try:
      separate = config['panorama']['separate']
    except KeyError:
      separate = None

    if separate is None or self._config['panorama']['separate'] == separate:
      # separate 설정이 변경되지 않은 경우
      vis_blend = OmegaConf.create()  # 빈 설정
    else:
      self._clear_separate_results()  # seg 이후 결과 삭제

      # 실화상 blend 설정 변경
      blend_type = 'feather' if separate else 'no'
      vis_blend = OmegaConf.from_dotlist(
          [f'panorama.blend.type.VIS={blend_type}'])

    self._config = update_config(self._wd, config, vis_blend)
    self.win.update_config(self._config)  # 이미 반영된 설정도 다시 업데이트 함

  def _clear_separate_results(self):
    # 다음 결과 폴더 내 모든 파일 삭제
    for d in (DIR.SEG, DIR.PANO, DIR.COR):
      files = self._fm.glob(d, '*')
      logger.debug('delete files in {}: {}', d,
                   [str(x.relative_to(self._wd)) for x in files])

      for f in files:
        f.unlink()

    # plot 리셋
    for pc in (self._rpc, self._spc, self._ppc, self._apc):
      pc.reset()

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
    self._rpc.save(panorama=self._config['panorama']['separate'])
    self.win.popup('Success', '저장 완료', timeout=1000)

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
      self._spc.plot(path, separate=self._config['panorama']['separate'])
    except FileNotFoundError:
      logger.warning('File not found: {}', path)

  @QtCore.Slot(str, str)
  def pano_plot(self, d, sp):
    try:
      self._ppc.plot(d=DIR[d], sp=SP[sp])
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
      logger.debug('Viewing angle: {}', angle)

  @QtCore.Slot()
  def pano_home(self):
    self._ppc.home()

  @QtCore.Slot(bool)
  def pano_crop_mode(self, value):
    self._ppc.crop_mode(value)

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
            self._ppc.viewing_angle, (roll, pitch, yaw), self._ppc.crop_range())
    process = mp.Process(name='save', target=_save_manual_correction, args=args)
    process.start()

  @QtCore.Slot()
  def rgst_pano_draw(self):
    if not self._fm:
      return

    ir = self._fm.panorama_path(d=DIR.PANO, sp=SP.IR)
    vis = self._fm.panorama_path(d=DIR.PANO, sp=SP.VIS)

    if any(not x.exists() for x in (ir, vis)):
      self.win.popup('Warning', '파노라마 생성 결과가 없습니다.')
      return

    # XXX 해상도 낮추기?
    self._rpc.set_images(fixed_image=IIO.read(ir), moving_image=IIO.read(vis))
    self._rpc.draw()

  @QtCore.Slot(bool, bool, bool)
  def analysis_plot(self, factor, segmentaion, vulnerable):
    try:
      self._apc.plot(factor=factor,
                     segmentation=segmentaion,
                     vulnerable=vulnerable)
    except OSError:
      pass
    except ValueError as e:
      self.win.popup('Error', str(e))
    else:
      self.win.panel_funtion('analysis', 'set_temperature_range',
                             *self._apc.temperature_range())
      self._analysis_summary()

  @QtCore.Slot(float, float)
  def analysis_set_clim(self, vmin, vmax):
    self._apc.set_clim(vmin=vmin, vmax=vmax)

  @QtCore.Slot(float, float)
  def analysis_correct_emissivity(self, wall, window):
    if np.isnan(wall) and np.isnan(window):
      return

    self._apc.remove_images()  # 기존 이미지 삭제하고 원본 이미지를 불러와서 보정

    try:
      ir, mask = self._apc.images
    except OSError:
      return

    meta_files = list(self._fm.subdir(DIR.IR).glob('*.yaml'))

    for idx, e1 in zip([1, 2], [wall, window]):
      ir0 = np.full_like(ir, np.nan)
      ir0[mask == idx] = ir[mask == idx]
      ir1 = analysis.correct_emissivity(image=ir0, meta_files=meta_files, e1=e1)
      ir[mask == idx] = ir1[mask == idx]

    self._apc.update_ir(ir)
    self.win.panel_funtion('analysis', 'set_temperature_range',
                           *self._apc.temperature_range())
    self._analysis_summary()

  @QtCore.Slot(float)
  def analysis_correct_temperature(self, temperature):
    try:
      ir = analysis.correct_temperature(*self._apc.images,
                                        coord=self._apc.coord,
                                        T1=temperature)
    except ValueError as e:
      self.win.popup('Error', str(e))
    else:
      self._apc.update_ir(ir)
      self.win.panel_funtion('analysis', 'show_point_temperature', temperature)
      self.win.panel_funtion('analysis', 'set_temperature_range',
                             *self._apc.temperature_range())
      self._analysis_summary()

  @QtCore.Slot(float, float)
  def analysis_set_teti(self, te, ti):
    self._apc.teti = (te, ti)

  @QtCore.Slot(float)
  def analysis_set_threshold(self, value):
    self._apc.threshold = value
    if not np.isnan(self._apc.teti).any():
      self._analysis_summary()

  def _analysis_summary(self):
    self.win.panel_funtion('analysis', 'clear_table')
    for cls, summary in self._apc.summary().items():
      row = {
          k: (v if isinstance(v, str) else f'{v:.2f}')
          for k, v in summary.items()
      }
      row['class'] = cls
      self.win.panel_funtion('analysis', 'add_table_row', row)

  @QtCore.Slot()
  def dist_plot(self):
    try:
      data = self._apc.plot()
    except OSError as e:
      self.win.popup('Warning', str(e))
    else:
      self.win.panel_funtion('descriptive', 'clear_table')

      for class_ in ('Wall', 'Window'):
        row = {key: f'{value:.2f}' for key, value in data[class_].items()}
        row['class'] = class_
        self.win.panel_funtion('descriptive', 'add_table_row', row)
