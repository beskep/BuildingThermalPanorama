import json
import multiprocessing as mp
from pathlib import Path
from typing import Any, Optional

from loguru import logger
import numpy as np
from omegaconf import DictConfig
from omegaconf import OmegaConf

from pano import utils
from pano.misc.imageio import ImageIO as IIO

from . import analysis
from .common.config import update_config
from .common.pano_files import DIR
from .common.pano_files import ImageNotFoundError
from .common.pano_files import init_directory
from .common.pano_files import replace_vis_images
from .common.pano_files import SP
from .common.pano_files import ThermalPanoramaFileManager
from .mbq import QtCore
from .mbq import QtGui
from .plot_controller import PlotControllers
from .plot_controller import save_manual_correction
from .plot_controller import WorkingDirNotSet
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
  utils.set_logger(loglevel)

  # pylint: disable=import-outside-toplevel
  from .pano_project import ThermalPanorama

  try:
    tp = ThermalPanorama(directory=directory)
  except (ValueError, OSError) as e:
    queue.put(f'{type(e).__name__}: {e}')
    return

  logger.info('Run {} command', command)

  try:
    fn = getattr(tp, f'{command}_generator', None)
    if fn is not None:
      for r in fn():
        queue.put(r)
    else:
      fn = getattr(tp, command)
      queue.put(fn())
  except (ValueError, RuntimeError, OSError) as e:
    logger.exception(e)
    queue.put(f'{type(e).__name__}: {e}')

  queue.put(1.0)


def _save_manual_correction(queue: mp.Queue, wd: str, subdir: str,
                            viewing_angle: float, angles: tuple,
                            crop_range: Optional[np.ndarray]):
  try:
    save_manual_correction(wd, subdir, viewing_angle, angles, crop_range)
  except (ValueError, RuntimeError, OSError) as e:
    logger.exception(e)
    queue.put(f'{type(e).__name__}: {e}')

  queue.put(1.0)


def _segments(flag_count: bool, seg_count: int, seg_length: float,
              building_width: float) -> int:
  if flag_count:
    segments = seg_count
  elif np.isnan(building_width):
    raise ValueError('건물 폭을 설정하지 않았습니다.')
  else:
    segments = int(building_width / seg_length)
    if segments < 1:
      raise ValueError('분할 길이 또는 건물 폭이 잘못 설정되었습니다.')

  return segments


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

    ok = title.lower() != 'error'
    utils.play_sound(ok=ok)

    if ok:
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
    self._pc: Any = None
    self._config: Optional[DictConfig] = None

  @property
  def win(self) -> _Window:
    return self._win

  @win.setter
  def win(self, win: QtGui.QWindow):
    self._win = _Window(win)

  @property
  def pc(self) -> PlotControllers:
    return self._pc

  def set_plot_controllers(self, controllers: dict):
    self._pc = PlotControllers(**controllers)
    self._pc.analysis.summarize = self._analysis_summarize

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

    try:
      self._config = init_directory(directory=wd)
    except ImageNotFoundError as e:
      self.win.popup('Error', f'{e.args[0]}\n({e.args[1]})')
      return

    self._wd = wd
    self._fm = ThermalPanoramaFileManager(wd)

    for controller in self.pc.controllers():
      controller.fm = self._fm
    self.pc.analysis.show_point_temperature = lambda x: self.win.panel_funtion(
        'analysis', 'show_point_temperature', x)

    self.win.update_config(self._config)

    self.prj_update_project_tree()
    self.update_image_view()

  def prj_update_project_tree(self):
    tree = tree_string(self._wd, width=40)
    self.win.panel_funtion('project', 'update_project_tree', tree)

  @QtCore.Slot(str)
  def prj_select_vis_images(self, files: str):
    assert self._fm is not None

    if not self._fm.subdir(DIR.VIS).exists():
      self.win.popup('Warning', '열·실화상 데이터를 먼저 추출해주세요.', 5000)
      return

    try:
      replace_vis_images(fm=self._fm, files=files)
    except OSError as e:
      logger.catch(e)  # type: ignore
      self.win.popup('Error', str(e))

    self.update_image_view()

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
        self.pc.panorama.plot(
            d=(DIR.PANO if command == 'panorama' else DIR.COR), sp=SP.IR)
        self.win.panel_funtion('panorama', 'reset')
      elif command == 'register':
        self.pc.registration.reset()
        self.pc.registration.reset_matrices()

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
    logger.debug('config: {}', config)

    configs = self._configure(self._config, config)
    self._config = update_config(self._wd, *configs)
    self.win.update_config(self._config)  # 이미 반영된 설정도 다시 업데이트 함

  def _configure(self, config0, config1):
    try:
      separate = config1['panorama']['separate']
    except KeyError:
      separate = None

    if separate is None or config0['panorama']['separate'] == separate:
      # separate 설정이 변경되지 않은 경우
      vis_blend = OmegaConf.create()  # 빈 설정
    else:
      self._clear_separate_results()  # seg 이후 결과 삭제

      # 실화상 blend 설정 변경
      blend_type = 'feather' if separate else 'no'
      vis_blend = OmegaConf.from_dotlist(
          [f'panorama.blend.type.VIS={blend_type}'])

    if 'output' in config1:
      self.pc.output.configure(config1['output'])
      self.output_plot()

    return config1, vis_blend

  def _clear_separate_results(self):
    # 다음 결과 폴더 내 관련 파일 삭제
    exts = {'.npy', '.jpg', '.png', '.webp'}
    for d in (DIR.SEG, DIR.PANO, DIR.COR):
      files = [
          x for x in self._fm.glob(d, '*')
          if x.is_file() and x.suffix.lower() in exts
      ]
      logger.debug('delete files in {}: {}', d, [x.name for x in files])

      for f in files:
        f.unlink()

    # plot 리셋
    for pc in self.pc.controllers():
      pc.reset()
      pc.draw()

  def update_image_view(self,
                        panels=('project', 'registration', 'segmentation')):
    if self._fm is None:
      logger.warning('Directory not set')
      return

    raw_files = [_path2url(x) for x in self._fm.raw_files()]
    if not raw_files:
      logger.debug('no files')
      return

    try:
      vis_files = [_path2url(x) for x in self._fm.files(DIR.VIS, error=False)]
    except FileNotFoundError:
      vis_files = raw_files

    for panel in panels:
      files = vis_files if panel == 'segmentation' else raw_files
      self.win.panel_funtion(panel, 'update_image_view', files)

  @QtCore.Slot(str)
  def rgst_plot(self, url):
    assert self._wd is not None
    path = _url2path(url)

    try:
      self.pc.registration.plot(path)
    except FileNotFoundError:
      logger.warning('Data not extracted: {}', path)
      self.win.popup('Error', f'파일 {path.name}의 데이터가 추출되지 않았습니다.')

  @QtCore.Slot()
  def rgst_save(self):
    self.pc.registration.save(panorama=self._config['panorama']['separate'])
    self.win.popup('Success', '저장 완료', timeout=1000)

  @QtCore.Slot()
  def rgst_reset(self):
    self.pc.registration.reset()

  @QtCore.Slot(bool)
  def rgst_set_grid(self, grid):
    self.pc.registration.set_grid(grid)

  @QtCore.Slot()
  def rgst_home(self):
    self.pc.registration.home()

  @QtCore.Slot(bool)
  def rgst_zoom(self, value):
    self.pc.registration.zoom(value)

  @QtCore.Slot(str)
  def seg_plot(self, url):
    assert self._wd is not None
    path = _url2path(url)

    try:
      self.pc.segmentation.plot(path,
                                separate=self._config['panorama']['separate'])
    except FileNotFoundError:
      self.win.popup('Error', '부위 인식 결과가 없습니다.')
      logger.warning('File not found: {}', path)

  @QtCore.Slot(str, str)
  def pano_plot(self, d, sp):
    try:
      self.pc.panorama.plot(d=DIR[d], sp=SP[sp])
    except WorkingDirNotSet:
      pass
    except FileNotFoundError as e:
      logger.debug('FileNotFound: "{}"', e)

  @QtCore.Slot(float, float, float, int)
  def pano_rotate(self, roll, pitch, yaw, limit):
    self.pc.panorama.project(roll=roll, pitch=pitch, yaw=yaw, limit=limit)

  @QtCore.Slot(bool)
  def pano_set_grid(self, grid):
    self.pc.panorama.set_grid(grid)

  @QtCore.Slot(str)
  def pano_set_viewing_angle(self, value):
    try:
      angle = float(value)
    except ValueError:
      pass
    else:
      self.pc.panorama.viewing_angle = angle
      logger.debug('Viewing angle: {}', angle)

  @QtCore.Slot()
  def pano_home(self):
    self.pc.panorama.home()

  @QtCore.Slot(bool)
  def pano_crop_mode(self, value):
    self.pc.panorama.crop_mode(value)

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

    pano = self.pc.panorama
    args = (queue, self._wd.as_posix(), pano.subdir, pano.viewing_angle,
            (roll, pitch, yaw), pano.crop_range())
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
    self.pc.registration.set_images(fixed_image=IIO.read(ir),
                                    moving_image=IIO.read(vis))
    self.pc.registration.draw()

  @QtCore.Slot(bool, bool, bool, bool)
  def analysis_plot(self, factor, segmentaion, vulnerable, distribution):
    self.pc.analysis.setting.factor = factor
    self.pc.analysis.setting.segmentation = segmentaion
    self.pc.analysis.setting.vulnerable = vulnerable
    self.pc.analysis.setting.distribution = distribution

    try:
      self.pc.analysis.plot()
    except WorkingDirNotSet:
      pass
    except FileNotFoundError as e:
      logger.debug('FileNotFound: "{}"', e)
    except ValueError as e:
      logger.catch(e)
      self.win.popup('Error', str(e))
    else:
      self.win.panel_funtion('analysis', 'set_temperature_range',
                             *self.pc.analysis.images.temperature_range())
      self._analysis_summarize()

  @QtCore.Slot(bool)
  def analysis_window_vulnerable(self, value):
    if self.pc.analysis.setting.window_vulnerable ^ value:
      self.pc.analysis.setting.window_vulnerable = value
      self.pc.analysis.plot()

  @QtCore.Slot()
  def analysis_read_multilayer(self):
    try:
      self.pc.analysis.read_multilayer()
    except (OSError, ValueError) as e:
      logger.catch(e)
      self.win.popup('Error', str(e))

  @QtCore.Slot()
  def analysis_cancel_selection(self):
    self.pc.analysis.cancel_selection()

  @QtCore.Slot(bool)
  def analysis_set_selector(self, point):
    self.pc.analysis.set_selector(point)

  @QtCore.Slot(float, float)
  def analysis_set_clim(self, vmin, vmax):
    self.pc.analysis.set_clim(vmin=vmin, vmax=vmax)

  @QtCore.Slot(float, float)
  def analysis_correct_emissivity(self, wall, window):
    if np.isnan(wall) and np.isnan(window):
      return

    try:
      # 기존 이미지 삭제, 원본 이미지를 불러와서 보정
      self.pc.analysis.images.reset_images()
      ir = self.pc.analysis.images.ir
      seg = self.pc.analysis.images.seg
    except WorkingDirNotSet:
      return

    meta_files = list(self._fm.subdir(DIR.IR).glob('*.yaml'))

    for idx, e1 in zip([1, 2], [wall, window]):
      ir0 = np.full_like(ir, np.nan)
      mask = seg == idx

      ir0[mask] = ir[mask]
      ir1 = analysis.correct_emissivity(image=ir0, meta_files=meta_files, e1=e1)
      ir[mask] = ir1[mask]

    self.pc.analysis.images.ir = ir
    self.pc.analysis.correction_params.e_wall = wall
    self.pc.analysis.correction_params.e_window = window

    self.win.panel_funtion('analysis', 'set_temperature_range',
                           *self.pc.analysis.images.temperature_range())

  @QtCore.Slot(float)
  def analysis_correct_temperature(self, temperature):
    try:
      ir, delta = analysis.correct_temperature(ir=self.pc.analysis.images.ir,
                                               mask=self.pc.analysis.images.seg,
                                               coord=self.pc.analysis.coord,
                                               T1=temperature)
    except ValueError as e:
      logger.catch(e)
      self.win.popup('Error', str(e))
    else:
      self.pc.analysis.images.ir = ir
      self.pc.analysis.correction_params.delta_temperature = delta

      self.win.panel_funtion('analysis', 'show_point_temperature', temperature)
      self.win.panel_funtion('analysis', 'set_temperature_range',
                             *self.pc.analysis.images.temperature_range())

  @QtCore.Slot(float, float)
  def analysis_set_teti(self, te, ti):
    self.pc.analysis.images.teti = (te, ti)

  @QtCore.Slot(float)
  def analysis_set_threshold(self, value):
    self.pc.analysis.images.threshold = value
    self._analysis_summarize()

  def _analysis_summarize(self):
    self.win.panel_funtion('analysis', 'clear_table')
    for cls, summary in self.pc.analysis.images.summarize().items():
      row = {
          k: (v if isinstance(v, str) else f'{v:.2f}')
          for k, v in summary.items()
      }
      row['class'] = cls
      self.win.panel_funtion('analysis', 'add_table_row', row)

  @QtCore.Slot()
  def analysis_save(self):
    try:
      self.pc.analysis.images.save()
      self.pc.analysis.save()
      self.pc.analysis.save_report()
    except WorkingDirNotSet:
      pass
    except ValueError as e:
      logger.catch(e)
      self.win.popup('Error', f'{e} 지표 및 분포 정보를 저장하지 못했습니다.')
      self.pc.analysis.plot()
    else:
      self.win.popup('Success', '저장 완료')

  @QtCore.Slot(str)
  def output_plot(self, image=None):
    if image:
      self.pc.output.setting.image = image

    try:
      self.pc.output.plot()
    except WorkingDirNotSet:
      pass
    except FileNotFoundError as e:
      logger.debug('FileNotFound: "{}"', e)

  @QtCore.Slot()
  def output_clear_lines(self):
    try:
      self.pc.output.lines.clear_lines()
      self.pc.output.draw()
    except WorkingDirNotSet:
      pass

  @QtCore.Slot()
  def output_estimate_edgelets(self):
    try:
      self.pc.output.estimate_edgelets()
    except WorkingDirNotSet:
      pass
    except FileNotFoundError as e:
      self.win.popup('Error', str(e))

  @QtCore.Slot(bool)
  def output_extend_lines(self, value):
    self.pc.output.lines.extend = value

  @QtCore.Slot(bool, int, float, float)
  def output_save(self, flag_count, seg_count, seg_length, building_width):
    try:
      segments = _segments(flag_count=flag_count,
                           seg_count=seg_count,
                           seg_length=seg_length,
                           building_width=building_width)
    except ValueError as e:
      logger.catch(e)
      self.win.popup('Error', str(e))
      segments = None

    logger.debug(
        'flag_count={} | count={} | length={} | width={} | segments={}',
        flag_count, seg_count, seg_length, building_width, segments)

    if not segments:
      return

    try:
      self.pc.output.save(segments)
    except WorkingDirNotSet:
      pass
    except (OSError, ValueError) as e:
      logger.catch(e)
      self.win.popup('Error', str(e))
    else:
      self.win.popup('Success', '저장 완료')
