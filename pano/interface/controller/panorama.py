import json
import multiprocessing as mp
from contextlib import suppress
from pathlib import Path

import numpy as np
from loguru import logger
from omegaconf import DictConfig, OmegaConf

import pano.interface.controller.controller as con
from pano.interface import analysis
from pano.interface import plot_controller as pc
from pano.interface.common import pano_files as pf
from pano.interface.common.config import update_config
from pano.interface.common.pano_files import DIR, SP
from pano.interface.mbq import QtCore, QtGui
from pano.interface.tree import tree_string
from pano.misc.imageio import ImageIO


def _save_manual_correction(
    queue: mp.Queue,
    wd: str,
    subdir: str,
    viewing_angle: float,
    angles: tuple,
    crop_range: np.ndarray | None,
):
  try:
    pc.save_manual_correction(wd, subdir, viewing_angle, angles, crop_range)
  except (ValueError, RuntimeError, OSError) as e:
    logger.exception(e)
    queue.put(f'{type(e).__name__}: {e}')

  queue.put(1.0)


def _segments(
    *,
    flag_count: bool,
    seg_count: int,
    seg_length: float,
    building_width: float,
) -> int:
  if flag_count:
    segments = seg_count
  elif np.isnan(building_width):
    raise ValueError('건물 폭을 설정하지 않았습니다.')
  else:
    segments = int(building_width / seg_length)
    if segments < 1:
      raise ValueError('분할 길이 또는 건물 폭이 잘못 설정되었습니다.')

  return segments


class Window(con.Window):

  def panel_function(self, panel: str, fn: str, *args):
    # TODO property()로 변경
    p = self._window.get_panel(panel)
    if p is None:
      msg = f'Invalid panel name: {panel}'
      raise ValueError(msg)

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
      'correct': '왜곡 보정',
  }

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
    self._pc = pc.PlotControllers(None, None, None, None, None)  # type: ignore[arg-type]
    self._config: DictConfig | None = None

  @property
  def win(self) -> Window:
    return self._win

  @win.setter
  def win(self, win: QtGui.QWindow):
    self._win = Window(win)

  @property
  def pc(self) -> pc.PlotControllers:
    return self._pc

  @property
  def fm(self) -> pf.ThermalPanoramaFileManager:
    if self._fm is None:
      raise ValueError('ThermalPanoramaFileManager not set')

    return self._fm

  def set_plot_controllers(self, controllers: dict):
    self._pc = pc.PlotControllers(**controllers)
    self._pc.analysis.summarize = self._analysis_summarize

  @QtCore.Slot(str)
  def log(self, message: str):
    con.log(message=message)

  @QtCore.Slot(str)
  def prj_select_working_dir(self, wd):
    wd = Path(wd).resolve()
    if not wd.exists():
      raise FileNotFoundError(wd)

    try:
      self._config = pf.init_directory(directory=wd)
    except pf.ImageNotFoundError as e:
      self.win.popup('Error', f'{e.args[0]}\n({e.args[1]})')
      return

    self._wd = wd
    self._fm = pf.ThermalPanoramaFileManager(wd)

    for controller in self.pc.controllers():
      controller.fm = self._fm
    self.pc.analysis.show_point_temperature = lambda x: self.win.panel_function(
        'analysis', 'show_point_temperature', x
    )

    self.win.update_config(self._config)

    self.prj_update_project_tree()
    self.update_image_view()

  def prj_update_project_tree(self):
    tree = tree_string(self._wd, width=40)
    self.win.panel_function('project', 'update_project_tree', tree)

  @QtCore.Slot(str)
  def prj_select_vis_images(self, files: str):
    if not self.fm.subdir(DIR.VIS).exists():
      self.win.popup('Warning', '열·실화상 데이터를 먼저 추출해주세요.', 5000)
      return

    try:
      pf.replace_vis_images(fm=self.fm, files=files)
    except OSError as e:
      logger.exception(e)
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
            d=(DIR.PANO if command == 'panorama' else DIR.COR), sp=SP.IR
        )
        self.win.panel_function('panorama', 'reset')
      elif command == 'register':
        self.pc.registration.reset()
        self.pc.registration.reset_matrices()

      self.win.popup('Success', f'{cmd_kr} 완료')
      self.win.pb_state(False)
      self.win.pb_value(1.0)

      with suppress(TypeError):
        self._consumer.done.disconnect()

    self._consumer.queue = queue
    self._consumer.done.connect(_done)
    self._consumer.start()

    process = mp.Process(
        name=command,
        target=con.producer,
        args=(queue, self._wd, command, self._loglevel),
    )
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
      vis_blend = OmegaConf.from_dotlist([f'panorama.blend.type.VIS={blend_type}'])

    if 'output' in config1:
      self.pc.output.configure(config1['output'])
      self.output_plot()

    return config1, vis_blend

  def _clear_separate_results(self):
    # 다음 결과 폴더 내 관련 파일 삭제
    exts = {'.npy', '.jpg', '.png', '.webp'}
    for d in (DIR.SEG, DIR.PANO, DIR.COR):
      files = [
          x for x in self.fm.glob(d, '*') if x.is_file() and x.suffix.lower() in exts
      ]
      logger.debug('delete files in {}: {}', d, [x.name for x in files])

      for f in files:
        f.unlink()

    # plot 리셋
    for pc in self.pc.controllers():
      pc.reset()
      pc.draw()

  def update_image_view(self, panels=('project', 'registration', 'segmentation')):
    raw_files = [con.path2uri(x) for x in self.fm.raw_files()]
    if not raw_files:
      logger.debug('no files')
      return

    try:
      vis_files = [con.path2uri(x) for x in self.fm.files(DIR.VIS, error=False)]
    except FileNotFoundError:
      vis_files = raw_files

    for panel in panels:
      files = vis_files if panel == 'segmentation' else raw_files
      self.win.panel_function(panel, 'update_image_view', files)

  @QtCore.Slot(str)
  def rgst_plot(self, url):
    assert self._wd is not None
    path = con.uri2path(url)

    try:
      self.pc.registration.plot(path)
    except FileNotFoundError:
      logger.warning('Data not extracted: {}', path)
      self.win.popup('Error', f'파일 {path.name}의 데이터가 추출되지 않았습니다.')

  @QtCore.Slot()
  def rgst_save(self):
    assert self._config is not None
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
    assert self._config is not None
    path = con.uri2path(url)

    try:
      self.pc.segmentation.plot(path, separate=self._config['panorama']['separate'])
    except FileNotFoundError:
      self.win.popup('Error', '부위 인식 결과가 없습니다.')
      logger.warning('File not found: {}', path)

  @QtCore.Slot(str, str)
  def pano_plot(self, d, sp):
    try:
      self.pc.panorama.plot(d=DIR[d], sp=SP[sp])
    except pc.WorkingDirNotSetError:
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

      with suppress(TypeError):
        self._consumer.done.disconnect()

    queue = mp.Queue()
    self._consumer.queue = queue
    self._consumer.done.connect(_done)
    self._consumer.start()

    pano = self.pc.panorama
    assert self._wd is not None
    args = (
        queue,
        self._wd.as_posix(),
        pano.subdir,
        pano.viewing_angle,
        (roll, pitch, yaw),
        pano.crop_range(),
    )
    process = mp.Process(name='save', target=_save_manual_correction, args=args)
    process.start()

  @QtCore.Slot()
  def rgst_pano_draw(self):
    if not self._fm:
      return

    ir = self.fm.panorama_path(d=DIR.PANO, sp=SP.IR)
    vis = self.fm.panorama_path(d=DIR.PANO, sp=SP.VIS)

    if any(not x.exists() for x in (ir, vis)):
      self.win.popup('Warning', '파노라마 생성 결과가 없습니다.')
      return

    # XXX 해상도 낮추기?
    self.pc.registration.set_images(
        fixed_image=ImageIO.read(ir), moving_image=ImageIO.read(vis)
    )
    self.pc.registration.draw()

  @QtCore.Slot(bool, bool, bool, bool)
  def analysis_plot(self, factor, segmentation, vulnerable, distribution):
    self.pc.analysis.setting.factor = factor
    self.pc.analysis.setting.segmentation = segmentation
    self.pc.analysis.setting.vulnerable = vulnerable
    self.pc.analysis.setting.distribution = distribution

    try:
      self.pc.analysis.plot()
    except pc.WorkingDirNotSetError:
      pass
    except FileNotFoundError as e:
      logger.debug('FileNotFound: "{}"', e)
    except ValueError as e:
      logger.exception(e)
      self.win.popup('Error', str(e))
    else:
      self.win.panel_function(
          'analysis',
          'set_temperature_range',
          *self.pc.analysis.images.temperature_range(),
      )
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
      logger.exception(e)
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
    except pc.WorkingDirNotSetError:
      return

    meta_files = list(self.fm.subdir(DIR.IR).glob('*.yaml'))

    for idx, e1 in zip([1, 2], [wall, window]):
      ir0 = np.full_like(ir, np.nan)
      mask = seg == idx

      ir0[mask] = ir[mask]
      ir1 = analysis.correct_emissivity(image=ir0, meta_files=meta_files, e1=e1)
      ir[mask] = ir1[mask]

    self.pc.analysis.images.ir = ir
    self.pc.analysis.correction_params.e_wall = wall
    self.pc.analysis.correction_params.e_window = window

    self.win.panel_function(
        'analysis',
        'set_temperature_range',
        *self.pc.analysis.images.temperature_range(),
    )

  @QtCore.Slot(float)
  def analysis_correct_temperature(self, temperature):
    try:
      ir, delta = analysis.correct_temperature(
          ir=self.pc.analysis.images.ir,
          mask=self.pc.analysis.images.seg,
          coord=self.pc.analysis.coord,
          T1=temperature,
      )
    except ValueError as e:
      logger.exception(e)
      self.win.popup('Error', str(e))
    else:
      self.pc.analysis.images.ir = ir
      self.pc.analysis.correction_params.delta_temperature = delta

      self.win.panel_function('analysis', 'show_point_temperature', temperature)
      self.win.panel_function(
          'analysis',
          'set_temperature_range',
          *self.pc.analysis.images.temperature_range(),
      )

  @QtCore.Slot(float, float)
  def analysis_set_teti(self, te, ti):
    self.pc.analysis.images.teti = (te, ti)

  @QtCore.Slot(float)
  def analysis_set_threshold(self, value):
    self.pc.analysis.images.threshold = value
    self._analysis_summarize()

  def _analysis_summarize(self):
    self.win.panel_function('analysis', 'clear_table')
    for cls, summary in self.pc.analysis.images.summarize().items():
      row = {k: v if isinstance(v, str) else f'{v:.2f}' for k, v in summary.items()}
      row['class'] = cls
      self.win.panel_function('analysis', 'add_table_row', row)

  @QtCore.Slot()
  def analysis_save(self):
    try:
      self.pc.analysis.save()
    except pc.WorkingDirNotSetError:
      pass
    except ValueError as e:
      logger.exception(e)
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
    except pc.WorkingDirNotSetError:
      pass
    except FileNotFoundError as e:
      logger.exception(e)

  @QtCore.Slot()
  def output_clear_lines(self):
    try:
      self.pc.output.lines.clear_lines()
      self.pc.output.draw()
    except pc.WorkingDirNotSetError:
      pass

  @QtCore.Slot()
  def output_estimate_edgelets(self):
    try:
      self.pc.output.estimate_edgelets()
    except pc.WorkingDirNotSetError:
      pass
    except FileNotFoundError as e:
      self.win.popup('Error', str(e))

  @QtCore.Slot(bool)
  def output_extend_lines(self, value):
    self.pc.output.lines.extend = value

  @QtCore.Slot(bool, int, float, float)
  def output_save(self, flag_count, seg_count, seg_length, building_width):
    try:
      segments = _segments(
          flag_count=flag_count,
          seg_count=seg_count,
          seg_length=seg_length,
          building_width=building_width,
      )
    except ValueError as e:
      logger.exception(e)
      self.win.popup('Error', str(e))
      segments = None

    logger.debug(
        'flag_count={} | count={} | length={} | width={} | segments={}',
        flag_count,
        seg_count,
        seg_length,
        building_width,
        segments,
    )

    if not segments:
      return

    try:
      self.pc.output.save(segments)
    except pc.WorkingDirNotSetError:
      pass
    except (OSError, ValueError) as e:
      logger.exception(e)
      self.win.popup('Error', str(e))
    else:
      self.win.popup('Success', '저장 완료')
