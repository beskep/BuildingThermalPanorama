import dataclasses as dc
import multiprocessing as mp
from collections.abc import Iterable
from contextlib import suppress
from enum import IntEnum
from pathlib import Path
from shutil import copy2

from loguru import logger
from omegaconf import DictConfig
from toolz import dicttoolz

import pano.interface.common.pano_files as pf
import pano.interface.controller.controller as con
from pano.flir.extractor import FlirExtractor
from pano.interface.mbq import QtCore, QtGui
from pano.interface.plot_controller.egs import DataNotFoundError, Images, PlotController
from pano.misc.sp import wkhtmltopdf
from pano.utils import DIR

# ruff: noqa: FBT003


class Window(con.Window):

  def __init__(self, window: QtGui.QWindow) -> None:
    super().__init__(window)
    self._panel = self._window.property('panel')

  @property
  def panel(self):
    return self._panel


class Mode(IntEnum):
  RAW = 0
  REGISTRATION = 1
  ANOMALY = 2
  REPORT = 3


@dc.dataclass
class ProjectData:
  location: str = ''
  building: str = ''
  date: str = ''
  part: str = ''
  etc: str = ''

  def replace(self, **kwargs):
    return dc.replace(self, **kwargs)

  def asdict(self):
    return dc.asdict(self)


@dc.dataclass
class IRParameter:
  Emissivity: float
  ReflectedApparentTemperature: float
  SubjectDistance: float
  RelativeHumidity: float
  AtmosphericTemperature: float

  @classmethod
  def fields(cls):
    return dc.fields(cls)

  @classmethod
  def from_file(cls, path):
    meta = FlirExtractor(str(path)).meta
    return cls(**{f.name: getattr(meta, f.name) for f in cls.fields()})

  def aslist(self):
    return [
        round(self.Emissivity, 4),
        f'{self.ReflectedApparentTemperature} ℃',
        f'{self.SubjectDistance} m',
        f'{self.RelativeHumidity:.1%}',
        f'{self.AtmosphericTemperature} ℃',
    ]

  def asdict(self):
    return {f.name: x for f, x in zip(self.fields(), self.aslist(), strict=True)}


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

    self._prj_data = ProjectData()
    self._summary: dict[Path, dict] = {}

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

  @QtCore.Slot(str, str)
  def set_project_data(self, key, value):
    self._prj_data = self._prj_data.replace(**{key: value})

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
    self.win.panel.update_image_view(files)

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
  def display(self, uri, mode):
    self.pc.reset()

    path = con.uri2path(uri) if uri else None
    self.display_params(path if mode == Mode.RAW else None)

    if not path:
      return

    if mode == Mode.REPORT:
      try:
        self.display_report(path)
      except DataNotFoundError as e:
        logger.debug(f'{e} | path="{e.path}"')
        self.win.popup('Error', str(e))

      return

    summary = self.plot(path=path, mode=mode)
    if summary is not None:
      self._summary[path] = summary
      self.display_stat(summary)

  def plot(self, path: Path, mode: int):
    self.pc.rgst = mode == Mode.REGISTRATION
    summary = None

    try:
      summary = self.pc.plot(path=path, mode=Mode(mode).name.lower())
    except DataNotFoundError as e:
      logger.debug(f'{e} | path="{e.path}"')
      self.win.popup('Error', str(e))

    return summary

  def display_params(self, path):
    if path:
      params = IRParameter.from_file(path).aslist()
    else:
      params = ['' for _ in IRParameter.fields()]

    parameter = self.win.panel.property('parameter')
    parameter.display(params)

  def display_stat(self, summary: dict[str, dict[str, float]]):
    self.win.panel.clear_stat()
    for c, d in summary.items():
      row = {k: v if isinstance(v, str) else f'{v:.2f}' for k, v in d.items()}
      row['class'] = {'normal': '정상 영역', 'anomaly': '이상 영역'}[c]
      self.win.panel.append_stat_row(row)

  def _report(self, image: Path, template: Path, dst: Path) -> str:
    prj = self._prj_data.asdict()
    params = IRParameter.from_file(image).asdict()

    _, _, summary = Images(image, self.fm).data()
    stat = {
        f'{x}_{y}': summary[x][y]
        for x in ['normal', 'anomaly']
        for y in ['avg', 'min', 'max']
    }

    images = {
        'IR': dst / f'{image.stem}.png',
        'AnomalyPlot': dst / f'{image.stem}_anomaly.png',
        'Histogram': dst / f'{image.stem}_hist.png',
    }

    fmt = dicttoolz.merge(
        prj,
        params,
        {k: f'{v:.1f} ℃' for k, v in stat.items()},
        {k: v.as_posix() for k, v in images.items()},
    )

    return template.read_text(encoding='UTF-8').format_map(fmt)

  def display_report(self, path: Path):
    assert self._wd is not None

    src = DIR.RESOURCE / 'report'
    dst = self._wd / '03 Report'  # TODO file manager
    dst.mkdir(exist_ok=True)

    report = self._report(image=path, template=src / 'EGReport.html', dst=dst)
    html = dst.joinpath(path.stem).with_suffix('.html')
    html.write_text(report, encoding='UTF-8')

    for css in src.glob('*.css'):
      copy2(css, dst)

    self.win.panel.web_view(html.as_posix())

  @QtCore.Slot(bool, bool)
  def plot_navigation(self, home, zoom):
    self.pc.navigation(home=home, zoom=zoom)

  @QtCore.Slot(str, str)
  def qml_command(self, commands: str, name: str):
    self.command(commands=map(str.strip, commands.split(',')), name=name)

  @QtCore.Slot(str, str)
  def save_report(self, html, pdf):
    html = con.uri2path(html)
    pdf = con.uri2path(pdf)
    wkhtmltopdf(src=html, dst=pdf)
    self.win.popup('Success', '보고서 저장 성공')
