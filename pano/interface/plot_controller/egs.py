from collections import defaultdict
from collections.abc import Iterable
from functools import cached_property
from itertools import chain
from pathlib import Path
from typing import ClassVar

import numpy as np
import seaborn as sns
import yaml
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseButton, MouseEvent
from matplotlib.cm import get_cmap
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import Bbox
from numpy.typing import NDArray
from skimage import transform

from pano.interface.analysis import summarize
from pano.interface.common.pano_files import DIR, ThermalPanoramaFileManager
from pano.interface.mbq import NavigationToolbar2QtQuick
from pano.misc import tools
from pano.misc.imageio import ImageIO

from .plot_controller import FigureCanvas, QtGui
from .plot_controller import PanoPlotController as _PanoPlotCtrl

CMAP = get_cmap('inferno')


class DataNotFoundError(ValueError):
  PROCESS = ''
  DATA = ''

  def __init__(self, path=None) -> None:
    self.path = path

  def __str__(self) -> str:
    data = f' ({self.DATA} 미발견)' if self.DATA else ''
    return f'{self.PROCESS} 단계가 필요합니다.{data}'


class NotExtractedError(DataNotFoundError):
  PROCESS = '파일 추출'


class NotRegisteredError(DataNotFoundError):
  PROCESS = '열·실화상 정합'
  DATA = ''


class NotSegmentedError(DataNotFoundError):
  PROCESS = '이상 영역 검출'
  DATA = '외피 분할 데이터'


class AnomalyThresholdError(DataNotFoundError):
  PROCESS = '이상 영역 검출'
  DATA = '이상 영역 임계치'


class Images:
  def __init__(self, path: Path, fm: ThermalPanoramaFileManager) -> None:
    self._path = path
    self._fm = fm

  @cached_property
  def ir(self):
    try:
      return ImageIO.read(self._fm.change_dir('IR', self._path))
    except FileNotFoundError as e:
      raise NotExtractedError(e.args[0]) from e

  @cached_property
  def vis(self):
    try:
      return ImageIO.read(self._fm.change_dir(DIR.VIS, self._path))
    except FileNotFoundError as e:
      raise NotExtractedError(e.args[0]) from e

  def rgst(self):
    try:
      return ImageIO.read(self._fm.change_dir(DIR.RGST, self._path))
    except FileNotFoundError as e:
      raise NotRegisteredError(e.args[0]) from e

  def seg(self):
    try:
      array = ImageIO.read(self._fm.change_dir(DIR.SEG, self._path))
    except FileNotFoundError as e:
      raise NotSegmentedError(e.args[0]) from e

    return tools.SegMask.vis2index(array)

  def threshold(self):
    try:
      with self._fm.anomaly_path().open() as f:
        threshold = yaml.safe_load(f)

      return threshold[self._path.stem]
    except (FileNotFoundError, KeyError):
      raise AnomalyThresholdError(self._path) from None

  def data(self):
    wall = self.seg() == tools.SegMask.WALL
    anomaly = wall & (self.ir > self.threshold())
    array = {
      'normal': self.ir[wall & ~anomaly].ravel(),
      'anomaly': self.ir[anomaly].ravel(),
    }
    summary = {k: summarize(v) for k, v in array.items()}

    return anomaly, array, summary


class _Axes:
  """Matplotlib Axes.

  [[left, center, right],
   [bottom]]
  """

  def __init__(self, fig: Figure) -> None:
    gs = GridSpec(2, 3, width_ratios=(1, 0.05, 1))
    self.gs = gs
    self._axes: tuple[Axes, ...] = tuple(
      fig.add_subplot(x) for x in [gs[0, 0], gs[0, 1], gs[0, 2], gs[1, :]]
    )

    self.anomaly = False
    self._hide_bottom = False

  def __iter__(self):
    yield from self.axes

  @property
  def axes(self):
    """(left, center, right, bottom)"""
    return self._axes

  @property
  def left(self) -> Axes:
    return self._axes[0]

  @property
  def center(self) -> Axes:
    return self._axes[1]

  @property
  def right(self) -> Axes:
    return self._axes[2]

  @property
  def bottom(self) -> Axes:
    return self._axes[3]

  @property
  def hide_bottom(self):
    return self._hide_bottom

  @hide_bottom.setter
  def hide_bottom(self, value: bool):
    value = bool(value)
    self._hide_bottom = value
    self.gs.set_height_ratios((1, 0.2) if value else (1, 1))

  def titles(self):
    return ('열화상', None, '이상영역' if self.anomaly else '실화상', '온도 분포')

  def set_style(self):
    self.left.set_axis_off()
    self.right.set_axis_off()

    for ax, title in zip(self.axes, self.titles(), strict=True):
      if title and ax.has_data():
        ax.set_title(title)

    colorbar = self.center.has_data()

    if colorbar:
      self.center.set_axis_on()
      self.center.set_ylabel('Temperature [℃]', rotation=90)
    else:
      self.center.set_axis_off()
      self.center.set_title('')

    if colorbar and self.bottom.has_data():
      # distribution
      self.bottom.set_axis_on()
      self.bottom.set_xlabel('Temperature [℃]')
      self.bottom.set_aspect('auto')
    else:
      # diff
      self.bottom.set_axis_off()
      self.bottom.set_title('')


class NavigationToolbar(NavigationToolbar2QtQuick):
  def save_figure(self, *_):
    pass

  @property
  def zoom_mode(self):
    return bool(self.mode)  # default mode: '', zoom mode: 'zoom rect'

  def zoom(self, *, value: bool) -> None:
    if self.zoom_mode == bool(value):
      return

    super().zoom()


class MousePoints:
  _REQUIRED = 4
  _SCATTER_ARGS: ClassVar[dict] = {'s': 50, 'edgecolors': 'w', 'linewidths': 1}

  def __init__(self) -> None:
    self._points = defaultdict(list)  # 선택된 점들의 mpl 오브젝트
    self._coords = defaultdict(list)  # 선택된 점들의 좌표

  @property
  def coords(self):
    return self._coords

  def add_point(self, ax: Axes, event: MouseEvent):
    if len(self._coords[ax]) >= self._REQUIRED:
      return

    p = event.inaxes.scatter(event.xdata, event.ydata, **self._SCATTER_ARGS)
    self._points[ax].append(p)
    self._coords[ax].append((event.xdata, event.ydata))

  def _remove_points(self, ax: Axes):
    for p in self._points[ax]:
      p.remove()

    self._points.pop(ax, None)
    self._coords.pop(ax, None)

  def remove_points(self, ax: Axes | None):
    axes = tuple(self._points.keys()) if ax is None else (ax,)
    for ax_ in axes:
      self._remove_points(ax_)

  def all_selected(self):
    return len(self._coords) >= 2 and all(  # noqa: PLR2004
      len(x) >= self._REQUIRED for x in self._coords.values()
    )


def _limit(data: NDArray, k=2.5):
  # XXX 이상치 제거 방법 변경?
  oa = tools.OutlierArray(data.ravel(), k=k)
  return (oa.lower, oa.upper)


def _bbox(axes: Axes | Iterable[Axes], *, full=False):
  """From https://stackoverflow.com/questions/4325733/save-a-subplot-in-matplotlib."""
  if isinstance(axes, Axes):
    axes = [axes]

  if not full:
    items = axes
  else:
    items = chain.from_iterable(
      [
        ax,
        ax.title,
        ax.xaxis.label,
        ax.yaxis.label,
        *ax.get_xticklabels(),
        *ax.get_yticklabels(),
      ]
      for ax in axes
    )

  return Bbox.union([x.get_window_extent() for x in items])


class PlotController(_PanoPlotCtrl):
  def __init__(self, parent=None) -> None:
    super().__init__(parent)
    self._axes: _Axes
    self._toolbar: NavigationToolbar

    self._last_file: Path | None = None
    self._threshold: dict[str, float] = {}

    self._points = MousePoints()
    self._cid = None  # button_press_event cid

  @property
  def axes(self) -> _Axes:
    return self._axes

  def init(self, app: QtGui.QGuiApplication, canvas: FigureCanvas):
    self._app = app
    self._canvas = canvas
    self._toolbar = NavigationToolbar(canvas=canvas)
    self._fig = canvas.figure
    self._axes = _Axes(self._fig)
    self.reset()

  def _set_style(self):
    self.axes.set_style()
    self.fig.subplots_adjust()
    self.fig.tight_layout()

  def reset(self):
    super().reset()
    self.axes.anomaly = False
    self._last_file = None
    self._points.remove_points(None)

  @property
  def rgst(self) -> bool:
    return self._cid is not None

  @rgst.setter
  def rgst(self, value: bool):
    if value and self._cid is None:
      self._cid = self.canvas.mpl_connect('button_press_event', self._on_click)

    if not value and self._cid is not None:
      self.canvas.mpl_disconnect(self._cid)
      self._cid = None

  def _on_click(self, event: MouseEvent):
    if not self.axes.left.has_data():
      return

    ax = event.inaxes
    if ax not in {self.axes.left, self.axes.right}:
      return

    if event.button == MouseButton.LEFT and not self._toolbar.zoom_mode:
      self._points.add_point(ax=ax, event=event)
    elif event.button == MouseButton.RIGHT:
      self._points.remove_points(ax=ax)
    else:
      return

    if self._points.all_selected():
      self._register()

    self.draw()

  def update_threshold(self):
    p = self.fm.anomaly_path()
    with p.open() as f:
      self._threshold = yaml.safe_load(f)

  def navigation(self, *, home: bool, zoom: bool):
    if self.axes.left.has_data() and home:
      self._toolbar.home()
      return

    self._toolbar.zoom(value=zoom)

  def plot(self, path: Path, mode: str):
    modes = {'raw', 'registration', 'anomaly'}
    if mode not in modes:
      msg = f'{mode!r} not in {modes!r}'
      raise ValueError(msg)

    self.reset()
    self._last_file = path
    self.axes.hide_bottom = mode == 'raw'  # raw plot의 경우 bottom 높이를 축소

    images = Images(path, self.fm)

    if mode == 'registration':
      self._plot_rgst(images)
      summary = None
    else:
      summary = self._plot(images, anomaly=mode == 'anomaly')

    if mode == 'anomaly':
      self._save(stem=path.stem)

    return summary

  def _plot(self, images: Images, *, anomaly: bool):
    lim = _limit(images.ir)
    im = self.axes.left.imshow(images.ir, cmap=CMAP, vmin=lim[0], vmax=lim[1])
    self.fig.colorbar(im, cax=self.axes.center, ticklocation='left')

    self.axes.right.imshow(images.rgst() if anomaly else images.vis)

    summary = None
    if anomaly:
      summary = self._plot_anomaly(images=images, lim=lim)
      self.axes.anomaly = True

    self.draw()

    return summary

  def _plot_anomaly(self, images: Images, lim: tuple[float, float]):
    anomaly, array, summary = images.data()

    self.axes.right.imshow(
      np.ma.MaskedArray(images.ir, ~anomaly), cmap=CMAP, vmin=lim[0], vmax=lim[1]
    )

    array = {'정상 영역': array['normal'], '이상 영역': array['anomaly']}
    sns.histplot(data=array, stat='probability', ax=self.axes.bottom)
    self.axes.bottom.set_xlim(*lim)

    return summary

  def _plot_rgst(self, images: Images):
    lim = _limit(images.ir)
    self.axes.left.imshow(images.ir, cmap=CMAP, vmin=lim[0], vmax=lim[1])
    self.axes.right.imshow(images.vis)

    try:
      rgst = images.rgst()
    except NotRegisteredError:
      pass
    else:
      self._plot_rgst_compare(ir=images.ir, registered=rgst)

    self.draw()

  def _plot_rgst_compare(self, ir: NDArray, registered: NDArray):
    checkerboard = tools.prep_compare_images(
      ir, registered, norm=True, eq_hist=True, method='checkerboard'
    )
    self.axes.bottom.imshow(checkerboard)

  def _register(self):
    assert self._last_file is not None

    src = np.array(self._points.coords[self.axes.right])
    dst = np.array(self._points.coords[self.axes.left])

    trsf = transform.ProjectiveTransform()
    trsf.estimate(src=src, dst=dst)

    image = Images(self._last_file, self.fm)
    registered = transform.warp(
      image=image.vis,
      inverse_map=trsf.inverse,
      output_shape=image.ir.shape[:2],
      preserve_range=True,
    )

    self._plot_rgst_compare(ir=image.ir, registered=registered)

    self.fm.subdir(DIR.RGST).mkdir(exist_ok=True)
    ImageIO.save(
      path=self.fm.change_dir(DIR.RGST, self._last_file),
      array=tools.uint8_image(registered),
    )

  def _save(self, stem: str, dpi=300):
    directory = self.fm.wd / '03 Report'  # TODO file manager
    directory.mkdir(exist_ok=True)

    axes = [self.axes.left, self.axes.right, self.axes.bottom]
    suffixes = ['', '_anomaly', '_hist']

    for ax, suffix in zip(axes, suffixes, strict=True):
      path = directory / f'{stem}{suffix}.png'
      bbox = _bbox(ax, full=suffix == '_hist').transformed(
        self.fig.dpi_scale_trans.inverted()
      )
      self.fig.savefig(path, bbox_inches=bbox, dpi=dpi)
