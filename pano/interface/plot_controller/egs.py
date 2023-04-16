from collections import defaultdict
from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseButton
from matplotlib.backend_bases import MouseEvent
from matplotlib.cm import get_cmap
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
import numpy as np
from numpy.typing import NDArray
import seaborn as sns
from skimage import transform
import yaml

from pano.interface.analysis import summarize
from pano.interface.common.pano_files import DIR
from pano.interface.mbq import NavigationToolbar2QtQuick
from pano.misc import tools
from pano.misc.imageio import ImageIO as IIO

from .plot_controller import FigureCanvas
from .plot_controller import PanoPlotController as _PanoPlotCtrl
from .plot_controller import QtGui


class _Axes:
  """
  [[left, center, right],
   [bottom]]
  """

  def __init__(self, fig: Figure, width_ratio=(1, 0.05, 1)) -> None:
    gs = GridSpec(2, 3, width_ratios=width_ratio)
    self._axes: tuple[Axes, ...] = tuple(
        fig.add_subplot(x) for x in [gs[0, 0], gs[0, 1], gs[0, 2], gs[1, :]])

    self._titles = ('열화상', None, '이상영역', '온도 분포')  # rgst일 때 수정

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

  def set_style(self):
    self.left.set_axis_off()
    self.right.set_axis_off()

    for ax, title in zip(self.axes, self._titles, strict=True):
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


class _NavigationToolbar(NavigationToolbar2QtQuick):

  def __init__(self, canvas, parent=None):
    super().__init__(canvas, parent)

  def zoom(self, value: bool) -> None:
    current = bool(self.mode)  # default mode: '', zoom mode: 'zoom rect'
    if current == bool(value):
      return

    super().zoom()


class _MousePoints:
  _REQUIRED = 4
  _SCATTER_ARGS = {'s': 50, 'edgecolors': 'w', 'linewidths': 1}

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
    for ax in axes:
      self._remove_points(ax)

  def all_selected(self):
    return (len(self._coords) >= 2 and
            all(len(x) >= self._REQUIRED for x in self._coords.values()))


class AnomalyThresholdNotSetError(ValueError):
  pass


def _limit(data: NDArray, k=2.5):
  # XXX 이상치 제거 방법 변경?
  oa = tools.OutlierArray(data.ravel(), k=k)
  return (oa.lower, oa.upper)


class PlotController(_PanoPlotCtrl):

  def __init__(self, parent=None) -> None:
    super().__init__(parent)
    self._axes: _Axes
    self._cmap = get_cmap('inferno')
    self._toolbar: _NavigationToolbar

    self._last_file: Path | None = None
    self._threshold: dict[str, float] = {}

    self._points = _MousePoints()
    self._cid = None  # button_press_event cid

  @property
  def axes(self) -> _Axes:
    return self._axes

  def init(self, app: QtGui.QGuiApplication, canvas: FigureCanvas):
    self._app = app
    self._canvas = canvas
    self._toolbar = _NavigationToolbar(canvas=canvas)
    self._fig = canvas.figure
    self._axes = _Axes(self._fig)
    self.reset()

  def _set_style(self):
    self.axes.set_style()
    self.fig.tight_layout()

  def reset(self):
    super().reset()
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
    if ax is None:
      return

    if event.button == MouseButton.LEFT:
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
    with p.open('r') as f:
      self._threshold = yaml.safe_load(f)

  def navigation(self, home: bool, zoom: bool):
    if self.axes.left.has_data() and home:
      self._toolbar.home()
      return

    self._toolbar.zoom(zoom)

  def plot(self, path: Path, mode: str):
    modes = {'raw', 'registration', 'anomaly'}
    if mode not in modes:
      raise ValueError(f'{mode!r} not in {modes!r}')

    self.reset()
    self._last_file = path

    if mode == 'registration':
      summary = self._plot_rgst(path)
    else:
      summary = self._plot(path, anomaly=mode == 'anomaly')

    return summary

  def _plot(self, path: Path, anomaly: bool):
    ir = IIO.read(self.fm.change_dir(DIR.IR, path))

    try:
      vis = IIO.read(self.fm.change_dir(DIR.RGST, path))
    except FileNotFoundError:
      vis = IIO.read(self.fm.change_dir(DIR.VIS, path))

    lim = _limit(ir)
    im = self.axes.left.imshow(ir, cmap=self._cmap, vmin=lim[0], vmax=lim[1])
    self.fig.colorbar(im, cax=self.axes.center, ticklocation='left')

    self.axes.right.imshow(vis)

    if anomaly:
      summary = self._plot_anomaly(path=path, ir=ir, lim=lim)
    else:
      summary = None

    self.draw()

    return summary

  def _plot_anomaly(self, path: Path, ir: NDArray, lim: tuple[float, float]):
    try:
      threshold = self._threshold[path.stem]
    except KeyError as e:
      raise AnomalyThresholdNotSetError(f'threshold not found: "{path}"') from e

    seg = IIO.read(self.fm.change_dir(DIR.SEG, path))
    seg = tools.SegMask.vis_to_index(seg)
    mw = seg == tools.SegMask.WALL  # wall mask
    ma = mw & (ir > threshold)  # anomaly mask

    self.axes.right.imshow(np.ma.MaskedArray(ir, ~ma),
                           cmap=self._cmap,
                           vmin=lim[0],
                           vmax=lim[1])

    data = {'정상 영역': ir[mw & ~ma].ravel(), '이상 영역': ir[ma].ravel()}
    sns.histplot(data=data, stat='probability', ax=self.axes.bottom)
    self.axes.bottom.set_xlim(*lim)

    return {k: summarize(v) for k, v in data.items()}

  def _plot_rgst(self, path: Path):
    ir = IIO.read(self.fm.change_dir(DIR.IR, path))
    vis = IIO.read(self.fm.change_dir(DIR.VIS, path))

    lim = _limit(ir)
    self.axes.left.imshow(ir, cmap=self._cmap, vmin=lim[0], vmax=lim[1])
    self.axes.right.imshow(vis)

    if (p := self.fm.change_dir(DIR.RGST, path)).exists():
      registered = IIO.read(p)
      self._plot_rgst_compare(ir=ir, registered=registered)

    self.draw()

  def _plot_rgst_compare(self, ir: NDArray, registered: NDArray | None = None):
    assert self._last_file is not None

    if registered is None:
      try:
        registered = IIO.read(self.fm.change_dir(DIR.RGST, self._last_file))
      except FileNotFoundError as e:
        raise ValueError('정합 영상이 없음') from e

    checkerboard = tools.prep_compare_images(ir,
                                             registered,
                                             norm=True,
                                             eq_hist=True,
                                             method='checkerboard')
    self.axes.bottom.imshow(checkerboard)

  def _register(self):
    assert self._last_file is not None

    src = np.array(self._points.coords[self.axes.right])
    dst = np.array(self._points.coords[self.axes.left])

    trsf = transform.ProjectiveTransform()
    trsf.estimate(src=src, dst=dst)

    ir = IIO.read(self.fm.change_dir(DIR.IR, self._last_file))
    vis = IIO.read(self.fm.change_dir(DIR.VIS, self._last_file))

    registered = transform.warp(image=vis,
                                inverse_map=trsf.inverse,
                                output_shape=ir.shape[:2],
                                preserve_range=True)

    self._plot_rgst_compare(ir=ir, registered=registered)

    self.fm.subdir(DIR.RGST).mkdir(exist_ok=True)
    IIO.save(path=self.fm.change_dir(DIR.RGST, self._last_file),
             array=tools.uint8_image(registered))
