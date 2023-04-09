from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.cm import get_cmap
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
import numpy as np
import seaborn as sns
import yaml

from pano.interface.common.pano_files import DIR
from pano.misc.imageio import ImageIO as IIO
from pano.misc.tools import OutlierArray
from pano.misc.tools import SegMask

from .plot_controller import FigureCanvas
from .plot_controller import PanoPlotController as _PanoPlotCtrl
from .plot_controller import QtGui


class _Axes:
  """
  [[IR, colorbar, anomaly],
   [dist]]
  """

  def __init__(self, fig: Figure, width_ratio=(1, 0.05, 1)) -> None:
    gs = GridSpec(2, 3, width_ratios=width_ratio)
    self._axes: tuple[Axes, ...] = tuple(
        fig.add_subplot(x) for x in [gs[0, 0], gs[0, 1], gs[0, 2], gs[1, :]])

    self._titles = ('열화상', None, '이상영역', '온도 분포')

  def __iter__(self):
    yield from self.axes

  @property
  def axes(self):
    """ir, colorbar, anomaly, dist"""
    return self._axes

  @property
  def ir(self) -> Axes:
    return self._axes[0]

  @property
  def colorbar(self) -> Axes:
    return self._axes[1]

  @property
  def anomaly(self) -> Axes:
    return self._axes[2]

  @property
  def dist(self) -> Axes:
    return self._axes[3]

  def set_style(self):
    self.ir.set_axis_off()
    self.anomaly.set_axis_off()

    for ax in (self.colorbar, self.dist):
      if ax.has_data():
        ax.set_axis_on()
      else:
        ax.set_axis_off()

    for ax, title in zip(self.axes, self._titles):
      if title and ax.has_data():
        ax.set_title(title)

    if self.colorbar.has_data():
      self.colorbar.set_ylabel('Temperature [℃]', rotation=90)
    if self.dist.has_data():
      self.dist.set_xlabel('Temperature [℃]')


class PlotController(_PanoPlotCtrl):

  def __init__(self, parent=None) -> None:
    super().__init__(parent)
    self._axes: _Axes
    self._threshold: dict[str, float] = {}
    self._cmap = get_cmap('inferno')

  @property
  def axes(self) -> _Axes:
    return self._axes

  def init(self, app: QtGui.QGuiApplication, canvas: FigureCanvas):
    self._app = app
    self._canvas = canvas
    self._fig = canvas.figure
    self._axes = _Axes(self._fig)
    self.reset()

  def _set_style(self):
    self.axes.set_style()
    self.fig.tight_layout()

  def update_threshold(self):
    p = self.fm.anomaly_path()
    with p.open('r') as f:
      self._threshold = yaml.safe_load(f)

  def plot(self, file: Path, anomaly: bool):
    ir = IIO.read(self.fm.change_dir(DIR.IR, file))
    vis = IIO.read(self.fm.change_dir(DIR.RGST, file))

    oa = OutlierArray(ir.ravel(), k=2.5)
    lim = (oa.lower, oa.upper)

    im = self.axes.ir.imshow(ir, cmap=self._cmap, vmin=lim[0], vmax=lim[1])
    self.fig.colorbar(im, cax=self.axes.colorbar, ticklocation='left')

    self.axes.anomaly.imshow(vis)

    if anomaly:
      try:
        threshold = self._threshold[file.stem]
      except KeyError as e:
        raise KeyError('threshold not found: "{}"', file) from e

      seg = IIO.read(self.fm.change_dir(DIR.SEG, file))
      seg = SegMask.vis_to_index(seg)
      mw = seg == SegMask.WALL  # wall mask
      ma = mw & (ir > threshold)  # anomaly mask

      self.axes.anomaly.imshow(np.ma.MaskedArray(ir, ~ma),
                               cmap=self._cmap,
                               vmin=lim[0],
                               vmax=lim[1])

      hist_data = {'정상 영역': ir[mw & ~ma].ravel(), '이상 영역': ir[ma].ravel()}
      sns.histplot(data=hist_data, stat='probability', ax=self.axes.dist)
      self.axes.dist.set_xlim(*lim)

    self.draw()
