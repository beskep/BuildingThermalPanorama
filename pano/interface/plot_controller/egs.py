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


class PlotController(_PanoPlotCtrl):

  def __init__(self, parent=None) -> None:
    super().__init__(parent)
    self._axes: tuple[Axes, ...]
    self._threshold: dict[str, float] = {}
    self._cmap = get_cmap('inferno')

  @property
  def axes(self) -> tuple[Axes, ...]:
    return self._axes

  def init(self, app: QtGui.QGuiApplication, canvas: FigureCanvas):
    self._app = app
    self._canvas = canvas

    fig: Figure = canvas.figure
    gs = GridSpec(2, 2)
    axes = tuple(fig.add_subplot(x) for x in [gs[0, 0], gs[0, 1], gs[1, :]])

    self._fig = fig
    self._axes = axes
    self.reset()

  def _set_style(self):
    self.axes[0].set_axis_off()
    self.axes[1].set_axis_off()

    ax = self.axes[2]
    if ax.has_data():
      ax.set_axis_on()
      ax.set_xlabel('Temperature [℃]')
    else:
      ax.set_axis_off()

    for ax, title in zip(self.axes, ('열화상', '실화상', '온도분포')):
      if ax.has_data():
        ax.set_title(title)

  def update_threshold(self):
    p = self.fm.anomaly_path()
    with p.open('r') as f:
      self._threshold = yaml.safe_load(f)

  def plot(self, file: Path, anomaly: bool):
    ir = IIO.read(self.fm.change_dir(DIR.IR, file))
    vis = IIO.read(self.fm.change_dir(DIR.RGST, file))

    oa = OutlierArray(ir.ravel(), k=2.5)
    lim = (oa.lower, oa.upper)

    # TODO colorbar
    self.axes[0].imshow(ir, cmap=self._cmap, vmin=lim[0], vmax=lim[1])
    self.axes[1].imshow(vis)

    if anomaly:
      try:
        threshold = self._threshold[file.stem]
      except KeyError as e:
        raise KeyError('threshold not found: "{}"', file) from e

      seg = IIO.read(self.fm.change_dir(DIR.SEG, file))
      seg = SegMask.vis_to_index(seg)
      mw = seg == SegMask.WALL  # wall mask
      ma = mw & (ir > threshold)  # anomaly mask

      self.axes[1].imshow(np.ma.MaskedArray(ir, ~ma),
                          cmap=self._cmap,
                          vmin=lim[0],
                          vmax=lim[1])

      hist_data = {'정상 영역': ir[mw & ~ma].ravel(), '이상 영역': ir[ma].ravel()}
      sns.histplot(data=hist_data, stat='probability', ax=self.axes[2])
      self.axes[2].set_xlim(*lim)

    self.draw()
