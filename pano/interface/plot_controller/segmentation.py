from pathlib import Path

import numpy as np
from matplotlib.cm import get_cmap
from matplotlib.patches import Patch

from pano.interface.common.pano_files import DIR
from pano.interface.mbq import FigureCanvas
from pano.misc.imageio import ImageIO
from pano.misc.tools import SegMask

from .plot_controller import PanoPlotController, QtGui


class SegmentationPlotController(PanoPlotController):
  _TITLES = ('실화상', '부위 인식 결과')
  _CLASSES = ('Background', 'Wall', 'Window', 'etc.')

  def __init__(self, parent=None) -> None:
    super().__init__(parent=parent)
    self._cmap = get_cmap('Dark2')
    self._legend = None

  @property
  def axes(self) -> np.ndarray:
    return super().axes

  def init(self, app: QtGui.QGuiApplication, canvas: FigureCanvas):
    self._app = app
    self._canvas = canvas

    self._fig = canvas.figure
    self._axes = self._fig.subplots(1, 2)

    self.draw()

  def _set_style(self):
    for ax, title in zip(self.axes.ravel(), self._TITLES, strict=True):
      if ax.has_data():
        ax.set_title(title)
      ax.set_axis_off()

  def plot(self, file: Path, *, separate: bool):
    vis_path = self.fm.change_dir((DIR.VIS if separate else DIR.RGST), file)
    mask_path = self.fm.change_dir(DIR.SEG, file)

    if not (vis_path.exists() and mask_path.exists()):
      raise FileNotFoundError(file)

    vis = ImageIO.read(vis_path)
    mask_vis = ImageIO.read(mask_path)
    mask = SegMask.vis_to_index(mask_vis)
    mask_cmap = self._cmap(mask)

    self.axes[0].clear()
    self.axes[1].clear()

    self.axes[0].imshow(vis)
    self.axes[1].imshow(vis)
    self.axes[1].imshow(mask_cmap, alpha=0.7)

    if self._legend is None:
      patches = [
          Patch(color=self._cmap(i), label=label)
          for i, label in enumerate(self._CLASSES)
      ]
      self._legend = self.fig.legend(
          handles=patches,
          ncol=len(patches),
          loc='lower center',
          bbox_to_anchor=(0.5, 0.01),
      )
      self.fig.tight_layout()

    self.draw()
