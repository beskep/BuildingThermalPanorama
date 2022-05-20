from typing import Any, Optional

from matplotlib import colors
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseEvent
from matplotlib.cm import get_cmap
from matplotlib.colorbar import make_axes_gridspec
from matplotlib.image import AxesImage
from matplotlib.patches import Patch
import numpy as np

from pano.interface import analysis
from pano.interface.common.pano_files import DIR
from pano.interface.common.pano_files import SP
from pano.interface.mbq import FigureCanvas
from pano.misc.imageio import ImageIO
from pano.misc.tools import SegMask

from .plot_controller import PanoPlotController
from .plot_controller import QtGui


def _vulnerable_area_ratio(factor, vulnerable, mask):
  valid = (~np.isnan(factor)) & mask

  return np.sum(vulnerable & valid) / np.sum(valid)


class AnalysisPlotController(PanoPlotController):
  # TODO 파노라마 생성/시점 보정 이후 image reset

  def __init__(self, parent=None) -> None:
    super().__init__(parent=parent)
    self._cax: Optional[Axes] = None
    self._images: Any = None
    self._axes_image: Optional[AxesImage] = None

    self._point = None  # 선택 지점 PathCollection
    self._coord = (-1, -1)  # 선택 지점 좌표 (y, x)

    self.show_point_temperature = lambda x: x / 0

    self._teti = (np.nan, np.nan)  # (exterior, interior temperature)
    self._threshold = 0.8  # 취약부위 임계치

    self._seg_cmap = get_cmap('Dark2')
    self._seg_legend = None

  @property
  def cax(self) -> Axes:
    if self._cax is None:
      raise ValueError('Colorbar ax not set')

    return self._cax

  def remove_images(self):
    self._images = None

  @property
  def images(self) -> tuple[np.ndarray, np.ndarray]:
    if self._images is None:
      if self.fm.panorama_path(DIR.COR, SP.SEG).exists():
        d = DIR.COR
      else:
        d = DIR.PANO

      ir = ImageIO.read(self.fm.panorama_path(d, SP.IR)).astype(np.float32)
      mask = ImageIO.read(self.fm.panorama_path(d, SP.MASK))
      ir[np.logical_not(mask)] = np.nan

      seg = ImageIO.read(self.fm.panorama_path(d, SP.SEG))[:, :, 0]
      seg = SegMask.vis_to_index(seg)

      self._images = (ir, seg)

    return self._images

  @property
  def coord(self):
    return self._coord

  @property
  def teti(self):
    return self._teti

  @teti.setter
  def teti(self, value):
    self._teti = (float(value[0]), float(value[1]))

  @property
  def threshold(self):
    return self._threshold

  @threshold.setter
  def threshold(self, value: float):
    v = float(value)
    if not (0 <= v <= 1):
      raise ValueError

    self._threshold = v

  def init(self, app: QtGui.QGuiApplication, canvas: FigureCanvas):
    self._app = app
    self._canvas = canvas

    self._fig = canvas.figure
    self._axes = self._fig.add_subplot(111)
    self._cax = make_axes_gridspec(self._axes)[0]
    self._fig.tight_layout()
    self.canvas.mpl_connect('button_press_event', self._on_click)

    self.draw()

  def temperature_factor(self):
    if np.isnan(self.teti).any():
      raise ValueError('실내외 온도가 설정되지 않았습니다.')

    return (self.images[0] - self.teti[0]) / (self.teti[1] - self.teti[0])

  def reset(self):
    self.axes.clear()
    self.cax.clear()

    if self._seg_legend is not None:
      self._seg_legend.remove()
      self._seg_legend = None

  def _set_style(self):
    self.axes.set_axis_off()

    if self.cax.has_data():
      self.cax.set_axis_on()
    else:
      self.cax.set_axis_off()

  def update_ir(self, ir):
    self._images = (ir, self._images[1])

  def _on_click(self, event: MouseEvent):
    ax: Axes = event.inaxes
    if ax is not self.axes:
      return

    if self._point is not None:
      self._point.remove()

    self._point = event.inaxes.scatter(event.xdata,
                                       event.ydata,
                                       s=60,
                                       c='seagreen',
                                       marker='x')
    self.draw()

    # 화면에 지점 온도 표시
    self._coord = (int(np.round(event.ydata)), int(np.round(event.xdata)))
    pt = self.images[0][self._coord[0], self._coord[1]]
    self.show_point_temperature('NA' if np.isnan(pt) else f'{pt:.1f}')

  @staticmethod
  def _get_cmap(factor=False, segmentation=False, vulnerable=False):
    if segmentation or vulnerable:
      name = 'gist_gray'
    elif factor:
      name = 'plasma'
    else:
      name = 'inferno'

    cmap = get_cmap(name).copy()

    if factor:
      cmap.set_over('white')
      cmap.set_under('black')

    return cmap

  def plot(self, factor=False, segmentation=False, vulnerable=False):
    self.reset()

    if vulnerable:
      factor = True

    if factor:
      image = self.temperature_factor()
      norm = colors.BoundaryNorm(boundaries=np.linspace(0, 1, 11), ncolors=256)
    else:
      image = self.images[0].copy()
      norm = None

    cmap = self._get_cmap(factor=factor,
                          segmentation=segmentation,
                          vulnerable=vulnerable)
    self._axes_image = self.axes.imshow(image, cmap=cmap, norm=norm)
    self.fig.colorbar(self._axes_image,
                      cax=self.cax,
                      ax=self.axes,
                      extend=('both' if factor else 'neither'))
    self.cax.set_ylabel('Temperature ' + ('Factor' if factor else '[℃]'),
                        rotation=90)

    if segmentation:
      seg = self._seg_cmap(self.images[1]).astype(float)
      seg[self.images[1] == 0] = np.nan
      seg[np.isnan(self.images[0])] = np.nan
      self.axes.imshow(seg, alpha=0.5)

      patches = [
          Patch(color=self._seg_cmap(i + 1), label=x)
          for i, x in enumerate(['Wall', 'Window', 'etc.'])
      ]
      self._seg_legend = self.fig.legend(handles=patches,
                                         ncol=len(patches),
                                         loc='lower center')

    if vulnerable:
      mask = (self.images[1] == 1) | (self.images[1] == 2)
      image[~mask] = np.nan
      image[mask & (image < self.threshold)] = np.nan  # 정상 부위 데이터 제외
      self.axes.imshow(image, cmap='inferno', vmin=0, vmax=1)

    self.draw()

  def temperature_range(self):
    mask = (self.images[1] == 1) | (self.images[1] == 2)
    image = self.images[0][mask]

    return (
        np.floor(np.nanmin(image)).item(),
        np.ceil(np.nanmax(image)).item(),
    )

  def set_clim(self,
               vmin: Optional[float] = None,
               vmax: Optional[float] = None):
    if self._axes_image is None:
      raise ValueError

    self._axes_image.set_clim(vmin, vmax)
    self.draw()

  def _summary(self, factor, index: int):
    mask = self.images[1] == index
    summ = analysis.summary(self.images[0][mask])

    if factor is None:
      summ['vulnerable'] = '-'
    else:
      v = _vulnerable_area_ratio(factor=factor,
                                 vulnerable=(factor >= self.threshold),
                                 mask=mask)
      summ['vulnerable'] = f'{v:.2%}'

    return summ

  def summary(self):
    try:
      factor = self.temperature_factor()
    except ValueError:
      factor = None

    return {
        'Wall': self._summary(factor, 1),
        'Window': self._summary(factor, 2)
    }
