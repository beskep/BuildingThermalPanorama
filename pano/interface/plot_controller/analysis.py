from typing import Any, Optional

from matplotlib import colors
from matplotlib.axes import Axes
from matplotlib.cm import get_cmap
from matplotlib.colorbar import make_axes_gridspec
from matplotlib.image import AxesImage
from matplotlib.patches import Patch
from matplotlib.widgets import _SelectorWidget
from matplotlib.widgets import PolygonSelector
import numpy as np
from skimage.draw import polygon2mask

from pano.interface import analysis
from pano.interface.common.pano_files import DIR
from pano.interface.common.pano_files import SP
from pano.interface.mbq import FigureCanvas
from pano.misc.imageio import ImageIO
from pano.misc.imageio import load_webp_mask
from pano.misc.tools import crop_mask
from pano.misc.tools import SegMask

from .plot_controller import PanoPlotController
from .plot_controller import QtGui


def _vulnerable_area_ratio(factor, vulnerable, mask):
  valid = (~np.isnan(factor)) & mask

  return np.sum(vulnerable & valid) / np.sum(valid)


class PointSelector(_SelectorWidget):

  def __init__(self, ax, onselect, markerprops: Optional[dict] = None) -> None:
    super().__init__(ax, onselect)

    self._marker = None
    self._markerprops = markerprops or dict(s=80, c='k', marker='x', alpha=0.8)

  @property
  def artists(self):
    artists = getattr(self, '_handles_artists', ())
    if self._marker is not None:
      artists += (self._marker,)

    return artists

  def _release(self, event):
    if self._marker is not None:
      self._marker.remove()

    self._marker = self.ax.scatter(event.xdata, event.ydata,
                                   **self._markerprops)

    coord = (int(np.round(event.ydata)), int(np.round(event.xdata)))
    self.onselect(coord)


class AnalysisPlotController(PanoPlotController):

  def __init__(self, parent=None) -> None:
    super().__init__(parent=parent)
    self._cax: Optional[Axes] = None
    self._images: Any = None
    self._axes_image: Optional[AxesImage] = None

    self._point = None  # 선택 지점 PathCollection
    self._coord = (-1, -1)  # 선택 지점 좌표 (y, x)
    self._selector: Optional[_SelectorWidget] = None
    self._multilayer = False

    self.show_point_temperature = lambda x: x / 0
    self.summarize = lambda: 1 / 0

    self._teti = (np.nan, np.nan)  # (exterior, interior temperature)
    self._threshold = 0.8  # 초기 취약부위 임계치

    self._seg_cmap = get_cmap('Dark2')
    self._seg_legend = None
    self._plot_setting = (False, False, False)

  @property
  def cax(self) -> Axes:
    if self._cax is None:
      raise ValueError('Colorbar ax not set')

    return self._cax

  def remove_images(self):
    self._images = None

  def _read_image(self, sp):
    return ImageIO.read(self.fm.panorama_path(DIR.COR, sp))

  def _read_images(self):
    ir = self._read_image(SP.IR).astype(np.float32)
    mask = self._read_image(SP.MASK)
    ir[np.logical_not(mask)] = np.nan

    if self._multilayer:
      path = self.fm.subdir(DIR.COR).joinpath('Panorama.webp')

      try:
        seg = load_webp_mask(path.as_posix())
      except ValueError as e:
        msg = '수동 수정 결과 인식 불가. 대상 파일에 부위 인식 레이어 (흑백)가 '
        if e.args[1] == 0:
          msg += '존재하지 않습니다.'
        else:
          msg += '두 개 이상 존재합니다.'

        raise ValueError(msg) from e
    else:
      seg = self._read_image(SP.SEG)[:, :, 0]
      seg = SegMask.vis_to_index(seg)

    return ir, seg

  @property
  def images(self) -> tuple[np.ndarray, np.ndarray]:
    if self._images is None:
      self._images = self._read_images()

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

    self.draw()

  def read_multilayer(self):
    self._multilayer = True
    self._images = None  # multilayer로부터 다시 읽기
    self.plot(factor=self._plot_setting[0],
              segmentation=True,
              vulnerable=self._plot_setting[2])

  def set_selector(self, point: Optional[bool] = None):
    if self._fm is None:
      self._selector = None
      return

    if self._selector is not None:
      self._selector.disconnect_events()
      for artist in self._selector.artists:
        artist.remove()

    if point is None:
      point = isinstance(self._selector, PointSelector)

    if point:
      self._selector = PointSelector(self.axes, self._on_point_select)
    else:
      self._selector = PolygonSelector(self.axes, self._on_polygon_select)

    self.draw()

  def cancel_selection(self):
    if isinstance(self._selector, PolygonSelector):
      self.remove_images()

    self.set_selector()
    self.plot(*self._plot_setting)

  def temperature_factor(self):
    if np.isnan(self.teti).any():
      raise ValueError('실내외 온도가 설정되지 않았습니다.')

    return np.absolute(
        (self.images[0] - self.teti[0]) / (self.teti[1] - self.teti[0]))

  def reset(self):
    self.axes.clear()
    self.cax.clear()

    if self._seg_legend is not None:
      self._seg_legend.remove()
      self._seg_legend = None

    self.set_selector()
    self.draw()

  def _set_style(self):
    self.axes.set_axis_off()

    if self.cax.has_data():
      self.cax.set_axis_on()
    else:
      self.cax.set_axis_off()

  def update_ir(self, ir):
    self._images = (ir, self._images[1])

  def _on_point_select(self, coord):
    self.draw()

    # 화면에 지점 온도 표시
    self._coord = coord
    pt = self.images[0][coord[0], coord[1]]
    self.show_point_temperature('NA' if np.isnan(pt) else f'{pt:.1f}')

  def _on_polygon_select(self, vertices):
    polygon = polygon2mask(image_shape=self.images[1].shape,
                           polygon=np.flip(vertices, 1))

    cr, cp = crop_mask(mask=polygon, morphology_open=False)

    ir = cr.crop(self.images[0])
    mask = cr.crop(self.images[1])
    ir[~cp] = np.nan
    mask[~cp] = False

    self._images = (ir, mask)
    self.plot(*self._plot_setting)

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
    factor = factor or vulnerable
    self._plot_setting = (factor, segmentation, vulnerable)

    if factor:
      # FIXME factor를 변경할 때마다 cax의 높이가 축소됨 (extend 설정 때문으로 추정)
      # https://matplotlib.org/stable/tutorials/intermediate/tight_layout_guide.html
      image = self.temperature_factor()
      norm = colors.BoundaryNorm(boundaries=np.linspace(0, 1, 11), ncolors=256)
    else:
      image = self.images[0].copy()
      norm = None

    cmap = self._get_cmap(factor=factor,
                          segmentation=segmentation,
                          vulnerable=vulnerable)

    self.reset()
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
                                         loc='lower right')

    if vulnerable:
      # TODO 벽 부위만 판단 옵션
      mask = (self.images[1] == 1) | (self.images[1] == 2)
      image[~mask] = np.nan
      image[mask & (image < self.threshold)] = np.nan  # 정상 부위 데이터 제외
      self.axes.imshow(image, cmap='inferno', vmin=0, vmax=1)

    self.set_selector()
    self.draw()
    self.summarize()

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
