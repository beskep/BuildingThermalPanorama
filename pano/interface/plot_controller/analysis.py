from contextlib import suppress
import dataclasses as dc
from shutil import copy2
from typing import Any, Optional

from matplotlib.axes import Axes
from matplotlib.cm import get_cmap
from matplotlib.colorbar import make_axes_gridspec
from matplotlib.colors import BoundaryNorm
from matplotlib.image import AxesImage
from matplotlib.patches import Patch
from matplotlib.widgets import _SelectorWidget
from matplotlib.widgets import PolygonSelector
import numpy as np
import seaborn as sns
from skimage.draw import polygon2mask

from pano import utils
from pano.interface import analysis
from pano.interface.common.pano_files import DIR
from pano.interface.common.pano_files import SP
from pano.interface.common.pano_files import ThermalPanoramaFileManager
from pano.interface.mbq import FigureCanvas
from pano.misc import tools
from pano.misc.cmap import apply_colormap
from pano.misc.imageio import ImageIO
from pano.misc.imageio import load_webp_mask
from pano.misc.subprocess import wkhtmltopdf

from .plot_controller import PanoPlotController
from .plot_controller import QtGui

SEG_CMAP = get_cmap('Dark2')


def _axes(fig):
  ax = fig.add_subplot(111)
  cax = make_axes_gridspec(ax)[0]
  return ax, cax


def _read_image(fm, sp):
  return ImageIO.read(fm.panorama_path(DIR.COR, sp))


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


def _vulnerable_area_ratio(factor, vulnerable, mask):
  valid = (~np.isnan(factor)) & mask

  return np.sum(vulnerable & valid) / np.sum(valid)


def _summarize(ir, seg, threshold, factor, index: int):
  mask = seg == index
  summ = analysis.summarize(ir[mask])

  if factor is None:
    summ['vulnerable'] = '-'
  else:
    v = _vulnerable_area_ratio(factor=factor,
                               vulnerable=(factor >= threshold),
                               mask=mask)
    summ['vulnerable'] = f'{v:.2%}'

  return summ


class Images:

  def __init__(self, fm: ThermalPanoramaFileManager) -> None:
    self._fm = fm

    self._ir: Optional[np.ndarray] = None  # IR image
    self._seg: Optional[np.ndarray] = None  # segmentation mask

    self._teti = (np.nan, np.nan)  # (exterior, interior temperature)
    self._threshold = 0.8  # 초기 취약부위 임계치
    self._multilayer = False
    self._crop: Optional[tuple[tools.CropRange, np.ndarray]] = None

    self.k = 2.0  # IQR 이상치 제거 변수

  @property
  def multilayer(self):
    return self._multilayer

  @multilayer.setter
  def multilayer(self, value):
    value = bool(value)

    if self._multilayer ^ value:
      self._seg = None

    self._multilayer = value

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
    if not (0.0 <= v <= 1.0):
      raise ValueError

    self._threshold = v

  def _read_ir(self):
    ir = _read_image(self._fm, SP.IR).astype(np.float32)
    mask = _read_image(self._fm, SP.MASK)
    ir[np.logical_not(mask)] = np.nan

    self._ir = ir

  def _read_seg(self):
    if not self._multilayer:
      seg = _read_image(self._fm, SP.SEG)[:, :, 0]
      seg = tools.SegMask.vis_to_index(seg)
    else:
      path = self._fm.subdir(DIR.COR).joinpath('Panorama.webp')

      try:
        seg = load_webp_mask(path.as_posix())
      except ValueError as e:
        msg = '수동 수정 결과 인식 불가. 대상 파일에 부위 인식 레이어 (흑백)가 '
        msg += ('존재하지 않습니다.' if e.args[1] == 0 else '두 개 이상 존재합니다.')
        raise ValueError(msg) from e

    self._seg = seg

  def reset_images(self):
    self._ir = None
    self._seg = None
    self._crop = None

  @property
  def ir(self):
    if self._ir is None:
      self._read_ir()
    return self._ir.copy()  # type: ignore

  @ir.setter
  def ir(self, value: np.ndarray):
    if not isinstance(value, np.ndarray):
      raise TypeError
    if self.ir.shape != value.shape:
      raise ValueError

    self._ir = value

  @property
  def seg(self):
    if self._seg is None:
      self._read_seg()
    return self._seg

  def temperature_range(self):
    mask = np.isin(self.seg, [1, 2])
    data = tools.OutlierArray(self.ir[mask], self.k).reject_outliers()

    return (
        np.floor(np.nanmin(data)).item(),
        np.ceil(np.nanmax(data)).item(),
    )

  def temperature_factor(self):
    if np.isnan(self.teti).any():
      raise ValueError('실내외 온도가 설정되지 않았습니다.')

    return np.absolute((self.ir - self.teti[0]) / (self.teti[1] - self.teti[0]))

  def vulnerable_area(self, window=True):
    if window:  # noqa: SIM108
      mask = np.isin(self.seg, [1, 2])
    else:
      mask = self.seg == 1

    factor = self.temperature_factor()
    factor[~mask] = np.nan
    factor[mask & (factor < self.threshold)] = np.nan  # 정상 부위 데이터 제외

    return factor

  def on_polygon_select(self, vertices):
    polygon = polygon2mask(image_shape=self.seg.shape,
                           polygon=np.flip(vertices, 1))
    cr, cp = tools.crop_mask(mask=polygon, morphology_open=False)

    self._ir = cr.crop(self.ir)
    self._ir[~cp] = np.nan  # type: ignore

    self._seg = cr.crop(self.seg)
    self._seg[~cp] = False  # type: ignore

    self._crop = (cr, cp)

  def summarize(self):
    try:
      factor = self.temperature_factor()
    except ValueError:
      factor = None

    return {
        'Wall': _summarize(self.ir, self.seg, self.threshold, factor, index=1),
        'Window': _summarize(self.ir, self.seg, self.threshold, factor, index=2)
    }

  def _path(self, sp, color=False):
    path = self._fm.panorama_path(DIR.ANLY, sp, error=False)
    if color:
      path = self._fm.color_path(path)
    if path.suffix != '.npy':
      path = path.parent / 'source' / path.name

    return path

  def save(self):
    self._path(SP.SEG).parent.mkdir(parents=True, exist_ok=True)

    # SEG
    ImageIO.save(self._path(SP.SEG), tools.SegMask.index_to_vis(self.seg))

    # IR
    ImageIO.save(self._path(SP.IR), self.ir)
    ImageIO.save(self._path(SP.IR, color=True),
                 apply_colormap(self.ir, cmap=_get_cmap(), na=True))

    # temperature factor
    with suppress(ValueError):
      ImageIO.save(self._path(SP.TF), self.temperature_factor())

    # VIS, mask
    if self._crop is not None:
      cr, cm = self._crop
      for sp in [SP.VIS, SP.MASK]:
        img = ImageIO.read(self._fm.panorama_path(DIR.COR, sp))
        img = cr.crop(img)
        img[~cm] = 0
        ImageIO.save(self._path(sp), img)
    else:
      copy2(self._fm.panorama_path(DIR.COR, SP.VIS), self._path(SP.VIS))
      copy2(self._fm.panorama_path(DIR.COR, SP.MASK), self._path(SP.MASK))


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


@dc.dataclass
class PlotSetting:
  factor: bool = False
  segmentation: bool = False
  vulnerable: bool = False
  distribution: bool = False
  window_vulnerable: bool = True  # 창문 취약부위를 plot에 표시할지 여부

  def update(self):
    self.factor = self.factor or self.vulnerable

  def __post_init__(self):
    self.update()


@dc.dataclass
class CorrectionParams:
  e_wall: float = np.nan
  e_window: float = np.nan
  delta_temperature: float = np.nan

  def asdict(self):
    return {
        k: ('-' if np.isnan(v) else f'{v:.2f}')
        for k, v in dc.asdict(self).items()
    }


class AnalysisPlotController(PanoPlotController):

  def __init__(self, parent=None) -> None:
    super().__init__(parent=parent)
    self._cax: Optional[Axes] = None
    self._axes_image: Optional[AxesImage] = None

    self._images: Any = None
    self._setting = PlotSetting()

    self._coord = (-1, -1)  # 선택 지점 좌표 (y, x)
    self._selector: Optional[_SelectorWidget] = None
    self._seg_legend = None

    self.show_point_temperature = lambda x: x / 0
    self.summarize = lambda: 1 / 0
    self.correction_params = CorrectionParams()

  @property
  def cax(self) -> Axes:
    if self._cax is None:
      raise ValueError('Colorbar ax not set')

    return self._cax

  @property
  def images(self) -> Images:
    if self._images is None:
      self._images = Images(fm=self.fm)
    return self._images

  @property
  def setting(self):
    return self._setting

  @property
  def coord(self):
    return self._coord

  def init(self, app: QtGui.QGuiApplication, canvas: FigureCanvas):
    self._app = app
    self._canvas = canvas

    self._fig = canvas.figure
    self._axes, self._cax = _axes(self.fig)

    self.draw()

  def read_multilayer(self):
    self.images.multilayer = True
    self._setting.segmentation = True
    self.plot()

  def set_selector(self, point: Optional[bool] = None, remove=False):
    if self._fm is None:
      self._selector = None
      return

    # 기존 selector 삭제
    if self._selector is not None:
      self._selector.disconnect_events()
      for artist in self._selector.artists:
        artist.remove()

    if remove:
      self._selector = None
      return

    # 새 selector 생성
    if point is None:
      point = isinstance(self._selector, PointSelector)

    if point:
      self._selector = PointSelector(self.axes, self._on_point_select)
    else:
      self._selector = PolygonSelector(self.axes, self._on_polygon_select)

    self.draw()

  def cancel_selection(self):
    if isinstance(self._selector, PolygonSelector):
      self.images.reset_images()

    self.set_selector()
    self.plot()

  def reset(self):
    self.axes.remove()
    self.cax.remove()
    self._axes, self._cax = _axes(self.fig)

    if self._seg_legend is not None:
      self._seg_legend.remove()
      self._seg_legend = None

    self.set_selector()
    self._set_style()

  def _set_style(self):
    if self._setting.distribution:
      self.axes.set_axis_on()  # histogram
      self.fig.tight_layout(pad=3)
    else:
      self.axes.set_axis_off()  # image
      self.fig.tight_layout(pad=2)

    if self.cax.has_data():
      self.cax.set_axis_on()
    else:
      self.cax.set_axis_off()

  def _on_point_select(self, coord):
    self.draw()

    # 화면에 지점 온도 표시
    self._coord = coord
    pt = self.images.ir[coord[0], coord[1]]
    self.show_point_temperature('NA' if np.isnan(pt) else f'{pt:.1f}')

  def _on_polygon_select(self, vertices):
    self.images.on_polygon_select(vertices=vertices)
    self.plot()

  def _plot_image(self, factor=False, segmentation=False, vulnerable=False):
    factor = factor or vulnerable

    if factor:
      # FIXME factor를 변경할 때마다 cax의 높이가 축소됨 (extend 설정 때문으로 추정)
      # https://matplotlib.org/stable/tutorials/intermediate/tight_layout_guide.html
      image = self.images.temperature_factor()
      norm = BoundaryNorm(boundaries=np.linspace(0, 1, 11), ncolors=256)
    else:
      image = self.images.ir
      norm = None

    cmap = _get_cmap(factor=factor,
                     segmentation=segmentation,
                     vulnerable=vulnerable)
    self._axes_image = self.axes.imshow(image, cmap=cmap, norm=norm)

    self.cax.set_visible(True)
    self.fig.colorbar(self._axes_image,
                      cax=self.cax,
                      ax=self.axes,
                      extend=('both' if factor else 'neither'))
    self.cax.set_ylabel('Temperature ' + ('Factor' if factor else '[℃]'),
                        rotation=90)

    if segmentation:
      seg = SEG_CMAP(self.images.seg).astype(float)
      seg[self.images.seg == 0] = np.nan
      seg[np.isnan(self.images.ir)] = np.nan
      self.axes.imshow(seg, alpha=0.5)

      patches = [
          Patch(color=SEG_CMAP(i + 1), label=x)
          for i, x in enumerate(['Wall', 'Window', 'etc.'])
      ]
      self._seg_legend = self.fig.legend(handles=patches,
                                         ncol=len(patches),
                                         loc='lower right')

    if vulnerable:
      va = self.images.vulnerable_area(window=self.setting.window_vulnerable)
      self.axes.imshow(va, cmap='inferno', vmin=0, vmax=1)

  def _plot_distribution(self):
    self.cax.set_visible(False)

    data = {
        'Wall': self.images.ir[self.images.seg == 1],
        'Window': self.images.ir[self.images.seg == 2]
    }
    data = {
        k: tools.OutlierArray(v, k=self.images.k).reject_outliers()
        for k, v in data.items()
    }
    data_range = (min(np.min(x) for x in data.values()),
                  max(np.max(x) for x in data.values()))

    sns.histplot(data=data,
                 stat='probability',
                 element='step',
                 binwidth=0.2,
                 ax=self.axes)
    self.axes.set_xlim(*data_range)
    self.axes.set_xlabel('Temperature [℃]')

  def plot(self):
    self.reset()

    self.setting.update()
    if self.setting.distribution:
      self._plot_distribution()
      self.set_selector(remove=True)
    else:
      self._plot_image(factor=self.setting.factor,
                       segmentation=self.setting.segmentation,
                       vulnerable=self.setting.vulnerable)
      self.set_selector()

    self.draw()
    self.summarize()

  def set_clim(self,
               vmin: Optional[float] = None,
               vmax: Optional[float] = None):
    if self._axes_image is None:
      raise ValueError

    self._axes_image.set_clim(vmin, vmax)
    self.draw()

  def save_plot(self):
    subdir = self.fm.subdir(DIR.ANLY) / 'source'
    subdir.mkdir(parents=True, exist_ok=True)
    distribution = self.setting.distribution

    # distribution
    self.reset()
    self.setting.distribution = True
    self._plot_distribution()
    self._set_style()
    self.fig.savefig(subdir.joinpath('TemperatureDistribution.png'), dpi=200)
    self.setting.distribution = distribution

    # factor
    self.reset()
    self._plot_image(factor=True)
    self._set_style()
    self.fig.savefig(subdir.joinpath(f'{SP.TF.value}.png'), dpi=200)

    # vulnerable area
    self.reset()
    self._plot_image(vulnerable=True)
    self._set_style()
    self.fig.savefig(subdir.joinpath('VulnerableArea.png'), dpi=200)

    self.plot()

  def _report_stats(self):
    summ = self.images.summarize()
    summ_wall = {f'{k}_wall': v for k, v in summ['Wall'].items()}
    summ_window = {f'{k}_window': v for k, v in summ['Window'].items()}
    summ_all = {
        k: (v if isinstance(v, str) else f'{v:.2f}℃')
        for k, v in (summ_wall | summ_window).items()
    }

    fmt = dict(exterior_temperature=self.images.teti[0],
               interior_temperature=self.images.teti[1],
               **self.correction_params.asdict(),
               **summ_all)

    return fmt

  def save_report(self):
    report = utils.DIR.RESOURCE / 'report'
    dst = self.fm.subdir(DIR.ANLY)

    fmt = self._report_stats()
    text = report.joinpath('Report.html').read_text(encoding='utf-8')
    text = text.format_map(fmt)

    # html
    html = dst / 'Report.html'
    html.write_text(data=text, encoding='utf-8')

    # pdf
    for css in report.glob('*.css'):
      copy2(css, dst / 'source')

    wkhtmltopdf(html, html.with_suffix('.pdf'))

  def save(self):
    self.images.save()
    self.save_plot()
    self.save_report()
