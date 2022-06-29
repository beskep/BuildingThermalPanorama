from dataclasses import dataclass
from typing import Any, Literal, Optional, Union

from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.widgets import _SelectorWidget
import numpy as np

from pano.interface.common.pano_files import DIR
from pano.interface.common.pano_files import SP
from pano.interface.common.pano_files import ThermalPanoramaFileManager
from pano.interface.mbq import FigureCanvas
from pano.misc import edgelet as edge
from pano.misc.edgelet import CannyOptions
from pano.misc.edgelet import Edgelets
from pano.misc.edgelet import HoughOptions
from pano.misc.imageio import ImageIO as IIO

from .plot_controller import PanoPlotController
from .plot_controller import QtGui
from .plot_controller import WorkingDirNotSet


def _suppress_edgelets(edgelets: Edgelets,
                       distance_threshold=10,
                       angle_threshold=10,
                       repeat: Optional[int] = None):
  repeat = repeat or edgelets.count
  angle_threshold = np.deg2rad(angle_threshold)

  edgelets.sort()
  edgelets.normalize()

  for idx in range(repeat):
    try:
      e = edgelets[idx]
    except IndexError:
      break

    normal = np.array([-e.directions[0, 1], e.directions[0, 0]])
    dist = np.sum(normal * (edgelets.locations - e.locations[0]), axis=1)

    cos = np.sum(e.directions * edgelets.directions, axis=1)
    angle = np.arccos(np.clip(cos, a_min=-1.0, a_max=1.0))

    valid = (np.abs(dist) > distance_threshold) & (angle <= angle_threshold)
    valid[idx] = True
    edgelets = edgelets[valid]

  return edgelets


def extend_lines(n: Any, p: Any, xlim: Any, ylim: Any) -> tuple[tuple, tuple]:
  """
  주어진 선 `dot(n, (X-p))=0`을 x, y 범위가 xlim, ylim인
  직사각형의 경계까지 확장하는 점 두개 반환.

  Parameters
  ----------
  n : Any
      (nx, ny)
  p : Any
      (px, py)
  xlim : Any
      (xmin, xmax) 직사각형의 x방향 경계
  ylim : Any
      (ymin, ymax) 직사각형의 y방향 경계

  Returns
  -------
  tuple[tuple, tuple]
      ((x0, y0), (x1, y1))
  """
  nx, ny = n
  px, py = p
  x0, x1 = xlim
  y0, y1 = ylim

  points = [
      (x0, py - np.divide(nx * (x0 - px), ny)),  # x=x0 선과 만나는 지점
      (x1, py - np.divide(nx * (x1 - px), ny)),  # x=x1 선과 만나는 지점
      (px - np.divide(ny * (y0 - py), nx), y0),  # y=y0 선과 만나는 지점
      (px - np.divide(ny * (y1 - py), nx), y1),  # y=y1 선과 만나는 지점
  ]
  argsort = np.argsort([x[0] for x in points])

  return points[argsort[1]], points[argsort[2]]  # x좌표가 중간인 두 점 반환


class LinesSelector(_SelectorWidget):
  DIST_THOLD = 10

  def __init__(self, ax, useblit=False, update=None) -> None:
    super().__init__(ax, onselect=lambda *args, **kwargs: None, useblit=useblit)

    self._props = dict(alpha=0.6,
                       animated=False,
                       color='k',
                       label='_nolegend_',
                       linestyle='-',
                       marker='D')
    self._extend = False

    self._lines: list[Line2D] = []
    self._current_line: Optional[Line2D] = None  # 마우스 클릭으로 생성 중인 line
    self._active_index = (-1, -1)  # line index, point index

    self._update = update or super().update

  @property
  def extend(self):
    return self._extend

  @extend.setter
  def extend(self, value):
    self._extend = bool(value)
    if self._extend:
      self.update()

  def update(self):
    if self._extend:
      self.extend_lines()
    self._update()

  def make_line(self, xs=None, ys=None):
    xs = xs if xs is not None else [np.nan]
    ys = ys if ys is not None else [np.nan]
    line = Line2D(xs, ys, **self._props)
    self.ax.add_line(line)
    self._lines.append(line)

    return line

  def clear_lines(self):
    for line in self._lines:
      line.remove()

    self._lines = []

  def add_lines(self, xs_arr: np.ndarray, ys_arr: np.ndarray):
    for xs, ys in zip(xs_arr, ys_arr):
      self.make_line(xs, ys)

  def closest(self, xdata, ydata):
    # 클릭한 지점의 화면 좌표
    pt = self.ax.transData.transform([xdata, ydata])

    # Line2D.get_data() -> [[x1, x2], [y1, y2]]
    # points -> [[x1, y1], [x2, y2], [x3, y3], ...]
    points = np.vstack([np.array(l.get_data()).T for l in self._lines])
    points = self.ax.transData.transform(points)  # 화면 좌표

    dist_sq = np.sum(np.square(points - pt), axis=1)
    argmin = np.argmin(dist_sq)
    line_index, point_index = divmod(argmin, 2)

    return line_index, point_index, np.sqrt(dist_sq[argmin])

  def _press(self, event):
    if not (self._current_line is None and self._lines):
      return

    lidx, pidx, dist = self.closest(event.xdata, event.ydata)

    if dist <= self.DIST_THOLD:
      self._active_index = (lidx, pidx)
    else:
      self._active_index = (-1, -1)

    self.update()

  def _onmove(self, event):
    if self._current_line is not None or self._active_index[0] < 0:
      return

    line = self._lines[self._active_index[0]]
    xs, ys = line.get_data()
    xs[self._active_index[1]] = event.xdata
    ys[self._active_index[1]] = event.ydata
    line.set_xdata(xs)
    line.set_ydata(ys)

    self.update()

  def _new_point(self, event):
    new_line = self._current_line is None

    if new_line:
      self._current_line = self.make_line()

    xs, ys = self._current_line.get_data()
    if new_line:
      xs[0] = event.xdata
      ys[0] = event.ydata
    else:
      xs.append(event.xdata)
      ys.append(event.ydata)

    self._current_line.set_xdata(xs)
    self._current_line.set_ydata(ys)

    if not new_line:
      self._current_line = None

  def _release(self, event):
    if self._active_index[0] < 0:
      self._new_point(event)
    else:
      self._active_index = (-1, -1)

    self.update()

  def extend_lines(self):
    xlim = self.ax.get_xlim()
    ylim = self.ax.get_ylim()

    for line in self._lines:
      xs, ys = line.get_data()
      if len(xs) == 1:
        continue

      n = [-ys[0] + ys[1], xs[0] - xs[1]]
      pt1, pt2 = extend_lines(n=n, p=[xs[0], ys[0]], xlim=xlim, ylim=ylim)
      line.set_xdata([pt1[0], pt2[0]])
      line.set_ydata([pt1[1], pt2[1]])


@dataclass
class EdgeletsOption:
  max_count: int = 10
  distance_threshold: float = 10.0  # pixel
  angle_threshold: float = 5  # deg


class Images:

  def __init__(self, fm: ThermalPanoramaFileManager) -> None:
    self._fm = fm

    self._ir: Any = None
    self._edges: Any = None

    self._edgelet_option = EdgeletsOption()
    self._canny_option = edge.CannyOptions(sigma=2.0)
    self._hough_option = edge.HoughOptions(threshold=10,
                                           line_gap=10,
                                           theta=self._theta())

  def _theta(self):
    # 수평선과 각도 오차가 허용치 이내인 edgelet만 추출하기 위한 HoughOption theta
    thold = self.edgelet_option.angle_threshold
    return np.deg2rad(np.linspace(-thold, thold, int(thold * 2 + 1)) + 90.0)

  def read(self, sp: SP):
    image = IIO.read(self._fm.panorama_path(DIR.ANLY, sp))
    if sp is SP.MASK:
      image = image.astype(bool)
    return image

  @property
  def ir(self) -> np.ndarray:
    if self._ir is None:
      self._ir = self.read(SP.IR)
    return self._ir

  @property
  def canny_option(self):
    return self._canny_option

  @canny_option.setter
  def canny_option(self, value: Union[dict, edge.CannyOptions]):
    if not isinstance(value, CannyOptions):
      value = CannyOptions(**value)

    self._canny_option = value
    self._edges = None

  @property
  def hough_option(self):
    return self._hough_option

  @hough_option.setter
  def hough_option(self, value: Union[dict, edge.HoughOptions]):
    if not isinstance(value, HoughOptions):
      value = HoughOptions(**value)
    value.theta = self._theta()
    self._hough_option = value

  @property
  def edgelet_option(self):
    return self._edgelet_option

  @edgelet_option.setter
  def edgelet_option(self, value: Union[dict, EdgeletsOption]):
    if not isinstance(value, EdgeletsOption):
      value = EdgeletsOption(**value)
    self._edgelet_option = value
    self._hough_option.theta = self._theta()

  @property
  def edges(self) -> np.ndarray:
    if self._edges is None:
      self._edges = edge.image2edges(self.ir,
                                     mask=self.read(SP.MASK),
                                     canny_option=self.canny_option)
    return self._edges

  def edgelets(self) -> Edgelets:
    edgelets = edge.edge2edgelets(self.edges, self.hough_option)
    print(self.hough_option.theta)

    # 가까운 edgelet 중 길이가 더 긴 것 선택
    opt = self.edgelet_option
    edgelets = _suppress_edgelets(edgelets,
                                  distance_threshold=opt.distance_threshold,
                                  angle_threshold=opt.angle_threshold)

    if edgelets.count > opt.max_count:
      edgelets = edgelets[:opt.max_count]

    return edgelets


@dataclass
class PlotSetting:
  image: Literal['ir', 'edges', 'vis', 'seg'] = 'ir'
  extend_lines: bool = False


class OutputPlotController(PanoPlotController):

  def __init__(self, parent=None) -> None:
    super().__init__(parent=parent)

    self._images: Optional[Images] = None
    self._axes_image: Optional[AxesImage] = None

    self._lines: Optional[LinesSelector] = None
    self._setting = PlotSetting()

  def init(self, app: QtGui.QGuiApplication, canvas: FigureCanvas):
    super().init(app, canvas)
    self._lines = LinesSelector(ax=self.axes, update=self.draw)

  def _set_style(self):
    self.axes.set_axis_off()

  @property
  def fm(self) -> ThermalPanoramaFileManager:
    return super().fm

  @fm.setter
  def fm(self, value):
    self._fm = value
    self._images = Images(value)

  @property
  def images(self) -> Images:
    if self._images is None:
      raise WorkingDirNotSet
    return self._images

  @property
  def lines(self):
    return self._lines

  @property
  def setting(self):
    return self._setting

  def plot(self):
    if self._axes_image is not None:
      self._axes_image.remove()

    it = self.setting.image
    cmap = 'inferno' if it == 'ir' else 'gist_gray'

    if it == 'ir':
      image = self.images.ir
    elif it == 'edges':
      image = np.logical_not(self.images.edges)
    elif it == 'vis':
      image = self.images.read(SP.VIS)
    elif it == 'seg':
      image = self.images.read(SP.SEG)
    else:
      raise ValueError(it)

    self._axes_image = self.axes.imshow(image, cmap=cmap)
    self.draw()

  def estimate_edgelets(self):
    self._lines.clear_lines()

    edgelets = self.images.edgelets()
    edgelets.sort()

    half = edgelets.strengths.reshape([-1, 1]) / 2.0
    pt1 = edgelets.locations - edgelets.directions * half  # [[x1, y1], ...]
    pt2 = edgelets.locations + edgelets.directions * half  # [[x2, y2], ...]
    xs = np.hstack([pt1[:, [0]], pt2[:, [0]]])  # [[x1, x2], ...]
    ys = np.hstack([pt1[:, [1]], pt2[:, [1]]])  # [[y1, y2], ...]

    self._lines.add_lines(xs, ys)
    self.draw()
