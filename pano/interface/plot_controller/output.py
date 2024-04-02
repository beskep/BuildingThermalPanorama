from collections.abc import Callable, Iterable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np
from matplotlib.backend_bases import MouseButton
from matplotlib.lines import Line2D
from matplotlib.widgets import _SelectorWidget  # noqa: PLC2701
from more_itertools import sliding_window
from numpy.typing import NDArray
from skimage import draw
from skimage.color import label2rgb

import pano.interface.common.pano_files as pf
from pano.interface.common.cmap import save_colormap
from pano.interface.common.pano_files import DIR, SP
from pano.interface.mbq import FigureCanvas
from pano.misc import edgelet as edge
from pano.misc.edgelet import Edgelets
from pano.misc.imageio import ImageIO
from pano.misc.tools import SegMask, normalize_image

from .plot_controller import PanoPlotController, QtGui

if TYPE_CHECKING:
  from matplotlib.image import AxesImage


def _suppress_edgelets(
  edgelets: Edgelets,
  distance_threshold=10,
  angle_threshold=10,
  repeat: int | None = None,
):
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


def _edgelets_between_edgelets(edgelets: Edgelets, weight=0.5):
  # 영상 수직 방향 좌표 정렬
  e = edgelets[np.argsort(edgelets.locations[:, 1])]
  e.normalize()

  # 수직 방향 인접한 edgelet의 위치 평균. weight가 낮을수록 아래 edgelet과 가까움
  locations = np.average(
    [e.locations[:-1, :], e.locations[1:, :]], axis=0, weights=(weight, 1 - weight)
  )

  # 왼쪽 방향 벡터에 -1 곱하기
  is_right = e.directions[:, 0] > 0
  if not np.all(is_right):
    e.directions *= np.where(is_right, 1, -1).reshape([-1, 1])

  # 수직 방향 인접한 edgelet의 방향 평균
  directions = np.average([e.directions[:-1, :], e.directions[1:, :]], axis=0)

  # strength (길이)는 1로 고정
  strengths = np.ones(edgelets.count - 1)

  return Edgelets(locations=locations, directions=directions, strengths=strengths)


def extend_lines(n: Any, p: Any, xlim: Any, ylim: Any) -> tuple[tuple, tuple]:
  """
  선 `dot(n, (X-p))=0`을 직사각형의 경계까지 확장하는 점 두개 반환.

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

  with np.errstate(divide='ignore'):
    points = [
      (x0, py - np.divide(nx * (x0 - px), ny)),  # x=x0 선과 만나는 지점
      (x1, py - np.divide(nx * (x1 - px), ny)),  # x=x1 선과 만나는 지점
      (px - np.divide(ny * (y0 - py), nx), y0),  # y=y0 선과 만나는 지점
      (px - np.divide(ny * (y1 - py), nx), y1),  # y=y1 선과 만나는 지점
    ]

  points = sorted(points, key=lambda x: x[0])

  return points[1], points[2]  # x좌표가 중간인 두 점 반환


def _segment_mask(storey_mask: np.ndarray, num: int):
  points = np.linspace(0, storey_mask.shape[1], num=(num + 1), endpoint=True)

  for idx1, idx2 in sliding_window(np.round(points), 2):
    mask = storey_mask.copy()
    mask[:, : int(idx1)] = False
    mask[:, int(idx2) :] = False

    yield mask


class SegmentsSummary:
  def __init__(
    self,
    functions: Iterable[Callable] | None = None,
    names: Iterable[str] | None = None,
  ) -> None:
    if functions is None:
      functions = (np.nanmean, np.nanmedian, np.nanmin, np.nanmax)
    if names is None:
      names = ('avg', 'median', 'min', 'max')

    self._fns = tuple(functions)
    self._names = tuple(names)

    if len(self._fns) != len(self._names):
      msg = '요약 함수와 이름의 개수가 불일치함'
      raise ValueError(msg)

    self._len = len(self._fns)

  @classmethod
  def mask(cls, shape: tuple[int, int], coords: NDArray):
    mask = np.full(shape=shape, fill_value=np.nan, dtype=np.uint8)

    for idx, lines in enumerate(sliding_window(coords, 2)):
      m = cls._mask(lines=lines, image_shape=shape)
      mask[m] = idx

    return mask

  @staticmethod
  def _mask(lines: tuple[NDArray, NDArray], image_shape: tuple[int, int]):
    # XXX 오류 발생 시
    # 1. ConvexHull로 선이 교차하는 문제 해결
    # 2. 대각선 (영상 수직/수평 경계를 모두 지나는 선) 때문에 발생하는 공백 구역 제거
    l1, l2 = lines
    polygon = np.array([l1[0], l2[0], l2[1], l1[1]])  # [[x, y], ...]
    polygon = np.flip(polygon, axis=1)  # [[y, x], ...]

    # 한 층에 해당하는 영역 mask
    return draw.polygon2mask(image_shape=image_shape, polygon=polygon)

  def __call__(
    self,
    arr: NDArray,
    coords: NDArray,
    num: int,
  ) -> tuple[dict[str, NDArray], NDArray]:
    """
    층별로 `num`개의 조각에 대해 `__init__`에서 지정한 stat 계산

    Parameters
    ----------
    arr : NDArray
        대상 array
    coords : NDArray
        각 층 구분선 좌표. `LinesSelector.coordinates()`로 계산.

        **수직방향으로 정렬되어 있어야 함.**
    num : int
        분할 개수

    Returns
    -------
    stats_dict : dict[str, NDArray]
        {name: stat}
    label : NDArray
        label image
    """
    assert np.all(coords[:, 0, 0] < coords[:, 1, 0])  # x1 < x2

    stats = []
    label = np.zeros_like(arr, dtype=int)  # segments 구분 시각화
    for idx, lines in enumerate(sliding_window(coords, 2)):
      mask = self._mask(lines=lines, image_shape=(arr.shape[0], arr.shape[1]))

      # 수직 선으로 `num`개 분할한 조각의 stat 계산
      ss = []
      li = idx % 2
      for m in _segment_mask(storey_mask=mask, num=num):
        marr = arr[m]

        if np.all(np.isnan(marr)):
          ss.append([np.nan] * len(self._fns))
        else:
          ss.append([fn(marr) for fn in self._fns])

        label[m] = li
        li = 0 if li else 1

      stats.append(ss)

    stats_arr = np.array(stats)
    stats_dict = {n: stats_arr[:, :, i] for i, n in enumerate(self._names)}

    return stats_dict, label


def _save_segments(
  subdir: Path,
  fname: str,
  arr: np.ndarray,
  coords: np.ndarray,
  num: int,
  label_thold=50,
):
  seg_summ = SegmentsSummary()
  stats, label = seg_summ(arr=arr, coords=coords, num=num)

  for k, v in stats.items():
    ImageIO.save(path=subdir.joinpath(f'{fname}-{k.title()}.csv'), array=v)

  if num <= label_thold:
    arr[np.isnan(arr)] = np.nanmin(arr)
    ImageIO.save(
      path=subdir.joinpath(f'{fname}-Label.png'),
      array=label2rgb(label, image=normalize_image(arr)),
    )


def _do_nothing(*_args, **_kwargs):
  return None


class LinesSelector(_SelectorWidget):
  DIST_THOLD = 10
  PROPS: ClassVar[dict] = {
    'alpha': 0.6,
    'animated': False,
    'color': 'k',
    'label': 'Floor Edgelet',
    'linestyle': '-',
    'marker': 'D',
  }
  PROPS_FIXED: ClassVar[dict] = PROPS | {
    'alpha': 0.8,
    'color': 'steelblue',
    'label': 'Window Edgelet',
  }

  def __init__(self, ax, *, useblit=False, update=None) -> None:
    super().__init__(ax, onselect=_do_nothing, useblit=useblit)

    self._lines: list[Line2D] = []
    self._fixed_lines: list[Line2D] = []  # 마우스로 수정 불가능한 선

    self._extend = False
    self._current_line: Line2D | None = None  # 마우스 클릭으로 생성 중인 line
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

  def make_line(self, xs=None, ys=None, *, editable=True):
    xs = xs if xs is not None else [np.nan]
    ys = ys if ys is not None else [np.nan]
    props = self.PROPS if editable else self.PROPS_FIXED

    line = Line2D(xs, ys, **props)
    self.ax.add_line(line)

    if editable:
      self._lines.append(line)
    else:
      self._fixed_lines.append(line)

    return line

  def clear_lines(self):
    for line in self._lines + self._fixed_lines:
      line.remove()

    self._lines = []
    self._fixed_lines = []

  def add_lines(self, xs_arr: np.ndarray, ys_arr: np.ndarray, *, editable=True):
    for xs, ys in zip(xs_arr, ys_arr, strict=True):
      self.make_line(xs, ys, editable=editable)

  def add_edgelets(self, edgelets: Edgelets, *, editable=True):
    half = edgelets.strengths.reshape([-1, 1]) / 2.0
    pt1 = edgelets.locations - edgelets.directions * half  # [[x1, y1], ...]
    pt2 = edgelets.locations + edgelets.directions * half  # [[x2, y2], ...]

    xs = np.hstack([pt1[:, [0]], pt2[:, [0]]])  # [[x1, x2], ...]
    ys = np.hstack([pt1[:, [1]], pt2[:, [1]]])  # [[y1, y2], ...]

    self.add_lines(xs, ys, editable=editable)

  def closest(self, xdata, ydata):
    # 클릭한 지점의 화면 좌표
    pt = self.ax.transData.transform([xdata, ydata])

    # Line2D.get_data() -> [[x1, x2], [y1, y2]]
    # points -> [[x1, y1], [x2, y2], [x3, y3], ...]
    points = np.vstack([np.array(line.get_data()).T for line in self._lines])
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

  def _onmove(self, event):
    if (
      event.button != 1
      or self._current_line is not None
      or self._active_index[0] < 0
      or not self._lines
    ):
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

    assert self._current_line is not None
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
    active = self._active_index[0]
    if event.button == 1:
      # left click -> 새 line 생성 혹은 active_index 초기화
      if active < 0:
        self._new_point(event)
      else:
        self._active_index = (-1, -1)
    elif event.button == MouseButton.RIGHT and active >= 0:
      # right click -> 가장 가까운 선 삭제
      self._lines[active].remove()
      self._lines.pop(active)

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

  def remove_window_line(self, seg: np.ndarray, threshold: float):
    def clip(c: float, axis=0):
      # 영상 내 shape 범위로 clip하고 int 형식으로 변환
      return int(np.clip(c, a_min=0, a_max=(seg.shape[axis] - 1)))

    lines = []
    for line in self._lines:
      xs, ys = line.get_data()

      # edgelet이 지나는 좌표
      lxs, lys = draw.line(
        r0=clip(ys[0], 1),
        c0=clip(xs[0], 0),
        r1=clip(ys[1], 1),
        c1=clip(xs[1], 0),
      )

      pixels = seg[lxs, lys]
      wall = np.sum(pixels == SegMask.WALL)
      window = np.sum(pixels == SegMask.WINDOW)

      if window >= threshold * (wall + window):
        line.remove()
      else:
        lines.append(line)

    self._lines = lines

  def coordinates(self, xmax: int, ymax: int):
    if not self._lines:
      msg = '추정한 층 구분선이 없습니다.'
      raise ValueError(msg)

    # Line2D.get_data():
    #   [[x1, x2],
    #    [y1, y2]]
    # coords:
    #   [[[x1, y1],
    #     [x2, y2]],
    #    ...       ]
    coords = [np.array(line.get_data()).T for line in self._lines]

    # 세로 방향 정렬
    argsort = np.argsort([np.average(x[:, 1]) for x in coords])
    coords = [coords[x] for x in argsort]

    coords.insert(0, np.array([[0, 0], [xmax, 0]]))  # 영상 위 경계 (y=0)
    coords.append(np.array([[0, ymax], [xmax, ymax]]))  # 영상 아래 경계 (y=ymax)

    return np.array(coords)


@dataclass
class EdgeletsOption:
  segmentation: bool = True  # True면 열화상이 아닌 segmentation mask로부터 추정
  window_threshold: float = 0.05
  slab_position: float = 0.5

  max_count: int = 10
  distance_threshold: float = 10.0  # pixel
  angle_threshold: float = 5  # deg


class Images:
  def __init__(self, fm: pf.ThermalPanoramaFileManager) -> None:
    self._fm = fm

    self._ir: Any = None
    self._edges: Any = None
    self._seg: Any = None

    self._edgelet_option = EdgeletsOption()
    self._canny_option = edge.CannyOptions()
    self._hough_option = edge.HoughOptions(theta=self._theta())

  def _theta(self):
    # 수평선과 각도 오차가 허용치 이내인 edgelet만 추출하기 위한 HoughOption theta
    thold = self.edgelet_option.angle_threshold
    return np.deg2rad(np.linspace(-thold, thold, int(thold * 2 + 1)) + 90.0)

  def read(self, sp: SP):
    path = self._fm.panorama_path(DIR.ANLY, sp)
    if sp is not SP.IR:
      path = path.parent / f'Image-{path.name}'

    image = ImageIO.read(path)
    if sp is SP.MASK:
      image = image.astype(bool)

    return image

  @property
  def ir(self) -> np.ndarray:
    if self._ir is None:
      self._ir = self.read(SP.IR)
    return self._ir

  @property
  def seg(self) -> np.ndarray:
    if self._seg is None:
      self._seg = SegMask.vis2index(self.read(SP.SEG)).astype(float)
    return self._seg

  @property
  def canny_option(self):
    return self._canny_option

  @canny_option.setter
  def canny_option(self, value: dict | edge.CannyOptions):
    if not isinstance(value, edge.CannyOptions):
      value = edge.CannyOptions(**value)

    self._canny_option = value
    self._edges = None

  @property
  def hough_option(self):
    return self._hough_option

  @hough_option.setter
  def hough_option(self, value: dict | edge.HoughOptions):
    if not isinstance(value, edge.HoughOptions):
      value = edge.HoughOptions(**value)
    value.theta = self._theta()
    self._hough_option = value

  @property
  def edgelet_option(self):
    return self._edgelet_option

  @edgelet_option.setter
  def edgelet_option(self, value: dict | EdgeletsOption):
    if not isinstance(value, EdgeletsOption):
      value = EdgeletsOption(**value)
    self._edgelet_option = value
    self._hough_option.theta = self._theta()

  @property
  def edges(self) -> np.ndarray:
    if self._edges is None:
      image = self.seg if self.edgelet_option.segmentation else self.ir
      self._edges = edge.image2edges(
        image=image.copy(),
        mask=self.read(SP.MASK),
        canny_option=self.canny_option,
      )
    return self._edges

  def edgelets(self) -> Edgelets:
    edgelets = edge.edge2edgelets(self.edges, self.hough_option)
    opt = self.edgelet_option

    # 가까운 edgelet 중 길이가 더 긴 것 선택
    edgelets = _suppress_edgelets(
      edgelets,
      distance_threshold=opt.distance_threshold,
      angle_threshold=opt.angle_threshold,
    )

    # 최대 n개 선택
    if edgelets.count > opt.max_count:
      edgelets = edgelets[: opt.max_count]

    return edgelets


@dataclass
class PlotSetting:
  image: Literal['ir', 'edges', 'vis', 'seg'] = 'ir'
  extend_lines: bool = False


class OutputPlotController(PanoPlotController):
  def __init__(self, parent=None) -> None:
    super().__init__(parent=parent)

    self._images: Images | None = None
    self._axes_image: AxesImage | None = None
    self._legend = None

    self._lines: LinesSelector | None = None
    self._setting = PlotSetting()

  def init(self, app: QtGui.QGuiApplication, canvas: FigureCanvas):
    super().init(app, canvas)
    self._lines = LinesSelector(ax=self.axes, update=self.draw)

  def _set_style(self):
    self.axes.set_axis_off()

  @property
  def fm(self) -> pf.ThermalPanoramaFileManager:
    return super().fm

  @fm.setter
  def fm(self, value):
    self._fm = value
    self._images = Images(value)

  @property
  def images(self) -> Images:
    if self._images is None:
      raise pf.WorkingDirNotSetError
    return self._images

  @property
  def lines(self):
    return self._lines

  @property
  def setting(self):
    return self._setting

  def configure(self, config):
    self.images.canny_option = config['canny']
    self.images.hough_option = config['hough']
    self.images.edgelet_option = config['edgelet']

  def plot(self):
    if self._axes_image is not None:
      with suppress(ValueError):
        self._axes_image.remove()

    seg = self.images.edgelet_option.segmentation
    if seg and self._legend is None:
      l1 = Line2D([], [], **self.lines.PROPS)
      l2 = Line2D([], [], **self.lines.PROPS_FIXED)
      self._legend = self.fig.legend(handles=[l1, l2])
    if not seg and self._legend is not None:
      self._legend.remove()
      self._legend = None

    it = self.setting.image
    cmap = 'inferno' if it == 'ir' else 'gist_gray'

    if it == 'ir':
      image = self.images.ir
    elif it == 'edges':
      image = np.logical_not(self.images.edges)
    elif it == 'vis':
      image = self.images.read(SP.VIS)
    elif it == 'seg':
      image = self.images.seg
    else:
      raise ValueError(it)

    self._axes_image = self.axes.imshow(image, cmap=cmap)
    self.draw()

  def estimate_edgelets(self):
    opt = self.images.edgelet_option
    edgelets = self.images.edgelets()

    self.lines.clear_lines()
    self.lines.add_edgelets(edgelets=edgelets, editable=(not opt.segmentation))

    if opt.segmentation:
      edgelets2 = _edgelets_between_edgelets(edgelets, weight=opt.slab_position)
      self.lines.add_edgelets(edgelets=edgelets2, editable=True)
      self.lines.extend_lines()

      self.lines.remove_window_line(seg=self.images.seg, threshold=opt.window_threshold)

    self.draw()

  def save(self, segments=20):
    subdir = self.fm.subdir(DIR.OUT, mkdir=True)

    # 열화상 (온도 행렬)
    ir = self.images.ir
    ImageIO.save(subdir.joinpath('Temperature.csv'), np.round(ir, 2))

    # 층 구분 이미지
    self.fig.savefig(subdir.joinpath('Edgelets.jpg'), dpi=150)

    self.lines.extend_lines()
    coords = self.lines.coordinates(xmax=ir.shape[1], ymax=ir.shape[0])

    # mask
    ImageIO.save(
      path=subdir / 'Floor.npy',
      array=SegmentsSummary.mask(shape=(ir.shape[0], ir.shape[1]), coords=coords),
    )

    # 벽 온도
    wall = ir.copy()
    wall[self.images.seg != SegMask.WALL] = np.nan
    _save_segments(
      subdir=subdir,
      fname='TemperatureWall',
      arr=wall,
      coords=coords,
      num=segments,
    )

    # 벽+창문 온도
    building = ir.copy()
    mask = np.isin(self.images.seg, [SegMask.WALL, SegMask.WINDOW])
    building[~mask] = np.nan
    _save_segments(
      subdir=subdir,
      fname='TemperatureBuilding',
      arr=building,
      coords=coords,
      num=segments,
    )

    # 벽+창문 factor
    factor = ImageIO.read(self.fm.panorama_path(DIR.ANLY, SP.TF))
    _save_segments(
      subdir=subdir, fname=SP.TF.value, arr=factor, coords=coords, num=segments
    )

    # 컬러맵
    save_colormap(path=subdir.joinpath('colormap.csv'))

    self.draw()
