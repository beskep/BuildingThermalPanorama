from collections import defaultdict
from pathlib import Path
from typing import Optional

from loguru import logger
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseEvent
from matplotlib.cm import get_cmap
from matplotlib.colorbar import make_axes_gridspec
from matplotlib.figure import Figure
from matplotlib.patches import Patch
import numpy as np
import seaborn as sns
from skimage import transform
from skimage.exposure import equalize_hist

from pano.distortion.projection import ImageProjection
from pano.misc.cmap import apply_colormap
from pano.misc.imageio import ImageIO
from pano.misc.tools import limit_image_size
from pano.misc.tools import prep_compare_images
from pano.misc.tools import SegMask
from pano.misc.tools import uint8_image

from .common.pano_files import DIR
from .common.pano_files import FN
from .common.pano_files import SP
from .common.pano_files import ThermalPanoramaFileManager
from .mbq import FigureCanvas
from .mbq import NavigationToolbar2QtQuick as NavToolbar
from .mbq import QtCore
from .mbq import QtGui
from .pano_project import ThermalPanorama

_DIRS = ('bottom', 'top', 'left', 'right')
_TICK_PARAMS = {key: False for key in _DIRS + tuple('label' + x for x in _DIRS)}


class WorkingDirNotSet(FileNotFoundError):

  def __str__(self) -> str:
    return self.args[0] if self.args else '대상 경로가 지정되지 않았습니다.'


class CropToolbar(NavToolbar):

  def none(self):
    """마우스 클릭에 반응하지 않는 모드"""
    self.mode = ''

  def crop(self):
    self.zoom()

  def save_figure(self, *args):
    pass


class PlotController(QtCore.QObject):

  def __init__(self, parent=None) -> None:
    super().__init__(parent=parent)

    self._app: Optional[QtGui.QGuiApplication] = None
    self._canvas: Optional[FigureCanvas] = None
    self._fig: Optional[Figure] = None
    self._axes: Optional[Axes] = None

  @property
  def app(self) -> QtGui.QGuiApplication:
    if self._app is None:
      raise ValueError('app not set')

    return self._app

  @property
  def canvas(self) -> FigureCanvas:
    if self._canvas is None:
      raise ValueError('canvas not set')

    return self._canvas

  @property
  def fig(self) -> Figure:
    if self._fig is None:
      raise ValueError('fig not set')

    return self._fig

  @property
  def axes(self) -> Axes:
    if self._axes is None:
      raise ValueError('axes not set')

    return self._axes

  def init(self, app: QtGui.QGuiApplication, canvas: FigureCanvas):
    self._app = app
    self._canvas = canvas

    self._fig = canvas.figure
    self._axes = self._fig.add_subplot(111)

    self.draw()

  def draw(self):
    self.canvas.draw()
    self.app.processEvents()


class _PanoPlotController(PlotController):

  def __init__(self, parent=None) -> None:
    super().__init__(parent=parent)
    self._fm: Optional[ThermalPanoramaFileManager] = None

  @property
  def fm(self) -> ThermalPanoramaFileManager:
    if self._fm is None:
      raise WorkingDirNotSet
    return self._fm

  @fm.setter
  def fm(self, value: ThermalPanoramaFileManager):
    self._fm = value

  def _set_style(self):
    pass

  def draw(self):
    self._set_style()
    super().draw()


class RegistrationPlotController(_PanoPlotController):
  _REQUIRED = 4
  _TITLES = ('열화상', '실화상', '비교 (Checkerboard)', '비교 (Difference)')
  _GRID_COUNTS = (8, 8)

  def __init__(self, parent=None) -> None:
    super().__init__(parent=parent)

    self._toolbar: Optional[NavToolbar] = None

    self._pnts = defaultdict(list)  # 선택된 점들의 mpl 오브젝트
    self._pnts_coord = defaultdict(list)  # 선택된 점들의 좌표
    self._zoom = False
    self._grid = False

    self._images: Optional[tuple] = None
    self._matrices: Optional[dict] = None

    self._file: Optional[Path] = None
    self._matrix: Optional[np.ndarray] = None  # 선택된 영상의 수동 정합 matrix
    self._registered_image: Optional[np.ndarray] = None

  @property
  def matrices(self) -> dict:
    if self._matrices is None:
      path = self.fm.rgst_matrix_path()
      if path.exists():
        npz = np.load(path)
        self._matrices = {f: npz[f] for f in npz.files}
      else:
        self._matrices = dict()

    return self._matrices

  def reset_matrices(self):
    self._matrices = None

  def update_matrices(self, stem: str, matrix: np.ndarray):
    self.matrices[stem] = matrix
    np.savez(self.fm.rgst_matrix_path(), **self.matrices)

  @property
  def axes(self) -> np.ndarray:
    return super().axes

  def init(self, app: QtGui.QGuiApplication, canvas: FigureCanvas):
    self._app = app
    self._canvas = canvas
    self._toolbar = NavToolbar(canvas=canvas)

    self._fig = canvas.figure
    self._axes = self.fig.subplots(2, 2)
    self._set_style()
    self._fig.tight_layout(pad=2)

    self.canvas.mpl_connect('button_press_event', self._on_click)
    self.draw()

  def home(self):
    if self._images is not None:
      self._toolbar.home()

  def zoom(self, value: bool):
    self._zoom = value
    if not self._zoom ^ value:
      assert self._toolbar is not None
      self._toolbar.zoom()

  def _set_style(self):
    for ax, title in zip(self.axes.ravel(), self._TITLES):
      if ax.has_data():
        ax.set_title(title)
      ax.set_axis_off()

    ar = self.axes[0, 0].get_aspect()
    self.axes[0, 1].set_aspect(ar)

    if self._grid:
      for ax in self.axes[0]:
        ax.set_axis_on()
        ax.tick_params(axis='both', which='both', **_TICK_PARAMS)

  def set_grid(self, grid: bool):
    if not self._grid ^ grid:
      return

    self._grid = grid
    self.draw()

  def _set_ticks(self, image: np.ndarray):
    ticks = tuple(
        np.linspace(
            0,
            image.shape[x],
            num=self._GRID_COUNTS[x],
            endpoint=True,
        ) for x in range(2))

    self.axes[0, 0].set_xticks(ticks[1])
    self.axes[0, 0].set_yticks(ticks[0])
    self.axes[0, 1].set_xticks(ticks[1])
    self.axes[0, 1].set_yticks(ticks[0])

  def reset(self):
    for ax in self.axes.ravel():
      ax.clear()

    self._pnts.clear()
    self._pnts_coord.clear()
    self._registered_image = None
    self._matrix = None
    self.draw()

  def set_images(self, fixed_image: np.ndarray, moving_image: np.ndarray):
    self.reset()

    if fixed_image.shape[:2] != moving_image.shape[:2]:
      moving_image = transform.resize(moving_image,
                                      output_shape=fixed_image.shape[:2],
                                      anti_aliasing=True)

    self._images = (fixed_image, moving_image)
    self.axes[0, 0].imshow(equalize_hist(fixed_image))
    self.axes[0, 1].imshow(moving_image)

    self._set_ticks(fixed_image)

  def plot(self, file: Path):
    self._file = file

    ir_path = self.fm.change_dir(DIR.IR, file)
    vis_path = self.fm.change_dir(DIR.VIS, file)
    ir = ImageIO.read(ir_path)
    vis = ImageIO.read(vis_path)

    self.set_images(ir, vis)

    matrix = self.matrices.get(file.stem, None)
    if matrix is not None:
      self._plot_registered(np.linalg.inv(matrix))

    self.draw()

  def _on_click(self, event: MouseEvent):
    logger.trace(event)
    if self._zoom or self._images is None:
      return

    ax: Axes = event.inaxes
    if ax is None:
      return

    axi = list(self.axes.ravel()).index(ax)
    if axi not in (0, 1):
      return

    if event.button == 1:
      self._add_point(axi, event=event)
    elif event.button == 3:
      self._remove_points(axi)
    else:
      return

    if self._registered_image is not None and self.all_points_selected():
      self._manual_register()

    self.draw()

  def _add_point(self, ax: int, event: MouseEvent):
    if len(self._pnts_coord[ax]) < self._REQUIRED:
      self._pnts_coord[ax].append((event.xdata, event.ydata))

      p = event.inaxes.scatter(event.xdata,
                               event.ydata,
                               s=50,
                               edgecolors='w',
                               linewidths=1)
      self._pnts[ax].append(p)

  def _remove_points(self, ax: int):
    self._pnts_coord.pop(ax, None)

    for p in self._pnts[ax]:
      p.remove()

    self._pnts.pop(ax, None)

  def all_points_selected(self):
    return all(len(self._pnts_coord[x]) == self._REQUIRED for x in range(2))

  def _plot_registered(self, matrix):
    registered = transform.warp(image=self._images[1],
                                inverse_map=matrix,
                                output_shape=self._images[0].shape[:2],
                                preserve_range=True)

    cb, diff = prep_compare_images(image1=self._images[0],
                                   image2=registered,
                                   norm=True,
                                   eq_hist=True,
                                   method=['checkerboard', 'diff'])

    self._axes[1, 0].imshow(cb)
    self._axes[1, 1].imshow(diff)

    self._registered_image = registered

  def _manual_register(self):
    src = np.array(self._pnts_coord[1])
    dst = np.array(self._pnts_coord[0])

    trsf = transform.ProjectiveTransform()
    trsf.estimate(src=src, dst=dst)

    self._matrix = trsf.params
    self._plot_registered(matrix=trsf.inverse)

  def save(self):
    if self._registered_image is None:
      logger.warning('저장할 정합 결과가 없습니다.')
      return

    if self._matrix is None:
      logger.warning('자동 정합 결과가 이미 저장되었습니다.')
      return

    assert self._file is not None
    path = self.fm.change_dir(DIR.RGST, self._file)
    ImageIO.save(path=path, array=uint8_image(self._registered_image))

    compare_path = path.with_name(f'{path.stem}{FN.RGST_MANUAL}{path.suffix}')
    self.fig.savefig(compare_path, dpi=300)

    self.update_matrices(path.stem, self._matrix)


class SegmentationPlotController(_PanoPlotController):
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
    for ax, title in zip(self.axes.ravel(), self._TITLES):
      if ax.has_data():
        ax.set_title(title)
      ax.set_axis_off()

  def plot(self, file: Path):
    vis_path = self.fm.change_dir(DIR.RGST, file)
    mask_path = self.fm.change_dir(DIR.SEG, file)

    if not (vis_path.exists() and mask_path.exists()):
      return

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
      self._legend = self.fig.legend(handles=patches,
                                     ncol=len(patches),
                                     loc='lower center',
                                     bbox_to_anchor=(0.5, 0.01))
      self.fig.tight_layout()

    self.draw()


class PanoramaPlotController(_PanoPlotController):
  _GRID_COUNTS = (7, 7)  # (height, width)

  def __init__(self, parent=None) -> None:
    super().__init__(parent=parent)

    self._cax: Optional[Axes] = None
    self._prj: Optional[ImageProjection] = None
    self._toolbar: Optional[CropToolbar] = None

    self._image: Optional[np.ndarray] = None
    self._dir = DIR.PANO
    self._angles: Optional[np.ndarray] = None
    self._va = np.deg2rad(42.0)

    self._cmap = get_cmap('inferno')
    self._limit = 0
    self._grid = True

  @property
  def cax(self) -> Axes:
    if self._cax is None:
      raise ValueError('Colorbar ax not set')

    return self._cax

  @property
  def subdir(self) -> str:
    return self._dir.name

  @property
  def viewing_angle(self):
    return self._va

  @viewing_angle.setter
  def viewing_angle(self, value):
    """
    Parameters
    ----------
    value : float
        Viewing angle [deg]
    """
    self._va = np.deg2rad(value)
    self._prj = None

  def init(self, app: QtGui.QGuiApplication, canvas: FigureCanvas):
    self._app = app
    self._canvas = canvas
    self._toolbar = CropToolbar(canvas=canvas)

    self._fig = canvas.figure
    self._axes = self._fig.add_subplot(111)
    self._cax = make_axes_gridspec(self._axes)[0]
    self._fig.tight_layout()

    self._axes.set_axis_off()
    self._cax.set_axis_off()

  def home(self):
    if self._image is not None:
      self._toolbar.home()

  def crop_mode(self, value: bool):
    assert self._toolbar is not None
    if value:
      self._toolbar.crop()
    else:
      self._toolbar.none()

  def crop_range(self) -> np.ndarray:
    """
    Returns
    -------
    np.ndarray
        [[xr0, yr0], [xr1, yr1]]
    """
    data_lim = self.axes.dataLim.get_points()
    width_height = data_lim[1] - data_lim[0]
    view_lim = self.axes.viewLim.get_points()

    return view_lim / width_height

  def _set_style(self):
    if self._grid:
      self.axes.set_axis_on()
      self.axes.tick_params(axis='both', which='both', **_TICK_PARAMS)
    else:
      self.axes.set_axis_off()

  def _load_panorama(self):
    pano_path = self.fm.panorama_path(DIR.COR, SP.IR)
    mask_path = self.fm.panorama_path(DIR.COR, SP.MASK)
    self._dir = DIR.COR

    if not pano_path.exists():
      pano_path = self.fm.panorama_path(DIR.PANO, SP.IR)
      mask_path = self.fm.panorama_path(DIR.PANO, SP.MASK)
      self._dir = DIR.PANO

      if not pano_path.exists():
        raise FileNotFoundError('파노라마가 생성되지 않았습니다.')

    if not mask_path.exists():
      raise FileNotFoundError(f'파노라마 마스크 파일이 존재하지 않습니다. ({mask_path})')

    pano = ImageIO.read(pano_path).astype(np.float32)
    mask = ImageIO.read(mask_path)
    pano[np.logical_not(mask)] = np.nan

    self._image = pano

  def resize(self, limit: int):
    if self._prj is not None and self._limit == limit:
      return

    assert self._image is not None
    self._limit = limit
    self._prj = ImageProjection(image=limit_image_size(image=self._image,
                                                       limit=limit),
                                viewing_angle=self.viewing_angle)

  def _set_ticks(self, image: np.ndarray):
    ticks = tuple(
        np.linspace(
            0,
            image.shape[x],
            num=self._GRID_COUNTS[x],
            endpoint=True,
        ) for x in range(2))

    self.axes.set_xticks(ticks[1])
    self.axes.set_yticks(ticks[0])

  def plot(self, force=False):
    if not (force or self._image is None):
      return

    self._load_panorama()

    self.axes.clear()
    self.cax.clear()

    im = self.axes.imshow(self._image, cmap=self._cmap)
    self.fig.colorbar(im, cax=self.cax, ax=self.axes)
    self.cax.get_yaxis().labelpad = 10
    self.cax.set_ylabel('Temperature [℃]', rotation=90)

    self._set_ticks(self._image)
    self.draw()

  def project(self, roll=0.0, pitch=0.0, yaw=0.0, limit=9999):
    if self._image is None:
      return

    self.resize(limit=limit)
    assert self._prj is not None

    self._angles = np.deg2rad([roll, pitch, yaw])
    image = self._prj.project(*self._angles)

    self.axes.clear()
    self.axes.imshow(image, cmap=self._cmap)

    self._set_ticks(image)
    self.draw()

  def set_grid(self, grid):
    if not self._grid ^ grid:
      return

    self._grid = grid
    self.draw()


def save_manual_correction(wd, subdir, viewing_angle, angles,
                           crop_range: Optional[np.ndarray]):
  tp = ThermalPanorama(wd, init_loglevel='TRACE')
  ir_pano = ImageIO.read(tp.fm.panorama_path(subdir, SP.IR))
  prj = ImageProjection(ir_pano, viewing_angle=viewing_angle)
  angles = np.deg2rad(angles)

  for sp in SP:
    if sp is SP.IR:
      image = ir_pano
    else:
      image = ImageIO.read(tp.fm.panorama_path(subdir, sp))

    corrected = prj.project(roll=angles[0],
                            pitch=angles[1],
                            yaw=angles[2],
                            cval=(None if sp is SP.IR else 0),
                            image=image)
    if sp is SP.MASK:
      corrected = uint8_image(corrected)

    if crop_range is not None:
      cr = np.multiply(crop_range, ir_pano.shape[::-1]).astype(int)
      xx = (np.min(cr[:, 0]), np.max(cr[:, 0]))
      yy = (np.min(cr[:, 1]), np.max(cr[:, 1]))
      corrected = corrected[yy[0]:yy[1], xx[0]:xx[1]]

    corrected = np.nan_to_num(corrected, nan=np.nanmin(corrected))

    path = tp.fm.panorama_path(DIR.COR, sp)
    path = path.parent.joinpath(f'Manual{path.name}')
    if not path.parent.exists():
      path.parent.mkdir()

    if sp is SP.IR:
      ImageIO.save_with_meta(path=path,
                             array=corrected,
                             exts=[FN.NPY, FN.LL],
                             dtype='uint16')
      # colormap 버전
      ImageIO.save(path=tp.fm.color_path(path),
                   array=apply_colormap(image=corrected, cmap=tp.cmap))

    else:
      ImageIO.save(path=path, array=corrected.astype(np.uint8))


class DistPlotController(_PanoPlotController):
  _CLASSES = ('Background', 'Wall', 'Window', 'etc.')

  def _set_style(self):
    if self.axes.has_data():
      self.axes.set_xlabel('Temperature [℃]')
      self.axes.tick_params(axis='both', which='both', top=False, right=False)
    else:
      self.axes.set_axis_off()

  def get_panorama_path(self):
    for subdir in (DIR.COR, DIR.PANO):
      path = self.fm.panorama_path(subdir, SP.IR)

      manual_path = path.parent.joinpath(f'Manual{path.name}')
      if manual_path.exists():
        return subdir, manual_path

      if path.exists():
        return subdir, path

    return None, None

  def read_images(self):
    subdir, ir_path = self.get_panorama_path()
    logger.debug('IR pano path: {}', ir_path)
    if ir_path is None:
      raise FileNotFoundError('생성된 파노라마가 없습니다.')

    mask_path = self.fm.panorama_path(subdir, SP.MASK)
    seg_path = self.fm.panorama_path(subdir, SP.SEG)
    if ir_path.stem.startswith('Manual'):
      mask_path = mask_path.parent.joinpath(f'Manual{mask_path.name}')
      seg_path = seg_path.parent.joinpath(f'Manual{seg_path.name}')

    ir = ImageIO.read(ir_path)
    mask = ImageIO.read(mask_path)
    ir[np.logical_not(mask)] = np.nan

    seg = ImageIO.read(seg_path)
    if seg.ndim == 3:
      seg = seg[:, :, 0]

    seg = SegMask.vis_to_index(seg)

    return ir, seg

  @staticmethod
  def remove_outliers(data: np.ndarray, k=1.5):
    data = data[~np.isnan(data)]

    q1 = np.quantile(data, q=0.25)
    q3 = np.quantile(data, q=0.75)
    iqr = q3 - q1

    lower = q1 - k * iqr
    upper = q3 + k * iqr

    return data[(lower < data) & (data < upper)]

  @staticmethod
  def bin_width(array: np.ndarray, min_width=0.5):
    r = np.nanmax(array) - np.nanmin(array)
    bin_width = r / np.histogram_bin_edges(array[~np.isnan(array)],
                                           bins='fd').size

    return max(min_width, bin_width)

  @staticmethod
  def _summary(arr: np.ndarray):
    return {
        'avg': np.average(arr),
        'std': np.std(arr),
        'q1': np.quantile(arr, 0.25),
        'median': np.median(arr),
        'q3': np.quantile(arr, 0.75),
    }

  def plot(self):
    self.axes.clear()

    ir, seg = self.read_images()
    data = {
        label: ir[seg == self._CLASSES.index(label)].ravel()
        for label in self._CLASSES[1:-1]
    }

    # TODO k 정하기
    data = {
        key: self.remove_outliers(value, k=1.5) for key, value in data.items()
    }

    sns.histplot(data=data,
                 stat='probability',
                 binwidth=self.bin_width(ir),
                 element='step',
                 ax=self.axes)

    self.draw()

    return {
        key: self._summary(value.astype(np.float64))
        for key, value in data.items()
    }
