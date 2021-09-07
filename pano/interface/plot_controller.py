from collections import defaultdict
from pathlib import Path
from typing import Optional

from loguru import logger
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseEvent
from matplotlib.cm import get_cmap
from matplotlib.figure import Figure
from matplotlib.patches import Patch
import numpy as np
from skimage import transform
from skimage.exposure import equalize_hist

from pano.misc.imageio import ImageIO
from pano.misc.tools import prep_compare_images
from pano.misc.tools import SegMask
from pano.misc.tools import uint8_image

from .common.pano_files import DIR
from .common.pano_files import FN
from .common.pano_files import ThermalPanoramaFileManager
from .mbq import FigureCanvas
from .mbq import QtCore
from .mbq import QtGui


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


class RegistrationPlotController(PlotController):
  _REQUIRED = 4
  _TITLES = ('열화상', '실화상', '비교 (Checkerboard)', '비교 (Difference)')

  def __init__(self, parent=None) -> None:
    super().__init__(parent=parent)

    self._pnts = defaultdict(list)  # 선택된 점들의 mpl 오브젝트
    self._pnts_coord = defaultdict(list)  # 선택된 점들의 좌표
    self._images: Optional[tuple] = None
    self._fm: Optional[ThermalPanoramaFileManager] = None
    self._matrices: Optional[dict] = None

    self._file: Optional[Path] = None
    self._matrix: Optional[np.ndarray] = None  # 선택된 영상의 수동 정합 matrix
    self._registered_image: Optional[np.ndarray] = None

  @property
  def fm(self) -> ThermalPanoramaFileManager:
    assert self._fm is not None
    return self._fm

  @fm.setter
  def fm(self, value: ThermalPanoramaFileManager):
    self._fm = value

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

    self._fig = canvas.figure
    self._axes = self.fig.subplots(2, 2)
    self._set_style()
    self._fig.tight_layout(pad=2)

    self.canvas.mpl_connect('button_press_event', self._on_click)
    self.draw()

  def _set_style(self):
    for ax, title in zip(self.axes.ravel(), self._TITLES):
      if ax.has_data():
        ax.set_title(title)
      ax.set_axis_off()

    ar = self.axes[0, 0].get_aspect()
    self.axes[0, 1].set_aspect(ar)

  def draw(self):
    self._set_style()
    return super().draw()

  def reset(self):
    for ax in self.axes.ravel():
      ax.clear()

    self._pnts.clear()
    self._pnts_coord.clear()
    self._registered_image = None
    self._matrix = None

  def set_images(self, fixed_image: np.ndarray, moving_image: np.ndarray):
    self.reset()

    if fixed_image.shape[:2] != moving_image.shape[:2]:
      moving_image = transform.resize(moving_image,
                                      output_shape=fixed_image.shape[:2],
                                      anti_aliasing=True)

    self._images = (fixed_image, moving_image)
    # self.axes[0, 0].imshow(fixed_image)
    self.axes[0, 0].imshow(equalize_hist(fixed_image))
    self.axes[0, 1].imshow(moving_image)

  def plot(self, file: Path):
    self._file = file

    ir_path = self.fm.change_dir(DIR.IR, file)
    vis_path = self.fm.change_dir(DIR.VIS, file)
    ir = ImageIO.read(ir_path)
    vis = ImageIO.read(vis_path)

    self.set_images(ir, vis)

    matrix = self.matrices[file.stem]
    if matrix is not None:
      self._plot_registered(np.linalg.inv(matrix))

    self.draw()

  def _on_click(self, event: MouseEvent):
    logger.trace(event)
    if self._images is None:
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

      p = event.inaxes.scatter(event.xdata, event.ydata, edgecolors='w', s=50)
      self._pnts[ax].append(p)

  def _remove_points(self, ax: int):
    self._pnts_coord.pop(ax)

    for p in self._pnts[ax]:
      p.remove()

    self._pnts.pop(ax)

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


class SegmentationPlotController(PlotController):
  _TITLES = ('실화상', '부위 인식 결과')
  _CLASSES = ('Background', 'Wall', 'Window', 'etc.')

  def __init__(self, parent=None) -> None:
    super().__init__(parent=parent)
    self._fm: Optional[ThermalPanoramaFileManager] = None
    self._cmap = get_cmap('Dark2')

  @property
  def axes(self) -> np.ndarray:
    return super().axes

  def init(self, app: QtGui.QGuiApplication, canvas: FigureCanvas):
    self._app = app
    self._canvas = canvas

    self._fig = canvas.figure
    self._axes = self._fig.subplots(1, 2)

    patches = [
        Patch(color=self._cmap(i), label=label)
        for i, label in enumerate(self._CLASSES)
    ]
    self._fig.legend(handles=patches, loc='right')

    self.draw()

  def _set_style(self):
    for ax, title in zip(self.axes.ravel(), self._TITLES):
      if ax.has_data():
        ax.set_title(title)
      ax.set_axis_off()

  def draw(self):
    self._set_style()
    return super().draw()

  @property
  def fm(self) -> ThermalPanoramaFileManager:
    assert self._fm is not None
    return self._fm

  @fm.setter
  def fm(self, value: ThermalPanoramaFileManager):
    self._fm = value

  def plot(self, file: Path):
    vis_path = self.fm.change_dir(DIR.RGST, file)
    mask_path = self.fm.change_dir(DIR.SEG, file)

    if not (vis_path.exists() and mask_path.exists()):
      return

    vis = ImageIO.read(vis_path)
    mask_vis = ImageIO.read(mask_path)
    mask = SegMask.vis_to_index(mask_vis)
    mask_cmap = self._cmap(mask)

    self.axes[0].imshow(vis)
    self.axes[1].imshow(vis)
    self.axes[1].imshow(mask_cmap, alpha=0.7)

    self.draw()
