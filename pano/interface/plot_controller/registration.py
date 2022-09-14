from collections import defaultdict
from pathlib import Path
from typing import Optional

from loguru import logger
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseEvent
import numpy as np
from skimage import transform
from skimage.exposure import equalize_hist

from pano.interface.common.pano_files import DIR
from pano.interface.common.pano_files import FN
from pano.interface.common.pano_files import SP
from pano.interface.mbq import FigureCanvas
from pano.interface.mbq import NavigationToolbar2QtQuick as NavToolbar
from pano.misc.imageio import ImageIO as IIO
from pano.misc.tools import Interpolation
from pano.misc.tools import prep_compare_images
from pano.misc.tools import uint8_image

from .plot_controller import PanoPlotController
from .plot_controller import QtGui
from .plot_controller import TICK_PARAMS


def _rename_file(p0: Path, p1: Path):
  if p1.exists():
    logger.debug('unlink existing file "{}"', p1)
    p1.unlink()

  p0.rename(p1)


class RegistrationPlotController(PanoPlotController):
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
        ax.tick_params(axis='both', which='both', **TICK_PARAMS)

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
    self._pnts.clear()
    self._pnts_coord.clear()
    self._registered_image = None
    self._matrix = None

    super().reset()

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

    ir = IIO.read(self.fm.change_dir(DIR.IR, file))
    vis = IIO.read(self.fm.change_dir(DIR.VIS, file))

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

    if self.all_points_selected():
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

  def _save(self):
    """개별 열/실화상 정합 결과 저장"""
    if self._matrix is None:
      logger.warning('자동 정합 결과가 이미 저장되었습니다.')
      return

    assert self._file is not None
    path = self.fm.change_dir(DIR.RGST, self._file)
    IIO.save(path=path, array=uint8_image(self._registered_image))

    compare_path = path.with_name(f'{path.stem}{FN.RGST_MANUAL}{path.suffix}')
    self.fig.savefig(compare_path, dpi=300)

    self.update_matrices(path.stem, self._matrix)

  def _save_pano(self):
    """파노라마 정합 결과 저장"""
    vis = self.fm.panorama_path(d=DIR.PANO, sp=SP.VIS)
    vis_unrgst = vis.with_stem(f'{vis.stem}Unregistered')
    seg = self.fm.panorama_path(d=DIR.PANO, sp=SP.SEG)
    seg_unrgst = seg.with_stem(f'{seg.stem}Unregistered')

    # 정합 안된 기존 파일 이름 변경
    _rename_file(vis, vis_unrgst)
    _rename_file(seg, seg_unrgst)

    # vis 저장
    IIO.save(path=vis, array=uint8_image(self._registered_image))

    # seg 저장
    shape = self._images[0].shape[:2]
    trsf = transform.ProjectiveTransform(matrix=self._matrix)
    seg_resized = transform.resize(IIO.read(seg_unrgst),
                                   order=Interpolation.NearestNeighbor,
                                   output_shape=shape)
    seg_rgst = transform.warp(image=seg_resized,
                              inverse_map=trsf.inverse,
                              output_shape=shape,
                              order=Interpolation.NearestNeighbor,
                              preserve_range=True)
    IIO.save(path=seg, array=uint8_image(seg_rgst))

  def save(self, panorama: bool):
    if self._registered_image is None:
      logger.warning('저장할 정합 결과가 없습니다.')
      return

    if not panorama:
      self._save()
    else:
      self._save_pano()
