from pathlib import Path

import numpy as np
from loguru import logger
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseButton, MouseEvent
from skimage import transform
from skimage.exposure import equalize_hist

from pano.interface.common.pano_files import DIR, FN, SP
from pano.interface.mbq import FigureCanvas
from pano.misc import tools
from pano.misc.imageio import ImageIO
from pano.misc.tools import INTRP

from .egs import MousePoints, NavigationToolbar
from .plot_controller import TICK_PARAMS, PanoPlotController, QtGui


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

    self._toolbar: NavigationToolbar
    self._points = MousePoints()
    self._grid = False

    self._images: tuple | None = None
    self._matrices: dict | None = None

    self._file: Path | None = None
    self._matrix: np.ndarray | None = None  # 선택된 영상의 수동 정합 matrix
    self._registered_image: np.ndarray | None = None

  @property
  def matrices(self) -> dict:
    if self._matrices is None:
      path = self.fm.rgst_matrix_path()
      if path.exists():
        npz = np.load(path)
        self._matrices = {f: npz[f] for f in npz.files}
      else:
        self._matrices = {}

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
    self._toolbar = NavigationToolbar(canvas=canvas)

    self._fig = canvas.figure
    self.fig.set_layout_engine('constrained')
    self._axes = self.fig.subplots(2, 2)

    self.canvas.mpl_connect('button_press_event', self._on_click)
    self.draw()

  def home(self):
    if self._images is not None:
      self._toolbar.home()

  def zoom(self, *, value: bool):
    self._toolbar.zoom(value=value)

  def _set_style(self):
    for ax, title in zip(self.axes.ravel(), self._TITLES, strict=True):
      if ax.has_data():
        ax.set_title(title)
      ax.set_axis_off()

    ar = self.axes[0, 0].get_aspect()
    self.axes[0, 1].set_aspect(ar)

    if self._grid:
      for ax in self.axes[0]:
        ax.set_axis_on()
        ax.tick_params(axis='both', which='both', **TICK_PARAMS)

  def set_grid(self, *, grid: bool):
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
      )
      for x in range(2)
    )

    self.axes[0, 0].set_xticks(ticks[1])
    self.axes[0, 0].set_yticks(ticks[0])
    self.axes[0, 1].set_xticks(ticks[1])
    self.axes[0, 1].set_yticks(ticks[0])

  def reset(self):
    self._points.remove_points(None)
    self._registered_image = None
    self._matrix = None

    super().reset()

  def set_images(self, fixed_image: np.ndarray, moving_image: np.ndarray):
    self.reset()

    if fixed_image.shape[:2] != moving_image.shape[:2]:
      moving_image = transform.resize(
        moving_image, output_shape=fixed_image.shape[:2], anti_aliasing=True
      )

    self._images = (fixed_image, moving_image)
    self.axes[0, 0].imshow(equalize_hist(fixed_image))
    self.axes[0, 1].imshow(moving_image)

    self._set_ticks(fixed_image)

  def plot(self, file: Path):
    self._file = file

    ir = ImageIO.read(self.fm.change_dir(DIR.IR, file))
    vis = ImageIO.read(self.fm.change_dir(DIR.VIS, file))

    self.set_images(ir, vis)

    matrix = self.matrices.get(file.stem, None)
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
    if axi not in {0, 1}:
      return

    if event.button == MouseButton.LEFT:
      self._points.add_point(ax=ax, event=event)
    elif event.button == MouseButton.RIGHT:
      self._points.remove_points(ax=ax)
    else:
      return

    if self._points.all_selected():
      self._manual_register()

    self.draw()

  def _plot_registered(self, matrix):
    assert self._images is not None

    registered = transform.warp(
      image=self._images[1],
      inverse_map=matrix,
      output_shape=self._images[0].shape[:2],
      preserve_range=True,
    )

    cb, diff = tools.prep_compare_images(
      image1=self._images[0],
      image2=registered,
      norm=True,
      eq_hist=True,
      method=['checkerboard', 'diff'],
    )

    self.axes[1, 0].imshow(cb)
    self.axes[1, 1].imshow(diff)

    self._registered_image = registered

  def _manual_register(self):
    src = np.array(self._points.coords[self.axes[0, 1]])
    dst = np.array(self._points.coords[self.axes[0, 0]])

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
    assert self._registered_image is not None

    path = self.fm.change_dir(DIR.RGST, self._file)
    ImageIO.save(path=path, array=tools.uint8_image(self._registered_image))

    compare_path = path.with_name(f'{path.stem}{FN.RGST_MANUAL}{path.suffix}')
    self.fig.savefig(compare_path, dpi=300)

    self.update_matrices(path.stem, self._matrix)

  def _save_pano(self):
    """파노라마 정합 결과 저장"""
    assert self._images is not None
    assert self._registered_image is not None

    vis = self.fm.panorama_path(d=DIR.PANO, sp=SP.VIS)
    vis_unrgst = vis.with_stem(f'{vis.stem}Unregistered')
    seg = self.fm.panorama_path(d=DIR.PANO, sp=SP.SEG)
    seg_unrgst = seg.with_stem(f'{seg.stem}Unregistered')

    # 정합 안된 기존 파일 이름 변경
    _rename_file(vis, vis_unrgst)
    _rename_file(seg, seg_unrgst)

    # vis 저장
    ImageIO.save(path=vis, array=tools.uint8_image(self._registered_image))

    # seg 저장
    shape = self._images[0].shape[:2]
    trsf = transform.ProjectiveTransform(matrix=self._matrix)
    seg_resized = transform.resize(
      ImageIO.read(seg_unrgst), order=INTRP.NearestNeighbor, output_shape=shape
    )
    seg_rgst: np.ndarray = transform.warp(
      image=seg_resized,
      inverse_map=trsf.inverse,
      output_shape=shape,
      order=INTRP.NearestNeighbor,
      preserve_range=True,
    )
    ImageIO.save(path=seg, array=seg_rgst.astype(np.uint8))

  def save(self, *, panorama: bool):
    if self._registered_image is None:
      logger.warning('저장할 정합 결과가 없습니다.')
      return

    if not panorama:
      self._save()
    else:
      self._save_pano()
