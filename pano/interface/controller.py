from collections import defaultdict
from pathlib import Path
from typing import Optional

from loguru import logger
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseEvent
from matplotlib.figure import Figure
import numpy as np
from PySide2 import QtCore
from PySide2 import QtGui
from skimage import transform

from pano.misc.tools import prep_compare_images

from .common.pano_files import init_directory
from .common.pano_files import ThermalPanoramaFileManager
from .mpl_qtquick.backend_qtquickagg import FigureCanvas
from .tree import tree_string


def _log(message: str):
  find = message.find('|')
  if find == -1:
    level = 'DEBUG'
  else:
    level = message[:find].upper()
    message = message[(find + 1):]

  logger.log(level, message)


class _Window:

  def __init__(self, window: QtGui.QWindow) -> None:
    self._window = window

  def panel_funtion(self, panel: str, fn: str, *args, **kwargs):
    p = self._window.get_panel(panel)
    if p is None:
      raise ValueError(f'Invalid panel name: {panel}')

    f = getattr(p, fn)

    return f(*args, **kwargs)


class Controller(QtCore.QObject):

  def __init__(self, win: QtGui.QWindow) -> None:
    super().__init__()

    self._win = _Window(win)
    self._wd: Optional[Path] = None

  @property
  def win(self) -> _Window:
    return self._win

  @win.setter
  def win(self, win: QtGui.QWindow):
    self._win = _Window(win)

  @QtCore.Slot(str)
  def log(self, message: str):
    _log(message=message)

  @QtCore.Slot(str)
  def select_working_dir(self, wd):
    wd = Path(wd).resolve()
    if not wd.exists():
      raise FileNotFoundError(wd)

    self._wd = wd
    self.update_project_tree()
    self.update_image_view()

  @QtCore.Slot()
  def update_project_tree(self):
    tree = tree_string(self._wd)
    self.win.panel_funtion('project', 'update_project_tree', tree)

  def update_image_view(self):
    fm = ThermalPanoramaFileManager(self._wd)
    files = fm.raw_files()
    if files:
      self.win.panel_funtion('project', 'update_image_view',
                             ['file:///' + x.as_posix() for x in files])

  @QtCore.Slot()
  def init_directory(self):
    init_directory(directory=self._wd)
    self.update_project_tree()
    self.update_image_view()


class PlotController(QtCore.QObject):

  def __init__(self, parent=None) -> None:
    super().__init__(parent=parent)

    self._canvas: Optional[FigureCanvas] = None
    self._fig: Optional[Figure] = None
    self._axes: Optional[Axes] = None

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

  def init(self, canvas: FigureCanvas):
    self._canvas = canvas
    self._fig = canvas.figure
    self._axes = self._fig.add_subplot(111)

    self.canvas.draw()


class RegistrationPlotController(PlotController):
  _REQUIRED = 4
  _TITLES = ['열화상', '실화상', '비교 (Checkerboard)', '비교 (Difference)']

  def __init__(self, parent=None) -> None:
    super().__init__(parent=parent)

    self._pnts = defaultdict(list)  # 선택된 점들의 mpl 오브젝트
    self._pnts_coord = defaultdict(list)  # 선택된 점들의 좌표
    self._registered = False  # TODO pnts 개수로 판단?
    self._images: Optional[tuple] = None

  @property
  def axes(self) -> np.ndarray:
    return super().axes

  def init(self, canvas: FigureCanvas):
    self._canvas = canvas
    self._fig = canvas.figure
    self._axes = self.fig.subplots(2, 2)
    # self._set_style()

    self.canvas.mpl_connect('button_press_event', self._on_click)
    self.canvas.draw()

  def _set_style(self):
    for ax, title in zip(self.axes.ravel(), self._TITLES):
      if ax.has_data():
        ax.set_title(title)

      ax.set_axis_off()

    ar = self.axes[0, 0].get_aspect()
    self.axes[0, 1].set_aspect(ar)

  def set_images(self, fixed_image: np.ndarray, moving_image: np.ndarray):
    self._images = (fixed_image, moving_image)
    self.axes[0, 0].imshow(fixed_image)
    self.axes[0, 1].imshow(moving_image)

  def _on_click(self, event: MouseEvent):
    logger.trace(event)

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

    self.canvas.draw()

    if not self._registered and self.all_points_selected():
      self._register()

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
    self.reset()

  def all_points_selected(self):
    return all(len(self._pnts_coord[x]) == self._REQUIRED for x in range(2))

  def _register(self):
    src = np.array(self._pnts_coord[1])
    dst = np.array(self._pnts_coord[0])

    trsf = transform.ProjectiveTransform()
    trsf.estimate(src=src, dst=dst)

    registered = transform.warp(image=self._moving_image,
                                inverse_map=trsf.inverse,
                                output_shape=self._fixed_image.shape[:2],
                                preserve_range=True)

    cb, diff = prep_compare_images(image1=self._fixed_image,
                                   image2=registered,
                                   norm=True,
                                   eq_hist=True,
                                   method=['checkerboard', 'diff'])

    self._axes[1, 0].imshow(cb)
    self._axes[1, 1].imshow(diff)

    self._set_style()
    self.canvas.draw()

    self._registered = True

  def reset(self):
    self._axes[1, 0].clear()
    self._axes[1, 1].clear()
    self._registered = False
    self._set_style()
