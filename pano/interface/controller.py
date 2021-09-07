from collections import defaultdict
import multiprocessing as mp
from pathlib import Path
from typing import Optional

from loguru import logger
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseEvent
from matplotlib.figure import Figure
import numpy as np
from skimage import transform

from pano.misc.imageio import ImageIO
from pano.misc.tools import prep_compare_images
from pano.utils import set_logger

from .common.pano_files import DIR
from .common.pano_files import FN
from .common.pano_files import init_directory
from .common.pano_files import ThermalPanoramaFileManager
from .mbq import FigureCanvas
from .mbq import QtCore
from .mbq import QtGui
from .tree import tree_string


def _log(message: str):
  find = message.find('|')
  if find == -1:
    level = 'DEBUG'
  else:
    level = message[:find].upper()
    message = message[(find + 1):]

  logger.log(level, message)


def _path2url(path):
  return 'file:///' + Path(path).as_posix()


def _url2path(url: str):
  return Path(url.replace('file:///', ''))


def _producer(queue: mp.Queue, directory, command: str, loglevel: int):
  set_logger(loglevel)

  # pylint: disable=import-outside-toplevel
  from .pano_project import ThermalPanorama

  tp = ThermalPanorama(directory=directory)
  for r in getattr(tp, command)():
    queue.put(r)


class _Consumer(QtCore.QThread):
  done = QtCore.Signal()  # TODO command 종료 후 화면 업데이트
  update = QtCore.Signal(float)

  def __init__(self) -> None:
    super().__init__()
    self._queue: Optional[mp.Queue] = None

  @property
  def queue(self):
    return self._queue

  @queue.setter
  def queue(self, value):
    self._queue = value

  def run(self):
    if self.queue is None:
      raise ValueError('queue not set')

    while True:
      if not self.queue.empty():
        r = self.queue.get()
        self.update.emit(r)

        if r >= 1.0:
          self.done.emit()
          break


class _Window:

  def __init__(self, window: QtGui.QWindow) -> None:
    self._window = window

  def pbar(self, value: float):
    return self._window.pbar(value)

  def panel_funtion(self, panel: str, fn: str, *args):
    p = self._window.get_panel(panel)
    if p is None:
      raise ValueError(f'Invalid panel name: {panel}')

    f = getattr(p, fn)

    return f(*args)


class Controller(QtCore.QObject):

  def __init__(self, win: QtGui.QWindow, loglevel=20) -> None:
    super().__init__()

    self._win = _Window(win)
    self._loglevel = loglevel

    self._consumer = _Consumer()
    self._consumer.update.connect(self.pbar)

    self._wd: Optional[Path] = None
    self._fm: Optional[ThermalPanoramaFileManager] = None
    self._rpc: Optional[RegistrationPlotController] = None

  @property
  def win(self) -> _Window:
    return self._win

  @win.setter
  def win(self, win: QtGui.QWindow):
    self._win = _Window(win)

  @property
  def rpc(self):
    return self._rpc

  @rpc.setter
  def rpc(self, value):
    self._rpc = value

  @QtCore.Slot(str)
  def log(self, message: str):
    _log(message=message)

  @QtCore.Slot(float)
  def pbar(self, r: float):
    self.win.pbar(r)

  @QtCore.Slot(str)
  def prj_select_working_dir(self, wd):
    wd = Path(wd).resolve()
    if not wd.exists():
      raise FileNotFoundError(wd)

    self._wd = wd
    self._fm = ThermalPanoramaFileManager(wd)
    self.prj_update_project_tree()
    self.update_image_view(panel='project')
    self.update_image_view(panel='registration')

  @QtCore.Slot()
  def prj_init_directory(self):
    init_directory(directory=self._wd)
    self.prj_update_project_tree()
    self.update_image_view(panel='project')
    self.update_image_view(panel='registration')

  def prj_update_project_tree(self):
    tree = tree_string(self._wd)
    self.win.panel_funtion('project', 'update_project_tree', tree)

  @QtCore.Slot(str)
  def command(self, command: str):
    if self._wd is None:
      logger.warning('Directory not set')
      return

    self.pbar(0.0)

    queue = mp.Queue()
    self._consumer.queue = queue
    self._consumer.start()

    process = mp.Process(name=command,
                         target=_producer,
                         args=(queue, self._wd, command, self._loglevel))
    process.start()

  def update_image_view(self, panel: str):
    if self._fm is None:
      logger.warning('Directory not set')
      return

    files = self._fm.raw_files()
    if not files:
      # logger.warning('No raw files')
      return

    self.win.panel_funtion(panel, 'update_image_view',
                           [_path2url(x) for x in files])

  @QtCore.Slot(str)
  def rgst_plot(self, url):
    path = _url2path(url)
    logger.debug('Register plot "{}"', path)

    irf = self._wd.joinpath(DIR.IR.value, path.stem).with_suffix(FN.NPY)
    visf = self._wd.joinpath(DIR.VIS.value, path.stem).with_suffix(FN.LL)
    ir = ImageIO.read(irf)
    vis = ImageIO.read(visf)

    self.rpc.set_images(fixed_image=ir, moving_image=vis)


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

  def reset(self):
    for ax in self.axes.ravel():
      ax.clear()

    self._pnts.clear()
    self._pnts_coord.clear()
    self._registered = False

    self._set_style()

  def set_images(self, fixed_image: np.ndarray, moving_image: np.ndarray):
    self.reset()

    self._images = (fixed_image, moving_image)
    self.axes[0, 0].imshow(fixed_image)
    self.axes[0, 1].imshow(moving_image)

    self._set_style()
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

    registered = transform.warp(image=self._images[1],
                                inverse_map=trsf.inverse,
                                output_shape=self._images[0].shape[:2],
                                preserve_range=True)

    cb, diff = prep_compare_images(image1=self._images[0],
                                   image2=registered,
                                   norm=True,
                                   eq_hist=True,
                                   method=['checkerboard', 'diff'])

    self._axes[1, 0].imshow(cb)
    self._axes[1, 1].imshow(diff)

    self._set_style()
    self.canvas.draw()

    self._registered = True
