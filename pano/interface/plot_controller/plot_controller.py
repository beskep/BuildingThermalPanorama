from typing import Optional

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np

from pano.interface.common.pano_files import ThermalPanoramaFileManager
from pano.interface.mbq import FigureCanvas
from pano.interface.mbq import NavigationToolbar2QtQuick as NavToolbar
from pano.interface.mbq import QtCore
from pano.interface.mbq import QtGui

_DIRS = ('bottom', 'top', 'left', 'right')
TICK_PARAMS = {key: False for key in _DIRS + tuple('label' + x for x in _DIRS)}


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

  def reset(self):
    if isinstance(self.axes, Axes):
      axs = (self.axes,)
    elif isinstance(self.axes, np.ndarray):
      axs = self.axes.ravel()
    else:
      axs = self.axes

    for ax in axs:
      ax.clear()


class PanoPlotController(PlotController):

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

  def reset(self):
    super().reset()
    self.draw()
