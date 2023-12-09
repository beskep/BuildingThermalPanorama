from collections.abc import Iterable

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import pano.interface.common.pano_files as pf
from pano.interface.mbq import FigureCanvas, QtCore, QtGui
from pano.interface.mbq import NavigationToolbar2QtQuick as NavToolbar

_DIRS = ('bottom', 'top', 'left', 'right')
TICK_PARAMS = {key: False for key in _DIRS + tuple('label' + x for x in _DIRS)}


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

    self._app: QtGui.QGuiApplication | None = None
    self._canvas: FigureCanvas | None = None
    self._fig: Figure | None = None
    self._axes: Axes | None = None

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
    self.canvas.draw_idle()
    self.app.processEvents()

  def reset(self):
    axs: Iterable[Axes]
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
    self._fm: pf.ThermalPanoramaFileManager | None = None

  @property
  def fm(self) -> pf.ThermalPanoramaFileManager:
    if self._fm is None:
      raise pf.WorkingDirNotSetError
    return self._fm

  @fm.setter
  def fm(self, value: pf.ThermalPanoramaFileManager):
    self._fm = value

  def _set_style(self):
    pass

  def draw(self):
    self._set_style()
    super().draw()

  def reset(self):
    super().reset()
    self.draw()
