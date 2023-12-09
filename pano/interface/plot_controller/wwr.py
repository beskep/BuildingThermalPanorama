from functools import cached_property
from typing import Literal, NamedTuple

import numpy as np
from loguru import logger
from matplotlib.image import AxesImage
from skimage.color import gray2rgb

import pano.interface.common.pano_files as pf
from pano.interface.common.pano_files import DIR, SP
from pano.interface.mbq import FigureCanvas, QtGui
from pano.misc.imageio import ImageIO
from pano.misc.tools import SegMask

from .plot_controller import PanoPlotController

Image = Literal['vis', 'seg']


class WWR(NamedTuple):
  wall: int
  window: int
  wwr: float


class Images:
  def __init__(self, fm: pf.ThermalPanoramaFileManager) -> None:
    self._fm = fm

  def read(self, sp: SP):
    path = self._fm.panorama_path(DIR.ANLY, sp=sp)
    if sp is not SP.IR:
      path = path.parent / f'Image-{path.name}'

    return ImageIO.read(path)

  def read_floor(self):
    return ImageIO.read(self._fm.subdir(DIR.OUT) / 'Floor.npy')

  @cached_property
  def vis(self):
    return self.read(SP.VIS)

  @cached_property
  def seg(self):
    return self.read(SP.SEG)

  @cached_property
  def floor(self):
    return self.read_floor()

  @cached_property
  def coverage(self):
    """외피 차폐율 (etc / (wall + window + etc))."""
    seg = SegMask.vis2index(self.seg)

    envelope = np.isin(seg, [SegMask.WALL, SegMask.WINDOW])
    objects = seg != SegMask.BG

    coverage = np.full_like(self.floor, fill_value=np.nan, dtype=np.float32)
    for idx in np.unique(self.floor):
      area = self.floor == idx
      coverage[area] = 1 - np.sum(envelope[area]) / np.sum(objects[area])

    return coverage

  def reset(self):
    for v in ['vis', 'seg', 'floor', 'coverage']:
      self.__dict__.pop(v, None)

  def exclude_area(self, threshold: float):
    return self.coverage > threshold

  def wwr(self, threshold: float):
    exclude = self.exclude_area(threshold)
    seg = SegMask.vis2index(self.seg)
    seg[exclude] = SegMask.BG

    wall = int(np.sum(seg == SegMask.WALL))
    window = int(np.sum(seg == SegMask.WINDOW))
    envelope = wall + window
    wwr = float('nan') if envelope == 0 else window / envelope

    return WWR(wall, window, wwr)


class WWRPlotController(PanoPlotController):
  ALPHA = 0.1

  def __init__(self, parent=None) -> None:
    super().__init__(parent)
    self._images: Images | None = None
    self._ai: AxesImage

    self._image: Image = 'vis'
    self._threshold = 0.1
    self._wwr = WWR(0, 0, float('nan'))

  def init(self, app: QtGui.QGuiApplication, canvas: FigureCanvas):
    super().init(app, canvas)
    self.fig.set_layout_engine('constrained')
    self.axes.set_axis_off()

  @property
  def fm(self):
    return super().fm

  @fm.setter
  def fm(self, value):
    self._fm = value
    self._images = Images(value)

  @property
  def images(self):
    if self._images is None:
      raise pf.WorkingDirNotSetError
    return self._images

  @property
  def image(self):
    return self._image

  @image.setter
  def image(self, value: Image):
    update = self._image != value
    self._image = value

    if update:
      self.plot()

  @property
  def threshold(self):
    return self._threshold

  @threshold.setter
  def threshold(self, value: float):
    if not (0 <= value <= 1):
      raise ValueError(value)

    update = self._threshold != value
    self._threshold = value

    if update:
      self.plot()

  @property
  def wwr(self):
    return self._wwr

  def update(self, image: Image, threshold: float, *, force=False):
    if not (0 <= threshold <= 1):
      raise ValueError(threshold)

    if force:
      self.images.reset()

    update = force or (self._image != image) or (self._threshold != threshold)
    self._image = image
    self._threshold = threshold

    if update:
      self._wwr = self.images.wwr(threshold)
      self.plot()

    return self._wwr

  def _issue(self):
    try:
      self.images.coverage  # noqa: B018
    except pf.WorkingDirNotSetError as e:
      return str(e)
    except FileNotFoundError as e:
      self.images.reset()
      return f'파일이 존재하지 않음: "{e}"'

    return None

  def plot(self):
    if issue := self._issue():
      logger.debug(issue)
      return

    match self._image:
      case 'vis':
        rgb = self.images.vis
      case 'seg':
        rgb = gray2rgb(self.images.seg)
      case _:
        raise ValueError(self._image)

    dmax = np.iinfo(rgb.dtype).max
    exclude = self.images.exclude_area(self.threshold)
    alpha = dmax * (1 - (1 - self.ALPHA) * exclude)
    rgba = np.dstack((rgb, alpha.astype(rgb.dtype)))

    try:
      self._ai.set_data(rgba)
    except AttributeError:
      self._ai = self.axes.imshow(rgba)

    self.draw()
