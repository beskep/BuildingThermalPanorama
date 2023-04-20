from typing import Optional

from matplotlib.axes import Axes
from matplotlib.cm import get_cmap
from matplotlib.colorbar import make_axes_gridspec
import numpy as np

from pano.distortion.projection import ImageProjection
from pano.interface.common.pano_files import DIR
from pano.interface.common.pano_files import FN
from pano.interface.common.pano_files import SP
from pano.interface.mbq import FigureCanvas
from pano.interface.pano_project import ThermalPanorama
from pano.misc.cmap import apply_colormap
from pano.misc.imageio import ImageIO
from pano.misc.tools import limit_image_size
from pano.misc.tools import uint8_image

from .plot_controller import CropToolbar
from .plot_controller import PanoPlotController
from .plot_controller import QtGui
from .plot_controller import TICK_PARAMS


class PanoramaPlotController(PanoPlotController):
  _GRID_COUNTS = (7, 7)  # (height, width)

  def __init__(self, parent=None) -> None:
    super().__init__(parent=parent)

    self._cax: Optional[Axes] = None
    self._prj: Optional[ImageProjection] = None
    self._toolbar: Optional[CropToolbar] = None

    self._dir = DIR.PANO
    self._sp = SP.IR
    self._angles: Optional[np.ndarray] = None
    self._va = np.deg2rad(42.0)

    self._cmap = get_cmap('inferno')
    self._grid = False

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

    self._set_style()

  def reset(self):
    self.axes.clear()
    self.cax.clear()
    self.draw()

  def home(self):
    if self.axes.has_data():
      assert self._toolbar is not None
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
    if self.axes.has_data() and self._grid:
      self.axes.set_axis_on()
      self.axes.tick_params(axis='both', which='both', **TICK_PARAMS)
    else:
      self.axes.set_axis_off()

    if self.cax.has_data():
      self.cax.set_axis_on()
    else:
      self.cax.set_axis_off()

  def _load_image(self, d: DIR, sp: SP):
    path = self.fm.panorama_path(d=d, sp=sp)
    if not path.exists() and d is DIR.COR:
      d = DIR.PANO
      path = self.fm.panorama_path(d=d, sp=sp, error=True)

    image = ImageIO.read(path)

    if sp is SP.IR:
      image = image.astype(np.float32)
      mask = ImageIO.read(self.fm.panorama_path(d=d, sp=SP.MASK, error=True))
      if image.shape[:2] != mask.shape[:2]:
        raise ValueError('파노라마와 마스크 파일의 shape이 일치하지 않습니다.')

      image[np.logical_not(mask)] = np.nan

    return d, image

  def _image_projection(self, limit=0):
    _, image = self._load_image(d=self._dir, sp=self._sp)

    if limit:
      # order=1 (bilinear interpolation)이 아니면
      # nan이 포함된 영상 크기 변환에 오류가 발생하는 듯
      image = limit_image_size(image=image, limit=limit, order=1)

    return ImageProjection(image=image, viewing_angle=self.viewing_angle)

  def _set_ticks(self, shape: tuple):
    ticks = tuple(
        np.linspace(0, shape[x], num=self._GRID_COUNTS[x], endpoint=True)
        for x in range(2))

    self.axes.set_xticks(ticks[1])
    self.axes.set_yticks(ticks[0])

  def plot(self, d: DIR, sp: SP):
    self._dir, image = self._load_image(d=d, sp=sp)
    self._sp = sp

    self.axes.clear()
    self.cax.clear()
    self._prj = None

    im = self.axes.imshow(image, cmap=self._cmap)

    if sp is SP.IR:
      self.fig.colorbar(im, cax=self.cax, ax=self.axes)
      self.cax.get_yaxis().labelpad = 10
      self.cax.set_ylabel('Temperature [℃]', rotation=90)

    self._set_ticks(image.shape)
    self.draw()

  def project(self, roll=0.0, pitch=0.0, yaw=0.0, limit=9999):
    if self._prj is None:
      self._prj = self._image_projection(limit=limit)

    self._angles = np.deg2rad([roll, pitch, yaw])
    image = self._prj.project(*self._angles)

    self.axes.clear()
    self.axes.imshow(image, cmap=self._cmap)

    self._set_ticks(image.shape)
    self.draw()

  def set_grid(self, grid):
    if not self._grid ^ grid:
      return

    self._grid = grid
    self.draw()


def save_manual_correction(wd, subdir, viewing_angle, angles,
                           crop_range: Optional[np.ndarray]):
  # FIXME 영상 시점 어긋나는 문제
  tp = ThermalPanorama(wd, init_loglevel='TRACE')
  ir_pano = ImageIO.read(tp.fm.panorama_path(subdir, SP.IR))
  prj = ImageProjection(ir_pano, viewing_angle=viewing_angle)
  angles = np.deg2rad(angles)

  for sp in [SP.IR, SP.VIS, SP.SEG, SP.MASK]:
    if sp is SP.IR:  # noqa: SIM108
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
    if not path.parent.exists():
      path.parent.mkdir()

    if sp is not SP.IR:
      ImageIO.save(path=path, array=corrected.astype(np.uint8))
    else:
      ImageIO.save_with_meta(path=path,
                             array=corrected,
                             exts=[FN.NPY, FN.LL],
                             dtype='uint16')
      # colormap 버전
      ImageIO.save(path=tp.fm.color_path(path),
                   array=apply_colormap(image=corrected, cmap=tp.cmap))

  tp.save_multilayer_panorama()
