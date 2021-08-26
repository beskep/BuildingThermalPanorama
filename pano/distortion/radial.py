"""
Alemán-Flores, M., Alvarez, L., Gomez, L., & Santana-Cedrés, D. (2014).
Automatic Lens Distortion Correction Using One-Parameter Division Models.
Image Processing On Line, 4, 327–343. https://doi.org/10.5201/ipol.2014.106

Bukhari, F., & Dailey, M. N. (2013). Automatic Radial Distortion Estimation
from a Single Image. Journal of Mathematical Imaging and Vision, 45(1), 31–45.
https://doi.org/10.1007/s10851-012-0342-2

테스트
"""

from typing import Optional

import numpy as np
from rich.progress import track
from scipy.linalg import lstsq
from skimage.draw import polygon_perimeter
from skimage.exposure import equalize_adapthist
from skimage.measure import CircleModel
from skimage.measure import find_contours
from skimage.measure import ransac
from skimage.restoration import denoise_bilateral

from pano import utils
from pano.misc import tools


class AbsResidualCircleModel(CircleModel):

  def residuals(self, data):
    residuals = np.abs(super().residuals(data))

    return residuals


class RadialDistortionModel:

  def __init__(self) -> None:
    self.params = None

  @staticmethod
  def _matrix(data: np.ndarray):
    if data.ndim != 2:
      raise ValueError
    if data.shape[1] != 3:
      raise ValueError

    A = 2 * data[:, :2]  # (2*xc, 2*yc)
    B = np.average(
        np.square(data),
        axis=1,
        weights=[1, 1, -1],
    ).reshape([-1, 1])  # (xc^2 + yc^2 - r^2)

    rows = list(range(1, data.shape[0])) + [0]
    Ap = A - A[rows]
    Bp = B - B[rows]

    return Ap, Bp

  @classmethod
  def static_estimate(cls, data: np.ndarray):
    Ap, Bp = cls._matrix(data=data)

    x0y0, _, _, _ = lstsq(a=Ap, b=Bp)
    x0 = x0y0.ravel()[0]
    y0 = x0y0.ravel()[1]

    M = data.copy()
    M[:, 0] -= x0
    M[:, 1] -= y0  # [xc - x0, yc - y0, r]
    invk1s = np.average(np.square(M), axis=1, weights=[1, 1, -1])
    k1s = np.true_divide(1.0,
                         invk1s,
                         out=np.full_like(invk1s, np.nan),
                         where=invk1s != 0)
    k1 = np.nanmean(k1s)

    return (x0, y0, k1)

  @classmethod
  def static_residuals(cls, data: np.ndarray, x0, y0):
    Ap, Bp = cls._matrix(data=data)
    error = Bp - Ap @ np.array([[x0], [y0]])

    return np.abs(error.ravel())

  def estimate(self, data):
    self.params = self.static_estimate(np.array(data))

    return True

  def residuals(self, data):
    data = np.array(data)
    x0, y0, k1 = self.params
    error = self.static_residuals(data=data, x0=x0, y0=y0)

    return error

  def inv_distort_map(self, xdyd: np.ndarray):
    if self.params is None:
      raise ValueError('parameter is None')

    x0y0 = np.array(self.params[:2]).reshape([1, 2])

    xd0_yd0 = xdyd - x0y0
    rdsq = np.sum(np.square(xd0_yd0), axis=1).reshape([-1, 1])  # (r_d)^2
    k1rdsq = self.params[2] * rdsq  # k_1 * (r_d)^2

    rr = np.divide(1 - np.sqrt(1 - 4 * k1rdsq),
                   2 * k1rdsq,
                   out=np.ones_like(k1rdsq),
                   where=k1rdsq != 0)
    assert np.all(rr >= 0.0)

    xu0yu0 = rr * xd0_yd0
    xuyu = xu0yu0 + x0y0

    return xuyu


def inv_radial_distort_map(xdyd: np.ndarray, k1: float, center) -> np.ndarray:
  center = np.array(center).reshape([-1, 2])
  xdyd_center = xdyd - center

  rdsq = np.sum(np.square(xdyd_center), axis=1).reshape([-1, 1])  # (r_d)^2
  k1rdsq = k1 * rdsq  # k_1 * (r_d)^2

  # if k1 >= 0:
  #   # pincushion distortion
  # else:
  #   # barrel distortion

  c = np.divide(1 - np.sqrt(1 - 4 * k1rdsq),
                2 * k1rdsq,
                out=np.ones_like(k1rdsq),
                where=k1rdsq != 0)
  assert np.all(c >= 0.0)

  xuyu = c * xdyd_center
  xuyu += center

  return xuyu


def radial_distort_map(xuyu: np.ndarray, k1: float, center) -> np.ndarray:
  center = np.array(center).reshape([-1, 2])
  xuyu_center = xuyu - center

  ru_sq = np.sum(np.square(xuyu_center), axis=1).reshape([-1, 1])
  ru = np.sqrt(ru_sq)

  k1_rusq_p1 = k1 * ru_sq + 1
  rd = np.divide(ru, k1_rusq_p1, out=np.ones_like(ru_sq), where=k1_rusq_p1 != 0)

  r_ratio = np.divide(rd, ru, out=np.ones_like(ru), where=ru != 0)
  xdyd_center = r_ratio * xuyu_center

  xdyd = xdyd_center + center

  return xdyd


class RadialDistortion:

  def __init__(self,
               perimeter_threshold: float,
               levels: list,
               clip_limit=0.05) -> None:
    self._perimeter_threshold = perimeter_threshold
    self._levels = levels
    self._clip_limit = clip_limit

    self._ransac_circle_args = dict(min_samples=3,
                                    residual_threshold=5,
                                    max_trials=1000)
    self._ransac_distort_args = dict(min_sample=4)

  def set_circle_ransac_options(self,
                                min_samples=10,
                                residual_threshold=5,
                                max_trials=10000,
                                **kwargs):
    self._ransac_circle_args.update(
        {
            'min_samples': min_samples,
            'residual_threshold': residual_threshold,
            'max_trials': max_trials
        },
        kwargs=kwargs)

  def preprocess(self, image: np.ndarray, mask: Optional[np.ndarray] = None):
    image = equalize_adapthist(image, clip_limit=self._clip_limit)
    image = denoise_bilateral(image=image, sigma_color=0.05, sigma_spatial=1)

    image = tools.normalize_image(image=image)

    if mask is not None:
      image[np.logical_not(mask)] = 0

    return image

  def _iterate_contours(self, image: np.ndarray, threshold: float):
    for level in track(self._levels,
                       description='Extracting contours...',
                       console=utils.console):
      contours = find_contours(image=image, level=level)

      for contour in contours:
        rr, cc = polygon_perimeter(contour[:, 1], contour[:, 0])
        contour_full = np.vstack([cc, rr]).T
        if contour_full.shape[0] < threshold:
          continue

        yield contour_full, level

  def detect_circles(self, image: np.ndarray):
    if image.ndim != 2:
      raise ValueError

    threshold = (self._perimeter_threshold * 2 *
                 (image.shape[0] + image.shape[1]))

    for contour, level in self._iterate_contours(image=image,
                                                 threshold=threshold):
      model, inliers = ransac(data=contour,
                              model_class=AbsResidualCircleModel,
                              **self._ransac_circle_args)

      yield level, contour, model, inliers
