"""Stitcher용 전처리 함수, 클래스"""

from typing import Optional, Tuple

import cv2 as cv
from loguru import logger
import numpy as np
from skimage.exposure import equalize_hist
from skimage.exposure import rescale_intensity

from pano.misc.tools import normalize_rgb_image_hist


class PanoramaPreprocess:
  """
  파노라마 생성을 위한 열화상, 실화상의 전처리
  """

  def __init__(self,
               is_numeric: bool,
               fillna: Optional[float] = 0.0,
               mask_threshold: Optional[float] = -30.0,
               contrast: Optional[str] = 'equalization',
               denoise: Optional[str] = 'bilateral'):
    """
    Parameters
    ----------
    is_numeric : bool
        전처리 대상 영상의 픽셀값이 물리적 의미가 있는지 (실화상인지) 여부.
        `True`인 경우, `mask_threshold` 적용.

    fillna : Optional[float]
        nan이 존재하는 경우 채워넣을 값.

    mask_threshold : Optional[float], optional
        `is_numeric`이 `True`인 경우, 픽셀 값이 `mask_threshold` 미만인 영역은
        파노라마 생성 시 제외함.

    contrast : Optional[str]
        명암비 개선 방법.

        `'equalization'`: Histogram equalization (히스토그램 균일화) 적용.
        `skimage.exposure.equalize_hist` 참조. 열화상에만 적용 권장.

        `'normalization'`: 영상의 밝기 정규화 적용. `cv2.normalize` 참조.
        RGB 형식의 실화상에만 적용.

    denoise : Optional[str]
        잡음 제거 방법.

        `'bilateral'`: Bilateral filter (양방향 필터) 적용.

        `'gaussian'`: Gaussian filter 적용.

    Raises
    ------
    ValueError
        전처리 방법 인자 설정 오류
    """
    if contrast not in (None, 'equalization', 'normalization'):
      raise ValueError
    if denoise not in (None, 'bilateral', 'gaussian'):
      raise ValueError

    self._is_numeric = is_numeric
    self._fillna = fillna
    self._mask_threshold = mask_threshold
    self._contrast_type = contrast
    self._denoise_type = denoise

    self._bilateral_args = {'d': -1, 'sigmaColor': 20, 'sigmaSpace': 10}
    self._gaussian_args = {'ksize': (5, 5), 'sigmaX': 0}

  @property
  def mask_threshold(self):
    """
    영상의 마스킹을 위한 기준 픽셀값.
    `mask_threshold` 미만의 영역은 파노라마 생성 영역에서 제외됨.
    """
    return self._mask_threshold

  @mask_threshold.setter
  def mask_threshold(self, value: Optional[float]):
    self._mask_threshold = value

  @property
  def bilateral_args(self):
    return self._bilateral_args

  @property
  def gaussian_args(self):
    return self._gaussian_args

  def set_bilateral_args(self, d=-1, sigmaColor=20, sigmaSpace=10):
    """
    uint8로 변환한 영상에 적용하는 Bilateral filter의 arguments 설정.
    `cv2.bilateralFilter` 참조.
    """
    self._bilateral_args = {
        'd': d,
        'sigmaColor': sigmaColor,
        'sigmaSpace': sigmaSpace
    }

  def set_gaussian_args(self, ksize=(5, 5), sigmaX=0):
    """
    uint8로 변환한 영상에 적용하는 Gaussian filter의 arguments 설정.
    `cv2.GaussianBlur` 참조.
    """
    self._gaussian_args = {'ksize': tuple(ksize), 'sigmaX': sigmaX}

  def _bilateral_filter(self, image: np.ndarray) -> np.ndarray:
    return cv.bilateralFilter(image, **self._bilateral_args)

  def _gaussian_filter(self, image: np.ndarray) -> np.ndarray:
    return cv.GaussianBlur(image, **self._gaussian_args)

  def masking(self,
              image: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    영상의 픽셀값에 따라 마스킹하고 영상과 마스크 반환.
    대상 영상이 열화상이 아닌 경우 (not `is_numeric`) 마스크는 `None` 반환.

    Parameters
    ----------
    image : np.ndarray

    Returns
    -------
    Tuple[np.ndarray, Optional[np.ndarray]]
        영상. nan이 존재하는 경우 fillna값으로 변경.

        마스크. 파노라마를 생성할 영역은 1, 이외 영역은 0.
    """
    if not self._is_numeric:
      return image, None

    if self.mask_threshold is None:
      threshold_mask = None
    else:
      threshold_mask = (self.mask_threshold < image).astype(bool)

    mask = np.logical_not(np.isnan(image))
    image = np.nan_to_num(image, nan=self._fillna)

    if threshold_mask is not None:
      mask = np.logical_and(mask, threshold_mask)

    return image, mask.astype(np.uint8)

  def adjust_contrast(self, image: np.ndarray) -> np.ndarray:
    """
    선택한 명암 개선 알고리즘 적용. 열화상의 경우 `normalization`을 선택 시
    Histogram normalization을 적용하지 않고 경고 메세지. 실화상의 경우
    `equalization` 선택 시 알고리즘은 적용하지만 경고 메세지를 보냄.

    Parameters
    ----------
    image : np.ndarray
        대상 영상.

    Returns
    -------
    np.ndarray
    """
    if self._contrast_type == 'equalization':
      if not self._is_numeric:
        logger.warning('실화상에 Histogram equalization을 적용합니다. '
                       '예상치 못한 오류가 발생할 수 있습니다.')

      res = equalize_hist(image)

    elif self._contrast_type == 'normalization':
      if not self._is_numeric:
        res = normalize_rgb_image_hist(image)
      else:
        res = image
        logger.warning('열화상에 Histogram normalization을 적용할 수 없습니다. '
                       '명암 보정을 적용하지 않습니다')

    else:
      res = image

    return res

  def denoise(self, image: np.ndarray) -> np.ndarray:
    """
    선택한 잡음 제거 알고리즘 적용.

    Parameters
    ----------
    image : np.ndarray
        대상 영상

    Returns
    -------
    np.ndarray
    """
    if self._denoise_type == 'bilateral':
      res = self._bilateral_filter(image)
    elif self._denoise_type == 'gaussian':
      res = self._gaussian_filter(image)
    else:
      res = image

    return res

  def __call__(self,
               image: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    마스킹, 명암비 개선, 잡음 제거 알고리즘 순차적으로 적용

    Parameters
    ----------
    image : np.ndarray
        대상 영상

    Returns
    -------
    image : np.ndarray
        전처리를 거친 영상
    mask : Optional[np.ndarray]
        마스크
    """
    image, mask = self.masking(image)

    image = self.adjust_contrast(image)
    image = rescale_intensity(image, out_range='uint8')
    image = self.denoise(image)

    return image, mask
