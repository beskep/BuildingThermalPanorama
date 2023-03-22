"""두 영상의 정합을 위한 전처리 및 Registrator 클래스"""

import abc
from typing import Callable, Optional, Tuple, Union

import numpy as np
from numpy.linalg import inv
from skimage.color import rgb2gray
from skimage.exposure import equalize_hist
from skimage.filters import sobel
from skimage.filters import unsharp_mask
from skimage.transform import resize
from skimage.transform import warp

from pano.misc.tools import normalize_image


class RegistrationPreprocess:

  def __init__(self,
               shape: tuple,
               fillnan: Optional[float] = 0.0,
               eqhist: Union[bool, Tuple[bool, bool]] = True,
               unsharp: Union[bool, Tuple[bool, bool]] = False,
               edge: Union[bool, Tuple[bool, bool]] = False) -> None:
    """
    영상 정합을 위한 전처리 방법.
    `eqhist`, `unsharp`는 상응하는 전처리 방법 적용 여부.
    tuple로 입력한 경우 (e.g. (`True`, `False`)), 각각 `fixed_preprocess`와
    `moving_preprocess`에 적용.

    Parameters
    ----------
    shape : tuple
        영상 shape
    eqhist : Union[bool, Tuple[bool, bool]], optional
        Histogram equalization, by default True
    unsharp : Union[bool, Tuple[bool, bool]], optional
        Unsharp masking filter, by default False
    edge : Union[bool, Tuple[bool, bool]], optional
        Detect edge of image, by default False
    """
    if isinstance(eqhist, bool):
      eqhist = (eqhist, eqhist)
    if isinstance(unsharp, bool):
      unsharp = (unsharp, unsharp)
    if isinstance(edge, bool):
      edge = (edge, edge)

    self._shape = shape

    self._fillnan = fillnan
    self._eqhist = eqhist
    self._unsharp = unsharp
    self._edge = edge

  def parameters(self):
    value: Union[bool, tuple[bool, bool]]
    for name, value in zip(['eqhist', 'unsharp'],
                           [self._eqhist, self._unsharp]):
      if isinstance(value, tuple) and value[0] == value[1]:
        value = value[0]

      yield name, value

  def fixed_preprocess(self, image: np.ndarray) -> np.ndarray:
    if self._fillnan is not None:
      image = np.nan_to_num(image, nan=self._fillnan)

    image = normalize_image(image)

    if self._eqhist[0]:
      image = equalize_hist(image)

    if self._unsharp[0]:
      image = unsharp_mask(image, radius=10)

    if self._edge[0]:
      image = sobel(image)

    return image

  def moving_preprocess(self, image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
      image = rgb2gray(image)

    if self._fillnan is not None:
      image = np.nan_to_num(image, nan=self._fillnan)

    image = normalize_image(image)

    if self._eqhist[1]:
      image = equalize_hist(image)

    if self._unsharp[1]:
      image = unsharp_mask(image, radius=10)

    if self._edge[1]:
      image = sobel(image)

    return image


class RegisteringImage:

  def __init__(self,
               image: np.ndarray,
               shape: Optional[tuple] = None,
               preprocess: Optional[Callable] = None) -> None:
    """
    정합 대상 영상 및 전처리 방법

    Parameters
    ----------
    image : Optional[np.ndarray], optional
        영상
    shape : Optional[tuple], optional
        크기를 조정할 경우 그 shape
    preprocess : Optional[Callable], optional
        전처리 함수
    """
    self._orig_image = image
    self._shape = shape
    self._prep = preprocess

    self._prep_image: Optional[np.ndarray] = None

    self._trsf_mtx: Optional[np.ndarray] = None
    self._trsf_fn: Optional[Callable] = None

  @property
  def matrix(self):
    return self._trsf_mtx

  @property
  def orig_image(self) -> np.ndarray:
    """원본 영상"""
    return self._orig_image

  @orig_image.setter
  def orig_image(self, image):
    self._orig_image = image
    self._resized_image = None
    self._prep_image = None

  def resized_image(self, gray=True) -> np.ndarray:
    """
    지정한 shape으로 영상의 크기 조정.

    `gray`가 `True`이고 원본 영상이 3차원 컬러 영상인 경우, 흑백 영상으로 변환.
    원본 영상은 RGB 인코딩이어야 함.

    Parameters
    ----------
    gray : bool
        흑백 영상으로 변환 여부

    Returns
    -------
    np.ndarray
    """
    if gray and self.orig_image.ndim == 3:
      img = rgb2gray(self.orig_image)
    else:
      img = self.orig_image

    if self._shape is not None and img.shape[:2] != self._shape:
      img = resize(img, output_shape=self._shape)

    return img

  def prep_image(self) -> np.ndarray:
    """
    `resized_image` 및 전처리를 거친 영상

    Returns
    -------
    np.ndarray
    """
    if self._prep_image is None:
      img = self.resized_image(gray=True)
      if self._prep is not None:
        img = self._prep(img)

      self._prep_image = img

    return self._prep_image

  def set_registration(
      self,
      transform_matrix: Optional[np.ndarray] = None,
      transform_function: Optional[Callable[[np.ndarray], np.ndarray]] = None):
    """
    Moving image의 registration 방법 설정

    Parameters
    ----------
    transform_matrix : Optional[np.ndarray], optional
        변환 행렬, by default None
    transform_function : Optional[Callable[[np.ndarray], np.ndarray]], optional
        변환 함수 (f: np.ndarray -> np.ndarray), by default None
    """
    self._trsf_mtx = transform_matrix
    self._trsf_fn = transform_function

  def is_registered(self) -> bool:
    """
    `set_registration`을 통해 정합 방법을 설정한 경우 `True`

    Returns
    -------
    bool
    """
    return (self._trsf_fn is not None) or (self._trsf_mtx is not None)

  def transform_by_matrix(self, image: np.ndarray) -> np.ndarray:
    """
    지정한 변환 행렬을 통해 moving image를 정합

    Parameters
    ----------
    image : np.ndarray
        변환할 영상

    Returns
    -------
    np.ndarray
        Fixed image의 시점에 맞게 변환 (정합)된 영상

    Raises
    ------
    ValueError
        영상 크기 오류
    """
    if self._shape is not None and self._shape != image.shape[:2]:
      raise ValueError('shape error')

    assert self._trsf_mtx is not None
    trsf_image = warp(image, inverse_map=inv(self._trsf_mtx))

    return trsf_image

  def transform(self, image: np.ndarray) -> np.ndarray:
    """
    지정한 registration 방법에 따라 주어진 영상 변환

    Parameters
    ----------
    image : np.ndarray
        변환할 영상

    Returns
    -------
    np.ndarray
        Fixed image의 시점에 맞게 변환 (정합)된 영상
    """
    if self._shape is not None and image.shape[:2] != self._shape:
      image = resize(image, output_shape=self._shape, anti_aliasing=True)

    if self._trsf_mtx is not None:
      trsf_image = self.transform_by_matrix(image)
    else:
      if self._trsf_fn is None:
        raise ValueError('Transform 지정되지 않음.')

      try:
        trsf_image = self._trsf_fn(image)
      except RuntimeError:
        trsf_image = np.stack(
            [self._trsf_fn(image[:, :, x]) for x in range(image.shape[-1])],
            axis=-1)

    return trsf_image

  def registered_orig_image(self) -> np.ndarray:
    """
    원본 영상을 정합한 결과

    Returns
    -------
    np.ndarray
    """
    return self.transform(self.resized_image(gray=False))

  def registered_prep_image(self) -> np.ndarray:
    """
    전처리를 거친 영상을 정합한 결과

    Returns
    -------
    np.ndarray
    """
    return self.transform(self.prep_image())


class BaseRegistrator(abc.ABC):

  @abc.abstractmethod
  def register(
      self,
      fixed_image: np.ndarray,
      moving_image: np.ndarray,
      **kwargs,
  ) -> Tuple[np.ndarray, Optional[Callable], Optional[np.ndarray]]:
    """
    Register image

    Parameters
    ----------
    fixed_image : np.ndarray
        Fixed image
    moving_image : np.ndarray
        Moving image
    **kwargs: dict, optional

    Returns
    -------
    tuple[np.ndarray, Optional[Callable], Optional[np.ndarray]]
        Registered image,
        Function: moving image -> registered image,
        Transform matrix
    """

  def prep_and_register(
      self,
      fixed_image: np.ndarray,
      moving_image: np.ndarray,
      preprocess: RegistrationPreprocess,
      **kwargs,
  ) -> Tuple[RegisteringImage, RegisteringImage]:
    fri = RegisteringImage(image=fixed_image,
                           shape=None,
                           preprocess=preprocess.fixed_preprocess)
    mri = RegisteringImage(image=moving_image,
                           shape=fixed_image.shape,
                           preprocess=preprocess.moving_preprocess)

    _, register, matrix = self.register(fixed_image=fri.prep_image(),
                                        moving_image=mri.prep_image(),
                                        **kwargs)
    mri.set_registration(transform_matrix=matrix, transform_function=register)

    return fri, mri
