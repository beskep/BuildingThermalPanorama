from itertools import product
from typing import Generator, Iterable, Tuple

import numpy as np
from numpy.linalg import inv
from skimage import transform as trsf


class ImageEnhance:

  def __init__(
      self,
      image: np.ndarray,
      scales: Iterable[float] = (1.0,),
      rotations: Iterable[float] = (0.0,),
      translations: Iterable[Iterable[float]] = ((0.0, 0.0),)
  ) -> None:
    """
    대상 영상에 similarity transform을 적용한 이미지를 반환

    Parameters
    ----------
    image : np.ndarray
        대상 영상
    scales : Iterable[float], optional
        scales, by default (1.0,)
    rotations : Iterable[float], optional
        rotations [radian], by default (0.0,)
    translations : Iterable[Iterable[float]], optional
        translations, by default ((0.0, 0.0),)
    """
    self._image = image
    self._scales = tuple(scales)
    self._rotations = tuple(rotations)
    self._translations = np.array(translations)

    center = np.array([image.shape[1] // 2, image.shape[0] // 2])

    self.mtx_to_center = np.eye(3)
    self.mtx_to_center[0:2, 2] = -center

    self.mtx_from_center = np.eye(3)
    self.mtx_from_center[0:2, 2] = center

  @property
  def original(self):
    return self._image.copy()

  def transform_matrix_at_center(self, mtx: np.ndarray) -> np.ndarray:
    """
    원점을 기준으로 transform하는 행렬

    Parameters
    ----------
    mtx : np.ndarray
        Transformation matrix

    Returns
    -------
    np.ndarray
        Transformation matrix
    """
    return self.mtx_from_center @ mtx @ self.mtx_to_center

  def similarity_matrix_at_center(self,
                                  scale=1,
                                  rotation=0,
                                  translation=(0, 0)) -> np.ndarray:
    """
    원점을 기준으로 영상에 scale, rotation, translation을 적용하는 matrix

    Parameters
    ----------
    scale : int, optional
        scale, by default 1
    rotation : int, optional
        rotation [radian], by default 0
    translation : tuple, optional
        translation, by default (0, 0)

    Returns
    -------
    np.ndarray
        Transformation matrix
    """
    mtx = trsf.SimilarityTransform(scale=scale,
                                   rotation=rotation,
                                   translation=translation).params
    mtx_at_center = self.transform_matrix_at_center(mtx)

    return mtx_at_center

  def transformed_image(self,
                        scale=1,
                        rotation=0,
                        translation=(0, 0)) -> np.ndarray:
    """
    원점을 기준으로 영상에 scale, rotation, translation을 적용

    Parameters
    ----------
    scale : int, optional
        scale, by default 1
    rotation : int, optional
        rotation [radian], by default 0
    translation : tuple, optional
        translation, by default (0, 0)

    Returns
    -------
    np.ndarray
        Transformed image
    """
    mtx = self.similarity_matrix_at_center(scale=scale,
                                           rotation=rotation,
                                           translation=translation)
    transformed_image = trsf.warp(self._image, inverse_map=inv(mtx))

    return transformed_image

  def transformed_images(
      self
  ) -> Generator[Tuple[np.ndarray, float, float, np.ndarray], None, None]:
    """
    입력한 scale, rotation, translation 조합 (product)을 적용한 영상의 iterator

    Yields
    ------
    transformed_image : np.ndarray
    scale : float
    rotation : float
        Rotation angle [rad]
    translation : np.ndarray
    """
    for scale, rotation, translation in product(
        self._scales,
        self._rotations,
        self._translations,
    ):
      if scale == 1 and rotation == 0 and tuple(translation) == (0, 0):
        image = self._image.copy()
      else:
        image = self.transformed_image(scale=scale,
                                       rotation=rotation,
                                       translation=translation)
      yield image, scale, rotation, translation
