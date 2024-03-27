"""지정한 roll, yaw, pitch를 적용한 영상 projection"""

from collections.abc import Collection
from typing import TypedDict

import numpy as np
from skimage import transform


class ProjectionMatrix:
  def __init__(self, image_shape: tuple, viewing_angle: float) -> None:
    """
    영상 시야각과 회전 각도로부터 projection matrix 계산

    Parameters
    ----------
    image_shape : tuple
        ndarray.shape
    viewing_angle : float
        Viewing angle of camera [rad]
    """
    self._cam_mtx = self.camera_matrix(
      image_shape=image_shape, viewing_angle=viewing_angle
    )
    self._inv_cam_mtx = np.linalg.inv(self._cam_mtx)

  def __call__(self, roll=0.0, pitch=0.0, yaw=0.0) -> np.ndarray:
    """
    대상 영상에 roll, pitch, yaw를 차례로 적용하는 projection matrix 계산.

    Parameters
    ----------
    roll : float, optional
        Roll angle [rad], by default 0.0
    pitch : float, optional
        Pitch angle [rad], by default 0.0
    yaw : float, optional
        Yaw angle [rad], by default 0.0

    Returns
    -------
    np.ndarray
    """
    rot = self.rotate(roll=roll, pitch=pitch, yaw=yaw)
    return np.linalg.multi_dot([self._cam_mtx, rot, self._inv_cam_mtx])  # matrix

  @staticmethod
  def camera_matrix(image_shape: tuple, viewing_angle: float) -> np.ndarray:
    """
    영상 해상도, 카메라 시야각 (폭)으로부터 camera matrix 추정

    Parameters
    ----------
    image_shape : tuple
        ndarray.shape
    viewing_angle : float
        Viewing angle of camera [rad]

    Returns
    -------
    np.ndarray
    """
    f = image_shape[0] / np.tan(viewing_angle / 2.0)
    translation = (image_shape[1] / 2.0, image_shape[0] / 2.0)
    trsf = transform.AffineTransform(scale=(f, f), translation=translation)
    return trsf.params

  @staticmethod
  def roll(angle=0.0) -> np.ndarray:
    """
    Roll의 projection matrix (평면 법선 방향을 축으로 회전).

    Parameters
    ----------
    angle : float, optional
        [rad], by default 0.0

    Returns
    -------
    np.ndarray
    """
    if angle:
      mtx = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0.0, 0.0, 1.0],
      ])
    else:
      mtx = np.identity(n=3)

    return mtx

  @staticmethod
  def yaw(angle=0.0) -> np.ndarray:
    """
    Yaw의 projection matrix (up-down).

    Parameters
    ----------
    angle : float, optional
        [rad], by default 0.0

    Returns
    -------
    np.ndarray
    """
    if angle:
      mtx_lr = np.array([
        [np.cos(angle), 0.0, np.sin(angle)],
        [0.0, 1.0, 0.0],
        [-np.sin(angle), 0.0, np.cos(angle)],
      ])
    else:
      mtx_lr = np.identity(n=3)

    return mtx_lr

  @staticmethod
  def pitch(angle=0.0) -> np.ndarray:
    """
    Pitch의 projection matrix (left-right).

    Parameters
    ----------
    angle : float, optional
        [rad], by default 0.0

    Returns
    -------
    np.ndarray
    """
    if angle:
      mtx_ud = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(angle), -np.sin(angle)],
        [0.0, np.sin(angle), np.cos(angle)],
      ])
    else:
      mtx_ud = np.identity(n=3)

    return mtx_ud

  @classmethod
  def rotate(cls, roll=0.0, pitch=0.0, yaw=0.0) -> np.ndarray:
    """
    roll, pitch, yaw를 차례로 적용하는 projection matrix 계산.

    Parameters
    ----------
    roll : float, optional
        Roll angle [rad], by default 0.0
    pitch : float, optional
        Pitch angle [rad], by default 0.0
    yaw : float, optional
        Yaw angle [rad], by default 0.0

    Returns
    -------
    np.ndarray
    """
    mtx_roll = cls.roll(angle=roll)
    mtx_pitch = cls.pitch(angle=pitch)
    mtx_yaw = cls.yaw(angle=yaw)
    return np.linalg.multi_dot([mtx_yaw, mtx_pitch, mtx_roll])


class WarpArgs(TypedDict, total=False):
  order: int | None
  cval: float | None
  clip: bool
  preserve_range: bool


class ImageProjection:
  def __init__(self, image: np.ndarray, viewing_angle: float) -> None:
    """
    ImageProjection

    Parameters
    ----------
    image : np.ndarray
    viewing_angle : float
        Viewing angle of camera [rad]
    """
    self._image = image
    self._prj_mtx = ProjectionMatrix(
      image_shape=image.shape[:2], viewing_angle=viewing_angle
    )

    self._vertex0 = np.array([
      [0, 0],
      [image.shape[1], 0],
      [image.shape[1], image.shape[0]],
      [0, image.shape[0]],
    ])

  @property
  def image(self):
    return self._image

  def _rotate_matrix(self, roll=0.0, pitch=0.0, yaw=0.0, *, scale=True):
    mtx_rot = self._prj_mtx(roll=roll, pitch=pitch, yaw=yaw)

    # 영상 원점을 맞추기 위한 translation
    rotated_vertex = transform.matrix_transform(self._vertex0, mtx_rot)
    translation = -np.min(rotated_vertex, axis=0)

    # scale factor 계산
    if not scale:
      scale_factor = 1.0
    else:
      rotated_area = 0.5 * np.abs(
        np.dot(rotated_vertex[:, 0], np.roll(rotated_vertex[:, 1], 1))
        - np.dot(rotated_vertex[:, 1], np.roll(rotated_vertex[:, 0], 1))
      )
      scale_factor = np.sqrt(self._image.shape[0] * self._image.shape[1] / rotated_area)

    mtx_fit = transform.AffineTransform(
      scale=scale_factor,
      translation=(translation * scale_factor),
    ).params

    return np.matmul(mtx_fit, mtx_rot)

  def project(
    self,
    angles: Collection[float],
    *,
    scale=True,
    image: np.ndarray | None = None,
    warp_args: WarpArgs | None = None,
    preserve_dtype=True,
  ) -> np.ndarray:
    if image is None:
      image = self._image
    elif image.shape[:2] != self._image.shape[:2]:
      msg = 'image.shape[:2] != self._image.shape[:2]'
      raise ValueError(msg)

    if all(x == 0 for x in angles):
      return image

    mtx = self._rotate_matrix(*angles, scale=scale)

    vertex = transform.matrix_transform(self._vertex0, mtx)
    shape = np.max(vertex, axis=0) - np.min(vertex, axis=0)

    warp_args = warp_args or WarpArgs()
    warp_args.setdefault('order', None)
    warp_args.setdefault('clip', False)
    warp_args.setdefault('preserve_range', True)
    isinteger = issubclass(image.dtype.type, np.integer)

    if warp_args.get('cval', None) is None:
      warp_args['cval'] = 0 if isinteger else float('nan')

    projected: np.ndarray = transform.warp(
      image=image,
      inverse_map=np.linalg.inv(mtx),
      output_shape=np.ceil(shape)[[1, 0]],
      **warp_args,
    )

    if preserve_dtype:
      if isinteger:
        projected[np.isnan(projected)] = 0

      projected = projected.astype(image.dtype)

    return projected


if __name__ == '__main__':
  # pylint: disable=ungrouped-imports
  import matplotlib.pyplot as plt
  from skimage import data, img_as_float

  img = img_as_float(data.chelsea())

  prj = ImageProjection(image=img, viewing_angle=np.deg2rad(42.0))
  a = (2, 0.1, 0.7)

  img_rot = prj.project(a, scale=False)
  img_fit = prj.project(a, scale=True)

  fig, axes = plt.subplots(1, 3)
  fig.set_layout_engine('constrained')

  axes[0].imshow(img)
  axes[1].imshow(img_rot)
  axes[2].imshow(img_fit)

  plt.show()
