"""지정한 roll, yaw, pitch를 적용한 영상 projection"""

from typing import Optional

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
    self._cam_mtx = self.camera_matrix(image_shape=image_shape,
                                       viewing_angle=viewing_angle)
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
    mtx = np.linalg.multi_dot([self._cam_mtx, rot, self._inv_cam_mtx])

    return mtx

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
    mtx = np.linalg.multi_dot([mtx_yaw, mtx_pitch, mtx_roll])

    return mtx


class ImageProjection:

  def __init__(self, image: np.ndarray, viewing_angle: float) -> None:
    """
    Parameters
    ----------
    image : np.ndarray
    viewing_angle : float
        Viewing angle of camera [rad]
    """
    self._image = image
    self._prj_mtx = ProjectionMatrix(image_shape=image.shape[:2],
                                     viewing_angle=viewing_angle)

    self._vertex0 = np.array([
        [0, 0],
        [image.shape[1], 0],
        [image.shape[1], image.shape[0]],
        [0, image.shape[0]],
    ])

  @property
  def image(self):
    return self._image

  def _rotate_matrix(self, roll=0.0, pitch=0.0, yaw=0.0, scale=True):
    mtx_rot = self._prj_mtx(roll=roll, pitch=pitch, yaw=yaw)

    # 영상 원점을 맞추기 위한 translation
    rotated_vertex = transform.matrix_transform(self._vertex0, mtx_rot)
    translation = -np.min(rotated_vertex, axis=0)

    # scale factor 계산
    if not scale:
      scale_factor = 1.0
    else:
      rotated_area = 0.5 * np.abs(
          np.dot(rotated_vertex[:, 0], np.roll(rotated_vertex[:, 1], 1)) -
          np.dot(rotated_vertex[:, 1], np.roll(rotated_vertex[:, 0], 1)))
      scale_factor = np.sqrt(self._image.shape[0] * self._image.shape[1] /
                             rotated_area)

    mtx_fit = transform.AffineTransform(
        scale=scale_factor,
        translation=(translation * scale_factor),
    ).params

    return np.matmul(mtx_fit, mtx_rot)

  def project(self,
              roll=0.0,
              pitch=0.0,
              yaw=0.0,
              scale=True,
              cval=None,
              image: Optional[np.ndarray] = None) -> np.ndarray:
    if image is None:
      image = self._image
    elif image.shape[:2] != self._image.shape[:2]:
      raise ValueError('image.shape[:2] != self._image.shape[:2]')

    if cval is None:
      cval = np.nan

    if all(x == 0.0 for x in [roll, pitch, yaw]):
      return image

    mtx = self._rotate_matrix(roll=roll, pitch=pitch, yaw=yaw, scale=scale)

    vertex = transform.matrix_transform(self._vertex0, mtx)
    shape = np.max(vertex, axis=0) - np.min(vertex, axis=0)

    projected = transform.warp(image=image,
                               inverse_map=np.linalg.inv(mtx),
                               output_shape=np.ceil(shape)[[1, 0]],
                               cval=cval,
                               clip=False,
                               preserve_range=True)

    return projected


if __name__ == '__main__':
  # pylint: disable=ungrouped-imports
  import matplotlib.pyplot as plt
  from skimage import data
  from skimage import img_as_float

  image = img_as_float(data.chelsea())

  prj = ImageProjection(image=image, viewing_angle=np.deg2rad(42.0))
  angles = (2, 0.1, 0.7)

  img_rot = prj.project(*angles, scale=False)
  img_fit = prj.project(*angles, scale=True)

  fig, axes = plt.subplots(1, 3)

  axes[0].imshow(image)
  axes[1].imshow(img_rot)
  axes[2].imshow(img_fit)

  plt.show()
