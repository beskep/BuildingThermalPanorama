"""
생성한 파노라마의 시점 보정을 위한 코드
"""

import numpy as np
from skimage import transform


class TransformMatrix:

  @staticmethod
  def camera_matrix(image_shape, viewing_angle):
    """영상 해상도와 카메라 시야각으로부터 camera matrix 추정

    Parameters
    ----------
    image_shape : tuple
        ndarray.shape
    viewing_angle : float
        viewing angle of camera [rad]

    Returns
    -------
    np.ndarray
        camera matrix
    """
    f = image_shape[0] / np.tan(viewing_angle / 2.0)
    translation = (image_shape[1] / 2.0, image_shape[0] / 2.0)
    trsf = transform.AffineTransform(scale=(f, f), translation=translation)

    return trsf.params

  @staticmethod
  def rotate_roll(angle=0.0):
    """roll

    Parameters
    ----------
    angle : float
        rotation angle [rad]
    """
    if angle:
      mtx_roll = np.array([
          [np.cos(angle), -np.sin(angle), 0],
          [np.sin(angle), np.cos(angle), 0],
          [0.0, 0.0, 1.0],
      ])
    else:
      mtx_roll = np.identity(n=3)

    return mtx_roll

  @staticmethod
  def rotate_lr(angle):
    """rotate left-right (yaw)

    Parameters
    ----------
    angle : float
        rotation angle [rad]
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
  def rotate_ud(angle):
    """rotate up-down (pitch)

    Parameters
    ----------
    angle : float
        rotation angle [rad]
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
  def rotate(cls, angle_roll=0.0, angle_ud=0.0, angle_lr=0.0):
    mtx_roll = cls.rotate_roll(angle=angle_roll)
    mtx_ud = cls.rotate_ud(angle=angle_ud)
    mtx_lr = cls.rotate_lr(angle=angle_lr)
    mtx = np.linalg.multi_dot([mtx_lr, mtx_ud, mtx_roll])

    return mtx


class ImageProjection:

  def __init__(self, image, angles, viewing_angle):
    """
    Parameters
    ----------
    image : np.ndarray
        대상 영상
    angles : tuple
        (roll, ud, lr) [rad]
    viewing_angle : float
        [rad]
    """
    self._image = image

    self._angles = angles
    self._viewing_angle = viewing_angle

    self._camera_matrix = None
    self._camera_matrix_inv = None
    self._rotate_matrix = None

  @property
  def camera_matrix(self):
    if self._camera_matrix is None:
      self._camera_matrix = TransformMatrix.camera_matrix(
          image_shape=self._image.shape[:2], viewing_angle=self._viewing_angle)

    return self._camera_matrix

  @property
  def camera_matrix_inv(self):
    if self._camera_matrix_inv is None:
      self._camera_matrix_inv = np.linalg.inv(self._camera_matrix)

    return self._camera_matrix_inv

  @property
  def rotate_matrix(self):
    if self._rotate_matrix is None:
      self._rotate_matrix = TransformMatrix.rotate(*self._angles)

    return self._rotate_matrix

  @property
  def angles(self):
    return self.angles

  @angles.setter
  def angles(self, value):
    self._rotate_matrix = None
    self._angles = value

  def original_vertex(self):
    coords = [
        [0, 0],
        [self._image.shape[1], 0],
        [self._image.shape[1], self._image.shape[0]],
        [0, self._image.shape[0]],
    ]

    return np.array(coords)

  def transformed_vertex(self, matrix):
    vertex = transform.matrix_transform(coords=self.original_vertex(),
                                        matrix=matrix)

    return vertex

  def project_matrix(self):
    mtx = np.linalg.multi_dot(
        [self.camera_matrix, self.rotate_matrix, self.camera_matrix_inv])

    return mtx

  def project(self, scale=True):
    """지정한 각도 따라 (카메라를 축으로) 회전한 영상 반환

    Parameters
    ----------
    scale : bool
        True인 경우, 변환한 영상의 크기를 원본과 비슷하게 조절

    Returns
    -------
    np.ndarray
        Projected image
    """
    # 시점 변환 matrix
    mtx_rot = self.project_matrix()

    # 영상 원점을 맞추기 위한 translation
    vertex_rot = self.transformed_vertex(mtx_rot)
    translation = -np.min(vertex_rot, axis=0)

    # scale factor 계산
    if not scale:
      scale_factor = 1.0
    else:
      area_rot = 0.5 * np.abs(
          np.dot(vertex_rot[:, 0], np.roll(vertex_rot[:, 1], 1)) -
          np.dot(vertex_rot[:, 1], np.roll(vertex_rot[:, 0], 1)))
      scale_factor = np.sqrt(self._image.shape[0] * self._image.shape[1] /
                             area_rot)

    # tranlate, scale matrix
    mtx_fit = transform.AffineTransform(
        scale=scale_factor,
        translation=(translation * scale_factor),
    ).params

    mtx = np.matmul(mtx_fit, mtx_rot)
    vertex_fit = self.transformed_vertex(mtx)
    shape_fit = np.max(vertex_fit, axis=0) - np.min(vertex_fit, axis=0)

    trsf_fit = transform.ProjectiveTransform(matrix=mtx)
    img = transform.warp(image=self._image,
                         inverse_map=trsf_fit.inverse,
                         output_shape=np.ceil(shape_fit)[[1, 0]],
                         clip=False)

    return img


if __name__ == '__main__':
  import matplotlib.pyplot as plt
  from skimage import data, img_as_float

  image = img_as_float(data.chelsea())

  img_prj = ImageProjection(image=image,
                            angles=(0.0, 0.1, 0.7),
                            viewing_angle=(42.0 * np.pi / 180.0))

  img_rot = img_prj.project(scale=False)
  img_fit = img_prj.project(scale=True)

  fig, axes = plt.subplots(1, 3)

  axes[0].imshow(image)
  axes[1].imshow(img_rot)

  axes[2].imshow(img_fit)

  plt.show()
