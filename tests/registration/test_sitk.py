import numpy as np
import pytest
import SimpleITK as sitk
from numpy.linalg import inv
from skimage.exposure import rescale_intensity
from skimage.io import imread
from skimage.transform import resize, warp

import pano.registration.registrator.simpleitk as rsitk
from pano.utils import DIR


class TestSITK:
  registrator = rsitk.SITKRegistrator()
  # test용으로 iteration 적게 설정
  registrator.method.SetOptimizerAsGradientDescent(
      learningRate=0.01, numberOfIterations=10
  )
  fixed_image: np.ndarray = None
  moving_image: np.ndarray = None

  def read_images(self):
    fixed_path = DIR.RESOURCE.joinpath('TestImage/SimpleITK_fixed_0.25.png')
    moving_path = DIR.RESOURCE.joinpath('TestImage/SimpleITK_moving_0.25.png')

    fixed_image = imread(fixed_path.as_posix())
    moving_image = imread(moving_path.as_posix())

    if fixed_image.shape != moving_image.shape:
      moving_image = resize(
          moving_image, output_shape=fixed_image.shape, anti_aliasing=True
      )

    fixed_image = rescale_intensity(fixed_image, out_range=(0.0, 1.0))
    moving_image = rescale_intensity(moving_image, out_range=(0.0, 1.0))

    # moving_image = warp(
    #     moving_image, AffineTransform(translation=(5, 5),
    #                                   rotation=np.deg2rad(2)))

    self.fixed_image = fixed_image
    self.moving_image = moving_image

  def get_images(self):
    if self.fixed_image is None:
      self.read_images()

    return self.fixed_image, self.moving_image

  def test_similarity_parameter(self):
    scale, angle, trnsl1, trnsl2 = range(1, 5)
    trsf = sitk.Similarity2DTransform()
    trsf.SetScale(scale)
    trsf.SetAngle(angle)
    trsf.SetTranslation([trnsl1, trnsl2])
    trsf.SetCenter([4.0, 5.0])

    assert trsf.GetParameters() == (scale, angle, trnsl1, trnsl2)

  def test_affine_parameter(self):
    mtx = (1, 2, 3, 4)
    trsl = (5, 6)

    trsf = sitk.AffineTransform(2)
    trsf.SetMatrix(mtx)
    trsf.SetTranslation(trsl)

    assert trsf.GetMatrix() == mtx
    assert trsf.GetTranslation() == trsl

    assert trsf.GetParameters() == mtx + trsl

  def test_affine_matrix(self):
    self.registrator.transformation = rsitk.Transformation.Affine

    fixed_image, moving_image = self.get_images()
    registered, fn, mtx = self.registrator.register(
        fixed_image, moving_image, set_origin=False
    )

    assert isinstance(registered, np.ndarray)
    assert isinstance(mtx, np.ndarray)

    registered_from_matrix = warp(moving_image, inverse_map=inv(mtx))
    corr = np.corrcoef(
        registered.ravel(), registered_from_matrix.ravel(), rowvar=False
    )[0, 1]
    assert corr > 0.99  # 동일한 변환인지 확인

  def test_similarity_matrix(self):
    self.registrator.transformation = rsitk.Transformation.Similarity

    fixed_image, moving_image = self.get_images()
    registered, fn, mtx = self.registrator.register(
        fixed_image, moving_image, set_origin=False
    )

    assert isinstance(registered, np.ndarray)
    assert isinstance(mtx, np.ndarray)

    registered_by_matrix = warp(moving_image, inverse_map=inv(mtx))
    corr = np.corrcoef(registered.ravel(), registered_by_matrix.ravel(), rowvar=False)[
        0, 1
    ]
    assert corr > 0.99  # 동일한 변환인지 확인

    registered_by_fn = fn(moving_image)
    corr2 = np.corrcoef(registered.ravel(), registered_by_fn.ravel(), rowvar=False)[
        0, 1
    ]
    assert corr2 > 0.99


if __name__ == '__main__':
  pytest.main(['-v'])
