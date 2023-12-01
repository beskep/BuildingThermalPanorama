# ruff: noqa: PLR2004

import numpy as np

from pano.misc import tools


def test_crop_by_mask():
  row = np.arange(5)
  image = row + 10 * row.reshape([-1, 1])

  mask = np.array(
    [
      [0, 0, 0, 0, 0],
      [0, 1, 1, 1, 0],
      [0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0],
      [0, 0, 0, 1, 0],
    ]
  )

  cr, cm = tools.crop_mask(mask=mask, morphology_open=False)
  ci = cr.crop(image)

  assert cr.cropped
  assert cr.x_min == 1
  assert cr.x_max == 4
  assert cr.y_min == 1
  assert cr.y_max == 5

  ci_ = np.array([1, 2, 3]) + 10 * np.array([1, 2, 3, 4]).reshape([-1, 1])
  cm_ = np.array(
    [
      [1, 1, 1],
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 1],
    ]
  )
  assert np.all(ci == ci_)
  assert np.all(cm == cm_)


def test_crop_by_mask_not_cropped():
  image = np.zeros([4, 4])
  mask = np.ones([4, 4])

  cr, cm = tools.crop_mask(mask=mask, morphology_open=False)
  ci = cr.crop(image)

  assert not cr.cropped
  assert np.all(image == ci)
  assert np.all(mask == cm)


def test_limit_image_size():
  image = np.zeros([4, 8])

  assert image.shape == tools.limit_image_size(image=image, limit=100).shape

  half = tools.limit_image_size(image=image, limit=4)
  assert half.shape[0] == 2
  assert half.shape[1] == 4
