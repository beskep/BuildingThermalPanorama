from context import DATA_DIR

import cv2 as cv
import numpy as np
import pytest
import skimage.exposure

import stitch
from misc.tools import ImageIO


def test_stitcher():
  img_dir = DATA_DIR.joinpath('MainBldgBackLoc1PanTiltTripod/IR')
  files = img_dir.glob('*.npy')
  arrays = [ImageIO.read_image(x) for x in files]

  def _prep(image):
    mask = (image > -30.0).astype(np.uint8)
    # mask = None
    image = skimage.exposure.equalize_hist(image)
    image = skimage.exposure.rescale_intensity(image=image, out_range='uint8')
    image = cv.bilateralFilter(image, d=-1, sigmaColor=20, sigmaSpace=10)

    return image, mask

  stitching_images = stitch.stitcher.StitchingImages(arrays=arrays,
                                                     preprocess=_prep)

  stitcher = stitch.stitcher.Stitcher(mode='pano')
  stitcher.warper_type = 'plane'

  stitched = stitcher.stitch(images=stitching_images, masks=None, names=None)

  assert isinstance(stitched.panorama, np.ndarray)
  assert isinstance(stitched.mask, np.ndarray)
  assert stitched.panorama.shape[:2] == stitched.mask.shape

  assert isinstance(stitched.graph, str)
  assert isinstance(stitched.indices, list)


if __name__ == "__main__":
  pytest.main(['-v', '-k', 'test_stitcher'])
