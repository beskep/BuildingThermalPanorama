from context import DATA_DIR

import cv2 as cv
import numpy as np
import pytest
import skimage.exposure
import skimage.io

import ivimages as ivi
import stitch


def test_stitcher():
  img_dir = DATA_DIR.joinpath('MainBldgBackLoc1PanTiltTripod')
  loader = ivi.ImageLoader(img_dir=img_dir, img_ext='npy')
  arrays = [loader.read(x).astype('float32') for x in loader.files]

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

  image, mask, graph, indices = stitcher.stitch(images=stitching_images,
                                                masks=None,
                                                image_names=None)

  assert isinstance(image, np.ndarray)
  assert isinstance(mask, np.ndarray)


if __name__ == "__main__":
  pytest.main(['-v', '-k', 'test_stitcher'])
