import numpy as np
import pytest

from pano.flir import FlirExtractor
from pano.flir.extract import FlirExtractor as DeprecatedExtractor
from pano.utils import DIR

image_dir = DIR.RESOURCE.joinpath('TestImage')
images = ['FLIR0239.jpg']


@pytest.mark.parametrize('image', images)
def test_flir_extractor(image):
  path = image_dir.joinpath(image)

  flir_data = FlirExtractor(path.as_posix()).extract()

  assert isinstance(flir_data.ir, np.ndarray)
  assert isinstance(flir_data.vis, np.ndarray)
  assert isinstance(flir_data.raw_refl, float)
  assert isinstance(flir_data.exif, dict)

  de = DeprecatedExtractor(path)

  assert np.isclose(flir_data.ir, de.extract_ir()).all()
  assert np.isclose(flir_data.vis, de.extract_vis()).all()


if __name__ == '__main__':
  pytest.main(['-v', '-k', 'test_flir'])
