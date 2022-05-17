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
  assert isinstance(flir_data.signal_reflected, float)
  assert isinstance(flir_data.exif, dict)

  de = DeprecatedExtractor(path)

  assert np.isclose(flir_data.ir, de.extract_ir()).all()
  assert np.isclose(flir_data.vis, de.extract_vis()).all()


@pytest.mark.parametrize('image', images)
@pytest.mark.parametrize('e0', [0.1, 0.6])
@pytest.mark.parametrize('e1', [0.2, 0.9])
def test_correct_emissivity(image, e0, e1):
  path = image_dir.joinpath(image)
  extractor = FlirExtractor(path.as_posix())

  extractor.meta.Emissivity = e1
  ir1, _ = extractor.ir()

  extractor.meta.Emissivity = e0
  ir0, signal_reflected = extractor.ir()

  assert not np.isclose(ir0, ir1).any()

  corrected = FlirExtractor.correct_emissivity(
      ir0, meta=extractor.meta, signal_reflected=signal_reflected, e0=e0, e1=e1)

  assert np.isclose(corrected, ir1).all()


if __name__ == '__main__':
  pytest.main(['-v', '-k', 'test_flir'])
