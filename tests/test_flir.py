import numpy as np
import pytest

from pano.flir import FlirExtractor
from pano.utils import DIR

image_dir = DIR.RESOURCE.joinpath('TestImage')
images = ['FLIR0239.jpg', 'IR_2020-11-10_0122.jpg']


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
    ir0, meta=extractor.meta, signal_reflected=signal_reflected, e0=e0, e1=e1
  )

  mask = np.isnan(corrected)

  assert np.isclose(corrected[~mask], ir1[~mask], rtol=0, atol=1e-4).all()
