import numpy as np
import pytest

from pano import flir
from pano.utils import DIR

extractor = flir.FlirExtractor()
image_dir = DIR.RESOURCE.joinpath('TestImage')
images = ['FLIR0239.jpg']


@pytest.mark.parametrize('image', images)
def test_process_image(image):
  path = image_dir.joinpath(image)
  extractor.process_image(path)
  assert extractor.extractor.thermal_image_np is not None
  assert extractor.extractor.rgb_image_np is not None


@pytest.mark.parametrize('image', images)
def test_extract_data(image):
  path = image_dir.joinpath(image)
  ir_array, vis_array = extractor.extract_data(path)

  assert isinstance(ir_array, np.ndarray)
  assert isinstance(vis_array, np.ndarray)
  assert vis_array.dtype == np.uint8


def test_extract_ir_non_flir_file():
  path = image_dir.joinpath('lena.jpg')

  with pytest.raises(flir.FlirExifNotFoundError):
    extractor.extract_ir(path)


if __name__ == '__main__':
  pytest.main(['-v', '-k', 'test_flir'])
