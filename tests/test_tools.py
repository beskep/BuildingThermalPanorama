from pathlib import Path
from tempfile import TemporaryDirectory

import context

import numpy as np
import pytest

from misc import tools
from misc.tools import ImageIO as IIO


class TestImageIO:
  arr = np.array([[0.0, 5.0], [5.0, 10.0]])
  test_dir = r'D:\test\tmp'
  fname = 'array'
  meta = {'test': [1, 2]}

  def save_image(self, directory):
    path = Path(directory).joinpath(self.fname)
    exts = ['.npy', '.csv', '.png']

    IIO.save_image_and_meta(path=path,
                            array=self.arr,
                            exts=exts,
                            meta=self.meta,
                            dtype='uint16')

    for ext in exts:
      assert path.with_suffix(ext).exists()

    meta_path = path.with_name(
        f'{path.stem}{IIO.META_SUFFIX}{tools.ImageIO.META_EXT}')
    assert meta_path.exists()

  def read_image(self, directory):
    path = Path(directory).joinpath(self.fname)

    img_npy, meta = IIO.read_image_and_meta(path.with_suffix('.npy'),
                                            scale=False)
    assert self.arr == pytest.approx(img_npy)

    meta_ = meta.copy()
    meta_['range'] = {'min': 0.0, 'max': 10.0}
    assert meta == meta_

    img_csv, _ = IIO.read_image_and_meta(path.with_suffix('.csv'), scale=False)
    assert self.arr == pytest.approx(img_csv)

    img_png, _ = IIO.read_image_and_meta(path.with_suffix('.png'), scale=True)
    assert self.arr == pytest.approx(img_png, rel=1e-3)

  def test_io(self):
    with TemporaryDirectory() as temp_dir:
      self.save_image(directory=temp_dir)
      self.read_image(directory=temp_dir)


if __name__ == '__main__':
  pytest.main(['-v', '-k', 'test_tools'])
