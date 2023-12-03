from pathlib import Path
from tempfile import TemporaryDirectory
from typing import ClassVar

import numpy as np
import pytest

from pano.misc.imageio import ImageIO


class TestImageIO:
  arr = np.array([[0.0, 5.0], [5.0, 10.0]])
  test_dir = r'D:\test\tmp'
  fname = 'array'
  meta: ClassVar = {'test': [1, 2]}

  def save_image(self, directory):
    path = Path(directory).joinpath(self.fname)
    exts = ['.npy', '.csv', '.png']

    ImageIO.save_with_meta(
      path=path, array=self.arr, exts=exts, meta=self.meta, dtype='uint16'
    )

    for ext in exts:
      assert path.with_suffix(ext).exists()

    meta_path = path.with_name(f'{path.stem}{ImageIO.META_SUFFIX}{ImageIO.META_EXT}')
    assert meta_path.exists()

  def read_image(self, directory):
    path = Path(directory).joinpath(self.fname)

    img_npy, meta = ImageIO.read_with_meta(path.with_suffix('.npy'), scale=False)
    assert self.arr == pytest.approx(img_npy)
    assert meta is not None

    meta_ = meta.copy()
    meta_['range'] = {'min': 0.0, 'max': 10.0}
    assert meta == meta_

    img_csv, _ = ImageIO.read_with_meta(path.with_suffix('.csv'), scale=False)
    assert self.arr == pytest.approx(img_csv)

    img_png, _ = ImageIO.read_with_meta(path.with_suffix('.png'), scale=True)
    assert self.arr == pytest.approx(img_png, rel=1e-3)

  def test_io(self):
    with TemporaryDirectory() as temp_dir:
      self.save_image(directory=temp_dir)
      self.read_image(directory=temp_dir)


def test_webp():
  rng = np.random.default_rng(42)
  image = rng.integers(low=0, high=255, size=(2, 2, 3), dtype=np.uint8)

  with TemporaryDirectory() as temp_dir:
    path = Path(temp_dir).joinpath('test.webp')
    ImageIO.save(path=path, array=image)
    image_ = ImageIO.read(path=path)[:, :, :3]

  assert image_ == pytest.approx(image)
