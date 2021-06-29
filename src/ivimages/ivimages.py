import os
import re
from pathlib import Path
from typing import List, Union
from warnings import warn

import numpy as np
import pandas as pd
import yaml
from skimage.exposure import rescale_intensity
from skimage.io import imread

from flir import FlirExtractor

__all__ = ['iv_pattern', 'get_iv', 'IVImages', 'ImageLoader', 'IVImagesLoader']

iv_pattern = re.compile(r'I[VR]_([0-9_-]+).*\.(.*)', re.IGNORECASE)


def get_iv(string):
  return iv_pattern.match(string).group(1)


class IVImages:

  def __init__(self,
               ir_array=None,
               vis_image=None,
               ir_path: Path = None,
               vis_path: Path = None,
               iv=None,
               fill_na=0.0,
               as_gray=False,
               exif: dict = None):
    self._ir_array = ir_array
    self._ir_image = None
    self._vis_image = vis_image
    self._ir_path: Union[None, Path] = ir_path
    self._vis_path: Union[None, Path] = vis_path
    self._iv = iv
    self._fill_na = fill_na
    self._as_gray = as_gray
    self._exif = exif

  @property
  def ir_array(self):
    if self._ir_array is None and self._ir_path is not None:
      if not self._ir_path.exists():
        raise FileNotFoundError(self._ir_path)

      self._ir_array = ImageLoader.read(self._ir_path)

    return self._ir_array

  @property
  def ir_image(self) -> Union[None, np.ndarray]:
    if self._ir_image is None:
      arr = self.ir_array

      if arr is None:
        self._ir_image = None
      else:
        self._ir_image = rescale_intensity(arr, out_range='uint8')

    return self._ir_image

  @property
  def vis_image(self) -> Union[None, np.ndarray]:
    if self._vis_image is None and self._vis_path is not None:
      if not self._vis_path.exists():
        raise FileNotFoundError(self._vis_path)

      self._vis_image = ImageLoader.read(self._vis_path, as_gray=self._as_gray)

    return self._vis_image

  @property
  def ir_nan_mask(self) -> np.ndarray:
    return np.isnan(self.ir_array)

  def ir_mask(self) -> Union[None, np.ndarray]:
    """
    nan가 아닌 영역의 mask 반환
    모두 nan이 아니거나 IR array가 None이면 None 반환

    Returns
    -------
    Union[None, np.ndarray]
        mask
    """
    if self.ir_array is None:
      mask = None
    else:
      nan_mask = self.ir_nan_mask

      if nan_mask.any():
        mask = np.logical_not(nan_mask).astype(np.int8)
      else:
        mask = None

    return mask

  @property
  def iv(self):
    return self._iv

  @property
  def ir_path(self):
    return self._ir_path

  @property
  def vis_path(self):
    return self._vis_path

  @property
  def exif(self):
    return self._exif


class ImageLoader:

  def __init__(self,
               img_dir: Union[str, os.PathLike],
               img_ext: str,
               pattern: str = None):
    self._img_dir = Path(img_dir).resolve()
    if not self._img_dir.exists():
      raise FileNotFoundError(self._img_dir)

    if (pattern is not None) and (not pattern.endswith(img_ext)):
      msg = '{} 외 파일이 선택될 수 있음 ' '(pattern: {})'.format(img_ext, pattern)
      warn(msg)

    self._img_ext = img_ext.strip('.')
    self._pattern = pattern

    self._files = None

  @property
  def directory(self):
    return self._img_dir

  @property
  def ext(self):
    return self._img_ext

  @property
  def pattern(self):
    return self._pattern

  def load_files(self):
    if self._pattern is None:
      pattern = '*.' + self._img_ext
    else:
      pattern = self._pattern

    self._files = list(self._img_dir.rglob(pattern))

  @property
  def files(self) -> List[Path]:
    if self._files is None:
      self.load_files()

    return self._files

  def file(self, iv: str, as_list=False):
    files = [x for x in self.files if iv in x.stem]

    if not files:
      raise FileNotFoundError('iv {} ({})가 {}에 존재하지 않음'.format(
          iv, self._img_ext, self._img_dir.as_posix()))

    if as_list:
      res = files
    else:
      if len(files) > 1:
        raise ValueError('iv {} ({})가 {}에 여러개 존재함'.format(
            iv, self._img_ext, self._img_dir.as_posix()))
      res = files[0]

    return res

  @property
  def iv_list(self):
    return [get_iv(x.name) for x in self.files]

  @staticmethod
  def read(path, as_gray=False) -> Union[np.ndarray, dict]:
    path = Path(path).resolve()
    if not path.exists():
      raise FileNotFoundError(path)

    if path.suffix == '.csv':
      res = pd.read_csv(path, header=None).values
    elif path.suffix == '.xlsx':
      res = pd.read_excel(path, header=None, na_values=['---']).values
    elif path.suffix == '.npy':
      res = np.load(path)
    elif path.suffix in ['.yaml', '.yml']:
      with open(path, 'r') as f:
        res = yaml.safe_load(f)
    else:
      res = imread(path.as_posix(), as_gray=as_gray)

    return res

  def read_by_iv(self, iv: str, as_gray=False) -> np.ndarray:
    file = self.file(iv, as_list=False)
    res = self.read(file, as_gray=as_gray)

    return res

  @staticmethod
  def read_flir(path) -> IVImages:
    path = Path(path)
    if not path.exists():
      FileNotFoundError(path)

    extractor = FlirExtractor()
    ir, vis = extractor.extract_data(path)

    img = IVImages(ir_array=ir, vis_image=vis)

    return img

  def read_flir_by_iv(self, iv: str) -> IVImages:
    file = self.file(iv, as_list=False)

    return self.read_flir(file)


class IVImagesLoader:

  def __init__(self,
               ir_dir: Union[str, os.PathLike],
               ir_ext='xlsx',
               ir_pattern=None,
               vis_dir: Union[str, os.PathLike] = None,
               vis_ext='jpg',
               vis_pattern=None,
               exif_dir: Union[str, os.PathLike] = None,
               exif_ext='yaml',
               exif_pattern=None):
    if not vis_dir and not ir_dir:
      raise ValueError

    if vis_dir:
      self._vis_loader = ImageLoader(img_dir=vis_dir,
                                     img_ext=vis_ext,
                                     pattern=vis_pattern)
    else:
      self._vis_loader = None

    if ir_dir:
      self._ir_loader = ImageLoader(img_dir=ir_dir,
                                    img_ext=ir_ext,
                                    pattern=ir_pattern)
    else:
      self._ir_loader = None

    if exif_dir:
      self._exif_loader = ImageLoader(img_dir=exif_dir,
                                      img_ext=exif_ext,
                                      pattern=exif_pattern)
    else:
      self._exif_loader = None

  @property
  def vis(self) -> Union[ImageLoader, None]:
    return self._vis_loader

  @property
  def ir(self) -> Union[ImageLoader, None]:
    return self._ir_loader

  @property
  def exif(self) -> Union[ImageLoader, None]:
    return self._exif_loader

  def load_files(self):
    for loader in [self.vis, self.ir, self.exif]:
      if loader is not None:
        loader.load_files()

  def read_by_iv(self, iv, as_gray=False, fill_na=0.0) -> IVImages:
    vis_path = None if self.vis is None else self.vis.file(iv)
    ir_path = None if self.ir is None else self.ir.file(iv)

    if self.exif is None:
      exif = None
    else:
      exif = self.exif.read_by_iv(iv)

    img = IVImages(ir_path=ir_path,
                   vis_path=vis_path,
                   iv=iv,
                   fill_na=fill_na,
                   as_gray=as_gray,
                   exif=exif)

    return img
