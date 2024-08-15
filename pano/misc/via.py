"""VGG Image Annotator (VIA)의 프로젝트 저장 결과로부터 annotation 해석"""
# ruff: noqa: DOC402 DOC501

import json
from collections import defaultdict
from pathlib import Path
from warnings import warn

import numpy as np
from skimage.draw import polygon as draw_polygon
from skimage.io import imsave


def draw_mask(shape, rows, cols):
  mask = np.zeros(shape=shape, dtype=int)
  rr, cc = draw_polygon(r=rows, c=cols)
  mask[rr, cc] = 1

  return mask


class VIAProject:
  def __init__(
    self,
    path: str | Path,
    attribute_name: str,
    attributes_ids: list[str],
  ):
    """
    VIA로 annotate한 영상의 영역 정보 해석

    Parameters
    ----------
    path : Union[str, Path]
        VIA project 파일 (json 형식) 경로
    attribute_name : str
        VIA를 통해 지정한 attribute 이름
    attributes_ids : List[str]
        VIA를 통해 지정한 attribute의 아이디 (annotation class)

    Raises
    ------
    FileNotFoundError
    """
    path = Path(path)
    if not path.exists():
      raise FileNotFoundError(path)

    self._path = path
    with path.open('r', encoding='utf-8-sig') as f:
      self._json = json.load(f)

    self._image_metadata = self._json['_via_img_metadata']
    self._fname_dict = {
      Path(self._image_metadata[x]['filename']).resolve().as_posix(): x
      for x in self._image_metadata
    }

    self._attr_name = attribute_name
    self._attr_ids = attributes_ids

  @property
  def files(self):
    return list(self._fname_dict.keys())

  def fname_key(self, fname: str):
    return self._fname_dict[fname]

  def meta_data(self, key):
    return self._image_metadata[key]

  def regions(self, fname: str):
    """대상 파일에 지정된 영역의 클래스 이름과 polygon 좌표 정보 generate"""
    meta_data = self.meta_data(self.fname_key(fname))
    regions = meta_data['regions']

    for region in regions:
      region_shape = region['shape_attributes']
      assert region_shape['name'] == 'polygon'

      ra = region['region_attributes'][self._attr_name]
      if not any(x in ra for x in self._attr_ids):
        warn(f'file {fname}: attribute 지정 안됨', UserWarning, stacklevel=1)
        continue

      for x in self._attr_ids:
        if ra.get(x):
          region_class = x
          break
      else:
        raise ValueError

      yield region_class, region_shape

  def write_masks(
    self,
    fname: str | Path,
    save_dir: str | Path,
    shape: tuple[int, int],
  ):
    """
    VIA로 지정한 영역 정보를 png 파일 형태 mask로 저장

    Parameters
    ----------
    fname : Union[str, Path]
        대상 영상 파일
    save_dir : Union[str, Path]
        저장 경로
    shape : Tuple[int, int]
        대상 영상의 shape
    """
    save_dir = Path(save_dir).resolve()
    fname = Path(fname)
    class_count = defaultdict(int)

    for rclass, rshape in self.regions(fname.name):
      class_count[rclass] += 1
      mask = draw_mask(
        shape=shape, rows=rshape['all_points_y'], cols=rshape['all_points_x']
      )
      mask = mask.astype('uint8') * 255

      path = save_dir.joinpath(
        f'{fname.stem}_{rclass}_{class_count[rclass]}'
      ).with_suffix('.png')
      imsave(fname=path.as_posix(), arr=mask, check_contrast=False)
