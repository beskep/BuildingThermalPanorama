"""추출해둔 FLIR 컬러맵, 또는 `matplotlib` 컬러맵 설정"""

from typing import Union

import numpy as np
from loguru import logger
from matplotlib.cm import get_cmap
from matplotlib.colors import Colormap, ListedColormap

from pano.misc.cmap import FLIRColormap, apply_colormap
from pano.utils import DIR

__all__ = [
    'apply_colormap',
    'get_iron_colormap',
    'get_mpl_colormap',
    'get_thermal_colormap',
    'save_colormap',
]

DEFAULT_CMAP = 'inferno'
_CMAP_PATH = DIR.RESOURCE.joinpath('iron_colormap_rgb.txt')


def get_iron_colormap() -> ListedColormap:
  """
  기존에 추출하고 저장한 컬러맵 파일 (`data/colormap_rgb.txt`)로부터 FLIR iron
  컬러맵 생성

  Returns
  -------
  ListedColormap
      FLIR Iron 컬러맵

  Raises
  ------
  FileNotFoundError
      파일이 존재하지 않는 경우
  """
  if not _CMAP_PATH.exists():
    raise FileNotFoundError(f'컬러맵 파일이 존재하지 않습니다. ({_CMAP_PATH})')

  return FLIRColormap.from_uint8_text_file(path=_CMAP_PATH)


def get_mpl_colormap(name: str) -> Colormap:
  """
  matplotlib 컬러맵 반환. 유효하지 않은 컬러맵 name을 입력한 경우, 기본 컬러맵
  (inferno)를 적용

  Parameters
  ----------
  name : str
      컬러맵 이름 (`matplotlib.cm.get_cmap` 참조)

  Returns
  -------
  Colormap
      컬러맵
  """
  try:
    cmap = get_cmap(name=name)
  except ValueError:
    logger.warning(
        '`{}`은/는 올바른 컬러맵 이름이 아닙니다. {}를 대신 적용합니다.',
        name,
        DEFAULT_CMAP,
    )
    cmap = get_cmap(DEFAULT_CMAP)

  return cmap


def get_thermal_colormap(name='iron') -> Colormap:
  """
  열화상 시각화를 위한 컬러맵 생성

  Parameters
  ----------
  name : str, optional
      컬러맵 이름. 기본값은 `'iron'` (FLIR 파일에서 추출한 컬러맵).
      iron 파일이 존재하지 않거나 입력한 컬러맵 이름이 유효하지 않은 경우 기본
      컬러맵 (inferno) 적용.

  Returns
  -------
  Colormap
      컬러맵
  """
  if name == 'iron':
    try:
      cmap = get_iron_colormap()
    except FileNotFoundError as e:
      logger.warning(e)
      cmap = get_mpl_colormap(DEFAULT_CMAP)
  else:
    cmap = get_mpl_colormap(name=name)

  return cmap


def save_colormap(path, cmap: Union[str, Colormap] = 'iron', num=101):
  if isinstance(cmap, str):
    cmap = get_thermal_colormap(cmap)

  with open(path, 'w', encoding='utf-8') as f:
    f.write('value,r,g,b\n')

    for value in np.linspace(0.0, 1.0, num=num, endpoint=True):
      rgb = cmap(value)
      f.write(f'{value},{rgb[0]},{rgb[1]},{rgb[2]}\n')
