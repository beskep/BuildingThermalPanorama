"""FLIR 촬영 영상에 저장된 colormap 추출 및 영상 데이터에 적용"""

from pathlib import Path

import cv2 as cv
import numpy as np
from matplotlib.colors import Colormap, ListedColormap

from pano.misc import sp, tools


def extract_flir_colormap(image_path: str, save_path: str, color_space: str = 'RGB'):
  """
  FLIR 촬영 파일로부터 열화상의 colormap 정보를 추출하고 text 파일로 저장

  Parameters
  ----------
  image_path : str
      FLIR 촬영 파일 경로
  save_path : str
      추출한 colormap의 text 파일 저장 경로
  color_space : str, optional
      Colormap을 저장할 colorspace. "RGB" 또는 "YCrCb".
  """
  color_space = color_space.lower()
  if color_space not in {'rgb', 'ycrcb'}:
    msg = f'{color_space} not in {{"rgb", "ycrcb"}}'
    raise ValueError(msg)

  palette = sp.get_exif_binary(path=image_path, tag='-Palette')
  palette_array = np.array(list(palette)).reshape([1, -1, 3]).astype(np.uint8)

  if color_space == 'rgb':
    palette_array = cv.cvtColor(palette_array, code=cv.COLOR_YCrCb2RGB)

  np.savetxt(fname=save_path, X=palette_array.reshape([-1, 3]), fmt='%d')


class FLIRColormap(ListedColormap):

  @classmethod
  def from_uint8_colors(
      cls, colors: np.ndarray, color_space: str = 'RGB'
  ) -> ListedColormap:
    """
    unit8 형태 색상 정보로부터 matplotlib cmap 생성

    Parameters
    ----------
    colors : np.ndarray
        색상 정보. shape: (Any, 3)
    color_space : str, optional
        Colorspace. "RGB" 또는 "YCrCb".

    Returns
    -------
    ListedColormap
    """
    color_space = color_space.lower()
    if color_space not in {'rgb', 'ycrcb'}:
      raise ValueError(color_space)

    if color_space == 'ycrcb':
      colors = colors.reshape([1, -1, 3])
      colors = cv.cvtColor(colors, code=cv.COLOR_YCrCb2RGB)

    colors = colors.astype('float').reshape([-1, 3]) / 255.0
    return cls(colors=colors)

  @classmethod
  def from_flir_file(cls, path: str) -> ListedColormap:
    """
    FLIR 촬영 파일로부터 colormap 정보 추출

    Parameters
    ----------
    path : str
        FLIR 촬영 파일

    Returns
    -------
    ListedColormap
    """
    palette = sp.get_exif_binary(path=path, tag='-Palette')
    colors = np.array(list(palette)).astype(np.uint8)
    return cls.from_uint8_colors(colors=colors, color_space='ycrcb')

  @classmethod
  def from_uint8_text_file(cls, path: str | Path, color_space='RGB') -> ListedColormap:
    """
    `extract_flir_colormap` 함수로 추출/저장한 텍스트 파일로부터 colormap 생성

    Parameters
    ----------
    path : Union[str, Path]
        파일 경로
    color_space : str, optional
        저장된 color space. "RGB" 또는 "YCrCb".

    Returns
    -------
    ListedColormap
    """
    colors = np.loadtxt(path)
    return cls.from_uint8_colors(colors=colors, color_space=color_space)


def apply_colormap(image: np.ndarray, cmap: Colormap, *, na=False) -> np.ndarray:
  """
  대상 영상에 지정한 colormap을 적용한 영상 반환.

  Normalize ([0, 1]) -> cmap() -> rescale to uint8

  Parameters
  ----------
  image : np.ndarray
      대상 영상
  cmap : Colormap
      Colormap
  na : bool
      `True`면 NA가 존재하는 영상 처리

  Returns
  -------
  np.ndarray
      Colormap을 적용한 영상
  """
  if not na:
    mask = None
  else:
    mask = np.isnan(image)
    image[mask] = np.nanmin(image)

  norm_image = tools.normalize_image(image=image)
  color_image = cmap(norm_image)
  uint8_image = tools.uint8_image(color_image)

  if mask is not None:
    image[mask] = 0

  return uint8_image
