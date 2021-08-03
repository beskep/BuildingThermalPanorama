"""기타 영상 처리 함수들"""

from pathlib import Path
from typing import Optional, Tuple
from warnings import warn

from utils import StrPath

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL.Image
import yaml
from skimage.exposure import equalize_hist, rescale_intensity
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.util import compare_images


def normalize_image(image: np.ndarray) -> np.ndarray:
  """
  영상의 pixel 값을 [0.0, 1.0] 범위로 정규화

  Parameters
  ----------
  image : np.ndarray
      대상 영상

  Returns
  -------
  np.ndarray
  """
  return rescale_intensity(image=image, out_range=(0.0, 1.0))


def uint8_image(image: np.ndarray) -> np.ndarray:
  return rescale_intensity(image, out_range='uint8')


def uint16_image(image: np.ndarray) -> np.ndarray:
  return rescale_intensity(image, out_range='uint16')


def normalize_rgb_image_hist(image: np.ndarray) -> np.ndarray:
  """
  밝기 정규화를 통해 컬러 영상의 명암비 개선

  Parameters
  ----------
  image : np.ndarray
      RGB uint8 영상

  Returns
  -------
  np.ndarray
  """
  image_yuv = cv.cvtColor(image, cv.COLOR_RGB2YUV)
  image_yuv[:, :, 0] = cv.normalize(image_yuv[:, :, 0],
                                    dst=None,
                                    alpha=0,
                                    beta=255,
                                    norm_type=cv.NORM_MINMAX)
  equalized = cv.cvtColor(image_yuv, cv.COLOR_YUV2RGB)

  return equalized


def _mask_range(mask, axis):
  mask_ = np.any(mask, axis=axis)
  c1 = np.argmax(mask_)
  c2 = len(mask_) - np.argmax(mask_[::-1])

  return int(c1), int(c2)


def mask_bbox(mask: np.ndarray,
              morphology_open=True) -> Tuple[int, int, int, int]:
  """
  마스크 영상 중 True인 영역의 bounding box 좌표를 찾음

  참조:
  https://stackoverflow.com/questions/39206986/numpy-get-rectangle-area-just-the-size-of-mask/48346079

  Parameters
  ----------
  mask : np.ndarray
      2차원 마스크. 대상 영역은 True, 대상 외 영역은 False.
  morphology_open : bool, optional
      Opening (morphological operation)을 통한 노이즈 제거 여부

  Returns
  -------
  Tuple[int]
      (x_min, x_max, y_min, y_max)

  Raises
  ------
  ValueError
      입력한 마스크가 2차원이 아닌 경우
  """
  if mask.ndim != 2:
    raise ValueError

  if morphology_open:
    kernel = np.ones(shape=(3, 3), dtype='uint8')
    mask_ = cv.morphologyEx(src=mask, op=cv.MORPH_OPEN, kernel=kernel)
  else:
    mask_ = mask

  xx = _mask_range(mask=mask_, axis=0)
  yy = _mask_range(mask=mask_, axis=1)

  return xx + yy


def erode(image: np.ndarray, iterations=1) -> np.ndarray:
  kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
  eroded = cv.erode(image, kernel, iterations=iterations)

  return eroded


def bin_size(image1: np.ndarray,
             image2: Optional[np.ndarray] = None,
             bins='auto') -> int:
  """
  영상의 histogram, entropy 계산을 위한 적정 bin 개수 추정.
  `numpy.histogram_bin_edges`함수를 이용함.

  Parameters
  ----------
  image1 : np.ndarray
      대상 영상.
  image2 : Optional[np.ndarray]
      대상 영상. 미입력 시 image1만 고려해서 bins 산정.

      입력 시 image1과 image2의 적정 bins의 평균으로 결정.
  bins : str, optional
      추정 방법. `numpy.histogram_bin_edges` 참조.

  Returns
  -------
  int
      Histogram의 적정 bin 개수
  """
  bins_count = np.histogram_bin_edges(image1, bins=bins).size
  if image2 is not None:
    bins_count2 = np.histogram_bin_edges(image2, bins=bins).size
    bins_count = (bins_count + bins_count2) // 2

  return bins_count


def _check_and_prep(image1: np.ndarray, image2: np.ndarray, normalize: bool,
                    eq_hist: bool) -> Tuple[np.ndarray, np.ndarray]:
  if image1.shape != image2.shape:
    raise ValueError('image1.shape != image2.shape')
  if image1.ndim != 2:
    raise ValueError('image1.ndim != 2')
  if image2.ndim != 2:
    raise ValueError('image2.ndim != 2')

  if normalize:
    image1 = normalize_image(image1)
    image2 = normalize_image(image2)

  if eq_hist:
    image1 = equalize_hist(normalize_image(image1))
    image2 = equalize_hist(normalize_image(image2))

  return image1, image2


def prep_compare_images(image1: np.ndarray,
                        image2: np.ndarray,
                        norm=False,
                        eq_hist=True,
                        method='checkerboard',
                        n_tiles=(8, 8)) -> np.ndarray:
  """
  두 영상의 전처리 및 비교 영상 생성.
  두 영상 모두 2차원 (gray image)이며 해상도 (ndarray.shape)이 동일해야 함.

  Parameters
  ----------
  image1 : np.ndarray
      대상 영상 1
  image2 : np.ndarray
      대상 영상 2
  norm : bool, optional
      `True`이면 픽셀 밝기 단위를 0에서 1까지 normalize
  eq_hist : bool, optional
      명암 개선을 위해 Histogram Equalization 적용 여부
  method : str, optional
      `skimage.exposure.compare_images` 옵션
  n_tiles : tuple, optional
      Checkerboard 타일 개수

  Returns
  -------
  np.ndarray

  Raises
  ------
  ValueError
      두 영상의 해상도 (shape)이 다르거나 2차원이 아닌 경우
  """
  img1, img2 = _check_and_prep(image1=image1,
                               image2=image2,
                               normalize=norm,
                               eq_hist=eq_hist)

  image = compare_images(img1, img2, method=method, n_tiles=n_tiles)

  return image


def prep_compare_fig(images: Tuple[np.ndarray, np.ndarray],
                     titles=('Image 1', 'Image 2'),
                     norm=False,
                     eq_hist=True,
                     n_tiles=(8, 8)):
  """
  두 영상의 전처리 및 비교 영상 생성.
  두 영상 모두 2차원 (gray image)이며 해상도 (ndarray.shape)이 동일해야 함.

  Parameters
  ----------
  images : Tuple[np.ndarray, np.ndarray]
      대상 영상
  titles : tuple
      figure에 표시할 각 영상 제목
  norm : bool, optional
      `True`이면 픽셀 밝기 단위를 0에서 1까지 normalize
  eq_hist : bool, optional
      명암 개선을 위해 Histogram Equalization 적용 여부
  n_tiles : tuple, optional
      Checkerboard 타일 개수

  Returns
  -------
  tuple:
      plt.Figure

      np.ndarray of plt.Axes

  Raises
  ------
  ValueError
      두 영상의 해상도 (shape)이 다르거나 2차원이 아닌 경우
  """
  img1, img2 = _check_and_prep(image1=images[0],
                               image2=images[1],
                               normalize=norm,
                               eq_hist=eq_hist)

  fig, axes = plt.subplots(2, 2, figsize=(16, 9))
  cmap = 'gray'

  axes[0, 0].imshow(images[0], cmap=cmap)
  axes[0, 0].set_title(titles[0])

  axes[0, 1].imshow(images[1], cmap=cmap)
  axes[0, 1].set_title(titles[1])

  cb = compare_images(img1, img2, method='checkerboard', n_tiles=n_tiles)

  axes[1, 0].imshow(cb, cmap=cmap)
  axes[1, 0].set_title('Compare (checkerboard)')

  diff = compare_images(img1, img2, method='diff')
  axes[1, 1].imshow(diff, cmap=cmap)
  axes[1, 1].set_title('Compare (difference)')

  for ax in axes.ravel():
    ax.set_axis_off()

  fig.tight_layout()

  return fig, axes


def limit_image_size(image: np.ndarray, limit: int) -> np.ndarray:
  max_shape = np.max(image.shape[:2]).astype(float)
  if max_shape <= limit:
    return image

  dtype = image.dtype

  ratio = limit / max_shape
  target_shape = (int(image.shape[0] * ratio), int(image.shape[1] * ratio))

  resized = resize(image=image.astype(np.float32),
                   output_shape=target_shape,
                   preserve_range=True,
                   anti_aliasing=True)

  return resized.astype(dtype)


class ImageIO:
  META_SUFFIX = '_meta'
  META_EXT = '.yaml'
  CSV_EXT = '.csv'
  NPY_EXT = '.npy'
  XLSX_EXT = '.xlsx'

  ENCODING = 'UTF-8-SIG'
  DELIMITER = ','

  @classmethod
  def meta_path(cls, path: Path) -> Path:
    return path.with_name(f'{path.stem}{cls.META_SUFFIX}{cls.META_EXT}')

  @classmethod
  def read(cls, path: StrPath) -> np.ndarray:
    """
    주어진 path의 확장자에 따라 영상 파일 해석

    Parameters
    ----------
    path : Union[str, Path]
        대상 파일 경로. `.npy`, `.csv`, `.xlsx` 및 `.png` 등 영상 확장자 지정 가능
        (`skimage.io.imread` 참조).

    Returns
    -------
    np.ndarray

    Raises
    ------
    FileNotFoundError
        대상 파일이 존재하지 않을 때
    """
    path = Path(path).resolve()
    if not path.exists():
      raise FileNotFoundError(path)

    if path.suffix == cls.NPY_EXT:
      image = np.load(file=path.as_posix())
    elif path.suffix == cls.CSV_EXT:
      image = np.loadtxt(fname=path.as_posix(), delimiter=cls.DELIMITER)
    elif path.suffix == cls.XLSX_EXT:
      image = pd.read_excel(path.as_posix())
      image = np.array(image)
    else:
      image = imread(fname=path.as_posix())

    return image

  @classmethod
  def read_with_meta(cls,
                     path: StrPath,
                     scale=False) -> Tuple[np.ndarray, Optional[dict]]:
    """
    주어진 path의 확장자에 따라 영상 파일 및 메타 정보 해석.
    조건에 따라 영상의 픽셀 값을 메타 정보에 기록된 원 범위로 scale함.

    Parameters
    ----------
    path : Union[str, Path]
        대상 파일 경로
    scale : bool, optional
        `True`이며 대상 파일이 영상 확장자 (`.png` 등)인 경우,
        `save_with_meta` 함수로 저장된 메타 정보 파일로부터 읽은
        원본 범위로 픽셀 값을 scale함.

    Returns
    -------
    image : np.ndarray
    meta_data : Optional[dict]
    """
    path = Path(path).resolve()
    image = cls.read(path)

    meta_path = cls.meta_path(path)
    if meta_path.exists():
      with open(meta_path, 'r', encoding=cls.ENCODING) as f:
        meta = yaml.safe_load(f)
    else:
      meta = None

    if scale and path.suffix not in ('.csv', '.npy'):
      if meta is None:
        warn(f'메타 정보 파일 ({meta_path.name})이 존재하지 않습니다. '
             f'영상의 밝기 범위를 변경하지 않습니다.')
      else:
        try:
          img_range = (meta['range']['min'], meta['range']['max'])
        except KeyError:
          warn(f'메타 정보 파일 ({meta_path.name})에 영상의 밝기 정보가 없습니다. '
               f'영상의 밝기 범위를 변경하지 않습니다.')
        else:
          image = rescale_intensity(image=image, out_range=img_range)

    return image, meta

  @classmethod
  def save(cls, path: StrPath, array: np.ndarray, check_contrast=False):
    """
    주어진 path의 확장자에 따라 영상 파일 저장

    Parameters
    ----------
    path : Union[str, Path]
        저장 경로. `.npy`, `.csv` 및 `.png` 등 영상 확장자 지정 가능
        (`skimage.io.imsave` 참조).
    array : np.ndarray
        저장할 영상
    check_contrast : bool, optional
        `True`인 경우, 영상 파일 확장자 저장 시 영상의 대비가 너무 낮으면 경고.
    """
    path = Path(path).resolve()
    if path.suffix == cls.NPY_EXT:
      np.save(file=path.as_posix(), arr=array)
    elif path.suffix == cls.CSV_EXT:
      np.savetxt(fname=path.as_posix(), X=array, delimiter=cls.DELIMITER)
    else:
      imsave(fname=path.as_posix(), arr=array, check_contrast=check_contrast)

  @staticmethod
  def _scale_and_save(array: np.ndarray, path: Path, exts: list, dtype='uint8'):
    if not exts:
      return
    if dtype not in ('uint8', 'uint16'):
      raise ValueError

    if dtype == 'uint8':
      image = uint8_image(array)
      for ext in exts:
        imsave(fname=path.with_suffix(ext).as_posix(),
               arr=image,
               check_contrast=False)
    else:
      image = uint16_image(array)
      pil_image = PIL.Image.fromarray(image)

      for ext in exts:
        pil_image.save(fp=path.with_suffix(ext))

  @staticmethod
  def _save_image(array: np.ndarray, path: Path, exts: list):
    for ext in exts:
      imsave(fname=path.with_suffix(ext).as_posix(), arr=array)

  @classmethod
  def save_with_meta(cls,
                     path: StrPath,
                     array: np.ndarray,
                     exts: list,
                     meta: Optional[dict] = None,
                     dtype: Optional[str] = None):
    """
    주어진 path와 각 확장자 (`exts`)에 따라 영상 파일 저장.
    조건에 따라 주어진 메타 정보를 함께 저장하며, 영상 확장자 (`.png` 등) 파일은
    영상 파일 형식에 맞게 픽셀 값의 범위를 조정함.

    Parameters
    ----------
    path : Union[str, Path]
        저장 경로 (확장자는 `exts`에 따라 변경함).
    array : np.ndarray
        저장할 영상.
    exts : list
        저장할 확장자 목록.
    meta : Optional[dict], optional
        영상의 메타데이터 (Exif 정보 등). `None`이 아니면 `{file name}_meta.yaml`
        파일에 메타 정보 저장.
    dtype : Optional[str], optional
        None이면 입력한 array를 그대로 영상 확장자로 저장. 'uint8' 혹은 'uint16'인
        경우, 해당 dtype의 범위로 정규화하고 저장.
    """
    path = Path(path)

    # npy 저장
    if cls.NPY_EXT in exts:
      np.save(file=path.with_suffix(cls.NPY_EXT).as_posix(), arr=array)
      exts.remove(cls.NPY_EXT)

    # csv 저장
    if cls.CSV_EXT in exts:
      np.savetxt(fname=path.with_suffix(cls.CSV_EXT).as_posix(),
                 X=array,
                 delimiter=cls.DELIMITER)
      exts.remove(cls.CSV_EXT)

    # 이미지 파일 저장
    if dtype is None:
      cls._save_image(array=array, path=path, exts=exts)
    else:
      cls._scale_and_save(array=array, path=path, exts=exts, dtype=dtype)

    # 메타 데이터 저장
    if meta is not None:
      meta['range'] = {
          'min': np.around(np.nanmin(array).item(), 4).item(),
          'max': np.around(np.nanmax(array).item(), 4).item()
      }
      meta_path = cls.meta_path(path)
      with open(meta_path, 'w', encoding=cls.ENCODING) as f:
        yaml.safe_dump(meta, f, indent=4, sort_keys=False)


class SegMask:
  _scale = 60

  @classmethod
  def index_to_vis(cls, array: np.ndarray):
    return array.astype(np.uint8) * cls._scale

  @classmethod
  def vis_to_index(cls, array: np.ndarray):
    return (array / cls._scale).astype(np.uint8)
