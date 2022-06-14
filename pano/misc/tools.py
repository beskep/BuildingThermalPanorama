"""기타 영상 처리 함수들"""

import dataclasses as dc
from enum import IntEnum
from typing import List, Optional, Tuple, Union

import cv2 as cv
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray
from skimage.color import rgba2rgb
from skimage.exposure import equalize_hist
from skimage.exposure import rescale_intensity
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


def erode(image: np.ndarray, iterations=1) -> np.ndarray:
  kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
  eroded = cv.erode(image, kernel, iterations=iterations)

  return eroded


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
      if mask.ndim != 2
  """
  if mask.ndim != 2:
    raise ValueError

  if morphology_open:
    kernel = np.ones(shape=(3, 3), dtype=np.uint8)
    mask_ = cv.morphologyEx(src=mask.astype(np.uint8),
                            op=cv.MORPH_OPEN,
                            kernel=kernel)
  else:
    mask_ = mask

  xx = _mask_range(mask=mask_, axis=0)
  yy = _mask_range(mask=mask_, axis=1)

  return xx + yy


@dc.dataclass
class CropRange:
  x_min: int
  x_max: int
  y_min: int
  y_max: int
  image_shape: tuple

  cropped: bool = dc.field(init=False)

  def __post_init__(self):
    self.cropped = ((self.x_min > 0 or self.x_max < self.image_shape[1]) and
                    (self.y_min > 0 or self.y_min < self.image_shape[0]))

  def as_tuple(self):
    return (self.x_min, self.x_max, self.y_min, self.y_max)

  def crop(self, image: np.ndarray, strict=True):
    if image.shape[:2] != self.image_shape[:2]:
      msg = (f'CropRange image shape {self.image_shape[:2]} '
             f'!= Input image shape {image.shape[:2]}')

      if strict:
        raise ValueError(msg)

      logger.warning(msg)
      image = resize(image,
                     output_shape=self.image_shape[:2],
                     preserve_range=True)

    return image[self.y_min:self.y_max, self.x_min:self.x_max]


def crop_mask(mask: np.ndarray,
              morphology_open=True) -> Tuple[CropRange, np.ndarray]:
  bbox = mask_bbox(mask=mask.astype(bool), morphology_open=morphology_open)
  cr = CropRange(*bbox, mask.shape[:2])

  if cr.cropped:
    cropped = cr.crop(mask)
  else:
    cropped = mask

  return cr, cropped


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


def gray_image(image: np.ndarray) -> np.ndarray:
  """
  RGB, RGBA 영상을 gray scale로 변환.
  2채널 흑백 영상은 원본 그대로 반환.

  Parameters
  ----------
  image : np.ndarray
      target image

  Returns
  -------
  np.ndarray

  Raises
  ------
  ValueError
      if image.ndim not in {2, 3}
  """
  if image.ndim == 3:
    if image.shape[2] == 4:
      image = rgba2rgb(image)

    return rgb2gray(image)

  if image.ndim != 2:
    raise ValueError

  return image


def _check_and_prep(image1: np.ndarray, image2: np.ndarray, normalize: bool,
                    eq_hist: bool) -> Tuple[np.ndarray, np.ndarray]:
  image1 = gray_image(image1)
  image2 = gray_image(image2)

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


def prep_compare_images(
    image1: np.ndarray,
    image2: np.ndarray,
    norm=False,
    eq_hist=True,
    method: Union[str, List[str]] = 'checkerboard',
    n_tiles=(8, 8)
) -> Union[np.ndarray, List[np.ndarray]]:
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
  method : Union[str, List[str]], optional
      `skimage.exposure.compare_images` 옵션.
      list로 주어지면 각 옵션을 적용한 비교 영상의 list 반환.
  n_tiles : tuple, optional
      Checkerboard 타일 개수

  Returns
  -------
  Union[np.ndarray, List[np.ndarray]]

  Raises
  ------
  ValueError
      두 영상의 해상도 (shape)이 다르거나 2차원이 아닌 경우
  """
  img1, img2 = _check_and_prep(image1=image1,
                               image2=image2,
                               normalize=norm,
                               eq_hist=eq_hist)
  if isinstance(method, str):
    res = compare_images(img1, img2, method=method, n_tiles=n_tiles)
  else:
    res = []
    for m in method:
      image = compare_images(img1, img2, method=m, n_tiles=n_tiles)
      res.append(image)

  return res


def prep_compare_fig(images: Tuple[np.ndarray, np.ndarray],
                     titles=('Image 1', 'Image 2', 'Compare (checkerboard)',
                             'Compare (difference)'),
                     norm=False,
                     eq_hist=True,
                     n_tiles=(8, 8),
                     cmap=None):
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
  cmap : Optional[str, Colormap]
      Colormap

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

  axes[0, 0].imshow(images[0], cmap=cmap)
  axes[0, 1].imshow(images[1], cmap=cmap)

  cb = compare_images(img1, img2, method='checkerboard', n_tiles=n_tiles)
  diff = compare_images(img1, img2, method='diff')

  axes[1, 0].imshow(cb, cmap=cmap)
  axes[1, 1].imshow(diff, cmap=cmap)

  for ax, title in zip(axes.ravel(), titles):
    ax.set_axis_off()
    ax.set_title(title)

  fig.tight_layout()

  return fig, axes


class Interpolation(IntEnum):
  NearestNeighbor = 0
  BiLinear = 1
  BiQuadratic = 2
  BiCubic = 3
  BiQuartic = 4
  BiQuintic = 5


def limit_image_size(image: np.ndarray,
                     limit: int,
                     order=Interpolation.BiCubic,
                     anti_aliasing=True) -> np.ndarray:
  max_shape = np.max(image.shape[:2]).astype(float)
  if max_shape <= limit:
    return image

  dtype = image.dtype

  ratio = limit / max_shape
  target_shape = (int(image.shape[0] * ratio), int(image.shape[1] * ratio))

  resized = resize(image=image.astype(np.float32),
                   order=int(order),
                   output_shape=target_shape,
                   preserve_range=True,
                   anti_aliasing=anti_aliasing)

  return resized.astype(dtype)


class SegMask:
  scale = 80

  @classmethod
  def index_to_vis(cls, array: np.ndarray):
    return array.astype(np.uint8) * cls.scale

  @classmethod
  def vis_to_index(cls, array: np.ndarray):
    return np.round(array / cls.scale).astype(np.uint8)


def reject_outliers(data: np.ndarray, m=1.5):
  data = data[~np.isnan(data)]
  q1, q3 = np.quantile(data, q=(0.25, 0.75))
  iqr = q3 - q1

  lower = q1 - m * iqr
  upper = q3 + m * iqr

  return data[(lower <= data) & (data <= upper)]
