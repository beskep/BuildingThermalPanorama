from collections.abc import Iterable
from pathlib import Path

import numpy as np
import yaml
from loguru import logger

from pano.flir import FlirExif, FlirExtractor

# ruff: noqa: N803, N806
# TODO misc로


def _read_meta(path: Path):
  with path.open('r', encoding='UTF-8-SIG') as f:
    return yaml.safe_load(f)


def _representative(values, name: str, threshold=0.1):
  array = np.array(values)
  std = np.std(array)
  if std > threshold:
    logger.warning('Standard deviation of {}={:.3e}', name, std)

  return np.median(array)


def correct_emissivity(
    image: np.ndarray, meta_files: Iterable[Path], e1: float, e0: float | None = None
):
  meta_list = [_read_meta(x) for x in meta_files]

  if any(
      meta_list[0]['Exif']['CameraModel'] != x['Exif']['CameraModel']
      for x in meta_list[1:]
  ):
    logger.warning('카메라 기종에 차이가 존재합니다. 오차가 발생할 수 있습니다.')

  signal_reflected = _representative(
      [x['signal_reflected'] for x in meta_list], name='signal_reflected'
  )
  if e0 is None:
    e0 = _representative(
        [x['Exif']['Emissivity'] for x in meta_list], name='Emissivity'
    )

  meta = FlirExif.from_dict(meta_list[0]['Exif'])

  return FlirExtractor.correct_emissivity(
      image=image, meta=meta, signal_reflected=signal_reflected, e0=e0, e1=e1
  )


def correct_temperature(ir: np.ndarray, mask: np.ndarray, coord: tuple, T1: float):
  T0 = ir[coord[0], coord[1]]
  if np.isnan(T0) or np.isnan(T1):
    raise ValueError('유효하지 않은 온도입니다.')

  index = mask[coord[0], coord[1]]
  if index not in [1, 2]:
    raise ValueError('벽 또는 창문을 선택해주세요')

  delta = T1 - T0
  ir[mask == index] += delta

  return ir, delta


def summarize(array: np.ndarray):
  arr = array[np.logical_not(np.isnan(array))]

  return {
      'avg': np.nanmean(arr),
      'std': np.nanstd(arr),
      'min': np.nanmin(arr),
      'q1': np.nanquantile(arr, 0.25),
      'median': np.nanmedian(arr),
      'q3': np.nanquantile(arr, 0.75),
      'max': np.nanmax(arr),
  }
