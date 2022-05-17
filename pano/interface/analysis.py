from pathlib import Path
from typing import Iterable, Optional

from loguru import logger
import numpy as np
import yaml

from pano.flir import FlirExif
from pano.flir import FlirExtractor


def _read_meta(path: Path):
  with path.open('r', encoding='UTF-8-SIG') as f:
    meta = yaml.safe_load(f)

  return meta


def _representative(values, name: str, threshold=0.1):
  array = np.array(values)
  std = np.std(array)
  if std > threshold:
    logger.warning('Standard deviation of {}={:.3e}', name, std)

  return np.median(array)


def correct_emissivity(image: np.ndarray,
                       meta_files: Iterable[Path],
                       e1: float,
                       e0: Optional[float] = None):
  meta_list = [_read_meta(x) for x in meta_files]

  if any(meta_list[0]['Exif']['CameraModel'] != x['Exif']['CameraModel']
         for x in meta_list[1:]):
    logger.warning('카메라 기종에 차이가 존재합니다. 오차가 발생할 수 있습니다.')

  signal_reflected = _representative([x['signal_reflected'] for x in meta_list],
                                     name='signal_reflected')
  if e0 is None:
    e0 = _representative([x['Exif']['Emissivity'] for x in meta_list],
                         name='Emissivity')

  meta = FlirExif.from_dict(meta_list[0]['Exif'])

  return FlirExtractor.correct_emissivity(image=image,
                                          meta=meta,
                                          signal_reflected=signal_reflected,
                                          e0=e0,
                                          e1=e1)


def correct_temperature(ir: np.ndarray, mask: np.ndarray, coord: tuple,
                        T1: float):
  T0 = ir[coord[0], coord[1]]
  if np.isnan(T0) or np.isnan(T1):
    raise ValueError('유효하지 않은 온도입니다.')

  if mask[coord[0], coord[1]] != 1:
    raise ValueError('벽을 선택해주세요')

  ir[mask == 1] += (T1 - T0)

  return ir
