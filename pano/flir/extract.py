"""deprecated"""

from pathlib import Path
from typing import Tuple
from warnings import warn

import numpy as np

from pano.misc.exif import EXIFTOOL_PATH
from pano.misc.imageio import ImageIO

from ._flir_image_extractor import FlirImageExtractor


class FlirExtractor:

  def __init__(self, path=None, exiftool_path=None, debug=False):
    warn('deprecated')

    if exiftool_path is None:
      exiftool_path = EXIFTOOL_PATH.as_posix()

    self._extractor = FlirImageExtractor(exiftool_path=exiftool_path,
                                         is_debug=debug)

    if path is not None:
      if not Path(path).exists():
        raise FileNotFoundError(path)

      self._extractor.process_image(path)

    self._path = path

  @property
  def extractor(self):
    return self._extractor

  def process_image(self, path):
    path = Path(path).resolve()
    if not path.exists():
      raise FileNotFoundError(path)

    self._extractor.process_image(path.as_posix())
    self._path = path

  def _check_path(self, path):
    if path is not None:
      path = Path(path).resolve()

      if path != self._path:
        self.process_image(path)
        self._path = path

    elif self._path is None:
      raise ValueError('이미지가 지정되지 않음')

  def _ir(self) -> np.ndarray:
    ir = self.extractor.thermal_image_np
    if ir is None:
      raise ValueError('열화상 추출 실패')

    ir = np.round(ir, 4)

    return ir

  def _vis(self) -> np.ndarray:
    vis = self.extractor.rgb_image_np
    if vis is None:
      # `EmbeddedImage` 태그가 없는 등의 이유로 FLIR 실화상 파일 추출에 실패한 경우,
      # 원본 파일을 대신 읽음
      vis = ImageIO.read(self._path)

    return vis

  def extract_data(self, path=None) -> Tuple[np.ndarray, np.ndarray]:
    self._check_path(path)
    ir = self._ir()
    vis = self._vis()

    return ir, vis

  def extract_ir(self, path=None) -> np.ndarray:
    self._check_path(path)

    return self._ir()

  def extract_vis(self, path=None) -> np.ndarray:
    self._check_path(path)

    return self._vis()
