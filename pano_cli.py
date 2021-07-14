"""
파노라마 영상처리 CLI
"""

import os
import sys

import cv2
import fire
import SimpleITK
import skimage
import skimage.io
import tensorflow
from loguru import logger

skimage.io.use_plugin('pil')
sys.path.insert(0, os.path.normpath('./src'))

# pylint: disable=wrong-import-position
import utils

from interface.pano import ThermalPanorama


class _ThermalPanorama(ThermalPanorama):

  def __init__(self, directory, debug=False) -> None:
    """
    Parameters
    ----------
    directory : str
        Working directory
    debug : bool, optional
        If true, log debug message
    """
    self._level = 'DEBUG' if debug else 'INFO'
    utils.set_logger(level=self._level)

    super().__init__(directory)

  def register(self):
    try:
      super().register()
    except (RuntimeError, ValueError, KeyError, OSError) as e:
      logger.exception(e)

  def segment(self):
    try:
      super().segment()
    except (RuntimeError, ValueError, KeyError, OSError) as e:
      logger.exception(e)

  def panorama(self):
    try:
      super().panorama()
    except (RuntimeError, ValueError, KeyError, OSError) as e:
      logger.exception(e)


if __name__ == '__main__':
  fire.Fire(_ThermalPanorama)
