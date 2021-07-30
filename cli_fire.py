"""파노라마 영상처리 CLI"""

import os
import sys

import fire
import skimage.io
from loguru import logger

skimage.io.use_plugin('pil')
sys.path.insert(0, os.path.normpath('./src'))

# pylint: disable=wrong-import-position
import utils


class CLI:
  """
  Thermal Panorama CLI

  Usage: pano_cli.py COMMAND DIRECTORY [OPTIONS]

  COMMAND:
      init       대상 폴더에 설정 파일을 복사
      register   열화상-실화상 정합
      segment    외피 부위 인식
      panorama   파노라마 생성
      correct    파노라마 왜곡 보정
      run        모든 command 순차 실행

  DIRECTORY:
      Target directory

  OPTIONS:
      --default  Use default configuration
      --debug    Show debug message
      loglevel   Log level (default: 20)
  """

  @staticmethod
  def _run(command: str,
           directory: str,
           default=False,
           debug=False,
           loglevel=20):
    level = min(loglevel, (10 if debug else 20))
    utils.set_logger(level=level)

    logger.debug('Directory: {}', directory)
    logger.debug('Command: {}', command)

    # pylint: disable=import-outside-toplevel
    from interface.pano import ThermalPanorama

    if command == 'init':
      default = True

    try:
      tp = ThermalPanorama(directory=directory, default_config=default)
    except (ValueError, OSError) as e:
      logger.error('{}: {}', type(e).__name__, e)
      return

    if command == 'init':
      return

    try:
      getattr(tp, command)()
    except (RuntimeError, ValueError, KeyError, OSError) as e:
      logger.exception(e)

  def init(self, directory: str, debug=False, loglevel=20):
    self._run('init', directory, True, debug, loglevel)

  def register(self, directory: str, default=False, debug=False, loglevel=20):
    self._run('register', directory, default, debug, loglevel)

  def segment(self, directory: str, default=False, debug=False, loglevel=20):
    self._run('segment', directory, default, debug, loglevel)

  def panorama(self, directory: str, default=False, debug=False, loglevel=20):
    self._run('panorama', directory, default, debug, loglevel)

  def correct(self, directory: str, default=False, debug=False, loglevel=20):
    self._run('correct', directory, default, debug, loglevel)

  def run(self, directory: str, default=False, debug=False, loglevel=20):
    self._run('run', directory, default, debug, loglevel)


if __name__ == '__main__':
  fire.Fire(CLI)
