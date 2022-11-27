"""파노라마 영상처리 CLI"""

import click
from loguru import logger

from pano import utils
from pano.interface.common.init import init_project

init_project(qt=False)

commands = {
    'init', 'extract', 'register', 'segment', 'panorama', 'correct', 'run'
}


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.argument('path', required=True)
@click.argument('command', default='run')
@click.option('--default', is_flag=True, help='기본 설정 사용')
@click.option('--config',
              help='설정 파일 경로',
              type=click.Path(exists=True, dir_okay=False))
@click.option('--debug', is_flag=True, help='디버그 메세지 출력')
@click.option('--raise', 'flag_raise', is_flag=True)
@click.option('-l', '--loglevel', default=20, help='로깅 레벨')
def cli(path: str, command: str, config: str, default: bool, debug: bool,
        flag_raise: bool, loglevel: int):
  """
  PATH: 대상 프로젝트 경로

  \b
  COMMAND:
    init      대상 폴더에 설정 파일을 복사
    extract   원본 파일로부터 열·실화상 추출
    register  열화상-실화상 정합
    segment   외피 부위 인식
    panorama  파노라마 생성
    correct   파노라마 왜곡 보정
    run       모든 command 순차 실행
  """
  level = min(loglevel, (10 if debug else 20))
  utils.set_logger(level=level)

  logger.info('Execute "{}" on "{}"', command.upper(), path)
  command = command.lower()

  if command not in commands:
    logger.error('Command는 {} 중 하나여야 합니다.', commands)
    return

  if command == 'init' and not config:
    default = True

  # pylint: disable=import-outside-toplevel
  from pano.interface.pano_project import ThermalPanorama

  try:
    tp = ThermalPanorama(directory=path, default_config=default)
  except (ValueError, OSError) as e:
    logger.exception(e)
    return

  if config:
    logger.debug('update config: {}', config)
    tp.update_config(config)

  if command == 'init':
    return

  fn = getattr(tp, command)
  errors = () if flag_raise else (RuntimeError, ValueError, KeyError, OSError)

  try:
    fn()
  except errors as e:
    logger.exception(e)


if __name__ == '__main__':
  # pylint: disable=no-value-for-parameter
  cli()
