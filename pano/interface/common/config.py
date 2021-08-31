"""파노라마 영상처리 설정 로드"""

from pathlib import Path
from shutil import copy2

from loguru import logger
from omegaconf import DictConfig
from omegaconf import OmegaConf
from yaml.error import YAMLError

from pano.utils import DIR

CONFIG_FNAME = 'config.yaml'
DEFAULT_CONFIG_PATH = DIR.RESOURCE.joinpath(CONFIG_FNAME)


def set_config(directory: Path, read_only=True, default=False) -> DictConfig:
  """
  config 파일 로드하고 유효한 config 설정 파일 저장.

  Parameters
  ----------
  directory : Path
      프로그램을 실행하는 working directory
  read_only : bool
      설정의 read only 여부
  default : bool
      `True`이면 `directory`에 저장된 설정을 무시함

  Returns
  -------
  DictConfig
  """
  config = OmegaConf.load(DEFAULT_CONFIG_PATH)
  config_path = directory.joinpath(CONFIG_FNAME)

  if not default and config_path.exists():
    try:
      wd_config = OmegaConf.load(config_path)
    except YAMLError:
      logger.error(
          '`{}` 파일의 형식이 올바르지 않습니다. 기본 설정을 사용합니다.',
          config_path,
      )
    else:
      # 기본 설정에 working dir의 설정을 덮어씌움
      config = OmegaConf.merge(config, wd_config)
  else:
    # 기본 설정 파일 복사
    copy2(src=DEFAULT_CONFIG_PATH, dst=config_path)

  if read_only:
    OmegaConf.set_readonly(config, True)

  return config
