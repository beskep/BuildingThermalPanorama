from pathlib import Path
from shutil import copy2

import utils

from loguru import logger
from omegaconf import DictConfig, OmegaConf
from yaml.error import YAMLError

CONFIG_FNAME = 'config.yaml'
DEFAULT_CONFIG_PATH = utils.SRC_DIR.joinpath(CONFIG_FNAME)


def read_config(wd: Path, read_only=True) -> DictConfig:
  """
  config 파일 로드

  Parameters
  ----------
  wd : Path
      프로그램을 실행하는 working directory

  Returns
  -------
  DictConfig
      설정
  """
  config = OmegaConf.load(DEFAULT_CONFIG_PATH)
  config_path = wd.joinpath(CONFIG_FNAME)

  if config_path.exists():
    try:
      wd_config = OmegaConf.load(config_path)
    except YAMLError:
      logger.error('`{}` 파일의 형식이 올바르지 않습니다. 기본 설정을 사용합니다.', config_path)
    else:
      # 기본 설정에 working dir의 설정을 덮어씌움
      config = OmegaConf.merge(config, wd_config)
  else:
    # 기본 설정 파일 복사
    copy2(src=DEFAULT_CONFIG_PATH, dst=config_path)

  if read_only:
    OmegaConf.set_readonly(config, True)

  return config
