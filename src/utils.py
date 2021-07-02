import sys
from pathlib import Path
from typing import Optional, Union

from loguru import logger
from rich.console import Console
from rich.logging import RichHandler

SRC_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SRC_DIR.parent
DATA_DIR = ROOT_DIR.joinpath('data')

_SRC_DIR = SRC_DIR.as_posix()
if _SRC_DIR not in sys.path:
  sys.path.insert(0, _SRC_DIR)

console = Console()


def set_logger(level: Optional[Union[int, str]] = None):
  logger.remove()

  if level is None:
    if any('debug' in x.lower() for x in sys.argv):
      level = 'DEBUG'
    else:
      level = 'INFO'

  rich_handler = RichHandler(show_time=True, console=console)
  logger.add(rich_handler, level=level, format='{message}', enqueue=True)
  logger.add('pano.log',
             level='DEBUG',
             rotation='1 week',
             retention='1 month',
             encoding='UTF-8-SIG',
             enqueue=True)
