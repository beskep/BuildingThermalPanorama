"""경로 및 로거 설정"""

from logging import LogRecord
from os import PathLike
from pathlib import Path
import sys
from typing import Optional, Union

try:
  import winsound
except ImportError:
  winsound = None

from loguru import logger
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import track as _track
from rich.theme import Theme


def is_frozen():
  return getattr(sys, 'frozen', False)


class DIR:
  if is_frozen():
    ROOT = Path(sys.executable).parent.resolve()
  else:
    ROOT = Path(__file__).parents[1].resolve()

  SRC = ROOT.joinpath('pano')
  RESOURCE = ROOT.joinpath('resource')


class _Handler(RichHandler):
  LVLS = {
      'TRACE': 5,
      'DEBUG': 10,
      'INFO': 20,
      'SUCCESS': 25,
      'WARNING': 30,
      'ERROR': 40,
      'CRITICAL': 50
  }
  BLANK_NO = 21
  _NEW_LVLS = {5: 'TRACE', 25: 'SUCCESS', BLANK_NO: ''}

  def emit(self, record: LogRecord) -> None:
    if record.levelno in self._NEW_LVLS:
      record.levelname = self._NEW_LVLS[record.levelno]

    return super().emit(record)


StrPath = Union[str, PathLike]
console = Console(theme=Theme({'logging.level.success': 'blue'}))
_handler = _Handler(console=console, log_time_format='[%X]')


def set_logger(level: Union[int, str] = 20):
  if isinstance(level, str):
    try:
      level = _Handler.LVLS[level.upper()]
    except KeyError as e:
      raise KeyError(f'`{level}` not in {list(_Handler.LVLS.keys())}') from e

  if getattr(logger, 'lvl', -1) != level:
    logger.remove()

    logger.add(_handler,
               level=level,
               format='{message}',
               backtrace=False,
               enqueue=True)
    logger.add('pano.log',
               level='DEBUG',
               rotation='1 month',
               retention='1 year',
               encoding='UTF-8-SIG',
               enqueue=True)

    setattr(logger, 'lvl', level)

  try:
    logger.level('BLANK')
  except ValueError:
    # 빈 칸 표시하는 'BLANK' level 새로 등록
    logger.level(name='BLANK', no=_Handler.BLANK_NO)


def track(sequence,
          description='Working...',
          total: Optional[float] = None,
          transient=True,
          **kwargs):
  """Track progress on console by iterating over a sequence."""
  return _track(sequence=sequence,
                description=description,
                total=total,
                console=console,
                transient=transient,
                **kwargs)


def ptrack(sequence,
           description='Working...',
           total: Optional[float] = None,
           transient=True,
           **kwargs):
  """
  Track progress on console by iterating over a sequence.
  Yield progress ratio and value.
  """
  if total is None:
    total = len(sequence)

  for idx, value in _track(sequence=enumerate(sequence),
                           description=description,
                           total=total,
                           console=console,
                           transient=transient,
                           **kwargs):
    yield (idx + 1) / total, value


def play_sound(ok=True):
  # TODO 모듈 교체 (pygame?)
  if hasattr(winsound, 'MessageBeep'):
    t = winsound.MB_OK if ok else winsound.MB_ICONHAND
    winsound.MessageBeep(t)
