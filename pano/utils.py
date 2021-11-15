"""경로 및 로거 설정"""

from logging import LogRecord
from os import PathLike
from pathlib import Path
import sys
from typing import Optional, Union

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


StrPath = Union[str, PathLike]

console = Console(theme=Theme({'logging.level.success': 'blue'}))
_BLANK_NO = 21


class _Handler(RichHandler):
  _levels = {5: 'TRACE', 25: 'SUCCESS', _BLANK_NO: ''}

  def emit(self, record: LogRecord) -> None:
    if record.levelno in self._levels:
      record.levelname = self._levels[record.levelno]

    return super().emit(record)


_handler = _Handler(console=console, log_time_format='[%X]')


def set_logger(level: Union[int, str] = 20):
  if isinstance(level, str):
    levels = {
        'TRACE': 5,
        'DEBUG': 10,
        'INFO': 20,
        'SUCCESS': 25,
        'WARNING': 30,
        'ERROR': 40,
        'CRITICAL': 50
    }
    try:
      level = levels[level.upper()]
    except KeyError as e:
      raise KeyError('`{}` not in {}'.format(level, set(levels.keys()))) from e

  if getattr(logger, 'lvl', -1) != level:
    logger.remove()

    logger.add(_handler,
               level=level,
               format='{message}',
               backtrace=False,
               enqueue=True)
    logger.add('pano.log',
               level='DEBUG',
               rotation='1 week',
               retention='1 month',
               encoding='UTF-8-SIG',
               enqueue=True)

    setattr(logger, 'lvl', level)

  try:
    logger.level('BLANK')
  except ValueError:
    # 빈 칸 표시하는 'BLANK' level 새로 등록
    logger.level(name='BLANK', no=_BLANK_NO)


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
