"""경로 및 로거 설정"""

import sys
from collections.abc import Iterable, Sequence
from logging import LogRecord
from operator import length_hint
from pathlib import Path
from typing import ClassVar, TypeVar

import rich

try:
  import winsound
except ImportError:
  winsound = None  # type: ignore[assignment]

from loguru import logger
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

  SRC = ROOT / 'pano'
  RESOURCE = ROOT / 'resource'
  QT = ROOT / 'qt'


class _Handler(RichHandler):
  LVLS: ClassVar[dict[str, int]] = {
    'TRACE': 5,
    'DEBUG': 10,
    'INFO': 20,
    'SUCCESS': 25,
    'WARNING': 30,
    'ERROR': 40,
    'CRITICAL': 50,
  }
  BLANK_NO = 21
  _NEW_LVLS: ClassVar[dict[int, str]] = {5: 'TRACE', 25: 'SUCCESS', BLANK_NO: ''}

  def emit(self, record: LogRecord) -> None:
    if record.levelno in self._NEW_LVLS:
      record.levelname = self._NEW_LVLS[record.levelno]

    return super().emit(record)


console = rich.get_console()
console.push_theme(Theme({'logging.level.success': 'bold blue'}))
_handler = _Handler(console=console, log_time_format='[%X]')


def set_logger(level: int | str = 20, name='pano'):
  if isinstance(level, str):
    try:
      level = _Handler.LVLS[level.upper()]
    except KeyError as e:
      msg = f'`{level}` not in {list(_Handler.LVLS.keys())}'
      raise KeyError(msg) from e

  logger.remove()
  logger.add(_handler, level=level, format='{message}', backtrace=False, enqueue=True)
  logger.add(
    f'{name}.log',
    level='DEBUG',
    rotation='1 month',
    retention='1 year',
    encoding='UTF-8-SIG',
    enqueue=True,
  )

  try:
    logger.level('BLANK')
  except ValueError:
    # 빈 칸 표시하는 'BLANK' level 새로 등록
    logger.level(name='BLANK', no=_Handler.BLANK_NO)


T = TypeVar('T')


def ptrack(
  sequence: Iterable[T] | Sequence[T],
  description='Working...',
  total: float | None = None,
  *,
  transient=True,
  **kwargs,
):
  if total is None:
    total = length_hint(sequence)

  if not total:
    msg = f'Invalid total value: {total}'
    raise ValueError(msg)

  for idx, value in _track(
    sequence=enumerate(sequence),
    description=description,
    total=total,
    console=console,
    transient=transient,
    **kwargs,
  ):
    yield (idx + 1) / total, value


def play_sound(*, ok=True):
  if hasattr(winsound, 'MessageBeep'):
    t = winsound.MB_OK if ok else winsound.MB_ICONHAND
    winsound.MessageBeep(t)
