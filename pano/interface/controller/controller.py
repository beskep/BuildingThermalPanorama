from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from PyQt5 import QtCore, QtGui

from pano import utils

if TYPE_CHECKING:
  import multiprocessing as mp
  from collections.abc import Iterable


def log(message: str):
  if (i := message.find('|')) == -1:
    level = 'DEBUG'
  else:
    level = message[:i].upper()
    message = message[(i + 1) :]

  logger.log(level, message)


def path2uri(path):
  return 'file:///' + Path(path).as_posix()


def uri2path(uri: str):
  return Path(uri.removeprefix('file:///'))


def _command(queue: mp.Queue, tp, command: str, step: int, count: float):
  fn = getattr(tp, f'{command}_generator', None)
  queue.put(fn is None)  # generator가 없으면 pb_state(True)

  try:
    if fn is None:
      getattr(tp, command)()
      queue.put(obj=False)
      queue.put((step + 1) / count)
    else:
      for r in fn():
        queue.put((step + r) / count)
  except (ValueError, RuntimeError, OSError) as e:
    logger.exception(e)
    queue.put(f'{type(e).__name__}: {e}')
    return False

  return True


def producer(
  queue: mp.Queue,
  directory,
  commands: str | Iterable[str],
  loglevel: int,
  logname='pano',
):
  utils.set_logger(loglevel, logname)

  # pylint: disable=import-outside-toplevel
  from pano.interface.pano_project import ThermalPanorama  # noqa: PLC0415

  try:
    tp = ThermalPanorama(directory=directory)
  except (ValueError, OSError) as e:
    queue.put(f'{type(e).__name__}: {e}')
    return

  commands = (commands,) if isinstance(commands, str) else tuple(commands)
  count = float(len(commands))
  queue.put(0.0)

  for step, cmd in enumerate(commands):
    logger.info('Run command "{}"', cmd)
    if not _command(queue=queue, tp=tp, command=cmd, step=step, count=count):
      queue.put(0.0)
      break
  else:
    queue.put(1.0)

  queue.put(obj=False)


class Consumer(QtCore.QThread):
  state = QtCore.Signal(bool)
  update = QtCore.Signal(float)  # 진행률 [0, 1] 업데이트
  done = QtCore.Signal()  # 작업 종료 signal (진행률 >= 1.0)
  fail = QtCore.Signal(str)  # 에러 발생 signal. 에러 메세지 emit

  def __init__(self) -> None:
    super().__init__()
    self._queue: mp.Queue | None = None

  @property
  def queue(self):
    return self._queue

  @queue.setter
  def queue(self, value):
    self._queue = value

  def run(self):
    if self.queue is None:
      msg = 'queue not set'
      raise ValueError(msg)

    while True:
      if not self.queue.empty():
        value = self.queue.get()

        if isinstance(value, str):
          self.fail.emit(value)
          break

        if isinstance(value, bool):
          self.state.emit(value)
          continue

        assert isinstance(value, float)
        if value >= 1:
          self.done.emit()
          break

        self.update.emit(value)

    self.quit()


class Window:
  def __init__(self, window: QtGui.QWindow) -> None:
    self._window = window

  def pb_value(self, value: float):
    return self._window.pb_value(value)

  def pb_state(self, indeterminate: bool):  # noqa: FBT001 (`update` signal)
    self._window.pb_state(indeterminate)

  def popup(self, title: str, message: str, timeout: int | None = None):
    ok = title.lower() != 'error'
    if not timeout:
      timeout = 2000 if ok else 10000

    logger.debug('[Popup] {}: {}', title, message)
    self._window.popup(title, message, timeout)
    utils.play_sound(ok=ok)

    if ok:
      msg = message.replace('\n', ' ')
      self._window.status_message(f'[{title}] {msg}')

  def error_popup(self, e: str | Exception):
    if isinstance(e, Exception):
      logger.exception(e)

    self.pb_state(indeterminate=False)
    self.pb_value(0.0)
    self.popup(title='Error', message=str(e))
