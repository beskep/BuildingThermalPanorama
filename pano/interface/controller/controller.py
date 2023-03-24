import multiprocessing as mp
from pathlib import Path
from typing import Optional

from loguru import logger

from pano import utils
from pano.interface.mbq import QtCore
from pano.interface.mbq import QtGui


def log(message: str):
  idx = message.find('|')

  if idx == -1:
    level = 'DEBUG'
  else:
    level = message[:idx].upper()
    message = message[(idx + 1):]

  logger.log(level, message)


def path2uri(path):
  return 'file:///' + Path(path).as_posix()


def uri2path(url: str):
  return Path(url.removeprefix('file:///'))


def producer(queue: mp.Queue, directory, command: str, loglevel: int):
  utils.set_logger(loglevel)

  # pylint: disable=import-outside-toplevel
  from pano.interface.pano_project import ThermalPanorama

  try:
    tp = ThermalPanorama(directory=directory)
  except (ValueError, OSError) as e:
    queue.put(f'{type(e).__name__}: {e}')
    return

  logger.info('Run {} command', command)

  try:
    fn = getattr(tp, f'{command}_generator', None)
    if fn is not None:
      for r in fn():
        queue.put(r)
    else:
      fn = getattr(tp, command)
      queue.put(fn())
  except (ValueError, RuntimeError, OSError) as e:
    logger.exception(e)
    queue.put(f'{type(e).__name__}: {e}')

  queue.put(1.0)


class Consumer(QtCore.QThread):
  update = QtCore.Signal(float)  # 진행률 [0, 1] 업데이트
  done = QtCore.Signal()  # 작업 종료 signal (진행률 >= 1.0)
  fail = QtCore.Signal(str)  # 에러 발생 signal. 에러 메세지 emit

  def __init__(self) -> None:
    super().__init__()
    self._queue: Optional[mp.Queue] = None

  @property
  def queue(self):
    return self._queue

  @queue.setter
  def queue(self, value):
    self._queue = value

  def run(self):
    if self.queue is None:
      raise ValueError('queue not set')

    while True:
      if not self.queue.empty():
        value = self.queue.get()

        if isinstance(value, str):
          self.fail.emit(value)
          break

        if value is None or value >= 1.0:
          self.done.emit()
          self.quit()
          break

        self.update.emit(value)


class Window:

  def __init__(self, window: QtGui.QWindow) -> None:
    self._window = window

  def pb_value(self, value: float):
    return self._window.pb_value(value)

  def pb_state(self, indeterminate: bool):
    self._window.pb_state(indeterminate)

  def popup(self, title: str, message: str, timeout=2000):
    logger.debug('[Popup] {}: {}', title, message)
    self._window.popup(title, message, timeout)

    ok = title.lower() != 'error'
    utils.play_sound(ok=ok)

    if ok:
      msg = message.replace('\n', ' ')
      self._window.status_message(f'[{title}] {msg}')
