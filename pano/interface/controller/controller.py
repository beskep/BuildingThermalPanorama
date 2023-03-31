import multiprocessing as mp
from pathlib import Path
from typing import Iterable, Optional

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


def _command(queue: mp.Queue, tp, command: str, step: int, count: float):
  fn = getattr(tp, f'{command}_generator', None)
  queue.put(fn is None)  # generator가 없으면 pb_state(True)

  try:
    if fn is None:
      getattr(tp, command)()
      queue.put(False)
      queue.put((step + 1) / count)
    else:
      for r in fn():
        queue.put((step + r) / count)
  except (ValueError, RuntimeError, OSError) as e:
    logger.exception(e)
    queue.put(f'{type(e).__name__}: {e}')
    return False

  return True


def producer(queue: mp.Queue,
             directory,
             commands: str | Iterable[str],
             loglevel: int,
             logname='pano'):
  utils.set_logger(loglevel, logname)

  # pylint: disable=import-outside-toplevel
  from pano.interface.pano_project import ThermalPanorama

  try:
    tp = ThermalPanorama(directory=directory)
  except (ValueError, OSError) as e:
    queue.put(f'{type(e).__name__}: {e}')
    return

  if isinstance(commands, str):
    commands = (commands,)
  else:
    commands = tuple(commands)

  count = float(len(commands))
  queue.put(0.0)

  for step, cmd in enumerate(commands):
    logger.info('Run command "{}"', cmd)
    if not _command(queue=queue, tp=tp, command=cmd, step=step, count=count):
      queue.put(0.0)
      break
  else:
    queue.put(1.0)

  queue.put(False)


class Consumer(QtCore.QThread):
  state = QtCore.Signal(bool)
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

        if isinstance(value, bool):
          self.state.emit(value)
          continue

        assert isinstance(value, float)
        if value >= 1.0:
          self.done.emit()
          break

        self.update.emit(value)

    self.quit()


class Window:

  def __init__(self, window: QtGui.QWindow) -> None:
    self._window = window

  def pb_value(self, value: float):
    return self._window.pb_value(value)

  def pb_state(self, indeterminate: bool):
    self._window.pb_state(indeterminate)

  def popup(self, title: str, message: str, timeout: Optional[int] = None):
    ok = title.lower() != 'error'
    if not timeout:
      timeout = 2000 if ok else 10000

    logger.debug('[Popup] {}: {}', title, message)
    self._window.popup(title, message, timeout)
    utils.play_sound(ok=ok)

    if ok:
      msg = message.replace('\n', ' ')
      self._window.status_message(f'[{title}] {msg}')

  def error_popup(self, message: str):
    self.pb_state(False)
    self.pb_value(0.0)
    self.popup(title='Error', message=message)
