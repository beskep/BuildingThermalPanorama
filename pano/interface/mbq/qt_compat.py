# type: ignore  # noqa: PGH003
# ruff: noqa: ARG001 PLW0603 PLC0415 N802
# pylint: disable-all
"""
Qt binding and backend selector.

The selection logic is as follows:
- if any of PyQt5, PySide2, PyQt4 or PySide have already been imported
  (checked in that order), use it;
- otherwise, if the QT_API environment variable (used by Enthought) is set, use
  it to determine which binding to use (but do not change the backend based on
  it; i.e. if the Qt5Agg backend is requested but QT_API is set to "pyqt4",
  then actually use Qt5 with PyQt5 or PySide2 (whichever can be imported);
- otherwise, use whatever the rcParams indicate.

Support for PyQt4 is deprecated.

(Copied and updated from matplotlib)
"""

import os
import sys
from distutils.version import LooseVersion

import matplotlib as mpl

QT_API_PYQT5 = 'PyQt5'
QT_API_PYSIDE2 = 'PySide2'
QT_API_PYQTv2 = 'PyQt4v2'
QT_API_PYSIDE = 'PySide'
QT_API_PYQT = 'PyQt4'  # Use the old sip v1 API (Py3 defaults to v2).
QT_API_ENV = os.environ.get('QT_API')
# Mapping of QT_API_ENV to requested binding.  ETS does not support PyQt4v1.
# (https://github.com/enthought/pyface/blob/master/pyface/qt/__init__.py)
_ETS = {
  'pyqt5': QT_API_PYQT5,
  'pyside2': QT_API_PYSIDE2,
  'pyqt': QT_API_PYQTv2,
  'pyside': QT_API_PYSIDE,
  None: None,
}
# First, check if anything is already imported.
if 'PyQt5.QtCore' in sys.modules:
  _QT_API = QT_API_PYQT5
elif 'PySide2.QtCore' in sys.modules:
  _QT_API = QT_API_PYSIDE2
elif 'PyQt4.QtCore' in sys.modules:
  _QT_API = QT_API_PYQTv2
elif 'PySide.QtCore' in sys.modules:
  _QT_API = QT_API_PYSIDE
# Otherwise, check the QT_API environment variable (from Enthought).  This can
# only override the binding, not the backend (in other words, we check that the
# requested backend actually matches).
elif mpl.rcParams['backend'] in {'Qt5Agg', 'Qt5Cairo', 'QtQuickAgg'}:
  _QT_API = _ETS[QT_API_ENV] if QT_API_ENV in {'pyqt5', 'pyside2'} else None
elif mpl.rcParams['backend'] in {'Qt4Agg', 'Qt4Cairo'}:
  _QT_API = _ETS[QT_API_ENV] if QT_API_ENV in {'pyqt4', 'pyside'} else None
# A non-Qt backend was selected but we still got there (possible, e.g., when
# fully manually embedding Matplotlib in a Qt app without using pyplot).
else:
  try:
    _QT_API = _ETS[QT_API_ENV]
  except KeyError as err:
    msg = (
      'The environment variable QT_API has the unrecognized value {!r};'
      "valid values are 'pyqt5', 'pyside2', 'pyqt', and "
      "'pyside'"
    )
    raise RuntimeError(msg) from err


def _setup_pyqt5():
  global \
    QtCore, \
    QtGui, \
    QtWidgets, \
    QtQuick, \
    QtQml, \
    __version__, \
    is_pyqt5, \
    _isdeleted, \
    _devicePixelRatio, \
    _setDevicePixelRatio, \
    _getSaveFileName

  if _QT_API == QT_API_PYQT5:
    import sip
    from PyQt5 import QtCore, QtGui, QtQml, QtQuick, QtWidgets

    __version__ = QtCore.PYQT_VERSION_STR
    QtCore.Signal = QtCore.pyqtSignal
    QtCore.Slot = QtCore.pyqtSlot
    QtCore.Property = QtCore.pyqtProperty
    _isdeleted = sip.isdeleted
  elif _QT_API == QT_API_PYSIDE2:
    import shiboken2
    from PySide2 import QtCore, QtGui, QtQml, QtQuick, QtWidgets, __version__

    def _isdeleted(obj):
      return not shiboken2.isValid(obj)

  else:
    msg = "Unexpected value for the 'backend.qt5' rcparam"
    raise ValueError(msg)
  _getSaveFileName = QtWidgets.QFileDialog.getSaveFileName

  def is_pyqt5():
    return True

  # self.devicePixelRatio() returns 0 in rare cases
  def _devicePixelRatio(obj):
    return obj.devicePixelRatio() or 1

  def _setDevicePixelRatio(obj, factor):
    obj.setDevicePixelRatio(factor)


def _setup_pyqt4():  # noqa: C901
  global \
    QtCore, \
    QtGui, \
    QtWidgets, \
    __version__, \
    is_pyqt5, \
    _isdeleted, \
    _devicePixelRatio, \
    _setDevicePixelRatio, \
    _getSaveFileName

  def _setup_pyqt4_internal(api):
    global QtCore, QtGui, QtWidgets, __version__, is_pyqt5, _isdeleted, _getSaveFileName  # noqa: PLW0602
    # List of incompatible APIs:
    # http://pyqt.sourceforge.net/Docs/PyQt4/incompatible_apis.html
    _sip_apis = [
      'QDate',
      'QDateTime',
      'QString',
      'QTextStream',
      'QTime',
      'QUrl',
      'QVariant',
    ]
    try:
      import sip
    except ImportError:
      pass
    else:
      for _sip_api in _sip_apis:
        try:  # noqa: SIM105
          sip.setapi(_sip_api, api)
        except ValueError:
          pass
    import sip  # Always succeeds *after* importing PyQt4.
    from PyQt4 import QtCore, QtGui

    __version__ = QtCore.PYQT_VERSION_STR
    # PyQt 4.6 introduced getSaveFileNameAndFilter:
    # https://riverbankcomputing.com/news/pyqt-46
    if __version__ < LooseVersion('4.6'):
      msg = 'PyQt<4.6 is not supported'
      raise ImportError(msg)
    QtCore.Signal = QtCore.pyqtSignal
    QtCore.Slot = QtCore.pyqtSlot
    QtCore.Property = QtCore.pyqtProperty
    _isdeleted = sip.isdeleted
    _getSaveFileName = QtGui.QFileDialog.getSaveFileNameAndFilter

  if QT_API_PYQTv2 == _QT_API:
    _setup_pyqt4_internal(api=2)
  elif _QT_API == QT_API_PYSIDE:
    import shiboken
    from PySide import QtCore, QtGui, __version__, __version_info__

    # PySide 1.0.3 fixed the following:
    # https://srinikom.github.io/pyside-bz-archive/809.html
    if __version_info__ < (1, 0, 3):
      msg = 'PySide<1.0.3 is not supported'
      raise ImportError(msg)

    def _isdeleted(obj):
      return not shiboken.isValid(obj)

    _getSaveFileName = QtGui.QFileDialog.getSaveFileName
  elif _QT_API == QT_API_PYQT:
    _setup_pyqt4_internal(api=1)
  else:
    msg = "Unexpected value for the 'backend.qt4' rcparam"
    raise ValueError(msg)
  QtWidgets = QtGui

  def is_pyqt5():
    return False

  def _devicePixelRatio(obj):
    return 1

  def _setDevicePixelRatio(obj, factor):
    pass


if _QT_API in {QT_API_PYQT5, QT_API_PYSIDE2}:
  _setup_pyqt5()
elif _QT_API in {QT_API_PYQTv2, QT_API_PYSIDE, QT_API_PYQT}:
  _setup_pyqt4()
elif _QT_API is None:
  if mpl.rcParams['backend'] == 'Qt4Agg':
    _candidates = [
      (_setup_pyqt4, QT_API_PYQTv2),
      (_setup_pyqt4, QT_API_PYSIDE),
      (_setup_pyqt4, QT_API_PYQT),
      (_setup_pyqt5, QT_API_PYQT5),
      (_setup_pyqt5, QT_API_PYSIDE2),
    ]
  else:
    _candidates = [
      (_setup_pyqt5, QT_API_PYQT5),
      (_setup_pyqt5, QT_API_PYSIDE2),
      (_setup_pyqt4, QT_API_PYQTv2),
      (_setup_pyqt4, QT_API_PYSIDE),
      (_setup_pyqt4, QT_API_PYQT),
    ]
  for _setup, _QT_API in _candidates:
    try:
      _setup()
    except ImportError:
      continue
    break
  else:
    msg = 'Failed to import any qt binding'
    raise ImportError(msg)
else:  # We should not get there.
  msg = f'Unexpected QT_API: {_QT_API}'
  raise AssertionError(msg)

# These globals are only defined for backcompatibility purposes.
ETS = {
  'pyqt': (QT_API_PYQTv2, 4),
  'pyside': (QT_API_PYSIDE, 4),
  'pyqt5': (QT_API_PYQT5, 5),
  'pyside2': (QT_API_PYSIDE2, 5),
}
QT_RC_MAJOR_VERSION = 5 if is_pyqt5() else 4

if not is_pyqt5():
  mpl.cbook.warn_deprecated('3.3', name='support for Qt4')
