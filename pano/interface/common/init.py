import os
import sys
from pathlib import Path

from pano.utils import DIR, is_frozen


def is_ascii(s: str):
  return all(ord(c) < 128 for c in s)  # noqa: PLR2004


def ascii_tempdir():
  tmp = ['TEMP', 'TMP']
  if all(is_ascii(os.environ[x]) for x in tmp):
    return

  tmpdir = Path('C:\\Temp')
  if not tmpdir.exists():
    tmpdir.mkdir()

  for t in tmp:
    os.environ[t] = str(tmpdir)


def init_project(*, qt: bool):
  ascii_tempdir()

  # pylint: disable=import-outside-toplevel
  import matplotlib as mpl
  import matplotlib.font_manager as fm
  import seaborn as sns
  import skimage.io

  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
  skimage.io.use_plugin('pil')

  if qt:
    if not is_frozen() and 'PySide2' in sys.modules:
      import PySide2

      pyside_dir = Path(PySide2.__file__).parent
      os.environ['QT_PLUGIN_PATH'] = str(pyside_dir / 'plugins')
      os.environ['QML2_IMPORT_PATH'] = str(pyside_dir / 'qml')

    mpl.use('Qt5Agg')

  font_name = 'Source Han Sans KR'
  font_path = DIR.RESOURCE.joinpath('font/SourceHanSansKR-Normal.otf')
  assert font_path.exists(), font_path

  fe = fm.FontEntry(fname=font_path.as_posix(), name=font_name)
  fm.fontManager.ttflist.insert(0, fe)

  sns.set_theme(
      context='notebook',
      style='whitegrid',
      font=font_name,
      rc={'axes.edgecolor': '0.2', 'grid.color': '0.8', 'image.cmap': 'inferno'},
  )
