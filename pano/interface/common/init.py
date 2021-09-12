import os
from pathlib import Path
import sys

import matplotlib as mpl
import seaborn as sns
import skimage.io

from pano.utils import DIR
from pano.utils import is_frozen


def init_project(qt: bool):
  # pylint: disable=import-outside-toplevel

  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
  skimage.io.use_plugin('pil')

  if qt:
    if not is_frozen() and 'PySide2' in sys.modules:
      import PySide2

      pyside_dir = Path(PySide2.__file__).parent
      plugins_dir = pyside_dir.joinpath('plugins')
      qml_dir = pyside_dir.joinpath('qml')

      os.environ['QT_PLUGIN_PATH'] = plugins_dir.as_posix()
      os.environ['QML2_IMPORT_PATH'] = qml_dir.as_posix()

    mpl.use('Qt5Agg')

  import matplotlib.font_manager as fm

  font_name = 'Noto Sans CJK KR'
  font_path = DIR.RESOURCE.joinpath('font/NotoSansCJKkr-Regular.otf')
  assert font_path.exists(), font_path

  fe = fm.FontEntry(fname=font_path.as_posix(), name=font_name)
  fm.fontManager.ttflist.insert(0, fe)

  sns.set_theme(context='notebook',
                style='whitegrid',
                font=font_name,
                rc={
                    'axes.edgecolor': '0.2',
                    'grid.color': '0.8'
                })

  mpl.rcParams['font.family'] = font_name
  mpl.rcParams['axes.unicode_minus'] = False
  mpl.rcParams['image.cmap'] = 'inferno'
