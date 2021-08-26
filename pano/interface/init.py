import os
from pathlib import Path

import skimage.io


def init_project(qt: bool):
  # pylint: disable=import-outside-toplevel

  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
  skimage.io.use_plugin('pil')

  if qt:
    import matplotlib
    import matplotlib.pyplot

    matplotlib.use('Qt5Agg')

    import PySide2

    plugins_path = Path(PySide2.__file__).parent.joinpath('plugins')
    os.environ['QT_PLUGIN_PATH'] = plugins_path.as_posix()
