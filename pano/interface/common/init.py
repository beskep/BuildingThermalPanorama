import os

import skimage.io

from pano.utils import DIR


def init_project(qt: bool):
  # pylint: disable=import-outside-toplevel

  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
  skimage.io.use_plugin('pil')

  if qt:
    # Matplotlib backend, style 설정
    import matplotlib as mpl
    mpl.use('Qt5Agg')

    import matplotlib.font_manager as fm

    font_name = 'Spoqa Han Sans Neo'
    font_path = DIR.RESOURCE.joinpath('font/SpoqaHanSansNeo-Regular.ttf')
    assert font_path.exists(), font_path

    fe = fm.FontEntry(fname=font_path.as_posix(), name=font_name)
    fm.fontManager.ttflist.insert(0, fe)

    mpl.rcParams['font.family'] = font_name
    mpl.rcParams['axes.unicode_minus'] = False

    import matplotlib.pyplot

    try:
      import seaborn as sns
    except ImportError:
      pass
    else:
      sns.set(style='whitegrid',
              font=font_name,
              rc={
                  'axes.edgecolor': '0.2',
                  'grid.color': '0.8'
              })
      # sns.set_context('talk', font_scale=0.8)
