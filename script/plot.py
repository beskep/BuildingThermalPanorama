from typing import Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def set_style(context='notebook', style='whitegrid', font_scale=1):
  font_name = 'Noto Sans CJK KR'
  mpl.rc('font', family=font_name)
  mpl.rcParams['axes.unicode_minus'] = False
  snsrc = {'axes.edgecolor': '0.2', 'grid.color': '0.8'}

  sns.set_theme(context=context,
                style=style,
                font=font_name,
                font_scale=font_scale,
                rc=snsrc)


class Subplots:

  def __init__(self,
               nrows=1,
               ncols=1,
               sharex=False,
               sharey=False,
               squeeze=True,
               figsize=(6.4, 4.8),
               dpi=100.0,
               subplot_kw=None,
               gridspec_kw=None,
               **fig_kw) -> None:
    if figsize is not None:
      fig_kw['figsize'] = figsize

    if dpi is not None:
      fig_kw['dpi'] = dpi

    subplots = plt.subplots(nrows=nrows,
                            ncols=ncols,
                            sharex=sharex,
                            sharey=sharey,
                            squeeze=squeeze,
                            subplot_kw=subplot_kw,
                            gridspec_kw=gridspec_kw,
                            **fig_kw)

    self._fig: plt.Figure = subplots[0]
    self._axes: Union[plt.Axes, np.ndarray] = subplots[1]

  def __enter__(self):
    return (self._fig, self._axes)

  def __exit__(self, exc_type, exc_value, traceback):
    plt.close(self._fig)
