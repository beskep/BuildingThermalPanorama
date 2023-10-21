import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from loguru import logger
from matplotlib import gridspec
from skimage import draw

from pano import utils
from pano.interface.plot_controller.output import (
    _edgelets_between_edgelets,
    _suppress_edgelets,
    extend_lines,
)
from pano.misc import edgelet
from pano.misc.imageio import ImageIO as IIO
from pano.misc.tools import SegMask, normalize_image

DIR = utils.DIR.RESOURCE / 'TestImage'
cmap = sns.color_palette('Dark2')


def _theta(thold):
  return np.deg2rad(np.linspace(-thold, thold, int(thold * 2 + 1)) + 90.0)


def _edgelets(image: np.ndarray, horiz_thold=2, max_count=10):
  edges = edgelet.image2edges(image, eqhist=False)

  hough_opts = edgelet.HoughOptions(line_gap=10, theta=_theta(horiz_thold))
  edge_window = edgelet.edge2edgelets(edges, hough_option=hough_opts)
  logger.info('# of window edgelets: {}', edge_window.count)

  edge_window = _suppress_edgelets(edge_window)
  edge_window.sort()
  if edge_window.count > max_count:
    edge_window = edge_window[:max_count]

  logger.info('# of window edgelets: {} (after suppress)', edge_window.count)

  edge_floor = _edgelets_between_edgelets(edge_window)
  edge_floor.strengths = 200 * np.ones_like(edge_floor.strengths)
  logger.info('# of floor edgelets: {} (after suppress)', edge_floor.count)

  return edges, edge_window, edge_floor


def _pnts_arr(edgelets: edgelet.Edgelets):
  half = edgelets.strengths.reshape([-1, 1]) / 2.0
  pt1 = edgelets.locations - edgelets.directions * half  # [[x1, y1], ...]
  pt2 = edgelets.locations + edgelets.directions * half  # [[x2, y2], ...]

  xarr = np.hstack([pt1[:, [0]], pt2[:, [0]]])  # [[x1, x2], ...]
  yarr = np.hstack([pt1[:, [1]], pt2[:, [1]]])  # [[y1, y2], ...]

  return xarr, yarr


class WindowLine:

  def __init__(self, mask: np.ndarray, threshold=0.05) -> None:
    self.mask = mask
    self._shape = mask.shape
    self._threshold = threshold

  def clip(self, c: float, axis=0):
    # 영상 내 shape 범위로 clip하고 int 형식으로 변환
    return int(np.clip(c, a_min=0, a_max=(self._shape[axis] - 1)))

  def is_window_line(self, xs, ys):
    # edgelet이 지나는 좌표
    lxs, lys = draw.line(
        r0=self.clip(ys[0], 1),
        c0=self.clip(xs[0], 0),
        r1=self.clip(ys[1], 1),
        c1=self.clip(xs[1], 0),
    )

    pixels = self.mask[lxs, lys]
    wall = np.sum(pixels == SegMask.WALL)
    window = np.sum(pixels == SegMask.WINDOW)

    return window >= self._threshold * (wall + window)


def draw_edgelets(
    edgelets: edgelet.Edgelets,
    ax: plt.Axes,
    wl: WindowLine,
    # color=None,
    extend=False,
):
  arr = _pnts_arr(edgelets)
  lim = dict(xlim=ax.get_xlim(), ylim=ax.get_ylim())

  for xs, ys in zip(*arr):
    if not extend:
      x, y = xs, ys
      color = cmap[0]
    else:
      n = [-ys[0] + ys[1], xs[0] - xs[1]]
      pt1, pt2 = extend_lines(n=n, p=[xs[0], ys[0]], **lim)
      x, y = [pt1[0], pt2[0]], [pt1[1], pt2[1]]
      color = cmap[1 if wl.is_window_line(x, y) else 2]

    ax.plot(x, y, color=color, linewidth=2, marker='D', markersize=5)


def example(path, img_cmap='gray'):
  image = IIO.read(path)
  edges, wedge, fedge = _edgelets(image)
  wl = WindowLine(SegMask.vis2index(image))  # FIXME

  fig = plt.figure()
  gs = gridspec.GridSpec(2, 2)
  axes = [
      fig.add_subplot(gs[0, 0]),
      fig.add_subplot(gs[0, 1]),
      fig.add_subplot(gs[1, :]),
  ]

  axes[0].imshow(image, cmap=img_cmap)
  axes[1].imshow(edges, cmap=img_cmap)
  axes[2].imshow(normalize_image(edges), cmap=img_cmap, vmin=0, vmax=2)

  draw_edgelets(wedge, axes[2], wl=wl)
  draw_edgelets(fedge, axes[2], wl=wl, extend=True)

  for ax in axes:
    ax.set_axis_off()

  return fig, axes


if __name__ == '__main__':
  sns.set_theme(
      context='talk',
      style='white',
      font='Source Han Sans KR',
      rc={'figure.figsize': [16, 9], 'savefig.dpi': 300},
  )

  path = DIR / 'floor-detection2.png'

  fig, _ = example(path=path)
  fig.savefig(DIR / 'floor-detection-plot.png')
