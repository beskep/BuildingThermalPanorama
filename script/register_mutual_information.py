"""
보고서 Mutual Information 설명용 plot

img1 img2 histogram 2d_hisgotram+MI
"""

from pathlib import Path
from typing import Optional

import click
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from skimage.exposure import equalize_hist
from skimage.transform import resize

from pano.misc import tools
from pano.misc.imageio import ImageIO
from pano.registration.evaluation import compute_ncc
from pano.registration.evaluation import compute_rmse
from pano.registration.evaluation import MutualInformation


def _read_img(path, eq=False) -> np.ndarray:
  img = ImageIO.read(path=path)
  img = tools.gray_image(img)

  if eq:
    img = tools.normalize_image(img)
    img = equalize_hist(img)
  else:
    img = (img - np.mean(img)) / np.std(img)

  return img


def pairwise(iterable):
  it = iter(iterable)
  return zip(it, it)


def mi_plot(paths: tuple,
            titles=('IR', 'Visible'),
            eqs: Optional[tuple] = None) -> tuple[plt.Figure, plt.Axes]:
  """
  Parameters
  ----------
  paths : tuple
      ((ir0, vis0),
       (ir1, vis0), ...)

  titles : tuple, optional
      ('IR', 'Visible')

  eqs : tuple, optionsl
      (bool, ...)

  Returns
  -------
  tuple[plt.Figure, plt.Axes]
  """
  sns.set_theme(context='notebook', style='whitegrid', font_scale=1.2)
  palette = sns.color_palette('Dark2', 2)[::-1]

  if not eqs:
    eqs = ((False, False),) * len(paths)

  nrows = len(paths)
  fig, axes = plt.subplots(nrows=nrows, ncols=4, figsize=(16, 8))

  for row, (p, eq) in enumerate(zip(paths, eqs)):
    img1 = _read_img(p[0], eq=eq[0])
    img2 = _read_img(p[1], eq=eq[1])

    if img1.shape != img2.shape:
      img2 = resize(img2, output_shape=img1.shape)

    images = (img1, img2)

    for col, img in enumerate(images):
      axes[row, col].imshow(img)
      axes[row, col].set_axis_off()
      # axes[row, col].set_title(titles[col])

    hist_data = {title: img.ravel() for title, img in zip(titles, images)}
    sns.histplot(data=hist_data,
                 ax=axes[row, 2],
                 stat='probability',
                 palette=palette)
    axes[row, 2].set_xlabel('Intensity')
    if not all(eq):
      axes[row, 2].set_xlim(-3, 3)
    axes[row, 2].set_box_aspect(0.8)

    mi = MutualInformation(image1=images[0], image2=images[1], bins=50)
    axes[row, 3].matshow(mi.joint_hist)
    # axes[row, 3].set_axis_off()
    axes[row, 3].set_xticks([])
    axes[row, 3].set_yticks([])
    axes[row, 3].set_ylabel(f'{titles[0]} intensity')
    axes[row, 3].set_xlabel(f'{titles[1]} intensity')

    logger.info('Row {} ncc: {}', row, compute_ncc(img1, img2))
    logger.info('Row {} mse: {}', row, np.square(compute_rmse(img1, img2)))
    logger.info('Row {} mi: {}', row, mi.mutual_information)

  fig.tight_layout()

  # axes[0, 0].set_title(titles[0])
  # axes[0, 1].set_title(titles[1])
  # axes[0, 2].set_title('Intensity distribution')
  # axes[0, 3].set_title('Joint intensity distribution')

  return fig, axes


@click.command()
@click.option('--directory', '-d')
@click.option('--output', '-o')
@click.option('--eqs', '-e', multiple=True, type=bool)
@click.argument('paths', nargs=-1)
def main(directory, output, eqs: tuple, paths):
  if not paths:
    raise ValueError

  if directory:
    directory = Path(directory)
    directory.stat()
    paths = [directory.joinpath(x) for x in paths]
  else:
    paths = [Path(x) for x in paths]

  for path in paths:
    path.stat()

  path_pair = tuple(pairwise(paths))

  if eqs:
    eqs = tuple(pairwise(eqs))

  fig, axes = mi_plot(paths=path_pair, eqs=eqs)

  if output:
    Path(output).parent.stat()
    fig.savefig(output, dpi=300)
    plt.close(fig)
  else:
    plt.show()


if __name__ == '__main__':
  # pylint: disable=no-value-for-parameter
  main()
