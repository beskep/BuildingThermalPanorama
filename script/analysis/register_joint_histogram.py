"""
히스토그램 평준화에 따른 Joint Histogram 변화
"""

from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from skimage.exposure import equalize_hist
from skimage.transform import resize

from pano.misc import tools
from pano.misc.imageio import ImageIO
from pano.registration.evaluation import MutualInformation


def _read_img(path, eq=False) -> np.ndarray:
  img = ImageIO.read(path=path)
  img = tools.gray_image(img)

  # img -= np.average(img)
  # img /= np.std(img)

  return img


@click.command()
@click.option('--directory', '-d')
@click.option('--output', '-o')
@click.argument('paths', nargs=2)
def main(directory, output, paths):
  if directory:
    directory = Path(directory)
    directory.stat()
    paths = [directory.joinpath(x) for x in paths]
  else:
    paths = [Path(x) for x in paths]

  for path in paths:
    path.stat()

  ir = _read_img(paths[0])
  vis = _read_img(paths[1])

  if ir.shape != vis.shape:
    vis = resize(vis, output_shape=ir.shape)

  fig, axes = plt.subplots(1, 2, figsize=(10, 4))

  for col in range(2):
    if col == 0:
      ir_jh = ir
      vis_jh = vis
    else:
      ir_jh = equalize_hist(ir)
      vis_jh = equalize_hist(vis)

    mi = MutualInformation(ir_jh, vis_jh, bins=50)
    hist = mi.joint_hist / np.sum(mi.joint_hist)
    sns.heatmap(data=hist, cbar=True, ax=axes[col])

    axes[col].set_xticks([])
    axes[col].set_yticks([])
    axes[col].set_ylabel('IR intensity')
    axes[col].set_xlabel('Visible intensity')
    axes[col].set_aspect('equal')

  axes[0].set_title('(a) Standardization')
  axes[1].set_title('(b) Histogram equalization')

  if output:
    Path(output).parent.stat()
    fig.savefig(output, dpi=300)
    plt.close(fig)
  else:
    plt.show()


if __name__ == '__main__':
  # pylint: disable=no-value-for-parameter
  main()
