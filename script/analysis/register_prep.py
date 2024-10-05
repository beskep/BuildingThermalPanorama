"""
열화상-표준화 열화상-표준화-히스토그램
열화상-histeq 열화상-histeq-히스토그램
"""

from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from skimage.exposure import equalize_hist

from pano.misc.imageio import ImageIO
from script.plot import set_style

set_style(font_scale=1.0)


@click.command()
@click.option('--output', '-o')
@click.argument('path')
def main(path, output):
  image = ImageIO.read(path)

  image_std = (image.astype(np.float32) - np.mean(image)) / np.std(image)
  image_eq = equalize_hist(image=image)

  fig, axes = plt.subplots(2, 2, figsize=(7, 4.5))

  for col, img in enumerate([image_std, image_eq]):
    axes[0, col].imshow(img)
    axes[0, col].set_axis_off()
    # axes[0, col].set_xticks([])
    # axes[0, col].set_yticks([])

    sns.histplot(
        data=img.ravel(),
        ax=axes[1, col],
        stat='probability',
        # bins=70,
    )
    axes[1, col].set_box_aspect(3 / 4)

  # axes[0, 0].set_title('Image')
  # axes[0, 1].set_title('Histogram')
  axes[0, 0].set_title('(a) Standardization')
  axes[0, 1].set_title('(b) Histogram equalization')

  # axes[0, 0].set_ylabel('(a) Standardization')
  # axes[1, 0].set_ylabel('(b) Histogram equalization')

  fig.tight_layout()

  if output:
    Path(output).parent.stat()
    fig.savefig(output, dpi=300)
    plt.close(fig)
  else:
    plt.show()


if __name__ == '__main__':
  # pylint: disable=no-value-for-parameter
  main()
