from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np

from pano.distortion import projection as prj
from pano.misc.imageio import ImageIO
from script.plot import set_style

DEFAULT_ANGLE = 30

set_style(font_scale=1.75)


@click.command()
@click.option('--roll', default=DEFAULT_ANGLE)
@click.option('--pitch', default=DEFAULT_ANGLE)
@click.option('--yaw', default=DEFAULT_ANGLE)
@click.option('--va', default=42.0)
@click.argument('path')
@click.argument('output', required=False)
def cli(roll, pitch, yaw, va, path, output):
  path = Path(path)
  path.stat()

  img = ImageIO.read(path).astype(np.float32)

  roll, pitch, yaw = np.deg2rad([roll, pitch, yaw])
  imgprj = prj.ImageProjection(image=img, viewing_angle=np.deg2rad(va))

  angles = (roll, pitch, yaw)
  titles = ('(a) Original Image', '(b) Roll', '(c) Pitch', '(d) Yaw')

  fig, axes = plt.subplots(2, 2, figsize=(16, 9))

  for idx, ax in enumerate(axes.ravel()):
    if idx == 0:
      ax.imshow(img)
    else:
      a = [0, 0, 0]
      a[idx - 1] = angles[idx - 1]
      img_rot = imgprj.project(*a)

      ax.imshow(img_rot)

    ax.set_title(titles[idx], pad=12)
    # ax.set_axis_off()
    ax.set_xticks([])
    ax.set_yticks([])

  fig.tight_layout(pad=1.2)

  if output:
    fig.savefig(output, dpi=200)
  else:
    plt.show()


if __name__ == '__main__':
  # pylint: disable=no-value-for-parameter
  cli()
