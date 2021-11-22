from collections import defaultdict

import click
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns
from skimage.exposure import equalize_hist
from skimage.transform import EuclideanTransform
from skimage.transform import resize
from skimage.transform import warp

from pano.misc import tools
from pano.misc.imageio import ImageIO
from pano.registration.evaluation import calculate_all_metrics

font_name = 'Noto Sans CJK KR'
mpl.rc('font', family=font_name)
mpl.rcParams['axes.unicode_minus'] = False
snsrc = {'axes.edgecolor': '0.2', 'grid.color': '0.8'}
sns.set_theme(context='paper',
              style='whitegrid',
              font=font_name,
              font_scale=1.75,
              rc=snsrc)


def metrics_by_trnsl(fixed_image, moving_image, translation_range, bins='auto'):
  metrics = defaultdict(list)

  for tr in translation_range:
    if tr == 0:
      mi = moving_image
      fi = fixed_image
    else:
      mi = warp(moving_image,
                inverse_map=EuclideanTransform(translation=(tr, 0)).inverse,
                cval=np.nan)

      mask = ~np.isnan(mi)
      mi = mi[mask]
      fi = fixed_image[mask]

    m = calculate_all_metrics(fi, mi, bins=bins)
    for key, value in m.items():
      metrics[key].append(value)

  fig, axes = plt.subplots(2, 2, figsize=(16, 9))

  for col, (img, title) in enumerate(
      zip(
          [fixed_image, moving_image],
          ['IR image', 'Visible image'],
      )):
    axes[0, col].imshow(img)
    axes[0, col].set_title(title)
    axes[0, col].set_axis_off()

  for col, metric in enumerate(['NCC', 'MI']):
    axes[1, col].plot(translation_range, metrics[metric])
    axes[1, col].set_title(metric)
    axes[1, col].set_xlabel('x translation [pixel]')

  return fig, axes


def _read(path):
  img = ImageIO.read(path)
  gray = tools.gray_image(img)
  norm = tools.normalize_image(gray)
  eq = equalize_hist(norm)

  return eq


@click.command()
@click.option('--output', '-o')
@click.option('--tmin', default=-50)
@click.option('--tmax', default=50)
@click.option('--tnum', default=50)
@click.argument('paths', nargs=2)
def main(paths, output, tmin, tmax, tnum):
  fixed_image = _read(paths[0])
  moving_image = _read(paths[1])

  if fixed_image.shape != moving_image.shape:
    moving_image = resize(moving_image,
                          output_shape=fixed_image.shape,
                          anti_aliasing=True)

  fig, axes = metrics_by_trnsl(
      fixed_image=fixed_image,
      moving_image=moving_image,
      translation_range=np.linspace(tmin, tmax, num=tnum, endpoint=True),
  )

  if output:
    fig.savefig(output, dpi=200)
  else:
    plt.show()


if __name__ == '__main__':
  # pylint: disable=no-value-for-parameter
  main()
