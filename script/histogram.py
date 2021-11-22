from pathlib import Path
from typing import Optional

import click
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from pano.misc.imageio import ImageIO
from pano.misc.tools import SegMask

sns.set_theme(style='white')


def outlier_mask(data: np.ndarray, k=1.5):
  data_nna = data[~np.isnan(data)]

  q1 = np.quantile(data_nna, q=0.25)
  q3 = np.quantile(data_nna, q=0.75)
  iqr = q3 - q1

  lower = q1 - k * iqr
  upper = q3 + k * iqr

  return (data < lower) | (upper < data)


def remove_outliers(data: np.ndarray, k=1.5):
  mask = outlier_mask(data=data, k=k)

  return data[~mask]


def image_outlier_map(image: np.ndarray,
                      label: Optional[np.ndarray] = None,
                      k=1.5):
  if label is None:
    label = np.zeros_like(image, dtype=int)
  else:
    assert image.shape == label.shape
    label = label.astype(int)

  if np.min(label) <= 0:
    label += np.min(label) + 1

  # non_na = ~np.isnan(image)
  outliers = np.full_like(image, fill_value=np.nan)

  for idx in np.unique(label):
    img = image.copy()
    img[label != idx] = np.nan

    om = outlier_mask(data=img, k=k)

    outliers[om & (label == idx)] = idx

  return outliers


@click.command()
@click.option('--directory')
@click.option('--ir', required=True)
@click.option('--mask', required=True)
@click.option('--seg', required=False)
@click.argument('ks', nargs=-1)
def main(directory, ir, mask, seg, ks):
  if directory:
    directory = Path(directory)
    ir = directory.joinpath(ir)
    mask = directory.joinpath(mask)
    if seg:
      seg = directory.joinpath(seg)

  ir_arr = ImageIO.read(ir).astype(np.float32)
  mask_arr = ImageIO.read(mask)
  ir_arr[np.logical_not(mask_arr)] = np.nan

  if seg:
    seg_arr = ImageIO.read(seg)
    if seg_arr.ndim == 3:
      seg_arr = seg_arr[:, :, 0]
    seg_arr = SegMask.vis_to_index(seg_arr)
  else:
    seg_arr = None

  axes_count = len(ks) + 1
  ncols = int(np.sqrt(16 * axes_count / 9))
  nrows = int(np.ceil(axes_count / ncols))

  fig, axes = plt.subplots(nrows, ncols)
  # axes.ravel()[0].imshow(ir_arr)
  for ax in axes.ravel():
    ax.imshow(ir_arr, cmap='autumn')

  for k, ax in zip(ks, axes.ravel()[1:]):
    outliers = image_outlier_map(image=ir_arr, label=seg_arr, k=float(k))
    ax.imshow(outliers)
    ax.set_title(f'k = {k}')

  fig.tight_layout()
  plt.show()


if __name__ == '__main__':
  # pylint: disable=no-value-for-parameter
  main()
