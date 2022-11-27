from typing import Optional

import cv2 as cv
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from PIL import Image
import seaborn as sns

from pano import utils

DIR = utils.DIR.RESOURCE / 'TestImage'


def pyr_down(image: NDArray, deep=4):
  img = image.copy()
  pyr = [img]
  for _ in range(deep - 1):
    img = cv.pyrDown(img)
    pyr.append(img)

  return pyr


def laplacian_pyramid(pyramid: list[NDArray]):
  lp = [pyramid[-1]]
  for idx in reversed(range(1, len(pyramid))):
    img = cv.pyrUp(pyramid[idx])
    shape = pyramid[idx - 1].shape
    resized = cv.resize(img,
                        dsize=(shape[1], shape[0]),
                        interpolation=cv.INTER_CUBIC)
    laplacian = cv.subtract(pyramid[idx - 1], resized)
    lp.append(laplacian)

  return lp


def multiband_blending(img1: NDArray,
                       img2: NDArray,
                       mask: Optional[NDArray] = None,
                       deep=4):
  if img1.shape != img2.shape:
    raise ValueError

  if mask is None:
    mask = np.zeros_like(img1)
    mask[:, int(img1.shape[1] / 2):] = 1

  gp1 = pyr_down(image=img1, deep=deep)
  gp2 = pyr_down(image=img2, deep=deep)
  gpm = pyr_down(image=mask, deep=deep)
  gpm = list(reversed(gpm))

  lp1 = laplacian_pyramid(gp1)
  lp2 = laplacian_pyramid(gp2)

  pyramid = [(1 - gpm[i]) * lp1[i] + gpm[i] * lp2[i] for i in range(deep)]
  blended = pyramid[0]
  for idx in range(1, deep):
    blended = cv.pyrUp(blended)
    shape = lp1[idx].shape
    blended = cv.resize(blended,
                        dsize=(shape[1], shape[0]),
                        interpolation=cv.INTER_CUBIC)
    blended = cv.add(blended, pyramid[idx])

  return blended


def feather_alpha(width: float, shape: tuple):
  half = width / 2.0
  xy = [[0, 0.5 - half, 0.5 + half, 1.0], [0, 0, 1, 1]]

  pnts = [int(x) for x in np.round(shape[1] * np.array(xy[0]))]
  line = np.zeros(shape[1])
  line[pnts[1]:pnts[2]] = np.linspace(0, 1, num=(pnts[2] - pnts[1]))
  line[int(pnts[2]):] = 1

  alpha = np.tile(line, (shape[0], 1))

  return xy, alpha


def example_no_blending(img1: NDArray, img2: NDArray):
  if img1.shape != img2.shape:
    raise ValueError

  # fig, axes = plt.subplots(1, 3)
  fig = plt.figure()
  gs = gridspec.GridSpec(2, 2)
  axes = [
      fig.add_subplot(gs[0, 0]),
      fig.add_subplot(gs[1, :]),
      fig.add_subplot(gs[0, 1])
  ]

  axes[0].imshow(img1)
  axes[2].imshow(img2)

  img_cut = img1.copy()
  half = int(img1.shape[1] / 2)
  img_cut[:, half:] = img2[:, half:]
  axes[1].imshow(img_cut)

  fig.savefig(DIR / 'blending-cut.png')
  plt.close(fig)

  for idx, img in enumerate([img1, img2, img_cut]):
    Image.fromarray(img).save(DIR / f'blending-cut-{idx}.png')


def _feather_blend(img1: NDArray, img2: NDArray, alpha: NDArray):
  imax = np.iinfo(img1.dtype).max
  imga1 = np.dstack([img1, imax * (1 - alpha)])
  imga2 = np.dstack([img2, imax * alpha])
  alpha3 = alpha[:, :, np.newaxis]
  blended = (1 - alpha3) * img1 + alpha3 * img2

  return imga1, imga2, blended


def example_feather(img1: NDArray, img2: NDArray, width=0.1):
  if img1.shape != img2.shape:
    raise ValueError

  if not (0 < width < 1):
    raise ValueError

  xy, alpha = feather_alpha(width, img1.shape)

  # feather-01
  fig, axes = plt.subplots(1, 2)
  axes[0].plot(*xy)
  axes[1].imshow(alpha, cmap='gray')

  axes[0].set_box_aspect(alpha.shape[0] / alpha.shape[1])
  axes[0].set_xlabel('Position')
  axes[0].set_ylabel('Alpha')
  fig.tight_layout()

  fig.savefig(DIR / 'blending-feather-01.png')
  plt.close(fig)

  # feather-02
  img1, img2, blended = _feather_blend(img1, img2, alpha)
  fig = plt.figure()
  gs = gridspec.GridSpec(2, 2)
  axes = [
      fig.add_subplot(gs[0, 0]),
      fig.add_subplot(gs[1, :]),
      fig.add_subplot(gs[0, 1])
  ]
  for idx, (ax, img) in enumerate(zip(axes, [img1, blended, img2])):
    imgu = img.astype(np.uint8)
    ax.imshow(imgu)
    Image.fromarray(imgu).save(DIR / f'blending-feather-02-{idx}.png')

  fig.tight_layout()
  fig.savefig(DIR / 'blending-feather-02.png')
  plt.close(fig)


def _pyramid_plot(pyramid: list, title=None):
  assert len(pyramid) <= 4
  fig, axes = plt.subplots(2, 2)

  for idx, (ax, img) in enumerate(zip(axes.ravel(), pyramid)):
    ax.imshow(img)
    if title:
      ax.set_title(f'{title} No. {idx+1}')

  for ax in axes.ravel():
    if not ax.has_data():
      ax.set_axis_off()

  fig.tight_layout()

  return fig, axes


def example_multiband(img1: NDArray,
                      img2: NDArray,
                      mask: Optional[NDArray] = None,
                      deep=4):
  if img1.shape != img2.shape:
    raise ValueError

  if mask is None:
    mask = np.zeros_like(img1)
    mask[:, int(img1.shape[1] / 2):] = 1

  gp1 = pyr_down(image=img1, deep=deep)
  gp2 = pyr_down(image=img2, deep=deep)
  gpm = pyr_down(mask)
  gpm = list(reversed(gpm))

  fig, _ = _pyramid_plot(gp1, title='Gaussian Pyramid')
  fig.savefig(DIR / 'blending-multiband-gaussian.png')
  plt.close(fig)

  lp1 = laplacian_pyramid(gp1)
  lp2 = laplacian_pyramid(gp2)
  fig, _ = _pyramid_plot([(1.5 * x).astype(np.uint8) for x in lp1[:0:-1]],
                         title='Laplacian Pyramid')
  fig.savefig(DIR / 'blending-multiband-laplacian.png')
  plt.close(fig)

  pyramid = [(1 - gpm[i]) * lp1[i] + gpm[i] * lp2[i] for i in range(deep)]
  fig, _ = _pyramid_plot(pyramid, title='Blended Images')
  fig.savefig(DIR / 'blending-multiband-blended_pyramid.png')
  plt.close(fig)


if __name__ == '__main__':
  sns.set_theme(context='talk',
                style='white',
                font='Source Han Sans KR',
                rc={
                    'figure.figsize': [16, 9],
                    'savefig.dpi': 300
                })

  img1 = cv.imread(str(DIR / 'blending01.png'))
  img2 = cv.imread(str(DIR / 'blending02.png'))

  example_no_blending(img1=img1, img2=img2)
  example_feather(img1=img1, img2=img2, width=0.2)
  example_multiband(img1=img1, img2=img2, mask=None, deep=4)

  for deep in range(2, 11, 2):
    blended = multiband_blending(img1=img1, img2=img2, deep=deep)
    Image.fromarray(blended).save(DIR /
                                  f'blending-multiband-blended-{deep}.png')
