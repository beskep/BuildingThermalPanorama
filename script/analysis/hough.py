import click
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from skimage.draw import line
from skimage.transform import hough_line, hough_line_peaks

font_name = 'Noto Sans CJK KR'
mpl.rc('font', family=font_name)
mpl.rcParams['axes.unicode_minus'] = False
snsrc = {'axes.edgecolor': '0.2', 'grid.color': '0.8', 'axes.titlepad': 12}


@click.group()
def cli():
  pass


@cli.command()
@click.option('--path', '-p', default=None)
def hough1(path):
  sns.set_theme(
      context='notebook', style='whitegrid', font=font_name, font_scale=1, rc=snsrc
  )
  w = [-1 / 2, 1]
  x0y0 = [2 / 5, 4 / 5]

  def xy(x):
    return w[0] * x + w[1]

  def theta_rho(theta, x, y):
    return x * np.cos(theta) + y * np.sin(theta)

  xy_line_xs = np.array([-1, 3])
  xy_line_ys = xy(xy_line_xs)

  xy_dots_xs = np.array([0.8, 1.2, 1.6])
  xy_dots_ys = xy(xy_dots_xs)

  fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4.5))

  axes[0].plot(xy_line_xs, xy_line_ys)
  axes[0].scatter(xy_dots_xs, xy_dots_ys, c='gray')
  axes[0].set_xlim(0, 2)
  axes[0].set_ylim(0, 1)
  axes[0].set_aspect('equal', 'box')
  axes[0].set_title(r'$(x, y)$ space')

  axes[0].plot([0, x0y0[0]], [0, x0y0[1]], color='gray')

  theta = np.linspace(0, np.pi, num=50)
  for x, y in zip(xy_dots_xs, xy_dots_ys, strict=False):
    rho = theta_rho(theta, x=x, y=y)
    axes[1].plot(theta, rho)

  axes[1].set_xlim(0, 3)
  axes[1].set_aspect(0.6)
  axes[1].set_title(r'$(r, \theta)$ space')

  if path:
    fig.savefig(path)
  else:
    plt.show()


@cli.command()
@click.option('--path', '-p', default=None)
@click.option('--fs', '-f', default=2)
def hough2(path, fs):
  sns.set_theme(
      context='paper',
      style='whitegrid',
      font=font_name,
      font_scale=float(fs),
      rc=snsrc,
  )

  # Constructing test image
  image = np.zeros((200, 200))
  idx = np.arange(20, 175)
  # image[idx, idx] = 255
  # image[line(45, 25, 25, 175)] = 255
  # image[line(25, 135, 175, 155)] = 255

  lines = (
      ((20, 70), (150, 160)),
      ((45, 25), (25, 175)),
      ((25, 135), (175, 155)),
  )
  for l in lines:
    image[line(l[0][0], l[0][1], l[1][0], l[1][1])] = 255

  # Classic straight-line Hough transform
  # Set a precision of 0.5 degree.
  tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
  h, theta, d = hough_line(image, theta=tested_angles)

  # Generating figure 1
  fig, axes = plt.subplots(1, 3, figsize=(15, 6))
  ax = axes.ravel()

  ax[0].imshow(image, cmap='gray')
  ax[0].set_title('(a) Input image')
  ax[0].set_axis_off()

  angle_step = 0.5 * np.diff(theta).mean()
  d_step = 0.5 * np.diff(d).mean()
  bounds = [
      np.rad2deg(theta[0] - angle_step),
      np.rad2deg(theta[-1] + angle_step),
      d[-1] + d_step,
      d[0] - d_step,
  ]
  ax[1].imshow(np.log(1 + h), extent=bounds, cmap='gray', aspect=0.5)
  ax[1].set_title('(b) Hough transform')
  ax[1].set_xlabel(r'$\theta$ [º]')
  ax[1].set_ylabel('$r$ [pixels]')
  # ax[1].axis('image')

  ax[2].imshow(image, cmap='gray', alpha=0.0)
  ax[2].set_ylim((image.shape[0], 0))
  # ax[2].set_axis_off()
  ax[2].set_title('(c) Detected lines')

  for _, angle, dist in zip(*hough_line_peaks(h, theta, d), strict=False):
    (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
    ax[2].axline((x0, y0), slope=np.tan(angle + np.pi / 2), linewidth=2)

  for l in lines:
    ax[2].plot([l[0][1], l[1][1]], [l[0][0], l[1][0]], color='orangered', linewidth=5)

  fig.tight_layout()

  if path:
    fig.savefig(path)
  else:
    plt.show()


if __name__ == '__main__':
  cli()
