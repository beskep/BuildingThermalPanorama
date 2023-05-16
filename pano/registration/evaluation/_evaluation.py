from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger
from rich.progress import track
from skimage.color import rgb2gray
from skimage.exposure import equalize_hist

from pano.misc import tools
from pano.misc.imageio import ImageIO

from ..registrator import BaseRegistrator, RegisteringImage
from .metrics import calculate_all_metrics


class BaseEvaluation:

  def __init__(
      self,
      case_names: list,
      fixed_files: list,
      moving_files: list,
      fixed_prep: Callable,
      moving_prep: Callable,
      save_fig=False,
  ) -> None:
    if len(case_names) != len(fixed_files):
      raise ValueError
    if len(case_names) != len(moving_files):
      raise ValueError

    self._cases = case_names
    self._fixed_files = fixed_files
    self._moving_files = moving_files
    self._fixed_prep = fixed_prep
    self._moving_prep = moving_prep
    self._save_fig = save_fig

    self._registrator: BaseRegistrator = None

  @property
  def registrator(self):
    return self._registrator

  def register_and_evaluate(
      self,
      case: str,
      fi: RegisteringImage,
      mi: RegisteringImage,
      df: defaultdict,
      **kwargs,
  ) -> tuple[RegisteringImage, dict]:
    try:
      rgst_image, register, matrix = self._registrator.register(
          fixed_image=fi.prep_image(), moving_image=mi.prep_image(), **kwargs
      )
    except (RuntimeError, ValueError, OSError) as e:
      logger.error(f'Case `{case}` failed')
      rgst_image = None
      register = None
      matrix = None

    # mi.registered_prep_image = rgst_image
    mi.set_registration(matrix, register)

    fi_eh = equalize_hist(fi.resized_image())
    mi_eh = equalize_hist(mi.registered_orig_image())
    metrics = calculate_all_metrics(fi_eh, mi_eh, bins=20, base=2)

    df['case'].append(case)
    for key, value in metrics.items():
      df[key].append(value)

    return mi, metrics

  @staticmethod
  def plot(fi: RegisteringImage, mi: RegisteringImage, title: str = None):
    fig, axes = plt.subplots(2, 4, figsize=(16, 9))

    compare_orig_cb = tools.prep_compare_images(fi.resized_image(), mi.resized_image())
    compare_reg_cb = tools.prep_compare_images(
        fi.prep_image(), mi.registered_prep_image()
    )
    compare_orig_diff = tools.prep_compare_images(
        fi.resized_image(), mi.resized_image(), method='diff'
    )
    compare_reg_diff = tools.prep_compare_images(
        fi.prep_image(), mi.registered_prep_image(), method='diff'
    )

    axes[0, 0].imshow(fi.orig_image)
    axes[1, 0].imshow(fi.prep_image)

    axes[0, 1].imshow(mi.orig_image)
    axes[1, 1].imshow(mi.registered_prep_image)

    axes[0, 2].imshow(compare_orig_cb)
    axes[1, 2].imshow(compare_reg_cb)

    axes[0, 3].imshow(compare_orig_diff)
    axes[1, 3].imshow(compare_reg_diff)

    for row, moved in enumerate(('original', 'registered')):
      for col, img in enumerate(('fixed', 'moving', 'checkerboard', 'diff')):
        axes[row, col].set_title(f'{moved} {img} image')

    for ax in axes.ravel():
      ax.set_axis_off()

    if title:
      fig.suptitle(title)

    fig.tight_layout()

    return fig, axes, compare_reg_cb

  def execute_once(self, case, ff, mf, df, output_dir, **kwargs):
    fixed_image = ImageIO.read(ff)
    moving_image = ImageIO.read(mf)
    if fixed_image.ndim == 3:
      fixed_image = rgb2gray(fixed_image)
    if moving_image.ndim == 3:
      moving_image = rgb2gray(moving_image)

    fi = RegisteringImage(image=fixed_image, preprocess=self._fixed_prep)
    mi = RegisteringImage(
        image=moving_image, preprocess=self._moving_prep, shape=fi.orig_image.shape
    )

    mi, metrics = self.register_and_evaluate(case=case, fi=fi, mi=mi, df=df, **kwargs)

    if self._save_fig and mi.is_registered():
      title = (
          case
          + ' | '
          + ' | '.join([f'{key}: {value:.3f}' for key, value in metrics.items()])
      )
      fig, axes, ci = self.plot(fi=fi, mi=mi, title=title)
      fig.savefig(output_dir.joinpath(f'fig_{case}.jpg'), dpi=150)
      plt.close(fig)

    return fi, mi

  def execute(self, output_dir, fname, **kwargs):
    df = defaultdict(list)
    output_dir = Path(output_dir)

    it = track(
        zip(self._cases, self._fixed_files, self._moving_files),
        total=len(self._cases),
    )
    for case, ff, mf in it:
      try:
        self.execute_once(
            case=case, ff=ff, mf=mf, df=df, output_dir=output_dir, **kwargs
        )
      except (RuntimeError, ValueError, TypeError) as e:
        logger.error(f'FAIL: {case}')
        continue

    df = pd.DataFrame(df)
    df.to_csv(
        output_dir.joinpath(fname).with_suffix('.csv'),
        index=False,
        encoding='utf-8-sig',
    )

  def execute_params(self, output_dir, fname, **kwargs):
    pass
