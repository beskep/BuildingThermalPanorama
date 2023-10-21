from collections.abc import Callable

from matplotlib import pyplot as plt
from misc.imageio import ImageIO
from skimage.color import rgb2gray

from ..registrator import feature_match as fm
from ._evaluation import BaseEvaluation


class FeatureEvaluation(BaseEvaluation):

  def __init__(
      self,
      case_names: list,
      fixed_files: list,
      moving_files: list,
      fixed_prep: Callable,
      moving_prep: Callable,
  ) -> None:
    super().__init__(case_names, fixed_files, moving_files, fixed_prep, moving_prep)
    self._registrator = fm.FeatureBasedRegistrator()

  @property
  def registrator(self) -> fm.FeatureBasedRegistrator:
    return self._registrator

  def execute_once(self, case, ff, mf, df, output_dir, **kwargs):
    fixed_image = ImageIO.read(ff)
    moving_image = ImageIO.read(mf)
    if fixed_image.ndim == 3:
      fixed_image = rgb2gray(fixed_image)
    if moving_image.ndim == 3:
      moving_image = rgb2gray(moving_image)

    mfig, maxes = plt.subplots(1, 1, figsize=(16, 9))
    mi, metrics = self.register_and_evaluate(
        case=case, fi=fixed_image, mi=moving_image, df=df, ax=maxes
    )
    mfig.savefig(output_dir.joinpath(f'match_{case}.png'))
    mfig.tight_layout()
    plt.close(mfig)

    title = (
        case
        + ' | '
        + ' | '.join([f'{key}: {value:.2f}' for key, value in metrics.items()])
    )
    fig, axes, ci = self.plot(fi=fixed_image, mi=moving_image, title=title)
    fig.savefig(output_dir.joinpath(f'fig_{case}.png'))
    plt.close(fig)

    return fixed_image, moving_image
