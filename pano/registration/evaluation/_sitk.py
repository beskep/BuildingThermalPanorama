from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Callable, List

from loguru import logger
import pandas as pd
from rich.progress import Progress
from skimage.exposure import rescale_intensity
from skimage.io import imsave
from skimage.util import compare_images

from ..registrator import simpleitk as rsitk
from ..registrator.registrator import RegistrationPreprocess
from ._evaluation import BaseEvaluation


class SITKEvaluation(BaseEvaluation):

  def __init__(self,
               case_names: list,
               fixed_files: list,
               moving_files: list,
               fixed_prep: Callable,
               moving_prep: Callable,
               save_fig=False) -> None:
    self._fixed_prep = None
    self._moving_prep = None
    super().__init__(case_names, fixed_files, moving_files, fixed_prep,
                     moving_prep, save_fig)
    self._registrator = rsitk.SITKRegistrator()

  @property
  def registrator(self) -> rsitk.SITKRegistrator:
    return super().registrator

  def execute_params(
      self,
      output_dir,
      fname,
      trsfs: List[rsitk.Transformation],
      metric_opts: List[dict],
      preps: List[RegistrationPreprocess],
      **kwargs,
  ):
    df = defaultdict(list)
    output_dir = Path(output_dir)
    total = len(self._cases) * len(trsfs) * len(metric_opts) * len(preps)

    logger.info('Number of files: {}', len(self._cases))
    logger.info('Number of parameters: {}',
                len(trsfs) * len(metric_opts) * len(preps))
    logger.info('Number of total cases: {}', total)

    save_image = kwargs.get('save_image', False)

    with Progress() as progress:
      task = progress.add_task('Registering...', total=total)

      it = product(range(len(self._cases)), trsfs, metric_opts, preps)
      for iidx, trsf, mopt, prep in it:
        self.registrator.transformation = trsf
        self.registrator.set_metric(**mopt)
        self._fixed_prep = prep.fixed_preprocess
        self._moving_prep = prep.moving_preprocess

        case = self._cases[iidx]
        ff = self._fixed_files[iidx]
        mf = self._moving_files[iidx]

        try:
          fi, mi = self.execute_once(case=case,
                                     ff=ff,
                                     mf=mf,
                                     df=df,
                                     output_dir=output_dir)
        except (RuntimeError, ValueError, TypeError) as e:
          logger.error(f'FAIL: {case}')
          # fi, mi = None, None
          raise ValueError from e

        df['param_trsf'].append(trsf.name)

        for key, value in mopt.items():
          v = getattr(value, 'name', str(value))
          df['param_' + str(key)].append(v)

        for key, value in prep.parameters():
          df['param_' + str(key)].append(str(value))

        if save_image and mi is not None:
          img_file = [
              f'{value[-1]}' for key, value in df.items()
              if key.startswith('param')
          ]
          img_file = case + '_' + '_'.join(img_file) + '.jpg'

          ci = compare_images(fi.prep_image,
                              mi.registered_prep_image,
                              method='diff')
          imsave(
              output_dir.joinpath(img_file).as_posix(),
              rescale_intensity(ci, out_range='uint8'))

        progress.advance(task)

    df = pd.DataFrame(df)
    df.to_csv(output_dir.joinpath(fname).with_suffix('.csv'),
              index=False,
              encoding='utf-8-sig')
