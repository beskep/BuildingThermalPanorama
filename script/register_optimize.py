"""
STIK 인자 최적화 효율 평가
대상: 전처리 (nomalization, standardization), STIK metric
"""

from itertools import product
from pathlib import Path
import shutil
from typing import Optional, Union

import click
from loguru import logger
import numpy as np

from pano import utils
from pano.interface.common.init import init_project
from pano.interface.common.pano_files import DIR
from pano.interface.pano_project import ThermalPanorama
from pano.misc import tools
from pano.misc.imageio import ImageIO as IIO
from pano.registration.registrator import RegistrationPreprocess
import pano.registration.registrator.simpleitk as rsitk


class _RegistrationPreprocess(RegistrationPreprocess):
  EXPOSURE = (
      # None,
      'eqhist',
      'norm',
  )

  def __init__(self, shape: tuple, exposure: Optional[str]) -> None:
    if exposure not in self.EXPOSURE:
      raise ValueError(exposure)

    self._shape = shape
    self._exposure = exposure

  def fixed_preprocess(self, image: np.ndarray) -> np.ndarray:
    if self._exposure == 'norm':
      image = (image - np.nanmean(image)) / np.std(image)

    image = tools.normalize_image(image)

    if self._exposure == 'eqhist':
      image = tools.equalize_hist(image)

    return image

  def moving_preprocess(self, image: np.ndarray) -> np.ndarray:
    image = tools.gray_image(image)

    return self.fixed_preprocess(image)


class _ThermalPanorama(ThermalPanorama):
  METRIC = (
      rsitk.Metric.JointHistMI,
      rsitk.Metric.Corr,
      rsitk.Metric.MeanSquare,
  )
  EXPOSURE = _RegistrationPreprocess.EXPOSURE
  BINS = (30, 50, 'auto')

  def __init__(self, directory: Union[str, Path], default_config=False) -> None:
    super().__init__(directory, default_config=default_config)

    self._metric = self.METRIC[0]
    self._exposure = self.EXPOSURE[0]
    self._bins = self.BINS[0]

  @property
  def metric(self):
    return self._metric

  @metric.setter
  def metric(self, value):
    if value not in self.METRIC:
      raise ValueError(value)
    self._metric = value

  @property
  def exposure(self):
    return self._exposure

  @exposure.setter
  def exposure(self, value):
    if value not in self.EXPOSURE:
      raise ValueError(value)
    self._exposure = value

  @property
  def bins(self):
    return self._bins

  @bins.setter
  def bins(self, value):
    if value not in self.BINS:
      raise ValueError(value)
    self._bins = value

  def _init_registrator(self, shape):
    registrator, _ = super()._init_registrator(shape)
    registrator.set_metric(metric=self.metric, bins=self.bins)

    prep = _RegistrationPreprocess(shape=shape, exposure=self._exposure)

    return registrator, prep

  def register(self):
    self.extract()
    self._fm.subdir(DIR.RGST, mkdir=True)

    files = self._fm.raw_files()
    registrator, prep, matrices = None, None, {}
    for file in utils.track(sequence=files, description='Registering...'):
      ir = IIO.read(self._fm.change_dir(DIR.IR, file))
      registrator, prep = self._init_registrator(shape=ir.shape)

      matrix = self._register(file=file, registrator=registrator, prep=prep)
      matrices[file.stem] = matrix

    np.savez(self._fm.rgst_matrix_path(), **matrices)
    logger.success('열화상-실화상 정합 완료')

  def evaluate(self):
    rgst_dir = self._fm.subdir(DIR.RGST)

    for metric, exposure, bins in product(self.METRIC, self.EXPOSURE,
                                          self.BINS):
      logger.info('Metric: "{}", Exposure: "{}", Bins: "{}"', metric, exposure,
                  bins)
      self.metric = metric
      self.exposure = exposure
      self.bins = bins

      self.register()

      subdir = self._fm._wd.joinpath(f'{metric.name}_{exposure}_{bins}')
      subdir.mkdir(exist_ok=True)

      for file in rgst_dir.glob('*'):
        shutil.copy2(file, subdir)

      shutil.rmtree(rgst_dir)


@click.command()
@click.argument('directory', required=True)
def main(directory):
  utils.set_logger(level=20)
  init_project(qt=True)

  tp = _ThermalPanorama(directory=directory)
  tp.evaluate()


if __name__ == '__main__':
  # pylint: disable=no-value-for-parameter
  main()
