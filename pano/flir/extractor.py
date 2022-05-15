import dataclasses as dc
from functools import cached_property
from io import BytesIO
import re

import numpy as np
from PIL import Image

from pano.misc import exif

T0 = -273.15


def _float(string):
  digits = re.findall(r'[-+]?\d*\.\d+|\d+', string)

  return float(digits[0])


@dc.dataclass
class FlirExif:
  AtmosphericTemperature: float
  Emissivity: float
  IRWindowTemperature: float
  IRWindowTransmission: float
  PlanckB: float
  PlanckF: float
  PlanckO: float
  PlanckR1: float
  PlanckR2: float
  RawThermalImageType: str
  ReflectedApparentTemperature: float
  RelativeHumidity: float
  SubjectDistance: float = 1.0

  def __post_init__(self):
    for field in dc.fields(self):
      value = getattr(self, field.name)

      if field.name == 'RawThermalImageType':
        setattr(self, field.name, value.strip().upper())
      elif isinstance(value, str):
        setattr(self, field.name, _float(value))

    # pylint: disable=no-member
    if self.RelativeHumidity > 1:
      self.RelativeHumidity /= 100


class Raw2Temperature:
  ATA1 = 0.006569
  ATA2 = 0.01262
  ATB1 = -0.002276
  ATB2 = -0.00667
  ATX = 1.9

  @classmethod
  def _transmission(cls, meta: FlirExif):
    h2o = (meta.RelativeHumidity *
           np.exp(1.5587 + 0.06939 * meta.AtmosphericTemperature -
                  0.00027816 * meta.AtmosphericTemperature**2 +
                  0.00000068455 * meta.AtmosphericTemperature**3))

    tau1 = (cls.ATX * np.exp(-np.sqrt(meta.SubjectDistance / 2) *
                             (cls.ATA1 + cls.ATB1 * np.sqrt(h2o))) +
            (1 - cls.ATX) * np.exp(-np.sqrt(meta.SubjectDistance / 2) *
                                   (cls.ATA2 + cls.ATB2 * np.sqrt(h2o))))

    tau2 = (cls.ATX * np.exp(-np.sqrt(meta.SubjectDistance / 2) *
                             (cls.ATA1 + cls.ATB1 * np.sqrt(h2o))) +
            (1 - cls.ATX) * np.exp(-np.sqrt(meta.SubjectDistance / 2) *
                                   (cls.ATA2 + cls.ATB2 * np.sqrt(h2o))))

    return tau1, tau2

  @staticmethod
  def _radiance_eq(meta: FlirExif, temperature: float):
    eq = np.exp(meta.PlanckB / (temperature - T0))
    eq2 = meta.PlanckR2 * (eq - meta.PlanckF)
    return meta.PlanckR1 / eq2 - meta.PlanckO

  @classmethod
  def _radiance(cls, raw: np.ndarray, meta: FlirExif):
    e = meta.Emissivity
    IRT = meta.IRWindowTransmission
    # emiss_wind = 1 - IRT
    tau1, tau2 = cls._transmission(meta)

    raw_refl1 = (cls._radiance_eq(meta, meta.ReflectedApparentTemperature) *
                 (1 - e) / e)
    # raw_refl2는 항상 0이 되어 생략

    ratm = cls._radiance_eq(meta, meta.AtmosphericTemperature)
    raw_atm1 = ratm * (1 - tau1) / (e * tau1)
    raw_atm2 = ratm * (1 - tau2) / (e * tau1 * tau2 * IRT)

    raw_wind = (cls._radiance_eq(meta, meta.IRWindowTemperature) * (1 - IRT) /
                (e * tau1 * IRT))

    raw_obj = ((raw / (e * tau1 * tau2 * IRT)) -
               (raw_atm1 + raw_atm2 + raw_wind + raw_refl1))

    return raw_obj, raw_refl1

  @staticmethod
  def rawobj2temperature(raw_obj: np.ndarray, meta: FlirExif):
    RSob = (meta.PlanckR1 / (meta.PlanckR2 * (raw_obj + meta.PlanckO)) +
            meta.PlanckF)

    mask = np.min(RSob) <= 0.0
    if np.any(mask):
      RSob[mask] = np.e
    else:
      mask = None

    temperature = meta.PlanckB / np.log(RSob) + T0
    if mask is not None:
      temperature[mask] = np.nan

    return temperature

  @classmethod
  def calculate(cls, raw: np.ndarray,
                meta: FlirExif) -> tuple[np.ndarray, float]:
    raw_obj, raw_refl = cls._radiance(raw, meta)
    temperature = cls.rawobj2temperature(raw_obj, meta)

    return temperature, raw_refl.item()


@dc.dataclass
class FlirData:
  ir: np.ndarray
  vis: np.ndarray
  raw_refl: float  # TODO 정확한 표기
  exif: dict


class FlirExtractor:
  TAGS = (
      '-AtmosphericTemperature',
      '-Emissivity',
      '-IRWindowTemperature',
      '-IRWindowTransmission',
      '-PlanckB',
      '-PlanckF',
      '-PlanckO',
      '-PlanckR1',
      '-PlanckR2',
      '-RawThermalImageType',
      '-ReflectedApparentTemperature',
      '-RelativeHumidity',
      '-SubjectDistance',
  )

  def __init__(self, path: str) -> None:
    self._path = path

  @property
  def path(self):
    return self._path

  @cached_property
  def meta(self):
    meta = exif.get_exif(files=self._path, tags=self.TAGS)[0]
    meta.pop('SourceFile')

    return FlirExif(**meta)

  def ir(self):
    raw_bytes = exif.get_exif_binary(self.path, '-RawThermalImage')
    raw_image = np.array(Image.open(BytesIO(raw_bytes)))

    if self.meta.RawThermalImageType == 'PNG':
      # fix endianness, the bytes in the embedded png are in the wrong order
      raw_image = np.vectorize(lambda x: (x >> 8) + ((x & 0x00ff) << 8))(
          raw_image)
    elif self.meta.RawThermalImageType != 'TIFF':
      raise ValueError('unexpected image type')

    return Raw2Temperature.calculate(raw_image, self.meta)

  def vis(self):
    if self.meta.RawThermalImageType == 'TIFF':
      tag = '-ThumbnailImage'
    else:
      tag = '-EmbeddedImage'

    vis_bytes = exif.get_exif_binary(self.path, tag)
    vis_image = Image.open(BytesIO(vis_bytes))

    return np.array(vis_image)

  def exif(self):
    return exif.get_exif(self.path)[0]

  def extract(self):
    ir, refl = self.ir()

    return FlirData(ir=ir, vis=self.vis(), raw_refl=refl, exif=self.exif())

  @staticmethod
  def correct_emissivity(image: np.ndarray, meta: FlirExif, raw_refl: float,
                         e0: float, e1: float):
    RSob = np.exp(meta.PlanckB / (image - T0))
    raw_obj0 = (meta.PlanckR1 / ((RSob - meta.PlanckF) * meta.PlanckR2) -
                meta.PlanckO)

    # TODO 검토
    raw_obj1 = (((raw_obj0 + raw_refl) * e0 / e1) -
                (raw_refl * (e0 / (1 - e0)) * ((1 - e1) / e1)))

    return Raw2Temperature.rawobj2temperature(raw_obj1, meta)
