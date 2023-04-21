import dataclasses as dc
from functools import cached_property
from io import BytesIO
import re

import numpy as np
from PIL import Image

from pano.misc import subprocess

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

  @classmethod
  def from_dict(cls, d: dict):
    fields = {x.name for x in dc.fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in fields})


class IRSignal:
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
  def temp2signal(temperature, meta: FlirExif):
    eq = np.exp(meta.PlanckB / (temperature - T0))
    eq2 = meta.PlanckR2 * (eq - meta.PlanckF)
    return meta.PlanckR1 / eq2 - meta.PlanckO

  @staticmethod
  def signal2temp(signal, meta: FlirExif):
    rsf = (meta.PlanckR1 / (meta.PlanckR2 * (signal + meta.PlanckO)) +
           meta.PlanckF)

    mask = rsf <= 0.0
    if np.any(mask):
      rsf[mask] = np.e
    else:
      mask = None

    temperature = meta.PlanckB / np.log(rsf) + T0
    if mask is not None:
      temperature[mask] = np.nan

    return temperature

  @classmethod
  def _signal(cls, raw: np.ndarray, meta: FlirExif):
    e = meta.Emissivity
    IRT = meta.IRWindowTransmission
    tau1, tau2 = cls._transmission(meta)

    signal_reflected = (
        cls.temp2signal(meta.ReflectedApparentTemperature, meta) * (1 - e) / e)

    atms = cls.temp2signal(meta.AtmosphericTemperature, meta)
    signal_atm1 = atms * (1 - tau1) / (e * tau1)
    signal_atm2 = atms * (1 - tau2) / (e * tau1 * tau2 * IRT)

    signal_wind = (cls.temp2signal(meta.IRWindowTemperature, meta) * (1 - IRT) /
                   (e * tau1 * IRT))

    signal_obj = ((raw / (e * tau1 * tau2 * IRT)) -
                  (signal_atm1 + signal_atm2 + signal_wind + signal_reflected))

    return signal_obj, signal_reflected

  @classmethod
  def calculate(cls, raw: np.ndarray,
                meta: FlirExif) -> tuple[np.ndarray, float]:
    signal_obj, signal_reflected = cls._signal(raw, meta)
    temperature = cls.signal2temp(signal_obj, meta)

    return temperature, signal_reflected.item()


@dc.dataclass
class FlirData:
  ir: np.ndarray
  vis: np.ndarray
  signal_reflected: float
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
    meta = subprocess.get_exif(files=self._path, tags=self.TAGS)[0]
    meta.pop('SourceFile')

    return FlirExif(**meta)

  def ir(self):
    raw_bytes = subprocess.get_exif_binary(self.path, '-RawThermalImage')
    raw_image = np.array(Image.open(BytesIO(raw_bytes)))

    if self.meta.RawThermalImageType == 'PNG':
      # fix endianness, the bytes in the embedded png are in the wrong order
      raw_image = np.vectorize(lambda x: (x >> 8) + ((x & 0x00ff) << 8))(
          raw_image)
    elif self.meta.RawThermalImageType != 'TIFF':
      raise ValueError('unexpected image type')

    return IRSignal.calculate(raw_image, self.meta)

  def vis(self):
    if self.meta.RawThermalImageType == 'TIFF':
      tag = '-ThumbnailImage'
    else:
      tag = '-EmbeddedImage'

    vis_bytes = subprocess.get_exif_binary(self.path, tag)
    vis_image = Image.open(BytesIO(vis_bytes))

    return np.array(vis_image)

  def exif(self):
    return subprocess.get_exif(self.path)[0]

  def extract(self):
    ir, signal_reflected = self.ir()

    return FlirData(ir=ir,
                    vis=self.vis(),
                    signal_reflected=signal_reflected,
                    exif=self.exif())

  @staticmethod
  def correct_emissivity(image: np.ndarray, meta: FlirExif,
                         signal_reflected: float, e0: float, e1: float):
    signal_obj0 = IRSignal.temp2signal(image, meta=meta)
    signal_obj1 = (((signal_obj0 + signal_reflected) * e0 / e1) -
                   (signal_reflected * (e0 / (1 - e0)) * ((1 - e1) / e1)))

    return IRSignal.signal2temp(signal_obj1, meta)
