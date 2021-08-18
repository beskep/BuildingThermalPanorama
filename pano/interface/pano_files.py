"""
ThermalPanorama 프로젝트 폴더/파일 목록 관리

FLIR 카메라로 촬영한 프로젝트의 경우 폴더 구조

ProjectDir
├── Raw
│   ├── file01.jpg
│   ⁝
├── 00 IR
│   ├── file01.npy
│   ├── file01_color.png
│   ├── file01_meta.yaml
│   ⁝
├── 00 VIS
│   ├── file01.png
│   ⁝
├── 01 Registration
│   ├── transform_matrix.npz
│   ├── file01.png
│   ├── file01_compare.png
│   ⁝
├── 02 Segmentation
│   ├── file01.png
│   ├── file01_fig.jpg
│   ⁝
├── 03 Panorama
│   ├── PanoramaIR.npy
│   ├── PanoramaIR_color.png
│   ├── PanoramaIR_meta.yaml
│   ├── PanoramaMask.png
│   ├── PanoramaSegmentation.png
│   └── PanoramaVIS.jpg
├── 04 Correction
│   ├── CorrectionProcess.jpg
│   ├── PanoramaIR.npy
│   ├── PanoramaIR_color.png
│   ├── PanoramaMask.png
│   ├── PanoramaSegmentation.png
│   └── PanoramaVIS.jpg
└── config.yaml
"""

from enum import Enum
from pathlib import Path
from typing import List, Optional, Union

from pano import utils

MODEL_PATH = utils.DIR.RESOURCE.joinpath('DeepLabV3/frozen_inference_graph.pb')


class DIR(Enum):
  RAW = 'Raw'
  IR = '00 IR'
  VIS = '00 VIS'
  RGST = '01 Registration'
  SEG = '02 Segmentation'
  PANO = '03 Panorama'
  COR = '04 Correction'


class FN:
  """Files suffix, ext"""
  NPY = '.npy'
  LL = '.png'  # Lossless
  LS = '.jpg'  # Loss

  COLOR = '_color'

  RGST_CMPR = '_compare'
  SEG_FIG = '_fig'


class SP(Enum):
  """Spectrum, image type"""
  IR = 'IR'
  VIS = 'VIS'
  SEG = 'Segmentation'
  MASK = 'Mask'


def _dir(d: Union[str, DIR]) -> DIR:
  if isinstance(d, str):
    d = DIR[d]

  return d


class ThermalPanoramaFileManager:

  def __init__(self, directory, raw_pattern: Optional[str] = None) -> None:
    self._wd = Path(directory)
    self._raw_pattern = raw_pattern

  def set_raw_pattern(self, pattern: str):
    self._raw_pattern = pattern

  def subdir(self, d: Union[str, DIR], mkdir=False):
    d = _dir(d)

    subdir = self._wd.joinpath(d.value)
    if mkdir and not subdir.exists():
      subdir.mkdir()

    return subdir

  def glob(self, d: Union[str, DIR], pattern: str):
    return list(self.subdir(d=d).glob(pattern=pattern))

  def raw_files(self) -> List[Path]:
    if self._raw_pattern is None:
      raise ValueError('Raw pattern not set')

    return self.glob(d=DIR.RAW, pattern=self._raw_pattern)

  def files(self, d: Union[str, DIR]) -> List[Path]:
    """
    프로젝트 경로 내 각 경로 중 수치 정보를 담고 있는 파일 목록.
    Raw 파일 목록과 대응하는 파일이 하나라도 존재하지 않으면 오류 발생.

    Parameters
    ----------
    d : Union[str, DIR]
        대상 directory.
        {RAW, IR, VIS, RGST, SEG}.

    Returns
    -------
    List[Path]

    Raises
    ------
    ValueError
        if d not in (DIR.RAW, DIR.IR, DIR.VIS, DIR.RGST, DIR.SEG)
    FileNotFoundError
        if d or any file not exist
    """
    d = _dir(d)
    if d not in (DIR.RAW, DIR.IR, DIR.VIS, DIR.RGST, DIR.SEG):
      raise ValueError(
          f'Available folders: {{RAW, IR, VIS, RGST, SEG}}, got {d}')

    raw_files = self.raw_files()
    if d is DIR.RAW:
      return raw_files

    subdir = self.subdir(d=d)
    if not subdir.exists():
      raise FileNotFoundError(subdir)

    ext = FN.NPY if d is DIR.IR else FN.LL
    files = [subdir.joinpath(f'{x.stem}{ext}') for x in raw_files]

    for file in files:
      if not file.exists():
        raise FileNotFoundError(file)

    return files

  def change_dir(self, d: Union[str, DIR], file: Union[str, Path]) -> Path:
    """
    다른 directory의 대응되는 파일 경로 반환.
    이미지 처리 결과를 저장할 때 사용.

    Parameters
    ----------
    d : DIR
        대상 directory.
        {IR, VIS, RGST, SEG}.
    file : Union[str, Path]
        (Raw 파일의) file name 혹은 Path.

    Returns
    -------
    Path

    Raises
    ------
    ValueError
        if d not in (DIR.IR, DIR.VIS, DIR.RGST, DIR.SEG)
    """
    d = _dir(d)
    if d not in (DIR.IR, DIR.VIS, DIR.RGST, DIR.SEG):
      raise ValueError(f'Available folders: {{IR, VIS, RGST, SEG}}, got {d}')

    subdir = self.subdir(d=d)
    ext = FN.NPY if d is DIR.IR else FN.LL
    if isinstance(file, Path):
      file = file.stem

    return subdir.joinpath(f'{file}{ext}')

  @staticmethod
  def color_path(path: Path):
    return path.with_name(f'{path.stem}{FN.COLOR}{FN.LL}')

  def rgst_matrix_path(self):
    return self.subdir(DIR.RGST).joinpath('transformation_matrix')

  @staticmethod
  def segment_model_path():
    if not MODEL_PATH.exists():
      raise FileNotFoundError(MODEL_PATH)

    return MODEL_PATH

  def panorama_path(self, d: DIR, sp: SP, error=False):
    subdir = self.subdir(d)
    ext = {SP.IR: FN.NPY, SP.VIS: FN.LS, SP.SEG: FN.LL, SP.MASK: FN.LL}[sp]
    path = subdir.joinpath(f'Panorama{sp.value}{ext}')

    if error and not path.exists():
      raise FileNotFoundError(path)

    return path

  def correction_plot_path(self):
    return self.subdir(DIR.COR).joinpath(f'CorrectionProcess{FN.LS}')
