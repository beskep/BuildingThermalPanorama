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
│   ├── file01_compare.jpg
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
├── 05 Analysis
│   ├── PanoramaIR.npy
│   ⁝
├── 06 Output
│   ├── Edgelets.jpg
│   ⁝
└── config.yaml
"""

from collections.abc import Iterable
from enum import Enum
from pathlib import Path
from shutil import copy2
from typing import ClassVar

from loguru import logger
from omegaconf import DictConfig

from pano import utils

from .config import set_config

MODEL_PATH = utils.DIR.RESOURCE / 'DeepLabV3PlusEfficientnetB6.onnx'


class DIR(Enum):
  RAW = 'Raw'
  IR = '00 IR'
  VIS = '00 VIS'
  RGST = '01 Registration'
  SEG = '02 Segmentation'
  PANO = '03 Panorama'
  COR = '04 Correction'
  ANLY = '05 Analysis'
  OUT = '06 Output'


class FN:
  """Files suffix, ext"""

  NPY = '.npy'
  LL = '.png'  # Lossless
  LS = '.jpg'  # Loss

  COLOR = '_color'

  RGST_CMPR = '_compare'
  RGST_AUTO = '_automatic'
  RGST_MANUAL = '_manual'
  SEG_FIG = '_fig'


class SP(Enum):
  """Spectrum, image type"""

  IR = 'IR'
  VIS = 'VIS'
  SEG = 'Segmentation'
  MASK = 'Mask'
  TF = 'TemperatureFactor'


def _dir(d: str | DIR) -> DIR:
  if isinstance(d, str):
    d = DIR[d]

  return d


class WorkingDirNotSetError(FileNotFoundError):
  def __str__(self) -> str:
    return self.args[0] if self.args else '프로젝트 폴더가 지정되지 않았습니다.'


class ThermalPanoramaFileManager:
  SP_EXT: ClassVar[dict[SP, str]] = {
    SP.IR: FN.NPY,
    SP.VIS: FN.LS,
    SP.SEG: FN.LL,
    SP.MASK: FN.LL,
    SP.TF: FN.NPY,
  }

  def __init__(self, directory, raw_pattern='*.jpg') -> None:
    self._wd = Path(directory)
    self._raw_pattern = raw_pattern

  @property
  def wd(self):
    return self._wd

  @property
  def raw_pattern(self):
    return self._raw_pattern

  @raw_pattern.setter
  def raw_pattern(self, value: str):
    self._raw_pattern = value

  def subdir(self, d: str | DIR, *, mkdir=False):
    d = _dir(d)

    subdir = self._wd.joinpath(d.value)
    if mkdir and not subdir.exists():
      subdir.mkdir()

    return subdir

  def glob(self, d: str | DIR, pattern: str):
    return list(self.subdir(d=d).glob(pattern=pattern))

  def raw_files(self) -> list[Path]:
    if self._raw_pattern is None:
      msg = 'Raw pattern not set'
      raise ValueError(msg)

    return self.glob(d=DIR.RAW, pattern=self._raw_pattern)

  def files(self, d: str | DIR, *, error=True) -> list[Path]:
    """
    프로젝트 경로 내 각 경로 중 수치 정보를 담고 있는 파일 목록.

    `error`가 `True`이고 Raw 파일 목록과 대응하는 파일이
    하나라도 존재하지 않으면 오류 발생.

    Parameters
    ----------
    d : Union[str, DIR]
        대상 directory.
        {RAW, IR, VIS, RGST, SEG}.
    error : bool
        `True`이면 대상 폴더에 Raw 파일 목록과 일치하지 않는 경우 오류 발생

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
    if d not in {DIR.RAW, DIR.IR, DIR.VIS, DIR.RGST, DIR.SEG}:
      msg = f'Available folders: {{RAW, IR, VIS, RGST, SEG}}, got {d}'
      raise ValueError(msg)

    raw_files = self.raw_files()
    if d is DIR.RAW:
      return raw_files

    subdir = self.subdir(d=d)
    if not subdir.exists():
      raise FileNotFoundError(subdir)

    ext = FN.NPY if d is DIR.IR else FN.LL
    files = [subdir.joinpath(f'{x.stem}{ext}') for x in raw_files]

    if error:
      for file in files:
        if not file.exists():
          raise FileNotFoundError(file)
    elif d in {DIR.VIS, DIR.SEG} and any(not x.exists() for x in files):
      # 다른 실화상을 입력한 경우, VIS/SEG 폴더에 존재하는 영상 목록 반환
      exts = {'.png'} if d is DIR.SEG else {'.png', '.jpg', '.webp'}
      files = [
        x
        for x in subdir.glob('*')
        if x.is_file() and x.suffix.lower() in exts and not x.name.endswith('_fig.png')
      ]

    if not files:
      msg = f'no file in {subdir}'
      raise FileNotFoundError(msg, subdir)

    return files

  def change_dir(self, d: str | DIR, file: str | Path) -> Path:
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
    if d not in {DIR.IR, DIR.VIS, DIR.RGST, DIR.SEG}:
      msg = f'Available folders: {{IR, VIS, RGST, SEG}}, got {d}'
      raise ValueError(msg)

    subdir = self.subdir(d=d)
    ext = FN.NPY if d is DIR.IR else FN.LL
    if isinstance(file, Path):
      file = file.stem

    return subdir.joinpath(f'{file}{ext}')

  @staticmethod
  def color_path(path: Path):
    return path.with_name(f'{path.stem}{FN.COLOR}{FN.LL}')

  def rgst_matrix_path(self):
    """
    자동 열/실화상 정합 결과인 transform matrix 저장 경로.

    실화상을 열화상 크기에 맞게 resize 후 matrix를 적용 필요.

    Returns
    -------
    Path
    """
    return self.subdir(DIR.RGST) / 'transformation_matrix.npz'

  @staticmethod
  def segment_model_path():
    if not MODEL_PATH.exists():
      raise FileNotFoundError(MODEL_PATH)

    return MODEL_PATH

  def panorama_path(self, d: DIR, sp: SP, *, error=False):
    subdir = self.subdir(d)
    prefix = '' if sp is SP.TF else 'Panorama'
    ext = self.SP_EXT[sp]
    path = subdir.joinpath(f'{prefix}{sp.value}{ext}')

    if error and not path.exists():
      raise FileNotFoundError(path)

    return path

  def correction_plot_path(self):
    return self.subdir(DIR.COR).joinpath(f'CorrectionProcess{FN.LS}')

  def anomaly_path(self):
    return self._wd / 'AnomalyThreshold.yaml'

  def add_images(self, paths: Iterable[str | Path]):
    raw_dir = self.wd / DIR.RAW.value
    raw_dir.mkdir(exist_ok=True)

    for src in paths:
      dst = raw_dir / Path(src).name

      if dst.exists():
        logger.warning('Raw 파일을 덮어씌웁니다: "{}"', dst)
      else:
        logger.debug('copy to "{}"', dst)

      copy2(src, dst)

  def replace_visual_images(self, paths: Iterable[str | Path]):
    """
    동시 추출된 실화상 대신 새 실화상 입력.

    Raises
    ------
    FileNotFoundError
    """
    if not (paths := list(paths)):
      msg = '선택된 파일이 없습니다.'
      raise FileNotFoundError(msg) from None

    p: str | Path

    # 이미 추출된 실화상을 IR 폴더로 옮김
    ir_dir = self.subdir(DIR.IR)
    for p in self.files(DIR.VIS, error=False):
      logger.debug('move "{}" to IR', p)
      p.replace(ir_dir / f'{p.stem}_vis{p.suffix}')

    # 새 실화상 파일 VIS 폴더에 복사
    vis_dir = self.subdir(DIR.VIS)
    for p in paths:
      logger.debug('copy "{}" to VIS', p)
      copy2(p, vis_dir)


class ImageNotFoundError(FileNotFoundError):
  pass


def init_directory(
  directory: Path,
  *,
  default_config=False,
  copy=True,
  raise_empty=True,
) -> DictConfig:
  """
  Working directory 초기화.

  대상 directory에 RAW 폴더가 존재하지 않는 경우,
  RAW 폴더를 생성하고 영상/엑셀 파일을 옮김.
  default config 파일 저장.

  Parameters
  ----------
  directory : Path
      작업 폴더
  default_config : bool, optional
      `True`면 기본 설정 적용.
  copy : bool, optional
      `True`면 영상 파일을 복사, `False`면 파일 이동.
  raise_empty : bool, optional
      `True`고 작업 폴더에 영상 파일이 없으면 `ImageNotFoundError` 발생.

  Returns
  -------
  DictConfig

  Raises
  ------
  ImageNotFoundError
      `raise_empty`가 `True`고 작업 폴더에 영상 파일이 없는 경우 raise.
  """
  if not (raw_dir := directory / DIR.RAW.value).exists():
    files = [
      x
      for x in directory.glob('*')
      if x.is_file() and x.suffix.lower() in {'.jpg', '.xlsx', '.png'}
    ]

    if not files and raise_empty:
      msg = f'영상 파일이 발견되지 않았습니다.: "{directory}"'
      raise ImageNotFoundError(msg) from None

    if files:
      raw_dir.mkdir()

    for file in files:
      if copy:
        copy2(file, raw_dir / file.name)
      else:
        file.replace(raw_dir / file.name)

  return set_config(directory=directory, default=default_config)
