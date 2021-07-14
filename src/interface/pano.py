"""외피 열화상 파노라마 영상처리 알고리즘의 CLI 인터페이스"""

from pathlib import Path
from typing import List, Optional, Union

import utils

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from rich.progress import track

import flir
import stitch
from misc import exif, tools
from misc.tools import ImageIO as IIO

from .cmap import apply_colormap, get_thermal_colormap
from .config import DictConfig, read_config


class DIR:
  # directory
  RAW = 'Raw'
  IR = '00 IR'
  VIS = '00 VIS'
  RGST = '01 Registration'
  SEG = '02 Segmentation'
  PANO = '03 Panorama'
  COR = '04 Correction'


class FN:
  # file suffix, ext

  NPY = '.npy'
  LL = '.png'  # Lossless
  LS = '.jpg'  # Loss

  COLOR = '_color'

  RGST_VIS = '_vis'
  RGST_CMPR = '_compare'

  SEG_MASK = '_mask'
  SEG_VIS = '_vis'
  SEG_FIG = '_fig'

  PANO_MASK = '_mask'


class ThermalPanorama:
  # TODO wd에 각 명령어 성공 여부 log 저장

  def __init__(self, directory: Union[str, Path], default_config=False) -> None:
    # working directory
    wd = Path(directory).resolve()
    if not wd.exists():
      raise FileNotFoundError(wd)
    self._wd = wd

    # config
    self._config = read_config(wd=wd, default=default_config)

    self._manufacturer = self._check_manufacturer()
    logger.debug('Manufacturer: {}', self._manufacturer)
    if self._manufacturer == 'FLIR':
      self._flir_ext = flir.FlirExtractor()
    else:
      self._flir_ext = None

    # colormap
    self._cmap = get_thermal_colormap(
        name=self._config['color'].get('colormap', 'iron'))

  @property
  def _size_limit(self):
    return self._config['file']['size_limit']

  def _check_manufacturer(self) -> str:
    fopt: DictConfig = self._config['file']
    flir_files = self._glob(folder=DIR.RAW, pattern=fopt['FLIR']['IR'])
    testo_ir_files = self._glob(folder=DIR.RAW, pattern=fopt['testo']['IR'])
    testo_vis_files = self._glob(folder=DIR.RAW, pattern=fopt['testo']['VIS'])

    if testo_ir_files and testo_vis_files:
      manufacturer = 'testo'
    elif flir_files:
      manufacturer = 'FLIR'
    else:
      raise ValueError('Raw 파일 설정 오류')

    return manufacturer

  def _glob(self, folder: str, pattern: str):
    return list(self._dir(folder=folder, mkdir=False).glob(pattern=pattern))

  def _files(self, d, error=False):
    """
    각 폴더에서 수치 정보를 담고 있는 파일 목록 반환

    Parameters
    ----------
    d : str
        대상 폴더
    warn : bool, optional
        `True`이고 파일 목록이 없을 경우 warning
    error : bool, optional
        `True`이고 파일 목록이 없을 경우 `FileNotFoundError`

    Returns
    -------
    List[Path]
    """
    if d == DIR.RAW:
      pattern = self._config['file'][self._manufacturer]['IR']
      files = self._glob(folder=DIR.RAW, pattern=pattern)
      target = 'Raw files'
    elif d == DIR.IR:
      files = self._glob(folder=d, pattern=f'*{FN.NPY}')
      target = '열화상 추출 결과'
    elif d == DIR.VIS:
      files = self._glob(folder=d, pattern=f'*{FN.LL}')
      target = '실화상 추출 결과'
    elif d == DIR.RGST:
      files = self._glob(folder=d, pattern=f'*{FN.RGST_VIS}{FN.LL}')
      target = '정합 결과'
    elif d == DIR.SEG:
      files = self._glob(folder=d, pattern=f'*{FN.SEG_MASK}{FN.LL}')
      target = '분할 결과'
    else:
      raise ValueError

    if not files:
      msg = f'{target}가 존재하지 않습니다.'

      if error:
        raise FileNotFoundError(msg)

      logger.warning(f'{target}가 존재하지 않습니다.')

    return files

  @property
  def _working_dir(self) -> Path:
    return self._wd

  def _dir(self, folder: str, mkdir=True):
    d = self._working_dir.joinpath(folder)
    if mkdir and not d.exists():
      d.mkdir()

    return d

  def _extract_flir_image(self, path: Path):
    ir, vis = self._flir_ext.extract_data(path)
    meta = {'Exif': exif.get_exif_tags(path.as_posix())[0]}

    # FIXME 임시로 만듬 - Exif Orientation 태그 정보에 따라 회전하기
    if ir.shape[0] > ir.shape[1]:
      logger.debug('rot90 ({})', path.name)
      ir = np.rot90(ir, k=1, axes=(0, 1))
      vis = np.rot90(vis, k=1, axes=(0, 1))

    return ir, vis, meta

  def _extract_testo_image(self, path: Path):
    vis_suffix = self._config['file']['testo']['VIS'].replace('*', '')
    vis_path = path.with_name(path.stem + vis_suffix)
    if not vis_path.exists():
      raise FileNotFoundError(vis_path)

    ir = IIO.read_image(path=path)
    vis = IIO.read_image(path=vis_path)

    return ir, vis

  def _save_extracted_image(self,
                            fname: str,
                            ir: np.ndarray,
                            vis: np.ndarray,
                            meta: Optional[dict] = None):
    """추출한 열/실화상을 각 폴더에 저장"""
    ir_path = self._dir(DIR.IR).joinpath(fname)
    vis_path = self._dir(DIR.VIS).joinpath(f'{fname}{FN.LL}')

    IIO.save_image_and_meta(path=ir_path, array=ir, exts=[FN.NPY], meta=meta)
    if self._config['color']['extract_color_image']:
      color_image = apply_colormap(ir, self._cmap)
      IIO.save_image(path=ir_path.with_name(f'{fname}{FN.COLOR}{FN.LL}'),
                     array=color_image)

    IIO.save_image(path=vis_path, array=vis)

  def _extract_raw_file(self, file: Path):
    """
    Raw 열화상, 실화상 파일 추출.

    Parameters
    ----------
    file : Path
        Raw 파일 경로

    Returns
    -------
    np.ndarray
        열화상
    np.ndarray
        실화상
    """
    if not file.exists():
      raise FileNotFoundError(file)

    ir_path = self._dir(DIR.IR).joinpath(f'{file.stat}{FN.NPY}')
    vis_path = self._dir(DIR.VIS).joinpath(f'{file.stat}{FN.LL}')

    if ir_path.exists() and vis_path.exists():
      return

    logger.debug('Extracting `{}`', file.name)

    if self._manufacturer == 'FLIR':
      ir, vis, meta = self._extract_flir_image(path=file)
    elif self._manufacturer == 'testo':
      ir, vis = self._extract_testo_image(path=file)
      meta = None
    else:
      raise ValueError

    assert ir is not None
    self._save_extracted_image(fname=file.stem, ir=ir, vis=vis, meta=meta)

  def _extract_raw_files(self):
    """기존에 raw 파일로부터 추출한 실화상/열화상이 없는 경우 추출 시행"""
    try:
      self._files(DIR.IR, error=True)
      self._files(DIR.VIS, error=True)
    except FileNotFoundError:
      files = self._files(DIR.RAW, error=True)
      for file in track(files,
                        description='Extracting images...',
                        console=utils.console):
        self._extract_raw_file(file=file)

  def _init_stitcher(self) -> stitch.Stitcher:
    sopt: DictConfig = self._config['panorama']['stitch']

    stitcher = stitch.Stitcher(mode=sopt['perspective'],
                               compose_scale=sopt['compose_scale'],
                               work_scale=sopt['work_scale'],
                               warp_threshold=sopt['warp_threshold'])
    stitcher.warper_type = sopt['warp']

    logger.debug('Stitching: Stitcher 초기화')

    return stitcher

  def _stitch(
      self,
      stitcher: stitch.Stitcher,
      images: List[np.ndarray],
      names: List[str],
      spectrum: str,
  ) -> stitch.StitchedImage:
    popt: DictConfig = self._config['panorama']['preprocess'][spectrum]

    # 전처리 정의
    prep = stitch.PanoramaPreprocess(is_numeric=(images[0].ndim == 2),
                                     mask_threshold=popt['masking_threshold'],
                                     contrast=popt['contrast'],
                                     denoise=popt['denoise'])
    if 'bilateral_args' in popt:
      prep.set_bilateral_args(**popt['bilateral_args'])
    if 'gaussian_args' in popt:
      prep.set_gaussian_args(**popt['gaussian_args'])

    # 대상 영상
    stitching_images = stitch.StitchingImages(arrays=images)
    stitching_images.set_preprocess(prep)
    logger.debug('Stitching: 대상 영상 & 전처리 설정')

    with utils.console.status('Stitching...'):
      res = stitcher.stitch(images=stitching_images,
                            masks=None,
                            names=names,
                            crop=self._config['panorama']['stitch']['crop'])

    return res

  def _save_panorama(self,
                     fname: str,
                     spectrum: str,
                     res: stitch.StitchedImage,
                     save_mask=True,
                     save_meta=True):
    pano_dir = self._dir(DIR.PANO)

    if save_meta:
      meta = {'panorama': {'graph': res.graph, 'image_indices': res.indices}}
    else:
      meta = None

    with utils.console.status('Saving...'):
      if spectrum == 'IR':
        # 적외선 수치, meta 정보 저장
        IIO.save_image_and_meta(
            path=pano_dir.joinpath(fname),
            array=res.panorama.astype(np.float16),  # TODO 추출부터 float16으로?
            exts=[FN.NPY],
            meta=meta)

        # 적외선 colormap 영상 저장
        color_panorama = apply_colormap(res.panorama, self._cmap)
        IIO.save_image(path=pano_dir.joinpath(f'{fname}{FN.COLOR}{FN.LL}'),
                       array=color_panorama)
      else:
        # 실화상 저장
        IIO.save_image_and_meta(path=pano_dir.joinpath(f'{fname}{FN.LS}'),
                                array=res.panorama,
                                exts=[FN.LS],
                                meta=meta)

      if save_mask:
        # 마스크 저장
        IIO.save_image(path=pano_dir.joinpath(f'{fname}{FN.PANO_MASK}{FN.LL}'),
                       array=tools.uint8_image(res.mask))

  @staticmethod
  def _pano_target(spectrum):
    if spectrum == 'IR':
      d = DIR.IR
    elif spectrum == 'VIS':
      d = DIR.RGST
    else:
      raise ValueError

    return d

  def panorama(self):
    # FIXME 함수 정리, size_limit 적용
    spectrum = self._config['panorama']['target'].upper()
    stitcher = self._init_stitcher()

    # Raw 파일 추출
    self._extract_raw_files()

    # 지정한 spectrum 파노라마
    files = self._files(self._pano_target(spectrum))
    images = [IIO.read_image(x) for x in files]

    # 파노라마 생성
    res = self._stitch(stitcher=stitcher,
                       images=images,
                       names=[x.stem for x in files],
                       spectrum=spectrum)

    # 저장
    warp = self._config['panorama']['stitch']['warp']
    self._save_panorama(fname=f'{spectrum}_{warp}', spectrum=spectrum, res=res)
