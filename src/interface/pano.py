'''
외피 열화상 파노라마 영상처리 알고리즘의 CLI 인터페이스
'''
from pathlib import Path
from typing import List, Optional, Tuple, Union

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


class _DIR:
  # directory
  RAW = 'Raw'
  IR = 'IR'
  VIS = 'VIS'
  PANO = 'Panorama'
  RGST = 'Registration'
  SEG = 'Segmentation'


class _FN:
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

  def __init__(self, directory: Union[str, Path]) -> None:
    # working directory
    wd = Path(directory).resolve()
    if not wd.exists():
      raise FileNotFoundError(wd)
    self._wd = wd

    # config
    self._config = read_config(wd)

    # 파일 형식
    self._exts = (self._config['file']['save_ext']['IR'],
                  self._config['file']['save_ext']['VIS'])

    self._manufacturer = self._check_manufacturer()
    logger.debug('Manufacturer: {}', self._manufacturer)
    if self._manufacturer == 'FLIR':
      self._flir_ext = flir.FlirExtractor()
    else:
      self._flir_ext = None

    # colormap
    self._cmap = get_thermal_colormap(
        name=self._config['color'].get('colormap', 'iron'))

  def _check_manufacturer(self) -> str:
    fopt: DictConfig = self._config['file']
    flir_files = self._glob(folder=_DIR.RAW, pattern=fopt['FLIR']['IR'])
    testo_ir_files = self._glob(folder=_DIR.RAW, pattern=fopt['testo']['IR'])
    testo_vis_files = self._glob(folder=_DIR.RAW, pattern=fopt['testo']['VIS'])

    if testo_ir_files and testo_vis_files:
      manufacturer = 'testo'
    elif flir_files:
      manufacturer = 'FLIR'
    else:
      raise ValueError('Raw 파일 설정 오류')

    return manufacturer

  def _glob(self, folder: str, pattern: str):
    return list(self._dir(folder=folder, mkdir=False).glob(pattern=pattern))

  def _raw_files(self) -> List[Path]:
    # FIXME Raw 폴더에 원본이 없는 경우에도 지원?
    pattern = self._config['file'][self._manufacturer]['IR']
    files = self._glob(folder=_DIR.RAW, pattern=pattern)

    if not files:
      logger.warning('{} 폴더에 영상 파일이 존재하지 않습니다.', _DIR.RAW)

    return files

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
    if d == _DIR.IR:
      files = self._glob(folder=d, pattern=f'*{_FN.NPY}')
      target = '열화상 추출 결과'
    elif d == _DIR.VIS:
      files = self._glob(folder=d, pattern=f'*{_FN.LL}')
      target = '실화상 추출 결과'
    elif d == _DIR.RGST:
      files = self._glob(folder=d, pattern=f'*{_FN.RGST_VIS}{_FN.LL}')
      target = '정합 결과'
    elif d == _DIR.SEG:
      files = self._glob(folder=d, pattern=f'*{_FN.SEG_MASK}{_FN.LL}')
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
    ir_path = self._dir(_DIR.IR).joinpath(fname)
    vis_path = self._dir(_DIR.VIS).joinpath(f'{fname}{self._exts[1]}')

    IIO.save_image_and_meta(path=ir_path,
                            array=ir,
                            exts=[self._exts[0]],
                            meta=meta)
    if self._config['color']['extract_color_image']:
      color_image = apply_colormap(ir, self._cmap)
      IIO.save_image(path=ir_path.with_name(f'{fname}{_FN.COLOR}{_FN.LL}'),
                     array=color_image)

    IIO.save_image(path=vis_path, array=vis)

  def _read_raw_file(self, file: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Raw 열화상, 실화상 파일을 읽음.
    각 해당하는 폴더에 해당 파일이 없으면 입력한 `file`로부터 추출하고 저장.

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

    ir_path = self._dir(_DIR.IR).joinpath(file.stem + self._exts[0])
    vis_path = self._dir(_DIR.VIS).joinpath(file.stem + self._exts[1])

    if ir_path.exists() and vis_path.exists():
      ir = IIO.read_image(ir_path)
      vis = IIO.read_image(vis_path)
    else:
      logger.debug('Extracting `{}`', file.name)

      if self._manufacturer == 'FLIR':
        ir, vis, meta = self._extract_flir_image(path=file)
      elif self._manufacturer == 'testo':
        ir, vis = self._extract_testo_image(path=file)
        meta = None
      else:
        raise ValueError

      self._save_extracted_image(fname=file.stem, ir=ir, vis=vis, meta=meta)

      assert ir is not None

    return ir, vis

  def _iter_raw_files(self):
    files = self._raw_files()
    for file in files:
      ir, vis = self._read_raw_file(file)

      yield ir, vis

  def _read_raw_files(self):
    ir_images = []
    vis_images = []

    for ir, vis in track(self._iter_raw_files(),
                         'Reading...',
                         total=len(self._raw_files()),
                         console=utils.console):
      ir_images.append(ir)
      vis_images.append(vis)

    return ir_images, vis_images

  def _init_stitcher(self) -> stitch.Stitcher:
    sopt: DictConfig = self._config['panorama']['stitch']

    # Stitcher
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
                            names=[x.stem for x in self._raw_files()],
                            crop=self._config['panorama']['stitch']['crop'])

    return res

  def _save_panorama(self,
                     fname: str,
                     spectrum: str,
                     res: stitch.StitchedImage,
                     save_mask=True,
                     save_meta=True):
    pano_dir = self._dir(_DIR.PANO)

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
            exts=['.npy'],
            meta=meta)

        # 적외선 colormap 영상 저장
        color_panorama = apply_colormap(res.panorama, self._cmap)
        IIO.save_image(path=pano_dir.joinpath(f'{fname}{_FN.COLOR}{_FN.LL}'),
                       array=color_panorama)
      else:
        # 실화상 저장
        IIO.save_image_and_meta(path=pano_dir.joinpath(f'{fname}{_FN.LS}'),
                                array=res.panorama,
                                exts=[_FN.LS],
                                meta=meta)

      if save_mask:
        # 마스크 저장
        IIO.save_image(
            path=pano_dir.joinpath(f'{fname}{_FN.PANO_MASK}{_FN.LL}'),
            array=tools.uint8_image(res.mask))

  @staticmethod
  def _pano_target(spectrum):
    if spectrum == 'IR':
      d = _DIR.IR
    elif spectrum == 'VIS':
      d = _DIR.RGST
    else:
      raise ValueError

    return d

  def panorama(self):
    spectrum = self._config['panorama']['target'].upper()
    if not spectrum in ('IR', 'VIS'):
      raise ValueError(spectrum)

    stitcher = self._init_stitcher()

    # Raw 파일 추출
    try:
      self._files(_DIR.IR, error=True)
    except FileNotFoundError:
      self._read_raw_files()

    # 지정한 spectrum 파노라마
    files = self._files(self._pano_target(spectrum))
    images = [IIO.read_image(x) for x in files]

    # 파노라마 생성
    res = self._stitch(stitcher=stitcher, images=images, spectrum=spectrum)

    # 저장
    warp = self._config['panorama']['stitch']['warp']
    self._save_panorama(fname=f'{spectrum}_{warp}', spectrum=spectrum, res=res)

    # 나머지 영상의 파노라마 생성/저장
    spectrum2 = 'VIS' if spectrum == 'IR' else 'IR'
    files2 = self._files(self._pano_target(spectrum2), error=True)
    files2 = [files2[x] for x in res.indices]  # FIXME 파일 이름과 대조
    images2 = [IIO.read_image(x) for x in files2]

    panorama2, _ = stitcher.warp_and_blend(
        images=stitch.StitchingImages(arrays=images2),
        cameras=res.cameras,
        masks=None,
        names=None)
    if res.crop_range:
      panorama2, _, _ = stitcher.crop(panorama2,
                                      mask=None,
                                      crop_range=res.crop_range)
    res.panorama = np.round(panorama2).astype(np.uint8)

    self._save_panorama(fname=f'{spectrum2}_{warp}',
                        spectrum=spectrum2,
                        res=res,
                        save_mask=False,
                        save_meta=False)

    logger.info('파노라마 생성 완료')
