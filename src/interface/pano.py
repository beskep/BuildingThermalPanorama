"""외피 열화상 파노라마 영상처리 알고리즘의 CLI 인터페이스"""

from pathlib import Path
from typing import List, Optional, Union

import utils

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from rich.progress import track

import flir
import registration.registrator.simpleitk as rsitk
import stitch
from distortion import perspective as persp
from misc import exif, tools
from misc.tools import ImageIO as IIO
from misc.tools import limit_image_size as limit_size

from .cmap import apply_colormap, get_thermal_colormap
from .config import DictConfig, read_config


class DIR:
  """Working directory"""
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

  RGST_VIS = '_vis'
  RGST_CMPR = '_compare'

  SEG_MASK = '_mask'
  SEG_FIG = '_fig'

  PANO_MASK = '_mask'


class ThermalPanorama:
  _SP_DIR = {'IR': DIR.IR, 'VIS': DIR.RGST}
  _SP_KOR = {'IR': '열화상', 'VIS': '실화상'}

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
      self._flir_ext: Optional[flir.FlirExtractor] = flir.FlirExtractor()
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

  def _files(self, d: str) -> List[Path]:
    """
    각 폴더에서 수치 정보를 담고 있는 파일 목록 반환.
    RAW 폴더의 경우 열화상 파일 (FLIR: `.jpg`, testo: `.xlsx`).

    Parameters
    ----------
    d : str
        대상 폴더

    Returns
    -------
    List[Path]

    Raises
    ------
    ValueError
        d not in `DIR`
    FileNotFoundError
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
      raise FileNotFoundError(f'{target}가 존재하지 않습니다.')

    return files

  def _dir(self, folder: str, mkdir=True):
    d = self._wd.joinpath(folder)
    if mkdir and not d.exists():
      d.mkdir()

    return d

  def _extract_flir_image(self, path: Path):
    assert self._flir_ext is not None
    ir, vis = self._flir_ext.extract_data(path)
    meta = {'Exif': exif.get_exif_tags(path.as_posix())[0]}

    if self._config['file']['force_horizontal'] and (ir.shape[0] > ir.shape[1]):
      # 수직 영상을 수평으로 만들기 위해 90도 회전
      logger.debug('rot90 `{}`', path.name)
      ir = np.rot90(ir, k=1, axes=(0, 1))
      vis = np.rot90(vis, k=1, axes=(0, 1))

    # FLIR One로 찍은 사진은 수직으로 촬영해도 orientation 번호가
    # `1` (`Horizontal (normal)`)로 표시되어서 아래 정보가 쓸모가 없음...

    # tag = exif.get_exif_tags(path.as_posix(), '-Orientation', '-n')
    # orientation = tag['Orientation']

    return ir, vis, meta

  def _extract_testo_image(self, path: Path):
    vis_suffix = self._config['file']['testo']['VIS'].replace('*', '')
    vis_path = path.with_name(path.stem + vis_suffix)

    ir = IIO.read(path=path)
    vis = IIO.read(path=vis_path)

    return ir, vis

  def _save_extracted_image(self,
                            fname: str,
                            ir: np.ndarray,
                            vis: np.ndarray,
                            meta: Optional[dict] = None):
    """추출한 열/실화상을 각 폴더에 저장"""
    ir_path = self._dir(DIR.IR).joinpath(fname)
    vis_path = self._dir(DIR.VIS).joinpath(f'{fname}{FN.LL}')

    IIO.save_with_meta(path=ir_path, array=ir, exts=[FN.NPY], meta=meta)

    if self._config['color']['extract_color_image']:
      color_image = apply_colormap(ir, self._cmap)
      IIO.save(path=ir_path.with_name(f'{fname}{FN.COLOR}{FN.LL}'),
               array=color_image)

    IIO.save(path=vis_path, array=vis)

  def _extract_raw_file(self, file: Path):
    """
    Raw 열화상, 실화상 파일 추출.

    Parameters
    ----------
    file : Path
        Raw 파일 경로
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
    """기존에 raw 파일로부터 추출한 실화상/열화상이 없는 경우 추출"""
    try:
      self._files(DIR.IR)
      self._files(DIR.VIS)
    except FileNotFoundError:
      files = self._files(DIR.RAW)

      for file in track(sequence=files,
                        description='Extracting images...',
                        console=utils.console):
        self._extract_raw_file(file=file)

  def _init_registrator(self, shape):
    ropt: DictConfig = self._config['registration']
    copt: DictConfig = self._config['camera']

    trsf = rsitk.Transformation[ropt['transformation']]
    metric = rsitk.Metric[ropt['metric']]
    optimizer = 'gradient_descent' if ropt['preprocess']['edge'] else 'powell'
    registrator = rsitk.SITKRegistrator(transformation=trsf,
                                        metric=metric,
                                        optimizer=optimizer,
                                        bins=ropt['bins'])
    aov = [
        np.deg2rad(copt[x]) if copt[x] else None for x in ['IR_AOV', 'VIS_AOV']
    ]
    registrator.set_initial_scale_factor(scale0=copt['IR_VIS_scale'],
                                         fixed_alpha=aov[0],
                                         moving_alpha=aov[1])
    prep = rsitk.RegistrationPreprocess(
        shape=shape,
        eqhist=ropt['preprocess']['equalize_histogram'],
        unsharp=ropt['preprocess']['unsharp'],
        edge=ropt['preprocess']['edge'])

    return registrator, prep

  def register(self):
    self._extract_raw_files()
    ir_files = self._files(DIR.IR)
    vis_files = self._files(DIR.VIS)
    rgst_dir = self._dir(DIR.RGST)

    registrator, prep = None, None
    for irf, visf in track(sequence=zip(ir_files, vis_files),
                           description='Registering...',
                           total=len(ir_files),
                           console=utils.console):
      ir = IIO.read(irf)
      vis = IIO.read(visf)

      if registrator is None:
        registrator, prep = self._init_registrator(shape=ir.shape)

      logger.debug('Registering `{}`', irf.stem)
      fri, mri = registrator.prep_and_register(fixed_image=ir,
                                               moving_image=vis,
                                               preprocess=prep)
      rgst_color_img = mri.registered_orig_image()

      # 정합한 실화상
      vis_fname = f'{irf.stem}{FN.RGST_VIS}{FN.LL}'
      IIO.save(path=rgst_dir.joinpath(vis_fname),
               array=tools.uint8_image(rgst_color_img))

      # 비교 영상
      compare_fname = f'{irf.stem}{FN.RGST_CMPR}{FN.LL}'
      compare_fig, _ = tools.prep_compare_fig(
          images=(fri.prep_image(), mri.registered_prep_image()),
          titles=('Thermal image (prep)', 'Visible image (prep)'))
      compare_fig.savefig(rgst_dir.joinpath(compare_fname))
      plt.close(compare_fig)

    logger.success('열화상-실화상 정합 완료')

  def _init_stitcher(self) -> stitch.Stitcher:
    sopt: DictConfig = self._config['panorama']['stitch']

    stitcher = stitch.Stitcher(mode=sopt['perspective'],
                               compose_scale=sopt['compose_scale'],
                               work_scale=sopt['work_scale'],
                               warp_threshold=sopt['warp_threshold'])
    stitcher.warper_type = sopt['warp']

    logger.debug('Stitcher 초기화')

    return stitcher

  def _stitch(
      self,
      stitcher: stitch.Stitcher,
      images: List[np.ndarray],
      names: List[str],
      spectrum: str,
  ) -> stitch.Panorama:
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
    logger.debug('Stitch 대상 영상 & 전처리 설정')

    with utils.console.status('Stitching...'):
      res = stitcher.stitch(images=stitching_images,
                            masks=None,
                            names=names,
                            crop=self._config['panorama']['stitch']['crop'])

    return res

  def _save_panorama(self,
                     fname: str,
                     spectrum: str,
                     panorama: stitch.Panorama,
                     save_mask=True,
                     save_meta=True):
    pano_dir = self._dir(DIR.PANO)

    if max(panorama.panorama.shape[:2]) > self._size_limit:
      panorama.panorama = limit_size(panorama.panorama, self._size_limit)
      panorama.mask = limit_size(panorama.mask, self._size_limit)

    meta = None
    if save_meta:
      meta = {
          'panorama': {
              'including': panorama.included(),
              'not_including': panorama.not_included(),
              'graph': panorama.graph_list()
          }
      }

    with utils.console.status('Saving...'):
      if spectrum == 'IR':
        # 적외선 수치, meta 정보 저장
        IIO.save_with_meta(path=pano_dir.joinpath(fname),
                           array=panorama.panorama.astype(np.float16),
                           exts=[FN.NPY],
                           meta=meta)

        # 적외선 colormap 영상 저장
        color_panorama = apply_colormap(panorama.panorama, self._cmap)
        IIO.save(path=pano_dir.joinpath(f'{fname}{FN.COLOR}{FN.LL}'),
                 array=color_panorama)
      else:
        # 실화상 저장
        IIO.save_with_meta(path=pano_dir.joinpath(f'{fname}{FN.LS}'),
                           array=np.round(panorama.panorama).astype(np.uint8),
                           exts=[FN.LS],
                           meta=meta)

      if save_mask:
        # 마스크 저장
        IIO.save(path=pano_dir.joinpath(f'{fname}{FN.PANO_MASK}{FN.LL}'),
                 array=tools.uint8_image(panorama.mask))

  def _stitch_others(
      self,
      stitcher: stitch.Stitcher,
      panorama: stitch.Panorama,
      files: List[Path],
      spectrum: str,
  ):
    if spectrum not in ('IR', 'VIS', 'seg'):
      raise ValueError

    warp = self._config['panorama']['stitch']['warp']
    images = [IIO.read(x) for x in files]
    pano, _, _ = stitcher.warp_and_blend(
        images=stitch.StitchingImages(arrays=images),
        cameras=panorama.cameras,
        masks=None,
        names=[x.name for x in files])

    if panorama.crop_range:
      pano, _, _ = stitcher.crop(pano,
                                 mask=None,
                                 crop_range=panorama.crop_range)

    if spectrum == 'IR':
      pano = pano[:, :, 0]
      pano = pano.astype(np.float16)
    else:
      pano = np.round(pano).astype(np.uint8)

    fname = f'{spectrum}_{warp}'
    if spectrum == 'seg':
      IIO.save(path=self._dir(DIR.PANO).joinpath(fname + FN.LL),
               array=limit_size(pano, self._size_limit))
    else:
      panorama.panorama = pano
      self._save_panorama(fname=fname,
                          spectrum=spectrum,
                          panorama=panorama,
                          save_mask=False,
                          save_meta=False)

  def panorama(self):
    spectrum = self._config['panorama']['target'].upper()
    sopt = self._config['panorama']['stitch']
    stitcher = self._init_stitcher()

    # Raw 파일 추출
    self._extract_raw_files()

    # 지정한 spectrum 파노라마
    files = self._files(self._SP_DIR[spectrum])
    images = [IIO.read(x) for x in files]

    # 파노라마 생성
    stitcher.set_blend_type(sopt['blend'][spectrum])
    pano = self._stitch(stitcher=stitcher,
                        images=images,
                        names=[x.stem for x in files],
                        spectrum=spectrum)

    # 저장
    self._save_panorama(fname='{}_{}'.format(spectrum, sopt['warp']),
                        spectrum=spectrum,
                        panorama=pano)

    # segmention mask 저장
    try:
      seg_files = self._files(DIR.SEG)
    except FileNotFoundError as e:
      logger.warning('{} 부위 인식 파노라마를 생성하지 않습니다.', e)
    else:
      logger.debug('부위 인식 파노라마 생성')
      stitcher.set_blend_type(False)
      self._stitch_others(stitcher=stitcher,
                          panorama=pano,
                          files=seg_files,
                          spectrum='seg')

    # 나머지 영상의 파노라마 생성/저장
    sp2 = 'VIS' if spectrum == 'IR' else 'IR'
    try:
      files2 = self._files(self._SP_DIR[sp2])
    except FileNotFoundError as e:
      logger.warning('{} {} 파노라마를 생성하지 않습니다.', e, self._SP_KOR[sp2])
    else:
      logger.debug('{} 파노라마 생성', self._SP_KOR[sp2])
      files2 = [files2[x] for x in pano.indices]  # TODO 파일 이름과 대조
      stitcher.set_blend_type(sopt['blend'][sp2])

      self._stitch_others(stitcher=stitcher,
                          panorama=pano,
                          files=files2,
                          spectrum=sp2)

    logger.success('파노라마 생성 완료')

  def _init_perspective_correction(self):
    options = self._config['distort_correction']

    canny_options = persp.CannyOptions(**options['canny'])
    hough_options = persp.HoughOptions(**options['hough'])
    correction_opts = persp.CorrectionOptions(**options['correction'])

    pc = persp.PerspectiveCorrection(canny_options=canny_options,
                                     hough_options=hough_options,
                                     correction_options=correction_opts)
    return pc

  def correct(self):
    pc = self._init_perspective_correction()

    warp = self._config['panorama']['stitch']['warp']
    ir_fname = f'IR_{warp}'  # FIXME {warp} 제거
    pano_dir = self._dir(DIR.PANO, mkdir=False)

    ir_path = pano_dir.joinpath(ir_fname + FN.NPY)
    if not ir_path.exists():
      logger.error('생성된 파노라마 파일이 없습니다.')
      return

    logger.debug('Init perspective correction')

    # 적외선 파노라마
    ir_pano = IIO.read(ir_path).astype(np.float32)
    mask = None
    if self._config['distort_correction']['apply_mask']:
      mask = IIO.read(ir_path.with_name(
          f'{ir_path.stem}{FN.PANO_MASK}{FN.LL}')).astype(bool)

    # 왜곡 보정
    try:
      corrected = pc.perspective_correct(image=ir_pano, mask=mask)
    except persp.NotEnoughEdgelets:
      logger.critical('시점 왜곡을 추정할 edge의 개수가 부족합니다. '
                      'Edge 추출 옵션을 변경하거나 높은 해상도의 파노라마를 사용하세요.')
      return

    logger.debug('IR 파노라마 왜곡 보정 완료 (shape: {})', corrected.output_shape)

    # plot 저장
    cor_dir = self._dir(DIR.COR)
    fig, _ = corrected.process_plot()
    fig.savefig(cor_dir.joinpath(f'plot{FN.LS}'))
    plt.close(fig)

    if corrected.success():
      # 적외선 파노라마 저장
      corrected_ir = limit_size(corrected.corrected_image, self._size_limit)
      IIO.save_with_meta(path=cor_dir.joinpath(ir_fname),
                         array=corrected_ir.astype(np.float16),
                         exts=[FN.NPY])
      # colormap 적용 버전 저장
      IIO.save(path=cor_dir.joinpath(f'{ir_fname}{FN.COLOR}{FN.LL}'),
               array=apply_colormap(corrected_ir, self._cmap))
      logger.debug('IR 파노라마 보정 파일 저장')

      # 실화상 파노라마 보정
      vis_path = pano_dir.joinpath(f'VIS_{warp}{FN.LS}')
      if not vis_path.exists():
        logger.warning('실화상 파노라마가 존재하지 않습니다.')
      else:
        vis_pano = IIO.read(path=vis_path)
        vis_corrected = limit_size(
            corrected.correct(vis_pano).astype(np.uint8), self._size_limit)
        IIO.save(path=cor_dir.joinpath(vis_path.name), array=vis_corrected)
        logger.debug('실화상 파노라마 왜곡 보정 저장')

      # segmentation 파노라마 보정
      seg_path = pano_dir.joinpath(f'seg_{warp}{FN.LL}')
      if not seg_path.exists():
        logger.warning('부위 인식 파노라마가 존재하지 않습니다.')
      else:
        seg_pano = IIO.read(path=seg_path)
        seg_corrected = limit_size(
            corrected.correct(seg_pano).astype(np.uint8), self._size_limit)
        IIO.save(path=cor_dir.joinpath(seg_path.name), array=seg_corrected)
        logger.debug('부위 인식 파노라마 왜곡 보정 저장')

    logger.success('파노라마 왜곡 보정 완료')
