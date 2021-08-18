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
from misc.imageio import ImageIO as IIO

from .cmap import apply_colormap, get_thermal_colormap
from .config import DictConfig, read_config
from .pano_files import DIR, FN, SP, ThermalPanoramaFileManager


class ThermalPanorama:
  _SP_DIR = {
      SP.IR.value: DIR.IR,
      SP.VIS.value: DIR.RGST,
      SP.SEG.value: DIR.SEG,
  }
  _SP_KOR = {
      SP.IR.value: '열화상',
      SP.VIS.value: '실화상',
      SP.SEG.value: '부위 인식',
  }

  def __init__(self, directory: Union[str, Path], default_config=False) -> None:
    # working directory
    wd = Path(directory).resolve()
    if not wd.exists():
      raise FileNotFoundError(wd)

    self._wd = wd
    self._config = read_config(wd=wd, default=default_config)
    self._fm = ThermalPanoramaFileManager(directory=wd)

    # 제조사, Raw 파일 패턴
    self._manufacturer = self._check_manufacturer()
    self._fm.set_raw_pattern(self._config['file'][self._manufacturer]['IR'])
    if self._manufacturer != 'FLIR':
      self._flir_ext = None
    else:
      self._flir_ext = flir.FlirExtractor()
    logger.info('Manufacturer: {}', self._manufacturer)

    # 카메라 기종
    self._camera = self._check_camera_model()
    logger.info('Camera: {}', self._camera)

    # 컬러맵
    self._cmap = get_thermal_colormap(
        name=self._config['color'].get('colormap', 'iron'))

  def _check_manufacturer(self) -> str:
    fopt: DictConfig = self._config['file']
    flir_files = self._fm.glob(d=DIR.RAW, pattern=fopt['FLIR']['IR'])
    testo_ir_files = self._fm.glob(d=DIR.RAW, pattern=fopt['testo']['IR'])
    testo_vis_files = self._fm.glob(d=DIR.RAW, pattern=fopt['testo']['VIS'])

    if testo_ir_files and testo_vis_files:
      manufacturer = 'testo'
    elif flir_files:
      manufacturer = 'FLIR'
    else:
      raise ValueError('지원하지 않는 Raw 파일 형식입니다.')

    return manufacturer

  def _check_camera_model(self) -> Optional[str]:
    tags = ['Model', 'CameraModel']
    raw_files = self._fm.raw_files()

    def _get_model(exif: dict):
      # iterable 중 조건을 만족하는 첫 element
      tag = next((x for x in exif.keys() if x in tags), None)
      if tag is None:
        return None

      return exif[tag]

    exifs = exif.get_exif(files=[x.as_posix() for x in raw_files], tags=tags)
    models = [_get_model(x) for x in exifs]
    models = [x for x in models if x is not None]

    if not models:
      logger.debug('Exif로부터 카메라 기종 추정 불가')
      return None

    if any(models[0] != x for x in models[1:]):
      logger.warning('다수의 카메라로 촬영된 Raw 파일을 입력했습니다 ({}). '
                     '카메라 기종을 추정할 수 없습니다.', set(models))
      return None

    return models[0]

  def _extract_flir_image(self, path: Path):
    assert self._flir_ext is not None
    ir, vis = self._flir_ext.extract_data(path)
    meta = {'Exif': exif.get_exif(files=path.as_posix())[0]}

    if self._config['file']['force_horizontal'] and (ir.shape[0] > ir.shape[1]):
      # 수직 영상을 수평으로 만들기 위해 90도 회전
      logger.debug('rot90 "{}"', path.name)
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
    ir_path = self._fm.change_dir(DIR.IR, file=fname)
    vis_path = self._fm.change_dir(DIR.VIS, file=fname)

    IIO.save_with_meta(path=ir_path, array=ir, exts=[FN.NPY], meta=meta)

    if self._config['color']['extract_color_image']:
      color_image = apply_colormap(ir, self._cmap)
      IIO.save(path=self._fm.color_path(ir_path), array=color_image)

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

    ir_path = self._fm.change_dir(DIR.IR, file)
    vis_path = self._fm.change_dir(DIR.VIS, file)
    if ir_path.exists() and vis_path.exists():
      return

    logger.debug('Extracting "{}"', file.name)

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
      self._fm.files(DIR.IR)
      self._fm.files(DIR.VIS)
    except FileNotFoundError:
      self._fm.subdir(DIR.IR, mkdir=True)
      self._fm.subdir(DIR.VIS, mkdir=True)

      files = self._fm.raw_files()
      for file in track(sequence=files,
                        description='Extracting images...',
                        console=utils.console):
        self._extract_raw_file(file=file)

  @property
  def _size_limit(self):
    return self._config['file']['size_limit']

  def limit_size(self, image: np.ndarray) -> np.ndarray:
    return tools.limit_image_size(image=image, limit=self._size_limit)

  def _init_registrator(self, shape):
    ropt: DictConfig = self._config['registration']

    if self._camera in self._config['camera']:
      camera = self._camera
    else:
      camera = 'default'

    copt: DictConfig = self._config['camera'][camera]
    logger.debug('Regestering preset: {}', camera)

    trsf = rsitk.Transformation[ropt['transformation']]
    metric = rsitk.Metric[ropt['metric']]
    optimizer = ropt['optimizer']

    registrator = rsitk.SITKRegistrator(transformation=trsf,
                                        metric=metric,
                                        optimizer=optimizer,
                                        bins=ropt['bins'])
    aov = [
        np.deg2rad(copt[x]) if copt[x] else None for x in ['IR_AOV', 'VIS_AOV']
    ]
    registrator.set_initial_params(scale=copt['scale'],
                                   fixed_alpha=aov[0],
                                   moving_alpha=aov[1],
                                   translation=copt['translation'])
    prep = rsitk.RegistrationPreprocess(
        shape=shape,
        eqhist=ropt['preprocess']['equalize_histogram'],
        unsharp=ropt['preprocess']['unsharp'],
        edge=ropt['preprocess']['edge'])

    return registrator, prep

  def register(self):
    self._extract_raw_files()
    self._fm.subdir(DIR.RGST, mkdir=True)

    files = self._fm.raw_files()
    registrator, prep, mtx = None, None, {}
    for file in track(sequence=files,
                      description='Registering...',
                      console=utils.console):
      ir = IIO.read(self._fm.change_dir(DIR.IR, file))
      vis = IIO.read(self._fm.change_dir(DIR.VIS, file))

      if registrator is None:
        registrator, prep = self._init_registrator(shape=ir.shape)

      logger.debug('Registering "{}"', file.stem)
      fri, mri = registrator.prep_and_register(fixed_image=ir,
                                               moving_image=vis,
                                               preprocess=prep)
      rgst_color_img = mri.registered_orig_image()
      mtx[file.stem] = mri.matrix

      # 정합한 실화상
      path = self._fm.change_dir(DIR.RGST, file)
      IIO.save(path=path, array=tools.uint8_image(rgst_color_img))

      # 비교 영상
      compare_path = path.with_name(f'{path.stem}{FN.RGST_CMPR}{path.suffix}')
      compare_fig, _ = tools.prep_compare_fig(
          images=(fri.prep_image(), mri.registered_prep_image()),
          titles=('Thermal image (prep)', 'Visible image (prep)'))
      compare_fig.savefig(compare_path.joinpath())
      plt.close(compare_fig)

    np.savez(self._fm.rgst_matrix_path(), **mtx)

    logger.success('열화상-실화상 정합 완료')

  def segment(self):
    # pylint: disable=import-outside-toplevel
    from segmentation import deeplab

    try:
      files = self._fm.files(DIR.RGST)
    except FileNotFoundError as e:
      logger.exception(e)
      logger.error('"{}" 파일을 찾을 수 없습니다. '
                   '열화상-실화상 정합을 먼저 시행해주세요.',
                   Path(e.args[0]).relative_to(self._wd))
      return

    # 모델 로드
    try:
      model_path = self._fm.segment_model_path()
    except FileNotFoundError as e:
      logger.exception(e)
      logger.error('부위 인식 모델 파일을 불러올 수 없습니다.')

    deeplab.tf_gpu_memory_config()
    model = deeplab.DeepLabModel(model_path.as_posix())

    self._fm.subdir(DIR.SEG, mkdir=True)
    for file in track(sequence=files,
                      description='Segmenting...',
                      console=utils.console):
      image = IIO.read(file)
      seg_map, _, fig = model.predict_and_visualize(image)

      path = self._fm.change_dir(DIR.SEG, file)
      IIO.save(path=path, array=tools.SegMask.index_to_vis(seg_map))

      fig_path = path.with_name(f'{path.stem}{FN.SEG_FIG}{FN.LS}')
      fig.savefig(fig_path)
      plt.close(fig)

    logger.success('외피 부위 인식 완료')

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
    logger.trace('Stitch 대상 영상 & 전처리 설정')

    with utils.console.status('Stitching...'):
      res = stitcher.stitch(images=stitching_images,
                            masks=None,
                            names=names,
                            crop=self._config['panorama']['stitch']['crop'])

    return res

  def _save_panorama(self,
                     spectrum: SP,
                     panorama: stitch.Panorama,
                     save_mask=True,
                     save_meta=True):
    self._fm.subdir(DIR.PANO, mkdir=True)

    if max(panorama.panorama.shape[:2]) > self._size_limit:
      panorama.panorama = self.limit_size(panorama.panorama)
      panorama.mask = self.limit_size(panorama.mask)

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
      path = self._fm.panorama_path(DIR.PANO, spectrum)
      if spectrum is SP.IR:
        # 적외선 수치, meta 정보 저장
        IIO.save_with_meta(path=path,
                           array=panorama.panorama.astype(np.float16),
                           exts=[FN.NPY],
                           meta=meta)

        # 적외선 colormap 영상 저장
        color_panorama = apply_colormap(panorama.panorama, self._cmap)
        IIO.save(path=self._fm.color_path(path), array=color_panorama)
      else:
        # 실화상 저장
        IIO.save_with_meta(path=path,
                           array=np.round(panorama.panorama).astype(np.uint8),
                           exts=[FN.LS],
                           meta=meta)

      if save_mask:
        # 마스크 저장
        IIO.save(path=self._fm.panorama_path(DIR.PANO, SP.MASK),
                 array=tools.uint8_image(panorama.mask))

  def _stitch_others(self, stitcher: stitch.Stitcher, panorama: stitch.Panorama,
                     spectrum: SP):
    try:
      files = self._fm.files(self._SP_DIR[spectrum.value])
    except FileNotFoundError as e:
      logger.exception(e)
      logger.error('"{}" 파일을 찾을 수 없습니다. {} 파노라마를 생성하지 않습니다.',
                   Path(e.args[0]).relative_to(self._wd),
                   self._SP_KOR[spectrum.value])
      return

    files = [files[x] for x in panorama.indices]
    images = [IIO.read(x) for x in files]

    pano, _, _ = stitcher.warp_and_blend(
        images=stitch.StitchingImages(arrays=images),
        cameras=panorama.cameras,
        masks=None,
        names=[x.name for x in files])

    if panorama.crop_range:
      pano = panorama.crop_range.crop(pano)

    if spectrum is SP.IR:
      pano = pano[:, :, 0]
      pano = pano.astype(np.float16)
    else:
      pano = np.round(pano).astype(np.uint8)

    if spectrum is SP.SEG:
      IIO.save(path=self._fm.panorama_path(DIR.PANO, spectrum),
               array=self.limit_size(pano))
    else:
      panorama.panorama = pano
      self._save_panorama(spectrum=spectrum,
                          panorama=panorama,
                          save_mask=False,
                          save_meta=False)

  def panorama(self):
    spectrum = self._config['panorama']['target'].upper()
    if spectrum not in ('IR', 'VIS'):
      raise ValueError(spectrum)

    sopt = self._config['panorama']['stitch']
    stitcher = self._init_stitcher()

    # Raw 파일 추출
    self._extract_raw_files()

    # 지정한 spectrum 파노라마
    files = self._fm.files(self._SP_DIR[spectrum])
    images = [IIO.read(x) for x in files]

    # 파노라마 생성
    stitcher.set_blend_type(sopt['blend'][spectrum])
    pano = self._stitch(stitcher=stitcher,
                        images=images,
                        names=[x.stem for x in files],
                        spectrum=spectrum)

    # 저장
    self._save_panorama(spectrum=SP[spectrum], panorama=pano)

    # segmention mask 저장
    stitcher.set_blend_type(False)
    self._stitch_others(stitcher=stitcher, panorama=pano, spectrum=SP.SEG)

    # 나머지 영상의 파노라마 생성/저장
    sp2 = 'VIS' if spectrum == 'IR' else 'IR'
    stitcher.set_blend_type(sopt['blend'][sp2])
    self._stitch_others(stitcher=stitcher, panorama=pano, spectrum=SP[sp2])

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

  def _correct_others(self, correction: persp.Correction, spectrum: SP):
    try:
      path = self._fm.panorama_path(DIR.PANO, spectrum, error=True)
    except FileNotFoundError as e:
      logger.exception(e)
      logger.error('{} 파노라마가 존재하지 않습니다.', self._SP_KOR[spectrum.value])
      return

    pano = IIO.read(path=path)
    pano_corrected = correction.correct(pano)[0].astype(np.uint8)
    pano_limited = self.limit_size(pano_corrected)

    IIO.save(path=self._fm.panorama_path(DIR.COR, spectrum), array=pano_limited)
    logger.debug('{} 파노라마 왜곡 보정 저장', self._SP_KOR[spectrum.value])

  def correct(self):
    pc = self._init_perspective_correction()

    try:
      ir_path = self._fm.panorama_path(DIR.PANO, SP.IR, error=True)
    except FileNotFoundError as e:
      logger.exception(e)
      logger.error('생성된 파노라마 파일이 없습니다.')

    logger.trace('Init perspective correction')

    # 적외선 파노라마
    pano = IIO.read(ir_path).astype(np.float32)
    if self._config['distort_correction']['apply_mask']:
      mask = IIO.read(self._fm.panorama_path(DIR.PANO, SP.MASK)).astype(bool)
    else:
      mask = None

    # 왜곡 보정
    try:
      crct = pc.perspective_correct(image=pano, mask=mask)
    except persp.NotEnoughEdgelets:
      logger.critical('시점 왜곡을 추정할 edge의 개수가 부족합니다. '
                      'Edge 추출 옵션을 변경하거나 높은 해상도의 파노라마를 사용하세요.')
      return

    # plot 저장
    self._fm.subdir(DIR.COR, mkdir=True)
    fig, _ = crct.process_plot(image=pano)
    fig.savefig(self._fm.correction_plot_path(), dpi=300)
    plt.close(fig)

    if not crct.success():
      logger.error('IR 파노라마 왜곡 보정 중 오류 발생. 저장된 plot을 참고해주세요.')
      return

    logger.debug('IR 파노라마 왜곡 보정 완료')

    # 적외선 파노라마 저장
    cpano, cmask = crct.correct(pano, mask)
    cpano = self.limit_size(cpano)
    path = self._fm.panorama_path(DIR.COR, SP.IR)
    IIO.save_with_meta(path=path, array=cpano.astype(np.float16), exts=[FN.NPY])

    # colormap 적용 버전 저장
    IIO.save(path=self._fm.color_path(path),
             array=apply_colormap(cpano, self._cmap))
    logger.debug('IR 파노라마 보정 파일 저장')

    # mask 저장
    if cmask is not None:
      cmask = self.limit_size(cmask)
      IIO.save(path=self._fm.panorama_path(DIR.COR, SP.MASK),
               array=tools.uint8_image(cmask))

    # 실화상, 부위인식 파노라마 보정
    self._correct_others(correction=crct, spectrum=SP.VIS)
    self._correct_others(correction=crct, spectrum=SP.SEG)

    logger.success('파노라마 왜곡 보정 완료')

  def run(self):
    logger.info('Start registering')
    self.register()
    logger.log('BLANK', '')

    logger.info('Start segmenting')
    self.segment()
    logger.log('BLANK', '')

    logger.info('Start panorama stitching')
    self.panorama()
    logger.log('BLANK', '')

    logger.info('Start distortion correction')
    self.correct()
