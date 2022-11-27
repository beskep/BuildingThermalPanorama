"""외피 열화상 파노라마 영상처리 알고리즘의 CLI 인터페이스"""

from pathlib import Path
from typing import List, Optional, Union

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

from pano import stitch
from pano import utils
from pano.distortion import perspective as persp
from pano.flir import FlirExtractor
from pano.misc import exif
from pano.misc import tools
from pano.misc.imageio import ImageIO as IIO
from pano.misc.imageio import save_webp_images
import pano.registration.registrator.simpleitk as rsitk
from pano.segmentation.onnx import SmpModel

from .common.cmap import apply_colormap
from .common.cmap import get_thermal_colormap
from .common.config import DictConfig
from .common.config import update_config
from .common.pano_files import DIR
from .common.pano_files import FN
from .common.pano_files import init_directory
from .common.pano_files import SP
from .common.pano_files import ThermalPanoramaFileManager


class ThermalPanorama:

  def __init__(self,
               directory: Union[str, Path],
               default_config=False,
               init_loglevel='INFO') -> None:
    # working directory
    wd = Path(directory).resolve()
    if not wd.exists():
      raise FileNotFoundError(wd)

    self._wd = wd
    self._config = init_directory(directory=wd, default=default_config)
    self._fm = ThermalPanoramaFileManager(directory=wd)

    # 제조사, Raw 파일 패턴
    self._manufacturer = self._check_manufacturer()
    self._fm.raw_pattern = self._config['file'][self._manufacturer]['IR']
    logger.log(init_loglevel, 'Manufacturer: {}', self._manufacturer)

    # 카메라 기종
    self._camera = self._check_camera_model()
    logger.log(init_loglevel, 'Camera: {}', self._camera)

    # 컬러맵
    self._cmap = get_thermal_colormap(
        name=self._config['color'].get('colormap', 'iron'))

  @property
  def fm(self):
    return self._fm

  @property
  def cmap(self):
    return self._cmap

  def update_config(self, config: Union[utils.StrPath, dict, DictConfig]):
    if isinstance(config, (str, Path)):
      config = OmegaConf.load(config)

    self._config = update_config(self._wd, config)

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
    data = FlirExtractor(path.as_posix()).extract()
    meta = {'Exif': dict(data.exif), 'signal_reflected': data.signal_reflected}

    force_horizontal = self._config['file']['force_horizontal']
    if force_horizontal and (data.ir.shape[0] > data.ir.shape[1]):
      # 수직 영상을 수평으로 만들기 위해 90도 회전
      try:
        k = {'CW': 3, 'CCW': 1}[force_horizontal]
      except IndexError as e:
        raise ValueError('force_horizontal must be one of {"CW", "CCW"}') from e

      logger.debug('Rotate "{}" (k={})', path.name, k)
      data.ir = np.rot90(data.ir, k=k, axes=(0, 1))
      data.vis = np.rot90(data.vis, k=k, axes=(0, 1))

    # FLIR One로 찍은 사진은 수직으로 촬영해도 orientation 번호가
    # `1` (`Horizontal (normal)`)로 표시되어서 아래 정보가 쓸모가 없음...

    # tag = exif.get_exif_tags(path.as_posix(), '-Orientation', '-n')
    # orientation = tag['Orientation']

    return data.ir, data.vis, meta

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

  def extract(self):
    try:
      self._fm.files(DIR.IR)
      self._fm.subdir(DIR.VIS).stat()
    except FileNotFoundError:
      self._fm.subdir(DIR.IR, mkdir=True)
      self._fm.subdir(DIR.VIS, mkdir=True)

      files = self._fm.raw_files()
      for file in utils.track(files, description='Extracting images...'):
        self._extract_raw_file(file=file)

  def extract_generator(self):
    try:
      self._fm.files(DIR.IR)
      self._fm.subdir(DIR.VIS).stat()
    except FileNotFoundError:
      self._fm.subdir(DIR.IR, mkdir=True)
      self._fm.subdir(DIR.VIS, mkdir=True)

      files = self._fm.raw_files()
      for r, file in utils.ptrack(files, description='Extracting images...'):
        self._extract_raw_file(file=file)
        yield r

  @property
  def _size_limit(self):
    return self._config['file']['size_limit']

  def limit_size(self, image: np.ndarray, aa=True) -> np.ndarray:
    if aa:
      order = tools.Interpolation.BiCubic
    else:
      order = tools.Interpolation.NearestNeighbor

    return tools.limit_image_size(image=image,
                                  limit=self._size_limit,
                                  order=order,
                                  anti_aliasing=aa)

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

  def _register(self, file, registrator: rsitk.SITKRegistrator,
                prep: rsitk.RegistrationPreprocess):
    ir = IIO.read(self._fm.change_dir(DIR.IR, file))
    vis = IIO.read(self._fm.change_dir(DIR.VIS, file))

    if registrator is None:
      registrator, prep = self._init_registrator(shape=ir.shape)

    logger.debug('Registering "{}"', file.stem)
    fri, mri = registrator.prep_and_register(fixed_image=ir,
                                             moving_image=vis,
                                             preprocess=prep)
    rgst_color_img = mri.registered_orig_image()

    # 정합한 실화상
    path = self._fm.change_dir(DIR.RGST, file)
    IIO.save(path=path, array=tools.uint8_image(rgst_color_img))

    # 비교 영상
    compare_path = path.with_name(f'{path.stem}{FN.RGST_AUTO}.jpg')
    compare_fig, _ = tools.prep_compare_fig(
        images=(fri.prep_image(), mri.registered_prep_image()),
        titles=('열화상', '실화상', '비교 (Checkerboard)', '비교 (Difference)'))
    compare_fig.savefig(compare_path, dpi=200)
    plt.close(compare_fig)

    return mri.matrix

  def register(self):
    self.extract()
    self._fm.subdir(DIR.RGST, mkdir=True)

    files = self._fm.raw_files()
    registrator, prep, matrices = None, None, {}
    for file in utils.track(sequence=files, description='Registering...'):
      if registrator is None:
        ir = IIO.read(self._fm.change_dir(DIR.IR, file))
        registrator, prep = self._init_registrator(shape=ir.shape)

      matrix = self._register(file=file, registrator=registrator, prep=prep)
      matrices[file.stem] = matrix

    np.savez(self._fm.rgst_matrix_path(), **matrices)
    logger.success('열화상-실화상 정합 완료')

  def register_generator(self):
    self._fm.subdir(DIR.RGST, mkdir=True)

    files = self._fm.raw_files()
    registrator, prep, matrices = None, None, {}
    for r, file in utils.ptrack(sequence=files, description='Registering...'):
      if registrator is None:
        ir = IIO.read(self._fm.change_dir(DIR.IR, file))
        registrator, prep = self._init_registrator(shape=ir.shape)

      try:
        matrix = self._register(file=file, registrator=registrator, prep=prep)
      except FileNotFoundError as e:
        msg = f'"{e}"를 찾을 수 없습니다. 열화상을 먼저 추출하세요.'
        raise FileNotFoundError(msg) from e

      matrices[file.stem] = matrix

      yield r

    np.savez(self._fm.rgst_matrix_path(), **matrices)
    logger.success('열화상-실화상 정합 완료')
    yield 1.0

  def _init_segment_model(self):
    if self._config['panorama']['separate']:
      files = self._fm.files(DIR.VIS, error=False)
    else:
      try:
        files = self._fm.files(DIR.RGST)
      except FileNotFoundError as e:
        raise FileNotFoundError('대상 경로를 찾을 수 없습니다. '
                                '열화상-실화상 정합을 먼저 시행해주세요.') from e

    try:
      path = self._fm.segment_model_path()
    except FileNotFoundError as e:
      raise FileNotFoundError('부위 인식 모델 파일을 불러올 수 없습니다.') from e

    model = SmpModel(str(path))

    return files, model

  def _segment(self, model: SmpModel, file):
    mask = model.predict(file)
    fig, _ = model.visualization(src=file, mask=mask)

    path = self._fm.change_dir(DIR.SEG, file)
    IIO.save(path=path, array=tools.SegMask.index_to_vis(mask))

    fig_path = path.with_name(f'{path.stem}{FN.SEG_FIG}{FN.LL}')
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

  def segment(self):
    files, model = self._init_segment_model()

    self._fm.subdir(DIR.SEG, mkdir=True)
    for file in utils.track(files, description='Segmenting...'):
      self._segment(model, file)

    logger.success('외피 부위 인식 완료')

  def segment_generator(self):
    files, model = self._init_segment_model()

    self._fm.subdir(DIR.SEG, mkdir=True)
    for r, file in utils.ptrack(files, description='Segmenting...'):
      self._segment(model, file)
      yield r

    logger.success('외피 부위 인식 완료')
    yield 1.0

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
      panorama.mask = self.limit_size(panorama.mask, aa=False)

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
                     sp: SP):
    try:
      files = self._fm.files(DIR[sp.name], error=(sp is SP.IR))
    except FileNotFoundError as e:
      raise FileNotFoundError('대상 파일을 찾을 수 없습니다. '
                              f'{sp.value} 파노라마를 생성할 수 없습니다.') from e

    files = [files[x] for x in panorama.indices]
    images = [IIO.read(x) for x in files]
    cameras = [panorama.cameras[x] for x in panorama.indices]

    pano, _, _ = stitcher.warp_and_blend(
        images=stitch.StitchingImages(arrays=images),
        cameras=cameras,
        masks=None,
        names=[x.name for x in files])

    if panorama.crop_range:
      pano = panorama.crop_range.crop(pano, strict=False)

    if sp is SP.IR:
      pano = pano[:, :, 0]
      pano = pano.astype(np.float16)
    elif sp is SP.SEG:
      pano = tools.SegMask.index_to_vis(np.round(pano / tools.SegMask.scale))
    else:
      pano = np.round(pano).astype(np.uint8)

    if sp is SP.SEG:
      IIO.save(path=self._fm.panorama_path(DIR.PANO, sp),
               array=self.limit_size(pano, aa=False))
    else:
      panorama.panorama = pano
      self._save_panorama(spectrum=sp,
                          panorama=panorama,
                          save_mask=False,
                          save_meta=False)

  def _panorama_join(self):
    cfg = self._config['panorama']['blend']
    stitcher = self._init_stitcher()

    spectrum = self._config['panorama']['target'].upper()
    if spectrum not in ('IR', 'VIS'):
      raise ValueError(spectrum)

    # 지정한 spectrum 파노라마
    files = self._fm.files(DIR[spectrum])
    images = [IIO.read(x) for x in files]

    # 파노라마 생성
    stitcher.blend_type = cfg['type'][spectrum]
    stitcher.blend_strength = cfg['strength'][spectrum]
    pano = self._stitch(stitcher=stitcher,
                        images=images,
                        names=[x.stem for x in files],
                        spectrum=spectrum)

    # 저장
    self._save_panorama(spectrum=SP[spectrum], panorama=pano)

    # 나머지 영상의 파노라마 생성/저장
    sp2 = 'VIS' if spectrum == 'IR' else 'IR'
    stitcher.blend_type = cfg['type'][sp2]
    stitcher.blend_strength = cfg['strength'][sp2]
    self._stitch_others(stitcher=stitcher, panorama=pano, sp=SP[sp2])

    # segmention mask 저장
    stitcher.blend_type = False
    stitcher.interp = stitch.Interpolation.NEAREST
    self._stitch_others(stitcher=stitcher, panorama=pano, sp=SP.SEG)

  def _panorama_separate(self):
    cfg = self._config['panorama']['blend']
    stitcher = self._init_stitcher()

    # IR 파노라마
    stitcher.blend_type = cfg['type']['IR']
    stitcher.blend_strength = cfg['strength']['IR']
    ir_files = self._fm.files(DIR.IR)
    ir_images = [IIO.read(x) for x in ir_files]

    try:
      ir_pano = self._stitch(stitcher=stitcher,
                             images=ir_images,
                             names=[x.stem for x in ir_files],
                             spectrum='IR')
    except (ValueError, RuntimeError) as e:
      logger.exception(e)
    else:
      self._save_panorama(spectrum=SP.IR, panorama=ir_pano)

    # VIS 파노라마
    stitcher.blend_type = cfg['type']['VIS']
    stitcher.blend_strength = cfg['strength']['VIS']
    vis_files = self._fm.files(DIR.VIS, error=False)
    vis_images = [IIO.read(x) for x in vis_files]

    try:
      vis_pano = self._stitch(stitcher=stitcher,
                              images=vis_images,
                              names=[x.stem for x in vis_files],
                              spectrum='VIS')
    except (ValueError, RuntimeError) as e:
      logger.exception(e)
    else:
      self._save_panorama(spectrum=SP.VIS, panorama=vis_pano, save_mask=False)

      # segmentation mask
      stitcher.blend_type = False
      stitcher.interp = stitch.Interpolation.NEAREST
      self._stitch_others(stitcher=stitcher, panorama=vis_pano, sp=SP.SEG)

  def panorama(self):
    # Raw 파일 추출
    self.extract()

    if self._config['panorama']['separate']:
      self._panorama_separate()
    else:
      self._panorama_join()

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

  def _correct_others(self,
                      correction: persp.Correction,
                      spectrum: SP,
                      crop_range: Optional[tools.CropRange] = None):
    try:
      path = self._fm.panorama_path(DIR.PANO, spectrum, error=True)
    except FileNotFoundError as e:
      raise FileNotFoundError(f'{spectrum.value} 파노라마가 존재하지 않습니다.') from e

    pano = IIO.read(path=path)

    if spectrum is SP.SEG:
      pano = tools.SegMask.vis_to_index(pano)
      order = tools.Interpolation.NearestNeighbor
    else:
      order = tools.Interpolation.BiCubic

    if crop_range is not None and crop_range.cropped:
      pano = crop_range.crop(pano)

    pano_corrected = correction.correct(pano, order=order)[0].astype(np.uint8)
    pano_limited = self.limit_size(pano_corrected, aa=(spectrum is not SP.SEG))

    if spectrum is SP.SEG:
      pano_limited = tools.SegMask.index_to_vis(pano_limited)

    IIO.save(path=self._fm.panorama_path(DIR.COR, spectrum), array=pano_limited)
    logger.debug('{} 파노라마 왜곡 보정 저장', spectrum.value)

  def correct(self):
    try:
      ir_path = self._fm.panorama_path(DIR.PANO, SP.IR, error=True)
    except FileNotFoundError as e:
      raise FileNotFoundError('생성된 파노라마 파일이 없습니다.') from e

    pc = self._init_perspective_correction()
    logger.trace('Init perspective correction')

    # 적외선 파노라마
    pano = IIO.read(ir_path).astype(np.float32)
    if self._config['distort_correction']['apply_mask']:
      mask = IIO.read(self._fm.panorama_path(DIR.PANO, SP.MASK)).astype(bool)
    else:
      mask = None

    seg = IIO.read(self._fm.panorama_path(DIR.PANO, SP.SEG))
    if (pano.shape[:2] != seg.shape[:2]):
      raise ValueError('열·실화상 파노라마의 크기가 다릅니다. 파노라마 정합을 먼저 시행해주세요.')

    # wall, window 영역만 추출
    seg = tools.SegMask.vis_to_index(seg[:, :, 0])
    seg_mask = np.logical_and(mask, np.logical_or(seg == 1, seg == 2))
    crop_range, _ = tools.crop_mask(seg_mask)
    if crop_range.cropped:
      pano = crop_range.crop(pano)
      mask = crop_range.crop(mask)

    # 왜곡 보정
    try:
      crct = pc.perspective_correct(image=pano, mask=mask)
    except persp.NotEnoughEdgelets as e:
      raise persp.NotEnoughEdgelets(
          '시점 왜곡을 추정할 edge의 개수가 부족합니다. '
          'Edge 추출 옵션을 변경하거나 높은 해상도의 파노라마를 사용하세요.') from e

    # plot 저장
    self._fm.subdir(DIR.COR, mkdir=True)
    fig, _ = crct.process_plot(image=pano)
    fig.savefig(self._fm.correction_plot_path(), dpi=300)
    plt.close(fig)

    if not crct.success():
      raise ValueError('IR 파노라마 왜곡 보정 중 오류 발생. 저장된 plot을 참고해주세요.')

    logger.debug('IR 파노라마 왜곡 보정 완료')

    # 적외선 파노라마 저장
    cpano, cmask = crct.correct(pano, mask)
    cpano = self.limit_size(cpano)
    path = self._fm.panorama_path(DIR.COR, SP.IR)
    IIO.save_with_meta(path=path,
                       array=cpano.astype(np.float16),
                       exts=[FN.NPY, FN.LL],
                       dtype='uint16')

    # colormap 적용 버전 저장
    IIO.save(path=self._fm.color_path(path),
             array=apply_colormap(cpano, self._cmap))
    logger.debug('IR 파노라마 보정 파일 저장')

    # mask 저장
    if cmask is not None:
      cmask = self.limit_size(cmask, aa=False)
      IIO.save(path=self._fm.panorama_path(DIR.COR, SP.MASK),
               array=tools.uint8_image(cmask))

    # 실화상, 부위인식 파노라마 보정
    self._correct_others(correction=crct,
                         spectrum=SP.VIS,
                         crop_range=crop_range)
    self._correct_others(correction=crct,
                         spectrum=SP.SEG,
                         crop_range=crop_range)

    self.save_multilayer_panorama()

    logger.success('파노라마 왜곡 보정 완료')

  def save_multilayer_panorama(self):
    images = [
        self._fm.panorama_path(DIR.COR, x, error=True)
        for x in [SP.IR, SP.VIS, SP.SEG]
    ]
    images[0] = self._fm.color_path(images[0])
    path = self.fm.subdir(DIR.COR).joinpath('Panorama.webp')
    save_webp_images(*images, path=path)

  def run(self):
    separate = self._config['panorama']['separate']

    logger.info('Start extracting')
    self.extract()

    if not separate:
      logger.info('Start registering')
      self.register()

    logger.info('Start segmenting')
    self.segment()

    logger.info('Start panorama stitching')
    self.panorama()

    if not separate:
      logger.info('Start distortion correction')
      self.correct()
