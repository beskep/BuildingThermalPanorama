"""
외피 열화상 파노라마 영상처리 알고리즘의 CLI 인터페이스
"""
from pathlib import Path
from typing import List, Optional

import utils

import matplotlib.pyplot as plt
import numpy as np
import yaml
from loguru import logger
from PIL import Image as PILImage
from rich.progress import track
from skimage.exposure import rescale_intensity
from skimage.io import imsave

import flir
import stitch
from misc import exif, tools
from misc.tools import ImageIO

CURRENT_DIR = Path('.').resolve()


class _DIR:
  RAW = 'Raw'
  IR = 'IR'
  VIS = 'VIS'
  PANO = 'Panorama'
  RGST = 'Registration'
  SEG = 'Segmentation'


class ThermalPanorama:
  # TODO: colormap

  def __init__(self, config) -> None:
    config = Path(config).resolve()
    logger.debug('Config path: `{}`', config)
    if not config.exists():
      raise FileNotFoundError(config)

    with open(config, 'r', encoding='utf-8') as f:
      self._config: dict = yaml.safe_load(f)

    wd = Path(self._config['path']['working_directory'])
    if not wd.is_absolute():
      wd = config.parent.joinpath(wd).resolve()
    if not wd.exists():
      raise FileNotFoundError(wd)

    self._wd = wd
    self._exts = (self._config['path']['save_ext']['IR'],
                  self._config['path']['save_ext']['VIS'])

    self._manufacturer = self._check_manufacturer()
    self._flir_ext = flir.FlirExtractor()
    logger.debug('Manufacturer: {}', self._manufacturer)

  def _check_manufacturer(self) -> str:
    raw_dir = self._dir(_DIR.RAW)

    flir_files = list(raw_dir.glob(self._config['path']['FLIR']['IR']))
    testo_ir_files = list(raw_dir.glob(self._config['path']['testo']['IR']))
    testo_vis_files = list(raw_dir.glob(self._config['path']['testo']['VIS']))

    if testo_ir_files and testo_vis_files:
      manufacturer = 'testo'
    elif flir_files:
      manufacturer = 'FLIR'
    else:
      raise ValueError('Raw 파일 설정 오류')

    return manufacturer

  def _files(self) -> List[Path]:
    # FIXME Raw 폴더에 원본이 없는 경우에도 지원?
    pattern = self._config['path'][self._manufacturer]['IR']
    files = [x.resolve() for x in self._dir(_DIR.RAW).glob(pattern)]

    if not files:
      logger.warning('{} 폴더에 영상 파일이 존재하지 않습니다.', _DIR.RAW)

    return files

  @property
  def working_dir(self) -> Path:
    return self._wd

  def _dir(self, folder: str):
    d = self.working_dir.joinpath(folder)
    if not d.exists():
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
    vis_suffix = self._config['path']['testo']['VIS'].replace('*', '')
    vis_path = path.with_name(path.stem + vis_suffix)
    if not vis_path.exists():
      raise FileNotFoundError(vis_path)

    ir = ImageIO.read_image(path=path)
    vis = ImageIO.read_image(path=vis_path)

    return ir, vis

  def _save_extracted_image(self,
                            fname: str,
                            ir: np.ndarray,
                            vis: np.ndarray,
                            meta: Optional[dict] = None):
    ir_path = self._dir(_DIR.IR).joinpath(fname)
    vis_path = self._dir(_DIR.VIS).joinpath(f'{fname}{self._exts[1]}')

    ImageIO.save_image_and_meta(path=ir_path,
                                array=ir,
                                exts=[self._exts[0], '.png'],
                                scale=True,
                                meta=meta,
                                save_meta=True)

    ImageIO.save_image(path=vis_path, array=vis)

  def _read_file(self, file: Path):
    if not file.exists():
      raise FileNotFoundError(file)

    ir_path = self._dir(_DIR.IR).joinpath(file.stem + self._exts[0])
    vis_path = self._dir(_DIR.VIS).joinpath(file.stem + self._exts[1])

    if ir_path.exists() and vis_path.exists():
      ir = ImageIO.read_image(ir_path)
      vis = ImageIO.read_image(vis_path)
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

  def _iter_files(self):
    files = self._files()
    for file in files:
      ir, vis = self._read_file(file)

      yield ir, vis

  def _read_files(self):
    ir_images = []
    vis_images = []

    for ir, vis in track(self._iter_files(),
                         'Reading...',
                         total=len(self._files()),
                         console=utils.console):
      ir_images.append(ir)
      vis_images.append(vis)

    return ir_images, vis_images

  def register(self):
    pass

  def segment(self):
    pass

  def panorama(self):
    wl = self._config['panorama']['target'].upper()
    if not wl in ('IR', 'VIS'):
      raise ValueError(wl)

    sopt: dict = self._config['panorama']['stitch']
    popt: dict = self._config['panorama']['preprocess'][wl]

    pano_dir = self._dir(_DIR.PANO)
    pano_fname = 'panorama_' + sopt['warp']
    if pano_dir.joinpath(pano_fname + '.png').exists():
      logger.info('기 생성된 파노라마 파일이 존재합니다.')
      return

    ir_images, vis_images = self._read_files()
    images = ir_images if wl == 'IR' else vis_images
    if not images:
      logger.warning('대상 영상이 없습니다.')
      return

    # 전처리
    prep = stitch.preprocess.PanoramaPreprocess(
        is_numeric=(images[0].ndim == 2),
        mask_threshold=popt['masking_threshold'],
        contrast=popt['contrast'],
        denoise=popt['denoise'])
    if 'bilateral_args' in popt:
      prep.set_bilateral_args(**popt['bilateral_args'])
    if 'gaussian_args' in popt:
      prep.set_gaussian_args(**popt['gaussian_args'])

    # 영상
    stitching_images = stitch.stitcher.StitchingImages(arrays=images)
    stitching_images.set_preprocess(prep)
    logger.debug('Stitching: 대상 영상 & 전처리 설정')

    # Stitcher
    stitcher = stitch.stitcher.Stitcher(mode=sopt['perspective'],
                                        compose_scale=sopt['compose_scale'],
                                        work_scale=sopt['work_scale'],
                                        warp_threshold=sopt['warp_threshold'])
    stitcher.warper_type = sopt['warp']
    logger.debug('Stitching: Stitcher 초기화')

    with utils.console.status('Stitching...'):
      panorama, mask, graph, indices = stitcher.stitch(
          images=stitching_images,
          masks=None,
          image_names=[x.stem for x in self._files()])

    if self._config['panorama']['stitch']['crop']:
      logger.debug('Crop panorama')
      x1, x2, y1, y2 = tools.mask_bbox(mask=mask, morphology_open=True)
      panorama = panorama[y1:y2, x1:x2]
      mask = mask[y1:y2, x1:x2]

    with utils.console.status('Saving...'):
      if wl == 'IR':
        meta = {
            'panorama': {
                'graph': graph,
                'image_indices': indices.ravel().tolist()
            }
        }
        ImageIO.save_image_and_meta(
            path=pano_dir.joinpath(pano_fname),
            # array=np.round(panorama, 3),
            array=panorama.astype(np.float16),  # TODO 추출부터 float16으로
            exts=['.npy', '.png'],
            scale=True,
            meta=meta,
            save_meta=True)
      else:
        panorama_image = tools.uint8_image(panorama)
        ImageIO.save_image(
            path=pano_dir.joinpath(pano_fname).with_suffix('.jpg'),
            array=panorama_image)

      ImageIO.save_image(path=pano_dir.joinpath(pano_fname + '_mask.png'),
                         array=rescale_intensity(mask, out_range='uint8'))

    logger.info('파노라마 생성 완료')
