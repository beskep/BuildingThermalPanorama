"""영상을 stitch하고 파노라마를 생성"""

# ruff: noqa: PLR2004

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import IntEnum
from itertools import repeat, starmap

import cv2 as cv
import numpy as np
from loguru import logger
from skimage.exposure import rescale_intensity

from pano.misc.tools import CropRange, crop_mask

_AVAILABLE_WARPER = {
  'affine',
  'compressedPlaneA1.5B1',
  'compressedPlaneA2B1',
  'compressedPlanePortraitA1.5B1',
  'compressedPlanePortraitA2B1',
  'cylindrical',
  'fisheye',
  'mercator',
  'paniniA1.5B1',
  'paniniA2B1',
  'paniniPortraitA1.5B1',
  'paniniPortraitA2B1',
  'plane',
  'spherical',
  'stereographic',
  'transverseMercator',
}


def _merge_mask(m1: np.ndarray | None, m2: np.ndarray | None):
  match m1, m2:
    case None, None:
      return None
    case _, None:
      return m1
    case None, _:
      return m2

  # mypy
  assert m1 is not None
  assert m2 is not None

  return np.logical_and(m1, m2)


class StitchError(ValueError):
  pass


class Interpolation(IntEnum):
  NEAREST = 0
  LINEAR = 1
  CUBIC = 2
  AREA = 3
  LANCZOS4 = 4
  LINEAR_EXACT = 5
  NEAREST_EXACT = 6


@dataclass
class Panorama:
  """Stitch 결과"""

  panorama: np.ndarray
  mask: np.ndarray
  graph: str
  indices: list
  cameras: list
  crop_range: CropRange | None
  image_names: list[str]

  def included(self):
    return [self.image_names[x] for x in sorted(self.indices)]

  def not_included(self):
    ni = [
      self.image_names[x] for x in range(len(self.image_names)) if x not in self.indices
    ]
    return ni if len(ni) else None

  def graph_list(self):
    gl = self.graph.split('\n')
    assert gl[0].startswith('graph')
    assert gl[-1] == '}'

    return [x.replace("'", '').replace('"', '') for x in gl[1:-1]]


class StitchingImages:
  def __init__(self, arrays: list[np.ndarray], preprocess: Callable | None = None):
    """Stitching 대상 이미지

    Parameters
    ----------
    arrays : List[np.ndarray]
        원본 이미지. dtype 상관 없음.
    preprocess : callable, optional
        Preprocessing function, by default None
        (image, mask)를 반환해야 함

    Raises
    ------
    ValueError
        preprocess가 None이나 callable이 아닌 경우
    """
    if (preprocess is not None) and not callable(preprocess):
      raise ValueError

    self._arrays = []
    self._arrays_count = 0
    self._ndim = 0

    self.arrays = arrays
    self._preprocess = preprocess

    minmax = np.array([[np.nanmin(x), np.nanmax(x)] for x in arrays])
    self._in_range = (np.min(minmax[:, 0]), np.max(minmax[:, 1]))

  @property
  def arrays(self) -> list[np.ndarray]:
    """영상 원본"""
    return self._arrays

  @arrays.setter
  def arrays(self, value: list[np.ndarray]):
    ndim = value[0].ndim
    if not all(x.ndim == ndim for x in value):
      msg = '영상의 채널 수가 동일하지 않음'
      raise ValueError(msg)

    self._arrays = value
    self._arrays_count = len(value)
    self._ndim = ndim

  @property
  def count(self):
    """영상 개수"""
    return self._arrays_count

  @property
  def ndim(self) -> int:
    """영상의 차원"""
    return self._ndim

  def set_preprocess(self, fn: Callable):
    self._preprocess = fn

  def select_images(self, indices: Iterable):
    """
    주어진 인덱스에 해당하는 영상 반환

    Parameters
    ----------
    indices : Iterable
        선택할 영상의 인덱스 목록
    """
    self.arrays = [self.arrays[x] for x in indices]

  def scale(self, image: np.ndarray, out_range) -> np.ndarray:
    """
    영상의 픽셀값 범위를 전체 영상 (`arrays`)의 범위로부터 `out_range` 범위로 조정.

    Parameters
    ----------
    image : np.ndarray
        대상 영상.
    out_range : Any
        조정할 픽셀값 범위 (`skimage.exposure.rescale_intensity` 참조).

    Returns
    -------
    np.ndarray
    """
    return rescale_intensity(image=image, in_range=self._in_range, out_range=out_range)

  def unscale(self, image: np.ndarray, out_range='image') -> np.ndarray:
    """
    `out_range` 범위로 픽셀값이 조정됐던 영상을 원 범위로 변환.

    Parameters
    ----------
    image : np.ndarray
        대상 영상.
    out_range : Any
        과거 변경했던 영상 범위 (`skimage.exposure.rescale_intensity` 참조).

    Returns
    -------
    np.ndarray
    """
    return rescale_intensity(image=image, in_range=out_range, out_range=self._in_range)

  def preprocess(self) -> tuple[list[np.ndarray], list[np.ndarray | None]]:
    """
    전처리 함수를 적용한 영상과 대상 영역 마스크 반환

    Returns
    -------
    images : List[np.ndarray]
        전처리를 적용한 영상 리스트
    masks : Optional[np.ndarray]
        마스크 리스트
    """
    if self._preprocess is None:
      images = self.arrays
      masks: list[np.ndarray | None] = [None for _ in range(self.count)]
    else:
      prep = [self._preprocess(x.copy()) for x in self.arrays]
      images = [x[0] for x in prep]
      masks = [x[1] for x in prep]

    if any(x.dtype != np.uint8 for x in images):
      images = [self.scale(x, out_range=np.uint8) for x in images]

    return images, masks


class Stitcher:
  def __init__(
    self,
    mode='pano',
    features_finder: cv.Feature2D | None = None,
    compose_scale=1.0,
    work_scale=1.0,
    warp_threshold=20.0,
    *,
    try_cuda=False,
  ):
    """
    파노라마 생성자

    Parameters
    ----------
    mode : str, optional
        파노라마 생성 모드. {`'pano'`, `'scan'`}. `'pano'`를 권장함.
    features_finder : Optional[cv.Feature2D], optional
        대상 영상의 특징점 추출 알고리즘. 지정하지 않는 경우 ORB 알고리즘 적용.
    compose_scale : float, optional
        구성될 영상의 원본 영상 대비 해상도 배율
    work_scale : float, optional
        특징 추출/정합을 위한 작업 영상의 원본 영상 대비 해상도 비율
    warp_threshold : float, optional
        정합되는 영상의 변형 한계. 변형되는 영상의 넓이 또는 폭이 원본 영상의
        `warp_threshold`배 이상인 경우, 오류로 인해 과도한 변형이 적용되는 것으로
        판단하고 해당 영상을 제외함.
    try_cuda : bool, optional
        NVIDIA CUDA 사용여부. 지원하는 OpenCV 버전을 설치해야 함.
    """
    self._mode = None
    self._estimator = None
    self._features_finder = None
    self._features_matcher = None
    self._bundle_adjuster = None
    self._refine_mask = None
    self._warper = None
    self._warper_type = None
    self._blend_type = 'no'
    self._blend_strength = 0.05
    self._interp: int = cv.INTER_LANCZOS4

    self._compose_scale = compose_scale
    self._work_scale = work_scale
    self._compose_work_aspect = compose_scale / work_scale
    self._warp_threshold = warp_threshold
    self._try_cuda = try_cuda

    self.features_finder = features_finder
    self.set_mode(mode.lower())

  @property
  def estimator(self) -> cv.detail_Estimator:
    """
    정합/영상 변환을 위한 Camera parameter 추정 방법.

    `mode`에 따라 결정됨 (`set_mode` 참조).
    """
    return self._estimator

  @estimator.setter
  def estimator(self, value: cv.detail_Estimator):
    if not isinstance(value, cv.detail_Estimator):
      raise TypeError

    self._estimator = value

  @property
  def features_finder(self) -> cv.Feature2D:
    """영상의 특징점 추출 알고리즘 (지정하지 않는 경우 ORB 알고리즘 적용)."""
    if self._features_finder is None:
      self._features_finder = cv.ORB_create()

    return self._features_finder

  @features_finder.setter
  def features_finder(self, value: cv.Feature2D):
    self._features_finder = value

  @property
  def features_matcher(self) -> cv.detail_FeaturesMatcher:
    """추출한 영상 특징점의 matching 방법. `set_features_matcher`로 설정."""
    return self._features_matcher

  @features_matcher.setter
  def features_matcher(self, value: cv.detail_FeaturesMatcher):
    self._features_matcher = value

  @property
  def bundle_adjuster(self) -> cv.detail_BundleAdjusterBase:
    """카메라 위치 최적화를 위한 Bundle Adjust 알고리즘"""
    return self._bundle_adjuster

  @bundle_adjuster.setter
  def bundle_adjuster(self, value: cv.detail_BundleAdjusterBase):
    self._bundle_adjuster = value

  @property
  def refine_mask(self) -> np.ndarray:
    """Bundle adjuster가 이용하는 Refinement mask"""
    if self._refine_mask is None:
      self.set_bundle_adjuster_refine_mask()
    assert self._refine_mask is not None

    return self._refine_mask

  @property
  def warper(self) -> cv.PyRotationWarper:
    """
    파노라마를 구성하는 영상의 변형 (warp) 방법.

    3차원 공간에 위치한 영상을 평면에 투영하는 방법을 결정함.
    `warper_type`, `set_warper`를 통해 설정.

    Returns
    -------
    cv.PyRotationWarper

    Raises
    ------
    ValueError
        Warper가 설정되지 않은 경우
    """
    if self._warper is None:
      msg = 'Warper 설정되지 않음'
      raise ValueError(msg)

    return self._warper

  @property
  def warper_type(self) -> str | None:
    """
    `warper`의 종류. `available_warper_types` 중에서 선택 가능함.

    `'plane'` : Rectilinear Projection.

    `'spherical'` : Stereographic Projection.

    References
    ----------
    [1] https://wiki.panotools.org/Stereographic_Projection
    """
    return self._warper_type

  @warper_type.setter
  def warper_type(self, value: str):
    if value not in _AVAILABLE_WARPER:
      raise ValueError(value)

    self._warper_type = value

  @property
  def blend_type(self) -> str:
    """
    파노라마를 구성하는 영상의 밝기 차이를 조정하기 위한 Blend 방법

    {`'multiband'`, `'feather'`, `'no'`}
    """
    return self._blend_type

  @blend_type.setter
  def blend_type(self, value: bool | str):
    if isinstance(value, str):
      value = value.lower()
      if value not in {'multiband', 'feather', 'no'}:
        msg = f'invalid blend_type: "{value}"'
        raise ValueError(msg)
    else:
      value = 'feather' if value else 'no'

    self._blend_type = value

  @property
  def blend_strength(self) -> float:
    """Blend 강도"""
    return self._blend_strength

  @blend_strength.setter
  def blend_strength(self, value: float):
    if not 0 <= value <= 1:
      msg = f'blender strength not in [0, 1]: {value}'
      raise ValueError(msg)

    self._blend_strength = value

  @property
  def interp(self):
    return self._interp

  @interp.setter
  def interp(self, value: int | Interpolation):
    self._interp = int(value)

  def set_features_matcher(
    self,
    matcher='affine',
    confidence: float | None = None,
    range_width=-1,
  ):
    """Set features matcher.

    Parameters
    ----------
    matcher : str
        matcher type
    confidence : Optional[float], optional
        Confidence for feature matching step.
        The default is 0.3 for ORB and 0.65 for other feature types.
    range_width : int
        uses range_width to limit number of images to match with
    """
    if confidence is None:
      if self._features_finder is None or isinstance(self._features_finder, cv.ORB):
        confidence = 0.30
      else:
        confidence = 0.65

    if matcher == 'affine':
      matcher = cv.detail_AffineBestOf2NearestMatcher(
        full_affine=False, try_use_gpu=self._try_cuda, match_conf=confidence
      )
    elif range_width == -1:
      matcher = cv.detail.BestOf2NearestMatcher_create(
        try_use_gpu=self._try_cuda, match_conf=confidence
      )
    else:
      matcher = cv.detail_BestOf2NearestRangeMatcher(
        range_width=range_width,
        try_use_gpu=self._try_cuda,
        match_conf=confidence,
      )
    self.features_matcher = matcher

  def set_bundle_adjuster_refine_mask(
    self, *, fx=True, skew=True, ppx=True, aspect=True, ppy=True
  ):
    """Set refinement mask for bundle adjustment"""
    refine_mask = np.zeros([3, 3], dtype=np.uint8)

    masks = [fx, skew, ppx, aspect, ppy]
    rows = [0, 0, 0, 1, 1]
    cols = [0, 1, 2, 1, 2]
    for mask, row, col in zip(masks, rows, cols, strict=True):
      if mask:
        refine_mask[row, col] = 1

    self._refine_mask = refine_mask

  def set_warper(self, scale):
    self._warper = cv.PyRotationWarper(type=self.warper_type, scale=scale)

  def set_mode(self, mode: str):
    """
    파노라마 생성 모드 ({`'pano'`, `'scan'`}) 및 모드별 적절한 알고리즘 설정.

    Parameters
    ----------
    mode : str
        "pano", "scan"

    Raises
    ------
    ValueError
        pano 값 오류
    """
    if mode.startswith('pano'):
      self.estimator = cv.detail_HomographyBasedEstimator()
      self.set_features_matcher('pano')
      self.bundle_adjuster = cv.detail_BundleAdjusterRay()
      self.warper_type = 'spherical'
    elif mode == 'scan':
      self.estimator = cv.detail_AffineBasedEstimator()
      self.set_features_matcher('affine')
      self.bundle_adjuster = cv.detail_BundleAdjusterAffinePartial()
      self.warper_type = 'affine'
    else:
      raise ValueError(mode)

    self._mode = mode

  def find_features(self, image: np.ndarray, mask: np.ndarray | None):
    """
    대상 영상의 특징점 추출

    Parameters
    ----------
    image : np.ndarray
        대상 영상
    mask : Optional[np.ndarray]
        대상 영역의 마스크

    Returns
    -------
    detail_ImageFeatures

    Raises
    ------
    ValueError
        self.features_finder가 지정되지 않은 경우
    """
    if self.features_finder is None:
      msg = 'features_finder가 지정되지 않음'
      raise ValueError(msg)

    return cv.detail.computeImageFeatures2(
      featuresFinder=self.features_finder, image=image, mask=mask
    )

  def stitch(
    self,
    images: StitchingImages,
    masks: list[np.ndarray | None] | None = None,
    names: list[str] | None = None,
    *,
    crop=True,
  ) -> Panorama:
    """
    영상의 특징점을 기반으로 정합 (stitch)하여 파노라마 영상 생성

    Parameters
    ----------
    images : StitchingImages
        대상 영상 목록.
    masks : Optional[List[Optional[np.ndarray]]], optional
        대상 영상의 마스크 목록., by default None
    names : Optional[List[str]], optional
        대상 영상의 이름 목록. 미지정 시 `Image n` 형식으로 지정.
    crop : bool
        `True`인 경우, 파노라마 영상 중 데이터가 존재하는 부분만 crop

    Returns
    -------
    Panorama
    """
    if names is None:
      names = [f'Image {x + 1}' for x in range(images.count)]

    prep_images, prep_masks = images.preprocess()

    if masks is None:
      masks = prep_masks
    else:
      masks = list(starmap(_merge_mask, zip(masks, prep_masks, strict=True)))

    # camera matrix 계산
    cameras, indices, matches_graph = self.calculate_camera_matrix(
      images=prep_images, image_names=names
    )

    if len(indices) != len(prep_images):
      images.select_images(indices=indices)
      logger.info(
        'Stitching에 필요 없는 이미지 제거 (indices: {})',
        set(range(len(prep_images))) - set(indices),
      )

    panorama, panorama_mask, warp_indices = self.warp_and_blend(
      images=images, cameras=cameras, masks=masks, names=names
    )
    indices = [indices[x] for x in warp_indices]

    if images.ndim == 2:
      # 원본 영상이 2차원인 경우 (열화상), 첫 번째 채널만 추출
      panorama = panorama[:, :, 0]

    # 파노라마 영상 중 데이터 없는 부분에 최소값 대입
    panorama[np.logical_not(panorama_mask)] = np.min(panorama)

    crop_range = None
    if crop:
      # 데이터가 존재하는 부분의 bounding box만 crop
      logger.debug('Crop panorama')
      crop_range, panorama_mask = crop_mask(mask=panorama_mask, morphology_open=True)
      if crop_range.cropped:
        panorama = crop_range.crop(panorama)

    return Panorama(
      panorama=panorama,
      mask=panorama_mask,
      graph=matches_graph,
      indices=indices,
      cameras=cameras,
      crop_range=crop_range,
      image_names=names,
    )

  def calculate_camera_matrix(
    self,
    images: list[np.ndarray],
    image_names: list[str],
  ) -> tuple[list[cv.detail_CameraParams], list[int], str]:
    """
    영상의 특성 추출/매칭을 통해 camera matrix 추정

    Parameters
    ----------
    images : List[np.ndarray]
        대상 영상 목록.
    image_names : List[str]
        대상 영상의 이름 목록.

    Returns
    -------
    cameras : List[cv.detail_CameraParams]
        각 영상의 camera parameter
    indices : List[int]
        매칭된 영상의 index 목록
    matches_graph : str
        매칭 graph (영상 간 연결 관계) 정보

    Raises
    ------
    StitchError
        이미지 개수 부족
    """
    logger.trace('Feature finding and matching')
    # note: find_features에는 마스크 적용하지 않음
    # (~mask에 0 대입한 영상으로 feature 탐색)
    features = [self.find_features(image=image, mask=None) for image in images]

    pairwise_matches = self.features_matcher.apply2(features=features)
    self.features_matcher.collectGarbage()

    indices_arr: np.ndarray = cv.detail.leaveBiggestComponent(
      features=features, pairwise_matches=pairwise_matches, conf_threshold=0.3
    )
    indices: list = indices_arr.ravel().tolist()
    if len(indices) < 2:
      msg = 'Need more images (valid images are less than two)'
      raise StitchError(msg)

    logger.trace('Matches graph')
    matches_graph: str = cv.detail.matchesGraphAsString(
      image_names, pairwise_matches=pairwise_matches, conf_threshold=1.0
    )

    logger.trace('Estimate camera')
    estimate_status, cameras = self.estimator.apply(
      features=features, pairwise_matches=pairwise_matches, cameras=None
    )
    if not estimate_status:
      msg = 'Homography estimation failed'
      raise StitchError(msg)

    logger.trace('Bundle adjust')
    self.bundle_adjuster.setConfThresh(1)
    self.bundle_adjuster.setRefinementMask(self.refine_mask)

    for cam in cameras:
      cam.R = cam.R.astype(np.float32)

    adjuster_status, cameras = self.bundle_adjuster.apply(
      features=features, pairwise_matches=pairwise_matches, cameras=cameras
    )
    if not adjuster_status:
      msg = 'Camera parameters adjusting failed'
      raise StitchError(msg)

    logger.trace('Wave correction')
    rs = [np.copy(camera.R) for camera in cameras]

    try:
      cv.detail.waveCorrect(rs, cv.detail.WAVE_CORRECT_HORIZ)
    except cv.error:
      logger.debug('Wave correction failed')
    else:
      for camera, r in zip(cameras, rs, strict=True):
        camera.R = r

    return cameras, indices, matches_graph

  def _warp_image(
    self,
    image: np.ndarray,
    mask: np.ndarray | None,
    camera: cv.detail_CameraParams,
  ) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """
    Camera parameter에 따라 영상을 변형.

    Parameters
    ----------
    image : np.ndarray
        대상 영상.
    mask : Optional[np.ndarray]
        대상 영상의 유의미한 영역 마스크.
    camera : cv.detail_CameraParams
        Camera parameter.

    Returns
    -------
    warped_image : np.ndarray
        변형된 영상
    warped_mask : np.ndarray
        변형된 영상의 마스크
    roi: Tuple[int, int, int, int]
        Region of interest

    Raises
    ------
    cv.error
        지나치게 과도한 영상 변형 시
    """  # noqa: DOC502
    if not np.isclose(self._compose_work_aspect, 1.0, rtol=1e-05, atol=0):
      camera.focal *= self._compose_work_aspect
      camera.ppx *= self._compose_work_aspect
      camera.ppy *= self._compose_work_aspect

    size = (
      int(image.shape[1] * self._compose_scale),
      int(image.shape[0] * self._compose_scale),
    )
    kmat = camera.K().astype(np.float32)
    rmat = camera.R
    roi: tuple[int, int, int, int] = self.warper.warpRoi(src_size=size, K=kmat, R=rmat)

    warped_shape = (roi[2] - roi[0], roi[3] - roi[1])
    if any(image.shape[x] * self._warp_threshold < warped_shape[x] for x in range(2)):
      raise cv.error  # noqa: DOC501

    if abs(self._compose_scale - 1) > 0.1:
      img = cv.resize(
        src=image,
        dsize=None,
        fx=self._compose_scale,
        fy=self._compose_scale,
        interpolation=self._interp,
      )
      if mask is not None:
        mask = cv.resize(
          src=mask,
          dsize=None,
          fx=self._compose_scale,
          fy=self._compose_scale,
          interpolation=cv.INTER_NEAREST,
        )
    else:
      img = image

    # NOTE `(roi[0], roi[1]) == corner
    _corner, warped_image = self.warper.warp(
      src=img,
      K=kmat,
      R=rmat,
      interp_mode=self._interp,
      border_mode=cv.BORDER_CONSTANT,
    )

    if mask is None:
      mask = np.ones(shape=img.shape[:2], dtype=np.uint8)

    _, warped_mask = self.warper.warp(
      src=mask,
      K=kmat,
      R=rmat,
      interp_mode=cv.INTER_NEAREST,
      border_mode=cv.BORDER_CONSTANT,
    )

    return warped_image, warped_mask, roi

  def _warp_images(
    self,
    images: Iterable[np.ndarray],
    cameras: list[cv.detail_CameraParams],
    masks: Iterable[np.ndarray | None] | None = None,
    names: Iterable[str] | None = None,
  ) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, list[int]]:
    """
    대상 영상들을 Camera parameter에 따라 변형.

    과도한 변형이 일어나는 경우 오류로 판단하고 파노라마를 구성하는 영상에서 제외함.

    Parameters
    ----------
    images : Iterable[np.ndarray]
        대상 영상 목록.
    cameras : List[cv.detail_CameraParams]
        대상 영상의 camera parameter 목록
    masks : Optional[Iterable[np.ndarray]]
        대상 영상의 마스크 목록.
    names: Optional[Iterable[str]]
        대상 영상의 이름 목록

    Returns
    -------
    warped_images : List[np.ndarray]
        변형된 영상 목록
    warped_masks : List[np.ndarray]
        마스크 목록
    rois : np.ndarray
        Region of interest 목록
    indices : List[int]
        제외되지 않은 영상의 index 목록
    """
    scale = np.median([x.focal for x in cameras])
    self.set_warper(scale=scale)

    if masks is None:
      masks = repeat(None)

    names = repeat('') if names is None else (f' ({x})' for x in names)

    warped_images = []
    warped_masks = []
    rois = []
    indices = []
    for idx, args in enumerate(zip(images, masks, cameras, names, strict=False)):
      try:
        wi, wm, roi = self._warp_image(*args[:3])
      except cv.error:
        logger.error('과도한 변형으로 인해 {}번 영상을 제외합니다.{}', idx + 1, args[3])
      else:
        warped_images.append(wi)
        warped_masks.append(wm)
        rois.append(roi)
        indices.append(idx)

    return warped_images, warped_masks, np.array(rois), indices

  def _blend(
    self,
    images: list[np.ndarray],
    masks: list[np.ndarray],
    rois: np.ndarray,
  ) -> tuple[np.ndarray, np.ndarray]:
    """
    파노라마를 구성하는 영상들의 밝기 조정

    Parameters
    ----------
    images : List[np.ndarray]
        대상 영상 목록. int16 형식만 입력 받음.
        1채널인 경우 자동으로 3채널 영상으로 변환.
    masks : List[np.ndarray]
        대상 영역의 마스크 목록
    rois : np.ndarray
        Region of interest

    Returns
    -------
    stitched_image : np.ndarray
        파노라마 영상
    stitched_mask : np.ndarray
        파노라마 영역의 마스크

    Raises
    ------
    ValueError
        blend_type 오류
    """
    corners = [(x[0].item(), x[1].item()) for x in rois]
    dst_size = cv.detail.resultRoi(corners=corners, images=images)

    # blend width 계산, blender type 결정
    blend_width = np.sqrt(dst_size[2] * dst_size[3]) * self.blend_strength
    blend_type = 'no' if blend_width < 1 else self.blend_type
    logger.trace('Blend type: {}', blend_type.title())

    # blender 생성
    if blend_type == 'no':
      blender = cv.detail.Blender_createDefault(
        type=cv.detail.Blender_NO, try_gpu=self._try_cuda
      )
    elif blend_type == 'multiband':
      blender = cv.detail_MultiBandBlender()
      bands_count = (np.log2(blend_width) - 1.0).astype(int)
      blender.setNumBands(bands_count)
    elif blend_type == 'feather':
      blender = cv.detail_FeatherBlender()
      blender.setSharpness(1.0 / blend_width)
    else:
      raise ValueError

    # blend
    blender.prepare(dst_size)
    for image, mask, corner in zip(images, masks, corners, strict=True):
      if image.ndim == 2:
        img = np.repeat(image[:, :, np.newaxis], repeats=3, axis=2)
      else:
        img = image

      blender.feed(img=img, mask=mask, tl=corner)

    stitched_image, stitched_mask = blender.blend(dst=None, dst_mask=None)

    return stitched_image, stitched_mask

  def warp_and_blend(
    self,
    images: StitchingImages,
    cameras: list[cv.detail_CameraParams],
    masks: Iterable[np.ndarray | None] | None = None,
    names: Iterable[str] | None = None,
  ) -> tuple[np.ndarray, np.ndarray, list[int]]:
    # warp each image
    warped_images, warped_masks, rois, indices = self._warp_images(
      images=images.arrays, cameras=cameras, masks=masks, names=names
    )

    # stitch and blend
    if self.blend_type == 'feather':

      def _scale(img):
        return images.scale(img, out_range=np.uint8).astype(np.int16)

    else:

      def _scale(img):
        return images.scale(img, out_range=np.int16)

    scaled_images = [_scale(x) for x in warped_images]
    scaled_panorama, panorama_mask = self._blend(
      images=scaled_images, masks=warped_masks, rois=rois
    )

    panorama = images.unscale(image=scaled_panorama)

    return panorama, panorama_mask, indices
