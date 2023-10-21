"""시점 왜곡 보정"""

import dataclasses as dc
from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from skimage.transform import warp

from pano.misc import edgelet as edge
from pano.misc import tools
from pano.misc.tools import INTRP

from . import rectification

# ruff: noqa: N803, N806


class NotEnoughEdgeletsError(ValueError):
  pass


@dc.dataclass
class CorrectionOptions:
  """CorrectionOptions

  Parameters
  ----------
  threshold: float
      RANSAC VP 판단 알고리즘의 threshold [degree]
  ransac_iter: int
      RANSAC 반복 횟수
  clip_factor: float
  vp_iter: int
      (Vanishing point가 영상 내부에 존재하는 경우) 최대 반복 연산 횟수.
  """

  threshold: float = 5.0
  ransac_iter: int = 1000
  clip_factor: float = 0.0
  vp_iter: int = 5
  erode: int = 50  # [pixel/iterations]

  strict: bool = False
  margin: float = 0.1


class VanishingPoint:
  INSIDE = 0
  HORIZ = 1
  VERT = 2
  DIAG = 3

  SOFT: ClassVar[set[int]] = {HORIZ, VERT, DIAG}
  STRICT: ClassVar[set[int]] = {HORIZ, VERT}

  _POS_DICT: ClassVar[dict[tuple[bool, bool], int]] = {
      (True, True): DIAG,
      (True, False): HORIZ,
      (False, True): VERT,
      (False, False): INSIDE,
  }

  def __init__(self, array: np.ndarray) -> None:
    if array.shape != (3,):
      msg = f'Invalid vanishing point shape {array.shape}'
      raise ValueError(msg)

    self._array = array
    self._xy = np.divide(self.array[:2], self._array[2])
    self._pos: int | None = None

  @property
  def array(self) -> np.ndarray:
    return self._array

  @property
  def xy(self) -> np.ndarray:
    return self._xy

  @property
  def pos(self) -> int:
    if self._pos is None:
      raise ValueError('position not computed')

    return self._pos

  def compute_position(self, image_shape: tuple[int, ...], margin=0.1):
    shape = np.array([image_shape[1], image_shape[0]])
    rxy = (self.xy - shape / 2.0) / shape
    hv = np.abs(rxy) > 0.5 + margin

    self._pos = self._POS_DICT[(hv[0], hv[1])]

  def __str__(self) -> str:
    arr = np.array2string(self.array.ravel(), precision=2, separator=',')
    xy = np.array2string(self.xy.ravel(), precision=2, separator=',')
    return f'VanishingPoint(coor={arr}, xy={xy})'


class _Rectify:
  """rectification warper"""

  @staticmethod
  def ransac_vanishing_point(
      edgelets: edge.Edgelets, num_ransac_iter=2000, threshold_inlier=5.0
  ) -> VanishingPoint:
    vp_array = rectification.ransac_vanishing_point(
        edgelets=edgelets.astuple(),
        num_ransac_iter=num_ransac_iter,
        threshold_inlier=threshold_inlier,
    )
    return VanishingPoint(array=vp_array)

  @staticmethod
  def compute_votes(
      vp: VanishingPoint, edgelets: edge.Edgelets, threshold_inlier=10.0
  ) -> np.ndarray:
    return rectification.compute_votes(
        edgelets=edgelets.astuple(),
        model=vp.array,
        threshold_inlier=threshold_inlier,
    )

  @classmethod
  def remove_inliers(
      cls, vp: VanishingPoint, edgelets: edge.Edgelets, threshold_inlier=10.0
  ) -> edge.Edgelets:
    votes = cls.compute_votes(
        vp=vp, edgelets=edgelets, threshold_inlier=threshold_inlier
    )
    inliers = votes > 0
    return edge.Edgelets(
        locations=edgelets.locations[~inliers],
        directions=edgelets.directions[~inliers],
        strengths=edgelets.strengths[~inliers],
    )


@dc.dataclass
class Homography:
  Hproject: np.ndarray | None = None
  Haffine: np.ndarray | None = None
  Htranslate: np.ndarray | None = None

  input_shape: tuple[int, ...] | None = None
  output_shape: tuple[int, int] | None = None

  def available(self):
    return self.Htranslate is not None

  def warp(self, image: np.ndarray, order=INTRP.BiCubic):
    if self.Htranslate is None:
      raise ValueError('Homography not set')

    if image.shape[:2] != self.input_shape:
      msg = (
          f'입력한 영상의 해상도 {image.shape[:2]}가 '
          '시점 왜곡을 추정한 영상의 해상도 '
          f'{self.input_shape}와 다릅니다.'
      )
      raise ValueError(msg)

    return warp(
        image=image,
        inverse_map=np.linalg.inv(self.Htranslate),
        order=order,
        output_shape=self.output_shape,
        preserve_range=True,
    )


@dc.dataclass
class Correction:
  """Perspective correction 결과"""

  edges: np.ndarray
  edgelets: edge.Edgelets

  vp1: VanishingPoint | None
  vp2: VanishingPoint | None

  homography: Homography
  crop_range: tools.CropRange | None

  def success(self) -> bool:
    return self.homography.available()

  def _warp_and_crop(self, image: np.ndarray, order=INTRP.BiCubic) -> np.ndarray:
    img = self.homography.warp(image=image, order=int(order))
    if self.crop_range is not None:
      img = self.crop_range.crop(img)

    return img

  def correct(
      self, image: np.ndarray, mask: np.ndarray | None = None, order=INTRP.BiCubic
  ) -> tuple[np.ndarray, np.ndarray | None]:
    """
    저장된 왜곡 보정 결과를 통해 새 영상 보정

    Parameters
    ----------
    image : np.ndarray
        보정 대상 영상.
        보정 계수를 추측한 영상과 크기 (width, height)가 동일해야 함.
    mask : np.ndarray
        보정 대상 영상의 분석 대상 영역 mask.
    order : Union[Interpolation, int]
        Interpolation 방법

    Returns
    -------
    np.ndarray
        Corrected image
    Optional[np.ndarray]
        Corrected mask

    Raises
    ------
    ValueError
        if image.shape[:2] != self.image.shape[:2]
    """
    img = self._warp_and_crop(image, order=order)

    if mask is None:
      msk = None
    else:
      msk = self._warp_and_crop(mask, order=INTRP.NearestNeighbor)
      img[np.logical_not(msk)] = np.nanmin(img)

    return img, msk

  def _visualize_model(self, ax: plt.Axes, vp: VanishingPoint):
    if self.edgelets is None:
      raise ValueError

    inliers = (
        _Rectify.compute_votes(vp=vp, edgelets=self.edgelets, threshold_inlier=10) > 0
    )

    locations = self.edgelets.locations[inliers]
    for idx in range(locations.shape[0]):
      ax.plot(
          [locations[idx, 0], vp.xy[0]],  # 소실점 선분 x 좌표
          [locations[idx, 1], vp.xy[1]],  # 소실점 선분 y 좌표
          'b--',
      )

  def process_plot(self, image: np.ndarray) -> tuple[plt.Figure, plt.Axes]:
    """보정 과정을 확인할 수 있는 matplotlib plot 생성"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # preprocessed image
    axes[0, 0].set_title('Preped Image')
    axes[0, 0].imshow(edge.edge_preprocess(image))

    # edges (canny) `
    axes[0, 1].set_title('Edges')
    axes[0, 1].imshow(self.edges)

    # lines (canny and hough lines)
    axes[0, 2].set_title('Lines')
    axes[0, 2].imshow(self.edges)

    half_strengths = self.edgelets.strengths.reshape([-1, 1]) / 2.0
    pt1 = self.edgelets.locations - self.edgelets.directions * half_strengths
    pt2 = self.edgelets.locations + self.edgelets.directions * half_strengths
    for idx in range(self.edgelets.count):
      axes[0, 2].plot(
          [pt1[idx, 0], pt2[idx, 0]],
          [pt1[idx, 1], pt2[idx, 1]],
          'r-',
      )

    # vanishing points
    axes[1, 0].set_title('Vanishing point 1')
    if self.vp1 is not None:
      axes[1, 0].imshow(self.edges)
      self._visualize_model(ax=axes[1, 0], vp=self.vp1)

    axes[1, 1].set_title('Vanishing point 2')
    if self.vp2 is not None:
      axes[1, 1].imshow(self.edges)
      self._visualize_model(ax=axes[1, 1], vp=self.vp2)

    # corrected image
    axes[1, 2].set_title('Corrected Image')
    if self.success():
      axes[1, 2].imshow(self.correct(image)[0])

    for ax in axes.ravel():
      ax.set_axis_off()

    fig.tight_layout()

    return fig, axes


class PerspectiveCorrection:
  MIN_EDGELETS = 10

  def __init__(
      self,
      canny: edge.CannyOptions | dict | None = None,
      hough: edge.HoughOptions | dict | None = None,
      correction: CorrectionOptions | dict | None = None,
  ) -> None:
    """
    영상의 소실점으로부터 시점 왜곡을 보정하는 Homography 행렬 추정.

    Parameters
    ----------
    canny : edge.CannyOptions | dict | None, optional
        `skimage.feature.canny` 옵션
    hough : edge.HoughOptions | dict | None, optional
        `skimage.transform.probabilistic_hough_line` 옵션
    correction : CorrectionOptions | dict | None, optional
        소실점, 시점 보정 행렬의 수치적 추정을 위한 옵션
    """
    if not isinstance(canny, edge.CannyOptions):
      canny = edge.CannyOptions(**(canny or {}))
    if not isinstance(hough, edge.HoughOptions):
      hough = edge.HoughOptions(**(hough or {}))
    if not isinstance(correction, CorrectionOptions):
      correction = CorrectionOptions(**(correction or {}))

    self._canny_options = canny
    self._hough_options = hough
    self._opt = correction

  @staticmethod
  def _compute_affine(vp1: np.ndarray, vp2: np.ndarray, H: np.ndarray) -> np.ndarray:
    # Find directions corresponding to vanishing points
    v_post1 = np.dot(H, vp1)
    v_post2 = np.dot(H, vp2)
    v_post1 = v_post1 / np.sqrt(v_post1[0] ** 2 + v_post1[1] ** 2)
    v_post2 = v_post2 / np.sqrt(v_post2[0] ** 2 + v_post2[1] ** 2)

    directions = np.array([
        [v_post1[0], -v_post1[0], v_post2[0], -v_post2[0]],
        [v_post1[1], -v_post1[1], v_post2[1], -v_post2[1]],
    ])

    thetas = np.arctan2(directions[0], directions[1])

    # Find direction closest to horizontal axis
    h_ind = np.argmin(np.abs(thetas))

    # Find positive angle among the rest for the vertical axis
    if h_ind // 2 == 0:
      v_ind = 2 + np.argmax([thetas[2], thetas[3]])
    else:
      v_ind = np.argmax([thetas[2], thetas[3]])

    A1 = np.array([
        [directions[0, v_ind], directions[0, h_ind], 0],
        [directions[1, v_ind], directions[1, h_ind], 0],
        [0, 0, 1],
    ])

    # Might be a reflection. If so, remove reflection.
    if np.linalg.det(A1) < 0:
      A1[:, 0] = -A1[:, 0]

    return np.linalg.inv(A1)

  @staticmethod
  def _compute_translate(H: np.ndarray, shape: tuple[int, ...], clip_factor: float):
    points = [
        [0, 0, shape[1], shape[1]],
        [0, shape[0], 0, shape[0]],
        [1, 1, 1, 1],
    ]
    cords = np.dot(H, points)
    cords = cords[:2] / cords[2]

    tx = min(0.0, cords[0].min())
    ty = min(0.0, cords[1].min())
    max_x = int(cords[0].max() - tx)
    max_y = int(cords[1].max() - ty)

    if clip_factor:
      max_offset = max(shape) * clip_factor / 2.0
      tx = max(tx, -max_offset)
      ty = max(ty, -max_offset)
      max_x = min(max_x, int(-tx + max_offset))
      max_y = min(max_y, int(-ty + max_offset))

    T = np.array([
        [1, 0, -tx],
        [0, 1, -ty],
        [0, 0, 1],
    ])

    return T, max_x, max_y

  def compute_homography(
      self,
      image_shape: tuple[int, ...],
      vp1: np.ndarray,
      vp2: np.ndarray,
  ) -> Homography:
    """
    두 개의 소실점으로부터 시점 보정을 위한 homography 행렬 추정

    Parameters
    ----------
    image_shape: Tuple[int, ...]
        대상 영상 shape
    vp1 : np.ndarray
        소실점 1
    vp2 : np.ndarray
        소실점 2

    Returns
    -------
    Homography
    """
    # Find Projective Transform
    vanishing_line = np.cross(vp1, vp2)
    Hproject = np.eye(3)
    Hproject[2] = vanishing_line / vanishing_line[2]
    Hproject = Hproject / Hproject[2, 2]

    # Computes affine transform to make axes corresponding to
    # vanishing points orthogonal
    A = self._compute_affine(vp1=vp1, vp2=vp2, H=Hproject)
    Haffine = np.dot(A, Hproject)

    # Image is translated so that the image is not missed.
    T, max_x, max_y = self._compute_translate(
        H=Haffine, shape=image_shape, clip_factor=self._opt.clip_factor
    )

    Htranslate = np.dot(T, Haffine)

    return Homography(
        Hproject=Hproject,
        Haffine=Haffine,
        Htranslate=Htranslate,
        input_shape=image_shape,
        output_shape=(max_y, max_x),
    )

  def _estimate_vanishing_point(
      self,
      edgelets: edge.Edgelets,
      image_shape: tuple[int, ...],
      target: int | None = None,
  ) -> tuple[VanishingPoint, edge.Edgelets]:
    """Vanishing point 추정

    RANSAC을 통해 주어진 edgelet으로부터 vanishing point를 추정하고,
    vanishing point와 vanishing point에 수렴하는 inlier를 제외한 edgelets 반환.

    추정한 vanishing point가 대상 영상 내부에 존재하는 경우,
    해당 inlier를 제외하고 재연산.

    Parameters
    ----------
    edgelets : Edgelets
    image_shape : Tuple[int, ...]
    target : Optional[int]
        추정 대상 VP의 위치 (VanishingPoint의 HORIZ, VERT)

    Returns
    -------
    VanishingPoint
        추정한 소실점
    Edgelets
        소실점에 수렴하는 edgelet을 제외한 edgelets

    Raises
    ------
    NotEnoughEdgelets
        조건에 맞지 않는 경우를 제외한 edgelet이 10개 미만인 경우
    ValueError
        Vanishing point 추정 실패
    """
    vp = None
    for idx in range(self._opt.vp_iter + 1):
      logger.debug('Vanishing point iter {}', idx)
      if edgelets.count < self.MIN_EDGELETS:
        raise NotEnoughEdgeletsError('Not enough edgelets')

      vp = _Rectify.ransac_vanishing_point(
          edgelets=edgelets,
          num_ransac_iter=self._opt.ransac_iter,
          threshold_inlier=self._opt.threshold,
      )
      edgelets = _Rectify.remove_inliers(
          vp=vp,
          edgelets=edgelets,
          threshold_inlier=(2 * self._opt.threshold),
      )

      vp.compute_position(image_shape=image_shape, margin=self._opt.margin)

      if target is None:
        if self._opt.strict:
          valid_positions = VanishingPoint.STRICT
        else:
          valid_positions = VanishingPoint.SOFT
        if vp.pos in valid_positions:
          break
      elif vp.pos == target:
        break

    else:
      raise ValueError('Vanishing point 추정 실패')

    return vp, edgelets

  def _estimate_vanishing_points(
      self,
      edgelets: edge.Edgelets,
      image_shape: tuple[int, ...],
  ) -> tuple[VanishingPoint | None, VanishingPoint | None]:
    """
    두 개의 vanishing point 추정

    Parameters
    ----------
    edgelets : Edgelets
        edgelets
    image_shape : Tuple[int, ...]
        대상 영상 shape

    Returns
    -------
    Optional[VanishingPoint]
        Vanishing point 1
    Optional[VanishingPoint]
        Vanishing point 2
    """
    vp1, vp2 = None, None
    vp1, edgelets2 = self._estimate_vanishing_point(
        edgelets=edgelets, image_shape=image_shape
    )
    logger.debug('VP1: {}', vp1)
    if vp1 is None:
      return None, None

    target = None
    if self._opt.strict:
      target = (
          VanishingPoint.VERT
          if vp1.pos == VanishingPoint.HORIZ
          else VanishingPoint.HORIZ
      )
    vp2, _ = self._estimate_vanishing_point(
        edgelets=edgelets2, image_shape=image_shape, target=target
    )

    # 영상 중심으로부터 두 vp의 방향이 유사한지 확인
    if vp2 is not None:
      center = np.array([image_shape[1], image_shape[0]]) / 2.0
      vp1xy = vp1.xy - center
      vp2xy = vp2.xy - center

      delta = np.rad2deg(
          np.arctan2(vp1xy[1], vp1xy[0])  # vp1 angle
          - np.arctan2(vp2xy[1], vp2xy[0])  # vp2 angle
      )

      if np.abs(delta) < 2 * self._opt.threshold:
        logger.warning('두 vanishing point가 유사함 (각도차: {:.2e} degree)', delta)
        vp2 = None

    logger.debug('VP2: {}', vp2)

    return vp1, vp2

  def perspective_correct(
      self, image: np.ndarray, mask: np.ndarray | None = None
  ) -> Correction:
    """
    시점 왜곡 보정

    Parameters
    ----------
    image : np.ndarray
        대상 영상
    mask : Optional[np.ndarray]
        대상 영상의 분석 영역 mask

    Returns
    -------
    Correction
    """
    if mask is None:
      edge_mask = None
    else:
      edge_mask = mask

      if self._opt.erode:
        logger.debug('Erode mask (iterations: {})', self._opt.erode)
        edge_mask = tools.erode(
            edge_mask.astype(np.uint8), iterations=self._opt.erode
        ).astype(bool)

    edgelets, edges = edge.image2edgelets(
        image=image,
        mask=edge_mask,
        canny_option=self._canny_options,
        hough_option=self._hough_options,
    )

    vp1, vp2 = self._estimate_vanishing_points(
        edgelets=edgelets, image_shape=image.shape[:2]
    )

    crop_range = None
    if vp1 is None or vp2 is None:
      # vp 추정 실패, 중간 과정만 저장
      homography = Homography()
    else:
      # Homography 행렬 추정
      homography = self.compute_homography(
          image_shape=image.shape[:2], vp1=vp1.array, vp2=vp2.array
      )
      if mask is not None:
        crop_range = tools.crop_mask(
            mask=homography.warp(mask, order=INTRP.NearestNeighbor)
        )[0]

    return Correction(
        edges=edges,
        edgelets=edgelets,
        vp1=vp1,
        vp2=vp2,
        homography=homography,
        crop_range=crop_range,
    )
