"""
두 영상의 특징점 추출/매칭을 통해 정합하는 코드.

SimpleITK를 이용한 최적화 방법이 정확도가 더 높음.
"""

import numpy as np
import skimage.feature.util as _fu
from skimage import feature, transform
from skimage.measure import ransac

from .registrator import BaseRegistrator


class BaseDetector(_fu.DescriptorExtractor, _fu.FeatureDetector):
  """특징점 추출 클래스"""

  def __init__(self):
    super().__init__()
    self.keypoints_ = None
    self.descriptors_ = None

  @property
  def keypoints(self):
    return self.keypoints_

  @property
  def descriptors(self):
    return self.descriptors_

  def detect(self, image):
    raise NotImplementedError

  def extract(self, image, keypoints):
    raise NotImplementedError

  def detect_and_extract(self, image, *args, **kwargs):
    raise NotImplementedError


class HoughLineFeatureDetector(BaseDetector):
  """`probabilistic_hough_line`을 통해 추출한 선분을 특징점으로 삼는 detector"""

  def __init__(self, *, hough=True, length=True, normalize=True):
    super().__init__()
    self._lines: np.ndarray = np.array([])
    self._hough = hough
    self._length = length
    self._normalize = normalize
    self.keypoints_: np.ndarray

  @property
  def keypoints(self):
    return self.keypoints_[:, ::-1]

  @property
  def descriptors(self):
    return self.descriptors_

  @property
  def params(self):
    return {
      'hough': self._hough,
      'length': self._length,
      'normalize': self._normalize,
    }

  def detect(self, image, canny_kwargs=None, hough_kwargs=None):
    # pylint: disable=arguments-differ
    if canny_kwargs is None:
      canny_kwargs = {}
    if hough_kwargs is None:
      hough_kwargs = {}

    image_canny = feature.canny(image=image, **canny_kwargs)
    lines = transform.probabilistic_hough_line(image=image_canny, **hough_kwargs)

    self._lines = np.array(lines)
    self.keypoints_ = np.average(self._lines, axis=1)

  def extract(self, image, keypoints=None):
    lines = self._lines.astype('float', copy=True)

    if self._normalize:
      lines[:, :, 0] /= image.shape[1]
      lines[:, :, 1] /= image.shape[0]
      diag_length = np.sqrt(np.sum(np.square(image.shape)))
    else:
      diag_length = None

    desc = lines.reshape([-1, 4])

    if self._hough or self._length:
      delta = lines[:, 0] - lines[:, 1]

      if self._hough:
        theta = np.arctan2(-delta[:, 0], delta[:, 1])
        rho = np.abs(
          self._lines[:, 0, 0] * np.cos(theta) + self._lines[:, 0, 1] * np.sin(theta)
        )
        if self._normalize:
          rho /= diag_length

        desc = np.hstack([desc, np.vstack([theta, rho]).T])

      if self._length:
        length = np.sqrt(np.sum(np.square(delta), axis=1))
        if self._normalize:
          length /= diag_length

        desc = np.hstack([desc, length.reshape([-1, 1])])

    self.descriptors_ = desc

  def detect_and_extract(self, image, canny_kwargs=None, hough_kwargs=None):
    # pylint: disable=arguments-differ
    self.detect(image, canny_kwargs=canny_kwargs, hough_kwargs=hough_kwargs)
    self.extract(image)


class ORBDetector(feature.ORB):
  """ORB detector

  https://docs.opencv.org/4.5.2/d1/d89/tutorial_py_orb.html
  """


class BRIEFDetector(BaseDetector):
  """BRIEF detector

  https://docs.opencv.org/4.5.2/dc/d7d/tutorial_py_brief.html
  """

  def __init__(
    self, descriptor_size=256, patch_size=49, mode='normal', sigma=1, sample_seed=1
  ):
    super().__init__()
    self._extractor = feature.BRIEF(
      descriptor_size=descriptor_size,
      patch_size=patch_size,
      mode=mode,
      sigma=sigma,
      sample_seed=sample_seed,
    )

  def detect(self, image):
    corner = feature.corner_harris(image)
    self.keypoints_ = feature.corner_peaks(corner, min_distance=5, threshold_rel=0.1)
    assert self.keypoints_ is not None

  def extract(self, image, keypoints):
    self._extractor.extract(image=image, keypoints=keypoints)
    self.descriptors_ = self._extractor.descriptors

  def detect_and_extract(self, image, *args, **kwargs):
    self.detect(image)
    self.extract(image, self.keypoints)


class FeatureBasedRegistrator(BaseRegistrator):
  """Feature 기반 정합

  지정한 detector를 통해 특징점을 추출, RANSAC 알고리즘을 통해 매칭하고 두 영상을 정합
  """

  def __init__(
    self,
    detector: BaseDetector | None = None,
    *,
    hough_transform=False,
    fixed_kwargs=None,
    moving_kwargs=None,
  ) -> None:
    if detector is None:
      detector = HoughLineFeatureDetector()
    self._detector = detector

    self._hough_transform = hough_transform
    if hough_transform:
      if not isinstance(detector, HoughLineFeatureDetector):
        raise ValueError
      if not detector.params['hough']:
        raise ValueError

    if fixed_kwargs is None:
      fixed_kwargs = {}
    if moving_kwargs is None:
      moving_kwargs = {}

    self._fixed_kwargs = fixed_kwargs
    self._moving_kwargs = moving_kwargs

  @property
  def detector(self) -> BaseDetector:
    return self._detector

  @detector.setter
  def detector(self, value):
    self._detector = value

  @property
  def detector_name(self):
    return self.detector.__class__.__name__

  def detect(self, image, **kwargs):
    if hasattr(self._detector, 'detect_and_extract'):
      self._detector.detect_and_extract(image, **kwargs)
      keypoints = self._detector.keypoints
      descriptors = self._detector.descriptors
    else:
      self._detector.detect(image, **kwargs)
      keypoints = self._detector.keypoints
      descriptors = self._detector.keypoints

    return keypoints, descriptors

  @staticmethod
  def match_features(
    fixed_keypoints,
    fixed_descriptors,
    moving_keypoints,
    moving_descriptors,
  ):
    matches = feature.match_descriptors(
      descriptors1=fixed_descriptors,
      descriptors2=moving_descriptors,
      cross_check=True,
    )
    keypoints = (
      fixed_keypoints[matches[:, 0]][:, ::-1],
      moving_keypoints[matches[:, 1]][:, ::-1],
    )
    trsf, inliers = ransac(
      data=keypoints,
      model_class=transform.ProjectiveTransform,
      min_samples=min(20, matches.shape[0] - 1),
      residual_threshold=5,
      max_trials=500,
    )
    matches_ransac = matches[inliers]

    return trsf, matches_ransac

  def register(self, fixed_image: np.ndarray, moving_image: np.ndarray, **kwargs):
    fixed_kps, fixed_dscs = self.detect(fixed_image, **self._fixed_kwargs)
    moving_kps, moving_dscs = self.detect(moving_image, **self._moving_kwargs)

    trsf, matches = self.match_features(
      fixed_keypoints=fixed_kps,
      fixed_descriptors=fixed_dscs,
      moving_keypoints=moving_kps,
      moving_descriptors=moving_dscs,
    )
    if trsf is None:
      return None, None, None

    if self._hough_transform and isinstance(self.detector, HoughLineFeatureDetector):
      delta_avg = np.average(
        fixed_dscs[matches[:, 0], 4] - moving_dscs[matches[:, 1], 4]
      )
      hough_trsf = transform.SimilarityTransform(
        rotation=delta_avg
      )  # TODO rho, theta, scale 각각 추출하고 적용

      def register(image: np.ndarray) -> np.ndarray:
        return transform.warp(image, hough_trsf.inverse)

      registered = register(moving_image)
      mtx = hough_trsf.params
    else:

      def register(image: np.ndarray) -> np.ndarray:
        return transform.warp(image, trsf)

      registered = register(moving_image)
      mtx = trsf.params  # .param 맞나?

    ax = kwargs.get('ax', None)
    if ax is not None:
      feature.plot_matches(
        ax=ax,
        image1=fixed_image,
        image2=moving_image,
        keypoints1=fixed_kps,
        keypoints2=moving_kps,
        matches=matches,
      )
      ax.set_axis_off()

    return registered, register, mtx
