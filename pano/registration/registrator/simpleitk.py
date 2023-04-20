"""SimpleITK 라이브러리와 수치적 최적화를 통해 영상 정합"""

import enum
from typing import Callable, Optional, Tuple, Union

from loguru import logger
import numpy as np
from numpy.linalg import inv
import SimpleITK as sitk  # noqa: N813
from skimage.transform import estimate_transform

from pano.misc.tools import bin_size

from .registrator import BaseRegistrator
from .registrator import RegisteringImage  # noqa: F401
from .registrator import RegistrationPreprocess  # noqa: F401


class Metric(enum.Enum):
  """최적화 대상 metric"""
  MeanSquare = enum.auto()
  Corr = enum.auto()
  ANTSNCorr = enum.auto()
  JointHistMI = enum.auto()
  MattesMI = enum.auto()


class Transformation(enum.Enum):
  """정합을 위한 영상 변환 방법"""
  Similarity = enum.auto()
  Affine = enum.auto()


def to_sitk_image(image: np.ndarray, set_origin=False) -> sitk.Image:
  """
  SimpleITK 형식 image로 변환

  Parameters
  ----------
  image : np.ndarray
      대상 영상
  set_origin : bool, optional
      True일 경우 영상 중심을 회전축으로 설정

  Returns
  -------
  sitk.Image
      변환된 영상
  """
  sitk_image = sitk.GetImageFromArray(image)
  if set_origin:
    sitk_image.SetOrigin(np.divide(image.shape, 2.0))

  return sitk_image


class SITKRegistrator(BaseRegistrator):
  # pylint: disable=no-member

  def __init__(self,
               transformation=Transformation.Similarity,
               metric=Metric.JointHistMI,
               optimizer='powell',
               bins: Union[str, int] = 'auto') -> None:
    """
    SimpleITK 라이브러리와 수치적 최적화를 통해 영상 정합

    Parameters
    ----------
    transformation : Transformation, optional
        정합하기 위한 moving_image의 변환 방법
    metric : Metric, optional
        최적화 대상 metric
    optimizer : str, optional
        최적화 방법. {'powell', 'gradient_descent'}
    bins : Union[str, int], optional
        `metric`이 `JointHistMI` 혹은 `MattesMI`인 경우 히스토그램과
        엔트로피 계산을 위한 bins 개수.
        string으로 설정하는 경우 `numpy.histogram_bin_edges`의 설정을 따름.
    """
    self._method = sitk.ImageRegistrationMethod()

    self._transformation: Transformation = transformation
    self._metric: Metric = metric
    self._bins = bins

    self._scale0: Optional[float] = None  # initial scale factor
    self._trnsl0: Optional[tuple] = None  # inital translation factor

    self._metric_options = {}
    if isinstance(bins, int):
      self.set_metric(metric=metric)

    self._method.SetMetricSamplingStrategy(self._method.NONE)  # 모든 픽셀 고려
    self._method.SetMetricSamplingPercentage(1.0)
    self.set_multi_resolution()

    self._method.SetInterpolator(sitk.sitkLinear)

    if optimizer == 'powell':
      self._method.SetOptimizerAsPowell(numberOfIterations=500,
                                        maximumLineIterations=100,
                                        stepLength=1,
                                        stepTolerance=1e-8,
                                        valueTolerance=1e-8)
    elif optimizer == 'gradient_descent':
      # TODO optimizer option 설정
      self._method.SetOptimizerAsGradientDescent(
          learningRate=0.01,
          numberOfIterations=500,
          convergenceMinimumValue=1e-4,
          convergenceWindowSize=20,
          maximumStepSizeInPhysicalUnits=2)
    else:
      raise ValueError(
          f'Optimizer `{optimizer}` not in ["powell", "gradient_descent"]')

    # 각 parameter의 scaling factor 결정 방법
    self._method.SetOptimizerScalesFromPhysicalShift()

  @property
  def method(self) -> sitk.ImageRegistrationMethod:
    """
    simpleitk 라이브러리의 영상 정합을 위한 클래스

    Returns
    -------
    sitk.ImageRegistrationMethod
    """
    return self._method

  @property
  def metric(self) -> Metric:
    """
    정합 결과 판단 지표의 종류

    Returns
    -------
    Metric
    """
    return self._metric

  def _set_method_metric(self,
                         metric: Metric,
                         bins=20,
                         pdf_var=1.5,
                         ants_radius=2):
    if metric is Metric.MeanSquare:
      self.method.SetMetricAsMeanSquares()
    elif metric is Metric.Corr:
      self.method.SetMetricAsCorrelation()
    elif metric is Metric.ANTSNCorr:
      self.method.SetMetricAsANTSNeighborhoodCorrelation(radius=ants_radius)
    elif metric is Metric.JointHistMI:
      self.method.SetMetricAsJointHistogramMutualInformation(
          numberOfHistogramBins=bins, varianceForJointPDFSmoothing=pdf_var)
    elif metric is Metric.MattesMI:
      self.method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=bins)
    else:
      raise ValueError(metric)

  def set_metric(self,
                 metric: Metric,
                 bins: Union[int, str] = 'auto',
                 pdf_var=1.5,
                 ants_radius=2):
    self._metric = metric
    self._bins = bins
    self._metric_options = {'pdf_var': pdf_var, 'ants_radius': ants_radius}

    if isinstance(bins, int):
      self._set_method_metric(metric=metric,
                              bins=bins,
                              pdf_var=pdf_var,
                              ants_radius=ants_radius)

  @property
  def transformation(self):
    return self._transformation

  @transformation.setter
  def transformation(self, value: Transformation):
    self._transformation = value

  @staticmethod
  def _get_transformation(transformation: Transformation,
                          scale: Optional[float] = None,
                          translation: Optional[tuple] = None):
    if scale is None:
      logger.warning('초기 scale이 설정되지 않았습니다. 정합 결과가 부정확할 수 있습니다.')
      scale = 1.0

    trsf: sitk.Transform

    # 초기 scale 설정
    if transformation is Transformation.Similarity:
      # params: (scale, angle, translation0, translation1)
      trsf = sitk.Similarity2DTransform()

      trsf.SetScale(scale)
      if translation is not None:
        trsf.SetTranslation(translation)

    elif transformation is Transformation.Affine:
      # params: mtx.flatten(), translation[0], translation[1]
      trsf = sitk.AffineTransform(2)

      if translation is None:
        translation = (0.0, 0.0)
      trsf.SetParameters(
          (scale, 0.0, 0.0, scale, translation[0], translation[1]))
    else:
      raise ValueError(transformation)

    return trsf

  @staticmethod
  def get_transform_matrix(transform: sitk.Transform) -> np.ndarray:
    """
    추정한 transform으로부터 transform matrix 추정

    Parameters
    ----------
    transform : sitk.Transform
        최적화를 통해 추정한 transform

    Returns
    -------
    np.ndarray
        Transform matrix
    """
    params = transform.GetParameters()
    assert len(params) in (4, 6)  # similarity의 경우 4개, affine은 6개

    # 수치적으로 tranform matrix 추정
    # (simpleitk의 문서가 부실해서 parameter를 해석할 수 없음)
    src = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype='float')
    dst = np.array([transform.TransformPoint(x) for x in src])
    ttype = 'similarity' if len(params) == 4 else 'affine'
    trsf = estimate_transform(ttype=ttype, src=src, dst=dst)
    matrix = inv(trsf.params)

    return matrix

  def set_multi_resolution(self,
                           shrink_factors=(4, 2, 1),
                           smoothing_sigmas=(2, 1, 0)):
    self.method.SetShrinkFactorsPerLevel(shrink_factors)
    self.method.SetSmoothingSigmasPerLevel(smoothing_sigmas)
    self.method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

  def set_initial_params(self,
                         scale: Optional[float] = None,
                         fixed_alpha: Optional[float] = None,
                         moving_alpha: Optional[float] = None,
                         translation: Optional[list] = None):
    """
    정합을 위한 translation의 초기 패러미터 지정.

    scale (fixed/moving)의 경우 fixed, moving image의 화각
    (AOV; Angle of View)을 통해 추정 가능.
    모두 None으로 입력하면 초기 scale factor를 설정하지 않음.

    translation은 기존 정합 결과를 참고해서 추정값 적용.
    e.g. FLIR T540의 경우 (5, -14).

    Parameters
    ----------
    scale : Optional[float], optional
        Initial scale factor
    fixed_alpha : Optional[float], optional
        AOV of fixed image [radian]
    moving_alpha : Optional[float], optional
        AOV of moving image [radian]
    translation : Optional[tuple], optional
        Inital translation
    """
    if scale is not None:
      self._scale0 = scale
    elif not (fixed_alpha is None or moving_alpha is None):
      self._scale0 = np.tan(fixed_alpha / 2.0) / np.tan(moving_alpha / 2.0)
    else:
      self._scale0 = None

    self._trnsl0 = None if translation is None else tuple(translation)
    logger.debug('Initial scale: {} | translation: {}', self._scale0,
                 self._trnsl0)

  def _registration_results(
      self,
      fixed_simg: sitk.Image,
      moving_simg: sitk.Image,
      trsf: sitk.Transform,
      set_origin: bool,
  ) -> Tuple[np.ndarray, Optional[Callable], Optional[np.ndarray]]:
    # 정합된 영상 추출
    registered = sitk.Resample(image1=moving_simg,
                               referenceImage=fixed_simg,
                               transform=trsf,
                               interpolator=sitk.sitkLinear,
                               defaultPixelValue=0.0,
                               outputPixelType=moving_simg.GetPixelID())
    registered_image = sitk.GetArrayFromImage(registered)

    # 변환 행렬
    matrix = self.get_transform_matrix(trsf)

    def register(image: np.ndarray) -> np.ndarray:
      """
      새 영상을 받아 추정한 transformation을 적용하는 함수

      Parameters
      ----------
      image : np.ndarray
          변환 대상 영상

      Returns
      -------
      np.ndarray
          변환된 영상
      """
      simg = to_sitk_image(image, set_origin=set_origin)
      registered_simg = sitk.Resample(simg,
                                      referenceImage=fixed_simg,
                                      transform=trsf,
                                      interpolator=sitk.sitkLinear,
                                      defaultPixelValue=0.0,
                                      outputPixelType=moving_simg.GetPixelID())
      registered_image = sitk.GetArrayFromImage(registered_simg)

      return registered_image

    return registered_image, register, matrix

  def register(
      self,
      fixed_image: np.ndarray,
      moving_image: np.ndarray,
      **kwargs,
  ) -> Tuple[np.ndarray, Optional[Callable], Optional[np.ndarray]]:
    """
    Register image

    Parameters
    ----------
    fixed_image : np.ndarray
        Fixed image (정합의 기준이 되는 고정된 영상)
    moving_image : np.ndarray
        Moving image (정합을 위해 변환하는 영상)
    **kwargs: dict, optional
        set_origin : bool
            `True`일 경우 초기 회전 중심을 영상 중심으로 설정 (default: `False`).
            `True`인 경우 **transform matrix를 적용 전 영상 중심을 조정해야 함**.

    Returns
    -------
    np.ndarray
        Registered image
    Optional[Callable]
        Registration function (f: moving image -> registered image)
    Optional[np.ndarray]
        Transform matrix
    """
    set_origin = kwargs.get('set_origin', False)

    fixed = to_sitk_image(fixed_image, set_origin=set_origin)
    moving = to_sitk_image(moving_image, set_origin=set_origin)

    if isinstance(self._bins, str):
      # 적절한 bin 개수 추정
      bins = bin_size(fixed_image, moving_image, bins=self._bins)
      self._set_method_metric(self.metric, bins=bins, **self._metric_options)

    trsf = self._get_transformation(self._transformation,
                                    scale=self._scale0,
                                    translation=self._trnsl0)

    # 초기 transformation 설정
    # (`operationMode=GEOMETRY` -> 영상의 기하학적 중심을 초기 회전축으로 설정)
    initial_trsf = sitk.CenteredTransformInitializer(
        fixedImage=fixed,
        movingImage=moving,
        transform=trsf,
        operationMode=sitk.CenteredTransformInitializerFilter.GEOMETRY)
    self.method.SetInitialTransform(initial_trsf, inPlace=False)

    # 연산
    final_trsf: sitk.Transform = self.method.Execute(
        sitk.Cast(fixed, sitk.sitkFloat32),
        sitk.Cast(moving, sitk.sitkFloat32),
    )
    logger.debug('Optimizer stopping condition: {}',
                 self.method.GetOptimizerStopConditionDescription())
    logger.debug('Final param: {}', np.round(final_trsf.GetParameters(), 3))

    return self._registration_results(fixed_simg=fixed,
                                      moving_simg=moving,
                                      trsf=final_trsf,
                                      set_origin=set_origin)

  # def prep_and_register(
  #     self,
  #     fixed_image: np.ndarray,
  #     moving_image: np.ndarray,
  #     preprocess: RegistrationPreprocess,
  #     **kwargs,
  # ) -> Tuple[RegisteringImage, RegisteringImage]:
  #   return super().prep_and_register(fixed_image, moving_image, preprocess,
  #                                    **kwargs)
  #   return super().prep_and_register(fixed_image, moving_image, preprocess,
  #                                    **kwargs)
