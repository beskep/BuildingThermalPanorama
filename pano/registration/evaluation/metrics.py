"""두 영상의 유사도/정합 정확도 평가"""

from functools import cached_property
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy as calculate_entropy

from pano.misc.tools import bin_size
from pano.misc.tools import normalize_image


def _check_shape(image1: np.ndarray, image2: np.ndarray):
  if image1.shape != image2.shape:
    raise ValueError('shape1 != shape2')


def compute_sse(image1: np.ndarray, image2: np.ndarray, norm=True):
  """
  Compute SSE (Sum of Squared Error) of image1 and image 2.

  Parameters
  ----------
  image1 : np.ndarray
      Target image 1
  image2 : np.ndarray
      Target image 2
  norm : bool, optional
      If true, normalize pixel values to range [0, 1], by default True

  Returns
  -------
  float
      Sum of Squared Error
  """
  _check_shape(image1, image2)

  if norm:
    image1 = normalize_image(image1)
    image2 = normalize_image(image2)

  sse = np.sum(np.square(image1 - image2))

  return sse


def compute_rmse(image1: np.ndarray, image2: np.ndarray, norm=True):
  """
  Compute RMSE (Root Mean Squared Error) of image1 and image 2.

  Parameters
  ----------
  image1 : np.ndarray
      Target image 1
  image2 : np.ndarray
      Target image 2
  norm : bool, optional
      If true, normalize pixel values to range [0, 1], by default True

  Returns
  -------
  float
      Root Mean Squared Error
  """
  sse = compute_sse(image1=image1, image2=image2, norm=norm)
  rmse = np.sqrt(sse / image1.size)

  return rmse


def compute_ncc(image1: np.ndarray, image2: np.ndarray):
  """
  Compute NCC (Normalized Cross Correlation) of image1 and image 2.

  Parameters
  ----------
  image1 : np.ndarray
      Target image 1
  image2 : np.ndarray
      Target image 2

  Returns
  -------
  float
      Normalized Cross Correlation

  References
  ----------
  [1] Dwith Chenna, Y., Ghassemi, P., Pfefer, T., Casamento, J., & Wang, Q.
  (2018). Free-Form Deformation Approach for Registration of Visible and
  Infrared Facial Images in Fever Screening. Sensors, 18(2), 125.
  https://doi.org/10.3390/s18010125
  """
  _check_shape(image1, image2)

  img1 = image1 - np.average(image1)
  img2 = image2 - np.average(image2)
  ncc = (np.sum(img1 * img2) /
         np.sqrt(np.sum(np.square(img1)) * np.sum(np.square(img2))))

  return ncc


def image_entropy(image: np.ndarray, bins: int, base=2) -> float:
  hist, edges = np.histogram(image.ravel(), bins=bins)
  entropy = calculate_entropy(hist, base=base)

  return entropy


class MutualInformation:
  """
  두 영상의 entropy, joint entropy, mutual information 계산

  Parameters
  ----------
  image1 : np.ndarray
      Target image 1
  image2 : np.ndarray
      Target image 2
  bins : Union[int, str], optional
      영상의 histogram, entropy 계산을 위한 bin 개수. str인 경우
      `numpy.histogram_bin_edges`의 설정을 따름.
  base : int, optional
      Entropy 계산을 위한 log의 밑.

  References
  ----------
  [1] Dwith Chenna, Y., Ghassemi, P., Pfefer, T., Casamento, J., & Wang, Q.
  (2018). Free-Form Deformation Approach for Registration of Visible and
  Infrared Facial Images in Fever Screening. Sensors, 18(2), 125.
  https://doi.org/10.3390/s18010125

  [2] Nan, A., Tennant, M., Rubin, U., & Ray, N. (2020, September).
  Drmime: Differentiable mutual information and matrix exponential
  for multi-resolution image registration. In Medical Imaging
  with Deep Learning (pp. 527-543). PMLR.
  """

  def __init__(self,
               image1: np.ndarray,
               image2: np.ndarray,
               bins: Union[int, str] = 'auto',
               base=2) -> None:
    _check_shape(image1, image2)

    self._image1 = image1
    self._image2 = image2
    self._base = base

    if isinstance(bins, str):
      self._bins = bin_size(image1=image1, image2=image2, bins=bins)
    else:
      self._bins = int(bins)

  @property
  def image1(self):
    return self._image1

  @property
  def image2(self):
    return self._image2

  @property
  def bins(self):
    return self._bins

  @property
  def base(self):
    return self._base

  @cached_property
  def image1_entropy(self):
    return image_entropy(image=self.image1, bins=self.bins, base=self.base)

  @cached_property
  def image2_entropy(self):
    return image_entropy(image=self.image2, bins=self.bins, base=self.base)

  @cached_property
  def joint_hist(self) -> np.ndarray:
    """
    Image 1, 2의 joint Histogram. Bin 개수는 초기 설정값을 따름.

    Returns
    -------
    np.ndarray
        Joint histogram (shape: (bins, bins))
    """
    return np.histogram2d(x=self._image1.ravel(),
                          y=self._image2.ravel(),
                          bins=self._bins)[0]

  @cached_property
  def joint_entropy(self) -> float:
    """
    Image 1, 2의 joint entropy. Histogram의 bin 개수, log의 밑은 초기
    설정값을 따름.

    Returns
    -------
    float
        Joint entropy
    """
    return calculate_entropy(self.joint_hist.ravel(), base=self.base)

  @cached_property
  def mutual_information(self) -> float:
    """
    Image 1, 2의 mutual information

    Returns
    -------
    float
        Mutual information

    References
    ----------
    [1] Dwith Chenna, Y., Ghassemi, P., Pfefer, T., Casamento, J., & Wang, Q.
    (2018). Free-Form Deformation Approach for Registration of Visible and
    Infrared Facial Images in Fever Screening. Sensors, 18(2), 125.
    https://doi.org/10.3390/s18010125
    """
    return self.image1_entropy + self.image2_entropy - self.joint_entropy

  @cached_property
  def mattes_mutual_information(self):
    """
    Mattes Mutual Information 계산.
    mutual_information()과 결과 동일함.

    Returns
    -------
    float
        Mattes mutual information

    References
    ----------
    [1] Nan, A., Tennant, M., Rubin, U., & Ray, N. (2020, September).
    Drmime: Differentiable mutual information and matrix exponential
    for multi-resolution image registration. In Medical Imaging
    with Deep Learning (pp. 527-543). PMLR.
    """
    hist = self.joint_hist
    pxy = hist / np.sum(hist)
    px = np.sum(pxy, axis=0)
    py = np.sum(pxy, axis=1).reshape([-1, 1])

    pxy_pxpy = np.divide(pxy, px * py, out=np.ones_like(pxy), where=(pxy != 0))
    mmi = np.sum(pxy * np.log2(pxy_pxpy)) / np.log2(self.base)

    return mmi

  def mi_plot(self) -> Tuple[plt.Figure, np.ndarray]:
    """
    두 영상, joint histogram을 plot

    Returns
    -------
    Tuple[plt.Figure, np.ndarray]
    """
    fig, axes = plt.subplots(1, 3)

    axes[0].imshow(self.image1, cmap='gray')
    axes[1].imshow(self.image2, cmap='gray')
    axes[2].matshow(self.joint_hist)

    axes[0].set_title('Image 1 | entropy: {:.3f}'.format(self.image1_entropy))
    axes[1].set_title('Image 2 | entropy: {:.3f}'.format(self.image2_entropy))
    axes[2].set_title('Entropy: {:.3f} | MI: {:.3f}'.format(
        self.joint_entropy, self.mutual_information))

    return fig, axes


def calculate_all_metrics(image1: np.ndarray,
                          image2: np.ndarray,
                          bins: Union[int, str] = 'auto',
                          base=2) -> dict:
  """
  지원하는 모든 metric을 계산

  Parameters
  ----------
  image1 : np.ndarray
      Target image 1
  image2 : np.ndarray
      Target image 2
  bins : Union[int, str], optional
      bins 개수 설정 (`MutualInformation` 설정 참조)
  base : int, optional
      log 함수의 밑 (`MutualInformation` 설정 참조)

  Returns
  -------
  dict
      {metric: value}
      metric list: ['RMSE', 'NCC', 'Entropy', 'MI']
  """
  mi = MutualInformation(image1=image1, image2=image2, bins=bins, base=base)
  rmse = compute_rmse(image1, image2)
  ncc = compute_ncc(image1, image2)

  metrics = {
      'RMSE': rmse,
      'NCC': ncc,
      'Entropy': mi.joint_entropy,
      'MI': mi.mutual_information,
      # 'MMI': mi.mattes_mutual_information,
  }

  return metrics
