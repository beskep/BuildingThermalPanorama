import dataclasses as dc

import numpy as np
from skimage.exposure.exposure import equalize_hist
from skimage.feature import canny
from skimage.transform import probabilistic_hough_line

from pano.misc import tools


@dc.dataclass
class Edgelets:
  """Edgelets.

  Parameters
  ----------
  locations : np.ndarray
      선분 중심 좌표. (count, 2).
  directions : np.ndarray
      선분 방향 벡터. Normalize 안함. (count, 2).
  strength : np.ndarray
      선분 길이. (count,).
  """

  locations: np.ndarray
  directions: np.ndarray
  strengths: np.ndarray

  def __getitem__(self, index):
    if isinstance(index, int):
      index = [index]

    return Edgelets(
      locations=self.locations[index],
      directions=self.directions[index],
      strengths=self.strengths[index],
    )

  def __post_init__(self):
    count = self.locations.shape[0]

    for var, ndim in zip(
      ['locations', 'directions', 'strengths'], [2, 2, 1], strict=True
    ):
      arr: np.ndarray = getattr(self, var)
      dim = (count, 2) if ndim == 2 else (count,)  # noqa: PLR2004
      if arr.shape != dim:
        msg = f'Invalid {var} shape {arr.shape}'
        raise ValueError(msg)

  @property
  def count(self):
    return self.locations.shape[0]

  def astuple(self):
    return (self.locations, self.directions, self.strengths)

  def normalize(self):
    self.directions /= np.linalg.norm(self.directions, axis=1).reshape([-1, 1])

  def copy(self):
    return Edgelets(
      locations=self.locations.copy(),
      directions=self.directions.copy(),
      strengths=self.strengths.copy(),
    )

  def sort(self):
    argsort = np.flip(np.argsort(self.strengths))
    self.locations = self.locations[argsort]
    self.directions = self.directions[argsort]
    self.strengths = self.strengths[argsort]


@dc.dataclass
class CannyOptions:
  """`skimage.feature.canny` 옵션"""

  sigma: float = 1.0
  low_threshold: float | None = None
  high_threshold: float | None = None
  use_quantiles: bool = False


@dc.dataclass
class HoughOptions:
  """`skimage.transform.probabilistic_hough_line` 옵션"""

  threshold: int = 10
  line_length: int = 50
  line_gap: int = 10
  theta: np.ndarray | None = None


def edge_preprocess(image: np.ndarray, *, eqhist=True) -> np.ndarray:
  """
  전처리 (회색 변환 및 히스토그램 평활화)

  Parameters
  ----------
  image : np.ndarray
  eqhist : bool, optional

  Returns
  -------
  np.ndarray
  """
  na = np.isnan(image)
  image[na] = np.nanmin(image)

  image = tools.gray_image(image=image)
  if eqhist:
    image = equalize_hist(image=image)

  return image


def image2edges(
  image: np.ndarray,
  mask: np.ndarray | None = None,
  canny_option: CannyOptions | None = None,
  *,
  eqhist=True,
) -> np.ndarray:
  if image.ndim != 2:  # noqa: PLR2004
    msg = 'image.ndim != 2'
    raise ValueError(msg)

  image = edge_preprocess(image, eqhist=eqhist)

  if canny_option is None:
    canny_option = CannyOptions()

  return canny(image=tools.normalize_image(image), mask=mask, **dc.asdict(canny_option))


def edge2edgelets(
  edges: np.ndarray, hough_option: HoughOptions | None = None
) -> Edgelets:
  kwargs = {} if hough_option is None else dc.asdict(hough_option)
  lines = np.array(probabilistic_hough_line(edges, **kwargs))

  locations = np.average(lines, axis=1)
  directions = lines[:, 1, :] - lines[:, 0, :]
  strengths = np.linalg.norm(directions, ord=2, axis=1)
  directions = np.divide(directions, strengths.reshape([-1, 1]))

  return Edgelets(locations=locations, directions=directions, strengths=strengths)


def image2edgelets(
  image: np.ndarray,
  mask: np.ndarray | None = None,
  canny_option: CannyOptions | None = None,
  hough_option: HoughOptions | None = None,
  *,
  eqhist=True,
):
  edges = image2edges(image=image, mask=mask, canny_option=canny_option, eqhist=eqhist)
  edgelets = edge2edgelets(edges=edges, hough_option=hough_option)

  return edgelets, edges
