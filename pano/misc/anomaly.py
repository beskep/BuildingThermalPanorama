from loguru import logger
import numpy as np
from numpy.typing import NDArray
from sklearn.mixture import GaussianMixture


def gaussian_mixture(array: NDArray, ks=(2, 3, 4, 5), **kwargs):
  gms = [GaussianMixture(n_components=k, **kwargs) for k in ks]

  for gm in gms:
    gm.fit(array)

  bic = [gm.bic(array) for gm in gms]
  logger.debug('BIC: {}', dict(zip(ks, (f'{x:.4e}' for x in bic))))

  model = gms[int(np.argmin(bic))]
  logger.debug('Selected k: {}', model.n_components)

  mask: NDArray = model.predict(array)

  return model, mask


def anomaly_threshold(array: NDArray, ks=(2, 3, 4, 5), **kwargs) -> float:
  """
  Kim, C., Choi, J.-S., Jang, H., & Kim, E.-J. (2021).
  Automatic Detection of Linear Thermal Bridges from Infrared Thermal Images
  Using Neural Network. Applied Sciences, 11(3), 931. https://doi.org/10.3390/app11030931

  Parameters
  ----------
  array : NDArray
      array
  ks : tuple, optional
      n_components of `GaussianMixture`, by default (2, 3, 4, 5)
  kwargs : dict, optional
      kwargs for `GaussianMixture`

  Returns
  -------
  float
      anomaly threshold
  """
  model, mask = gaussian_mixture(array.reshape([-1, 1]), ks=ks, **kwargs)
  counts = [np.sum(mask == i) for i in range(model.n_components)]
  ref = int(np.argmax(counts))  # 면적이 가장 많은 군집
  threshold = float(np.mean(array.ravel()[mask == ref]))

  return threshold
