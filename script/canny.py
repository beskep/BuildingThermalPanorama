import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import ndimage as ndi
from skimage import feature, filters
from skimage.util import random_noise

from pano.interface.common.init import init_project


def main():
  init_project(qt=True)
  sns.set_context(
      'talk',
      # font_scale=0.8,
  )

  # Generate noisy image of a square
  image = np.zeros((128, 128), dtype=float)
  image[32:-32, 32:-32] = 1

  image = ndi.rotate(image, 15, mode='constant')
  image = ndi.gaussian_filter(image, 4)
  image = random_noise(image, mode='speckle', mean=0.1)

  gaussian = filters.gaussian(image=image, sigma=1.5)

  blur = filters.gaussian(image=image, sigma=1.5)
  edges_sobel = np.hypot(
      np.abs(ndi.sobel(blur, axis=1)), np.abs(ndi.sobel(blur, axis=0))
  )

  edges_canny = feature.canny(image=image, sigma=3)

  # display results
  fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
  ax = axes.ravel()

  ax[0].imshow(image, cmap='gray')
  ax[0].set_title('(a) Original Image')

  ax[1].imshow(gaussian, cmap='gray')
  ax[1].set_title('(b) Gaussian Blur')

  ax[2].imshow(edges_sobel, cmap='gray')
  ax[2].set_title('(c) Sobel Kernel')

  ax[3].imshow(edges_canny, cmap='gray')
  ax[3].set_title('(d) Thresholding')

  for a in ax:
    a.axis('off')

  fig.tight_layout()
  plt.show()


if __name__ == '__main__':
  main()
