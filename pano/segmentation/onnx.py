"""segmentation-models-pytorch 학습, onnx export한 DeepLabV3+ 모델"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap
from matplotlib.patches import Patch
from onnxruntime import InferenceSession
from PIL import Image

RSMP = Image.Resampling


class SmpModel:
  CMAP = 'Dark2'
  IMG_SIZE = (640, 640)
  TITLES = ('Input Image', 'Segmentation Overlay', 'Segmentation Map')
  LABELS: tuple[str, ...] = ('Background', 'Wall', 'Window', 'etc.')

  def __init__(self, path: str) -> None:
    self._sess = InferenceSession(path, providers=['CPUExecutionProvider'])
    self._input_name = self._sess.get_inputs()[0].name

    self._cmap = get_cmap(self.CMAP)
    self._handles = [
        Patch(facecolor=self._cmap(i), label=label)
        for i, label in enumerate(self.LABELS)
    ]

  def predict(self, src: str | bytes | Path):
    image = Image.open(src)
    inputs = np.array(image.resize(self.IMG_SIZE, resample=RSMP.LANCZOS))
    inputs = np.moveaxis(inputs, -1, 0).astype(np.float32)  # HWC -> CHW

    outputs = self._sess.run(output_names=None, input_feed={self._input_name: inputs})
    pred = np.argmax(outputs[0][0], axis=0).astype(np.int8)
    resized = Image.fromarray(pred).resize(image.size, resample=RSMP.NEAREST)

    return np.array(resized)

  def visualization(self, src: str | Path | Image.Image | np.ndarray, mask: np.ndarray):
    if isinstance(src, Image.Image | np.ndarray):  # noqa: SIM108
      image = src
    else:
      image = Image.open(src)

    mask_cmap = self._cmap(mask)

    fig, axes = plt.subplots(1, 3, figsize=(8, 3))
    axes[0].imshow(image)
    axes[1].imshow(image)
    axes[1].imshow(mask_cmap, alpha=0.75)
    axes[2].imshow(mask_cmap)
    for ax, title in zip(axes, self.TITLES, strict=True):
      ax.set_axis_off()
      ax.set_title(title)

    fig.legend(handles=self._handles, loc='lower center', ncol=5)  # TODO cmap 개수
    fig.tight_layout()

    return fig, axes


class SmpModel9(SmpModel):
  LABELS = (
      'Background',
      'Wall',
      'Window',
      'etc.',
      'Tree',
      'Lamp',
      'Car',
      'Banner',
      'Canopy',
  )

  def predict(self, src: str | bytes | Path):
    mask = super().predict(src)

    # 5번이었던 `etc` index를 3으로 당기기
    #    background, wall, window, tree, lamp, etc, car, banner, canopy
    # -> background, wall, window, etc, tree, lamp, car, banner, canopy
    etc = mask == 5  # noqa: PLR2004
    mask[np.isin(mask, [3, 4])] += 1
    mask[etc] = 3

    return mask
