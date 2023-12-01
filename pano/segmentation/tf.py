"""DeepLabV3+"""

# ruff: noqa

from pathlib import Path

import numpy as np
import PIL.Image
import tensorflow.compat.v1 as tf
from loguru import logger
from matplotlib import gridspec
from matplotlib import pyplot as plt
from PIL.Image import Image as PILImage
from rich.progress import track
from skimage.transform import resize as _resize


def tf_gpu_memory_config():
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  return tf.Session(config=config)


def create_pascal_label_colormap() -> np.ndarray:
  """
  Create a label colormap used in PASCAL VOC segmentation benchmark.

  Returns
  -------
  np.ndarray
      A Colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=int)
  ind = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((ind >> channel) & 1) << shift
    ind >>= 3

  return colormap


def label_to_color_image(label: np.ndarray, cmap='pascal') -> np.ndarray:
  """
  Add color defined by the dataset colormap to the label.

  Parameters
  ----------
  label : np.ndarray
      A 2D array with integer type, storing the segmentation label.
  cmap : str, optional
      Colormap. 'pascal' or matplotlib's colormap name.

  Returns
  -------
  np.ndarray
      A 2D array with floating type. The element of the array is the color
      indexed by the corresponding element in the input label to the PASCAL
      color map.

  Raises
  ------
  ValueError
      If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  if cmap == 'pascal':
    colormap = create_pascal_label_colormap()
  else:
    colormap = np.array(plt.get_cmap(cmap).colors) * 255

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]


LABEL_NAMES = np.asarray(['Background', 'Wall', 'Window', 'etc.'])

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
# FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)


def vis_segmentation(
  image: np.ndarray | PILImage, seg_map: np.ndarray, *, show=False, cmap='pascal'
) -> tuple[plt.Figure, np.ndarray]:
  """
  Visualizes input image, segmentation map and overlay view.

  Parameters
  ----------
  image : Union[np.ndarray, PILImage]
      Target image
  seg_map : np.ndarray
      Segmentation map.
  show : bool, optional
      Whether show matplotlib plot, by default False
  cmap : str, optional
      Colormap, by default 'pascal'

  Returns
  -------
  fig : plt.Figure
      figure
  seg_image: np.ndarray
      Segmentation map with color
  """
  fig = plt.figure(figsize=(15, 5))
  grid_spec = gridspec.GridSpec(1, 4, width_ratios=[8, 8, 8, 1])

  plt.subplot(grid_spec[0])
  plt.imshow(image)
  plt.axis('off')
  plt.title('input image')

  plt.subplot(grid_spec[1])
  seg_image = label_to_color_image(seg_map, cmap=cmap).astype(np.uint8)
  plt.imshow(seg_image)
  plt.axis('off')
  plt.title('segmentation map')

  plt.subplot(grid_spec[2])
  plt.imshow(image)
  plt.imshow(seg_image, alpha=0.7)
  plt.axis('off')
  plt.title('segmentation overlay')

  full_color_map = label_to_color_image(FULL_LABEL_MAP, cmap=cmap)
  unique_labels = np.unique(seg_map)
  ax = plt.subplot(grid_spec[3])
  plt.imshow(full_color_map[unique_labels].astype(np.uint8), interpolation='nearest')
  ax.yaxis.tick_right()
  plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
  plt.xticks([], [])
  ax.tick_params(width=0.0)
  plt.grid('off')

  fig.tight_layout()

  if show:
    plt.show()

  return fig, seg_image


def predict(
  model_path: str,
  images: list[np.ndarray],
  output: Path,
  cmap='Dark2',
  names: list[str] | None = None,
):
  """
  입력받은 각 영상을 DeepLabV3+ 모델을 통해 segment하고 결과를 저장.

  Parameters
  ----------
  model_path : str
      학습한 DeepLabV3+ 모델의 frozen graph 경로.
  images : List[np.ndarray]
      영상 목록
  output : Path
      결과 저장 경로
  cmap : str, optional
      Colormap, by default 'Dark2'
  names : Optional[List[str]], optional
      저장할 파일 이름 목록. 미지정 시 'Image n' 형식으로 저장.
  """
  tf_gpu_memory_config()

  model = DeepLabModel(graph_path=model_path)

  if names is None:
    names = [f'Image {x+1}' for x in range(len(images))]

  for image, fname in track(zip(images, names), total=len(images)):
    pil_image = PIL.Image.fromarray(image)

    resized_image, seg_map = model.run(pil_image)
    fig, seg_image = vis_segmentation(resized_image, seg_map, show=False, cmap=cmap)

    mask = PIL.Image.fromarray(seg_map)
    mask.save(output.joinpath(fname + '_mask').with_suffix('.png'))

    seg_pil = PIL.Image.fromarray(seg_image)
    seg_pil.save(output.joinpath(fname + '_vis').with_suffix('.png'))

    fig.savefig(output.joinpath(fname + '_fig').with_suffix('.png'))
    plt.close(fig)


class DeepLabModel:
  """
  학습된 DeepLabV3+ 모델의 실행 인터페이스를 제공하는 클래스.

  https://github.com/tensorflow/models/tree/master/research/deeplab 참조
  """

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 641
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, graph_path):
    """Create and loads pretrained deeplab model."""
    self.graph = tf.Graph()

    try:
      graph_def = tf.GraphDef()
      with tf.io.gfile.GFile(graph_path, 'rb') as f:
        graph_def.ParseFromString(f.read())
    except (RuntimeError, ValueError, OSError) as e:
      logger.exception('loading failed')
      raise RuntimeError('failed to load graph') from e

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    self.sess = tf.Session(graph=self.graph)

  def run(self, image: PILImage) -> tuple[PILImage, np.ndarray]:
    """
    Run inference on a single image.

    Parameters
    ----------
    image : PILImage
        A PIL.Image object, raw input image.

    Returns
    -------
    resized_image: PILImage
        RGB image resized from original input image.
    seg_map: np.ndarray
        Segmentation map of `resized_image`.
    """
    width, height = image.size
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, PIL.Image.ANTIALIAS)
    batch_seg_map = self.sess.run(
      self.OUTPUT_TENSOR_NAME,
      feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]},
    )
    seg_map = batch_seg_map[0]

    return resized_image, seg_map

  def predict_and_visualize(
    self, image: np.ndarray, cmap='Dark2', *, resize=True
  ) -> tuple[np.ndarray, np.ndarray, plt.Figure]:
    """
    주어진 영상으로 segmentation 예측 및 시각화

    Parameters
    ----------
    image : np.ndarray
        대상 영상
    cmap : str, optional
        Colormap, by default 'Dark2'
    resize : bool, optional
        `True`이면 원본 영상으로 resize한 결과 반환

    Returns
    -------
    segmentation_map : np.ndarray
        Segmentation 인덱스 맵.
        {0: 'Background', 1: 'Wall', 2: 'Window', 3: 'etc.'}
    segmentation_image : np.ndarray
        지정한 colormap을 통해 시각화한 segmentation map
    fig : plt.Figure
        Raw image, segmentation map, overlay 영상을 담은 matplotlib figure
    """
    pil_image = PIL.Image.fromarray(image)
    resized_image, seg_map = self.run(image=pil_image)

    resized_shape = (resized_image.height, resized_image.width)
    if not (resize or image.shape[:2] == resized_shape):
      vis_image = resized_image
    else:
      vis_image = image
      seg_map = _resize(
        seg_map,
        output_shape=image.shape[:2],
        order=0,
        preserve_range=True,
        anti_aliasing=False,
      )
      seg_map = np.round(seg_map).astype(np.uint8)

    fig, seg_image = vis_segmentation(
      image=vis_image, seg_map=seg_map, show=False, cmap=cmap
    )

    return seg_map, seg_image, fig
