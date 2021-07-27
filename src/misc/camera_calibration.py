"""카메라 렌즈의 왜곡 보정"""

from pathlib import Path
from typing import List, Union

import cv2 as cv
import numpy as np
import yaml

from misc.tools import ImageIO


def _detect_chessboard(file: Path, save_dir: Path, img_size, pattern_size,
                       criteria):
  """체스보드 패턴 검출"""
  image = cv.imread(file.as_posix())
  gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

  if img_size is None:
    img_size = gray.shape[::-1]
  else:
    assert img_size == gray.shape[::-1]

  ret, corners = cv.findChessboardCorners(gray,
                                          patternSize=pattern_size,
                                          corners=None)
  if ret:
    corners2 = cv.cornerSubPix(gray,
                               corners=corners,
                               winSize=(11, 11),
                               zeroZone=(-1, -1),
                               criteria=criteria)

    cv.drawChessboardCorners(image,
                             patternSize=pattern_size,
                             corners=corners2,
                             patternWasFound=ret)

    path = save_dir.joinpath(file.with_suffix('.jpg').name)
    ImageIO.save(path=path, array=image)

  return img_size, ret, corners


def _calibrate_camera(object_points, image_points, image_size, save_dir: Path):
  """카메라 패러미터 산정, 저장"""
  ret, mtx, dist_coeff, rvecs, tvecs = cv.calibrateCamera(
      objectPoints=object_points,
      imagePoints=image_points,
      imageSize=image_size,
      cameraMatrix=None,
      distCoeffs=None)

  res_dict = {
      'image_size': list(image_size),
      'ret': ret,
      'matrix': mtx.tolist(),
      'dist_coeff': dist_coeff.tolist(),
      'rvecs': np.array(rvecs).tolist(),
      'tvecs': np.array(tvecs).tolist(),
  }
  with open(save_dir.joinpath('parameters.yaml'), 'w') as f:
    yaml.dump(res_dict, stream=f)

  np.savez(file=save_dir.joinpath('parameters'),
           image_size=image_size,
           matrix=mtx,
           dist_coeff=dist_coeff,
           rvecs=np.array(rvecs),
           tvecs=np.array(tvecs))


def compute_camera_matrix(files: List[Union[str, Path]],
                          save_dir: Union[str, Path],
                          pattern_size=(3, 3)):
  """
  주어진 영상으로부터 Chessboard 패턴을 검출하고 카메라 보정 패러미터 산정.

  Parameters
  ----------
  files : List[Union[str, Path]]
      영상 파일 목록
  save_dir : Union[str, Path]
      결과 저장 경로
  pattern_size : tuple, optional
      검출한 체스보드 패턴 개수, by default (3, 3)
  """
  save_dir = Path(save_dir).resolve()

  criteria = (cv.TermCriteria_EPS + cv.TermCriteria_MAX_ITER, 30, 0.001)
  objp = np.zeros((pattern_size[0] * pattern_size[1], 3), dtype=np.float32)
  objp[:, :2] = np.mgrid[0:pattern_size[0],
                         0:pattern_size[1]].T.reshape([-1, 2])

  obj_points = []
  img_points = []
  img_size = None
  for file in files:
    file = Path(file)
    img_size, ret, corners = _detect_chessboard(file=file,
                                                save_dir=save_dir,
                                                img_size=img_size,
                                                pattern_size=pattern_size,
                                                criteria=criteria)
    if ret:
      obj_points.append(objp)
      img_points.append(corners)

  if not obj_points:
    raise ValueError('Chessboard 추출 실패')

  _calibrate_camera(object_points=obj_points,
                    image_points=img_points,
                    image_size=img_size,
                    save_dir=save_dir)


class CameraCalibration:
  """
  추출한 Camera Calibration 패러미터를 읽고 주어진 영상을 calibrate하는 클래스
  """

  def __init__(self, params_path: Union[str, Path]):
    """
    Parameters
    ----------
    params_path : Union[str, Path]
        `compute_camera_matrix`를 통해 산정한 카메라 패러미터 파일.
        `.npz` 또는 `.yaml` 파일 입력 가능.
    """
    params_path = Path(params_path)
    if not params_path.exists():
      raise FileNotFoundError(params_path)

    ext = params_path.suffix
    if ext == '.npz':
      npz_file = np.load(params_path.as_posix())

      self._img_size = tuple(npz_file['image_size'])
      self._matrix = npz_file['matrix']
      self._dist_coeff = npz_file['dist_coeff']
    elif ext in ['.yaml', '.yml']:
      with open(params_path, 'r') as f:
        params = yaml.safe_load(f)

      self._img_size = tuple(params['image_size'])
      self._matrix = np.array(params['matrix'])
      self._dist_coeff = params['dist_coeff']
    else:
      raise ValueError('지원하지 않는 파일 형식입니다.')

  @property
  def image_size(self):
    return self._img_size

  @property
  def matrix(self):
    return self._matrix

  @property
  def dist_coeff(self):
    return self._dist_coeff

  def calibrate(self, image: np.ndarray) -> np.ndarray:
    """
    새 영상에 카메라 보정 적용

    Parameters
    ----------
    image : np.ndarray
        대상 영상. OpenCV 지원 dtype이어야 함. `cv2.undistort` 참조

    Returns
    -------
    np.ndarray
        보정된 영상.
    """
    new_matrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix=self._matrix,
                                                   distCoeffs=self._dist_coeff,
                                                   imageSize=self._img_size,
                                                   alpha=1,
                                                   newImgSize=self._img_size)

    calibrated = cv.undistort(image,
                              cameraMatrix=self._matrix,
                              distCoeffs=self._dist_coeff,
                              dst=None,
                              newCameraMatrix=new_matrix)

    return calibrated

  def mask(self) -> np.ndarray:
    """
    카메라 보정 결과 영역의 uint8 마스크

    Returns
    -------
    np.ndarray
    """
    blank = np.full(shape=self._img_size[::-1], fill_value=255, dtype=np.uint8)
    mask = self.calibrate(blank)

    return mask
