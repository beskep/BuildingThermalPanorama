"""영상 파일의 Exif 정보 추출"""

from subprocess import check_output
from subprocess import DEVNULL
from typing import List, Optional, Union

import yaml

from pano import utils

EXIFTOOL_PATH = utils.DIR.RESOURCE.joinpath('exiftool.exe')
if not EXIFTOOL_PATH.exists():
  raise FileNotFoundError(EXIFTOOL_PATH)


def exiftool(*args) -> bytes:
  """
  ExifTool 프로그램을 통해 영상 파일의 메타 데이터 (Exif) 추출
  """
  args = (EXIFTOOL_PATH.as_posix(),) + args

  return check_output(args, stderr=DEVNULL)


def get_exif(files: Union[str, List[str]],
             tags: Optional[List[str]] = None) -> List[dict]:
  """
  촬영 파일로부터 Exif tag 추출

  Parameters
  ----------
  files : Union[str, List[str]]
      대상 파일 경로, 혹은 경로의 목록
  tags : Optional[List[str]], optional
      추출할 tag. `None`인 경우 모든 tag 추출, by default None

  Returns
  -------
  List[dict]
  """
  if isinstance(files, str):
    files = [files]

  if tags is None:
    tags = []
  else:
    tags = [(x if x.startswith('-') else '-' + x) for x in tags]

  exifs_byte = exiftool('-j', *tags, *files)
  exifs = yaml.safe_load(exifs_byte.decode())

  return exifs


def get_exif_binary(image_path: str, tag: str) -> bytes:
  """
  Exif 정보로부터 binary 데이터 추출.
  FLIR 촬영 파일의 열화상, 실화상 정보 추출에 이용.

  Parameters
  ----------
  image_path : str
      파일 경로
  tag : str
      추출 대상 태그

  Returns
  -------
  bytes
  """
  return exiftool(tag, '-b', image_path)
