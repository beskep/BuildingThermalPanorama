"""영상 파일의 Exif 정보 추출"""

from subprocess import DEVNULL, check_output

import utils

import yaml

EXIFTOOL_PATH = utils.DIR.SRC.joinpath('exiftool.exe')
if not EXIFTOOL_PATH.exists():
  raise FileNotFoundError(EXIFTOOL_PATH)


def run_exif_tool(*args):
  """
  ExifTool 프로그램을 통해 영상 파일의 메타 데이터 (Exif) 추출
  """
  args = (EXIFTOOL_PATH.as_posix(),) + args
  res = check_output(args, stderr=DEVNULL)

  return res


def get_exif_tags(image_path: str, *args) -> dict:
  """
  지정한 Exif tag 추출

  Parameters
  ----------
  image_path : str
      영상 경로
  *args
      추출할 태그 목록

  Returns
  -------
  dict
  """
  tag_byte = run_exif_tool(image_path, '-j', *args)
  tag = yaml.safe_load(tag_byte.decode())

  return tag


def get_exif_binary(image_path: str, tag):
  """
  Exif 정보로부터 binary 데이터 추출. FLIR 촬영 파일의 열화상, 실화상
  정보 추출에 이용.

  Parameters
  ----------
  image_path : str
      파일 경로
  tag : str
      추출 대상 태그
  """
  res = run_exif_tool(tag, '-b', image_path)

  return res
