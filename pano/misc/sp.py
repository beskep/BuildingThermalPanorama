# ruff: noqa: S603

import subprocess as sp
from collections.abc import Iterable

import yaml

from pano.utils import DIR


class PATH:
  BIN = DIR.RESOURCE / 'bin'
  EXIFTOOL = BIN / 'exiftool.exe'
  WKHTMLTOPDF = BIN / 'wkhtmltopdf.exe'
  WKHTMLTOIMAGE = BIN / 'wkhtmltoimage.exe'


def exiftool(*args) -> bytes:
  """ExifTool 프로그램을 통해 영상 파일의 메타 데이터 (Exif) 추출."""
  PATH.EXIFTOOL.stat()
  args = (PATH.EXIFTOOL, *args)
  return sp.check_output(list(map(str, args)), stderr=sp.DEVNULL)


def get_exif(files: str | list[str], tags: Iterable[str] | None = None) -> list[dict]:
  """
  촬영 파일로부터 Exif tag 추출

  Parameters
  ----------
  files : str | list[str]
      대상 파일 경로, 혹은 경로의 목록
  tags : Iterable[str] | None, optional
      추출할 tag. `None`인 경우 모든 tag 추출.

  Returns
  -------
  list[dict]
  """
  if isinstance(files, str):
    files = [files]

  tags = [(x if x.startswith('-') else '-' + x) for x in tags or []]

  exifs_byte = exiftool('-j', *tags, *files)
  return yaml.safe_load(exifs_byte.decode())


def get_exif_binary(path: str, tag: str) -> bytes:
  """Exif 정보로부터 binary 데이터 추출.

  FLIR 촬영 파일의 열화상, 실화상 정보 추출에 이용.

  Parameters
  ----------
  path : str
      파일 경로
  tag : str
      추출 대상 태그

  Returns
  -------
  bytes
  """
  return exiftool(tag, '-b', path)


def wkhtmltopdf(src, dst, *, capture=False):
  PATH.WKHTMLTOPDF.stat()
  args = (PATH.WKHTMLTOPDF, '--enable-local-file-access', src, dst)
  sp.check_output(list(map(str, args)), stderr=None if capture else sp.DEVNULL)
