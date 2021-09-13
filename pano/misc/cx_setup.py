"""cx_Freeze를 통해 실행 파일 생성을 위한 코드"""

from pathlib import Path
import sys

from cx_Freeze import Executable
from cx_Freeze import setup

from pano.utils import DIR


def build():
  resources = [
      x.relative_to(DIR.ROOT).as_posix()
      for x in DIR.RESOURCE.iterdir()
      if not x.name.lower().startswith('test')
  ]
  include_files = [(x, x) for x in resources]

  includes = [
      'click',
      'cv2',
      'loguru',
      'matplotlib',
      'omegaconf',
      'pdb',
      'PIL',
      'pydoc',
      'rich.console',
      'rich.logging',
      'rich.progress',
      'rich',
      'scipy.spatial.transform._rotation_groups',
      'seaborn.cm',
      'SimpleITK',
      'skimage.feature._orb_descriptor_positions',
      'skimage.io._plugins.pil_plugin',
      'skimage',
      'tensorflow',
  ]
  excludes = ['tkinter', 'locket', 'mypy']
  zip_include_packages = []

  bins = ['ITK']
  lib_bin = Path(sys.base_prefix).joinpath('Library/bin')
  for b in bins:
    files = list(lib_bin.glob(f'*{b}*'))
    for file in files:
      include_files.append((file.as_posix(), f'lib/{file.name}'))

  options = {
      'build_exe': {
          'include_files': include_files,
          'includes': includes,
          'zip_include_packages': zip_include_packages,
          'excludes': excludes,
          'optimize': 1
      }
  }

  executables = [
      Executable(script=r'pano\interface\cli.py', target_name='CLI'),
      Executable(script=r'pano\interface\gui.py', target_name='GUI'),
  ]

  setup(name='app',
        version='0.1',
        description='ThermalPanorama',
        options=options,
        executables=executables)


if __name__ == '__main__':
  build()
