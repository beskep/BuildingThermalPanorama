"""cx_Freeze를 통해 실행 파일 생성을 위한 코드"""

import sys
import sysconfig
from datetime import datetime
from pathlib import Path

from cx_Freeze import Executable, setup
from pytz import timezone

from pano.utils import DIR, play_sound

sys.setrecursionlimit(5000)


def build():
  resources = [
    str(x.relative_to(DIR.ROOT))
    for x in DIR.RESOURCE.iterdir()
    if not x.name.lower().startswith('test')
  ]
  include_files = [(x, x) for x in resources]
  include_files.append(('qt', 'qt'))

  includes = [
    'click',
    'cv2',
    'loguru',
    'matplotlib',
    'omegaconf',
    'onnxruntime',
    'pdb',
    'PIL',
    'pydoc',
    'rich.console',
    'rich.logging',
    'rich.progress',
    'rich',
    'scipy.integrate.lsoda',
    'scipy.integrate.vode',
    'scipy.integrate',
    'scipy.io',
    'scipy.optimize.nonlin',
    'scipy.optimize.zeros',
    'scipy.optimize',
    'scipy.spatial.transform._rotation_groups',
    'seaborn.cm',
    'SimpleITK',
    'skimage.color.colorlabel',
    'skimage.draw._polygon2mask',
    'skimage.draw.draw',
    'skimage.feature._canny',
    'skimage.feature._orb_descriptor_positions',
    'skimage.feature.orb',
    'skimage.filters._unsharp_mask',
    'skimage.filters.edges',
    'skimage.io._plugins.pil_plugin',
    'skimage.measure.block',
    'skimage.measure.fit',
    'skimage.transform._warps',
    'skimage.transform.hough_transform',
    'skimage',
    'webp',
  ]
  excludes = ['locket', 'mypy', 'PySide2', 'tkinter', 'resource']
  zip_include_packages = []

  bins = ['ITK']
  lib_bin = Path(sys.base_prefix).joinpath('Library/bin')
  for b in bins:
    include_files.extend(
      [(f.as_posix(), f'lib/{f.name}') for f in lib_bin.glob(f'*{b}*')]
    )

  sys_info = (sysconfig.get_platform(), sysconfig.get_python_version())
  version = datetime.now(tz=timezone('Asia/Seoul')).date().isoformat().replace('-', '.')

  options = {
    'build_exe': {
      'build_exe': (
        f'build/BuildingThermalPanorama-{version}-exe.{sys_info[0]}-{sys_info[1]}'
      ),
      'include_files': include_files,
      'includes': includes,
      'zip_include_packages': zip_include_packages,
      'excludes': excludes,
      'optimize': 1,
      'silent_level': 1,
    }
  }

  executables = [
    Executable(script=r'pano\interface\cli.py', target_name='CLI'),
    Executable(script=r'pano\interface\gui.py', target_name='GUI'),
  ]

  setup(
    name='app',
    version='0.1',
    description='ThermalPanorama',
    options=options,
    executables=executables,
    packages=[],
  )


if __name__ == '__main__':
  build()
  play_sound()
