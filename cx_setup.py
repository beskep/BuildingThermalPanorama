from pathlib import Path

import cx_Freeze
from cx_Freeze import Executable, setup

include_files = [
    'src',
    'config.yaml',
    ('data/DeepLabV3', 'data/DeepLabV3'),
    ('data/MainBldgBackLoc1PanTiltTripod',
     'data/MainBldgBackLoc1PanTiltTripod'),
]
includes = [
    'loguru',
    'rich',
    'rich.console',
    'rich.progress',
    'rich.logging',
    'pydoc',
    'pdb',
    'PIL',
    'scipy.spatial.transform._rotation_groups',
    'matplotlib',
    'skimage',
    'skimage.feature._orb_descriptor_positions',
    'skimage.io._plugins.pil_plugin',
    'cv2',
    'tensorflow',
    'PyQt5',
    'SimpleITK',
]
excludes = ['tkinter', 'locket', 'PySide2']
zip_include_packages = []
bins = ['ITK']

try:
  import mkl
  bins.append('mkl_intel_thread')
  includes.append('mkl')
except ImportError:
  pass

lib_bin = Path(cx_Freeze.__path__[0]).parents[2].joinpath('Library/bin')
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
    }
}

executables = [
    Executable('pano_cli.py'),
]

setup(name='app',
      version='0.1',
      description='description',
      options=options,
      executables=executables)
