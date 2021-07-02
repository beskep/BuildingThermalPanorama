import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parents[1].resolve()
SRC_DIR = ROOT_DIR.joinpath('src')

SCRIPT_DIR = ROOT_DIR.joinpath('scripts')
DATA_DIR = ROOT_DIR.joinpath('data')

_SRC_DIR = SRC_DIR.as_posix()
if _SRC_DIR not in sys.path:
  sys.path.insert(0, _SRC_DIR)

# pylint: disable=wrong-import-position
import distortion
import flir
import misc
import registration
import stitch
