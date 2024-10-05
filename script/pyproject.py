import json
from itertools import chain
from pathlib import Path

ROOT = Path(__file__).parents[1]
assert ROOT.name == 'BuildingThermalPanorama'

if __name__ == '__main__':
  ds = {'dist', 'build'}
  exts = ['.py', '.qml']

  fc = chain.from_iterable((f for f in ROOT.rglob(f'*{e}')) for e in exts)
  fl = sorted(str(f) for f in fc if all(p not in ds for p in f.parts))

  pyprj = ROOT / 'panorama.pyproject'
  with pyprj.open('w', encoding='utf-8') as f:
    json.dump({'files': fl}, f, indent=4)
