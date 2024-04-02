import os
import re
from collections import defaultdict
from pathlib import Path

import click
import pandas as pd

p_dist = r'([0-9.]+)m'


def is_target(directory: Path):
  if directory.name in ['Raw', 'VIS']:
    return False

  if directory.joinpath('Raw').exists():
    return True

  if list(directory.glob('*.jpg')):
    return True

  return False


def get_date(directory: Path):
  return directory.parent.name[:10]


def get_camera(directory: Path):
  name = directory.name.upper()
  if not (name.startswith('FLIR') or name.startswith('TESTO')):
    return None

  if name.startswith('TESTO'):
    return 'testo 882'

  return name[: name.find(' ', 5)]


def get_images_count(directory: Path, camera: str | None = None):
  raw = directory.joinpath('Raw')
  d = raw if raw.exists() else directory

  if camera is None or camera.upper().startswith('FLIR'):
    ext = '.jpg'
  else:
    ext = '.xlsx'

  return len(list(d.glob(f'*{ext}')))


def get_distance(directory: Path):
  m = re.search(p_dist, directory.name)
  return None if m is None else m.group(1)


@click.command()
@click.argument('directory')
@click.argument('output', required=False)
def cli(directory, output):
  directory = Path(directory)
  directory.stat()

  dd = defaultdict(list)
  for root, dirs, files in os.walk(directory):
    root = Path(root)
    if directory == root:
      continue

    if not is_target(root):
      continue

    dd['path'].append(root.as_posix())
    dd['case'].append(root.name)
    dd['date'].append(get_date(root))

    camera = get_camera(root)
    dd['camera'].append(camera)
    dd['images_count'].append(get_images_count(root, camera=camera))

    dd['distance'].append(get_distance(root))

  df = pd.DataFrame(dd)
  print(df)

  if output:
    df.to_csv(output, encoding='utf-8-sig')


if __name__ == '__main__':
  # pylint: disable=no-value-for-parameter
  cli()
