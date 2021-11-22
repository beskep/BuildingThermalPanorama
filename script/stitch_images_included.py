from collections import defaultdict
from pathlib import Path
from typing import DefaultDict

import click
from loguru import logger
import pandas as pd
import yaml


def _images_count(images):
  if images is None:
    return 0

  return len(images)


def _read_pano_meta(path: Path):
  with path.open('r', encoding='utf-8') as f:
    meta = yaml.safe_load(f)

  including = _images_count(meta['panorama']['including'])
  not_including = _images_count(meta['panorama']['not_including'])

  return including / (including + not_including)


def _read_case(path: Path):
  meta_path = path.joinpath('PanoramaIR_meta.yaml')
  if not meta_path.exists():
    return 0

  return _read_pano_meta(meta_path)


@click.command()
@click.option('--output', '-o')
@click.argument('path')
def main(path, output):
  path = Path(path)
  pano_dirs = path.rglob('03 Panorama')

  dd = DefaultDict(list)
  for pano_dir in pano_dirs:
    case = pano_dir.parent.name
    r = _read_case(pano_dir)

    dd['case'].append(case)
    dd['r'].append(r)

  df = pd.DataFrame(dd)
  print(df)

  if output:
    df.to_csv(output, index=False, encoding='utf-8-sig')


if __name__ == '__main__':
  # pylint: disable=no-value-for-parameter
  main()
