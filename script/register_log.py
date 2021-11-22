from collections import defaultdict
from pathlib import Path
import re

import click
import numpy as np
import pandas as pd

p_quote = re.compile(r'"(.*)"')
p_camera = re.compile(r'Camera: (.*)$')
p_param = re.compile(r'Final param: \[(.*)\]$')


def read_row(line: str):
  if '_register:260' in line:
    return 'image', p_quote.findall(line)[0]

  if 'register:393' in line:
    return 'param', np.fromstring(p_param.findall(line)[0], count=4, sep=' ')

  if 'run_cases:33' in line:
    return 'case', p_quote.findall(line)[0]

  if '__init__:63' in line:
    return 'camera', p_camera.findall(line)[0]

  return None, None


@click.command()
@click.option('--output', '-o')
@click.argument('path')
def read_log(path, output):
  path = Path(path)

  row = {}
  data = defaultdict(list)
  with path.open('r', encoding='utf-8') as f:
    while True:
      line = f.readline()
      if not line:
        break

      key, value = read_row(line)
      if key is None:
        continue

      if key == 'case':
        row = {key: value}
      else:
        row[key] = value

      if key == 'param':
        for k, v in row.items():
          data[k].append(v)

  params = np.vstack(data.pop('param'))
  df_params = pd.DataFrame(params, columns=['s', 'theta', 'tx', 'ty'])
  df_key = pd.DataFrame(data)
  df = pd.concat([df_key, df_params], axis=1)

  print(df)

  if output:
    df.to_csv(output, encoding='utf-8-sig', index=False)


if __name__ == '__main__':
  # pylint: disable=no-value-for-parameter
  read_log()
