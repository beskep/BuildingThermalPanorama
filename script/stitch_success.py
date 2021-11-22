from pathlib import Path
from typing import Optional

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from script.plot import set_style

set_style()


def show_or_save(fig, output: Optional[Path] = None, fname=None):
  if output:
    if fname is None:
      fname = 'plot'

    path = output.joinpath(fname).with_suffix('.png')
    fig.savefig(path, dpi=200)
  else:
    plt.show()

  plt.close(fig)


def _plot_success_rate(df: pd.DataFrame, output):
  df_p = df[['Photographer',
             'success']].groupby('Photographer').agg('mean').reset_index()

  print(df_p)

  fig, ax = plt.subplots(1, 1)
  sns.barplot(data=df_p, x='Photographer', y='success', ax=ax)
  ax.set_ylim(0, 1)
  show_or_save(fig, output=output, fname='success_p')

  df_pc = df[['Photographer', 'success',
              'Camera']].groupby(['Photographer',
                                  'Camera']).agg('mean').reset_index()
  print(df_pc)
  fig, ax = plt.subplots(1, 1)
  sns.barplot(data=df_pc, x='Photographer', y='success', hue='Camera', ax=ax)
  plt.legend(loc=2)
  fig.tight_layout()
  show_or_save(fig, output=output, fname='success_pc')


def _plot_scatter(df: pd.DataFrame, outout):
  dfc = df.copy()
  dfc['success'] = ['Success' if x else 'Fail' for x in dfc['success']]

  fig, ax = plt.subplots(1, 1)
  sns.scatterplot(data=dfc,
                  x='Distance(m)',
                  y='ImagesCount',
                  hue='success',
                  hue_order=['Success', 'Fail'],
                  style='Camera',
                  palette='Dark2',
                  alpha=0.75,
                  s=75,
                  ax=ax)

  show_or_save(fig, output=outout, fname='scatter')


def _plot_grid(df: pd.DataFrame, output):
  dfc = df.copy().reset_index(drop=True)
  dfc['success'] = [int(x > 0) for x in dfc['success']]

  # df['DistOverCount'] = df['Distance(m)'] / df['ImagesCount']
  dfc['CountOverDist'] = dfc['ImagesCount'] / dfc['Distance(m)']
  dfc['CountOverFOV'] = dfc['ImagesCount'] / dfc['FOV']
  dfc['DistOverFOV'] = dfc['Distance(m)'] / dfc['FOV']

  variables = [
      'Distance(m)',
      'ImagesCount',
      'FOV',
      'CountOverDist',
      'DistOverFOV',
  ]
  for var in variables:
    if np.all(dfc[var][0] == dfc[var]):
      variables.remove(var)

  grid = sns.PairGrid(
      data=dfc,
      hue='success',
      hue_order=[1, 0],
      despine=False,
      palette='Dark2',
      vars=variables,
      diag_sharey=False,
  )
  grid.map_diag(sns.histplot, stat='density', kde=True, common_norm=False)
  grid.map_lower(sns.scatterplot, alpha=0.5)

  try:
    grid.map_upper(sns.kdeplot)
  except Exception:
    pass

  grid.add_legend()

  show_or_save(grid.fig, output=output, fname='grid')


def _plot_heatmap(df: pd.DataFrame, output):
  dfc = df.copy()

  dedges = np.linspace(0, 36, num=13, endpoint=True)
  cedges = np.linspace(0, 60, num=16, endpoint=True).astype(int)

  didx = np.digitize(dfc['Distance(m)'], dedges)
  cidx = np.digitize(dfc['ImagesCount'], cedges)

  dfs = pd.DataFrame({'didx': didx, 'cidx': cidx, 'success': dfc['success']})
  dfs['distance'] = [
      f'{dedges[x-1]:02.0f}–{dedges[x]:02.0f}' for x in dfs['didx']
  ]
  dfs['images_count'] = [
      f'{cedges[x-1]:02d}–{cedges[x]:02d}' for x in dfs['cidx']
  ]
  dfs = dfs.groupby(['distance', 'images_count']).agg('mean').reset_index()

  fig, ax = plt.subplots(1, 1, figsize=(8, 6))

  sns.heatmap(
      data=dfs.pivot('images_count', 'distance', 'success'),
      linewidths=0.2,
      annot=True,
      fmt='.2f',
      cmap=sns.color_palette('crest_r'),
      ax=ax,
  )
  ax.invert_yaxis()
  ax.set_xlabel('Distance (m)')
  ax.set_ylabel('Images Count')

  fig.tight_layout()

  show_or_save(fig=fig, output=output, fname='heatmap')


@click.command()
@click.option('--output', '-o')
@click.option('--ratio', is_flag=True)
@click.option('--photographer', '-p', multiple=True)
@click.option('--camera', '-c', multiple=True)
@click.argument('path')
def main(path, photographer, camera, output, ratio):
  df: pd.DataFrame = pd.read_csv(path)
  df['Camera'] = [
      'FLIR One' if x.lower().startswith('flir one') else x
      for x in df['Camera']
  ]
  df = df.dropna(axis=0, subset=['Distance(m)'])

  if ratio:
    df.loc[df['success'] == 0, 'images_ratio'] = 0
    df['success'] = df['images_ratio']

  print(df)
  print(df.dtypes)

  # pylint: disable=no-member, unsubscriptable-object
  if photographer:
    df = df.loc[[x in photographer for x in df['Photographer']], :]

  if camera:
    df = df.loc[[x in camera for x in df['Camera']], :]

  if output:
    output = Path(output)

  _plot_heatmap(df, output=output)
  _plot_success_rate(df, output=output)
  _plot_scatter(df, outout=output)
  _plot_grid(df, output=output)


if __name__ == '__main__':
  # pylint: disable=no-value-for-parameter
  main()
