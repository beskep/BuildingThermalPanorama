"""Document 생성"""

import re
from pathlib import Path
from warnings import catch_warnings

import pdoc
from loguru import logger

from pano import utils

OUTPUT_DIR = utils.DIR.ROOT.joinpath('docs')


def module_path(m: pdoc.Module, ext: str) -> Path:
  parts = re.sub(r'\.html$', ext, m.url()).split('/')
  parts.remove('pano')
  path = OUTPUT_DIR.joinpath(*parts)

  return path


def recursive_htmls(mod: pdoc.Module):
  yield mod.name, mod.html(), module_path(mod, ext='.html')

  for submod in mod.submodules():
    yield from recursive_htmls(submod)


def main():
  module_names = ['pano']
  context = pdoc.Context()

  modules = []
  for name in module_names:
    try:
      with catch_warnings(record=True) as warnings:
        module = pdoc.Module(name, context=context)

        for w in warnings:
          logger.warning(w)
    except ImportError as e:
      logger.exception(e)
    else:
      modules.append(module)

  pdoc.link_inheritance(context)

  for mod in modules:
    for module_name, html, path in recursive_htmls(mod):
      if not path.parent.exists():
        path.parent.mkdir(parents=True)

      with path.open('w', encoding='utf-8') as f:
        f.write(html)


if __name__ == '__main__':
  main()
