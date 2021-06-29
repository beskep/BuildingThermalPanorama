import re
from pathlib import Path

import utils

import pdoc
from loguru import logger

OUTPUT_DIR = utils.ROOT_DIR.joinpath('docs')


def module_path(m: pdoc.Module, ext: str) -> Path:
  parts = re.sub(r'\.html$', ext, m.url()).split('/')
  parts.remove('src')
  path = OUTPUT_DIR.joinpath(*parts)

  return path


def recursive_htmls(mod: pdoc.Module):
  yield mod.name, mod.html(), module_path(mod, ext='.html')

  for submod in mod.submodules():
    yield from recursive_htmls(submod)


if __name__ == '__main__':
  # module_names = [x.name for x in utils.SRC_DIR.iterdir() if x.is_dir()]
  module_names = ['src']
  context = pdoc.Context()

  modules = []
  for name in module_names:
    try:
      module = pdoc.Module(name, context=context)
    except ImportError as e:
      logger.error(e)
    else:
      modules.append(module)

  pdoc.link_inheritance(context)

  for mod in modules:
    for module_name, html, path in recursive_htmls(mod):
      if not path.parent.exists():
        path.parent.mkdir(parents=True)

      with path.open('w', encoding='utf-8') as f:
        f.write(html)
