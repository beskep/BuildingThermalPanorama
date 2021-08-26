from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.text import Text
from rich.tree import Tree


def walk_directory(directory: Path, tree: Tree) -> None:
  """Recursively build a Tree with directory contents."""
  paths = sorted(Path(directory).iterdir(),
                 key=lambda path: (path.is_file(), path.name.lower()))

  for path in paths:
    if path.name.startswith('.'):
      continue
    if path.suffix == '.npy':
      continue

    if path.is_dir():
      if 'raw' in path.name.lower():
        branch = tree.add(f'📂 {escape(path.name)}')
        walk_directory(path, branch)
      else:
        branch = tree.add(f'📁 {escape(path.name)}')
    else:
      icon = '🖼️' if path.suffix in ('.png', '.jpg') else '📄'
      # icon = '📷' if path.suffix in ('.png', '.jpg') else '📄'
      text = Text(icon + ' ' + path.name)
      tree.add(text)


def tree_string(directory, width=80):
  directory = Path(directory)
  tree = Tree(label=f'📂 {directory.as_posix()}')
  walk_directory(directory=directory, tree=tree)

  console = Console(width=width)
  with console.capture() as capture:
    console.print(tree)

  return capture.get()
