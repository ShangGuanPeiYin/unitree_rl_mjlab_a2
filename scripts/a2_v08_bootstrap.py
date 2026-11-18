"""Standard-library-only isolation of ``src.tasks`` for the A2 stairs v0.8 line.

The legacy ``src/tasks/__init__.py`` scans every task package with
``import_packages``.  v0.8 runs must bind only their explicit dependency
closure and must not execute that scanner or register any older task.  This
module installs a namespace-like ``src.tasks`` module in ``sys.modules`` before
the first v0.8 import.  It deliberately does not import ``src.tasks`` itself.

Usage (must be the first task import in the process)::

    from a2_v08_bootstrap import activate
    activate('/path/to/project')
    from src.tasks.velocity.v08 import spec

The root is recorded on the module so a second call with the same resolved root
is idempotent.  A normally imported or foreign ``src.tasks`` is never replaced.
An isolated namespace previously installed by another A2 line for the *same*
root (for example ``scripts/a2_v05_r2_bootstrap.py`` or
``scripts/a2_v04_bootstrap.py`` in a shared test process) is adopted and
additionally marked for v0.8 instead of being rejected: it is already the
exact isolated module this line requires, has no ``__file__``, no loader and the
same ``src/tasks`` search path.

The module is a byte-level derivative of ``scripts/a2_v05_r2_bootstrap.py``; only
the v0.8 marker names, the usage docstring and the adoption wording differ
(SDS 8.2).  The r1 bootstrap is never modified.
"""
from __future__ import annotations

import importlib
from importlib.machinery import ModuleSpec
from pathlib import Path
import sys
import types

_MARKER = '__a2_v08_bootstrap__'
_ROOT = '__a2_v08_bootstrap_root__'
_ANY_MARKER_SUFFIX = '_bootstrap__'


def _default_root():
    return Path(__file__).resolve().parents[1]


def _isolated_namespace(module, root):
    """Return True for a namespace module this project already isolated."""
    tasks_path = str((Path(root) / 'src' / 'tasks').resolve())
    if getattr(module, '__file__', None) is not None:
        return False
    if list(getattr(module, '__path__', [])) != [tasks_path]:
        return False
    spec = getattr(module, '__spec__', None)
    if spec is None or spec.loader is not None:
        return False
    markers = [name for name in vars(module)
               if name.startswith('__a2_') and name.endswith(_ANY_MARKER_SUFFIX)]
    if not markers:
        return False
    return any(getattr(module, name.replace('_bootstrap__', '_bootstrap_root__'), None) == str(Path(root).resolve()) and getattr(module, name, False) is True for name in markers)


def activate(project_root=None):
    """Install the isolated ``src.tasks`` package and return it.

    Raises ``RuntimeError`` if ``src`` resolves elsewhere, if ``src.tasks`` was
    already loaded normally, or if the supplied root is not a project root with
    ``src/tasks``.
    """
    root = Path(project_root).resolve() if project_root is not None else _default_root()
    root = root.resolve()
    if not (root / 'src' / 'tasks').is_dir():
        raise RuntimeError(f'Not an A2 project root (missing src/tasks): {root}')
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    src = sys.modules.get('src')
    if src is None:
        src = importlib.import_module('src')
    src_file = Path(getattr(src, '__file__', '')).resolve()
    if not src_file.is_relative_to(root):
        raise RuntimeError(
            f'src was imported from {src_file}, outside requested root {root}; use a clean process')

    tasks_path = str((root / 'src' / 'tasks').resolve())
    existing = sys.modules.get('src.tasks')
    if existing is not None:
        if (getattr(existing, _MARKER, False) and getattr(existing, _ROOT, None) == str(root)):
            return existing
        if _isolated_namespace(existing, root):
            setattr(existing, _MARKER, True)
            setattr(existing, _ROOT, str(root))
            setattr(src, 'tasks', existing)
            return existing
        raise RuntimeError(
            'src.tasks is already loaded normally or from another source; '
            'refusing to replace it. Use a clean process for the v0.8 entry point.')

    module = types.ModuleType('src.tasks')
    module.__package__ = 'src.tasks'
    module.__file__ = None
    module.__path__ = [tasks_path]
    module.__spec__ = ModuleSpec('src.tasks', loader=None, is_package=True)
    module.__spec__.submodule_search_locations = [tasks_path]
    setattr(module, _MARKER, True)
    setattr(module, _ROOT, str(root))
    module.__doc__ = (
        'Isolated namespace replacement for src.tasks installed by '
        'scripts/a2_v08_bootstrap.py; the legacy import_packages root initializer '
        'is not executed.')
    sys.modules['src.tasks'] = module
    setattr(src, 'tasks', module)
    return module


def is_active(project_root=None):
    tasks = sys.modules.get('src.tasks')
    if tasks is None or not getattr(tasks, _MARKER, False):
        return False
    if project_root is None:
        return True
    return getattr(tasks, _ROOT, None) == str(Path(project_root).resolve())


def active_root():
    tasks = sys.modules.get('src.tasks')
    if tasks is None or not getattr(tasks, _MARKER, False):
        return None
    return getattr(tasks, _ROOT, None)
