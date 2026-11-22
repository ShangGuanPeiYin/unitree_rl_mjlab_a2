"""V0.4 ``src.tasks`` bootstrap isolation (SDS 0.4 §8.4).

This module intentionally runs the real legacy scanner never: the suite entry
point activates ``scripts/a2_v04_bootstrap.py`` before discovery.  Rejection and
scanner-bypass evidence is collected in fresh subprocesses so it cannot depend on
test ordering or on other tests' imports.
"""
import ast
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'


def bootstrap():
    try:
        return importlib.import_module('a2_v04_bootstrap')
    except ModuleNotFoundError:
        raise AssertionError('scripts/a2_v04_bootstrap.py is missing') from None


def probe(source):
    environment = dict(os.environ)
    environment['CUDA_VISIBLE_DEVICES'] = ''
    prefix = str(SCRIPTS)
    if environment.get('PYTHONPATH'):
        prefix += os.pathsep + environment['PYTHONPATH']
    environment['PYTHONPATH'] = prefix
    completed = subprocess.run([sys.executable, '-c', source], cwd=str(ROOT),
                               capture_output=True, text=True, env=environment, timeout=180)
    if completed.returncode != 0:
        raise AssertionError(
            f'bootstrap subprocess failed (exit {completed.returncode})\n'
            f'stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}')
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise AssertionError('bootstrap subprocess produced no output')
    return json.loads(lines[-1])


class BootstrapTests(unittest.TestCase):
    def test_activate_is_idempotent_and_installs_isolated_namespace(self):
        b = bootstrap()
        root = str(ROOT.resolve())
        current = sys.modules.get('src.tasks')
        self.assertIsNotNone(current, 'the suite must be launched after bootstrap.activate()')
        self.assertIs(b.activate(root), current)
        self.assertIs(b.activate(), current)
        self.assertTrue(b.is_active())
        self.assertTrue(b.is_active(root))
        self.assertEqual(b.active_root(), root)
        self.assertIsNone(current.__file__)
        self.assertIsNone(current.__spec__.loader)
        self.assertEqual(list(current.__path__), [str((ROOT / 'src' / 'tasks').resolve())])
        self.assertEqual(list(current.__spec__.submodule_search_locations),
                         [str((ROOT / 'src' / 'tasks').resolve())])
        self.assertTrue(getattr(current, '__a2_v04_bootstrap__'))
        self.assertIs(getattr(sys.modules['src'], 'tasks'), current)
        self.assertEqual(current.__package__, 'src.tasks')
        self.assertFalse(b.is_active(str(ROOT / 'not-the-root')))
        # The legacy initializer is still present on disk, untouched; it is
        # simply not what occupies sys.modules.
        self.assertTrue((ROOT / 'src' / 'tasks' / '__init__.py').is_file())

    def test_wrong_root_is_rejected_without_replacing_the_active_module(self):
        b = bootstrap()
        current = sys.modules['src.tasks']
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / 'not-a-project'
            with self.assertRaises(RuntimeError):
                b.activate(missing)
            foreign = Path(temporary)
            (foreign / 'src' / 'tasks').mkdir(parents=True)
            # ``src`` is already imported from the real project, so a foreign
            # root is rejected instead of silently mixing two source trees.
            with self.assertRaises(RuntimeError):
                b.activate(foreign)
        self.assertIs(sys.modules['src.tasks'], current)
        self.assertTrue(b.is_active(ROOT))

    def test_ordinary_or_foreign_src_tasks_is_rejected_in_a_clean_process(self):
        ordinary = probe(f'''
import json, sys, types
sys.path.insert(0, {str(SCRIPTS)!r})
import a2_v04_bootstrap as b
fake = types.ModuleType('src.tasks')
sys.modules['src.tasks'] = fake
result = {{'rejected': False, 'message': ''}}
try:
    b.activate({str(ROOT.resolve())!r})
except RuntimeError as exc:
    result = {{'rejected': True, 'message': str(exc)}}
print(json.dumps(result))
''')
        self.assertTrue(ordinary['rejected'], ordinary)
        self.assertIn('already loaded', ordinary['message'])

        foreign_marker = probe(f'''
import json, sys, types
sys.path.insert(0, {str(SCRIPTS)!r})
import a2_v04_bootstrap as b
fake = types.ModuleType('src.tasks')
fake.__a2_v04_bootstrap__ = True
fake.__a2_v04_bootstrap_root__ = '/tmp/foreign-project'
sys.modules['src.tasks'] = fake
result = {{'rejected': False, 'message': ''}}
try:
    b.activate({str(ROOT.resolve())!r})
except RuntimeError as exc:
    result = {{'rejected': True, 'message': str(exc)}}
print(json.dumps(result))
''')
        self.assertTrue(foreign_marker['rejected'], foreign_marker)

        with tempfile.TemporaryDirectory() as temporary:
            foreign = Path(temporary)
            (foreign / 'src' / 'tasks').mkdir(parents=True)
            (foreign / 'src' / '__init__.py').write_text('')
            (foreign / 'src' / 'tasks' / '__init__.py').write_text('')
            imported = probe(f'''
import json, sys
sys.path.insert(0, {str(SCRIPTS)!r})
import a2_v04_bootstrap as b
sys.path.insert(0, {str(foreign)!r})
import src.tasks
result = {{'rejected': False, 'message': '', 'foreign_file': getattr(src.tasks, '__file__', None)}}
try:
    b.activate({str(ROOT.resolve())!r})
except RuntimeError as exc:
    result = {{'rejected': True, 'message': str(exc), 'foreign_file': src.tasks.__file__}}
print(json.dumps(result))
''')
            self.assertTrue(imported['rejected'], imported)
            self.assertIn('outside requested root', imported['message'])
            self.assertIn(str(foreign), imported['foreign_file'])

    def test_legacy_root_scanner_is_not_executed_and_modules_stay_bounded(self):
        observed = probe(f'''
import json, sys
sys.path.insert(0, {str(SCRIPTS)!r})
import a2_v04_bootstrap as b
pre = sorted(name for name in sys.modules if name == 'src.tasks' or name.startswith('src.tasks.'))
module = b.activate({str(ROOT.resolve())!r})
after_activate = sorted(name for name in sys.modules if name == 'src.tasks' or name.startswith('src.tasks.'))
import src.tasks.velocity.v04.rewards
post = sorted(name for name in sys.modules if name == 'src.tasks' or name.startswith('src.tasks.'))
print(json.dumps({{
    'pre': pre, 'after_activate': after_activate, 'post': post, 'file': module.__file__,
    'loader': None if module.__spec__.loader is None else type(module.__spec__.loader).__name__,
    'config_loaded': 'src.tasks.velocity.config' in sys.modules,
    'mdp_loaded': 'src.tasks.velocity.mdp' in sys.modules,
    'scanner_module_loaded': 'mjlab.utils.lab_api.tasks.importer' in sys.modules,
}}))
''')
        self.assertEqual(observed['pre'], [])
        self.assertEqual(observed['after_activate'], ['src.tasks'])
        self.assertIsNone(observed['file'])
        self.assertIsNone(observed['loader'])
        self.assertEqual(set(observed['post']), {
            'src.tasks', 'src.tasks.velocity', 'src.tasks.velocity.v04',
            'src.tasks.velocity.v04.rewards', 'src.tasks.velocity.v04.spec'})
        self.assertFalse(observed['config_loaded'])
        self.assertFalse(observed['mdp_loaded'])
        # The legacy scanner itself is never imported through the task root.
        self.assertFalse(observed['scanner_module_loaded'])

    def test_bootstrap_is_standard_library_only(self):
        b = bootstrap()
        source = (ROOT / 'scripts' / 'a2_v04_bootstrap.py').read_text()
        tree = ast.parse(source)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split('.')[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported.add(node.module.split('.')[0])
        self.assertEqual(imported - set(sys.stdlib_module_names), set())
        self.assertNotIn('src', imported)
        self.assertNotIn('torch', imported)
        self.assertNotIn('mjlab', imported)
        self.assertTrue(callable(b.activate))
        self.assertTrue(callable(b.is_active))
        self.assertTrue(callable(b.active_root))


if __name__ == '__main__':
    unittest.main()
