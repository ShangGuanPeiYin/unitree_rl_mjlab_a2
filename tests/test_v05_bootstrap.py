"""V0.5 ``src.tasks`` bootstrap isolation (SDS 8.1 / 9.1 item 18).

This module never runs the legacy ``import_packages`` scanner: the v0.5 entry
point activates ``scripts/a2_v05_bootstrap.py`` before any task import.  The
rejection, namespace-boundedness and traceback evidence is collected in fresh
subprocesses so it cannot depend on test ordering or on other tests' imports.
"""
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

sys.path.insert(0, str(SCRIPTS))
import a2_v05_bootstrap  # noqa: E402

a2_v05_bootstrap.activate(ROOT)


def bootstrap():
    try:
        return importlib.import_module('a2_v05_bootstrap')
    except ModuleNotFoundError:
        raise AssertionError('scripts/a2_v05_bootstrap.py is missing') from None


def probe(source):
    environment = dict(os.environ)
    environment['CUDA_VISIBLE_DEVICES'] = ''
    prefix = str(SCRIPTS)
    if environment.get('PYTHONPATH'):
        prefix += os.pathsep + environment['PYTHONPATH']
    environment['PYTHONPATH'] = prefix
    completed = subprocess.run([sys.executable, '-c', source], cwd=str(ROOT),
                               capture_output=True, text=True, env=environment, timeout=300)
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
        from src.tasks.velocity.v05 import spec  # noqa: F401  (registers nothing)
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
        self.assertTrue(getattr(current, '__a2_v05_bootstrap__'))
        self.assertIs(getattr(sys.modules['src'], 'tasks'), current)
        self.assertEqual(current.__package__, 'src.tasks')
        self.assertFalse(b.is_active(str(ROOT / 'not-the-root')))
        # The legacy initializer still exists on disk, untouched and unused.
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
            # ``src`` is already imported from the real project, so a foreign root
            # is rejected instead of silently mixing two source trees.
            with self.assertRaises(RuntimeError):
                b.activate(foreign)
        self.assertIs(sys.modules['src.tasks'], current)
        self.assertTrue(b.is_active(ROOT))

    def test_ordinary_or_foreign_src_tasks_is_rejected_in_a_clean_process(self):
        ordinary = probe(f'''
import json, sys, types
sys.path.insert(0, {str(SCRIPTS)!r})
import a2_v05_bootstrap as b
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
import a2_v05_bootstrap as b
fake = types.ModuleType('src.tasks')
fake.__a2_v05_bootstrap__ = True
fake.__a2_v05_bootstrap_root__ = '/tmp/foreign-project'
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
import a2_v05_bootstrap as b
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

    def test_an_isolated_namespace_from_another_line_is_adopted(self):
        observed = probe(f'''
import json, sys, types
from importlib.machinery import ModuleSpec
sys.path.insert(0, {str(SCRIPTS)!r})
import src
import a2_v05_bootstrap as b
root = {str(ROOT.resolve())!r}
tasks_path = root + '/src/tasks'
module = types.ModuleType('src.tasks')
module.__package__ = 'src.tasks'
module.__file__ = None
module.__path__ = [tasks_path]
module.__spec__ = ModuleSpec('src.tasks', loader=None, is_package=True)
module.__spec__.submodule_search_locations = [tasks_path]
module.__a2_v04_bootstrap__ = root
module.__a2_v04_bootstrap_root__ = root
sys.modules['src.tasks'] = module
adopted = b.activate(root)
print(json.dumps({{'same_object': adopted is module, 'v05_marker': bool(getattr(adopted, '__a2_v05_bootstrap__', False)), 'v04_marker': bool(getattr(adopted, '__a2_v04_bootstrap__', False)), 'is_active': b.is_active(root)}}))
''')
        self.assertTrue(observed['same_object'], observed)
        self.assertTrue(observed['v05_marker'], observed)
        self.assertTrue(observed['v04_marker'], observed)
        self.assertTrue(observed['is_active'], observed)

    def test_legacy_root_scanner_is_not_executed_and_modules_stay_bounded(self):
        observed = probe(f'''
import json, sys
sys.path.insert(0, {str(SCRIPTS)!r})
import a2_v05_bootstrap as b
pre = sorted(name for name in sys.modules if name == 'src.tasks' or name.startswith('src.tasks.'))
module = b.activate({str(ROOT.resolve())!r})
after_activate = sorted(name for name in sys.modules if name == 'src.tasks' or name.startswith('src.tasks.'))
import src.tasks.velocity.v05.spec
import src.tasks.velocity.v05.rewards
import src.tasks.velocity.v05.model
post = sorted(name for name in sys.modules if name == 'src.tasks' or name.startswith('src.tasks.'))
old = sorted(name for name in post if any(part in name for part in
             ('.v02', '.v02_formal', '.v03', '.v04')))
print(json.dumps({{
    'pre': pre, 'after_activate': after_activate, 'post': post, 'old': old, 'file': module.__file__,
    'loader': None if module.__spec__.loader is None else type(module.__spec__.loader).__name__,
    'config_loaded': 'src.tasks.velocity.config' in sys.modules,
    'mdp_loaded': 'src.tasks.velocity.mdp' in sys.modules,
    'scanner_module_loaded': 'mjlab.utils.lab_api.tasks.importer' in sys.modules,
    'registry': [key for key in vars(sys.modules.get('src.tasks')).keys()],
}}))
''')
        self.assertEqual(observed['pre'], [])
        self.assertEqual(observed['after_activate'], ['src.tasks'])
        self.assertIsNone(observed['file'])
        self.assertIsNone(observed['loader'])
        # Every loaded task module is either the isolated namespace, a shared
        # version-neutral helper package or a v0.5 module; no older version and no
        # legacy registry/scanner module may appear.
        for name in observed['post']:
            self.assertTrue(name in ('src.tasks', 'src.tasks.velocity')
                            or name.startswith('src.tasks.velocity.v05')
                            or name.startswith('src.tasks.velocity.rl'), name)
        for name in ('src.tasks.velocity.v05', 'src.tasks.velocity.v05.spec',
                     'src.tasks.velocity.v05.rewards', 'src.tasks.velocity.v05.model',
                     'src.tasks.velocity.rl.ideal_heightmap_actor_critic'):
            self.assertIn(name, observed['post'])
        self.assertEqual(observed['old'], [])
        self.assertFalse(observed['config_loaded'])
        self.assertFalse(observed['mdp_loaded'])
        self.assertFalse(observed['scanner_module_loaded'])

    def test_nested_traceback_keeps_only_v05_identifiers(self):
        observed = probe(f'''
import json, sys, traceback
sys.path.insert(0, {str(SCRIPTS)!r})
import a2_v05_bootstrap as b
b.activate({str(ROOT.resolve())!r})
from src.tasks.velocity.v05 import spec
from src.tasks.velocity.v05.model import V05ActorCritic
diagnostics = {{}}
try:
    V05ActorCritic.from_eth(spec.INIT_PATH, '0' * 64, 42)
except ValueError as error:
    diagnostics = {{'error': type(error).__name__, 'message': str(error),
                    'trace': traceback.format_exc()}}
formatted = diagnostics.get('trace', '')
import src.tasks.velocity.v05.rewards as rewards
import torch
nested = {{}}
try:
    rewards.speed_reward(torch.zeros(1), torch.zeros(1), torch.zeros(1),
                         torch.ones(1, dtype=torch.bool), enabled=torch.ones(1, dtype=torch.bool))
except ValueError as error:
    nested = {{'error': type(error).__name__, 'message': str(error), 'trace': traceback.format_exc()}}
print(json.dumps({{'first': diagnostics, 'first_trace': formatted, 'nested': nested,
                   'modules': sorted(name for name in sys.modules if name.startswith('src.tasks')),
                   'forbidden_in_trace': [token for token in
                       ('v02', 'v02_formal', 'v03', 'v04', 'V04', 'V03', '0.3.0', '0.4.0', '0.4-r1')
                       if token in (formatted + nested.get('trace', ''))]}}))
''')
        self.assertEqual(observed['first']['error'], 'EthMigrationError')
        self.assertEqual(observed['nested']['error'], 'ValueError')
        self.assertIn('v05/model.py', observed['first_trace'])
        self.assertIn('v05/rewards.py', observed['nested']['trace'])
        self.assertEqual(observed['forbidden_in_trace'], [])
        for name in observed['modules']:
            self.assertTrue(name in ('src.tasks', 'src.tasks.velocity')
                            or name.startswith('src.tasks.velocity.v05')
                            or name.startswith('src.tasks.velocity.rl'), name)
        for name in ('src.tasks.velocity.v05.spec', 'src.tasks.velocity.v05.model',
                     'src.tasks.velocity.v05.rewards'):
            self.assertIn(name, observed['modules'])

    def test_bootstrap_is_standard_library_only(self):
        b = bootstrap()
        source = (ROOT / 'scripts' / 'a2_v05_bootstrap.py').read_text()
        import ast
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
        for name in ('activate', 'is_active', 'active_root'):
            self.assertTrue(callable(getattr(b, name)), name)


if __name__ == '__main__':
    unittest.main()
