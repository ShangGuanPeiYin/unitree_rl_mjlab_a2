"""v07 must isolate task imports without consulting or activating the C1 flag."""
from pathlib import Path
import pytest
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]

def run(source):
    return subprocess.run([sys.executable,'-c',source],cwd=ROOT,text=True,capture_output=True)

def test_native_namespace_ignores_legacy_activation_and_does_not_import_legacy_tasks():
    result=run('''
import os,sys
sys.path.insert(0,'scripts')
os.environ['A2_V06_CONTINUATION']='invalid-legacy-flag-must-not-be-read'
import a2_v07_bootstrap
first=a2_v07_bootstrap.activate('.')
assert a2_v07_bootstrap.activate('.') is first
from src.tasks.velocity.v07 import spec
assert spec.ARCHITECTURE=='v06-route-state-v1'
assert spec.terrain_spec(17).lesson_id=='D25'
assert 'torch' not in sys.modules
assert not any(k.startswith('src.tasks.velocity.v06') for k in sys.modules)
assert 'src.tasks.velocity.v07.c1_contract' not in sys.modules
assert first.__file__ is None and first.__spec__.loader is None
''')
    assert result.returncode==0,result.stderr

def test_bootstrap_refuses_to_replace_a_normally_loaded_task_package():
    result=run('''
import sys,types
from pathlib import Path
from importlib.machinery import ModuleSpec
sys.path.insert(0,'scripts')
import a2_v07_bootstrap
import src
normal=types.ModuleType('src.tasks');normal.__file__=str(Path('src/tasks/__init__.py').resolve())
normal.__path__=[str(Path('src/tasks').resolve())]
normal.__spec__=ModuleSpec('src.tasks',loader=object(),is_package=True)
sys.modules['src.tasks']=normal
try:a2_v07_bootstrap.activate('.')
except RuntimeError:pass
else:raise AssertionError('legacy loaded package was silently replaced')
assert sys.modules['src.tasks'] is normal
''')
    assert result.returncode==0,result.stderr

@pytest.mark.parametrize('name',['run','test','evaluate'])
def test_native_entry_scripts_exist_and_do_not_dispatch_v06(name):
    path=ROOT/'scripts'/f'{name}_a2_ideal_stairs_v07.py'
    assert path.is_file()
    content=path.read_text()
    assert 'a2_v07_bootstrap' in content
    assert 'src.tasks.velocity.v06' not in content and 'a2_v06_bootstrap' not in content
