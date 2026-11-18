from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
def test_v08_adopts_existing_isolated_old_namespace_without_old_global_mutation():
    code="""import sys;sys.path.insert(0,'scripts')
import a2_v07_bootstrap as old
old.activate()
import a2_v08_bootstrap as new
new.activate()
assert new.is_active()
from src.tasks.velocity.v08 import spec
from src.tasks.velocity.v07 import spec as legacy
assert spec.RECIPE_VERSION.startswith('0.8') and legacy.RECIPE_VERSION.startswith('0.7')
"""
    result=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
