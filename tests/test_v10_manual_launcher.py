"""Manual dynamic-N entry wiring; no simulation or physical samples."""
import importlib.util
from pathlib import Path
import sys
import types
import pytest
ROOT=Path(__file__).resolve().parents[1]

def launcher(monkeypatch):
    calls=[]
    fake=types.ModuleType('src.tasks.velocity.v10.dynamic_execution')
    def run(**kwargs):calls.append(kwargs);return {'passed':True,'executed':False}
    fake.run_dynamic_execution=run
    monkeypatch.setitem(sys.modules,fake.__name__,fake)
    spec=importlib.util.spec_from_file_location('manual_launcher',ROOT/'scripts/run_a2_v10_formal.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module,calls

def test_check_is_explicit_no_training(monkeypatch):
    module,calls=launcher(monkeypatch)
    assert module.main(['--mode','check','--num-envs','2048'])==0
    assert calls[0]['check_only'] is True and calls[0]['num_envs']==2048

def test_all_passes_configurable_env_count_and_fresh_root(monkeypatch):
    module,calls=launcher(monkeypatch)
    assert module.main(['--mode','all','--num-envs','2048'])==0
    assert calls[0]['run_dir']==ROOT/'logs/a2_v10/server_fresh'
    assert calls[0]['profile_path'].name=='execution_profile_N2048.json'
    assert calls[0]['teachers_only'] is False
    assert calls[0]['fresh'] is True

def test_too_small_env_count_rejected(monkeypatch):
    module,calls=launcher(monkeypatch)
    with pytest.raises(SystemExit):module.main(['--num-envs','0'])
    assert not calls
