"""Qv07-002 real file protocol, immutable errors and latched operator requests."""
import importlib
import json
from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import a2_v07_bootstrap
a2_v07_bootstrap.activate(ROOT)

def module():return importlib.import_module('src.tasks.velocity.v07.operator_control')

def test_missing_and_continue_do_not_request_stop(tmp_path):
    m=module();path=tmp_path/'control.json';control=m.ControlProtocol(path,tmp_path/'evidence')
    assert control.poll() is None
    m.write_control(path,dict(action='continue',reason='stop',resume_authorized=True))
    assert control.poll() is None

@pytest.mark.parametrize('action,reason',[('stop','user_requested_stop'),('pause','operator_pause')])
def test_first_valid_request_latches_and_ignores_later_continue(tmp_path,action,reason):
    m=module();path=tmp_path/'control.json';m.write_control(path,dict(action=action,reason='continue',resume_authorized=True))
    original_raw=path.read_bytes()
    c=m.ControlProtocol(path,tmp_path/'evidence');request=c.poll()
    assert request['reason']==reason and request['resume_authorized'] is False
    assert request['pause_requested'] is (action=='pause')
    m.write_control(path,dict(action='continue'))
    assert c.poll()==request
    assert Path(request['evidence_path']).read_bytes()==original_raw

@pytest.mark.parametrize('raw',[b'{bad json',b'[]',b'{}',b'{"action":"resume"}',b'{"action":true}'])
def test_invalid_control_retains_exact_bytes_and_raises_named_engineering_error(tmp_path,raw):
    m=module();path=tmp_path/'control.json';path.write_bytes(raw)
    with pytest.raises(m.ControlProtocolError) as caught:m.ControlProtocol(path,tmp_path/'evidence').poll()
    assert path.read_bytes()==raw
    assert Path(caught.value.audit['evidence_path']).read_bytes()==raw
    assert caught.value.audit['error_type']=='ControlProtocolError'

def test_read_error_is_not_default_continue_and_original_is_not_modified(tmp_path,monkeypatch):
    m=module();path=tmp_path/'control.json';path.write_bytes(b'{"action":"stop"}')
    original=Path.read_bytes
    def broken(self):
        if self==path:raise OSError('injected read error')
        return original(self)
    with monkeypatch.context() as patch:
        patch.setattr(Path,'read_bytes',broken)
        with pytest.raises(m.ControlProtocolError):m.ControlProtocol(path,tmp_path/'evidence').poll()
    assert path.read_bytes()==b'{"action":"stop"}'

def test_supervisor_keeps_bad_user_control_bytes_on_worker_engineering_error(tmp_path,monkeypatch):
    from src.tasks.velocity.v07 import orchestration,spec
    m=module();raw=b'{operator bad json'
    def worker(contract,out):
        path=out/'control.json';path.write_bytes(raw)
        m.ControlProtocol(path,out/'control_evidence').poll()
    monkeypatch.setattr(orchestration,'acquire_single_worker',lambda *a,**k:None)
    monkeypatch.setattr(orchestration,'release_single_worker',lambda *a,**k:None)
    args=orchestration.build_parser().parse_args(['--execute'])
    snapshot=dict(whole_device_mib=0,process_mib=None,process_mib_method='not_initialized',host_available_mib=10000,disk_free_gib=100)
    supervisor=orchestration.V07Orchestrator(args,worker_factory=worker,root=ROOT,
        preflight_dir=tmp_path/'preflight',resource_probe=lambda **kwargs:snapshot,gpu_registry=tmp_path/'lease.json')
    supervisor.contract=dict(spec.recipe_config(),execute=True,mode='resume',output_dir=str(tmp_path/'run'))
    with pytest.raises(m.ControlProtocolError):supervisor.execute()
    assert (tmp_path/'run'/'control.json').read_bytes()==raw
    assert json.loads((tmp_path/'run'/'error.json').read_text())['error_type']=='ControlProtocolError'

def test_valid_request_remains_latched_even_when_its_evidence_write_first_fails(tmp_path,monkeypatch):
    m=module();path=tmp_path/'control.json';m.write_control(path,dict(action='pause'))
    control=m.ControlProtocol(path,tmp_path/'evidence');original=control._snapshot
    def fail(raw):raise OSError('injected evidence persistence error')
    with monkeypatch.context() as patch:
        patch.setattr(control,'_snapshot',fail)
        with pytest.raises(m.ControlProtocolError):control.poll()
    m.write_control(path,dict(action='continue'))
    request=control.poll()
    assert request['action']=='pause' and request['reason']=='operator_pause'
