"""Consumed FIFO rows must be durable before their EMA application."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import a2_v07_bootstrap
a2_v07_bootstrap.activate(ROOT)
from src.tasks.velocity.v07 import spec
from test_v07_curriculum import outcomes

def module():return importlib.import_module('src.tasks.velocity.v07.window_evidence')
def payload():
    rows=outcomes(56)
    return dict(schema='v07-frontier-window-v1',identity=dict(update=4818,phase='D',substage='D25',generation=73,window_index=1),
        records=rows,records_sha256=__import__('hashlib').sha256(spec.canonical_json(rows).encode()).hexdigest(),
        previous_ema=dict(s=None,f=None,q=None,batches=0),diagnostic_only=True)

def test_install_roundtrip_idempotent_and_conflict_refusal(tmp_path):
    m=module();p=payload();path=tmp_path/'first.json';before=deepcopy(p)
    binding=m.write_window(path,p)
    assert p==before and m.verify_window(binding,p)
    assert m.write_window(path,p)==binding
    changed=deepcopy(p);changed['identity']['update']+=1
    with pytest.raises(ValueError):m.write_window(path,changed)
    path.write_text('{}')
    with pytest.raises(ValueError):m.verify_window(binding,p)

@pytest.mark.parametrize('change',['count','hash','generation','role','tile','order','nan','flags'])
def test_invalid_consumption_evidence_refused(tmp_path,change):
    m=module();p=payload()
    if change=='count':p['records'].pop()
    if change=='hash':p['records_sha256']='0'*64
    if change=='generation':p['records'][0]['generation']=72
    if change=='role':p['records'][0]['role']='near'
    if change=='tile':p['records'][0]['tile_id']=18
    if change=='order':p['records'][0]['step']=9999
    if change=='nan':p['records'][0]['tracking_score']=float('nan')
    if change=='flags':p['records'][0].update(success=False,failure=False,timeout=False)
    if change not in ('hash','nan'):p['records_sha256']=__import__('hashlib').sha256(spec.canonical_json(p['records']).encode()).hexdigest()
    with pytest.raises(ValueError):m.write_window(tmp_path/'bad.json',p)
    assert not (tmp_path/'bad.json').exists()

def test_install_failure_keeps_complete_partial_and_retry_syncs_same_rows(tmp_path,monkeypatch):
    m=module();p=payload();path=tmp_path/'first.json';original=m.os.link
    def fail(*args,**kwargs):raise OSError('injected link failure')
    monkeypatch.setattr(m.os,'link',fail)
    with pytest.raises(OSError):m.write_window(path,p)
    assert json.loads(path.with_name('first.json.partial').read_text())==p
    monkeypatch.setattr(m.os,'link',original)
    assert m.verify_window(m.write_window(path,p),p)
