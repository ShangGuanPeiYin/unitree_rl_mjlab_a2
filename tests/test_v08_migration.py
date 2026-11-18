"""Native6400 migration must retain the complete learned state, not old course."""
from pathlib import Path
import sys
import copy
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import a2_v08_bootstrap
a2_v08_bootstrap.activate(ROOT)

def test_v08_native_migration_module_exists():
    assert (ROOT/'src/tasks/velocity/v08/continuation_v08.py').is_file()

def source():
    from src.tasks.velocity.v08 import continuation_v08 as m
    return m,m.load_designated_source()

def test_source_exact_learning_and_preservation():
    m,s=source(); new,diff=m.migrate_stop_6400(s,source_sha256=m.SOURCE_SHA256)
    assert new['learning']==s['learning']
    for key in ('model','optimizer','normalizer','learning','rng'):
        assert m.tree_digest(new[key])==m.tree_digest(s[key])
    assert len(new['optimizer']['state'])==45
    assert {float(v['step']) for v in new['optimizer']['state'].values()}=={128000.}
    assert new['learning']['learning_rate']==1.5000000000000005e-05
    assert new['curriculum']['state']['phase']=='A' and new['curriculum']['state']['last_update']==6400
    assert new['ledger']['pending_boundary'] is None
    assert new['source']['parent']['terminal']==s['ledger']['terminal']
    assert diff['first_new_update']==6401

def test_wrong_source_and_tampered_adam_refused():
    m,s=source()
    with pytest.raises(ValueError):m.migrate_stop_6400(s,source_sha256='0'*64)
    s=copy.deepcopy(s);next(iter(s['optimizer']['state'].values()))['exp_avg'].add_(1.)
    with pytest.raises(ValueError):m.migrate_stop_6400(s,source_sha256=m.SOURCE_SHA256)
    with pytest.raises(ValueError):m.load_designated_source('/tmp/noncanonical_6400.pt')
