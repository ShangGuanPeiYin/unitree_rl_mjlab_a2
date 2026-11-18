from test_v08_evaluation import failure_raw
from pathlib import Path
import sys
import copy
import random
import numpy as np
import torch
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import a2_v08_bootstrap
a2_v08_bootstrap.activate(ROOT)

def test_freeze_covers_old_dependencies_and_fails_on_missing(tmp_path):
    from src.tasks.velocity.v08 import recovery
    frozen=recovery.build_freeze(ROOT)
    assert 'src/tasks/velocity/v07/spec.py' in frozen['files']
    assert 'src/tasks/velocity/rl/ideal_heightmap_actor_critic.py' in frozen['files']
    assert 'scripts/a2_v08_bootstrap.py' in frozen['files']
    recovery.verify_freeze(frozen,ROOT)
    broken=copy.deepcopy(frozen);broken['files']['src/nonexistent.py']='0'*64
    with pytest.raises(ValueError):recovery.verify_freeze(broken,ROOT)

def test_rng_restoration_requires_compatible_device_and_env():
    from src.tasks.velocity.v08 import recovery
    from src.tasks.velocity.v08.continuation_v08 import load_designated_source,tree_digest
    source=load_designated_source();before=recovery.capture_rng()
    class Env:
        def load_rng_state(self,x): self.env=x
        def load_sampler_rng_state(self,x): self.sampler=x
    env=Env()
    # CPU engineering explicitly records unconsumed CUDA state rather than claiming exact restoration.
    receipt=recovery.restore_rng(source['rng'],env=env,device='cpu',allow_cpu_cuda_provenance=True)
    assert receipt['cuda']=='source_preserved_cpu_engineering_only'
    assert tree_digest(env.env)==tree_digest(source['rng']['env'])
    assert tree_digest(env.sampler)==tree_digest(source['rng']['sampler'])
    with pytest.raises(ValueError):recovery.restore_rng(source['rng'],env=object(),device='cpu')
    recovery.restore_rng(before,device='cpu')

def test_atomic_save_reload_checks_sha(tmp_path):
    from src.tasks.velocity.v08 import recovery
    data={'value':torch.tensor([1.,2.])}
    path=tmp_path/'snapshot.pt';sha=recovery.atomic_save(path,data)
    assert torch.equal(recovery.load_verified(path,sha)['value'],data['value'])
    with pytest.raises(ValueError):recovery.load_verified(path,'0'*64)

def test_evaluation_reference_reloads_raw_sidecar_and_refuses_missing_or_tamper(tmp_path):
    from src.tasks.velocity.v08 import recovery,spec
    from src.tasks.velocity.v08.evaluation import make_cases,evaluate_cases
    from src.tasks.velocity.v08.evaluation_records import pack_result,write_sidecar
    from src.tasks.velocity.v08.continuation_v08 import SOURCE_SHA256
    table=make_cases('A',kind='gate')
    def failure(c):return dict(failure_raw(c),reason='physical_failure',
        collector_batch_id=recovery.digest([c['case_id']]),collector_batch_control_samples=1)
    result=evaluate_cases(None,table,failure)
    identity=dict(update=6500,phase='A',kind='gate',case_table_sha256=table.sha256,
        source_sha256=SOURCE_SHA256,recipe_sha256=recovery.digest(recovery.jsonable(spec.contract())))
    payload=pack_result(result,identity);path=tmp_path/'raw.json';binding=write_sidecar(path,payload)
    entry=dict(binding=binding,identity=identity,summary=result.summary,control_samples=len(table.cases))
    loaded=recovery.load_evaluation_reference(entry)
    assert loaded.summary==result.summary
    path.write_text('{}')
    with pytest.raises(ValueError):recovery.load_evaluation_reference(entry)
    path.unlink()
    with pytest.raises((ValueError,FileNotFoundError)):recovery.load_evaluation_reference(entry)

def test_unfinished_evaluation_intent_blocks_recovery_before_any_fresh_reset(tmp_path):
    from src.tasks.velocity.v08 import recovery
    directory=tmp_path/'evaluation_transactions';directory.mkdir()
    recovery.write_json(directory/'gate_A_006600.intent.json',dict(schema='v08-evaluation-intent-v1',
        identity={'update':6600},sidecar_path=str(tmp_path/'missing.json'),batches=[['case0']]))
    with pytest.raises(ValueError,match='evaluation'):
        recovery.verify_evaluation_journal(tmp_path)

def test_orphan_batch_progress_cannot_be_treated_as_no_evaluation(tmp_path):
    from src.tasks.velocity.v08 import recovery
    progress=tmp_path/'evaluation_transactions'/'h00_A_006600.batches';progress.mkdir(parents=True)
    recovery.write_json(progress/'batch_0000.commit.json',{'control_samples':24})
    with pytest.raises(ValueError,match='evaluation'):
        recovery.verify_evaluation_journal(tmp_path)
