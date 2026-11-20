"""Production CLI preflight identity failures, CPU-only and no simulator."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from a2_v07_bootstrap import activate
activate(ROOT)
from src.tasks.velocity.v09 import spec
from src.tasks.velocity.v09.cases import build_s1_manifest, freeze_manifest
from src.tasks.velocity.v09.checkpoint import save_checkpoint
from src.tasks.velocity.v09.model import V09ActorCritic

CURRENT = 'c' * 64
OLD = 'a' * 64


def script(name):
    definition = importlib.util.spec_from_file_location('q010_' + name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(definition)
    definition.loader.exec_module(module)
    return module


@pytest.fixture
def cli_inputs(tmp_path, monkeypatch):
    # Only volatile source enumeration is replaced; file bytes, schema and CLI
    # parsing remain real. No model/environment/GPU operation is mocked.
    monkeypatch.setattr(spec, 'validate_sources', lambda: {})
    monkeypatch.setattr(spec, 'code_snapshot', lambda: {'code_sha256': CURRENT, 'files': {}})
    model = V09ActorCritic()
    optimizer = torch.optim.Adam(model.optimizer_parameters(), lr=3e-5)
    metadata = dict(stage='S1', course='S1.1', value_scale=1., manifest_sha256='d'*64,
        ledger_run_id='test-run', learning_samples=0, code_sha256=OLD,
        gate_state=dict(next_suite='gate_A', attempts=0, final_suites_used=0,
                        recoveries=0, next_gate_samples=100000000, passes=[]),
        role_state=dict(num_envs=512, episode_counters={}, assignments=[]),
        random_state={'python': (), 'numpy': ()})
    checkpoint = tmp_path/'old_student.pt'
    digest = save_checkpoint(checkpoint, model, optimizer, metadata)
    for suite in ('gate_A','S1_DIAG'):
        freeze_manifest(tmp_path/'cases'/f'{suite}.json', build_s1_manifest(suite))
    calibration = tmp_path/'calibration.json'
    calibration.write_text(json.dumps(dict(sds_sha256=spec.SDS_SHA256,
        source_sha256=spec.SOURCE_SHA256, code_sha256=CURRENT, value_scale=1.)))
    return tmp_path, checkpoint, digest, calibration


def test_student_evaluation_dry_run_rejects_old_checkpoint_before_physics(cli_inputs, monkeypatch):
    run, checkpoint, digest, _ = cli_inputs
    args = ['evaluate_a2_v09_s1', '--run-dir',str(run),'--run-id','test-run',
        '--suite','gate_A','--policy','student','--checkpoint',str(checkpoint),
        '--checkpoint-sha256',digest,'--code-sha256',CURRENT,
        '--obligation-id','candidate_gate_01','--segment-id','eval',
        '--batch-size','20','--device','cpu']
    monkeypatch.setattr(sys,'argv',args)
    with pytest.raises(ValueError, match='code'):
        script('evaluate_a2_v09_s1').main()
    assert not (run/'evidence').exists() and not (run/'budget.jsonl').exists()


def test_student_video_dry_run_rejects_old_checkpoint_before_physics(cli_inputs, monkeypatch):
    run, checkpoint, digest, _ = cli_inputs
    monkeypatch.setattr(sys,'argv',['video_a2_v09_s1','--run-dir',str(run),
        '--run-id','test-run','--segment-id','video','--code-sha256',CURRENT,
        '--policy','student','--checkpoint',str(checkpoint),'--checkpoint-sha256',digest,
        '--obligation-id','video_init','--output-dir',str(run/'videos'),'--device','cpu'])
    with pytest.raises(ValueError, match='code'):
        script('video_a2_v09_s1').main()
    assert not (run/'videos').exists() and not (run/'budget.jsonl').exists()


def test_diagnostic_comparison_rejects_old_student_checkpoint(cli_inputs, monkeypatch):
    run, checkpoint, digest, calibration = cli_inputs
    monkeypatch.setattr(sys,'argv',['compare_a2_v09_diagnostics','--run-dir',str(run),'--run-id','test-run',
        '--before-dir',str(run/'before'),'--after-dir',str(run/'after'),
        '--initial-checkpoint',str(checkpoint),'--initial-checkpoint-sha256',digest,
        '--adapted-checkpoint',str(checkpoint),'--adapted-checkpoint-sha256',digest,
        '--calibration',str(calibration),'--code-sha256',CURRENT,'--device','cpu',
        '--output',str(run/'comparison.json')])
    with pytest.raises(ValueError, match='code'):
        script('compare_a2_v09_diagnostics').main()
    assert not (run/'comparison.json').exists()


def test_training_resume_dry_run_rejects_old_checkpoint_before_admissions(cli_inputs, monkeypatch):
    run, checkpoint, digest, calibration = cli_inputs
    monkeypatch.setattr(sys,'argv',['train_a2_v09_s1','--run-dir',str(run),'--run-id','test-run',
        '--segment-id','train','--resource-admission',str(run/'resource.json'),
        '--baseline-admission',str(run/'baseline.json'),'--calibration',str(calibration),
        '--code-sha256',CURRENT,'--num-envs','512','--device','cpu',
        '--resume-checkpoint',str(checkpoint),'--resume-checkpoint-sha256',digest])
    with pytest.raises(ValueError, match='code'):
        script('train_a2_v09_s1').main()
    assert not (run/'budget.jsonl').exists()


def test_current_checkpoint_loader_accepts_exact_current_code_and_sha(cli_inputs):
    import hashlib
    from src.tasks.velocity.v09.s1_runner import load_current_checkpoint
    _, path, _, _ = cli_inputs
    payload = torch.load(path, map_location='cpu', weights_only=False)
    payload['metadata']['code_sha256'] = CURRENT
    torch.save(payload, path)
    raw = path.read_bytes()
    expected = hashlib.sha256(raw).hexdigest()
    accepted = load_current_checkpoint(path, code_sha256=CURRENT, checkpoint_sha256=expected)
    assert accepted['metadata']['code_sha256'] == CURRENT
    assert path.read_bytes() == raw
    with pytest.raises(ValueError, match='SHA mismatch'):
        load_current_checkpoint(path, code_sha256=CURRENT, checkpoint_sha256='f'*64)


@pytest.mark.parametrize('entry', ['evaluate_a2_v09_s1','video_a2_v09_s1'])
def test_teacher_dry_run_cannot_relabel_student_checkpoint_as_unique_6400(cli_inputs, monkeypatch, entry):
    run, checkpoint, digest, _ = cli_inputs
    common = [entry,'--run-dir',str(run),'--run-id','test-run','--segment-id','teacher',
        '--code-sha256',CURRENT,'--policy','teacher','--checkpoint',str(checkpoint),
        '--checkpoint-sha256',digest,'--device','cpu']
    if entry.startswith('evaluate'):
        common += ['--suite','gate_A','--obligation-id','teacher_gate_A','--batch-size','20']
    else:
        common += ['--obligation-id','video_6400','--output-dir',str(run/'videos')]
    monkeypatch.setattr(sys,'argv',common)
    with pytest.raises(ValueError, match='6400|source'):
        script(entry).main()
    assert not (run/'videos').exists() and not (run/'evidence').exists()


def test_direct_resume_rejects_old_code_before_teacher_device_allocation(cli_inputs, monkeypatch):
    from types import SimpleNamespace
    from src.tasks.velocity.v09.s1_runner import RealS1Trainer, ScheduleJournal
    from src.tasks.velocity.v09.model import Frozen6400Teacher
    run, checkpoint, _, _ = cli_inputs
    class Ledger:
        def snapshot(self):
            return {'stages':{'S1':{'allocations':{},'spent':{'learning':0},'progress':{}}},'operations':{}}
    def forbidden_allocation(*args,**kwargs):
        pytest.fail('invalid resume checkpoint reached model/device allocation')
    monkeypatch.setattr(Frozen6400Teacher,'from_6400_checkpoint',forbidden_allocation)
    with pytest.raises(ValueError,match='code'):
        RealS1Trainer(run_dir=run,run_id='test-run',segment_id='resume',device='cuda:0',
            code_sha256=CURRENT,admissions={'num_envs':512,'value_scale':1.},
            ledger=Ledger(),journal=ScheduleJournal(run/'schedule.jsonl'),
            evidence_hooks=SimpleNamespace(),resume=checkpoint)


@pytest.mark.parametrize('entry,suite', [('evaluate_a2_v09_s1','gate_A'),('video_a2_v09_s1','S1_DIAG')])
def test_current_student_cli_reads_only_approved_frozen_inputs_in_dry_run(tmp_path,monkeypatch,entry,suite):
    from test_v09_consumer_identity import make_consumer_identity
    f = make_consumer_identity(tmp_path)
    monkeypatch.setattr(spec,'validate_sources',lambda: {})
    monkeypatch.setattr(spec,'code_snapshot',lambda: {'code_sha256':CURRENT,'files':{}})
    manifest = build_s1_manifest(suite)
    path = f.run_dir/'cases'/f'{suite}.json'
    freeze_manifest(path,manifest)
    f.bind_file(path,kind='frozen_manifest',suite=suite,
        manifest_sha256=manifest['manifest_sha256'],category='frozen_inputs')
    f.seal()
    model = V09ActorCritic()
    optimizer = torch.optim.Adam(model.optimizer_parameters(),lr=3e-5)
    metadata = dict(stage='S1',course='S1.1',value_scale=2.,manifest_sha256='a'*64,
        ledger_run_id=f.run_id,learning_samples=0,code_sha256=CURRENT,
        gate_state=dict(next_suite='gate_A',attempts=0,final_suites_used=0,recoveries=0,next_gate_samples=100000000,passes=[]),
        role_state=dict(num_envs=512,episode_counters={},assignments=[]),random_state={'python':(),'numpy':()})
    checkpoint = f.run_dir/'new-student.pt'
    digest = save_checkpoint(checkpoint,model,optimizer,metadata)
    args = [entry,'--run-dir',str(f.run_dir),'--run-id',f.run_id,'--segment-id','dry',
        '--policy','student','--checkpoint',str(checkpoint),'--checkpoint-sha256',digest,
        '--code-sha256',CURRENT,'--device','cpu','--evidence-identity',str(f.approval_path),
        '--evidence-identity-sha256',f.approval_sha256]
    if entry.startswith('evaluate'):
        args += ['--suite',suite,'--obligation-id','candidate_gate_01','--batch-size','20']
    else:
        args += ['--obligation-id','video_init','--output-dir',str(f.run_dir/'videos')]
    before = f.ledger_path.read_bytes()
    monkeypatch.setattr(sys,'argv',args)
    assert script(entry).main() in (None,0)
    assert f.ledger_path.read_bytes() == before and not (f.run_dir/'videos').exists()


@pytest.mark.parametrize('binding',['missing','wrong_sha','wrong_path','unexpected_context'])
def test_formal_resume_rejects_current_checkpoint_with_wrong_identity_before_allocation(cli_inputs,monkeypatch,binding):
    import hashlib
    from src.tasks.velocity.v09.evidence_identity import EvidenceIdentityContext
    from src.tasks.velocity.v09.s1_runner import RealS1Trainer,ScheduleJournal,identity_binding
    from src.tasks.velocity.v09.model import Frozen6400Teacher
    from v09_identity_fixtures import make_identity_fixture
    _, path, _, _ = cli_inputs
    f = make_identity_fixture(path.parent/'identity',CURRENT)
    ctx = EvidenceIdentityContext.from_approval(f.approval_path,f.approval_sha256,**f.context_kwargs())
    payload = torch.load(path,map_location='cpu',weights_only=False)
    payload['metadata']['code_sha256'] = CURRENT
    payload['metadata']['ledger_run_id'] = f.run_id
    import random
    import numpy as np
    payload['metadata']['value_scale'] = 2.
    payload['metadata']['random_state'] = {'python': random.getstate(), 'numpy': np.random.get_state()}
    payload['metadata']['external_ledger'] = f.ledger.checkpoint_metadata()
    payload['metadata']['shuffle_generator'] = torch.Generator().manual_seed(42).get_state()
    expected = identity_binding(ctx)
    if binding == 'wrong_sha':
        payload['metadata']['evidence_identity'] = dict(expected,approval_sha256='f'*64)
    elif binding == 'wrong_path':
        payload['metadata']['evidence_identity'] = dict(expected,approval_path=str(f.run_dir/'other.json'))
    elif binding == 'unexpected_context':
        payload['metadata']['evidence_identity'] = expected
    torch.save(payload,path)
    assert hashlib.sha256(path.read_bytes()).hexdigest()
    def forbidden_allocation(*args,**kwargs):
        pytest.fail('wrong identity reached teacher/model device allocation')
    monkeypatch.setattr(Frozen6400Teacher,'from_6400_checkpoint',forbidden_allocation)
    with pytest.raises(ValueError,match='identity'):
        RealS1Trainer(run_dir=f.run_dir,run_id=f.run_id,segment_id='resume',device='cuda:0',
            code_sha256=CURRENT,admissions={'num_envs':512,'value_scale':2.,
                'identity_context':None if binding == 'unexpected_context' else ctx},
            ledger=f.ledger,journal=ScheduleJournal(f.run_dir/'s1_schedule.jsonl'),
            evidence_hooks=object(),resume=path)
