"""Q010 consumer boundaries with real identity context, files and external ledger."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from a2_v07_bootstrap import activate
activate(ROOT)
from src.tasks.velocity.v09 import spec
from src.tasks.velocity.v09.cases import sha256
from src.tasks.velocity.v09.s1_runner import validate_admissions, validate_learning_approval, PlanningRequired
from v09_identity_fixtures import OLD_CODE, file_sha, make_identity_fixture

CURRENT = 'c'*64
SUITES = ('gate_A','gate_B','final_1','final_2','final_3')


def context(f):
    module = importlib.import_module('src.tasks.velocity.v09.evidence_identity')
    return module.EvidenceIdentityContext.from_approval(f.approval_path,f.approval_sha256,
                                                       **f.context_kwargs())


def test_formal_checkpoint_recovery_requires_all_same_context_fields(tmp_path):
    from src.tasks.velocity.v09.s1_runner import validate_formal_checkpoint_identity
    f = make_identity_fixture(tmp_path, CURRENT)
    ctx = context(f)
    binding = dict(approval_path=str(f.approval_path.resolve()),
        approval_sha256=f.approval_sha256, consumer_code_sha256=CURRENT,
        run_id=f.run_id, run_dir=str(f.run_dir.resolve()))
    payload = {'metadata': {'evidence_identity': binding}}
    original = deepcopy(payload)
    validate_formal_checkpoint_identity(payload, ctx)
    assert payload == original  # producer metadata remains the saved binding
    for key, value in (('run_id', 'other-run'), ('run_dir', str(tmp_path/'other')),
                       ('consumer_code_sha256', 'f'*64)):
        changed = {'metadata': {'evidence_identity': dict(binding, **{key: value})}}
        with pytest.raises(ValueError, match='identity'):
            validate_formal_checkpoint_identity(changed, ctx)
    with pytest.raises(ValueError, match='identity'):
        validate_formal_checkpoint_identity({'metadata': {'evidence_identity': None}}, ctx)


def make_consumer_identity(directory):
    f = make_identity_fixture(directory, CURRENT)
    duties = {'engineering': dict(kind='engineering',control_samples=0)}
    for suite in SUITES:
        duties['teacher_'+suite] = dict(kind='teacher_baseline',control_samples=1,
            suite=suite,manifest_sha256='a'*64)
    duties['calibration'] = dict(kind='calibration',control_samples=1,
        suite='train_calibration',manifest_sha256='a'*64)
    duties['probe_512'] = dict(kind='probe',control_samples=1,
        work_items={'rollout_001':dict(num_envs=1,control_steps=1,require_full_steps=True)})
    plan = dict(stage='S1', admitted=True, sds_sha256=spec.SDS_SHA256,
                total_control_samples=7, obligations=duties, case_catalogue={})
    plan['plan_sha256'] = sha256(plan)
    f.ledger.reserve_stage(plan)
    for name,duty in duties.items():
        if name == 'engineering':
            continue
        key = f.ledger.reserve(segment_id='fixture',operation_id=name,stage='S1',
            course='S1_RESOURCE' if name.startswith('probe') else 'S1.0',
            category='evaluation' if name.startswith('teacher') else 'engineering',
            num_envs=1,max_control_steps=1,checkpoint_sha256=spec.SOURCE_SHA256,
            reason='synthetic unit fixture, no physical execution',obligation_id=name,
            case_sha256='a'*64 if not name.startswith('probe') else None,
            work_item_id='rollout_001' if name.startswith('probe') else None)
        f.ledger.commit(key,actual_control_steps=1)
        f.ledger.close_obligation('S1',name,reason='synthetic complete fixture')
    prefix = f.ledger_path.read_bytes()
    state = f.ledger.snapshot()
    f.approval['ledger_prefix'] = dict(path=str(f.ledger_path.resolve()),byte_length=len(prefix),
        sha256=hashlib.sha256(prefix).hexdigest(),sequence=state['sequence'],event_sha256=state['ledger_sha256'])
    caches = {}
    for suite in SUITES:
        folder = f.run_dir/'evidence'/('teacher_'+suite)
        episodes = folder/'episodes.jsonl'
        folder.mkdir(parents=True)
        episodes.write_text(json.dumps(dict(sealed_final_marker=suite,case_id=0,success=True,
            checkpoint_sha256=spec.SOURCE_SHA256,manifest_sha256='a'*64))+'\n')
        summary = f.put_json(folder/'summary.json',dict(code_sha256=OLD_CODE,
            suite=suite,checkpoint_sha256=spec.SOURCE_SHA256,manifest_sha256='a'*64,
            total=1500 if suite.startswith('gate') else 7500,success=1))
        f.bind_file(episodes,kind='teacher_episodes',suite=suite,manifest_sha256='a'*64)
        f.bind_file(summary,kind='teacher_summary',suite=suite,manifest_sha256='a'*64)
        caches[suite] = dict(episodes_path=str(episodes.resolve()),summary_path=str(summary.resolve()),
            episodes_file_sha256=file_sha(episodes),summary_file_sha256=file_sha(summary),
            checkpoint_sha256=spec.SOURCE_SHA256,manifest_sha256='a'*64,code_sha256=OLD_CODE)
    f.baseline_path = f.put_json(f.run_dir/'baseline_admission.json',dict(schema=11,baseline_complete=True,
        sds_sha256=spec.SDS_SHA256,source_sha256=spec.SOURCE_SHA256,code_sha256=OLD_CODE,
        calibration_path=str(f.calibration_path.resolve()),calibration_sha256=file_sha(f.calibration_path),
        calibration_manifest_sha256='a'*64,value_scale=2.,teacher_cache=caches))
    f.bind_file(f.baseline_path,kind='baseline_index',suite=None)
    candidate = dict(num_envs=512,warmups_completed=5,measurements_completed=50,
        restored_iteration_completed=True,full_s1_2_workload_verified=True,
        evaluation_control_steps=5005,evaluation_num_envs=100,video_control_steps=5005,
        post_evidence_iteration_completed=True,error=None,
        memory=dict(monitoring_valid=True,conservative_whole_card_peak_bytes=1024**3,
            host_min_available_bytes=8*1024**3,phases=[{'name':name} for name in
                ('training','full_s1_2','unloaded_child_evaluation','unloaded_child_video','resume_training')]))
    resource = dict(schema_version=11,code_sha256=CURRENT,sds_sha256=spec.SDS_SHA256,
        source_sha256=spec.SOURCE_SHA256,admitted=True,selected_num_envs=512,candidates=[candidate],
        calibration=dict(f.calibration,calibration_file_sha256=file_sha(f.calibration_path)))
    resource['report_sha256'] = sha256(resource)
    f.resource_path = f.put_json(f.run_dir/'resource.json',resource)
    f.seal()
    from dataclasses import asdict
    resource['calibration']['evidence_ref'] = asdict(context(f).expected_ref(
        f.calibration_path,kind='calibration',suite='train_calibration'))
    resource['report_sha256'] = sha256({k:v for k,v in resource.items() if k != 'report_sha256'})
    f.put_json(f.resource_path,resource)
    return f


def test_legacy_baseline_calibration_admission_uses_real_context_without_final_decode(tmp_path,monkeypatch):
    f = make_consumer_identity(tmp_path)
    ctx = context(f)
    original_loads = json.loads
    def no_final_decode(raw,*args,**kwargs):
        text = raw.decode() if isinstance(raw,bytes) else raw
        if 'sealed_final_marker' in text and any(suite in text for suite in ('final_1','final_2','final_3')):
            raise AssertionError('learning admission must not decode sealed final teacher results')
        return original_loads(raw,*args,**kwargs)
    monkeypatch.setattr(json,'loads',no_final_decode)
    before = f.ledger_path.read_bytes()
    result = validate_admissions(run_dir=f.run_dir,resource_path=f.resource_path,
        baseline_path=f.baseline_path,calibration_path=f.calibration_path,
        code_sha256=CURRENT,ledger=f.ledger,identity_context=ctx)
    assert result['num_envs'] == 512 and result['value_scale'] == 2.
    assert result['baseline']['code_sha256'] == OLD_CODE
    assert result['calibration']['code_sha256'] == OLD_CODE
    assert result['calibration']['evidence_ref']['approval_sha256'] == f.approval_sha256
    assert f.ledger_path.read_bytes() == before


def test_current_resource_outer_sha_cannot_be_legacy_even_with_valid_context(tmp_path):
    f = make_consumer_identity(tmp_path)
    ctx = context(f)
    resource = json.loads(f.resource_path.read_text())
    resource['code_sha256'] = OLD_CODE
    resource['report_sha256'] = sha256({k:v for k,v in resource.items() if k != 'report_sha256'})
    f.put_json(f.resource_path,resource)
    with pytest.raises(ValueError,match='resource'):
        validate_admissions(run_dir=f.run_dir,resource_path=f.resource_path,
            baseline_path=f.baseline_path,calibration_path=f.calibration_path,
            code_sha256=CURRENT,ledger=f.ledger,identity_context=ctx)


def test_learning_approval_must_bind_same_identity_path_and_sha(tmp_path):
    f = make_consumer_identity(tmp_path)
    ctx = context(f)
    approval = dict(approved=True,stage='S1',sds_sha256=spec.SDS_SHA256,
        source_sha256=spec.SOURCE_SHA256,code_sha256=CURRENT,run_id=f.run_id,
        run_dir=str(f.run_dir.resolve()),scope=['S1.1_learning','S1.2_learning'],
        evidence_identity_path=str(f.approval_path.resolve()),evidence_identity_sha256=f.approval_sha256)
    for changed in (dict(approval,evidence_identity_sha256='f'*64),
                    dict(approval,evidence_identity_path=str(f.run_dir/'other.json'))):
        with pytest.raises(PlanningRequired):
            validate_learning_approval(changed,run_dir=f.run_dir,run_id=f.run_id,
                                      code_sha256=CURRENT,identity_context=ctx)
    assert validate_learning_approval(approval,run_dir=f.run_dir,run_id=f.run_id,
                                     code_sha256=CURRENT,identity_context=ctx)


def test_final_legacy_read_allowed_after_child_close_only_until_parent_finish(tmp_path):
    """Real authorization window only; zero spent learning is not training proof."""
    from src.tasks.velocity.v09.s1_runner import S1Schedule, ScheduleJournal
    f = make_identity_fixture(tmp_path)
    candidate, manifest = 'b'*64, 'a'*64
    folder = f.run_dir/'evidence/teacher_final_1'
    summary = f.put_json(folder/'summary.json',dict(code_sha256=OLD_CODE,
        checkpoint_sha256=spec.SOURCE_SHA256,manifest_sha256=manifest))
    episodes = folder/'episodes.jsonl'
    episodes.write_text('{"synthetic_case":0}\n')
    for path,kind in ((summary,'teacher_summary'),(episodes,'teacher_episodes')):
        f.bind_file(path,kind=kind,suite='final_1',manifest_sha256=manifest)
    f.seal()
    ctx = context(f)
    duties = dict(engineering=dict(kind='engineering',control_samples=0),
        candidate_final_1=dict(kind='candidate_finals',control_samples=1,
                               suite='final_1',manifest_sha256=manifest))
    plan = dict(stage='S1',admitted=True,sds_sha256=spec.SDS_SHA256,
                total_control_samples=1,obligations=duties,case_catalogue={})
    plan['plan_sha256'] = sha256(plan)
    f.ledger.reserve_stage(plan)
    gates = [dict(attempt=i,suite=suite,candidate_sha256=candidate,status='complete',
        passed=True,engineering_valid=True,report_sha256='d'*64)
        for i,suite in enumerate(('gate_A','gate_B'),1)]
    schedule = S1Schedule(512,segment='main',gates=gates,used_slots=[1,2],
                          final_pending=True,engineering_adapt_passed=True)
    journal = ScheduleJournal(f.run_dir/'s1_schedule.jsonl')
    f.ledger.advance_state('S1',gate_attempts=2)
    journal.append(schedule,event='synthetic_same_sha_AB_fixture')
    operation = f.ledger.reserve(segment_id='synthetic',operation_id='final_1',
        stage='S1',course='S1.2',category='evaluation',num_envs=1,max_control_steps=1,
        checkpoint_sha256=candidate,case_sha256=manifest,obligation_id='candidate_final_1',
        reason='CPU fixture only, no physics')
    f.ledger.commit(operation,actual_control_steps=1)
    f.ledger.close_obligation('S1','candidate_final_1',reason='synthetic child completed')
    raw,ref = ctx.read_legacy_bytes(episodes,kind='teacher_episodes',suite='final_1',
                                    candidate_sha256=candidate)
    assert raw == b'{"synthetic_case":0}\n' and ref.producer_code_sha256 == OLD_CODE
    schedule.finish_final(candidate_sha256=candidate,passed=True,
                          engineering_valid=True,report_sha256='e'*64)
    f.ledger.advance_state('S1',final_suites_used=1)
    journal.append(schedule,event='synthetic_parent_finish_final_fixture')
    with pytest.raises(PermissionError):
        ctx.read_legacy_bytes(episodes,kind='teacher_episodes',suite='final_1',
                              candidate_sha256=candidate)


@pytest.mark.parametrize('candidate_code',[CURRENT,OLD_CODE])
def test_paired_gate_uses_verified_old_teacher_but_rejects_old_candidate_output(tmp_path,monkeypatch,candidate_code):
    from src.tasks.velocity.v09.cases import build_s1_manifest,freeze_manifest,canonical_json
    from src.tasks.velocity.v09.s1_runner import SubprocessEvaluator
    f = make_identity_fixture(tmp_path)
    manifest = build_s1_manifest('gate_A')
    manifest_path = f.run_dir/'cases/gate_A.json'
    freeze_manifest(manifest_path,manifest)
    f.bind_file(manifest_path,kind='frozen_manifest',suite='gate_A',
        manifest_sha256=manifest['manifest_sha256'],category='frozen_inputs')
    metrics = dict(speed_rmse=.01,pitch_rms_deg=1.,slip=.1,action_change=.1,stride=.2,frequency=2.,drift=.01)
    teacher = [dict(case_id=case['case_id'],case_sha256=case['case_sha256'],
        checkpoint_sha256=spec.SOURCE_SHA256,manifest_sha256=manifest['manifest_sha256'],
        success=True,failure=False,timeout=False,reason='success',forward_distance=2.,
        failure_masks={'body':False},metrics=dict(metrics)) for case in manifest['cases']]
    folder = f.run_dir/'evidence/teacher_gate_A'
    folder.mkdir(parents=True)
    episodes = folder/'episodes.jsonl'
    episodes.write_text(''.join(canonical_json(row)+'\n' for row in teacher))
    summary = f.put_json(folder/'summary.json',dict(code_sha256=OLD_CODE,suite='gate_A',
        total=1500,checkpoint_sha256=spec.SOURCE_SHA256,manifest_sha256=manifest['manifest_sha256']))
    for path,kind in ((summary,'teacher_summary'),(episodes,'teacher_episodes')):
        f.bind_file(path,kind=kind,suite='gate_A',manifest_sha256=manifest['manifest_sha256'])
    f.seal()
    ctx = context(f)
    cache = {'gate_A':dict(episodes_path=str(episodes),summary_path=str(summary),
        episodes_file_sha256=file_sha(episodes),summary_file_sha256=file_sha(summary),
        checkpoint_sha256=spec.SOURCE_SHA256,manifest_sha256=manifest['manifest_sha256'],code_sha256=OLD_CODE)}
    calls = []
    candidate_sha = 'd'*64
    def child(command,*,cwd,check):
        calls.append(command)
        output = f.run_dir/'evidence'/command[command.index('--segment-id')+1]
        output.mkdir()
        rows = [dict(row,checkpoint_sha256=candidate_sha) for row in teacher]
        (output/'episodes.jsonl').write_text(''.join(canonical_json(row)+'\n' for row in rows))
        f.put_json(output/'summary.json',dict(suite='gate_A',total=1500,
            checkpoint_sha256=candidate_sha,manifest_sha256=manifest['manifest_sha256'],code_sha256=candidate_code))
    monkeypatch.setattr('subprocess.run',child)  # actual external physics boundary only
    evaluator = SubprocessEvaluator(run_dir=f.run_dir,run_id=f.run_id,device='cpu',
        code_sha256=CURRENT,python=sys.executable,teacher_cache=cache,identity_context=ctx)
    original = episodes.read_bytes()
    if candidate_code == OLD_CODE:
        with pytest.raises(ValueError,match='code'):
            evaluator(suite='gate_A',candidate_path=f.run_dir/'new.pt',candidate_sha256=candidate_sha,
                      attempt=1,segment_id='candidate_gate')
    else:
        result = evaluator(suite='gate_A',candidate_path=f.run_dir/'new.pt',candidate_sha256=candidate_sha,
                           attempt=1,segment_id='candidate_gate')
        assert result['passed'] and result['fixed_denominator'] == 1500
        assert result['teacher_episodes_evidence_ref']['producer_code_sha256'] == OLD_CODE
        assert result['code_sha256'] == CURRENT
        assert calls[0][calls[0].index('--evidence-identity')+1] == str(f.approval_path)
        assert calls[0][calls[0].index('--evidence-identity-sha256')+1] == f.approval_sha256
    assert episodes.read_bytes() == original


def test_diagnostic_hooks_forward_same_context_and_explicit_run_id(tmp_path,monkeypatch):
    from src.tasks.velocity.v09.cases import build_s1_manifest,freeze_manifest
    from src.tasks.velocity.v09.s1_runner import SubprocessEvidenceHooks
    f = make_identity_fixture(tmp_path)
    manifest = build_s1_manifest('S1_DIAG')
    path = f.run_dir/'cases/S1_DIAG.json'
    freeze_manifest(path,manifest)
    f.bind_file(path,kind='frozen_manifest',suite='S1_DIAG',
                manifest_sha256=manifest['manifest_sha256'],category='frozen_inputs')
    f.seal()
    ctx = context(f)
    calls = []
    def child(command,*,cwd,check):
        calls.append(command)
        if '--output' in command:
            f.put_json(Path(command[command.index('--output')+1]),dict(sds_sha256=spec.SDS_SHA256,
                code_sha256=CURRENT,engineering_valid=True,finite=True,migration_valid=True,contacts_valid=True))
        else:
            segment = command[command.index('--segment-id')+1]
            f.put_json(f.run_dir/'evidence'/segment/'summary.json',dict(suite='S1_DIAG',total=1500,
                checkpoint_sha256=command[command.index('--checkpoint-sha256')+1],
                manifest_sha256=manifest['manifest_sha256'],code_sha256=CURRENT))
    monkeypatch.setattr('subprocess.run',child)
    hooks = SubprocessEvidenceHooks(python=sys.executable,run_dir=f.run_dir,run_id=f.run_id,
        segment_id='train',device='cpu',code_sha256=CURRENT,calibration_path=f.calibration_path,
        identity_context=ctx)
    before = hooks.diagnostic(phase='before',checkpoint=f.run_dir/'initial.pt',checkpoint_sha256='b'*64)
    after = hooks.diagnostic(phase='after',checkpoint=f.run_dir/'adapted.pt',checkpoint_sha256='d'*64)
    result = hooks.compare(before=before,after=after)
    assert result['engineering_valid'] and result['report_sha256']
    for command in calls:
        assert command[command.index('--evidence-identity')+1] == str(f.approval_path)
        assert command[command.index('--evidence-identity-sha256')+1] == f.approval_sha256
        assert command[command.index('--run-id')+1] == f.run_id
