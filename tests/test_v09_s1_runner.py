"""S1 control-flow/denominator tests: CPU only, no environment construction."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import a2_v07_bootstrap

a2_v07_bootstrap.activate(ROOT)
from src.tasks.velocity.v09 import spec
from src.tasks.velocity.v09.cases import build_s1_manifest, sha256
from src.tasks.velocity.v09.s1_runner import (
    CAPS, S1Schedule, ScheduleJournal, PlanningRequired, LedgerEpisodeIds,
    SubprocessEvidenceHooks, compare_suite,
)


def main_schedule(n=2048):
    result = S1Schedule(n)
    result.spent['adapt'] = result.effective_cap
    result.finish_adapt(finite=True, migration_valid=True, contacts_valid=True,
                        comparison_report_sha256='e' * 64)
    return result


def finish_pair(schedule, candidate='a' * 64):
    first = schedule.start_gate(candidate)
    schedule.finish_gate(candidate_sha256=candidate, passed=True,
                         engineering_valid=True, report_sha256='1' * 64)
    second = schedule.start_gate(candidate)
    schedule.finish_gate(candidate_sha256=candidate, passed=True,
                         engineering_valid=True, report_sha256='2' * 64)
    return first, second


def history(schedule, count):
    schedule.used_slots = list(range(1, count + 1))
    schedule.gates = [dict(attempt=i, original_slot=i, status='complete',
                          suite='gate_A' if i % 2 else 'gate_B', passed=False,
                          candidate_sha256='0' * 64) for i in range(1, count + 1)]


def test_adaptation_requires_floor_complete_rollouts_and_no_behavior_gate():
    for n in spec.ENV_CANDIDATES:
        schedule = S1Schedule(n)
        assert schedule.due_slot() is None
        assert schedule.effective_cap == 20000000 // (n * 48) * (n * 48)
        with pytest.raises(ValueError):
            schedule.start_gate('a' * 64)
        schedule.spent['adapt'] = schedule.effective_cap - schedule.rollout_samples
        with pytest.raises(ValueError):
            schedule.finish_adapt(finite=True, migration_valid=True, contacts_valid=True,
                                  comparison_report_sha256='e' * 64)
        schedule.consume_rollout()
        assert not schedule.can_collect()
        with pytest.raises(PlanningRequired):
            schedule.finish_adapt(finite=True, migration_valid=True, contacts_valid=False,
                                  comparison_report_sha256='e' * 64)
        schedule.finish_adapt(finite=True, migration_valid=True, contacts_valid=True,
                              comparison_report_sha256='e' * 64)
        assert schedule.segment == 'main' and schedule.course == 'S1.2'
        assert not schedule.gates and schedule.recoveries == 0


def test_main_threshold_mapping_and_same_sha_confirmation_blocks_learning():
    schedule = main_schedule()
    boundary = (100000000 + schedule.rollout_samples - 1) // schedule.rollout_samples * schedule.rollout_samples
    schedule.spent['main'] = boundary - schedule.rollout_samples
    assert schedule.due_slot() is None
    schedule.consume_rollout()
    assert schedule.due_slot() == 1
    first = schedule.start_gate('a' * 64)
    assert first['suite'] == 'gate_A'
    schedule.finish_gate(candidate_sha256='a' * 64, passed=True,
                         engineering_valid=True, report_sha256='1' * 64)
    assert not schedule.can_collect()
    with pytest.raises(ValueError, match='frozen candidate'):
        schedule.start_gate('b' * 64)
    with pytest.raises(ValueError, match='frozen'):
        schedule.consume_rollout()
    second = schedule.start_gate('a' * 64)
    assert second['suite'] == 'gate_B' and second['original_slot'] == 2
    schedule.finish_gate(candidate_sha256='a' * 64, passed=True,
                         engineering_valid=True, report_sha256='2' * 64)
    assert schedule.final_pending and schedule.begin_final() == 'final_1'
    assert schedule.cancelled_slots == [3, 4, 5, 6, 7]


def test_q006_seven_to_eight_borrows_gate_only_never_retry_learning():
    schedule = main_schedule()
    history(schedule, 6)
    schedule.spent['main'] = schedule.effective_cap
    first, second = finish_pair(schedule)
    assert (first['original_slot'], second['original_slot']) == (7, 8)
    assert (first['attempt'], second['attempt']) == (7, 8)
    assert schedule.segment == 'main' and schedule.spent['retry_1'] == 0
    schedule.finish_final(candidate_sha256='a' * 64, passed=False,
                          engineering_valid=True, report_sha256='f' * 64)
    assert schedule.segment == 'retry_1' and schedule.recoveries == 1
    assert schedule.due_slot() is None
    schedule.spent['retry_1'] = CAPS['retry_1'] // schedule.rollout_samples * schedule.rollout_samples
    first, second = finish_pair(schedule, 'b' * 64)
    assert (first['original_slot'], second['original_slot']) == (9, 10)
    schedule.finish_final(candidate_sha256='b' * 64, passed=False,
                          engineering_valid=True, report_sha256='e' * 64)
    assert schedule.segment == 'retry_2'
    with pytest.raises(PlanningRequired, match='only one'):
        schedule.due_slot()


def test_q006_ten_to_eleven_and_three_final_sets():
    schedule = main_schedule()
    schedule.segment = 'retry_2'
    schedule.recoveries, schedule.final_suites_used = 2, 2
    history(schedule, 9)
    schedule.spent['retry_2'] = 50000000 // schedule.rollout_samples * schedule.rollout_samples + schedule.rollout_samples
    first, second = finish_pair(schedule)
    assert [first['original_slot'], second['original_slot']] == [10, 11]
    assert [first['suite'], second['suite']] == ['gate_B', 'gate_A']
    assert schedule.begin_final() == 'final_3'
    schedule.finish_final(candidate_sha256='a' * 64, passed=True,
                          engineering_valid=True, report_sha256='f' * 64)
    assert schedule.accepted and schedule.final_suites_used == 3


def test_q007_early_segment_close_cancels_thresholds_and_keeps_attempts_contiguous():
    schedule = main_schedule()
    schedule.spent['main'] = 100000000 // schedule.rollout_samples * schedule.rollout_samples + schedule.rollout_samples
    first, second = finish_pair(schedule)
    assert [first['original_slot'], second['original_slot']] == [1, 2]
    assert schedule.cancelled_slots == [3, 4, 5, 6, 7]
    old_main_spent = schedule.spent['main']
    schedule.finish_final(candidate_sha256='a' * 64, passed=False,
                          engineering_valid=True, report_sha256='f' * 64)
    assert schedule.segment == 'retry_1' and schedule.spent['main'] == old_main_spent
    schedule.spent['retry_1'] = 50000000 // schedule.rollout_samples * schedule.rollout_samples + schedule.rollout_samples
    first, second = finish_pair(schedule, 'b' * 64)
    assert [first['original_slot'], second['original_slot']] == [8, 9]
    assert [first['attempt'], second['attempt']] == [3, 4]
    assert [first['suite'], second['suite']] == ['gate_A', 'gate_B']
    schedule.finish_final(candidate_sha256='b' * 64, passed=False,
                          engineering_valid=True, report_sha256='e' * 64)
    assert schedule.segment == 'retry_2'
    schedule.spent['retry_2'] = 50000000 // schedule.rollout_samples * schedule.rollout_samples + schedule.rollout_samples
    first, second = finish_pair(schedule, 'c' * 64)
    assert [first['original_slot'], second['original_slot']] == [10, 11]
    assert [first['attempt'], second['attempt']] == [5, 6]
    assert schedule.begin_final() == 'final_3'


def test_engineering_final_does_not_consume_behavior_set_or_unlock_learning():
    schedule = main_schedule()
    schedule.spent['main'] = 100100000
    finish_pair(schedule)
    schedule.finish_final(candidate_sha256='a' * 64, passed=False,
                          engineering_valid=False, report_sha256='f' * 64)
    assert schedule.final_suites_used == 0 and schedule.recoveries == 0
    assert schedule.final_pending and schedule.blocked
    assert not schedule.can_collect()


def test_failed_confirmation_at_cap_requires_planning():
    schedule = main_schedule()
    history(schedule, 6)
    schedule.spent['main'] = schedule.effective_cap
    first = schedule.start_gate('a' * 64)
    schedule.finish_gate(candidate_sha256='a' * 64, passed=True,
                         engineering_valid=True, report_sha256='1' * 64)
    schedule.start_gate('a' * 64)
    schedule.finish_gate(candidate_sha256='a' * 64, passed=False,
                         engineering_valid=True, report_sha256='2' * 64)
    assert schedule.blocked and not schedule.final_pending and schedule.recoveries == 0


def test_interrupted_physics_retains_consumption_without_learning_from_partial():
    schedule = main_schedule()
    schedule.consume_rollout(13, complete=False)
    assert schedule.spent['main'] == schedule.num_envs * 13
    assert schedule.complete_rollouts == 0 and schedule.blocked
    with pytest.raises(ValueError):
        main_schedule().consume_rollout(47, complete=True)
    schedule.spent['main'] = CAPS['main'] - 1
    with pytest.raises(PlanningRequired):
        schedule.consume_rollout()


def test_journal_resume_keeps_cancelled_slots_attempts_and_nonrewind_state(tmp_path):
    schedule = main_schedule()
    schedule.spent['main'] = 100100000
    finish_pair(schedule)
    journal = ScheduleJournal(tmp_path / 'schedule.jsonl')
    journal.append(schedule, event='same_sha_AB')
    restored = S1Schedule(**journal.read()[-1]['schedule'])
    assert restored.state_dict() == schedule.state_dict()
    assert restored.begin_final() == 'final_1'
    with journal.path.open('a') as stream:
        stream.write('{"partial":')
    with pytest.raises(PlanningRequired, match='incomplete'):
        journal.read()


class FakeLedger:
    def __init__(self):
        self.progress = {}
    def snapshot(self):
        return {'stages': {'S1': {'progress': dict(self.progress)}}}
    def advance_state(self, stage, **counters):
        assert all(value >= self.progress.get(key, 0) for key, value in counters.items())
        self.progress.update(counters)


def test_episode_blocks_skip_unused_ids_on_restart_and_N_change():
    ledger = FakeLedger()
    first = LedgerEpisodeIds(ledger, block_size=4)
    assert [first.reserve('flat_walk') for _ in range(2)] == [0, 1]
    assert first.reserve('flat_stop') == 0
    restored = LedgerEpisodeIds(ledger, block_size=4)
    assert restored.reserve('flat_walk') == 4
    assert restored.reserve('flat_stop') == 4
    assert ledger.progress['episode_flat_walk'] == 8


@pytest.fixture(scope='module')
def paired_suite():
    manifest = build_s1_manifest('gate_A')
    metrics = dict(speed_rmse=.01, pitch_rms_deg=1., slip=.1, action_change=.1,
                   stride=.2, frequency=2., drift=.01)
    rows = [dict(case_sha256=case['case_sha256'], case_id=case['case_id'],
            manifest_sha256=manifest['manifest_sha256'], success=True, metrics=dict(metrics),
            failure_masks={'body': False}, forward_distance=2.) for case in manifest['cases']]
    return manifest, rows


def test_compare_suite_uses_all_75_real_group_identities(paired_suite):
    manifest, rows = paired_suite
    report = compare_suite(manifest, rows, deepcopy(rows))
    assert report['passed'] and report['group_count'] == 75
    assert report['fixed_denominator'] == 1500
    assert all(group['N'] == 20 for group in report['groups'])
    assert len({json.dumps(group['group_identity']) for group in report['groups']}) == 75


def test_compare_suite_rejects_missing_duplicate_or_wrong_manifest(paired_suite):
    manifest, rows = paired_suite
    for changed in (rows[:-1], rows[:-1] + rows[:1]):
        with pytest.raises(ValueError, match='every fixed case'):
            compare_suite(manifest, rows, changed)
    changed = deepcopy(rows)
    changed[0]['manifest_sha256'] = 'bad'
    with pytest.raises(ValueError, match='manifest identity'):
        compare_suite(manifest, rows, changed)


def test_missing_real_evidence_driver_fails_before_any_physics(tmp_path):
    with pytest.raises(PlanningRequired, match='not implemented'):
        SubprocessEvidenceHooks(script=tmp_path / 'missing.py', python=sys.executable,
            run_dir=tmp_path, run_id='r', segment_id='s', device='cuda:0', code_sha256='a' * 64,
            calibration_path=tmp_path / 'calibration.json')


def test_external_allocation_mismatch_and_unknown_never_free_on_resume():
    from src.tasks.velocity.v09.s1_runner import validate_schedule_ledger, ALLOCATIONS
    schedule = main_schedule()
    stage = dict(allocations={ALLOCATIONS[key]: value for key, value in schedule.spent.items()},
                 spent={'learning': schedule.learning_samples}, progress={})
    state = dict(stages={'S1': stage}, operations={})
    assert validate_schedule_ledger(schedule, state)
    stage['allocations']['S1.2_main'] = 98304
    with pytest.raises(PlanningRequired, match='consumption differs'):
        validate_schedule_ledger(schedule, state)
    stage['allocations']['S1.2_main'] = 0
    state['operations']['pending'] = dict(category='learning', settled=False)
    with pytest.raises(PlanningRequired, match='unsettled learning'):
        validate_schedule_ledger(schedule, state)


def test_baseline_complete_schema_and_sealed_final_metadata_admission(tmp_path, monkeypatch):
    from src.tasks.velocity.v09.s1_runner import validate_admissions, file_sha256
    from src.tasks.velocity.v09 import resources
    monkeypatch.setattr(resources, 'qualifies', lambda result: result.get('verified') is True)
    code = 'c' * 64
    calibration = dict(sds_sha256=spec.SDS_SHA256, source_sha256=spec.SOURCE_SHA256,
        code_sha256=code, manifest_sha256='m' * 64, value_scale=2.,
        method='population_std_max_1', physical_control_samples=100)
    cp = tmp_path / 'calibration.json'
    cp.write_text(json.dumps(calibration))
    entries = {}
    obligations = {'calibration': {'closed': True}, 'probe_2048': {'closed': True}}
    for suite in ('gate_A', 'gate_B', 'final_1', 'final_2', 'final_3'):
        ep = tmp_path / (suite + '_episodes.jsonl')
        ep.write_text('SEALED INVALID JSON MUST NEVER BE PARSED DURING TRAINING ADMISSION')
        summary = tmp_path / (suite + '_summary.json')
        summary.write_text('SEALED SUMMARY ALSO MUST NOT BE PARSED')
        entries[suite] = dict(episodes_path=str(ep), summary_path=str(summary),
            checkpoint_sha256=spec.SOURCE_SHA256, manifest_sha256='f' * 64,
            episodes_file_sha256=file_sha256(ep))
        obligations['teacher_' + suite] = {'closed': True}
    baseline = dict(schema=11, baseline_complete=True, resource_admitted=False,
        behavior_restored=False, sds_sha256=spec.SDS_SHA256, source_sha256=spec.SOURCE_SHA256,
        code_sha256=code, calibration_path=str(cp.resolve()), calibration_sha256=file_sha256(cp),
        calibration_manifest_sha256=calibration['manifest_sha256'], value_scale=2., teacher_cache=entries)
    bp = tmp_path / 'baseline.json'
    bp.write_text(json.dumps(baseline))
    resource = dict(schema_version=11, sds_sha256=spec.SDS_SHA256, source_sha256=spec.SOURCE_SHA256,
        code_sha256=code, admitted=True, selected_num_envs=2048,
        calibration=dict(value_scale=2., calibration_file_sha256=file_sha256(cp)),
        candidates=[dict(num_envs=2048, verified=True)])
    resource['report_sha256'] = sha256(resource)
    rp = tmp_path / 'resource.json'
    rp.write_text(json.dumps(resource))
    class Ledger:
        def snapshot(self):
            return {'stages': {'S1': {'obligations': obligations}}}
    result = validate_admissions(run_dir=tmp_path, resource_path=rp, baseline_path=bp,
                                calibration_path=cp, code_sha256=code, ledger=Ledger())
    assert result['num_envs'] == 2048 and result['value_scale'] == 2.
    baseline['baseline_complete'] = False
    bp.write_text(json.dumps(baseline))
    with pytest.raises(PlanningRequired, match='baseline caches'):
        validate_admissions(run_dir=tmp_path, resource_path=rp, baseline_path=bp,
                            calibration_path=cp, code_sha256=code, ledger=Ledger())


def test_final_unseal_requires_authoritative_adjacent_same_sha_AB(tmp_path):
    from src.tasks.velocity.v09.s1_runner import validate_final_unseal, ALLOCATIONS
    schedule = main_schedule()
    schedule.spent['main'] = 100100000
    class Ledger:
        def snapshot(self):
            return dict(stages={'S1': dict(
                allocations={ALLOCATIONS[k]: v for k, v in schedule.spent.items()},
                spent={'learning': schedule.learning_samples},
                progress={'gate_attempts': len(schedule.gates),
                          'final_suites_used': schedule.final_suites_used,
                          'final_recoveries': schedule.recoveries})}, operations={})
    ledger = Ledger()
    with pytest.raises(PlanningRequired, match='remains sealed'):
        validate_final_unseal(run_dir=tmp_path, ledger=ledger, suite='final_1', candidate_sha256='a' * 64)
    finish_pair(schedule)
    journal = ScheduleJournal(tmp_path / 's1_schedule.jsonl')
    journal.append(schedule, event='complete_same_sha_AB')
    proof = validate_final_unseal(run_dir=tmp_path, ledger=ledger, suite='final_1', candidate_sha256='a' * 64)
    assert proof['gate_attempts'] == [1, 2]
    for suite, digest in (('final_2', 'a' * 64), ('final_1', 'b' * 64)):
        with pytest.raises(PlanningRequired, match='remains sealed'):
            validate_final_unseal(run_dir=tmp_path, ledger=ledger, suite=suite, candidate_sha256=digest)


def test_s12_transition_resets_physical_episodes_keeps_adam_momentum_and_ids():
    import torch
    from src.tasks.velocity.v09.s1_runner import RealS1Trainer
    class Student(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.))
            self.course = 'S1.1'
        def set_s1_course(self, value):
            self.course = value
    student = Student()
    optimizer = torch.optim.Adam(student.parameters(), lr=3e-5)
    student.weight.square().backward()
    optimizer.step()
    momentum = optimizer.state[student.weight]['exp_avg'].clone()
    adam_step = optimizer.state[student.weight]['step'].clone()
    ledger = FakeLedger()
    ids = LedgerEpisodeIds(ledger, block_size=4)
    old_id = ids.reserve('flat_walk')
    class Env:
        course = 'S1.1'
        teacher_fraction = .5
        resets = 0
        def set_course(self, value):
            self.course = value
        def reset(self):
            self.resets += 1
            assert self.course == 'S1.2'
            self.new_episode = ids.reserve('flat_walk')
            self.teacher_fraction = .1
    trainer = RealS1Trainer.__new__(RealS1Trainer)
    trainer.student, trainer.optimizer, trainer.ids, trainer.env = student, optimizer, ids, Env()
    trainer._switch_to_s1_2()
    assert trainer.optimizer is optimizer and trainer.ids is ids
    assert trainer.env.resets == 1 and trainer.env.teacher_fraction == .1
    assert trainer.env.new_episode > old_id
    assert torch.equal(optimizer.state[student.weight]['exp_avg'], momentum)
    assert torch.equal(optimizer.state[student.weight]['step'], adam_step)
    assert student.course == trainer.env.course == 'S1.2'


@pytest.mark.parametrize('saved_identity',['matching','mismatch','missing'])
def test_q009_evidence_offload_restores_same_adam_rng_and_new_ids(tmp_path, monkeypatch, saved_identity):
    import random
    import numpy as np
    import torch
    from src.tasks.velocity.v09 import checkpoint
    from src.tasks.velocity.v09 import s1_runner as runner
    class Student(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.))
            self.std = torch.nn.Parameter(torch.tensor(.3), requires_grad=False)
            self.register_buffer('normalizer_mean', torch.tensor(.2))
    student = Student()
    optimizer = torch.optim.Adam(student.parameters(), lr=3e-5)
    student.weight.square().backward()
    optimizer.step()
    teacher = torch.nn.Linear(1,1)
    ledger = FakeLedger()
    ledger.validate_checkpoint = lambda metadata: ledger.snapshot()
    ids = LedgerEpisodeIds(ledger, block_size=4)
    roles = ['flat_walk', 'flat_stop']
    old_ids = [ids.reserve(role) for role in roles]
    class Env:
        def __init__(self, episode_ids):
            self.episode_length_buf = torch.tensor([13,0])
            self.case_episode_ids = episode_ids
            self.v09_cases = [{'case_sha256': 'a'*64}, {'case_sha256': 'b'*64}]
            self.closed = False
        def close(self):
            self.closed = True
    env = Env(old_ids)
    trainer = runner.RealS1Trainer.__new__(runner.RealS1Trainer)
    trainer.student, trainer.teacher, trainer.optimizer = student, teacher, optimizer
    trainer.schedule = main_schedule(512)
    trainer.env, trainer.roles, trainer.ids, trainer.ledger = env, roles, ids, ledger
    trainer.device = 'cpu'
    trainer.code_sha256 = 'c'*64
    trainer.generator = torch.Generator().manual_seed(42)
    trainer.buffers = [torch.ones(2), torch.ones(3)]
    trainer.evidence_session_state = None
    trainer._iteration_in_progress = False
    events = []
    trainer._record = lambda event, **details: events.append((event, details))
    monkeypatch.setattr(runner, 'validate_schedule_ledger', lambda schedule, state: True)
    saves = []
    def save(label):
        path = tmp_path / f'{len(saves)}.pt'
        metadata = dict(stage='S1', course='S1.2', value_scale=1., manifest_sha256='d'*64,
            ledger_run_id='session-run', learning_samples=trainer.schedule.learning_samples,
            code_sha256='c'*64, external_ledger={},
            gate_state=dict(next_suite='gate_A', attempts=0, final_suites_used=0,
                            recoveries=0, next_gate_samples=100000000, passes=[]),
            role_state=dict(num_envs=512, episode_counters=ids.state_dict(), assignments=roles),
            random_state={'python': random.getstate(), 'numpy': np.random.get_state()})
        if saved_identity == 'matching':
            metadata['evidence_identity'] = None
        elif saved_identity == 'mismatch':
            metadata['evidence_identity'] = {'approval_sha256': 'f'*64}
        digest = checkpoint.save_checkpoint(path, student, optimizer, metadata)
        saves.append(path)
        return path, digest
    trainer._save = save
    def new_env():
        torch.rand(3)  # construction must not perturb the learner RNG
        return Env([ids.reserve(role) for role in roles])
    trainer._new_env = new_env
    fingerprint = runner.training_state_fingerprint(student, optimizer)
    adam_step = optimizer.state[student.weight]['step'].clone()
    before_rng = trainer._capture_training_rng()
    before_samples = trainer.schedule.learning_samples
    trainer._begin_evidence_session('gate_A')
    assert env.closed and trainer.env is None and trainer.buffers == [None,None]
    assert trainer.optimizer is optimizer and trainer.student is student
    truncated = next(details['episodes'] for event, details in events if event == 'evidence_external_truncations')
    assert len(truncated) == 1 and truncated[0]['physical_control_steps'] == 13
    assert truncated[0]['success'] is None and truncated[0]['failure'] is None
    assert truncated[0]['reward_added'] == 0. and truncated[0]['control_steps_already_billed']
    trainer._begin_evidence_session('gate_B')
    assert len(saves) == 1  # same-SHA contiguous evidence never repeats save/Adam
    random.random(); np.random.rand(); torch.rand(2)
    trainer.generator.manual_seed(19)
    if saved_identity != 'matching':
        with pytest.raises(ValueError,match='identity'):
            trainer._end_evidence_session()
        assert trainer.schedule.learning_samples == before_samples
        assert trainer.env is None
        return
    trainer._end_evidence_session()
    assert trainer.schedule.learning_samples == before_samples
    assert trainer.optimizer is optimizer and trainer.ids is ids
    assert torch.equal(optimizer.state[student.weight]['step'], adam_step)
    assert runner.training_state_fingerprint(student, optimizer) == fingerprint
    assert all(new > old for new, old in zip(trainer.env.case_episode_ids, old_ids))
    after_rng = trainer._capture_training_rng()
    assert before_rng['python'] == after_rng['python']
    assert np.array_equal(before_rng['numpy'][1], after_rng['numpy'][1])
    assert torch.equal(before_rng['torch'], after_rng['torch'])
    assert torch.equal(before_rng['shuffle'], after_rng['shuffle'])
    with pytest.raises(PlanningRequired, match='fully committed'):
        trainer._iteration_in_progress = True
        trainer._begin_evidence_session('partial_update')


def test_q009_plain_checkpoint_save_does_not_reset_environment(tmp_path):
    import torch
    from src.tasks.velocity.v09.s1_runner import RealS1Trainer
    trainer = RealS1Trainer.__new__(RealS1Trainer)
    trainer.run_dir, trainer.segment_id, trainer.run_id = tmp_path, 'plain', 'r'
    trainer.schedule = main_schedule(512)
    trainer.student = torch.nn.Linear(1,1)
    trainer.optimizer = torch.optim.Adam(trainer.student.parameters(), lr=3e-5)
    trainer.generator = torch.Generator().manual_seed(42)
    trainer.roles = ['flat_walk']
    trainer.ids = LedgerEpisodeIds(FakeLedger())
    trainer.admissions = {'value_scale': 1., 'calibration': {'manifest_sha256': 'a'*64}}
    trainer.code_sha256 = 'c'*64
    trainer.loss_reports, trainer.latest = [], []
    class Ledger:
        def checkpoint_metadata(self):
            return {'sequence': 1}
    trainer.ledger = Ledger()
    trainer._record = lambda *args, **kwargs: None
    class Env:
        resets = 0
        def reset(self):
            self.resets += 1
    original = trainer.env = Env()
    first, first_sha = trainer._save('course_entry')
    original_bytes = first.read_bytes()
    second, _ = trainer._save('course_entry')
    assert first != second and first.is_file() and second.is_file()
    assert first.read_bytes() == original_bytes
    from src.tasks.velocity.v09.s1_runner import file_sha256
    assert file_sha256(first) == first_sha
    assert trainer.env is original and original.resets == 0


def test_q001_unjudgeable_final_keeps_opened_set_and_blocks_retry_learning():
    schedule = main_schedule()
    schedule.spent['main'] = 100100000
    finish_pair(schedule)
    before_samples = schedule.learning_samples
    schedule.finish_final(candidate_sha256='a' * 64, passed=False, engineering_valid=True,
        report_sha256='f' * 64, planning_required=True)
    assert schedule.blocked and 'Q001' in schedule.blocked
    assert schedule.final_suites_used == 1  # already unsealed, never silently reuse
    assert schedule.final_outcomes[-1]['suite'] == 'final_1'
    assert schedule.final_outcomes[-1]['report_sha256'] == 'f' * 64
    assert schedule.final_outcomes[-1]['planning_required'] is True
    assert schedule.recoveries == 0 and schedule.segment == 'main'
    assert schedule.spent['retry_1'] == schedule.spent['retry_2'] == 0
    assert schedule.learning_samples == before_samples
    assert not schedule.accepted and not schedule.final_pending and not schedule.can_collect()
    with pytest.raises(PlanningRequired):
        schedule.begin_final()
    restored = S1Schedule(**schedule.state_dict())
    assert restored.blocked and restored.final_suites_used == 1 and restored.recoveries == 0


def test_q001_planning_required_cannot_produce_acceptance_even_if_pass_flag_is_true():
    schedule = main_schedule()
    schedule.spent['main'] = 100100000
    finish_pair(schedule)
    schedule.finish_final(candidate_sha256='a' * 64, passed=True, engineering_valid=True,
        report_sha256='f' * 64, planning_required=True)
    assert schedule.blocked and not schedule.accepted and schedule.recoveries == 0


def test_run_final_planning_result_never_collects_retry_or_emits_restored(tmp_path):
    from types import SimpleNamespace
    from src.tasks.velocity.v09.s1_runner import RealS1Trainer
    trainer = RealS1Trainer.__new__(RealS1Trainer)
    trainer.schedule = main_schedule()
    trainer.schedule.spent['main'] = 100100000
    finish_pair(trainer.schedule)
    trainer.run_dir, trainer.segment_id = tmp_path, 'test'
    trainer.candidate_path, trainer.candidate_sha256 = tmp_path/'candidate.pt', 'a'*64
    trainer.env = None
    trainer.evidence_hooks = SimpleNamespace(video=lambda **kwargs: None)
    trainer.evaluator = lambda **kwargs: dict(group_count=75, fixed_denominator=7500,
        passed=False, engineering_valid=True, planning_required=True, report_sha256='f'*64)
    labels = []
    trainer._save = lambda label: (labels.append(label) or tmp_path/label, 'a'*64)
    trainer._record = lambda *args, **kwargs: None
    trainer._begin_evidence_session = lambda *args: None
    trainer._end_evidence_session = lambda **kwargs: None
    def forbidden_collect():
        raise AssertionError('unjudgeable final cannot unlock or collect retry learning')
    trainer._collect_and_train = forbidden_collect
    with pytest.raises(PlanningRequired, match='Q001'):
        trainer.run()
    assert trainer.schedule.recoveries == 0 and trainer.schedule.final_suites_used == 1
    assert 'final_planning_blocked' in labels
    assert 'final_failure_recovery_entry' not in labels
    assert not (tmp_path/'S1_RESTORED.json').exists()


def test_learning_approval_requires_both_course_scopes_and_rejects_s10(tmp_path):
    from src.tasks.velocity.v09.s1_runner import validate_learning_approval
    full = dict(approved=True, stage='S1', approval_type='independent_learning_entry',
        sds_sha256=spec.SDS_SHA256, code_sha256='c'*64, source_sha256=spec.SOURCE_SHA256,
        run_id='formal-run', run_dir=str(tmp_path.resolve()),
        scope=['S1.1_learning','S1.2_learning'])
    assert validate_learning_approval(full, run_dir=tmp_path, run_id='formal-run', code_sha256='c'*64)
    for changed in (dict(full, approval_type='independent_engineering_entry_only'),
                    dict(full, course='S1.0'), dict(full, scope=['S1.0']),
                    dict(full, scope=['S1.1_learning']),
                    dict(full, scope='S1.1_learning S1.2_learning')):
        with pytest.raises(PlanningRequired):
            validate_learning_approval(changed, run_dir=tmp_path, run_id='formal-run', code_sha256='c'*64)


def test_learning_approval_binds_run_and_unique_6400_source(tmp_path):
    from src.tasks.velocity.v09.s1_runner import validate_learning_approval
    full = dict(approved=True, stage='S1', sds_sha256=spec.SDS_SHA256, code_sha256='c'*64,
        source_sha256=spec.SOURCE_SHA256, run_id='formal-run', run_dir=str(tmp_path.resolve()),
        scope=['S1.1_learning','S1.2_learning'])
    for changed in (dict(full, run_id='other-run'), dict(full, run_dir=str(tmp_path/'other')),
                    dict(full, source_sha256='b'*64), dict(full, code_sha256='d'*64)):
        with pytest.raises(PlanningRequired):
            validate_learning_approval(changed, run_dir=tmp_path, run_id='formal-run', code_sha256='c'*64)
    assert validate_learning_approval(full, run_dir=tmp_path/'.', run_id='formal-run', code_sha256='c'*64)


def test_rrun04_zero_learning_resume_reuses_completed_before_without_evidence_replay(tmp_path):
    from types import SimpleNamespace
    from src.tasks.velocity.v09.s1_runner import RealS1Trainer, file_sha256
    schedule = S1Schedule(512)
    initial = tmp_path / 'frozen_initial.pt'
    initial.write_bytes(b'fixed initial checkpoint bytes')
    checkpoint_sha = file_sha256(initial)
    folder = tmp_path / 'initial_diagnostic'
    folder.mkdir()
    summary = dict(suite='S1_DIAG', total=1500, checkpoint_sha256=checkpoint_sha,
                   manifest_sha256='m'*64, code_sha256='c'*64)
    summary_path = folder / 'summary.json'
    summary_path.write_text(json.dumps(summary))
    report = dict(sds_sha256=spec.SDS_SHA256, code_sha256='c'*64, folder=str(folder),
        checkpoint=str(initial), checkpoint_sha256=checkpoint_sha, summary=summary,
        summary_file_sha256=file_sha256(summary_path))
    report['report_sha256'] = sha256(report)
    journal = ScheduleJournal(tmp_path / 's1_schedule.jsonl')
    journal.append(schedule, event='Q008_before_evidence', report=report)
    obligations = {name: dict(closed=True, bound_checkpoint_sha256=checkpoint_sha)
                   for name in ('diag_before','video_init')}
    obligations['diag_before']['manifest_sha256'] = 'm'*64
    obligations['video_6400'] = dict(closed=True, bound_checkpoint_sha256=spec.SOURCE_SHA256)
    class Ledger:
        def snapshot(self):
            return dict(stages={'S1': {'obligations': obligations}}, total_control_samples=123456)
        def reserve(self, **kwargs):
            raise AssertionError('completed before evidence must not be reserved/billed again')
        def commit(self, **kwargs):
            raise AssertionError('completed before evidence must not be billed twice')
    trainer = RealS1Trainer.__new__(RealS1Trainer)
    trainer.schedule, trainer.journal, trainer.ledger = schedule, journal, Ledger()
    trainer.code_sha256, trainer.before_diagnostic = 'c'*64, None
    trainer.run_dir, trainer.env = tmp_path, None
    events = []
    trainer._record = lambda event, **details: events.append((event,details))
    trainer._save = lambda label: (tmp_path/'resumed_entry.pt','a'*64)
    def forbidden(*args, **kwargs):
        raise AssertionError('closed before diagnostic/video must not rerun physics')
    trainer.evidence_hooks = SimpleNamespace(diagnostic=forbidden, video=forbidden)
    trainer._begin_evidence_session = forbidden
    trainer._end_evidence_session = lambda **kwargs: None
    class StopBeforeLearning(Exception):
        pass
    def stop():
        raise StopBeforeLearning
    trainer._collect_and_train = stop
    with pytest.raises(StopBeforeLearning):
        trainer.run()
    assert trainer.before_diagnostic == report
    assert schedule.spent['adapt'] == 0 and trainer.ledger.snapshot()['total_control_samples'] == 123456
    assert any(event == 'Q008_before_evidence_reused' for event, _ in events)
    initial.write_bytes(b'tampered old referenced checkpoint')
    with pytest.raises(ValueError, match='changed'):
        trainer._reuse_before_evidence()
