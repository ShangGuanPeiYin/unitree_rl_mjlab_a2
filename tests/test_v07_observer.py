"""Read-only audit behavior: real files, independent counts, no simulator."""
import importlib.util
import json
from pathlib import Path

import pytest

OBSERVER = Path(__file__).resolve().parents[1] / 'logs/a2_ideal_stairs_v07/implementation/coordination/v07_observer.py'


def module():
    assert OBSERVER.is_file(), 'v07 read-only observer is not implemented'
    spec = importlib.util.spec_from_file_location('_v07_readonly_observer', OBSERVER)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture_run(tmp_path):
    run = tmp_path / 'production'
    write(run / 'contract.json', dict(recipe='0.7-d22-eval-regression-r1',
        source_sha256='a41963a8bf5e6f1d4810b0a0c912ac65747d61bd481ea3432ef638a99e076e64',
        phase_budgets=dict(F0=1000, H=12000, D=21000, L=6000, V=4000, R=4000)))
    write(run / 'recovery_v07/receipt.json', dict(resume_kind='designated_4817',
        generation=73, generation_updates=0, resume_from_update=4817,
        actual_learning=dict(update=4817, optimizer_step=96340, adam_steps=96340,
                             control_samples=118382592, physics_substeps=473530368)))
    for update, substage, generation in [(4818, 'D25', 73), (4819, 'D22', 74), (4820, 'D22', 74)]:
        write(run / f'commit_{update:06}.json', dict(update=update, committed=True,
            phase='D', substage=substage, generation=generation,
            spent=dict(F0=200, H=2500, D=2117 + update - 4817, L=0, V=0, R=0)))
    return run


def test_contiguous_stage_accounting_feasible_budget_and_idempotent_reads(tmp_path):
    target = module(); run = fixture_run(tmp_path); audit = tmp_path / 'audit'
    before = {str(p.relative_to(run)): p.read_bytes() for p in run.rglob('*') if p.is_file()}
    state = target.observe(run, audit)
    assert state['last_update'] == 4820
    assert state['stage_counts_segment'] == dict(D25=1, D22=2)
    assert state['generation_counts'] == {'73': 1, '74': 2}
    row = json.loads((audit / 'progress.jsonl').read_text().splitlines()[-1])
    assert row['active_branch_cap'] == 48000 and row['active_remaining'] == 43180
    assert row['phase_cap_upper_bound_remaining'] == 32880
    assert row['active_control_samples'] == 118456320
    assert row['active_adam_updates'] == 96400 and row['formal_billed_updates'] == 4920
    target.observe(run, audit)
    assert len((audit / 'progress.jsonl').read_text().splitlines()) == 1
    assert before == {str(p.relative_to(run)): p.read_bytes() for p in run.rglob('*') if p.is_file()}
    write(run / 'stop.json', {'reason': 'user_requested_stop'})
    target.observe(run, audit)
    span = json.loads((audit / 'd22_spans.jsonl').read_text().splitlines()[-1])
    assert span['first_billed_update'] == 4819 and span['last_billed_update'] == 4820
    assert span['billed_updates'] == 2 and span['closed'] is True


def test_refuses_in_run_audit_output_and_a_missing_billed_update(tmp_path):
    target = module(); run = fixture_run(tmp_path)
    with pytest.raises(ValueError, match='outside'):
        target.observe(run, run / 'audit')
    assert not (run / 'audit').exists()
    (run / 'commit_004819.json').unlink()
    with pytest.raises(ValueError, match='gap'):
        target.observe(run, tmp_path / 'gap_audit')


def test_retention_alarm_is_summary_only_and_below_twelve_is_not_perfect_success():
    target = module(); table = 'e' * 64
    rows = [dict(c=c, total=5, case_pass_count=n, case_pass_required=18)
            for c, n in zip((.3, .5, .6), (2, 2, 1))]
    gate = dict(summary=dict(per_speed=rows, gate_eligible=False, case_table_sha256=table),
                gate=dict(case_table_sha256=table, failed_cases=10, passed=False))
    retention = dict(case_table_sha256=table, failed_cases=10, streak=0, stop=False)
    result = target.interpret(gate, retention, 1, (.3, .5, .6))
    assert result['evidence_level'] == 'summary_only'
    assert result['failed_cases'] == 10 and result['streak_after'] == 0
    assert result['retention_alarm'] is False and result['raw_passed'] is False


@pytest.mark.parametrize('batch_mode', ['one_batch', 'separate_reads'])
def test_real_null_substages_have_distinct_json_stage_keys_and_close_spans(tmp_path, batch_mode):
    target = module(); run = fixture_run(tmp_path); audit = tmp_path / 'audit'
    if batch_mode == 'separate_reads': target.observe(run, audit)
    for update, phase, substage, generation in [(4821, 'D', 'D20', 75),
            (4822, 'V', None, 76), (4823, 'R', None, 77)]:
        spent = dict(F0=200, H=2500, D=2121, L=0,
                     V=int(update >= 4822), R=int(update >= 4823))
        write(run / f'commit_{update:06}.json', dict(update=update, committed=True,
            phase=phase, substage=substage, generation=generation, spent=spent))
        if batch_mode == 'separate_reads': target.observe(run, audit)
    before = {str(p.relative_to(run)): p.read_bytes() for p in run.rglob('*') if p.is_file()}
    state = target.observe(run, audit)
    assert state['stage_counts_segment'] == dict(D25=1, D22=2, D20=1, V=1, R=1)
    assert state['last_update'] == 4823 and state['open_spans'] == {}
    assert json.loads((audit / 'observer_state.json').read_text()) == state
    rows = (audit / 'progress.jsonl').read_bytes()
    target.observe(run, audit)
    assert (audit / 'progress.jsonl').read_bytes() == rows
    for stage, start, end in [('d22', 4819, 4820), ('d20', 4821, 4821)]:
        spans = (audit / (stage + '_spans.jsonl')).read_text().splitlines()
        assert len(spans) == 1
        span = json.loads(spans[0])
        assert span['first_billed_update'] == start and span['last_billed_update'] == end
        assert span['billed_updates'] == end - start + 1 and span['closed'] is True
    latest = json.loads(rows.splitlines()[-1])
    assert latest['active_control_samples'] == 118382592 + 6 * 24576
    assert latest['active_adam_updates'] == 96340 + 6 * 20
    assert before == {str(p.relative_to(run)): p.read_bytes() for p in run.rglob('*') if p.is_file()}
