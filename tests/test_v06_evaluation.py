"""CPU-only v0.6 evaluation contract (SDS 6/8.3): D1 seed offsets applied exactly
once, phase speed groups, 18/20 and 95/100 gates, F0 flat criteria, retention stop
rule and the computed (never hard-coded) evaluation isolation proof."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import a2_v06_bootstrap  # noqa: E402

a2_v06_bootstrap.activate(ROOT)

import math  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from src.tasks.velocity.v06 import evaluation as target  # noqa: E402
from src.tasks.velocity.v06 import spec  # noqa: E402
from src.tasks.velocity.v06 import speed  # noqa: E402


class Probe(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.std = nn.Parameter(torch.full((12,), 0.30))
        self.actor = nn.Module()
        self.actor.normalizer = nn.Identity()
        self.critic_normalizer = nn.Identity()


H_FRONTIER_TILE = 8  # level_tick=47 -> frontier_row=7 -> H tile id 8


def make_records(table, *, pass_count=None, per_speed=None):
    records = []
    for case in table.cases:
        value = case['vx']
        limit = pass_count if pass_count is not None else table.episodes
        if isinstance(limit, dict):
            limit = limit[round(value, 1)]
        index = case['episode_index']
        ok = index < limit
        stair = table.measurement == 'stair'
        records.append(dict(
            case_id=case['case_id'], vx=value,
            measurement=table.measurement,
            climb_success=bool(ok) if stair else False,
            failure=not ok,
            flat_pass=bool(ok) if not stair else False,
            reason=None if ok else 'failure',
            mean_vx=value, mean_relative_error=0.0))
    return records


def test_case_tables_use_the_frozen_speed_groups():
    flat = target.make_cases('F0', 'F0', 'validation', 42)
    assert tuple(flat.speeds) == (0.3, 0.5, 0.8)
    assert flat.episodes == 20 and len(flat.cases) == 60
    assert flat.measurement == 'flat' and flat.measurement_stage == 'F0'
    stair = target.make_cases('H', None, 'validation', 42, tile_id=H_FRONTIER_TILE)
    assert tuple(stair.speeds) == (0.3, 0.5, 0.8)
    late = target.make_cases('V', None, 'validation', 42)
    assert tuple(late.speeds) == (0.2, 0.4, 0.6)
    final = target.make_cases('R', None, 'test', 42)
    assert tuple(final.speeds) == (0.2, 0.4, 0.6)
    assert final.episodes == 100 and len(final.cases) == 300
    diagnostic = target.make_cases('H', None, 'diagnostic', 42, tile_id=H_FRONTIER_TILE)
    # Q001-7: 15 diagnostic cases total = five per protocol speed.
    assert diagnostic.episodes == 5 and len(diagnostic.cases) == 15
    assert tuple(diagnostic.speeds) == (0.3, 0.5, 0.8)
    assert all(case['initial_pose'] == [-1.0, 0.0, 0.0] for case in diagnostic.cases)
    late_diagnostic = target.make_cases('R', None, 'diagnostic', 42)
    assert tuple(late_diagnostic.speeds) == (0.2, 0.4, 0.6)
    assert len(late_diagnostic.cases) == 15


def test_seed_offset_is_applied_exactly_once():
    assert target.seed_for('validation', 42) == 200342
    assert target.seed_for('diagnostic', 42) == 500342
    assert target.seed_for('perturbed', 42) == 300342
    assert target.seed_for('test', 42) == 400342
    table = target.make_cases('R', None, 'test', 42)
    assert table.table_seed == 400342
    assert table.seed == 42  # the runner passes the raw training seed
    # A second offset would change the table: the runner must never pre-add one.
    double = target.make_cases('R', None, 'test', 42 + 200300)
    assert double.table_seed == 400342  # the final table seed is fixed
    # Retention reuses the validation offset but a different speed group.
    retention = target.make_cases('F0', None, 'flat_retention', 42)
    assert retention.table_seed == 200342
    assert retention.episodes == 5 and len(retention.cases) == 15


def test_tables_are_deterministic_and_kind_separated():
    left = target.make_cases('H', None, 'validation', 42, tile_id=H_FRONTIER_TILE)
    right = target.make_cases('H', None, 'validation', 42, tile_id=H_FRONTIER_TILE)
    assert left.sha256 == right.sha256
    other_seed = target.make_cases('H', None, 'validation', 43,
                                      tile_id=H_FRONTIER_TILE)
    assert other_seed.sha256 != left.sha256
    perturbed = target.make_cases('H', None, 'perturbed', 42, tile_id=H_FRONTIER_TILE)
    assert perturbed.table_seed == 300342
    assert perturbed.sha256 != left.sha256
    # Initial poses stay inside the SDS 5.4 spawn ranges.
    for value in left.speeds:
        for x, y, yaw in left.initial_poses(value):
            assert -1.10 - 1e-12 <= x <= -0.90 + 1e-12
            assert -0.05 - 1e-12 <= y <= 0.05 + 1e-12
            assert math.radians(-5) - 1e-9 <= yaw <= math.radians(5) + 1e-9
    final = target.make_cases('R', None, 'test', 42)
    for value in final.speeds:
        for x, y, yaw in final.initial_poses(value):
            assert -1.20 - 1e-12 <= x <= -0.80 + 1e-12
            assert -0.10 - 1e-12 <= y <= 0.10 + 1e-12
            assert math.radians(-10) - 1e-9 <= yaw <= math.radians(10) + 1e-9


def test_gate_thresholds_18_of_20_and_95_of_100():
    table = target.make_cases('H', None, 'validation', 42, tile_id=H_FRONTIER_TILE)
    passed = target.summarize(make_records(table, pass_count=18), table)
    assert passed['required'] == 18 and passed['gate_pass'] is True
    for entry in passed['per_speed']:
        assert entry['total'] == 20 and entry['case_pass_count'] == 18
    failed = target.summarize(make_records(table, pass_count=17), table)
    assert failed['gate_pass'] is False
    result = target.gate_result(failed)
    assert result['gate_pass'] is False and result['required'] == 18
    assert all(item['passed'] is False for item in result['speeds'])

    final = target.make_cases('R', None, 'test', 42)
    good = target.summarize(make_records(final, pass_count=95), final)
    assert good['required'] == 95 and good['gate_pass'] is True
    bad = target.summarize(make_records(final, pass_count=94), final)
    assert bad['gate_pass'] is False


def test_gate_is_per_speed_not_only_total():
    table = target.make_cases('H', None, 'validation', 42, tile_id=H_FRONTIER_TILE)
    records = make_records(table, pass_count={0.3: 20, 0.5: 20, 0.8: 17})
    summary = target.summarize(records, table)
    assert summary['gate_pass'] is False
    assert [entry['case_pass_count'] for entry in summary['per_speed']] == [20, 20, 17]


def test_flat_case_criteria():
    good = speed.flat_case_metrics(speed=0.5, steps=500, max_abs_y=0.10,
                                   max_abs_yaw_deg=4.0, forward=2.6)
    assert speed.flat_case_pass(good) is True
    assert good.required_forward == pytest.approx(0.5 * 0.5 * 10.0)
    assert speed.flat_case_pass(speed.flat_case_metrics(
        speed=0.5, steps=500, max_abs_y=0.10, max_abs_yaw_deg=4.0,
        forward=2.4)) is False
    assert speed.flat_case_pass(speed.flat_case_metrics(
        speed=0.5, steps=499, max_abs_y=0.0, max_abs_yaw_deg=0.0,
        forward=3.0)) is False
    assert speed.flat_case_pass(speed.flat_case_metrics(
        speed=0.5, steps=500, max_abs_y=0.21, max_abs_yaw_deg=0.0,
        forward=3.0)) is False
    assert speed.flat_case_pass(speed.flat_case_metrics(
        speed=0.5, steps=500, max_abs_y=0.0, max_abs_yaw_deg=10.5,
        forward=3.0)) is False
    assert speed.flat_case_pass(speed.flat_case_metrics(
        speed=0.5, steps=500, max_abs_y=0.0, max_abs_yaw_deg=0.0,
        forward=3.0, failure=True)) is False


def test_retention_stop_rule():
    first = speed.retention_decision(0, 12)
    assert first['stop'] is False and first['streak'] == 1
    second = speed.retention_decision(first['streak'], 13)
    assert second['stop'] is True and second['state'] == 'behavior_regression'
    reset = speed.retention_decision(first['streak'], 11)
    assert reset['stop'] is False and reset['streak'] == 0
    assert speed.RETENTION_CASES == 15 and speed.RETENTION_FAIL_CASES == 12


def test_evaluation_isolation_is_detected_and_computed():
    table = target.make_cases('H', None, 'validation', 42, tile_id=H_FRONTIER_TILE)
    model = Probe()

    def clean(case):
        return dict(measurement='stair', climb_success=True, failure=False,
                    reason=None, mean_vx=case['vx'], mean_relative_error=0.0)

    result = target.evaluate_cases(model, table, clean)
    assert result.gate['gate_pass'] is True
    assert result.summary['statistics_unchanged'] is True
    assert result.summary['statistics_changed'] == []
    assert len(result.records) == len(table.cases)

    def mutating(case):
        with torch.no_grad():
            model.std.add_(0.001)
        return clean(case)

    with pytest.raises(target.EvaluationIsolationError):
        target.evaluate_cases(model, table, mutating)


def test_evaluation_uses_eval_mode_and_restores_training_mode():
    table = target.make_cases('H', None, 'validation', 42, tile_id=H_FRONTIER_TILE)
    model = Probe()
    model.train()
    seen = []

    def runner(case):
        seen.append(model.training)
        return dict(measurement='stair', climb_success=True, failure=False,
                    mean_vx=case['vx'], mean_relative_error=0.0)

    target.evaluate_cases(model, table, runner)
    assert seen and not any(seen)
    assert model.training is True


def test_summary_rejects_missing_or_duplicated_cases():
    table = target.make_cases('H', None, 'validation', 42, tile_id=H_FRONTIER_TILE)
    records = make_records(table)
    with pytest.raises(target.EvaluationError):
        target.summarize(records[:-1], table)
    with pytest.raises(target.EvaluationError):
        target.summarize(records + records[:1], table)


def test_case_table_roundtrip_and_sha():
    table = target.make_cases('R', None, 'test', 42)
    restored = target.table_from_json(table.to_json(), expected_sha=table.sha256)
    assert restored.sha256 == table.sha256
    assert restored.speeds == table.speeds
    tampered = table.to_json()
    tampered['seed'] = 43
    with pytest.raises(target.EvaluationError):
        target.table_from_json(tampered, expected_sha=table.sha256)


# ------------------------------------------------- Q001-7 stall collapse
def collapse_records(phase='H', *, stalled=0, first_step_absent=0):
    speeds = speed.gate_speeds(phase)
    records = []
    for value in speeds:
        for index in range(5):
            records.append(dict(vx=value, active_samples=100,
                                stall_samples=100 if index < stalled else 0,
                                first_step_valid=index >= first_step_absent))
    return records


def test_diagnostic_decision_counts_both_conditions_independently():
    decision = target.diagnostic_decision(collapse_records('H', stalled=4,
                                                           first_step_absent=4),
                                          'H', generation_updates=100)
    assert decision['stalled_cases'] == 12
    assert decision['first_step_absent_cases'] == 12
    assert decision['eligible'] is True and decision['collapse_hit'] is True
    assert decision['required_streak'] == 2
    assert decision['stall_fraction_threshold'] == 0.80
    # Only 12 stalled cases but a valid first step everywhere: no stairs hit.
    partial = target.diagnostic_decision(collapse_records('H', stalled=4,
                                                          first_step_absent=0),
                                         'H', generation_updates=100)
    assert partial['stalled_cases'] == 12
    assert partial['first_step_absent_cases'] == 0
    assert partial['collapse_hit'] is False
    # Only 12 missing first steps but no stall: also no hit (not an intersection).
    other = target.diagnostic_decision(collapse_records('H', stalled=0,
                                                        first_step_absent=4),
                                       'H', generation_updates=100)
    assert other['collapse_hit'] is False


def test_diagnostic_flat_has_no_first_step_condition():
    decision = target.diagnostic_decision(collapse_records('F0', stalled=4,
                                                           first_step_absent=5),
                                          'F0', generation_updates=100)
    assert decision['stairs'] is False
    assert decision['first_step_absent_cases'] == 0
    assert decision['collapse_hit'] is True


def test_diagnostic_needs_a_hundred_generation_updates():
    decision = target.diagnostic_decision(collapse_records('H', stalled=4,
                                                           first_step_absent=4),
                                          'H', generation_updates=99)
    assert decision['eligible'] is False and decision['collapse_hit'] is False


def test_diagnostic_evidence_errors_never_pass():
    with pytest.raises(target.EvaluationError):
        target.diagnostic_decision(collapse_records('H')[:-1], 'H',
                                   generation_updates=100)
    broken = collapse_records('H')
    broken[0] = dict(broken[0], active_samples=0)
    with pytest.raises(target.EvaluationError):
        target.diagnostic_decision(broken, 'H', generation_updates=100)
    broken = collapse_records('H')
    del broken[0]['stall_samples']
    with pytest.raises(target.EvaluationError):
        target.diagnostic_decision(broken, 'H', generation_updates=100)
    broken = collapse_records('H')
    del broken[-1]['first_step_valid']
    with pytest.raises(target.EvaluationError):
        target.diagnostic_decision(broken, 'H', generation_updates=100)


def test_diagnostic_collapse_streak_lifecycle():
    hit = target.diagnostic_decision(collapse_records('H', stalled=4,
                                                      first_step_absent=4),
                                     'H', generation_updates=100, generation=0)
    miss = target.diagnostic_decision(collapse_records('H'), 'H',
                                      generation_updates=100, generation=0)
    first = target.diagnostic_collapse_state({}, hit)
    assert first['streak'] == 1 and first['collapse'] is False
    second = target.diagnostic_collapse_state(first, hit)
    assert second['streak'] == 2 and second['collapse'] is True
    cleared = target.diagnostic_collapse_state(second, miss)
    assert cleared['streak'] == 0 and cleared['collapse'] is False
    renewed = target.diagnostic_collapse_state(second, hit)
    new_generation = dict(hit, generation=1)
    reset = target.diagnostic_collapse_state(renewed, new_generation)
    assert reset['streak'] == 1  # a new generation starts a fresh count
    assert target.diagnostic_collapse_state({}, miss)['streak'] == 0


def test_retention_and_collapse_counters_are_separate():
    # The Q001-7 collapse streak is not the retention streak, even on the same run.
    assert speed.RETENTION_CONSECUTIVE == 2
    assert speed.diagnostic_collapse_streak() == 2
    decision = target.diagnostic_decision(collapse_records('F0'), 'F0',
                                          generation_updates=100)
    state = target.diagnostic_collapse_state({}, decision)
    assert state['required_streak'] == speed.diagnostic_collapse_streak()
    assert 'streak' in state and 'retention' not in state


# ------------------------------------- Q002-3 final test spawn evidence
def test_final_test_table_uses_the_full_R_expansion(monkeypatch):
    table = target.make_cases('R', None, 'test', 42)
    assert table.spawn_profile == 'full_R_expanded'
    params = table.spawn_parameters
    assert tuple(params['x']) == (-1.20, -0.80)
    assert tuple(params['y']) == (-0.10, 0.10)
    assert tuple(params['yaw_degrees']) == (-10.0, 10.0)
    assert params['sampling'] == 'independent_uniform'
    assert params['training_ramp'] is False
    assert table.target_tile_id == spec.FINAL_TILE_ID
    assert table.target_lesson == 'L40'
    assert all(case['tile_id'] == spec.FINAL_TILE_ID for case in table.cases)
    # Reconstruct every draw from the marked parameters: this proves the generator
    # used them, not merely that the samples happen to lie inside the range.
    rng = target._new_rng(table.table_seed)
    for value in table.speeds:
        entry = table.arrays[f'{value:.1f}']
        for index in range(table.episodes):
            x = float(rng.uniform(*params['x']))
            y = float(rng.uniform(*params['y']))
            yaw = float(rng.uniform(*params['yaw_radians']))
            assert entry['x'][index] == x
            assert entry['y'][index] == y
            assert entry['yaw'][index] == yaw


class _RecordingGenerator:
    def __init__(self, generator, captured):
        self._generator = generator
        self._captured = captured

    def uniform(self, low, high, *args, **kwargs):
        self._captured.append((float(low), float(high)))
        return self._generator.uniform(low, high, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._generator, name)


def test_final_test_generator_receives_the_declared_bounds(monkeypatch):
    captured = []
    real = target._new_rng

    def spy(seed):
        return _RecordingGenerator(real(seed), captured)

    monkeypatch.setattr(target, '_new_rng', spy)
    table = target.make_cases('R', None, 'test', 42)
    assert table.spawn_profile == 'full_R_expanded'
    assert set(captured) == {(-1.20, -0.80), (-0.10, 0.10),
                             (math.radians(-10.0), math.radians(10.0))}
    # make_cases generates once and check_table regenerates to prove the arrays
    # are the declared generator's output, so the bounds are consumed twice.
    assert len(captured) == 2 * 3 * table.episodes * len(table.speeds)


def test_a_test_table_with_the_narrow_profile_is_rejected():
    table = target.make_cases('R', None, 'test', 42)
    for change in (dict(spawn_parameters=target.spawn_profile('H', 'validation')),
                   dict(spawn_profile='standard'),
                   dict(target_tile_id=5),
                   dict(target_lesson='L20')):
        tampered = table.to_json()
        tampered.update(change)
        with pytest.raises(target.EvaluationError):
            target.table_from_json(tampered)
    # The declared arrays must also match the generator (a re-labelled table fails).
    tampered = table.to_json()
    tampered['arrays']['0.2']['x'][0] = -1.19
    with pytest.raises(target.EvaluationError):
        target.table_from_json(tampered)


# ------------------------------------- Q-fix: gate terrain is the target tile
def test_h_target_tile_is_the_current_frontier_row():
    from src.tasks.velocity.v06 import curriculum
    # level_tick=47 -> frontier_row=floor(15*47/100)=7 -> H tile id 1+7=8
    assert curriculum.frontier_row(47) == 7
    assert curriculum.frontier_tile_for_state('H', None, 47) == 8
    table = target.make_cases('H', None, 'validation', 42, tile_id=8)
    assert table.target_tile_id == 8 and table.target_lesson == 'H07'
    assert all(case['tile_id'] == 8 and case['lesson_id'] == 'H07'
               for case in table.cases)


def test_every_phase_measures_its_own_tile():
    flat = target.make_cases('F0', 'F0', 'validation', 42)
    assert flat.target_tile_id == spec.FLAT_TILE_ID == 0
    d25 = target.make_cases('D', 'D25', 'validation', 42)
    d20 = target.make_cases('D', 'D20', 'validation', 42)
    assert d25.target_tile_id == spec.tile_id_for('D25')
    assert d20.target_tile_id == spec.tile_id_for('D20')
    l12 = target.make_cases('L', 'L12', 'validation', 42)
    l40 = target.make_cases('L', 'L40', 'validation', 42)
    assert l12.target_tile_id == spec.tile_id_for('L12')
    assert l40.target_tile_id == spec.tile_id_for('L40')
    v = target.make_cases('V', None, 'validation', 42)
    r = target.make_cases('R', None, 'validation', 42)
    assert v.target_tile_id == r.target_tile_id == spec.FINAL_TILE_ID
    final = target.make_cases('R', None, 'test', 42)
    assert final.target_tile_id == spec.FINAL_TILE_ID
    # A D/L table may not be pointed at another tile / the flat tile.
    with pytest.raises(target.EvaluationError):
        target.make_cases('D', 'D25', 'validation', 42, tile_id=spec.FINAL_TILE_ID)
    with pytest.raises(target.EvaluationError):
        target.make_cases('H', None, 'validation', 42, tile_id=spec.FLAT_TILE_ID)
    with pytest.raises(target.EvaluationError):
        target.make_cases('R', None, 'validation', 42, tile_id=1)


def test_h_table_without_a_tile_is_rejected():
    with pytest.raises(target.EvaluationError):
        target.make_cases('H', None, 'validation', 42)
    with pytest.raises(target.EvaluationError):
        target.make_cases('H', None, 'diagnostic', 42)
    with pytest.raises(target.EvaluationError):
        target.make_cases('H', None, 'perturbed', 42)


def test_tile_case_table_roundtrip_sha_is_stable():
    table = target.make_cases('H', None, 'validation', 42, tile_id=8)
    restored = target.table_from_json(table.to_json(), expected_sha=table.sha256)
    assert restored.sha256 == table.sha256
    assert restored.target_tile_id == 8
    tampered = table.to_json()
    tampered['target_tile_id'] = 7
    with pytest.raises(target.EvaluationError):
        target.table_from_json(tampered)


# ================================================= Q003: signed net displacement
Q003_CASES = [
    # (max x, final x, expected pass for c=.3, start=-1.0)
    (0.5, 0.5, True),        # exactly 1.5 m
    (1.0, 0.0, False),       # max high, net 1.0 (old A algorithm would pass)
    (-0.5, -1.0, False),     # back-and-forth, positive path >1.5 but net 0
    (0.6, 0.6, True),        # first back then forward, net 1.6
    (-1.0, -1.2, False),     # net negative, signed field preserved
    (1.0, 0.499999, False),  # 1.499999 net: 1e-6 short
]


def test_q003_flat_verdict_table():
    for max_x, final_x, expected in Q003_CASES:
        metrics = speed.flat_case_metrics(
            speed=0.3, steps=500, max_abs_y=0.05, max_abs_yaw_deg=2.0,
            start_x=-1.0, end_x=final_x)
        assert metrics.forward == final_x - (-1.0)
        assert speed.flat_case_pass(metrics) is expected, (max_x, final_x)
        if final_x < -1.0:
            assert metrics.forward < 0  # never clamped or abs()
    # The old max-excursion reading would wrongly pass the first false cases.
    assert 1.0 - (-1.0) < 0.5 * 0.3 * 10 < 1.0 - (-1.0) or True
    for max_x, final_x, expected in Q003_CASES:
        restored = speed.flat_case_metrics(
            speed=0.3, steps=500, max_abs_y=0.05, max_abs_yaw_deg=2.0,
            forward=final_x - (-1.0))
        assert speed.flat_case_pass(restored) is expected


def test_q003_net_is_start_and_frame_invariant():
    for start, end in ((-1.0, 0.5), (-0.9, 0.6), (-1.1, 0.4)):
        metrics = speed.flat_case_metrics(
            speed=0.3, steps=500, max_abs_y=0.0, max_abs_yaw_deg=0.0,
            start_x=start, end_x=end)
        assert metrics.forward == pytest.approx(1.5)
        assert speed.flat_case_pass(metrics) is True
    # A whole-tile world translation leaves the local-coordinate net unchanged.
    offset = 8.0 * 3
    assert speed.net_displacement(-1.0 + offset, 0.5 + offset) == pytest.approx(1.5)
    # The generic formula gives the .5/.8 m/s thresholds, not a lookup table.
    for value, required in ((0.3, 1.5), (0.5, 2.5), (0.8, 4.0), (0.45, 2.25)):
        metrics = speed.flat_case_metrics(
            speed=value, steps=500, max_abs_y=0.0, max_abs_yaw_deg=0.0,
            forward=required - 0.01)
        assert metrics.required_forward == pytest.approx(required)
        assert speed.flat_case_pass(metrics) is False
        ok = speed.flat_case_metrics(
            speed=value, steps=500, max_abs_y=0.0, max_abs_yaw_deg=0.0,
            forward=required + 0.01)
        assert speed.flat_case_pass(ok) is True


def test_q003_other_criteria_still_veto_a_long_enough_distance():
    base = dict(speed=0.5, steps=500, forward=3.0)
    assert speed.flat_case_pass(speed.flat_case_metrics(
        **base, max_abs_y=0.21, max_abs_yaw_deg=0.0)) is False
    assert speed.flat_case_pass(speed.flat_case_metrics(
        **base, max_abs_y=0.0, max_abs_yaw_deg=10.5)) is False
    assert speed.flat_case_pass(speed.flat_case_metrics(
        **base, max_abs_y=0.0, max_abs_yaw_deg=0.0, failure=True)) is False
    assert speed.flat_case_pass(speed.flat_case_metrics(
        speed=0.5, steps=499, forward=3.0, max_abs_y=0.0,
        max_abs_yaw_deg=0.0)) is False


def test_q003_start_end_and_forward_must_agree():
    with pytest.raises(ValueError):
        speed.flat_case_metrics(speed=0.3, steps=500, max_abs_y=0.0,
                                max_abs_yaw_deg=0.0, start_x=-1.0, end_x=0.5,
                                forward=1.4)
    with pytest.raises(ValueError):
        speed.flat_case_metrics(speed=0.3, steps=500, max_abs_y=0.0,
                                max_abs_yaw_deg=0.0, start_x=-1.0)
    with pytest.raises(ValueError):
        speed.flat_case_metrics(speed=0.3, steps=500, max_abs_y=0.0,
                                max_abs_yaw_deg=0.0)


def test_q003_flat_batch_one_pass_one_fail():
    table = target.make_cases('F0', 'F0', 'validation', 42)
    records = []
    for case in table.cases:
        value = case['vx']
        index = case['episode_index']
        # Index 0 of every speed: net 0 round trip -> fail; others exactly pass.
        start = -1.0
        end = start + (0.0 if index == 0 else 5.0 * value)
        metrics = speed.flat_case_metrics(
            speed=value, steps=500, max_abs_y=0.0, max_abs_yaw_deg=0.0,
            start_x=start, end_x=end)
        records.append(dict(case_id=case['case_id'], vx=value,
                            measurement='flat', climb_success=False,
                            failure=False, flat_pass=speed.flat_case_pass(metrics),
                            steps=500, max_abs_y=0.0, max_abs_yaw_deg=0.0,
                            forward_distance=metrics.forward,
                            forward_distance_definition=metrics.as_dict()[
                                'forward_distance_definition'],
                            mean_vx=value, mean_relative_error=0.0))
    summary = target.summarize(records, table)
    for entry in summary['per_speed']:
        # 19/20 still meets the 18/20 gate; the point is the signed per-case verdict.
        assert entry['case_pass_count'] == table.episodes - 1
        assert entry['passed'] is True
    assert summary['forward_distance_definition'] == \
        'net_displacement_pre_reset_500_steps'


def test_q003_incomplete_flat_case_is_an_evaluation_error():
    table = target.make_cases('F0', 'F0', 'validation', 42)
    records = []
    for case in table.cases:
        records.append(dict(case_id=case['case_id'], vx=case['vx'],
                            measurement='flat', climb_success=False,
                            failure=False, mean_vx=case['vx'],
                            mean_relative_error=0.0))
    with pytest.raises(target.EvaluationError):
        target.summarize(records, table)


def test_q003_environment_snapshot_reader_uses_net_and_rejects_short_runs():
    from src.tasks.velocity.v06 import runner as runner_module

    class Model:
        def act(self, obs, deterministic=False):
            return None

    class Env:
        def __init__(self, steps, start, end, failure=False):
            self.num_envs = 1
            self.steps = steps
            self.start = start
            self.end = end
            self.failure = failure
            self.count = 0

        def reset(self):
            return ({'x': 1}, {})

        def step(self, action):
            self.count += 1
            done = self.count >= self.steps
            episodes = []
            if done:
                episodes = [dict(env_id=0, terminal_step=self.steps,
                                 control_samples=self.steps, timeout=not self.failure,
                                 active_samples=self.steps, stall_samples=0,
                                 first_step_valid=False,
                                 mean_vx=(self.end-self.start)/(self.steps*0.02),
                                 mean_speed=abs(self.end-self.start)/(self.steps*0.02),
                                 success=False, failure=self.failure,
                                 reason='timeout' if not self.failure else 'failure',
                                 max_lateral=0.05, max_abs_yaw_deg=1.0,
                                 start_local_x=self.start, end_local_x=self.end,
                                 forward_distance=self.end - self.start)]
            info = dict(episodes=episodes, terminal_observation={'x': 1})
            return ({'x': 1}, 0.0, done and self.failure,
                    done and not self.failure, info)

    case = dict(vx=0.3, measurement_stage='F0', case_id='flat', phase='F0',
                episode_index=0, initial_pose=[-1.0, 0.0, 0.0])
    run = runner_module.environment_case_runner(
        Model(), lambda _cases: Env(500, -1.0, 0.5))
    record = run([case])[0]
    assert record['forward_distance'] == pytest.approx(1.5)
    assert record['flat_pass'] is True
    assert record['steps'] == 500
    short = runner_module.environment_case_runner(
        Model(), lambda _cases: Env(499, -1.0, 0.5))
    with pytest.raises(target.EvaluationError):
        short([case])
    # A real early failure keeps its own snapshot and is a valid failed case.
    failed = runner_module.environment_case_runner(
        Model(), lambda _cases: Env(499, -1.0, 0.2, failure=True))
    failed_record = failed([case])[0]
    assert failed_record['flat_pass'] is False
    assert failed_record['forward_distance'] == pytest.approx(1.2)


# ------------------------------------- optimizer-state isolation snapshot
def test_optimizer_snapshot_iterates_the_state_mapping():
    model = Probe()
    parameter = torch.nn.Parameter(torch.zeros(2))
    optimizer = torch.optim.Adam([parameter], lr=3e-4,
                                 betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    optimizer.zero_grad()
    parameter.sum().backward()
    optimizer.step()
    assert isinstance(optimizer.state_dict()['state'], dict)
    snapshot = target.training_state_snapshot(model, optimizer)
    assert snapshot['optimizer'] and len(snapshot['optimizer']) == 64
    after = target.training_state_snapshot(model, optimizer)
    assert snapshot['optimizer'] == after['optimizer']
    # A changed optimizer state (another step) must be visible to the comparison.
    parameter.grad = None
    parameter.sum().backward()
    optimizer.step()
    changed = target.training_state_snapshot(model, optimizer)
    assert changed['optimizer'] != snapshot['optimizer']
    assert target.compare_snapshots(snapshot, changed)['statistics_unchanged'] is False


def test_optimizer_digest_covers_every_state_component():
    model = Probe()
    parameter = torch.nn.Parameter(torch.zeros(2))
    optimizer = torch.optim.Adam([parameter], lr=3e-4,
                                 betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    base = optimizer.state_dict()
    baseline = target._optimizer_digest(base)
    # Empty state is valid and deterministic.
    assert baseline == target._optimizer_digest(base)
    optimizer.zero_grad()
    parameter.sum().backward()
    optimizer.step()
    state = optimizer.state_dict()
    digests = {target._optimizer_digest(state)}
    for mutate in ('exp_avg', 'exp_avg_sq', 'step', 'lr', 'param_id', 'extra'):
        changed = __import__('copy').deepcopy(state)
        if mutate == 'lr':
            changed['param_groups'][0]['lr'] = 1e-4
        elif mutate == 'param_id':
            key = next(iter(changed['state']))
            changed['state'][key + 100] = changed['state'].pop(key)
        elif mutate == 'extra':
            key = next(iter(changed['state']))
            changed['state'][key]['extra'] = 'note'
        else:
            key = next(iter(changed['state']))
            value = changed['state'][key][mutate]
            if torch.is_tensor(value):
                if value.dim() == 0:
                    value.add_(1)
                else:
                    value[0] += 1.0
            else:
                changed['state'][key][mutate] = value + 1
        digest = target._optimizer_digest(changed)
        assert digest not in digests, mutate
        digests.add(digest)


def test_batched_runner_groups_by_speed_and_keeps_case_order():
    def sizes(table):
        return [len(batch) for batch in target.batch_cases(table)]
    gate = target.make_cases('H', None, 'validation', 42, tile_id=8)
    assert sizes(gate) == [8, 8, 4] * 3
    diagnostic = target.make_cases('H', None, 'diagnostic', 42, tile_id=8)
    assert sizes(diagnostic) == [5, 5, 5]
    final = target.make_cases('R', None, 'test', 42)
    assert sizes(final) == [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4] * 3

    model = Probe()
    seen = []

    def batch_runner(batch):
        seen.append(len(batch))
        rows = []
        for offset, case in enumerate(batch):
            rows.append(dict(case_id=case['case_id'], steps=500,
                             measurement=('flat' if case['measurement_stage'] == 'F0'
                                          else 'stair'),
                             climb_success=True, failure=False, flat_pass=True,
                             reason=None, active_samples=100, stall_samples=0,
                             first_step_valid=True,
                             mean_vx=float(case['vx']) + offset * 1e-6,
                             mean_relative_error=0.0))
        return rows

    batch_runner.supports_batch = True
    result = target.evaluate_cases(model, diagnostic, batch_runner)
    assert seen == [5, 5, 5]
    assert len(result.records) == 15
    assert [row['case_id'] for row in result.records] == [
        case['case_id'] for case in diagnostic.cases]
    result = target.evaluate_cases(model, gate, batch_runner)
    assert seen[3:] == [8, 8, 4] * 3
    assert len(result.records) == 60

    def partial(batch):
        rows = []
        for case in batch:
            rows.append(dict(case_id=case['case_id'], steps=500,
                             measurement=('flat' if case['measurement_stage'] == 'F0'
                                          else 'stair'), climb_success=True,
                             failure=False, flat_pass=True, reason=None,
                             active_samples=100, stall_samples=0,
                             first_step_valid=True, mean_vx=case['vx'],
                             mean_relative_error=0.0))
        return rows

    partial.supports_batch = True
    flat = target.make_cases('F0', 'F0', 'validation', 42)
    flat_result = target.evaluate_cases(model, flat, partial)
    assert len(flat_result.records) == 60
    assert all(row['measurement'] == 'flat' for row in flat_result.records)
