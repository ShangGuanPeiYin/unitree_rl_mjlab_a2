"""SDS 9.1 items 13/14/15/16: curriculum windows, EMA, generations and budgets.

Pure-CPU tests of :class:`CurriculumManager`; no environment or GPU is built.
"""
import json

import pytest

from src.tasks.velocity.v06 import curriculum, spec
from src.tasks.velocity.v06.curriculum import (CurriculumManager, CurriculumState,
                                               EpisodeOutcome, TransitionEvent)


def _state(**overrides):
    base = CurriculumState(
        phase='H', substage=None, level_tick=0, generation=0,
        generation_updates=100, last_update=100, cooldown=0,
        spent_by_phase={name: (200 if name == 'H' else 0)
                        for name in curriculum.PHASE_ORDER},
        spent_by_substage={name: 0 for name in curriculum.SUBSTAGE_BUDGETS},
        frontier_tile_id=1,
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def _manager(**overrides):
    return CurriculumManager(num_envs=4, seed=42, state=_state(**overrides).__dict__)


def _outcome(env_id, step, *, success=True, failure=False, timeout=False,
             role='frontier', tile_id=1, score=1.0, generation=None):
    return EpisodeOutcome(env_id=env_id, step=step, role=role, tile_id=tile_id,
                          success=success, failure=failure, timeout=timeout,
                          reason='success' if success else ('fell' if failure else 'timeout'),
                          tracking_score=score, command=0.5, generation=generation)


def _feed_window(manager, step_base, *, success=True, failure=False, score=1.0,
                 role='frontier', generation=None):
    outcomes = []
    for index in range(curriculum.WINDOW):
        outcomes.append(_outcome(index % 4, step_base + index,
                                 success=success and not failure, failure=failure,
                                 timeout=not success and not failure, role=role,
                                 score=score, generation=generation))
    manager.observe(outcomes)


def _gate(manager, update, passed=True):
    manager.on_update(update)
    manager.record_gate(passed=passed, update=update)


def test_level_tick_boundaries_and_terrain_map():
    assert curriculum.frontier_row(0) == 0
    assert curriculum.frontier_row(6) == 0
    assert curriculum.frontier_row(7) == 1
    assert curriculum.frontier_row(99) == 14
    assert curriculum.frontier_row(100) == 15
    for tick in range(101):
        row = curriculum.frontier_row(tick)
        assert row == int(15 * tick / 100)
        assert curriculum.frontier_tile_for_state('H', None, tick) == row + 1
    assert curriculum.frontier_tile_for_state('H', None, 0) == spec.tile_id_for('H00')
    assert curriculum.frontier_tile_for_state('H', None, 100) == spec.tile_id_for('H15')
    assert curriculum.frontier_tile_for_state('D', 'D25', 100) == spec.tile_id_for('D25')
    assert curriculum.frontier_tile_for_state('L', 'L40', 100) == spec.tile_id_for('L40')


def test_role_allocation_is_sds_exact_on_512_and_largest_remainder_otherwise():
    roles = curriculum.allocate_roles(512, 'H')
    counts = {name: roles.count(name) for name in curriculum.ROLE_NAMES}
    assert counts == {'flat': 102, 'frontier': 164, 'near': 123, 'easy': 123}
    assert curriculum.allocate_roles(512, 'F0') == ('flat',) * 512
    small = curriculum.allocate_roles(4, 'H')
    assert {name: small.count(name) for name in curriculum.ROLE_NAMES} == {
        'flat': 1, 'frontier': 1, 'near': 1, 'easy': 1}
    for count in (1, 3, 7, 100, 513):
        allocated = curriculum.allocate_roles(count, 'L')
        assert len(allocated) == count
        assert allocated[:allocated.count('flat')] == ('flat',) * allocated.count('flat')
        assert set(allocated) <= set(curriculum.ROLE_NAMES)


def test_schedule_tiles_follow_frontier_near_easy_rules():
    manager = _manager()
    k0 = manager.schedule()
    assert k0.frontier_tile_id == 1
    assert set(k0.tile_ids) <= {0, 1}
    # k = 20 (L40/V/R): near is T18/T19, easy is T0..T17.
    manager.state.phase = 'L'
    manager.state.substage = 'L40'
    manager.state.level_tick = 100
    manager.state.frontier_tile_id = 21
    assignment = manager.schedule()
    roles = list(assignment.roles)
    assert assignment.frontier_tile_id == 21
    near = {assignment.tile_ids[i] for i, role in enumerate(roles) if role == 'near'}
    easy = {assignment.tile_ids[i] for i, role in enumerate(roles) if role == 'easy'}
    flat = {assignment.tile_ids[i] for i, role in enumerate(roles) if role == 'flat'}
    frontier = {assignment.tile_ids[i] for i, role in enumerate(roles) if role == 'frontier'}
    assert frontier == {21} and flat == {0}
    assert near <= {spec.tile_id_for('L12'), spec.tile_id_for('L20')}
    assert easy <= set(range(1, spec.tile_id_for('D20') + 1))
    # k = 16 (D25): near T14/T15, easy T0..T13.
    manager.state.substage = 'D25'
    manager.state.frontier_tile_id = spec.tile_id_for('D25')
    d25 = manager.schedule()
    roles = list(d25.roles)
    near = {d25.tile_ids[i] for i, role in enumerate(roles) if role == 'near'}
    easy = {d25.tile_ids[i] for i, role in enumerate(roles) if role == 'easy'}
    assert near <= {spec.tile_id_for('H14'), spec.tile_id_for('H15')}
    assert easy <= set(range(1, spec.tile_id_for('H14')))
    # Regression truncates the sampleable set: H row 3 (k = 3) keeps near
    # T1/T2 and easy T0 only, never T3 or above.
    manager.state.phase = 'H'
    manager.state.substage = None
    manager.state.level_tick = 20
    manager.state.frontier_tile_id = spec.tile_id_for('H00') + 3
    small = manager.schedule()
    roles = list(small.roles)
    near = {small.tile_ids[i] for i, role in enumerate(roles) if role == 'near'}
    easy = {small.tile_ids[i] for i, role in enumerate(roles) if role == 'easy'}
    assert near <= {spec.tile_id_for('H00') + 1, spec.tile_id_for('H00') + 2}
    assert easy == {spec.tile_id_for('H00')}
    # V/R fixed frontier k=20.
    for phase in ('V', 'R'):
        manager.state.phase = phase
        manager.state.substage = None
        manager.state.frontier_tile_id = spec.V_R_FRONTIER_TILE_ID
        assignment = manager.schedule()
        assert assignment.frontier_tile_id == spec.V_R_FRONTIER_TILE_ID


def test_fifo_orders_by_end_step_then_env_id_and_consumes_one_window():
    manager = _manager()
    full = [_outcome(env, 100 + index) for index, env in enumerate(range(64))]
    manager.observe(reversed(full))
    assert len(manager.state.episode_fifo) == 64
    entries = manager.state.episode_fifo
    assert [(e['step'], e['env_id']) for e in entries] == sorted(
        (e['step'], e['env_id']) for e in entries)
    assert manager.state.ema['batches'] == 0
    manager.on_update(101)
    assert manager.state.ema['batches'] == 1
    assert manager.state.window_metrics['size'] == 64
    assert manager.state.episode_fifo == []
    # A short window is retained, never dropped.
    manager.observe([_outcome(0, 500)])
    manager.on_update(102)
    assert len(manager.state.episode_fifo) == 1
    assert manager.state.ema['batches'] == 1
    # Two full windows pending: exactly one is consumed per boundary.
    for base in (2000, 3000):
        _feed_window(manager, base)
    assert len(manager.state.episode_fifo) == 129
    manager.on_update(103)
    assert len(manager.state.episode_fifo) == 65
    assert manager.state.ema['batches'] == 2


def test_ema_first_batch_then_097_003():
    manager = _manager()
    _feed_window(manager, 100, success=True, score=1.0)
    manager.on_update(101)
    first = dict(manager.state.ema)
    assert first == dict(s=1.0, f=0.0, q=1.0, batches=1)
    _feed_window(manager, 200, failure=True, score=0.0)
    manager.on_update(102)
    second = dict(manager.state.ema)
    assert second['s'] == pytest.approx(0.97 * 1.0 + 0.03 * 0.0)
    assert second['f'] == pytest.approx(0.97 * 0.0 + 0.03 * 1.0)
    assert second['q'] == pytest.approx(0.97 * 1.0 + 0.03 * 0.0)
    assert second['batches'] == 2


def test_h00_requires_two_complete_gates_before_adaptive_upgrade():
    manager = _manager(level_tick=0)
    _feed_window(manager, 100, success=True, score=1.0)
    manager.on_update(101)
    assert manager.state.ready_streak == 1
    assert manager.state.level_tick == 0
    _gate(manager, 102)
    assert manager.state.gate_streak == 1
    assert manager.state.level_tick == 0
    _gate(manager, 103)
    assert manager.state.gate_streak == 2
    assert manager.state.retention_enabled['h00'] is True
    assert manager.state.retention_qualified_at['h00'] == 103
    assert manager.state.retention_next_due['h00'] == 200
    # Five consecutive passing windows and cooldown == 0 -> one tick.
    manager.state.ready_streak = 0
    manager.state.cooldown = 0
    for index in range(5):
        _feed_window(manager, 1000 + index * 100, success=True, score=1.0)
        manager.on_update(104 + index)
    assert manager.state.level_tick == 1
    assert manager.state.cooldown == curriculum.COOLDOWN
    assert manager.state.ready_streak == 0


def test_upgrade_needs_five_checks_and_cooldown_after_each_level():
    manager = _manager(level_tick=1, gate_streak=2, cooldown=5)
    update = 200
    for index in range(4):
        update += 1
        _feed_window(manager, 1000 + index * 100, success=True, score=1.0)
        manager.on_update(update)
    assert manager.state.level_tick == 1, 'four checks must not upgrade'
    assert manager.state.cooldown == 1
    update += 1
    _feed_window(manager, 1400, success=True, score=1.0)
    manager.on_update(update)
    assert manager.state.level_tick == 2
    assert manager.state.cooldown == curriculum.COOLDOWN
    # A ready streak at 5 is still blocked while the cooldown is positive.
    manager.state.ready_streak = curriculum.ADVANCE['streak']
    manager.state.cooldown = 2
    manager._h_window_decision()
    assert manager.state.level_tick == 2
    manager.state.cooldown = 0
    manager.state.ready_streak = curriculum.ADVANCE['streak']
    manager._h_window_decision()
    assert manager.state.level_tick == 3


def test_regress_two_hard_checks_subtract_three_ticks_and_priority():
    manager = _manager(level_tick=10, gate_streak=2)
    _feed_window(manager, 100, failure=True)
    manager.on_update(201)
    assert manager.state.level_tick == 10
    _feed_window(manager, 200, failure=True)
    manager.on_update(202)
    assert manager.state.level_tick == 7
    assert manager.state.cooldown == curriculum.COOLDOWN
    # Regress wins when both an upgrade and a downgrade are eligible.
    manager = _manager(level_tick=10)
    manager.state.ready_streak = curriculum.ADVANCE['streak']
    manager.state.hard_streak = curriculum.REGRESS['streak']
    manager.state.cooldown = 0
    assert manager._adaptive_allowed()
    manager._h_window_decision()
    assert manager.state.level_tick == 7
    # A downgrade at the lowest level is a no-op but still resets the streak.
    manager = _manager(level_tick=0, gate_streak=2)
    manager.state.hard_streak = curriculum.REGRESS['streak']
    manager._h_window_decision()
    assert manager.state.level_tick == 0
    assert manager.state.hard_streak == 0


def test_row_change_starts_new_generation_and_clears_window_state():
    manager = _manager(level_tick=6, gate_streak=2, cooldown=0)
    manager.state.ema = dict(s=1.0, f=0.0, q=1.0, batches=5)
    manager.state.episode_fifo = [dict(_outcome(0, 1).as_dict())]
    manager.state.ready_streak = curriculum.ADVANCE['streak'] - 1
    _feed_window(manager, 100, success=True, score=1.0)
    generation = manager.state.generation
    events = manager.on_update(201)
    assert manager.state.level_tick == 7
    assert manager.state.generation == generation + 1
    assert any(event.kind == 'generation' for event in events)
    assert manager.state.ema['batches'] == 0
    assert manager.state.episode_fifo == []
    assert manager.state.gate_streak == 0
    assert manager.state.cooldown == curriculum.COOLDOWN
    assert manager.state.frontier_tile_id == spec.tile_id_for('H01')
    # Same row: no generation change, only the cooldown.
    tick = manager.state.level_tick
    manager.state.ready_streak = curriculum.ADVANCE['streak']
    manager.state.cooldown = 0
    manager._h_window_decision()
    assert manager.state.generation == generation + 1
    assert manager.state.level_tick == tick + 1


def test_old_generation_outcomes_are_log_only():
    manager = _manager(level_tick=10)
    generation = manager.state.generation
    manager.state.generation = generation + 1
    _feed_window(manager, 100, success=True, generation=generation)
    assert manager.state.episode_fifo == []
    _feed_window(manager, 200, success=True, generation=generation + 1)
    assert len(manager.state.episode_fifo) == 64
    # Non-frontier roles never enter the window even with the right generation.
    manager.state.episode_fifo = []
    _feed_window(manager, 300, success=True, role='easy')
    assert manager.state.episode_fifo == []


def test_d_substage_advance_two_gates_then_regress_keeps_budget():
    manager = _manager(phase='D', substage='D25', level_tick=100,
                       generation_updates=100, last_update=100)
    manager.state.spent_by_phase = {name: 0 for name in curriculum.PHASE_ORDER}
    manager.state.spent_by_substage = {name: 0 for name in curriculum.SUBSTAGE_BUDGETS}
    _gate(manager, 101)
    assert manager.state.substage == 'D25'
    _gate(manager, 102)
    assert manager.state.substage == 'D25', 'transition commits on the next boundary'
    manager.on_update(103)
    assert manager.state.substage == 'D20'
    assert manager.state.generation == 1
    spent_d25 = manager.state.spent_by_substage['D25']
    assert spent_d25 >= 2
    # Two hard windows regress back to D25 and keep its spent budget.
    _feed_window(manager, 100, failure=True)
    manager.on_update(104)
    _feed_window(manager, 200, failure=True)
    events = manager.on_update(105)
    assert manager.state.substage == 'D25'
    assert manager.state.spent_by_substage['D25'] == spent_d25, 'regress keeps spent'
    assert any(event.kind == 'substage_regress' for event in events)
    # The old pass is not reused: gate streak is zero after the regress and the
    # new generation must first complete 100 updates before gates count again.
    assert manager.state.gate_streak == 0
    manager.on_update(204)
    manager.record_gate(passed=True, update=204)
    assert manager.state.gate_streak == 0
    assert manager.state.gate_log[-1]['reason'] == 'generation_younger_than_100_updates'
    manager.on_update(205)
    manager.record_gate(passed=True, update=205)
    assert manager.state.gate_streak == 1
    assert manager.state.substage == 'D25'


def test_returning_to_exhausted_substage_stops_with_budget_failure():
    manager = _manager(phase='D', substage='D25', level_tick=100,
                       generation_updates=100, last_update=100)
    manager.state.spent_by_substage = {name: 0 for name in curriculum.SUBSTAGE_BUDGETS}
    manager.state.spent_by_phase = {name: 0 for name in curriculum.PHASE_ORDER}
    _gate(manager, 101)
    _gate(manager, 102)
    manager.on_update(103)
    assert manager.state.substage == 'D20'
    manager.state.spent_by_substage['D25'] = curriculum.SUBSTAGE_BUDGETS['D25']
    # Regress to D25, whose budget is now exhausted.
    _feed_window(manager, 100, failure=True)
    manager.on_update(104)
    _feed_window(manager, 200, failure=True)
    events = manager.on_update(105)
    assert manager.state.finished
    assert manager.state.stop_status == 'budget_exhausted'
    assert any(event.kind == 'final' and event.status == 'budget_exhausted'
               for event in events)
    # The exhausted subtage's budget was never zeroed.
    assert manager.state.spent_by_substage['D25'] >= curriculum.SUBSTAGE_BUDGETS['D25']


def test_phase_budget_exhaustion_stops_without_transition():
    manager = _manager(phase='H', level_tick=10)
    budget = curriculum.spec.phase_budget('H')
    manager.state.spent_by_phase['H'] = budget
    events = manager.on_update(200)
    assert manager.state.finished
    assert manager.state.stop_status == 'budget_exhausted'
    assert any(event.kind == 'final' for event in events)
    # Idempotent: further boundaries do nothing.
    assert manager.on_update(201) == []


def test_v_r_gate_streak_start_and_minimum_updates_are_decoupled_from_ramp():
    manager = _manager(phase='V', substage=None, level_tick=100)
    manager.state.spent_by_phase = {name: 0 for name in curriculum.PHASE_ORDER}
    manager.state.spent_by_phase['V'] = 900
    manager.state.last_update = 900
    manager.state.generation_updates = 100
    manager.on_update(900)  # no-op boundary (delta 0)
    manager.record_gate(passed=True, update=900)
    assert manager.state.gate_streak == 0, 'gates before 1000 V updates never count'
    assert manager.state.gate_log[-1]['reason'] == 'v_gate_streak_starts_at_1000'
    manager.on_update(1000)
    manager.record_gate(passed=True, update=1000)
    assert manager.state.gate_streak == 1
    manager.on_update(1100)
    manager.record_gate(passed=True, update=1100)
    assert manager.state.gate_streak == 2
    assert manager.state.phase == 'V', 'V needs 1200 updates before promotion'
    events = manager.on_update(1200)
    assert manager.state.phase == 'R'
    assert manager.state.frontier_tile_id == spec.V_R_FRONTIER_TILE_ID
    assert any(event.kind == 'advance' for event in events)
    # R: streaks start at 400 updates and promotion needs 600.
    manager.on_update(1400)
    manager.record_gate(passed=True, update=1400)
    assert manager.state.gate_streak == 0, 'R gate before 400 R updates never counts'
    manager.on_update(1500)
    manager.record_gate(passed=True, update=1500)
    manager.on_update(1600)
    manager.record_gate(passed=True, update=1600)
    manager.on_update(1700)
    manager.record_gate(passed=True, update=1700)
    assert manager.state.gate_streak == 2
    assert manager.state.phase == 'R' and not manager.state.finished
    events = manager.on_update(1800)
    assert manager.state.finished
    assert manager.state.stop_status == 'complete'
    assert any(event.kind == 'final' and event.status == 'complete' for event in events)


def test_f0_two_gates_advance_and_start_the_flat_retention_schedule():
    manager = _manager(phase='F0', substage=None, level_tick=0, last_update=100,
                       generation_updates=100)
    _gate(manager, 101)
    assert manager.state.phase == 'F0'
    _gate(manager, 102)
    assert manager.state.phase == 'F0'
    manager.on_update(103)  # transition commits on the next boundary
    assert manager.state.phase == 'H'
    assert manager.state.level_tick == curriculum.LEVEL_TICK_MIN
    assert manager.state.retention_enabled['flat'] is True
    assert manager.state.retention_qualified_at['flat'] == 103
    assert manager.state.retention_next_due['flat'] == 200
    assert manager.retention_due(103) is None
    assert manager.retention_due(199) is None
    assert manager.retention_due(200) == ('flat',)
    manager.record_retention(kind='flat', failed_cases=0, update=200)
    assert manager.state.retention_next_due['flat'] == 400


def test_retention_due_returns_every_due_set_in_fixed_order():
    manager = _manager(phase='F0', substage=None, level_tick=0, last_update=0)
    assert manager.retention_due(100) is None
    assert manager.retention_due(200) is None
    manager._qualify_retention('flat', 100)
    assert manager.retention_due(100) is None
    assert manager.retention_due(150) is None
    assert manager.retention_due(200) == ('flat',)
    manager.record_retention(kind='flat', failed_cases=0, update=200)
    # H00 qualifies later; both sets are due on the same 200-update boundary.
    manager._qualify_retention('h00', 150)
    assert manager.retention_due(200) == ('h00',)
    manager.record_retention(kind='h00', failed_cases=0, update=200)
    assert manager.retention_due(399) is None
    assert manager.retention_due(400) == ('flat', 'h00')
    manager.record_retention(kind='flat', failed_cases=0, update=400)
    assert manager.retention_due(400) == ('h00',)
    manager.record_retention(kind='h00', failed_cases=0, update=400)
    assert manager.retention_due(400) is None
    assert manager.retention_due(600) == ('flat', 'h00')
    assert curriculum.retention_gate_kind('flat') == 'flat_retention'
    assert curriculum.retention_gate_kind('h00') == 'stair_retention'


def test_retention_streaks_are_independent_and_stop_per_set():
    manager = _manager(phase='H', last_update=0)
    first = manager.record_retention(kind='flat_retention', failed_cases=12,
                                     update=200)
    assert first['streak'] == 1 and first['stop'] is False
    assert manager.state.retention_streaks['h00'] == 0
    second = manager.record_retention(kind='flat_retention', failed_cases=13,
                                      update=400)
    assert second['streak'] == 2 and second['stop'] is True
    assert manager.state.retention_streaks == {'flat': 2, 'h00': 0}
    # A repeated completion at the same update is idempotent, never double
    # counted (Q002-6 recovery replay).
    replay = manager.record_retention(kind='flat_retention', failed_cases=13,
                                      update=400)
    assert replay['duplicate'] is True and replay['streak'] == 2
    # H00 keeps its own counter: it starts at 1 and does not inherit flat's 2.
    h00 = manager.record_retention(kind='stair_retention', failed_cases=12,
                                   update=600)
    assert h00['streak'] == 1 and h00['stop'] is False
    assert manager.state.retention_streaks == {'flat': 2, 'h00': 1}
    # A passing H00 result clears only H00; flat's stop counter is untouched.
    manager.record_retention(kind='stair_retention', failed_cases=11, update=800)
    assert manager.state.retention_streaks == {'flat': 2, 'h00': 0}
    # ... and a passing flat result clears only flat.
    manager.record_retention(kind='flat_retention', failed_cases=5, update=1000)
    assert manager.state.retention_streaks == {'flat': 0, 'h00': 0}
    # Two more H00 failures reach H00's own stop, with flat still at zero.
    manager.record_retention(kind='stair_retention', failed_cases=12, update=1200)
    decision = manager.record_retention(kind='stair_retention', failed_cases=14,
                                        update=1400)
    assert decision['stop'] is True and decision['streak'] == 2
    assert manager.state.retention_streaks == {'flat': 0, 'h00': 2}
    # Unknown kinds/out-of-range counts are hard errors, never silent defaults.
    with pytest.raises(ValueError):
        manager.record_retention(kind='other', failed_cases=12)
    with pytest.raises(ValueError):
        manager.record_retention(kind='flat', failed_cases=16)
    # Both counters and the per-set schedules are part of the serializable state.
    state = manager.state_dict()
    assert state['retention_streaks'] == {'flat': 0, 'h00': 2}
    restored = CurriculumManager(num_envs=4, seed=42, state=state)
    assert restored.state.retention_streaks == {'flat': 0, 'h00': 2}


def test_retention_both_sets_due_at_same_boundary_are_each_reported():
    manager = _manager(phase='H', last_update=0)
    manager._qualify_retention('flat', 200)
    manager._qualify_retention('h00', 350)
    due = manager.retention_due(400)
    assert tuple(due) == ('flat', 'h00')
    # Each due set is fed independently at that boundary.
    decisions = [manager.record_retention(
        kind=curriculum.retention_gate_kind(kind), failed_cases=12, update=400)
        for kind in due]
    assert [d['kind'] for d in decisions] == ['flat', 'h00']
    assert manager.state.retention_streaks == {'flat': 1, 'h00': 1}


def test_retention_schedule_uses_strict_next_200_multiple():
    manager = _manager(phase='H', last_update=0)
    # q = 200 -> first check 400, never 200.
    info = manager._qualify_retention('flat', 200)
    assert (info['qualified_at'], info['next_due']) == (200, 400)
    assert manager.retention_due(200) is None
    assert manager.retention_due(399) is None
    assert manager.retention_due(400) == ('flat',)
    decision = manager.record_retention(kind='flat', failed_cases=3, update=400)
    assert decision['next_due'] == 600 and decision['last_completed'] == 400
    assert manager.retention_due(500) is None
    assert manager.retention_due(600) == ('flat',)
    # q = 350 -> 400 (not 550); q = 400 -> 600 (not 400).
    for qualified, expected in ((350, 400), (400, 600), (0, 200), (199, 200)):
        manager = _manager(phase='H', last_update=0)
        manager._qualify_retention('h00', qualified)
        assert manager.state.retention_qualified_at['h00'] == qualified
        assert manager.state.retention_next_due['h00'] == expected, qualified
        assert manager.retention_due(qualified) is None
    # The two sets keep independent schedules on the same boundary.
    manager = _manager(phase='H', last_update=0)
    manager._qualify_retention('flat', 350)   # due 400
    manager._qualify_retention('h00', 150)    # due 200
    assert manager.retention_due(200) == ('h00',)
    assert manager.retention_due(400) == ('flat', 'h00')
    manager.record_retention(kind='h00', failed_cases=0, update=400)
    assert manager.state.retention_next_due == {'flat': 400, 'h00': 600}


def test_retention_schedule_survives_regression_and_generation_changes():
    manager = _manager(phase='H', level_tick=10, last_update=0)
    manager._qualify_retention('flat', 200)
    manager._qualify_retention('h00', 250)
    before = dict(manager.state.retention_qualified_at), dict(
        manager.state.retention_next_due)
    # A regression / new generation must never postpone the first check.
    manager._new_generation('test')
    manager.state.retention_enabled = {'flat': True, 'h00': True}
    assert (dict(manager.state.retention_qualified_at),
            dict(manager.state.retention_next_due)) == before
    assert manager.retention_due(400) == ('flat', 'h00')
    # Re-qualifying an already qualified set does not move its first check.
    manager._qualify_retention('flat', 399)
    assert manager.state.retention_next_due['flat'] == 400


def test_retention_pending_evaluation_resumes_once_without_skip_or_duplicate():
    manager = _manager(phase='H', last_update=0)
    manager._qualify_retention('flat', 350)
    state = manager.state_dict()
    restored = CurriculumManager(num_envs=4, seed=42, state=state)
    assert restored.retention_due(400) == ('flat',)
    first = restored.record_retention(kind='flat', failed_cases=12, update=400)
    assert first['duplicate'] is False and first['streak'] == 1
    assert restored.state.retention_next_due['flat'] == 600
    assert restored.state.retention_last_completed['flat'] == 400
    # Replaying the same pending completion after a resume is idempotent.
    again = restored.record_retention(kind='flat', failed_cases=12, update=400)
    assert again['duplicate'] is True and again['streak'] == 1
    assert restored.state.retention_next_due['flat'] == 600
    assert restored.retention_due(500) is None
    # A late (skipped boundary) completion lands on the next strict multiple.
    late = restored.record_retention(kind='flat', failed_cases=0, update=650)
    assert late['duplicate'] is False and late['next_due'] == 800
    assert restored.retention_due(799) is None
    assert restored.retention_due(800) == ('flat',)
    # The schedule fields are serialized.
    state = restored.state_dict()
    assert state['retention_qualified_at'] == {'flat': 350, 'h00': None}
    assert state['retention_next_due']['flat'] == 800
    assert state['retention_last_completed']['flat'] == 650
    rebuilt = CurriculumManager(num_envs=4, seed=42, state=state)
    assert rebuilt.retention_due(800) == ('flat',)


def test_minimum_level_failure_stops_after_four_windows():
    manager = _manager(phase='H', level_tick=0, gate_streak=2)
    manager.state.spent_by_phase['H'] = curriculum.spec.MIN_LEVEL_PHASE_UPDATES
    for index in range(3):
        _feed_window(manager, 100 + index * 100, failure=True)
        manager.on_update(201 + index)
        assert not manager.state.finished, index
    _feed_window(manager, 400, failure=True)
    events = manager.on_update(204)
    assert manager.state.finished
    assert manager.state.stop_status == 'minimum_level_failure'
    assert any(event.kind == 'final' and event.status == 'minimum_level_failure'
               for event in events)


def test_on_update_is_idempotent_and_tracks_spent_budgets():
    manager = _manager(phase='H', level_tick=5)
    manager.on_update(101)
    spent = manager.state.spent_by_phase['H']
    assert spent == 201
    assert manager.state.generation_updates == 101
    assert manager.on_update(101) == []
    assert manager.state.spent_by_phase['H'] == spent


def test_state_dict_is_json_safe_and_roundtrips():
    manager = _manager(level_tick=7, gate_streak=2)
    manager.state.ema = dict(s=0.7, f=0.2, q=0.5, batches=3)
    _feed_window(manager, 100, success=True)
    manager.on_update(201)
    manager.record_retention(kind='stair_retention', failed_cases=12)
    state = manager.state_dict()
    encoded = json.dumps(state)
    restored = CurriculumManager(num_envs=4, seed=42, state=json.loads(encoded))
    assert restored.state_dict() == state
    assert restored.schedule().as_dict() == manager.schedule().as_dict()
    loaded = CurriculumState(**{name: state[name] for name in state})
    assert loaded.level_tick == manager.state.level_tick


def test_numeric_anomaly_in_window_raises():
    manager = _manager()
    outcomes = [_outcome(0, 1) for _ in range(curriculum.WINDOW)]
    for outcome in outcomes:
        manager.observe(outcome)
    # Corrupt one episode so it is neither success, failure nor timeout.
    manager.state.episode_fifo[0]['success'] = False
    manager.state.episode_fifo[0]['failure'] = False
    manager.state.episode_fifo[0]['timeout'] = False
    with pytest.raises(ValueError):
        manager.on_update(101)
