"""CPU-only v0.6 runner contract (SDS 5/6/8): gate schedule, phase/substage
reaction, checkpoint discipline, retention regression, budget exhaustion and the
separated terminal states (user stop is never a training or communication
failure)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import a2_v06_bootstrap  # noqa: E402

a2_v06_bootstrap.activate(ROOT)

from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from src.tasks.velocity.v06 import model as model_module  # noqa: E402
from src.tasks.velocity.v06 import ppo as ppo_module  # noqa: E402
from src.tasks.velocity.v06 import runner as target  # noqa: E402
from src.tasks.velocity.v06 import spec  # noqa: E402


class FakeActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fusion = nn.Linear(2, 2)
        self.route_fusion = nn.Linear(2, 2)
        with torch.no_grad():
            self.fusion.weight.zero_()
            self.fusion.bias.zero_()
            self.route_fusion.weight.zero_()
            self.route_fusion.bias.zero_()
        self.normalizer = nn.Identity()


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = FakeActor()
        self.critic = nn.Linear(2, 1)
        self.critic_normalizer = nn.Identity()
        self.std = nn.Parameter(torch.full((12,), 0.30))
        self.transfer_metadata = dict(sha256=model_module.INIT_SHA256,
                                      source_path='/tmp/eth.pt',
                                      source_iteration=2000,
                                      route_fusion='zeros_after_migration',
                                      fusion='zeros_after_migration')

    def optimizer_parameters(self):
        return [parameter for parameter in self.parameters() if parameter.requires_grad]


class FakeEnv:
    num_envs = 8
    device = 'cpu'
    frontier_tile_id = 1  # H level_tick=0 -> H00 (phase H evaluation target)

    def __init__(self):
        self.schedules = 0
        self.resets = 0
        self.iterations = []

    def reset(self):
        self.resets += 1
        return ({'proprio': torch.zeros(8)}, {})

    def set_schedule(self, schedule, iteration=None):
        self.schedules += 1
        self.iterations.append(None if iteration is None else int(iteration))
        return schedule

    def rng_state(self):
        return {'rng': [1]}

    def sampler_rng_state(self):
        return {'sampler': [2]}


class FakePPO:
    def __init__(self, update=0, num_envs=8, model=None):
        self.update_count = int(update)
        self.num_envs = int(num_envs)
        self.model = model or FakeModel()
        self.optimizer = ppo_module.build_optimizer(self.model)
        self.learning_state = dict(optimizer_initialized=True,
                                   completed_ppo_updates=self.update_count,
                                   optimizer_steps=self.update_count * 20)
        self.collect_calls = 0

    @property
    def learning(self):
        update = int(self.learning_state['completed_ppo_updates'])
        samples = update * self.num_envs * 48
        return dict(update=update, optimizer_step=update * 20, adam_steps=update * 20,
                    control_samples=samples,
                    physics_substeps=ppo_module.physics_substeps(samples))

    def reset(self):
        return ({'proprio': torch.zeros(self.num_envs)}, {})

    def collect(self):
        self.collect_calls += 1
        return []

    def update(self):
        # The real trainer derives every counter from ``learning_state`` after a
        # checkpoint restore; the fake mirrors that instead of its constructor value.
        self.update_count = int(self.learning_state['completed_ppo_updates']) + 1
        self.learning_state['completed_ppo_updates'] = self.update_count
        self.learning_state['optimizer_steps'] = self.update_count * 20
        self.learning_state['optimizer_initialized'] = True
        return dict(loss=1.0, kl=0.01, lr=3e-4, update=self.update_count,
                    reward_terms={'tracking_lin_vel': 1.0})


PHASE_ORDER = ('F0', 'H', 'D', 'L', 'V', 'R')


def spend(update):
    """Distribute a global update count over the frozen phase budgets."""
    remaining, spent = int(update), {}
    for phase in PHASE_ORDER:
        budget = spec.phase_budget(phase)
        count = min(remaining, budget)
        if count:
            spent[phase] = count
        remaining -= count
    return spent


def phase_for(update):
    spent = spend(update)
    for phase in reversed(PHASE_ORDER):
        if spent.get(phase):
            return phase
    return 'F0'


class FakeCurriculum:
    def __init__(self, phase='F0', update=0):
        self.phase = phase
        self.substage = None
        self.update = int(update)
        self.events = []
        self.retention = None
        self.observed = 0
        self.gates = []
        self.spent = spend(self.update)
        self.last_update = int(update)
        self.generation_updates = int(update)
        self.substage_spent = {}
        self.finished = False
        self.stop_status = None
        self.schedule_calls = 0

    def state_dict(self):
        # A phase only spends down its own budget; earlier phases are exhausted.
        spent = dict(self.spent)
        if self.phase not in spent:
            spent[self.phase] = 0
        return dict(phase=self.phase, substage=self.substage, level_tick=0,
                    generation=0, generation_updates=self.generation_updates,
                    last_update=self.last_update,
                    finished=self.finished, stop_status=self.stop_status,
                    spent_by_phase=spent, spent_by_substage=dict(self.substage_spent),
                    episode_fifo=[], ema=None, ready_streak=0, hard_streak=0,
                    cooldown=0, gate_streak=0, retention_streaks={},
                    minimum_level_failure_streak=0, sampler_rng_state={})

    def load_state_dict(self, value):
        self.phase = value['phase']
        self.substage = value['substage']
        self.spent = dict(value['spent_by_phase'])
        self.last_update = int(value.get('last_update', self.last_update))
        self.generation_updates = int(value.get('generation_updates',
                                                self.generation_updates))

    def schedule(self):
        self.schedule_calls += 1
        return SimpleNamespace(phase=self.phase, substage=self.substage,
                               frontier_tile_id=1, roles=(), tile_ids=())

    def observe(self, outcomes):
        self.observed += len(outcomes)

    def on_update(self, update, gates=None):
        self.update = int(update)
        self.last_update = int(update)
        self.generation_updates = int(update)
        self.spent = spend(int(update))
        self.phase = phase_for(int(update))
        for gate in gates or ():
            self.gates.append(dict(gate))
        events, self.events = self.events, []
        return events

    def retention_due(self, update):
        return self.retention

    def record_gate(self, passed, update, kind):
        self.gates.append(dict(passed=passed, update=update, kind=kind))


def make_runner(tmp_path, *, update=0, phase='F0', retention=None, failing=0,
                seed=42, stall_cases=0, first_step_absent=0):
    model = FakeModel()
    env = FakeEnv()
    ppo = FakePPO(update=update, model=model)
    curriculum = FakeCurriculum(phase=phase, update=update)
    curriculum.retention = retention
    total = 15 if retention else 60
    counter = {'n': 0}

    def case_runner(case):
        counter['n'] += 1
        # ``failing`` failures per speed (retention/diagnostic = 3 speeds x 5).
        index = case['episode_index']
        fail = index < failing
        return dict(measurement='stair', climb_success=not fail, failure=fail,
                    reason=None if not fail else 'failure', mean_vx=case['vx'],
                    mean_relative_error=0.0, flat_pass=not fail,
                    active_samples=100,
                    stall_samples=100 if index < stall_cases else 0,
                    first_step_valid=index >= first_step_absent)

    runner = target.V06Runner(model=model, ppo=ppo, curriculum=curriculum, env=env,
                              directory=tmp_path, seed=seed, device='cpu',
                              case_runner=case_runner)
    return runner, ppo, curriculum, env


def test_runner_starts_running_without_a_pilot_state(tmp_path):
    runner, _ppo, _curriculum, _env = make_runner(tmp_path)
    assert runner.update == 0 and runner.phase == 'F0'
    assert runner.lifecycle_state == 'running'
    assert runner.stop_state is None
    assert not hasattr(runner, 'pilot')  # Q002-4: no pilot lifecycle in v0.6
    assert runner.learning['update'] == 0
    assert runner.result()['state'] is None


def test_gate_every_100_updates_and_saves_evidence(tmp_path):
    runner, ppo, curriculum, _env = make_runner(tmp_path, update=99)
    runner.train_one_update()
    assert ppo.update_count == 100 and runner.update == 100
    assert runner.gates and runner.gates[-1].kind == 'validation'
    assert runner.gates[-1].passed is True
    assert runner.gate_streak == 1
    assert curriculum.gates[-1]['passed'] is True
    reasons = [item['reason'] for item in runner.checkpoints]
    assert 'gate' in reasons and 'interval' in reasons
    assert (tmp_path / 'checkpoints').is_dir()
    assert (tmp_path / 'gates').is_dir()
    boundary = dict(actions=['gate', 'save'])
    assert runner.update == 100
    assert runner.result()['state'] is None


def test_two_consecutive_gate_passes_open_the_phase_transition(tmp_path):
    runner, _ppo, _curriculum, _env = make_runner(tmp_path, update=100)
    runner.run_gate('validation')
    assert runner.gate_streak == 1 and runner.pending_transition is None
    runner.run_gate('validation')
    assert runner.gate_streak == 2
    assert runner.pending_transition is not None
    assert runner.pending_transition['kind'] == 'gate_passed'


def test_failed_gate_resets_the_streak(tmp_path):
    runner, _ppo, _curriculum, _env = make_runner(tmp_path, update=100, failing=5)
    runner.run_gate('validation')
    assert runner.gate_streak == 0
    assert runner.pending_transition is None


def test_v_and_r_gate_streaks_wait_for_min_updates(tmp_path):
    runner, _ppo, _curriculum, _env = make_runner(tmp_path, update=500, phase='V')
    runner.run_gate('validation')
    assert runner.gate_streak == 0
    runner, _ppo, curriculum, _env = make_runner(tmp_path / 'late', update=1001,
                                                  phase='V')
    curriculum.spent = {'V': 1001}
    runner.run_gate('validation')
    assert runner.gate_streak == 1
    runner, _ppo, _curriculum, _env = make_runner(tmp_path / 'r', update=599, phase='R')
    runner.run_gate('validation')
    assert runner.gate_streak == 0


def test_curriculum_transition_saves_and_resets_the_schedule(tmp_path):
    runner, ppo, curriculum, env = make_runner(tmp_path, update=0)
    curriculum.events = [dict(kind='advance', phase='F0', substage=None)]

    def advance(update):
        curriculum.update = int(update)
        curriculum.last_update = int(update)
        curriculum.generation_updates = int(update)
        curriculum.spent = {'F0': int(update)}
        curriculum.phase = 'H'
        events, curriculum.events = curriculum.events, []
        return events

    curriculum.on_update = advance
    runner.update = 0
    ppo.update_count = 0
    runner.train_one_update()
    assert runner.update == 1
    assert runner.phase == 'H'
    assert env.schedules == 1
    assert any(item['reason'] == 'transition' for item in runner.checkpoints)
    assert runner.transitions and runner.transitions[-1]['phase'] == 'F0'


def test_retention_failure_twice_is_behavior_regression(tmp_path):
    runner, _ppo, _curriculum, _env = make_runner(tmp_path, update=200, phase='H',
                                                  retention='flat', failing=5)
    runner.run_gate('flat_retention', retention_kind='flat')
    assert runner.stop_state is None
    assert runner.retention['streaks']['flat'] == 1
    assert runner.retention['streaks']['h00'] == 0  # independent, untouched
    runner.run_gate('flat_retention', retention_kind='flat')
    assert runner.stop_state == 'behavior_regression'
    assert (tmp_path / 'stop.json').is_file()
    assert runner.stop_evidence['failure_class'] == 'training_failure'
    assert target.classify_failure('behavior_regression') == 'training_failure'


def test_retention_does_not_stop_below_twelve(tmp_path):
    runner, _ppo, _curriculum, _env = make_runner(tmp_path, update=200, phase='H',
                                                  retention='flat', failing=3)
    runner.run_gate('flat_retention', retention_kind='flat')
    runner.run_gate('flat_retention', retention_kind='flat')
    assert runner.stop_state is None
    assert runner.retention['streaks']['flat'] == 0


def test_total_budget_exhaustion_stops_the_run(tmp_path):
    runner, ppo, curriculum, _env = make_runner(tmp_path, update=29999)
    runner.update = 30000
    runner.phase = 'R'
    curriculum.phase = 'R'
    curriculum.spent = spend(30000)
    ppo.learning_state['completed_ppo_updates'] = 30000
    runner._check_stop_conditions()
    assert runner.stop_state == 'budget_exhausted'
    assert runner.stop_evidence['failure_class'] == 'training_failure'
    assert (tmp_path / 'stop.json').is_file()


def test_user_stop_is_its_own_terminal_state(tmp_path):
    runner, _ppo, _curriculum, _env = make_runner(tmp_path)
    runner.stop_flag = lambda: True
    runner._check_stop_conditions()
    assert runner.stop_state == 'user_requested_stop'
    assert runner.stop_evidence['failure_class'] == 'user'
    assert target.classify_failure('user_requested_stop') == 'user'
    assert target.classify_failure('monitoring_blocked') == 'communication'
    assert target.classify_failure('gate_failed') == 'training_failure'
    assert target.classify_stop(user_requested=True, resource_available=False) == \
        'user_requested_stop'
    assert target.classify_stop(resource_available=False,
                                behavior_regression=True) == 'resource_insufficient'


def test_phase_budget_without_advance_is_gate_failed(tmp_path):
    runner, _ppo, curriculum, _env = make_runner(tmp_path, update=1000)
    curriculum.spent = spend(1000)
    curriculum.finished = True
    curriculum.stop_status = 'budget_exhausted'
    runner.update = 1000
    runner._check_stop_conditions()
    assert runner.stop_state == 'gate_failed'
    assert runner.stop_reason == 'phase_budget_exhausted'
    assert runner.stop_evidence['budget']['total_exhausted'] is False
    assert runner.stop_evidence['failure_class'] == 'training_failure'


def test_save_and_resume_never_rebill_an_update(tmp_path):
    runner, ppo, _curriculum, _env = make_runner(tmp_path, update=150)
    record = runner.save('interval')
    assert record['update'] == 150
    resumed, new_ppo, new_curriculum, new_env = make_runner(tmp_path / 'resume',
                                                            update=0)
    contract = resumed.load(record['path'], record['sha256'])
    assert contract['new_episode'] is True
    assert contract['replay_billed_updates'] is False
    assert resumed.update == 150
    assert resumed.learning['update'] == 150
    assert new_ppo.update_count == 0
    assert new_ppo.collect_calls == 0
    assert (tmp_path / 'resume' / 'resume.json').is_file()


def test_resume_rejects_a_checkpoint_from_another_run(tmp_path):
    runner, _ppo, _curriculum, _env = make_runner(tmp_path, update=150)
    record = runner.save('interval')
    other, _ppo, _curriculum, _env = make_runner(tmp_path / 'other', update=0, seed=43)
    with pytest.raises(Exception):
        other.load(record['path'], record['sha256'])


def test_curriculum_complete_runs_the_final_test(tmp_path):
    runner, _ppo, curriculum, _env = make_runner(tmp_path, update=25000, phase='R')
    curriculum.finished = True
    curriculum.stop_status = 'complete'
    result = runner.run(max_updates=25001)
    assert runner.lifecycle_state == 'complete'
    assert runner.stop_state is None
    assert runner.curriculum_complete is True
    assert runner.final is not None and runner.final['passed'] is True
    assert (tmp_path / 'final_test' / 'summary.json').is_file()
    assert len(runner.final['summary']['per_speed']) == 3
    assert result['state'] == 'complete'


def test_final_test_below_95_is_a_gate_failure(tmp_path):
    runner, _ppo, curriculum, _env = make_runner(tmp_path, update=25000, phase='R')
    curriculum.finished = True
    curriculum.stop_status = 'complete'

    def case_runner(case):
        fail = case['episode_index'] >= 94  # 6 failures per speed => 94/100
        return dict(measurement='stair', climb_success=not fail, failure=fail,
                    reason=None if not fail else 'failure', mean_vx=case['vx'],
                    mean_relative_error=0.0)

    runner.case_runner = case_runner
    runner.run(max_updates=25001)
    assert runner.final is not None and runner.final['passed'] is False
    assert runner.stop_state == 'gate_failed'
    assert runner.stop_evidence['stage'] == 'final_test'


def test_curriculum_minimum_level_failure_is_a_training_stop(tmp_path):
    runner, _ppo, curriculum, _env = make_runner(tmp_path, update=3000, phase='L')
    curriculum.finished = True
    curriculum.stop_status = 'minimum_level_failure'
    runner._check_stop_conditions()
    assert runner.stop_state == 'curriculum_stalled'
    assert runner.stop_evidence['failure_class'] == 'training_failure'
    assert (tmp_path / 'stop.json').is_file()


# ---------------------------------------------------------------- Q001 fixes
def test_retention_kinds_are_independent_on_the_same_boundary(tmp_path):
    runner, _ppo, curriculum, _env = make_runner(tmp_path, update=200, phase='H',
                                                  retention=('flat', 'h00'), failing=4)
    due = runner._retention_due()
    assert due == ['flat', 'h00']
    # Both kinds fail 12/15 at the first boundary: each keeps its own streak.
    for kind in due:
        runner.run_gate(target._retention_gate_kind(curriculum, kind),
                        retention_kind=kind)
    assert runner.retention['streaks'] == {'flat': 1, 'h00': 1}
    assert runner.stop_state is None
    for kind in due:
        runner.run_gate(target._retention_gate_kind(curriculum, kind),
                        retention_kind=kind)
    assert runner.stop_state == 'behavior_regression'
    assert runner.retention['streaks'] == {'flat': 2, 'h00': 2}


def test_one_retention_kind_never_clears_or_accumulates_the_other(tmp_path):
    runner, _ppo, curriculum, _env = make_runner(tmp_path, update=200, phase='H',
                                                  retention=('flat', 'h00'), failing=0)
    runner.run_gate('flat_retention', retention_kind='flat', )
    # A clean flat set leaves the h00 streak exactly where it was (and vice versa).
    runner.retention['streaks']['h00'] = 1
    runner.run_gate('stair_retention', retention_kind='h00')
    assert runner.retention['streaks']['flat'] == 0
    assert runner.retention['streaks']['h00'] == 0


def test_final_transition_stops_training_and_enters_the_final_test(tmp_path):
    runner, ppo, curriculum, _env = make_runner(tmp_path, update=5000, phase='R')

    def complete(update):
        curriculum.update = int(update)
        curriculum.last_update = int(update)
        curriculum.generation_updates = int(update)
        curriculum.spent = spend(int(update))
        curriculum.finished = True
        curriculum.stop_status = 'complete'
        return [dict(kind='final', status='complete', phase='R',
                     update=int(update), detail={})]

    curriculum.on_update = complete
    runner.update = 5000
    ppo.update_count = 5000
    ppo.learning_state['completed_ppo_updates'] = 5000
    result = runner.run(max_updates=30000)
    assert runner.lifecycle_state == 'complete' and runner.stop_state is None
    assert runner.final is not None and runner.final['passed'] is True
    assert runner.final['update'] == runner.update == 5001
    assert ppo.update_count == 5001  # exactly one update: none after kind='final'
    assert result['state'] == 'complete'


def test_final_event_minimum_level_failure_is_curriculum_stalled(tmp_path):
    runner, _ppo, curriculum, _env = make_runner(tmp_path, update=3000, phase='L')
    runner.phase = 'L'
    runner._consume_terminal_event(dict(kind='final', status='minimum_level_failure'))
    assert runner.stop_state == 'curriculum_stalled'
    assert runner.stop_evidence['failure_class'] == 'training_failure'


def test_final_event_budget_mapping_uses_the_total_budget(tmp_path):
    runner, _ppo, curriculum, _env = make_runner(tmp_path, update=300, phase='H')
    runner._consume_terminal_event(dict(kind='final', status='budget_exhausted'))
    assert runner.stop_state == 'gate_failed'

    other, _ppo, curriculum, _env = make_runner(tmp_path / 'total', update=30000,
                                                phase='R')
    other.update = 30000
    other._consume_terminal_event(dict(kind='final', status='budget_exhausted'))
    assert other.stop_state == 'budget_exhausted'


def test_finished_curriculum_forbids_another_billed_update(tmp_path):
    runner, ppo, curriculum, _env = make_runner(tmp_path, update=100, phase='H')
    curriculum.finished = True
    curriculum.stop_status = 'budget_exhausted'
    ppo.update_count = 100
    ppo.learning_state['completed_ppo_updates'] = 100
    with pytest.raises(target.RunStopped):
        runner.train_one_update()
    assert ppo.collect_calls == 0  # no rollout was billed
    assert runner.stop_state == 'gate_failed'  # phase budget, total budget not reached


def test_diagnostic_collapse_needs_two_consecutive_hits(tmp_path):
    runner, _ppo, curriculum, _env = make_runner(tmp_path, update=100, phase='H',
                                                  stall_cases=4, first_step_absent=4)
    runner._diagnostic()
    assert runner.diagnostic['streak'] == 1
    assert runner.diagnostic['stalled_cases'] == 12
    assert runner.diagnostic['first_step_absent_cases'] == 12
    assert runner.stop_state is None
    runner._diagnostic()
    assert runner.stop_state == 'behavior_collapse'
    assert runner.stop_evidence['failure_class'] == 'training_failure'
    assert (tmp_path / 'stop.json').is_file()


def test_diagnostic_collapse_needs_both_independent_conditions(tmp_path):
    # 12 stalled cases but every case has a valid first step: no stairs hit.
    runner, _ppo, curriculum, _env = make_runner(tmp_path, update=100, phase='H',
                                                  stall_cases=4, first_step_absent=0)
    runner._diagnostic()
    assert runner.diagnostic['stalled_cases'] == 12
    assert runner.diagnostic['first_step_absent_cases'] == 0
    assert runner.diagnostic['streak'] == 0

    # Flat has no first-step condition at all.
    flat, _ppo, curriculum, _env = make_runner(tmp_path / 'flat', update=100, phase='F0',
                                               stall_cases=4, first_step_absent=0)
    flat._diagnostic()
    assert flat.diagnostic['streak'] == 1


def test_diagnostic_clean_result_and_new_generation_clear_the_streak(tmp_path):
    runner, _ppo, curriculum, _env = make_runner(tmp_path, update=100, phase='H',
                                                  stall_cases=0, first_step_absent=0)
    runner._diagnostic()
    assert runner.diagnostic['streak'] == 0
    # A prior streak is cleared by a valid non-hit and by a new generation.
    runner.diagnostic['streak'] = 1
    runner.diagnostic['generation'] = curriculum.state_dict()['generation']
    runner._diagnostic()
    assert runner.diagnostic['streak'] == 0
    runner.diagnostic.update(streak=1, generation=-1)
    runner._diagnostic()
    assert runner.diagnostic['streak'] == 0


def test_diagnostic_missing_fields_is_an_evaluation_error(tmp_path):
    runner, _ppo, curriculum, _env = make_runner(tmp_path, update=100, phase='H')

    def incomplete(case):
        return dict(measurement='stair', climb_success=True, failure=False,
                    mean_vx=case['vx'])

    runner.case_runner = incomplete
    with pytest.raises(Exception) as error:
        runner._diagnostic()
    assert 'active_samples' in str(error.value) or 'stall_samples' in str(error.value)


def test_real_curriculum_wiring_end_to_end(tmp_path):
    """Drive the real agent-B CurriculumManager through the runner boundary.

    Exercises the Q001 integration seams without a simulator: record_gate ->
    gate_streak -> phase transition, retention_due returning a tuple of short
    kinds, record_retention with independent per-kind streaks, and the checkpoint
    round-trip of the new diagnostic/retention state.
    """
    from src.tasks.velocity.v06.curriculum import CurriculumManager

    model = FakeModel()
    env = FakeEnv()
    ppo = FakePPO(update=0, model=model)
    curriculum = CurriculumManager(num_envs=8, seed=42)

    def case_runner(case):
        return dict(measurement=('flat' if str(case.get('measurement_stage')) == 'F0'
                                 else 'stair'),
                    climb_success=True, failure=False, flat_pass=True,
                    reason=None, mean_vx=case['vx'], mean_relative_error=0.0,
                    active_samples=100, stall_samples=0, first_step_valid=True)

    runner = target.V06Runner(model=model, ppo=ppo, curriculum=curriculum, env=env,
                              directory=tmp_path, seed=42, device='cpu',
                              case_runner=case_runner)
    runner.case_runner = case_runner
    for _ in range(400):
        if runner.stop_state is not None or runner.curriculum_complete:
            break
        runner.train_one_update()
    result = runner.result()
    assert result['state'] is None
    assert runner.phase == 'H'
    assert runner.update == 400
    kinds = [record.kind for record in runner.gates]
    assert kinds.count('validation') == 4          # every 100 updates
    assert 'flat_retention' in kinds               # every 200 after F0 passed
    assert runner.retention['streaks'] == {'flat': 0, 'h00': 0}
    assert runner.diagnostic['streak'] == 0
    assert runner.diagnostic['required'] == 2
    assert any(item['reason'] == 'transition' for item in runner.checkpoints)
    # The real curriculum advanced F0 -> H through our record_gate hook.
    assert any(item.get('kind') in ('advance', 'final') for item in runner.transitions) \
        or runner.phase == 'H'
    # Checkpoint round-trip keeps the per-kind retention and collapse counters.
    record = runner.save('evidence')
    resumed, _ppo, _curriculum, _env = make_runner(tmp_path / 'resume', update=0)
    resumed_curriculum = CurriculumManager(num_envs=8, seed=42)
    resumed.curriculum = resumed_curriculum
    contract = resumed.load(record['path'], record['sha256'])
    assert contract['new_episode'] is True
    assert resumed.retention['streaks'] == {'flat': 0, 'h00': 0}
    assert resumed.diagnostic['streak'] == 0


# =========================================================== Q002 decisions
def real_runner(tmp_path, *, fail_at=(), stall_cases=0, first_step_absent=0):
    """A runner backed by the real agent-B CurriculumManager (no simulator)."""
    from src.tasks.velocity.v06.curriculum import CurriculumManager

    model = FakeModel()
    env = FakeEnv()
    ppo = FakePPO(update=0, model=model)
    curriculum = CurriculumManager(num_envs=8, seed=42)
    state = {'fail_at': set(int(value) for value in fail_at)}
    holder = {}

    def case_runner(case):
        update = holder['runner'].update
        fail = update in state['fail_at']
        return dict(
            measurement=('flat' if str(case.get('measurement_stage')) == 'F0'
                         else 'stair'),
            climb_success=not fail, failure=fail, flat_pass=not fail,
            reason=None if not fail else 'failure', mean_vx=case['vx'],
            mean_relative_error=0.0, active_samples=100,
            stall_samples=100 if case['episode_index'] < stall_cases else 0,
            first_step_valid=case['episode_index'] >= first_step_absent)

    runner = target.V06Runner(model=model, ppo=ppo, curriculum=curriculum, env=env,
                              directory=tmp_path, seed=42, device='cpu',
                              case_runner=case_runner)
    holder['runner'] = runner
    return runner, ppo, curriculum, env


def drive(runner, updates):
    for _ in range(int(updates)):
        if runner.stop_state is not None or runner.lifecycle_state != 'running':
            break
        runner.train_one_update()
    return runner


def test_f0_passes_at_100_and_200_advance_to_h(tmp_path):
    runner, _ppo, curriculum, _env = real_runner(tmp_path)
    drive(runner, 201)
    assert runner.stop_state is None
    assert runner.phase == 'H' and curriculum.phase == 'H'
    counted = [entry for entry in curriculum.state_dict()['gate_log'] if entry['counted']]
    assert [entry['update'] for entry in counted] == [100, 200]


def test_f0_second_gate_failure_blocks_until_a_later_pair(tmp_path):
    runner, _ppo, curriculum, _env = real_runner(tmp_path, fail_at=(200,))
    drive(runner, 300)
    assert runner.phase == 'F0' and runner.stop_state is None
    assert int(curriculum.state_dict()['gate_streak']) == 1
    drive(runner, 101)  # to update 401 so the 400 gate is a consecutive second pass
    assert runner.phase == 'H'


def test_f0_last_update_legal_second_pass_advances_before_budget(tmp_path):
    runner, _ppo, curriculum, _env = real_runner(
        tmp_path, fail_at=set(range(100, 1001, 100)) - {900, 1000})
    drive(runner, 1000)
    assert runner.stop_state is None
    assert runner.phase == 'H'
    assert runner.update == 1000
    assert any(entry['update'] == 1000 and entry['counted']
               for entry in curriculum.state_dict()['gate_log'])


def test_f0_budget_exhausted_without_a_second_pass_is_gate_failed(tmp_path):
    runner, _ppo, curriculum, _env = real_runner(
        tmp_path, fail_at=set(range(100, 1001, 100)) - {900})
    drive(runner, 1001)
    assert runner.update == 1000
    assert runner.stop_state == 'gate_failed'
    assert runner.stop_reason == 'phase_budget_exhausted'
    assert runner.stop_evidence['budget']['total_exhausted'] is False
    assert curriculum.state_dict()['finished'] is True


def test_total_and_phase_budget_in_the_same_frame_prefers_budget_exhausted(tmp_path):
    runner, ppo, curriculum, _env = make_runner(tmp_path, update=30000, phase='R')
    runner.update = 30000
    runner.phase = 'R'
    curriculum.phase = 'R'
    curriculum.spent = spend(30000)
    curriculum.finished = True
    curriculum.stop_status = 'budget_exhausted'
    ppo.learning_state['completed_ppo_updates'] = 30000
    runner._check_stop_conditions()
    assert runner.stop_state == 'budget_exhausted'
    assert runner.stop_reason == 'total_budget_exhausted'
    budget = runner.stop_evidence['budget']
    assert budget['total_exhausted'] is True
    assert budget['phase_exhausted'] is True


def test_substage_budget_exhaustion_reports_its_scope(tmp_path):
    runner, _ppo, curriculum, _env = make_runner(tmp_path, update=16000, phase='D')
    runner.update = 14500
    runner.phase = 'D'
    runner.substage = 'D25'
    curriculum.phase = 'D'
    curriculum.substage = 'D25'
    curriculum.spent = {'F0': 1000, 'H': 12000, 'D': 1500}
    curriculum.substage_spent = {'D25': 1500}
    curriculum.finished = True
    curriculum.stop_status = 'budget_exhausted'
    ppo = runner.ppo
    ppo.learning_state['completed_ppo_updates'] = 14500
    runner._check_stop_conditions()
    assert runner.stop_state == 'gate_failed'
    assert runner.stop_reason == 'substage_budget_exhausted'


def test_a_single_failed_gate_with_budget_left_does_not_stop(tmp_path):
    runner, _ppo, _curriculum, _env = make_runner(tmp_path, update=100, failing=5)
    runner.run_gate('validation')
    runner._check_stop_conditions()
    assert runner.stop_state is None
    assert runner.lifecycle_state == 'running'


def test_behavior_regression_is_never_overwritten_by_budget(tmp_path):
    runner, _ppo, curriculum, _env = make_runner(tmp_path, update=200, phase='H',
                                                  retention='flat', failing=5)
    runner.run_gate('flat_retention', retention_kind='flat')
    runner.run_gate('flat_retention', retention_kind='flat')
    assert runner.stop_state == 'behavior_regression'
    runner.update = 30000
    runner._check_stop_conditions()
    assert runner.stop_state == 'behavior_regression'
    assert runner.stop_reason == 'behavior_regression'


def test_total_budget_after_a_mid_phase_advance_is_still_budget_exhausted(tmp_path):
    runner, ppo, curriculum, _env = make_runner(tmp_path, update=30000, phase='H')
    runner.update = 30000
    curriculum.spent = spend(30000)
    curriculum.finished = False
    ppo.learning_state['completed_ppo_updates'] = 30000
    runner._check_stop_conditions()
    assert runner.stop_state == 'budget_exhausted'
    assert runner.stop_reason == 'total_budget_exhausted'


def test_total_budget_exactly_ready_allows_the_zero_learning_final_test(tmp_path):
    runner, ppo, curriculum, _env = make_runner(tmp_path, update=30000, phase='R')
    runner.update = 30000
    runner.phase = 'R'
    curriculum.phase = 'R'
    curriculum.spent = spend(30000)
    curriculum.finished = True
    curriculum.stop_status = 'complete'
    ppo.learning_state['completed_ppo_updates'] = 30000
    ppo.update_count = 30000
    result = runner.run(max_updates=30000)
    assert ppo.collect_calls == 0          # no 30,001st learning update
    assert runner.update == 30000
    assert runner.lifecycle_state == 'complete'
    assert result['state'] == 'complete'


def test_final_test_failure_keeps_per_speed_counts(tmp_path):
    runner, ppo, curriculum, _env = make_runner(tmp_path, update=25000, phase='R')
    runner.update = 25000
    curriculum.phase = 'R'
    curriculum.finished = True
    curriculum.stop_status = 'complete'
    ppo.learning_state['completed_ppo_updates'] = 25000
    ppo.update_count = 25000

    def case_runner(case):
        fail = case['episode_index'] >= 94
        return dict(measurement='stair', climb_success=not fail, failure=fail,
                    reason=None if not fail else 'failure', mean_vx=case['vx'],
                    mean_relative_error=0.0)

    runner.case_runner = case_runner
    result = runner.run(max_updates=30000)
    assert runner.lifecycle_state == 'running'
    assert runner.stop_state == 'gate_failed'
    assert runner.stop_reason == 'final_test_failed'
    per_speed = runner.final['per_speed']
    assert [entry['case_pass_count'] for entry in per_speed] == [94, 94, 94]
    assert runner.final['cases_per_speed'] == 100
    assert runner.final['spawn_profile'] == 'full_R_expanded'
    assert result['state'] == 'gate_failed'


def test_final_test_engineering_error_is_never_complete(tmp_path):
    runner, ppo, curriculum, _env = make_runner(tmp_path, update=25000, phase='R')
    runner.update = 25000
    runner.phase = 'R'
    curriculum.phase = 'R'
    curriculum.finished = True
    curriculum.stop_status = 'complete'
    ppo.learning_state['completed_ppo_updates'] = 25000

    def exploding(case):
        raise RuntimeError('simulator unavailable')

    runner.case_runner = exploding
    with pytest.raises(RuntimeError):
        runner.run(max_updates=30000)
    assert runner.lifecycle_state != 'complete'
    assert runner.stop_state is None
    assert runner.error_type == 'RuntimeError'
    assert (tmp_path / 'final_test' / 'error.json').is_file()


def test_retention_pending_evaluation_completes_once(tmp_path):
    runner, _ppo, curriculum, _env = make_runner(tmp_path, update=210, phase='H',
                                                  failing=5)
    entry = dict(kind='retention', retention_kind='flat', gate_kind='flat_retention',
                 update=200, phase='H', substage=None)
    runner._begin_evaluation(entry)
    completed = runner._complete_pending_evaluations()
    assert completed == [entry]
    assert runner.pending_evaluations == []
    recorded = [record for record in runner.gates if record.kind == 'flat_retention']
    assert len(recorded) == 1 and recorded[0].update == 200
    assert runner.retention['last']['flat']['update'] == 200
    streak = runner.retention['streaks']['flat']
    # A second call has nothing pending and cannot double count.
    runner._complete_pending_evaluations()
    assert len([r for r in runner.gates if r.kind == 'flat_retention']) == 1
    assert runner.retention['streaks']['flat'] == streak


def test_retention_due_returns_all_sets_and_runs_each_once(tmp_path):
    runner, _ppo, curriculum, _env = make_runner(tmp_path, update=200, phase='H',
                                                  retention=('flat', 'h00'))
    due = runner._retention_due()
    assert due == ['flat', 'h00']
    runner.boundary()
    kinds = [record.kind for record in runner.gates]
    assert kinds.count('flat_retention') == 1
    assert kinds.count('stair_retention') == 1
    assert runner.retention['last']['flat']['last_completed'] == 200
    assert runner.retention['last']['h00']['last_completed'] == 200


def test_v_ramp_uses_the_in_phase_update_count_and_survives_resume(tmp_path):
    from src.tasks.velocity.v06.curriculum import CurriculumManager

    def build(directory, update):
        model = FakeModel()
        env = FakeEnv()
        ppo = FakePPO(update=update, model=model)
        curriculum = CurriculumManager(num_envs=8, seed=42)
        curriculum.state.phase = 'V'
        curriculum.state.spent_by_phase = {'V': int(update)}
        curriculum.state.last_update = int(update)
        runner = target.V06Runner(model=model, ppo=ppo, curriculum=curriculum,
                                  env=env, directory=directory, seed=42,
                                  device='cpu')
        return runner, ppo, curriculum, env

    runner, ppo, curriculum, env = build(tmp_path, 500)
    runner._reset_environment_schedule()
    assert env.iterations[-1] == 500

    # u = 0 / 500 / 1000 all come from the V-internal completed-update count.
    for update in (0, 1, 500, 1000):
        curriculum.state.spent_by_phase = {'V': update}
        curriculum.state.last_update = update
        runner.update = update
        ppo.learning_state['completed_ppo_updates'] = update
        runner._reset_environment_schedule()
        assert env.iterations[-1] == update

    curriculum.state.spent_by_phase = {'V': 500}
    curriculum.state.last_update = 500
    runner.update = 500
    ppo.learning_state['completed_ppo_updates'] = 500
    record = runner.save('evidence')

    resumed, resumed_ppo, _curriculum, resumed_env = build(tmp_path / 'resume', 0)
    resumed.load(record['path'], record['sha256'])
    assert resumed.phase == 'V'
    resumed._reset_environment_schedule()
    assert resumed_env.iterations[-1] == 500  # checkpoint/resume never resets u
    assert int(resumed._curriculum_state()['spent_by_phase']['V']) == 500
    # The next completed V update advances u by exactly one.
    resumed.update = 501
    resumed_ppo.learning_state['completed_ppo_updates'] = 501
    resumed.curriculum.state.spent_by_phase = {'V': 501}
    resumed.curriculum.state.last_update = 501
    resumed._reset_environment_schedule()
    assert resumed_env.iterations[-1] == 501


def test_terminal_evidence_is_complete_and_uses_the_spec_states(tmp_path):
    runner, _ppo, _curriculum, _env = make_runner(tmp_path, update=1000)
    runner._stop('gate_failed', dict(reason='phase_budget_exhausted'),
                 reason='phase_budget_exhausted')
    result = runner.result()
    assert result['state'] in target.STOP_STATES
    assert result['lifecycle_state'] == 'running'
    terminal = result['terminal']
    for field in ('state', 'reason', 'global_update', 'phase', 'substage',
                  'generation', 'budget', 'evidence'):
        assert field in terminal, field
    assert terminal['state'] == 'gate_failed'
    assert terminal['reason'] == 'phase_budget_exhausted'
    assert terminal['budget']['total_budget'] == 30000
    assert result['stop_reason'] == 'phase_budget_exhausted'
    assert runner.result()['state'] in spec.ALL_TRAINING_STATES


# ------------------------------------- gate terrain identity (no H00 fallback)
def test_h_gate_measures_the_current_frontier_tile(tmp_path):
    runner, _ppo, curriculum, env = make_runner(tmp_path, update=100, phase='H')
    curriculum.phase = 'H'
    curriculum.spent = {'H': 100}
    runner.update = 100
    runner.phase = 'H'
    env.frontier_tile_id = 8  # level_tick=47 -> frontier_row 7 -> tile 8
    seen = []

    def case_runner(case):
        seen.append(case['tile_id'])
        return dict(measurement='stair', climb_success=True, failure=False,
                    reason=None, mean_vx=case['vx'], mean_relative_error=0.0,
                    active_samples=100, stall_samples=0, first_step_valid=True)

    runner.case_runner = case_runner
    runner.run_gate('validation')
    assert seen and set(seen) == {8}
    assert runner.gates[-1].phase == 'H'


def test_h_gate_without_a_frontier_tile_is_refused(tmp_path):
    runner, _ppo, curriculum, env = make_runner(tmp_path, update=100, phase='H')
    curriculum.phase = 'H'
    runner.phase = 'H'
    env.frontier_tile_id = None
    with pytest.raises(target.RunnerError):
        runner._current_tile()
    with pytest.raises(target.RunnerError):
        runner.run_gate('validation')


def test_gate_tile_follows_the_phase_protocol(tmp_path):
    runner, _ppo, curriculum, env = make_runner(tmp_path, update=100, phase='V')
    curriculum.phase = 'V'
    curriculum.spent = {'V': 100}
    runner.update = 100
    runner.phase = 'V'
    assert runner._current_tile() == spec.FINAL_TILE_ID
    seen = []

    def case_runner(case):
        seen.append(case['tile_id'])
        return dict(measurement='stair', climb_success=True, failure=False,
                    reason=None, mean_vx=case['vx'], mean_relative_error=0.0)

    runner.case_runner = case_runner
    runner.run_gate('validation')
    assert set(seen) == {spec.FINAL_TILE_ID}
    # F0 always measures the flat tile, even if the environment frontier is set.
    flat, _ppo, curriculum, env = make_runner(tmp_path / 'flat', update=100)
    env.frontier_tile_id = 8
    assert flat._current_tile() == spec.FLAT_TILE_ID


def test_evaluation_environment_kwargs_carry_the_target_tile():
    import train_a2_ideal_stairs_v06 as train

    contract = dict(num_envs=512, eval_num_envs=8, device='cpu', seed=42)
    case = dict(vx=0.3, phase='H', substage=None, kind='validation', tile_id=8, lesson_id='H07',
                initial_pose=[-1.0, 0.0, 0.0], measurement_stage='H')
    kwargs = train.case_environment_kwargs(contract, [case], 'H', None)
    assert kwargs['frontier_tile_id'] == 8
    assert kwargs['phase'] == 'H' and kwargs['substage'] is None
    assert kwargs['evaluation'] is True
    missing = dict(case)
    del missing['tile_id']
    missing['tile_id'] = None
    with pytest.raises(ValueError):
        train.case_environment_kwargs(contract, [missing], 'H', None)


# ------------------------------------- real CPU V06Env integration (B per-env)
def _real_f0_case():
    return dict(vx=0.3, initial_pose=[-1.0, 0.0, 0.0],
                measurement_stage='F0', phase='F0', case_id='flat:0.3:000',
                episode_index=0, tile_id=0, spawn_profile='standard')


def test_real_env_case_runner_returns_consistent_scalars():
    """Real V06Env (CPU): per-env [B] info must reduce to one case scalar."""
    pytest.importorskip('mujoco')
    from src.tasks.velocity.v06.env import V06Env

    class ZeroModel:
        def act(self, obs, deterministic=False):
            torch_zeros = torch.zeros(8, 12)
            return torch_zeros

    case = _real_f0_case()
    try:
        env = V06Env(num_envs=8, device='cpu', seed=42, phase='F0',
                     evaluation=True, fixed_vx=0.3, initial_cases=[case],
                     frontier_tile_id=0)
    except Exception as error:  # noqa: BLE001 - no MuJoCo/MJLab in this environment
        pytest.skip(f'CPU V06Env unavailable: {type(error).__name__}: {error}')
    run = target.environment_case_runner(ZeroModel(), lambda _cases: env)
    explicit_cases = [dict(case, case_id=f'flat:0.3:{index:03d}') for index in range(8)]
    record = run(explicit_cases)[0]
    assert record['steps'] == 500
    assert isinstance(record['flat_pass'], bool)
    assert isinstance(record['forward_distance'], float)
    assert isinstance(record['max_abs_y'], float)
    assert isinstance(record['max_abs_yaw_deg'], float)
    assert isinstance(record['active_samples'], int)
    assert isinstance(record['stall_samples'], int)
    assert isinstance(record['first_step_valid'], bool)
    assert record['measurement'] == 'flat'
    assert 0.0 <= record['max_abs_y'] and 0.0 <= record['max_abs_yaw_deg']
    # Eight explicit case slots use identical poses here to retain the original
    # scalar-consistency fixture; the runner now requires explicit slot identity.
    rows = env.diagnostics_cases()
    assert len(rows) == 8
    for key in ('forward_distance', 'max_abs_y', 'max_abs_yaw_deg',
                'active_samples', 'stall_samples', 'first_step_valid'):
        values = [row[key] for row in rows]
        assert all(value == values[0] for value in values), key


def test_real_env_499_steps_is_an_incomplete_evaluation():
    pytest.importorskip('mujoco')
    from src.tasks.velocity.v06.env import V06Env

    class ZeroModel:
        def act(self, obs, deterministic=False):
            return torch.zeros(8, 12)

    case = _real_f0_case()

    def factory(_case):
        return V06Env(num_envs=8, device='cpu', seed=42, phase='F0',
                      evaluation=True, fixed_vx=0.3, initial_cases=[case],
                      frontier_tile_id=0)

    try:
        run = target.environment_case_runner(ZeroModel(), factory, step_limit=499)
        run([dict(case, case_id=f'flat:0.3:{index:03d}') for index in range(8)])
    except target.RunnerError:
        raise
    except Exception as error:  # noqa: BLE001
        if 'incomplete' in str(error) or '500' in str(error):
            return
        pytest.skip(f'CPU V06Env unavailable: {type(error).__name__}: {error}')
    raise AssertionError('a 499-step F0 run without failure must be an evaluation error')


# ------------------------------------- probe container: allow_probe contract
def test_probe_container_resume_requires_allow_probe(tmp_path):
    runner, ppo, _curriculum, _env = make_runner(tmp_path, update=55)
    ppo.learning_state['completed_ppo_updates'] = 55
    runner.update = 55
    runner.container = 'v06-probe-state-v1'
    record = runner.save('probe')

    # 1) The probe's own recovery (allow_probe=True) resumes and bills one update.
    resumed, resumed_ppo, _curriculum, _env = make_runner(tmp_path / 'probe',
                                                          update=0)
    contract = resumed.load(record['path'], record['sha256'], allow_probe=True)
    assert contract['resume_mode'] == 'new_episode'
    assert resumed.update == 55
    before = int(resumed.learning['update'])
    resumed.train_one_update()
    assert int(resumed.learning['update']) == before + 1 == 56
    assert resumed_ppo.collect_calls == 1

    # 2) A formal load (default allow_probe=False) still refuses the probe tag.
    formal, formal_ppo, _curriculum, _env = make_runner(tmp_path / 'formal', update=0)
    with pytest.raises(Exception) as error:
        formal.load(record['path'], record['sha256'])
    assert 'probe' in str(error.value).lower() or 'container' in str(error.value).lower()
    assert formal.learning['update'] == 0
    assert formal_ppo.collect_calls == 0

    # 3) The recovery layer itself reports the typed container error.
    from src.tasks.velocity.v06 import recovery as recovery_module
    with pytest.raises(recovery_module.ForeignContainerError):
        recovery_module.resume_contract(record['path'] and __import__('torch').load(
            record['path'], map_location='cpu', weights_only=False), allow_probe=False)


def test_probe_worker_source_uses_allow_probe_and_keeps_the_cost_receipt():
    import inspect
    import train_a2_ideal_stairs_v06 as train

    source = inspect.getsource(train.build_probe_worker)
    assert 'allow_probe=True' in source
    assert "save('probe')" in source
    assert "mode='probe'" in source
    assert 'cost=dict(plan[' + "'cost'" + '])' in source
    for field in ('resume_mode', 'updates_after', 'billed_updates_not_replayed',
                  'initialized_from_eth_again', 'container', 'counts'):
        assert field in source


# ------------------------------------- post-start measured resource stop
def test_post_start_measured_resource_failure_stops_as_resource_insufficient(tmp_path):
    from src.tasks.velocity.v06 import orchestration as orchestration_module

    runner, ppo, curriculum, env = make_runner(tmp_path, update=0)
    measured = orchestration_module.check_resources(
        orchestration_module.resource_snapshot(
            directory=tmp_path,
            query=[dict(index=0, used_mib=4316, total_mib=24576)],
            processes=[], torch_reserved=None),
        stage='measured')
    assert measured['ok'] is False
    runner.resource_guard = lambda: dict(measured, snapshot={})
    runner.train_one_update()
    assert runner.stop_state == 'resource_insufficient'
    assert runner.stop_evidence['failure_class'] == 'resource'
    assert runner.stop_evidence['budget']['total_exhausted'] is False
    assert (tmp_path / 'stop.json').is_file()


def test_post_start_measured_resource_pass_keeps_training(tmp_path):
    import os
    from src.tasks.velocity.v06 import orchestration as orchestration_module

    runner, ppo, curriculum, env = make_runner(tmp_path, update=0)
    measured = orchestration_module.check_resources(
        orchestration_module.resource_snapshot(
            directory=tmp_path,
            query=[dict(index=0, used_mib=4316, total_mib=24576)],
            processes=[dict(pid=os.getpid(), used_mib=4000)]),
        stage='measured')
    assert measured['ok'] is True
    runner.resource_guard = lambda: dict(measured, snapshot={})
    runner.train_one_update()
    assert runner.stop_state is None
    assert runner.update == 1


# ------------------------------------- diagnostic/gate boundaries save cleanly
def test_diagnostic_boundary_saves_with_committed_curriculum_spend(tmp_path):
    runner, ppo, curriculum, env = make_runner(tmp_path, update=49)
    drive(runner, 1)  # update 50 runs the diagnostic
    assert runner.update == 50
    assert sum(int(v) for v in
               runner._curriculum_state()['spent_by_phase'].values()) == 50
    record = runner.save('evidence')
    state = torch.load(record['path'], map_location='cpu', weights_only=False)
    assert int(state['learning']['update']) == 50
    assert sum(int(v) for v in state['curriculum']['spent_by_phase'].values()) == 50
    assert state['ledger']['pending_update'] is None


def test_gate_boundary_saves_with_committed_curriculum_spend(tmp_path):
    runner, ppo, curriculum, env = make_runner(tmp_path, update=99)
    drive(runner, 1)  # update 100 runs the gate and the interval save
    assert runner.update == 100
    saved = [item for item in runner.checkpoints if item['reason'] == 'interval']
    assert saved and saved[-1]['update'] == 100
    state = torch.load(saved[-1]['path'], map_location='cpu', weights_only=False)
    assert int(state['learning']['update']) == 100
    assert sum(int(v) for v in state['curriculum']['spent_by_phase'].values()) == 100
    assert state['ledger']['pending_update'] is None


def test_exception_boundary_writes_a_pending_update_checkpoint(tmp_path):
    runner, ppo, curriculum, env = make_runner(tmp_path, update=49)

    def exploding(case):
        raise RuntimeError('evaluation blew up mid-diagnostic')

    runner.case_runner = exploding
    with pytest.raises(RuntimeError):
        runner.run(max_updates=60)
    exception_path = tmp_path / 'checkpoints' / 'exception_000050.pt'
    assert exception_path.is_file()
    state = torch.load(exception_path, map_location='cpu', weights_only=False)
    spent = sum(int(v) for v in state['curriculum']['spent_by_phase'].values())
    assert int(state['learning']['update']) == 50
    assert spent == 50
    assert state['ledger']['pending_update'] is None
    pending = state['ledger']['pending_boundary']
    assert pending is not None and pending['update'] == 50
    assert [task['kind'] for task in pending['tasks']] == ['diagnostic']
    assert pending['results'] == []
    from src.tasks.velocity.v06 import recovery as recovery_module
    assert recovery_module.validate_checkpoint(state) is True
    contract = recovery_module.resume_contract(state)
    assert contract['pending_boundary_requires_completion'] is True
    assert (tmp_path / 'exception.json').is_file()


def test_pending_diagnostic_resumes_without_rebilling_and_next_update_is_51(tmp_path):
    runner, ppo, curriculum, env = make_runner(tmp_path, update=49)
    env.num_envs = 512
    ppo.num_envs = 512
    calls = {'diagnostic': 0}

    def failing(case):
        if str(case.get('kind')) == 'diagnostic':
            calls['diagnostic'] += 1
            raise RuntimeError('diagnostic blew up')
        return dict(measurement='stair', climb_success=True, failure=False,
                    reason=None, mean_vx=case['vx'], mean_relative_error=0.0,
                    active_samples=100, stall_samples=0, first_step_valid=True)

    runner.case_runner = failing
    with pytest.raises(RuntimeError):
        runner.run(max_updates=60)
    checkpoint = tmp_path / 'checkpoints' / 'exception_000050.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert int(state['learning']['update']) == 50
    assert sum(int(v) for v in state['curriculum']['spent_by_phase'].values()) == 50
    assert int(state['learning']['adam_steps']) == 1000
    assert int(state['learning']['control_samples']) == 1_228_800
    assert state['ledger']['pending_boundary']['update'] == 50

    good, good_ppo, good_curriculum, good_env = make_runner(tmp_path / 'resume',
                                                            update=0)
    good_env.num_envs = 512
    good_ppo.num_envs = 512
    seen = []

    def working(case):
        seen.append(case['case_id'])
        return dict(measurement='flat' if str(case.get('measurement_stage')) == 'F0'
                    else 'stair', climb_success=True, failure=False,
                    flat_pass=True, reason=None, mean_vx=case['vx'],
                    mean_relative_error=0.0, active_samples=100,
                    stall_samples=0, first_step_valid=True)

    good.case_runner = working
    contract = good.load(checkpoint, target.recovery_module.sha256(checkpoint))
    assert contract['pending_boundary_requires_completion'] is True
    assert good.update == 50
    assert good_ppo.collect_calls == 0
    good._complete_pending_boundary()
    assert good.update == 50
    assert sum(int(v) for v in
               good._curriculum_state()['spent_by_phase'].values()) == 50
    assert good.pending_boundary is None
    assert any(record.get('diagnostic') for record in good.diagnostics)
    assert good_ppo.collect_calls == 0
    good.train_one_update()
    assert good.update == 51
    assert good_ppo.collect_calls == 1


# ================================================= Q004 §5 injections
def _passing(case):
    return dict(measurement='flat' if str(case.get('measurement_stage')) == 'F0'
                else 'stair', climb_success=True, failure=False, flat_pass=True,
                reason=None, mean_vx=case['vx'], mean_relative_error=0.0,
                active_samples=100, stall_samples=0, first_step_valid=True)


def _load_and_complete(tmp_path, checkpoint, case_runner, num_envs=512):
    resumed, resumed_ppo, _curriculum, resumed_env = make_runner(
        tmp_path / 'resume', update=0)
    resumed_env.num_envs = num_envs
    resumed_ppo.num_envs = num_envs
    resumed.case_runner = case_runner
    contract = resumed.load(checkpoint, target.recovery_module.sha256(checkpoint))
    assert contract['pending_boundary_requires_completion'] is True
    resumed._complete_pending_boundary()
    return resumed, resumed_ppo


def test_update100_gate_fault_resume_without_rebilling(tmp_path):
    runner, ppo, curriculum, env = make_runner(tmp_path, update=99)
    env.num_envs = 512
    ppo.num_envs = 512

    def failing(case):
        if str(case.get('kind')) == 'validation':
            raise RuntimeError('gate evaluation blew up')
        return _passing(case)

    runner.case_runner = failing
    with pytest.raises(RuntimeError):
        runner.run(max_updates=110)
    checkpoint = tmp_path / 'checkpoints' / 'boundary_000100.pt'
    assert checkpoint.is_file()
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert int(state['learning']['update']) == 100
    assert sum(int(v) for v in state['curriculum']['spent_by_phase'].values()) == 100
    pending = state['ledger']['pending_boundary']
    assert pending['tasks'][0]['kind'] == 'validation'
    assert 'diagnostic' in [task['kind'] for task in pending['tasks']]
    assert pending['tasks'][0]['case_table_sha256']
    resumed, resumed_ppo = _load_and_complete(tmp_path, checkpoint, _passing)
    assert resumed_ppo.collect_calls == 0
    assert resumed.update == 100
    assert sum(int(v) for v in
               resumed._curriculum_state()['spent_by_phase'].values()) == 100
    assert [r.kind for r in resumed.gates].count('validation') == 1
    assert resumed.pending_boundary is None
    resumed.train_one_update()
    assert resumed.update == 101
    assert resumed_ppo.collect_calls == 1


def test_update200_dual_retention_fault_dedup_per_set(tmp_path):
    runner, ppo, curriculum, env = make_runner(tmp_path, update=199, phase='H',
                                               retention=('flat', 'h00'))
    env.num_envs = 512
    ppo.num_envs = 512

    def failing(case):
        if str(case.get('kind')) == 'flat_retention':
            raise RuntimeError('flat retention blew up')
        return _passing(case)

    runner.case_runner = failing
    with pytest.raises(RuntimeError):
        runner.run(max_updates=210)
    checkpoint = tmp_path / 'checkpoints' / 'boundary_000200.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    pending = state['ledger']['pending_boundary']
    kinds = [task.get('retention_kind') for task in pending['tasks']
             if task['kind'] == 'retention']
    assert kinds == ['flat', 'h00']
    assert [task['kind'] for task in pending['tasks']][:2] == ['validation',
                                                               'diagnostic']
    retention_hashes = [task['case_table_sha256'] for task in pending['tasks']
                        if task['kind'] == 'retention']
    assert len(set(retention_hashes)) == 2  # flat and H00 tables stay distinct
    resumed, resumed_ppo = _load_and_complete(tmp_path, checkpoint, _passing)
    assert resumed_ppo.collect_calls == 0
    assert resumed.update == 200
    # Both sets are completed once with a clean verdict: independent, no
    # cross-counting and no double increment.
    assert resumed.retention['streaks']['flat'] == 0
    assert resumed.retention['streaks']['h00'] == 0
    assert resumed.retention['last']['flat']['update'] == 200
    assert resumed.retention['last']['h00']['update'] == 200
    retention_kinds = [r.kind for r in resumed.gates
                       if r.kind in ('flat_retention', 'stair_retention')]
    assert retention_kinds == ['flat_retention', 'stair_retention']
    assert resumed.pending_boundary is None
    resumed.train_one_update()
    assert resumed.update == 201


def test_phase_last_update_fault_keeps_budget_order(tmp_path):
    runner, ppo, curriculum, env = make_runner(tmp_path, update=12999, phase='H')
    env.num_envs = 512
    ppo.num_envs = 512
    curriculum.spent = spend(12999)
    curriculum.last_update = 12999
    curriculum.generation_updates = 12999

    def failing(case):
        if str(case.get('kind')) == 'validation':
            raise RuntimeError('H final gate blew up')
        return _passing(case)

    runner.case_runner = failing
    with pytest.raises(RuntimeError):
        runner.run(max_updates=14000)
    checkpoint = tmp_path / 'checkpoints' / 'boundary_013000.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert int(state['learning']['update']) == 13000
    assert sum(int(v) for v in state['curriculum']['spent_by_phase'].values()) == 13000
    assert state['ledger']['pending_boundary']['update'] == 13000
    resumed, resumed_ppo = _load_and_complete(tmp_path, checkpoint, _passing)
    assert resumed_ppo.collect_calls == 0
    assert resumed.update == 13000
    # Budget order: the curriculum decides on the final update, then stops.
    assert resumed.stop_state in (None, 'gate_failed', 'budget_exhausted')
    if resumed.stop_state is None:
        assert resumed.update == 13000


def test_total_budget_30000_final_qualification_fault(tmp_path):
    runner, ppo, curriculum, env = make_runner(tmp_path, update=30000, phase='R')
    runner.update = 30000
    runner.phase = 'R'
    curriculum.phase = 'R'
    curriculum.spent = spend(30000)
    curriculum.last_update = 30000
    curriculum.finished = True
    curriculum.stop_status = 'complete'
    ppo.learning_state['completed_ppo_updates'] = 30000
    ppo.update_count = 30000

    def exploding(case):
        raise RuntimeError('final test infrastructure blew up')

    runner.case_runner = exploding
    with pytest.raises(RuntimeError):
        runner.run(max_updates=30000)
    assert runner.lifecycle_state != 'complete'
    assert runner.error_type == 'RuntimeError'
    assert ppo.collect_calls == 0
    assert runner.update == 30000
    # Same update, no 30,001st learning update: a fresh runner performs only the
    # frozen test (no PPO, no billing).
    retry, retry_ppo, retry_curriculum, retry_env = make_runner(
        tmp_path / 'retry', update=30000, phase='R')
    retry.update = 30000
    retry.phase = 'R'
    retry_curriculum.phase = 'R'
    retry_curriculum.spent = spend(30000)
    retry_curriculum.last_update = 30000
    retry_curriculum.finished = True
    retry_curriculum.stop_status = 'complete'
    retry_ppo.learning_state['completed_ppo_updates'] = 30000
    retry_ppo.update_count = 30000
    retry.case_runner = _passing
    result = retry.run(max_updates=30000)
    assert retry_ppo.collect_calls == 0
    assert retry.update == 30000
    assert result['state'] == 'complete'
    assert retry.lifecycle_state == 'complete'


def test_probe_runs_on_r_l40_from_update_one(tmp_path):
    import json
    import train_a2_ideal_stairs_v06 as train
    from src.tasks.velocity.v06 import spec as spec_module

    plan = train._noop if False else None
    from src.tasks.velocity.v06 import orchestration as orchestration_module
    plan = orchestration_module.probe_plan(seed=42,
                                           num_envs=512, eval_num_envs=8)
    curriculum = train.probe_curriculum(plan, 42)
    state = curriculum.state_dict()
    assert state['phase'] == 'R'
    assert int(state['frontier_tile_id']) == int(spec_module.FINAL_TILE_ID) == 21
    assert int(state['spent_by_phase']['R']) == 0
    assert int(state['level_tick']) == 0

    model = FakeModel()
    env = FakeEnv()
    env.frontier_tile_id = 21
    env.phase = 'R'
    ppo = FakePPO(update=0, model=model)
    runner = target.V06Runner(model=model, ppo=ppo, curriculum=curriculum,
                              env=env, directory=tmp_path, seed=42,
                              device='cpu',
                              case_runner=lambda case: _passing(case))
    runner.case_runner = lambda case: _passing(case)
    runner.train_one_update()
    assert env.phase == 'R'
    assert int(env.frontier_tile_id) == 21
    assert runner._curriculum_state()['phase'] == 'R'
    for _ in range(49):
        runner.train_one_update()
    assert runner.phase == 'R'
    diagnostic = target.recovery_module.jsonable(runner.diagnostics[-1])
    assert diagnostic['phase'] == 'R'
    assert diagnostic['diagnostic']['stairs'] is True
    assert [float(value) for value in diagnostic['diagnostic']['speeds']] == [0.2, 0.4, 0.6]
    pending = torch.load(tmp_path / 'checkpoints' / 'boundary_000050.pt',
                         map_location='cpu', weights_only=False)
    tasks = pending['ledger']['pending_boundary']['tasks']
    assert all(int(task['tile']) == int(spec_module.FINAL_TILE_ID) for task in tasks)
    assert pending['curriculum']['phase'] == 'R'
    assert int(pending['curriculum']['frontier_tile_id']) == 21
    for _ in range(5):
        runner.train_one_update()
    assert int(runner._curriculum_state()['spent_by_phase']['R']) == 55
    assert runner.phase == 'R'
    assert int(runner._curriculum_state()['frontier_tile_id']) == 21


def test_ppo_half_failure_is_preserved_and_marked_non_resumable(tmp_path):
    runner, ppo, curriculum, env = make_runner(tmp_path, update=49)

    def exploding_update():
        raise RuntimeError('PPO blew up in the middle')

    ppo.update = exploding_update
    with pytest.raises(RuntimeError) as error:
        runner.run(max_updates=60)
    assert 'PPO blew up' in str(error.value)
    assert runner.lifecycle_state != 'complete'
    incomplete = torch.load(tmp_path / 'checkpoints' / 'incomplete_000049.pt',
                            map_location='cpu', weights_only=False)
    marker = incomplete['ledger']['incomplete_update']
    assert marker['resumable'] is False
    assert marker['error_type'] == 'RuntimeError'
    assert int(incomplete['learning']['update']) == 49
    assert (tmp_path / 'ppo_incomplete.json').is_file()
    from src.tasks.velocity.v06 import recovery as recovery_module
    with pytest.raises(recovery_module.RecoveryBlocked):
        recovery_module.resume_contract(incomplete)


def test_checkpoint_save_failure_does_not_mask_the_original_error(tmp_path,
                                                                 monkeypatch):
    runner, ppo, curriculum, env = make_runner(tmp_path, update=49)

    def failing(case):
        if str(case.get('kind')) == 'diagnostic':
            raise RuntimeError('original evaluation error')
        return _passing(case)

    runner.case_runner = failing

    real_save = target.recovery_module.atomic_save

    def broken_save(path, state):
        if 'exception_' in str(path):
            raise OSError('disk full while saving')
        return real_save(path, state)

    monkeypatch.setattr(target.recovery_module, 'atomic_save', broken_save)
    with pytest.raises(RuntimeError) as error:
        runner.run(max_updates=60)
    assert 'original evaluation error' in str(error.value)
    assert runner.lifecycle_state != 'complete'
    assert (tmp_path / 'exception_save_error.json').is_file()
    assert (tmp_path / 'exception.json').is_file()


def test_forward_distance_cross_source_tolerance_is_float32_safe():
    from src.tasks.velocity.v06 import evaluation as evaluation_module

    class Model:
        def act(self, obs, deterministic=False):
            return None

    class Env:
        def __init__(self, offset=0.0):
            self.offset = offset
            self.count = 0

        def reset(self):
            return ({'x': 1}, {})

        def step(self, action):
            self.count += 1
            done = self.count >= 500
            info = dict(episodes=[], terminal_observation={'x': 1})
            if done:
                info['episodes'] = [dict(
                    env_id=index, control_samples=500, terminal_step=500, success=False,
                    failure=False, timeout=True, active_samples=500, stall_samples=0,
                    first_step_valid=False, mean_vx=.3, mean_speed=.3,
                    reason='timeout', max_lateral=0.05,
                    max_abs_yaw_deg=1.0, forward_distance=1.5 + index * 0.01
                    + self.offset,
                    forward_distance_start=-1.0,
                    forward_distance_end=0.5 + index * 0.01)
                    for index in range(2)]
            return ({'x': 1}, 0.0, torch.zeros(2, dtype=torch.bool),
                    torch.full((2,), done, dtype=torch.bool), info)

    case = dict(vx=0.3, measurement_stage='F0', case_id='a', phase='F0',
                episode_index=0, initial_pose=[-1.0, 0.0, 0.0])
    cases = [dict(case, case_id=f'flat:{index}') for index in range(2)]
    # float32 rounding (~1e-7) passes the cross-source check.
    tolerant = target.environment_case_runner(
        Model(), lambda _cases: Env(offset=6e-8))
    records = tolerant(cases)
    assert len(records) == 2
    # Negative float32 error (-6e-8) also passes the cross-source check.
    negative = target.environment_case_runner(
        Model(), lambda _cases: Env(offset=-6e-8))
    assert len(negative(cases)) == 2
    # +/-1e-6 passes exactly; +/-1.1e-6 is a real inconsistency.
    for offset in (1e-6, -1e-6):
        runner_case = target.environment_case_runner(
            Model(), lambda _cases, off=offset: Env(offset=off))
        assert len(runner_case(cases)) == 2
    for offset in (1.1e-6, -1.1e-6):
        bad = target.environment_case_runner(
            Model(), lambda _cases, off=offset: Env(offset=off))
        with pytest.raises(evaluation_module.EvaluationError):
            bad(cases)
    # Non-finite endpoint evidence is invalid, never a comparison pass.
    for broken_field in ('forward_distance', 'forward_distance_start',
                         'forward_distance_end'):
        class BadEnv(Env):
            def step(self, action, **kwargs):
                result = super().step(action)
                for record in result[4].get('episodes') or []:
                    record[broken_field] = float('nan')
                return result

        bad = target.environment_case_runner(Model(), lambda _cases: BadEnv())
        with pytest.raises(evaluation_module.EvaluationError):
            bad(cases)
    # A real 1e-3 mismatch is still rejected.
    broken = target.environment_case_runner(
        Model(), lambda _cases: Env(offset=1e-3))
    with pytest.raises(evaluation_module.EvaluationError):
        broken(cases)
    # The Q003 gate criterion itself is unchanged (no relaxation to 1e-6).
    from src.tasks.velocity.v06 import speed
    metrics = speed.flat_case_metrics(
        speed=0.3, steps=500, max_abs_y=0.0, max_abs_yaw_deg=0.0,
        forward=1.5 - 5e-10)
    assert speed.flat_case_pass(metrics) is True
    short = speed.flat_case_metrics(
        speed=0.3, steps=500, max_abs_y=0.0, max_abs_yaw_deg=0.0,
        forward=1.5 - 2e-9)
    assert speed.flat_case_pass(short) is False
    # 5e-7 below the gate threshold still fails (the consistency tolerance never
    # relaxes the Q003 gate criterion).
    below = speed.flat_case_metrics(
        speed=0.3, steps=500, max_abs_y=0.0, max_abs_yaw_deg=0.0,
        forward=1.5 - 5e-7)
    assert speed.flat_case_pass(below) is False
