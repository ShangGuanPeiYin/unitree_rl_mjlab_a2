"""Q006 CPU contract: ETH metadata recovery, verified resume evidence and the
30100 formal-cost ledger.

The tests exercise the *production* resume entry as far as a CPU can: a bare
``V06ActorCritic`` (no metadata, no optimizer state), ``recovery.preload_checkpoint``
(the worker preload), ``V06Runner.load``, an atomic save and a formal
``preload_checkpoint`` reload into an independent bare model plus a fresh single
Adam.  The PPO/environment doubles only avoid GPU rollouts; every counter, Adam
entry, normalizer and curriculum value still flows through the real
runner/recovery/evaluation code, and no learning count is edited by hand.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tests'))
import a2_v06_bootstrap  # noqa: E402

a2_v06_bootstrap.activate(ROOT)

from copy import deepcopy  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

from src.tasks.velocity.v06 import ledger as ledger_module  # noqa: E402
from src.tasks.velocity.v06 import model as model_module  # noqa: E402
from src.tasks.velocity.v06 import ppo as ppo_module  # noqa: E402
from src.tasks.velocity.v06 import recovery as recovery_module  # noqa: E402
from src.tasks.velocity.v06 import runner as runner_module  # noqa: E402
from src.tasks.velocity.v06 import speed  # noqa: E402
from src.tasks.velocity.v06 import spec  # noqa: E402

CHECKPOINT_LR = 8.888888888888888e-05
REAL_BOUNDARY = (ROOT / 'logs/a2_ideal_stairs_v06/train_seed42_trial02/checkpoints/'
                 'boundary_000100.pt')
PHASE_ORDER = ('F0', 'H', 'D', 'L', 'V', 'R')


def spend(update):
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


class Q006Env:
    device = 'cpu'
    frontier_tile_id = 1

    def __init__(self, num_envs=512):
        self.num_envs = int(num_envs)
        self.resets = 0
        self.schedules = 0

    def reset(self):
        self.resets += 1
        return ({'proprio': torch.zeros(self.num_envs)}, {})

    def set_schedule(self, schedule, iteration=None):
        self.schedules += 1
        return schedule

    def rng_state(self):
        return {'env': [1, 2, 3]}

    def sampler_rng_state(self):
        return {'sampler': [4, 5, 6]}


class Q006Curriculum:
    def __init__(self, update=0):
        self.phase = phase_for(int(update))
        self.substage = None
        self.update = int(update)
        self.last_update = int(update)
        self.generation_updates = int(update)
        self.spent = spend(int(update))
        self.gates = []
        self.observed = 0

    def state_dict(self):
        spent = dict(self.spent)
        spent.setdefault(self.phase, 0)
        return dict(phase=self.phase, substage=self.substage, level_tick=0,
                    generation=0, generation_updates=self.generation_updates,
                    last_update=self.last_update, finished=False, stop_status=None,
                    spent_by_phase=spent, spent_by_substage={}, episode_fifo=[],
                    ema=None, ready_streak=0, hard_streak=0, cooldown=0,
                    gate_streak=0, retention_streaks={},
                    minimum_level_failure_streak=0, sampler_rng_state={})

    def load_state_dict(self, value):
        self.phase = value['phase']
        self.substage = value.get('substage')
        self.spent = dict(value['spent_by_phase'])
        self.last_update = int(value.get('last_update', self.last_update))
        self.generation_updates = int(value.get('generation_updates',
                                                self.generation_updates))

    def schedule(self):
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
        return []

    def retention_due(self, update):
        return None

    def record_gate(self, passed, update, kind):
        self.gates.append(dict(passed=passed, update=update, kind=kind))


class Q006PPO:
    """A real single Adam over the real model; only rollouts are synthetic."""

    def __init__(self, model, *, update=0, num_envs=512):
        self.model = model
        self.num_envs = int(num_envs)
        self.optimizer = ppo_module.build_optimizer(model)
        self.learning_state = dict(optimizer_initialized=True,
                                   completed_ppo_updates=int(update),
                                   optimizer_steps=int(update) * 20)
        self.collect_calls = 0
        self.adam_steps = 0
        for group in self.optimizer.param_groups:
            group['lr'] = CHECKPOINT_LR
        self._adam_step()

    def _adam_step(self):
        # A real Adam step without autograd: the state tensors are materialized
        # exactly as a training step would materialize them.
        for parameter in self.model.parameters():
            if parameter.requires_grad:
                parameter.grad = torch.full_like(parameter, 1e-3)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.adam_steps += 1

    @property
    def learning(self):
        update = int(self.learning_state['completed_ppo_updates'])
        samples = update * self.num_envs * int(ppo_module.ROLLOUT)
        return dict(update=update, optimizer_step=update * 20,
                    adam_steps=update * 20, control_samples=samples,
                    physics_substeps=ppo_module.physics_substeps(samples))

    def reset(self):
        return ({'proprio': torch.zeros(self.num_envs)}, {})

    def collect(self):
        self.collect_calls += 1
        return []

    def update(self):
        self.update_count = int(self.learning_state['completed_ppo_updates']) + 1
        self.learning_state['completed_ppo_updates'] = self.update_count
        self.learning_state['optimizer_steps'] = self.update_count * 20
        self.learning_state['optimizer_initialized'] = True
        self._adam_step()
        return dict(loss=1.0, kl=0.01, update=self.update_count)


def passing(case):
    return dict(measurement='flat' if str(case.get('measurement_stage')) == 'F0'
                else 'stair', climb_success=True, failure=False, flat_pass=True,
                reason=None, mean_vx=case['vx'], mean_relative_error=0.0,
                active_samples=100, stall_samples=0, first_step_valid=True)


def make_runner(directory, *, update=0, num_envs=512, require_resume_evidence=False,
                container=None, evaluate=True, telemetry=None, model=None,
                case_runner=passing):
    model = model if model is not None else model_module.V06ActorCritic.from_eth(
        seed=42, device='cpu')
    env = Q006Env(num_envs=num_envs)
    ppo = Q006PPO(model, update=update, num_envs=num_envs)
    curriculum = Q006Curriculum(update=update)
    runner = runner_module.V06Runner(
        model=model, ppo=ppo, curriculum=curriculum, env=env, directory=directory,
        seed=42, device='cpu', case_runner=case_runner, container=container,
        evaluate=evaluate, require_resume_evidence=require_resume_evidence,
        telemetry=telemetry)
    return runner, ppo, curriculum, env


def drive(runner, count):
    for _ in range(int(count)):
        runner.train_one_update()


# ------------------------------------------------------- metadata round trip
def test_bare_model_load_save_roundtrip_and_metadata_deepcopy(tmp_path):
    """The Q006 defect: a bare model cannot save; the fixed load can and verifies."""
    source, source_ppo, _curriculum, _env = make_runner(tmp_path / 'source')
    drive(source, 100)
    record = source.save('source')
    source_state = torch.load(record['path'], map_location='cpu', weights_only=False)
    assert source_state['learning']['update'] == 100
    assert source_state['initialization']['transfer_metadata']['sha256'] == \
        model_module.INIT_SHA256

    # Old behaviour: load the weights into a bare model (no metadata adoption),
    # then a checkpoint save refuses the run (this is the original reported bug).
    bare = model_module.V06ActorCritic()
    old_runner, old_ppo, _c, _e = make_runner(tmp_path / 'old', model=bare)
    preloaded = recovery_module.preload_checkpoint(record['path'], record['sha256'],
                                                   mode='resume')
    recovery_module.load_model_state(old_runner.model, preloaded['model'])
    assert not getattr(old_runner.model, 'transfer_metadata', None)
    with pytest.raises(runner_module.RunnerError):
        old_runner.save('legacy')

    # New behaviour: the worker preload -> mirror runner.load -> save -> formal
    # recovery reload carries the ETH metadata and every training state.
    resumed, resumed_ppo, _curriculum2, _env2 = make_runner(
        tmp_path / 'resume', require_resume_evidence=True,
        model=model_module.V06ActorCritic())
    assert not getattr(resumed.model, 'transfer_metadata', None)
    contract = resumed.load(record['path'], record['sha256'])
    assert contract['replay_billed_updates'] is False
    metadata = resumed.model.transfer_metadata
    assert metadata['sha256'] == model_module.INIT_SHA256
    assert metadata['source_actor_keys']
    assert metadata is not source_state['initialization']['transfer_metadata']
    # Deep copy: mutating the restored metadata cannot reach the source object.
    metadata['source_actor_keys'].append('mutated')
    assert 'mutated' not in source_state['initialization']['transfer_metadata'][
        'source_actor_keys']
    metadata['source_actor_keys'].pop()

    saved = resumed.save_verified('resume_loaded')
    assert saved['verification']['aligned'] is True
    checks = saved['verification']['checks']
    assert checks['model_tensors_match'] and checks['optimizer_state_match']
    assert checks['normalizer_match'] and checks['learning_rate_match']
    assert checks['metadata_deepcopied'] and checks['metadata_match']
    assert checks['active_model_unchanged'] and checks['active_rng_unchanged']
    assert checks['counters']['update'] == 100
    payload = torch.load(saved['path'], map_location='cpu', weights_only=False)
    assert payload['initialization']['resume_origin']['source_checkpoint'] == \
        record['sha256']
    assert payload['initialization']['transfer_metadata']['sha256'] == \
        model_module.INIT_SHA256
    assert payload['optimizer']['state']
    assert payload['normalizer'] and payload['rng']['env'] is not None
    assert sum(int(v) for v in payload['curriculum']['spent_by_phase'].values()) == 100
    assert abs(float(payload['learning']['learning_rate']) - CHECKPOINT_LR) <= 0
    # Active learning rate stayed the checkpoint value (never reset to 3e-4).
    assert float(resumed.ppo.optimizer.param_groups[0]['lr']) == CHECKPOINT_LR
    assert float(payload['optimizer']['param_groups'][0]['lr']) == CHECKPOINT_LR
    assert resumed_ppo.collect_calls == 0

    # The saved file is a legitimate second resume source under the same source.
    second, _second_ppo, _c3, _e3 = make_runner(
        tmp_path / 'second', model=model_module.V06ActorCritic())
    contract2 = second.load(saved['path'], saved['sha256'])
    assert contract2['replay_billed_updates'] is False
    assert second.update == 100
    second.save_verified('resume_loaded')
    assert second.model.transfer_metadata['sha256'] == model_module.INIT_SHA256


# ------------------------------------------------- metadata must be admissible
@pytest.mark.parametrize('kind', ['missing', 'forged', 'architecture',
                                 'route_fusion', 'scope'])
def test_missing_or_forged_metadata_is_refused_before_learning(tmp_path, kind):
    source, _ppo, _c, _e = make_runner(tmp_path / 'source')
    record = source.save('update0')
    state = torch.load(record['path'], map_location='cpu', weights_only=False)
    state.pop('checkpoint_sha256', None)
    metadata = state['initialization']['transfer_metadata']
    if kind == 'missing':
        state['initialization']['transfer_metadata'] = {}
    elif kind == 'forged':
        metadata['sha256'] = 'a' * 64
    elif kind == 'architecture':
        metadata['architecture'] = 'v05-route-state-v0'
    elif kind == 'route_fusion':
        metadata['route_fusion'] = 'random_init'
    elif kind == 'scope':
        metadata['scope'] = 'resume'
    path = tmp_path / f'tampered_{kind}.pt'
    torch.save(state, path)
    resumed, resumed_ppo, _curriculum, _env = make_runner(tmp_path / 'resume')
    with pytest.raises((ValueError, recovery_module.RecoveryBlocked,
                        recovery_module.ForeignContainerError)):
        resumed.load(path)
    assert resumed_ppo.collect_calls == 0
    assert resumed.update == 0
    assert not (tmp_path / 'resume' / 'checkpoints' /
                'resume_loaded_000000.pt').exists()


# ------------------------------------- real 100 pending -> ready -> no replay
def _pending_100_runner(tmp_path):
    runner, ppo, _curriculum, _env = make_runner(tmp_path / 'source', update=99)

    def failing(case):
        if str(case.get('kind')) == 'validation':
            raise RuntimeError('gate evaluation blew up')
        return passing(case)

    runner.case_runner = failing
    with pytest.raises(RuntimeError):
        runner.run(max_updates=100)
    boundary = tmp_path / 'source' / 'checkpoints' / 'boundary_000100.pt'
    assert boundary.is_file()
    return runner, ppo, boundary


def test_real_100_pending_resume_ready_and_no_double_streak(tmp_path):
    _source, _source_ppo, boundary = _pending_100_runner(tmp_path)
    resumed, resumed_ppo, _curriculum, _env = make_runner(
        tmp_path / 'resume', require_resume_evidence=True)
    resumed.load(boundary, recovery_module.sha256(boundary))
    assert resumed.pending_boundary is not None
    loaded = resumed.save_verified('resume_loaded')
    assert loaded['update'] == 100 and loaded['verification']['aligned']
    result = resumed.run(max_updates=100)
    assert resumed.pending_boundary is None
    assert resumed.resume_ready is not None
    ready = resumed.resume_ready
    assert ready['update'] == 100 and ready['verification']['aligned']
    ready_state = torch.load(ready['path'], map_location='cpu', weights_only=False)
    assert ready_state['ledger']['pending_boundary'] is None
    assert int(ready_state['learning']['update']) == 100
    assert sum(int(v) for v in
               ready_state['curriculum']['spent_by_phase'].values()) == 100
    assert resumed.gate_streak == 1
    assert [record.kind for record in resumed.gates].count('validation') == 1
    assert len(resumed.diagnostics) == 1
    assert int(resumed.diagnostics[-1]['update']) == 100
    assert resumed_ppo.collect_calls == 0
    assert result['update'] == 100

    # Second load of the ready evidence: the committed 100 transaction is never
    # re-run and the streak is not incremented twice.
    second, second_ppo, _c2, _e2 = make_runner(tmp_path / 'second',
                                                require_resume_evidence=True)
    second.load(ready['path'], ready['sha256'])
    assert second.update == 100 and second.pending_boundary is None
    second.save_verified('resume_loaded')
    assert second.gate_streak == 1
    assert [record.kind for record in second.gates].count('validation') == 1
    second.train_one_update()
    assert second.update == 101
    assert second_ppo.collect_calls == 1


def test_write_failure_keeps_the_last_verified_loaded_evidence(tmp_path, monkeypatch):
    _source, _source_ppo, boundary = _pending_100_runner(tmp_path)
    resumed, _ppo, _c, _e = make_runner(tmp_path / 'resume',
                                        require_resume_evidence=True)
    resumed.load(boundary, recovery_module.sha256(boundary))
    loaded = resumed.save_verified('resume_loaded')
    original = recovery_module.atomic_save

    def flaky(path, state):
        if Path(path).name.startswith('resume_ready'):
            raise OSError('simulated disk failure')
        return original(path, state)

    monkeypatch.setattr(recovery_module, 'atomic_save', flaky)
    with pytest.raises(OSError):
        resumed.save_verified('resume_ready')
    assert resumed.resume_ready is None
    assert not (tmp_path / 'resume' / 'checkpoints' /
                'resume_ready_000100.pt').exists()
    # The earlier loaded evidence is intact and still formally recoverable.
    recovery_module.preload_checkpoint(loaded['path'], loaded['sha256'],
                                       mode='resume')
    assert resumed.resume_loaded['sha256'] == loaded['sha256']


# -------------------------------------------- 150 diagnostic / 200 gate saves
class RaisingTelemetry:
    def update(self, metrics, boundary):
        raise RuntimeError('telemetry transport blew up after the boundary')

    def gate(self, record):
        return None

    def retention(self, record):
        return None

    def start(self, *args, **kwargs):
        return None

    def finish(self, *args, **kwargs):
        return None


def test_150_update_counts_after_telemetry_failure(tmp_path):
    runner, _ppo, _curriculum, _env = make_runner(
        tmp_path, update=149, telemetry=RaisingTelemetry())
    with pytest.raises(RuntimeError):
        runner.run(max_updates=150)
    # The pre-diagnostic boundary checkpoint exists (the save that failed on the
    # original machine) and the completed PPO update is billed regardless of the
    # missing telemetry record.
    boundary = tmp_path / 'checkpoints' / 'boundary_000150.pt'
    assert boundary.is_file()
    state = torch.load(boundary, map_location='cpu', weights_only=False)
    assert int(state['learning']['update']) == 150
    assert int(state['learning']['adam_steps']) == 3000
    assert int(state['learning']['control_samples']) == 3_686_400
    assert sum(int(v) for v in state['curriculum']['spent_by_phase'].values()) == 150
    assert state['ledger']['pending_boundary'] is not None
    assert (tmp_path / 'commit_000150.json').is_file()
    exception = torch.load(tmp_path / 'checkpoints' / 'exception_000150.pt',
                           map_location='cpu', weights_only=False)
    assert int(exception['learning']['update']) == 150
    assert exception['ledger']['pending_update'] is None
    assert runner.update == 150


def test_200_gate_boundary_saves_a_recoverable_checkpoint(tmp_path):
    runner, _ppo, _curriculum, _env = make_runner(tmp_path, update=199)
    drive(runner, 1)
    gate = tmp_path / 'checkpoints' / 'gate_000200.pt'
    assert gate.is_file()
    state = torch.load(gate, map_location='cpu', weights_only=False)
    assert int(state['learning']['update']) == 200
    assert sum(int(v) for v in state['curriculum']['spent_by_phase'].values()) == 200
    assert state['ledger']['pending_boundary'] is None
    assert recovery_module.validate_checkpoint(state) is True
    contract = recovery_module.resume_contract(state)
    assert contract['update'] == 200


# ------------------------------------------------------ source / closure gate
def test_source_closure_mismatch_is_refused_and_new_files_pass(tmp_path):
    source, _ppo, _c, _e = make_runner(tmp_path / 'source')
    drive(source, 1)
    record = source.save('source')
    state = torch.load(record['path'], map_location='cpu', weights_only=False)
    state.pop('checkpoint_sha256', None)
    state['source']['source_closure_sha256'] = 'b' * 64
    path = tmp_path / 'foreign.pt'
    torch.save(state, path)
    resumed, _ppo2, _c2, _e2 = make_runner(tmp_path / 'resume')
    with pytest.raises(recovery_module.RecoveryBlocked):
        resumed.load(path)

    # A checkpoint saved by the repaired source resumes through the normal
    # source-closure check: no bridge flag and no forged old SHA.
    clean, _ppo3, _c3, _e3 = make_runner(tmp_path / 'clean')
    saved = clean.save_verified('resume_loaded')
    other, _ppo4, _c4, _e4 = make_runner(tmp_path / 'other')
    contract = other.load(saved['path'], saved['sha256'])
    assert contract['bridge'] is None
    assert contract['replay_billed_updates'] is False


# --------------------------------------------------------- Q006 bridge wiring
@pytest.mark.skipif(not REAL_BOUNDARY.is_file(),
                    reason='the documented trial02 boundary checkpoint is absent')
def test_q006_bridge_on_the_real_trial02_boundary(tmp_path):
    actual = recovery_module.sha256(REAL_BOUNDARY)
    assert actual == recovery_module.BRIDGE_CHECKPOINT_SHA256
    with pytest.raises(recovery_module.RecoveryBlocked):
        recovery_module.preload_checkpoint(REAL_BOUNDARY, mode='resume')
    state = recovery_module.preload_checkpoint(REAL_BOUNDARY, actual, mode='resume',
                                               allow_bridge=True)
    assert int(state['learning']['update']) == 100
    assert state['initialization']['transfer_metadata']['sha256'] == \
        model_module.INIT_SHA256
    contract = recovery_module.resume_contract(state, allow_bridge=True)
    bridge = contract['bridge']
    assert bridge['q005_sha256'] == recovery_module.Q005_DECISION_SHA256
    assert bridge['q006_sha256'] == ledger_module.Q006_DECISION_SHA256
    assert bridge['q006_errata_sha256'] == ledger_module.Q006_ERRATA_SHA256
    assert bridge['rollback_from_update'] == 150
    assert bridge['resume_from_update'] == 100
    assert bridge['lost_updates'] == 50
    assert bridge['replay_billed_updates'] is True
    assert contract['replay_billed_updates'] is True
    assert bridge['segment']['segment_id'] == ledger_module.Q006_SEGMENT_ID
    assert bridge['independent_acceptance_sha256'] == \
        ledger_module.Q006_ACCEPTANCE_SHA256

    runner, _ppo, _c, _e = make_runner(tmp_path, require_resume_evidence=True)
    loaded_contract = runner.load(REAL_BOUNDARY, actual, allow_bridge=True)
    assert loaded_contract['replay_billed_updates'] is True
    assert runner.model.transfer_metadata['sha256'] == model_module.INIT_SHA256
    assert runner.model.transfer_metadata['source_path']
    assert float(runner.ppo.optimizer.param_groups[0]['lr']) == \
        pytest.approx(8.888888888888888e-05)
    assert runner.resume_origin['bridge']['q006_sha256'] == \
        ledger_module.Q006_DECISION_SHA256
    saved = runner.save_verified('resume_loaded')
    assert saved['verification']['aligned'] is True
    payload = torch.load(saved['path'], map_location='cpu', weights_only=False)
    assert payload['initialization']['resume_origin']['segment']['segment_id'] == \
        ledger_module.Q006_SEGMENT_ID
    # Completing the committed update-100 transaction must use the recorded case
    # tables (phase/update/generation identity) and leave a verified ready file;
    # update 101 is the first new learning update.
    runner.run(max_updates=101)
    assert runner.resume_ready is not None
    assert int(runner.resume_ready['update']) == 100
    assert runner.pending_boundary is None
    ready_state = torch.load(runner.resume_ready['path'], map_location='cpu',
                             weights_only=False)
    assert ready_state['ledger']['pending_boundary'] is None
    assert int(ready_state['learning']['update']) == 100
    assert [record.kind for record in runner.gates].count('validation') == 1
    assert [int(record['update']) for record in runner.diagnostics].count(100) == 1
    assert runner.update == 101

    # The newly saved file is a normal second-resume source (no bridge, no old SHA).
    second, _ppo2, _c2, _e2 = make_runner(tmp_path / 'second')
    second_contract = second.load(saved['path'], saved['sha256'])
    assert second_contract['bridge'] is None
    assert second.update == 100


def test_q006_bridge_flag_preflight_records_the_segment(tmp_path):
    from src.tasks.velocity.v06 import orchestration as orchestration_module

    if not REAL_BOUNDARY.is_file():
        pytest.skip('the documented trial02 boundary checkpoint is absent')
    manifest = ROOT / ledger_module.Q006_BRIDGE_MANIFEST_PATH
    if not manifest.is_file():
        pytest.skip('the Q006 recovery manifest is absent')
    base = ['--mode', 'train', '--resume-run', str(REAL_BOUNDARY),
            '--allow-q006-bridge']
    output = tmp_path / 'run'
    argv = base + ['--output-dir', str(output)]
    orchestrator = orchestration_module.V06Orchestrator(
        orchestration_module.build_parser().parse_args(argv),
        preflight_dir=tmp_path / 'preflight')
    record = orchestrator.preflight()
    contract = record['contract']
    assert contract['allow_q006_bridge'] is True
    assert contract['allow_bridge'] is True
    segment = contract['segment']
    assert segment['segment_id'] == ledger_module.Q006_SEGMENT_ID
    assert segment['run_id'] == 'trial02'
    assert segment['rollback_from_update'] == 150
    assert segment['resume_from_update'] == 100
    assert segment['lost_updates'] == 50
    assert segment['replay_billed_updates'] is True
    assert contract['resume']['replay_billed_updates'] is True
    assert contract['resume']['bridge']['q006_sha256'] == \
        ledger_module.Q006_DECISION_SHA256
    assert contract['q006_bridge_manifest']['sha256'] == \
        recovery_module.sha256(manifest)
    assert not output.exists()

    # The default segment directory for the bridge run is the unique _q006 one.
    default_args = orchestration_module.build_parser().parse_args(base)
    default_contract = orchestration_module.validate_arguments(default_args)
    assert default_contract['output_dir'].endswith(
        'train_seed42_trial02_resume_000100_q006')

    # The flag is meaningless (and refused) without a resume source.
    with pytest.raises(orchestration_module.PreflightError):
        orchestration_module.validate_arguments(
            orchestration_module.build_parser().parse_args(
                ['--mode', 'train', '--allow-q006-bridge',
                 '--output-dir', str(tmp_path / 'nope')]))


# --------------------------------------------------------------- probe entry
def test_probe_container_save_verify_roundtrip(tmp_path):
    from src.tasks.velocity.v06.orchestration import probe_container

    runner, _ppo, _c, _e = make_runner(
        tmp_path / 'probe', update=55, container=probe_container(), evaluate=False)
    checkpoint = runner.save('probe')
    resumed, resumed_ppo, _c2, _e2 = make_runner(
        tmp_path / 'probe_resume', container=probe_container(), evaluate=False,
        require_resume_evidence=True)
    resumed.load(checkpoint['path'], checkpoint['sha256'], allow_probe=True)
    loaded = resumed.save_verified('resume_loaded', allow_probe=True)
    assert loaded['verification']['aligned'] is True
    resumed.train_one_update()
    assert resumed.update == 56
    ready = resumed.save_verified('resume_ready', allow_probe=True)
    assert ready['verification']['aligned'] is True
    assert resumed_ppo.collect_calls == 1


def test_production_case_runner_runs_a_real_f0_case():
    """The resume evidence is not verified against a mock-only case runner."""
    pytest.importorskip('mujoco')
    import train_a2_ideal_stairs_v06 as train

    class ZeroModel:
        def act(self, obs, deterministic=False):
            return torch.zeros(1, 12)

    contract = dict(num_envs=8, eval_num_envs=8, device='cpu', seed=42)
    case_runner = train.build_case_runner(ZeroModel(), contract, 'F0', None)
    assert getattr(case_runner, 'supports_batch', False) is True
    case = dict(vx=0.3, initial_pose=[-1.0, 0.0, 0.0], measurement_stage='F0',
                phase='F0', case_id='flat:0.3:000', episode_index=0, tile_id=0,
                spawn_profile='standard',kind='validation',lesson_id='FLAT')
    # Installed MuJoCo must execute the actual protocol; implementation errors
    # must fail this test instead of being mislabeled as host unavailability.
    record = case_runner([case])[0]
    assert record['case_id'] == case['case_id']
    assert record['measurement'] == 'flat'
    assert 0 < int(record['steps']) <= int(speed.F0_GATE_STEPS)
    assert isinstance(record['flat_pass'], bool)
    assert record['forward_distance_definition'] == \
        speed.forward_distance_definition()


def test_probe_worker_uses_the_verified_resume_entry():
    import inspect
    import train_a2_ideal_stairs_v06 as train

    source = inspect.getsource(train.build_probe_worker)
    assert "save_verified('resume_loaded'" in source
    assert "save_verified('resume_ready'" in source
    assert 'require_resume_evidence=(mode == ' in source
    assert "source=recovery_module.source_block(seed=seed, device=device,\n" in source
    assert "mode='train')" in source


# ------------------------------------------------------------------- ledger
def test_q006_ledger_keeps_lost_and_replayed_costs_apart():
    ledger = ledger_module.build_current_ledger()
    assert ledger_module.validate_ledger(ledger) is True
    totals = ledger['totals']
    assert totals['effective_learning_update'] == 100
    assert totals['physical_training_updates'] == 200
    assert totals['lost_branch_updates'] == 100
    budget = ledger['budget']
    assert budget['per_run_cap'] == 30000
    assert budget['combined_formal_cap'] == 30100
    assert budget['remaining_active_branch_after_rollback'] == 29900
    full = budget['full_cap_cost']
    assert full == dict(control_samples=739_737_600,
                        physics_substeps=2_958_950_400, adam_updates=602_000)
    assert ledger_module.training_cost(30100) == full
    # 101-150 exists on two distinct segments and neither is deduplicated.
    q005 = [row for row in ledger['entries']
            if row['segment_id'] == 'trial02_q005_resume_000100']
    assert [row['update'] for row in q005] == list(range(101, 151))
    assert all(row['lost_branch_updates'] == 1
               and row['effective_learning_update'] is False for row in q005)
    original = [row for row in ledger['entries']
                if row['segment_id'] == 'trial02_original']
    assert len(original) == 100
    assert all(row['effective_learning_update'] is True for row in original)
    with pytest.raises(ValueError):
        ledger_module.validate_entries(list(ledger['entries']) + [dict(
            ledger['entries'][0])])
    replay = ledger_module.entry('trial02', ledger_module.Q006_SEGMENT_ID, 101)
    assert replay['effective_learning_update'] is True
    assert replay['lost_branch_updates'] == 0
    q006_rows = [ledger_module.entry('trial02', ledger_module.Q006_SEGMENT_ID,
                                     index) for index in range(101, 151)]
    extended = ledger_module.build_ledger(
        entries=list(ledger['entries']) + q006_rows,
        effective_learning_update=150, physical_training_updates=250,
        lost_branch_updates=100)
    assert ledger_module.validate_ledger(extended) is True
    # The replay entry and the lost entry share update 101 but not the triple.
    assert {(row['run_id'], row['segment_id'], row['update'])
            for row in extended['entries']} >= {
        ('trial02', 'trial02_q005_resume_000100', 101),
        ('trial02', ledger_module.Q006_SEGMENT_ID, 101)}
    with pytest.raises(ValueError):
        ledger_module.build_ledger(physical_training_updates=30101)
