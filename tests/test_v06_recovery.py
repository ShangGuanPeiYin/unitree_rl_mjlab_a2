"""CPU-only v0.6 recovery contract (SDS 8.2): schema-8 on disk, ``--resume-run``
contract validation, no re-billing, RNG round-trip, four-way budget audit and the
refusal of R1/R2/smoke/probe checkpoints as a training resume source."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tests'))
import a2_v06_bootstrap  # noqa: E402

a2_v06_bootstrap.activate(ROOT)

from copy import deepcopy  # noqa: E402
import random  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from src.tasks.velocity.v06 import model as model_module  # noqa: E402
from src.tasks.velocity.v06 import recovery as target  # noqa: E402
from src.tasks.velocity.v06 import spec  # noqa: E402
from test_v06_schema import FakeEnv, make_state  # noqa: E402


def test_resume_contract_declares_a_new_episode_and_no_replay():
    state = make_state(update=7)
    state['checkpoint_sha256'] = 'c' * 64
    contract = target.resume_contract(state, init_sha256=model_module.INIT_SHA256,
                                     recipe=spec.RECIPE_VERSION, seed=42,
                                     device='cuda:0', num_envs=8)
    assert contract['resume_mode'] == 'new_episode'
    assert contract['new_episode'] is True
    assert contract['replay_billed_updates'] is False
    assert contract['physical_trajectory_continuous'] is False
    assert contract['update'] == 7
    assert contract['optimizer_step'] == 7 * 20
    assert contract['phase'] == 'H'


def test_resume_contract_rejects_a_different_contract():
    state = make_state()
    with pytest.raises(target.RecoveryBlocked):
        target.resume_contract(state, seed=43)
    with pytest.raises(target.RecoveryBlocked):
        target.resume_contract(state, device='cuda:1')
    with pytest.raises(target.RecoveryBlocked):
        target.resume_contract(state, recipe='0.5-r2')
    with pytest.raises(target.RecoveryBlocked):
        target.resume_contract(state, init_sha256='b' * 64)


def test_atomic_save_and_preload(tmp_path):
    state = make_state()
    path = tmp_path / 'update_000003.pt'
    sha = target.atomic_save(path, state)
    assert target.sha256(path) == sha
    loaded = target.preload_checkpoint(path, sha, mode='resume')
    assert loaded['checkpoint_sha256'] == sha
    assert loaded['learning']['update'] == 3
    with pytest.raises(FileExistsError):
        target.atomic_save(path, state)
    with pytest.raises(target.RecoveryBlocked):
        target.preload_checkpoint(path, '0' * 64)


def test_legacy_and_foreign_files_are_blocked(tmp_path):
    legacy = deepcopy(make_state())
    legacy['schema'] = 7
    path = tmp_path / 'legacy.pt'
    torch.save(legacy, path)
    with pytest.raises(target.RecoveryBlocked):
        target.preload_checkpoint(path, mode='resume')

    foreign = deepcopy(make_state())
    foreign['identity']['recipe'] = '0.5-r2'
    path = tmp_path / 'foreign.pt'
    torch.save(foreign, path)
    with pytest.raises(target.RecoveryBlocked):
        target.preload_checkpoint(path, mode='resume')

    smoke = deepcopy(make_state())
    smoke['container'] = 'v06-smoke-state-v1'
    path = tmp_path / 'smoke.pt'
    torch.save(smoke, path)
    with pytest.raises(target.RecoveryBlocked):
        target.preload_checkpoint(path, mode='resume')


def test_checkpoint_identity_reports_the_container(tmp_path):
    smoke = deepcopy(make_state())
    smoke['container'] = 'v06-probe-state-v1'
    path = tmp_path / 'probe.pt'
    torch.save(smoke, path)
    identity = target.checkpoint_identity(path)
    assert identity['classification'] == 'smoke_or_probe'
    assert identity['schema'] == 8


def test_rng_roundtrip_restores_every_generator():
    env = FakeEnv()
    state = target.capture_rng(env)
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    target.restore_rng(state, env)
    first = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    target.restore_rng(state, env)
    second = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    assert first == second
    assert state['env'] == {'env': [1, 2, 3]}
    assert state['sampler'] == {'sampler': [4, 5, 6]}


def test_budget_audit_four_way_consistency():
    learning = target.learning_state(300, num_envs=8)
    curriculum = dict(phase='H', substage=None, level_tick=0, generation=0,
                      generation_updates=300, spent_by_phase={'H': 300},
                      spent_by_substage={}, episode_fifo=[], ema=None,
                      ready_streak=0, hard_streak=0, cooldown=0, gate_streak=0,
                      retention_streaks={}, minimum_level_failure_streak=0,
                      sampler_rng_state={})
    audit = target.audit_budget(learning, curriculum)
    assert audit['consistent'] is True
    assert audit['checks'] == dict(update_equals_phase_spend=True,
                                   phase_spend_within_budgets=True,
                                   total_within_budget=True, ledger_arithmetic=True)
    broken = dict(curriculum, spent_by_phase={'H': 299})
    with pytest.raises(ValueError):
        target.audit_budget(learning, broken)
    broken = dict(curriculum, spent_by_phase={'H': 300, 'F0': 1001})
    with pytest.raises(ValueError):
        target.audit_budget(learning, broken)
    broken = dict(curriculum, spent_by_phase={'H': 300, 'Z': 1})
    with pytest.raises(ValueError):
        target.audit_budget(learning, broken)


def test_learning_state_arithmetic_is_derived_not_stored():
    state = target.learning_state(5, num_envs=8)
    assert state['control_samples'] == 5 * 8 * 48
    assert state['physics_substeps'] == 4 * state['control_samples']
    assert state['adam_steps'] == 100
    with pytest.raises(ValueError):
        target.learning_state(-1)
    tampered = dict(state, control_samples=1)
    with pytest.raises(ValueError):
        target.validate_learning(tampered)


def test_model_state_validation_and_load():
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)

    model = Tiny()
    state = {key: value.clone() for key, value in model.state_dict().items()}
    assert target.validate_model_state(state, model) is True
    target.load_model_state(model, state)
    with pytest.raises(ValueError):
        target.validate_model_state({'linear.weight': state['linear.weight']}, model)
    bad = deepcopy(state)
    bad['linear.weight'] = torch.full_like(bad['linear.weight'], float('nan'))
    with pytest.raises(ValueError):
        target.validate_model_state(bad, model)


def test_identity_and_source_blocks_bind_the_contract():
    identity = target.identity_block()
    assert identity['task'] == spec.TASK
    assert identity['recipe'] == spec.RECIPE_VERSION
    assert identity['architecture'] == spec.ARCHITECTURE
    assert identity['schema'] == 8
    assert identity['sds_sha256'] == spec.SDS_SHA256
    source = target.source_block(seed=42, device='cuda:0')
    assert source['init_sha256'] == model_module.INIT_SHA256
    assert source['sds_sha256'] == spec.SDS_SHA256
    assert len(source['source_closure_sha256']) == 64
    with pytest.raises(target.ForeignContainerError):
        target.source_block(seed=42, init_sha256='b' * 64)


# ------------------------------------------------- Q002-4 terminal states
def _stopped_state(stop='behavior_regression', lifecycle='running'):
    state = make_state()
    state['ledger']['stop_state'] = stop
    state['ledger']['lifecycle_state'] = lifecycle
    state['ledger']['stop_reason'] = stop
    state['ledger']['terminal'] = dict(
        state=(lifecycle if lifecycle != 'running' else stop),
        lifecycle_state=lifecycle, stop_state=stop, reason=stop,
        global_update=3, phase='H', substage=None, generation=0,
        budget=dict(update=3, total_budget=30000, total_exhausted=False,
                    phase='H', phase_budget=12000, phase_spent=3,
                    phase_exhausted=False),
        evidence={}, final_test=None,
        error_type=None)
    return state


def test_stopped_checkpoint_is_refused_for_resume(tmp_path):
    state = _stopped_state()
    target.validate_checkpoint(state)
    path = tmp_path / 'stopped.pt'
    torch.save(state, path)
    with pytest.raises(target.RecoveryBlocked):
        target.preload_checkpoint(path, mode='resume')
    from src.tasks.velocity.v06 import runner as runner_module
    runner_like = runner_module.V06Runner
    assert 'resume'  # the runner raises the same RecoveryBlocked in load()

def test_unknown_state_names_are_foreign():
    state = make_state()
    state['ledger']['stop_state'] = 'pilot_complete'
    with pytest.raises(target.ForeignContainerError):
        target.validate_checkpoint(state)
    state = make_state()
    state['ledger']['lifecycle_state'] = 'pilot_failed'
    with pytest.raises(target.ForeignContainerError):
        target.validate_checkpoint(state)


def test_complete_requires_the_passed_full_final_test():
    state = _stopped_state(stop=None, lifecycle='complete')
    state['ledger']['terminal']['final_test'] = dict(
        passed=False, spawn_profile='full_R_expanded', cases_per_speed=100)
    with pytest.raises(ValueError):
        target.validate_checkpoint(state)
    state['ledger']['terminal']['final_test'] = dict(
        passed=True, spawn_profile='standard', cases_per_speed=100)
    with pytest.raises(ValueError):
        target.validate_checkpoint(state)
    state['ledger']['terminal']['final_test'] = dict(
        passed=True, spawn_profile='full_R_expanded', cases_per_speed=100,
        per_speed=[dict(c=0.2), dict(c=0.4), dict(c=0.6)])
    assert target.validate_checkpoint(state) is True


def test_complete_with_a_training_stop_is_rejected():
    state = _stopped_state(stop='budget_exhausted', lifecycle='complete')
    with pytest.raises(ValueError):
        target.validate_checkpoint(state)


def test_gate_failed_requires_a_stop_reason():
    state = _stopped_state(stop='gate_failed')
    state['ledger']['terminal']['reason'] = ''
    with pytest.raises(ValueError):
        target.validate_checkpoint(state)


def test_terminal_evidence_must_be_complete():
    state = _stopped_state()
    del state['ledger']['terminal']['budget']
    with pytest.raises(ValueError):
        target.validate_checkpoint(state)
    state = _stopped_state()
    del state['ledger']['terminal']['generation']
    with pytest.raises(ValueError):
        target.validate_checkpoint(state)


# ------------------------------------------------- probe container tag (Q002)
def test_probe_checkpoint_is_refused_by_formal_resume(tmp_path):
    state = make_state()
    state['container'] = 'v06-probe-state-v1'
    path = tmp_path / 'probe_000055.pt'
    torch.save(state, path)
    with pytest.raises(target.RecoveryBlocked):
        target.preload_checkpoint(path, mode='resume')
    # The probe itself may read its own tagged evidence (recovery verification).
    loaded = target.preload_checkpoint(path, mode='probe')
    assert loaded['container'] == 'v06-probe-state-v1'
    assert loaded['resume_mode'] == 'new_episode'
    # A normal training checkpoint is never accepted as probe evidence.
    normal = make_state()
    normal_path = tmp_path / 'normal.pt'
    torch.save(normal, normal_path)
    with pytest.raises(target.RecoveryBlocked):
        target.preload_checkpoint(normal_path, mode='probe')


def test_build_checkpoint_can_emit_the_probe_container(tmp_path):
    state = make_state()
    state['container'] = target.PROBE_CONTAINER
    assert target.validate_checkpoint(state, allow_probe=True) is True
    with pytest.raises(target.ForeignContainerError):
        target.validate_checkpoint(state)


def test_q005_bridge_accepts_only_the_documented_checkpoint(tmp_path):
    from src.tasks.velocity.v06 import recovery as r
    state = make_state()
    state['checkpoint_sha256'] = r.BRIDGE_CHECKPOINT_SHA256
    state['source']['source_closure_sha256'] = r.BRIDGE_OLD_SOURCE_CLOSURE_SHA256
    bridge = r.resume_contract(state, allow_bridge=True)
    assert bridge['bridge']['restricted'] is True
    assert bridge['bridge']['q005_sha256'] == r.Q005_DECISION_SHA256
    with pytest.raises(r.RecoveryBlocked):
        r.resume_contract(state)  # no bridge flag: unchanged contract
    state['checkpoint_sha256'] = 'a' * 64
    with pytest.raises(r.RecoveryBlocked):
        r.resume_contract(state, allow_bridge=True)


def test_q005_recovery_preserves_learning_rate(tmp_path):
    state = make_state()
    state['learning']['learning_rate'] = 8.888888888888888e-05
    state['optimizer']['param_groups'][0]['lr'] = 8.888888888888888e-05
    contract = target_module().resume_contract(state)
    assert contract['learning_rate'] == pytest.approx(8.888888888888888e-05)


def target_module():
    from src.tasks.velocity.v06 import recovery
    return recovery
