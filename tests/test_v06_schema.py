"""CPU-only v0.6 schema-8 contract (SDS 7.2/8.2): the full checkpoint envelope,
budget arithmetic, refusal of legacy/foreign/smoke containers and the four-way
budget audit."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import a2_v06_bootstrap  # noqa: E402

a2_v06_bootstrap.activate(ROOT)

from copy import deepcopy  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

from src.tasks.velocity.v06 import model as model_module  # noqa: E402
from src.tasks.velocity.v06 import ppo as ppo_module  # noqa: E402
from src.tasks.velocity.v06 import recovery as target  # noqa: E402
from src.tasks.velocity.v06 import spec  # noqa: E402


class FakeEnv:
    def rng_state(self):
        return {'env': [1, 2, 3]}

    def sampler_rng_state(self):
        return {'sampler': [4, 5, 6]}


def make_optimizer_state():
    parameter = torch.nn.Parameter(torch.zeros(2))
    optimizer = torch.optim.Adam([parameter], lr=3e-4, betas=(0.9, 0.999),
                                 eps=1e-8, weight_decay=0.0)
    optimizer.zero_grad()
    parameter.sum().backward()
    optimizer.step()
    return optimizer.state_dict()


def make_curriculum(update, *, phase='H'):
    return dict(phase=phase, substage=None, level_tick=0, generation=0,
                generation_updates=update, spent_by_phase={phase: update},
                spent_by_substage={}, episode_fifo=[], ema=None, ready_streak=0,
                hard_streak=0, cooldown=0, gate_streak=0, retention_streaks={},
                minimum_level_failure_streak=0,
                sampler_rng_state={'state': [0.0]})


def make_initialization():
    return dict(contract={'task': spec.TASK}, model={'x': 1}, source_path='eth.pt',
                source_sha256=model_module.INIT_SHA256,
                migrated_model_sha256='a' * 64,
                transfer_metadata={'sha256': model_module.INIT_SHA256,
                                   'route_fusion': 'zeros_after_migration'},
                resume_mode='new_episode', resume=False)


def make_state(*, update=3, num_envs=8, phase='H'):
    learning = target.learning_state(update, num_envs=num_envs)
    learning['learning_rate'] = 3e-4
    model_state = {key: torch.zeros(2) for key in model_module.BYPASS_KEYS}
    model_state['std'] = torch.full((12,), 0.30)
    return target.build_checkpoint(
        model_state=model_state, optimizer_state=make_optimizer_state(),
        learning_rate=3e-4,
        normalizer={'actor': {'mean': torch.zeros(1, 45)},
                    'critic': {'mean': torch.zeros(80)}},
        learning=learning, rng=target.capture_rng(FakeEnv()),
        curriculum=make_curriculum(update, phase=phase),
        evaluation={'cases': {}, 'latest': None},
        initialization=make_initialization(),
        gate={'identity': {'task': spec.TASK}},
        retention={'streak': 0}, ledger={'checkpoints': []},
        source=target.source_block(seed=42, device='cuda:0'),
        recipe=spec.recipe_config())


def test_valid_schema8_roundtrip():
    state = make_state()
    assert set(state) == set(target.TOP_LEVEL_KEYS)
    assert state['schema'] == 8
    assert state['resume_mode'] == 'new_episode'
    assert target.validate_checkpoint(state) is True
    audit = target.audit_budget(state['learning'], state['curriculum'])
    assert audit['consistent'] is True
    assert all(audit['checks'].values())
    assert audit['total_budget'] == 30000


def test_budget_arithmetic_of_one_update_and_the_full_budget():
    learning = target.learning_state(1, num_envs=512)
    assert learning['control_samples'] == 24576
    assert learning['physics_substeps'] == 98304
    assert learning['adam_steps'] == 20
    full = target.learning_state(30000, num_envs=512)
    assert full['control_samples'] == 737_280_000
    assert full['physics_substeps'] == 2_949_120_000
    assert full['adam_steps'] == 600_000


def test_legacy_schema_is_refused():
    state = deepcopy(make_state())
    state['schema'] = 7
    with pytest.raises(target.LegacySchemaError):
        target.validate_checkpoint(state)
    kind, reason = target.classify(state)
    assert kind == 'legacy' and 'schema' in reason


def test_r1_r2_recipe_is_refused():
    for recipe in ('0.5', '0.5-r2', '0.4'):
        state = deepcopy(make_state())
        state['identity']['recipe'] = recipe
        with pytest.raises(target.ForeignContainerError):
            target.validate_checkpoint(state)
    state = deepcopy(make_state())
    state['identity']['task'] = 'Unitree-A2-IdealHeightMap-Stairs-V05R2'
    with pytest.raises(target.ForeignContainerError):
        target.validate_checkpoint(state)


def test_smoke_and_probe_containers_are_refused():
    for container in ('v05-smoke-state-v1', 'v06-smoke-state-v1',
                      'v06-probe-state-v1'):
        state = deepcopy(make_state())
        state['container'] = container
        with pytest.raises(target.ForeignContainerError):
            target.validate_checkpoint(state)
    state = deepcopy(make_state())
    state['source']['mode'] = 'probe'
    with pytest.raises(target.ForeignContainerError):
        target.validate_checkpoint(state)


def test_wrong_initialization_source_is_refused():
    state = deepcopy(make_state())
    state['source']['init_sha256'] = 'b' * 64
    with pytest.raises(target.ForeignContainerError):
        target.validate_checkpoint(state)
    state = deepcopy(make_state())
    state['initialization']['transfer_metadata']['route_fusion'] = 'random'
    with pytest.raises(ValueError):
        target.validate_checkpoint(state)


def test_envelope_key_contract_is_exact():
    state = deepcopy(make_state())
    del state['retention']
    with pytest.raises(ValueError):
        target.validate_checkpoint(state)
    state = deepcopy(make_state())
    state['extra'] = 1
    with pytest.raises(ValueError):
        target.validate_checkpoint(state)


def test_resume_mode_is_mandatory():
    state = deepcopy(make_state())
    state['resume_mode'] = 'resume'
    with pytest.raises(ValueError):
        target.validate_checkpoint(state)
    kind, _reason = target.classify(state)
    assert kind == 'invalid'


def test_single_adam_state_is_validated():
    state = deepcopy(make_state())
    group = state['optimizer']['param_groups'][0]
    group['betas'] = (0.8, 0.99)
    with pytest.raises(ValueError):
        target.validate_optimizer_state(state['optimizer'])
    state = deepcopy(make_state())
    state['optimizer']['param_groups'][0]['weight_decay'] = 0.01
    with pytest.raises(ValueError):
        target.validate_optimizer_state(state['optimizer'])
    state = deepcopy(make_state())
    state['optimizer']['param_groups'][0]['lr'] = 1e-3
    with pytest.raises(ValueError):
        target.validate_optimizer_state(state['optimizer'])
    state = deepcopy(make_state())
    state['optimizer']['param_groups'].append(deepcopy(
        state['optimizer']['param_groups'][0]))
    with pytest.raises(ValueError):
        target.validate_optimizer_state(state['optimizer'])


def test_learning_rate_block_must_match_optimizer():
    state = deepcopy(make_state())
    state['learning']['learning_rate'] = 1e-4
    with pytest.raises(ValueError):
        target.validate_checkpoint(state)


def test_budget_mismatch_is_refused():
    state = deepcopy(make_state())
    state['curriculum']['spent_by_phase'] = {'H': 4}
    with pytest.raises(ValueError):
        target.validate_curriculum(state['curriculum'], learning=state['learning'])
    state = deepcopy(make_state())
    state['curriculum']['spent_by_phase'] = {'H': 2}
    with pytest.raises(ValueError):
        target.validate_curriculum(state['curriculum'], learning=state['learning'])
    state = deepcopy(make_state(update=1001, phase='H'))
    state['curriculum']['spent_by_phase'] = {'F0': 1001}
    with pytest.raises(ValueError):
        target.validate_curriculum(state['curriculum'], learning=state['learning'])


def test_rng_block_requires_env_and_sampler():
    state = deepcopy(make_state())
    state['rng']['env'] = None
    with pytest.raises(ValueError):
        target.validate_rng(state['rng'])
    state['rng'] = target.capture_rng(None)
    with pytest.raises(ValueError):
        target.validate_rng(state['rng'])


def test_curriculum_state_fields_are_complete():
    state = deepcopy(make_state())
    for field in target.CURRICULUM_FIELDS:
        assert field in state['curriculum'], field
    state['curriculum']['spent_by_substage'] = {'D25': 1501}
    state['curriculum']['spent_by_phase'] = {'D': 3}
    with pytest.raises(ValueError):
        target.validate_curriculum(state['curriculum'])


def test_mode_and_pilot_are_never_resumable():
    for mode in ('pilot', 'smoke', 'probe'):
        state = deepcopy(make_state())
        state['source']['mode'] = mode
        with pytest.raises(target.ForeignContainerError):
            target.reject_foreign_container(state)
