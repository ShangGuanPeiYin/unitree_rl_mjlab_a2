"""CPU-only v0.6 ETH migration audit (SDS 9.2): SHA gate, complete Actor tensor
manifest, zero bypass, std 0.30, fresh Critic/Adam and the deterministic action
mean equality bound -- never a random-action sampling comparison."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import a2_v06_bootstrap  # noqa: E402

a2_v06_bootstrap.activate(ROOT)

import json  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

import check_a2_ideal_stairs_v06_migration as audit_cli  # noqa: E402
from src.tasks.velocity.v06 import model as target  # noqa: E402


@pytest.fixture(scope='module')
def report():
    return audit_cli.audit(seed=42, device='cpu')


def test_all_migration_checks_pass(report):
    assert report['passed'] is True
    assert all(report['checks'].values()), report['checks']
    assert report['source_checkpoint_sha256'] == target.INIT_SHA256
    assert report['forbidden_source_blocks'] == ['critic_state_dict',
                                           'optimizer_state_dict']


def test_action_mean_equivalence_and_method(report):
    mean = report['action_mean']
    assert mean['method'] == 'deterministic_mean'
    assert mean['max_abs_error'] <= 1e-6
    assert 'not accepted' in mean['method_note']
    assert 'random' in mean['method_note']


def test_bypass_zero_and_std(report):
    assert report['bypass_all_zero'] is True
    assert report['bypass']['all_zero'] is True
    assert sorted(report['bypass']['keys']) == sorted(target.BYPASS_KEYS)
    assert len(report['bypass']['sha256']) == 64
    assert report['initial_std_exact'] is True
    assert all(abs(value - 0.30) <= 1e-7 for value in report['initial_std'])
    assert report['std_range'] == [0.05, 1.00]


def test_critic_is_fresh_and_normalizer_frozen(report):
    assert report['critic_input_dims'] == 211
    assert report['critic_fresh'] is True
    assert report['normalizer_frozen'] is True
    assert report['optimizer'] == 'fresh_adam_not_built_by_model'


def test_eth_source_manifest_is_complete():
    _source, state, forbidden = target.load_validated_eth(target.INIT_CHECKPOINT,
                                                         target.INIT_SHA256)
    assert set(state) == set(target.ETH_ACTOR_KEYS)
    assert len(target.ETH_ACTOR_KEYS) == 13
    assert forbidden == {'critic_state_dict': True, 'optimizer_state_dict': True}
    assert 'optimizer' not in forbidden


def test_migration_cli_writes_a_new_report(tmp_path, capsys):
    output = tmp_path / 'migration.json'
    code = audit_cli.main(['--seed', '42', '--output', str(output)])
    assert code == 0
    payload = json.loads(output.read_text(encoding='utf-8'))
    assert payload['passed'] is True
    with pytest.raises(SystemExit):
        audit_cli.main(['--seed', '42', '--output', str(output)])  # never overwritten
    printed = capsys.readouterr().out
    assert 'max_abs_error' in printed


def test_wrong_sha_is_refused_by_the_cli(tmp_path):
    with pytest.raises(target.EthMigrationError):
        audit_cli.audit(target.INIT_CHECKPOINT, '0' * 64)
