"""Reject old schemas and bind recovery to the approved specification."""
import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from a2_v07_bootstrap import activate
activate()
from src.tasks.velocity.v09.checkpoint import save_checkpoint, load_checkpoint


def test_checkpoint_roundtrip_and_wrong_sds_rejected(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    path = tmp_path / 'checkpoint.pt'
    metadata = dict(stage='S1', course='S1.1', value_scale=1.,
                    manifest_sha256='a'*64, ledger_run_id='test',
                    learning_samples=0, code_sha256='c'*64,
                    gate_state=dict(next_suite='gate_A', attempts=0, final_suites_used=[],
                                    recoveries=0, next_gate_samples=100_000_000, passes=[]),
                    role_state=dict(num_envs=512, episode_counters={}, assignments=[]),
                    random_state=dict(python=(), numpy=()))
    save_checkpoint(path, model, optimizer, metadata)
    payload = load_checkpoint(path)
    assert payload['metadata'] == metadata
    payload['sds_sha256'] = 'b'*64
    torch.save(payload, path)
    with pytest.raises(ValueError, match='sds_sha256'):
        load_checkpoint(path)


def test_reject_legacy_checkpoint(tmp_path):
    path = tmp_path / 'old.pt'
    torch.save({'model': {}}, path)
    with pytest.raises(ValueError, match='schema'):
        load_checkpoint(path)

@pytest.mark.parametrize('field,value', [('value_scale', float('nan')),
    ('value_scale', float('inf')), ('learning_samples', .5),
    ('learning_samples', float('nan')), ('manifest_sha256', 'bad')])
def test_reject_invalid_numeric_and_identity_metadata(tmp_path, field, value):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    metadata = dict(stage='S1', course='S1.1', value_scale=1.,
                    manifest_sha256='a'*64, ledger_run_id='test',
                    learning_samples=0, code_sha256='c'*64,
                    gate_state=dict(next_suite='gate_A', attempts=0, final_suites_used=[],
                                    recoveries=0, next_gate_samples=100_000_000, passes=[]),
                    role_state=dict(num_envs=512, episode_counters={}, assignments=[]),
                    random_state=dict(python=(), numpy=()))
    metadata[field] = value
    with pytest.raises(ValueError):
        save_checkpoint(tmp_path / 'bad.pt', model, optimizer, metadata)
