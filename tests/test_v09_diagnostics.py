"""Reject broadcast teacher labels and unbound saved diagnostic states."""
import hashlib
import json
from pathlib import Path
import sys
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from a2_v07_bootstrap import activate
activate()
from src.tasks.velocity.v09 import spec
from src.tasks.velocity.v09.diagnostics import loss_report


class Toy(torch.nn.Module):
    def forward(self, obs):
        return torch.zeros(obs['proprio'].shape[0], 12)
    def evaluate(self, obs):
        return torch.zeros(obs['proprio'].shape[0], 1)


def saved(tmp_path, *, labels=2):
    code, policy, manifest_sha = 'a' * 64, 'b' * 64, 'c' * 64
    manifest = dict(manifest_sha256=manifest_sha, cases=[dict(case_sha256=f'case{i}',
        task='flat_walk', lesson='FLAT', speed=.3, stop_mode='none') for i in range(2)])
    identity = dict(sds_sha256=spec.SDS_SHA256, code_sha256=code,
        manifest_sha256=manifest_sha, policy_checkpoint_sha256=policy,
        teacher_source_sha256=spec.SOURCE_SHA256, suite='S1_DIAG')
    certificate = dict(validated=True, validator='v07.terrain.index_geometry',
                       code_sha256=code, sds_sha256=spec.SDS_SHA256)
    certificate['certificate_sha256'] = hashlib.sha256(json.dumps(certificate,
        sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    batch = dict(identity=identity, geometry=certificate, case_sha256s=['case0', 'case1'],
        diagnostic_samples=[dict(slots=torch.arange(2),
            student_observation={'proprio': torch.zeros(2, 3)},
            teacher_mean=torch.ones(labels, 12), teacher_value=torch.ones(labels, 1))])
    torch.save(batch, tmp_path / 'batch_0000.pt')
    (tmp_path / 'summary.json').write_text(json.dumps(dict(checkpoint_sha256=policy,
        code_sha256=code, manifest_sha256=manifest_sha)))
    return manifest, dict(code_sha256=code, dataset_checkpoint_sha256=policy)


def test_reject_broadcast_teacher_labels(tmp_path):
    manifest, kwargs = saved(tmp_path, labels=1)
    with pytest.raises(ValueError, match='exact sample count'):
        loss_report(Toy(), tmp_path, manifest, value_scale=1., device='cpu', **kwargs)


def test_reject_zero_value_scale(tmp_path):
    manifest, kwargs = saved(tmp_path)
    with pytest.raises(ValueError, match='value_scale'):
        loss_report(Toy(), tmp_path, manifest, value_scale=0., device='cpu', **kwargs)


def test_fixed_state_loss_and_source_binding(tmp_path):
    manifest, kwargs = saved(tmp_path)
    result = loss_report(Toy(), tmp_path, manifest, value_scale=2., device='cpu', **kwargs)
    assert result['case_count'] == 2
    assert next(iter(result['groups'].values()))['action_loss'] == 1.
    assert next(iter(result['groups'].values()))['value_loss'] == .25
    kwargs['dataset_checkpoint_sha256'] = 'd' * 64
    with pytest.raises(ValueError, match='identity'):
        loss_report(Toy(), tmp_path, manifest, value_scale=2., device='cpu', **kwargs)
