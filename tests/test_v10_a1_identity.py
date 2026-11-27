"""Checkpoint teacher identity includes the approved command adapter."""
import torch
import pytest
from src.tasks.velocity.v10 import checkpoint as cp,spec
from src.tasks.velocity.v10.teachers import teacher_identity


def test_checkpoint_binds_adapter_and_rejects_weights_only_identity(tmp_path):
    path=tmp_path/'small.pt'
    cp.checkpoint_save(path,dict(model={},learner={},num_envs=1024,physical_sha256='physics'))
    saved=torch.load(path,map_location='cpu',weights_only=False)
    assert saved['teachers']==teacher_identity()
    assert 'blind_adapter' in saved['teachers']
    cp.checkpoint_validate(saved,num_envs=1024,physical_sha256='physics')
    saved['teachers']={'blind':spec.BLIND_SHA256,'stairs':spec.STAIRS_SHA256}
    with pytest.raises(ValueError,match='identity'):
        cp.checkpoint_validate(saved,num_envs=1024,physical_sha256='physics')
