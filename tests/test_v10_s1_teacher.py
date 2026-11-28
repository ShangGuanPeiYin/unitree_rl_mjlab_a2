"""Actual S1 weights in every DualTeacher caller, S2 student unchanged."""
import torch
import pytest
from src.tasks.velocity.v10 import spec
from src.tasks.velocity.v10.model import V10ActorCritic
from src.tasks.velocity.v10.teachers import DualTeacher,teacher_identity


def test_actual_s1_teacher_preserves_s2_student():
    torch.set_num_threads(1)
    student=V10ActorCritic.from_stairs(device='cpu')
    before={k:v.clone() for k,v in student.state_dict().items()}
    teacher=DualTeacher(student)
    s1=torch.load(spec.STAIRS_PATH,map_location='cpu',weights_only=False)['model']
    s2=torch.load(spec.INITIAL_PATH,map_location='cpu',weights_only=False)['model']
    assert all(torch.equal(v,s1[k]) for k,v in teacher.stairs.state_dict().items())
    assert all(torch.equal(v,s2[k]) and torch.equal(v,before[k]) for k,v in student.state_dict().items())
    assert not any(p.requires_grad for p in teacher.parameters())
    assert teacher_identity()['stairs']=='efa9caa8ce99e2cdff967212063e198b7ae68a6415727083b1db027f615ead24'


def test_wrong_teacher_bytes_rejected(tmp_path,monkeypatch):
    path=tmp_path/'wrong.pt';path.write_bytes(b'not S1')
    monkeypatch.setattr(spec,'STAIRS_PATH',path)
    with pytest.raises(ValueError,match='stairs teacher identity mismatch'):
        DualTeacher(torch.nn.Linear(1,1))
