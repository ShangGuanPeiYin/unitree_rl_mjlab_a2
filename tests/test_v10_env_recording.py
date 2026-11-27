"""Independent same-state check of the exact approved recording-only insertion."""
import ast
import copy
import math
import random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch


def test_recording_node_preserves_physics_observations_teacher_input_and_rng():
    from src.tasks.velocity.v10.teachers import blind_yaw_command
    source=ast.parse(Path('src/tasks/velocity/v10/env.py').read_text())
    nodes=[n for n in ast.walk(source) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
        and isinstance(n.value.func,ast.Attribute) and isinstance(n.value.func.value,ast.Name)
        and n.value.func.value.id=='frame' and n.value.func.attr=='update']
    assert len(nodes)==1
    expected=ast.parse('''frame.update(yaw=yaw.clone(),foot_position_local=feet_local.clone(),top_upward_force=upward.clone(),
        failure_masks=dict(body_hit=body_hit.clone(),long_hit=long_hit.clone(),
            attitude=(roll.abs()>math.radians(70))|(pitch.abs()>math.radians(70)),
            route_fall=d.root_link_pos_w[:,2]<route_h-.5,bounds=~bounds,timeout=frame['timeout'].clone()))''').body[0]
    assert ast.dump(nodes[0],include_attributes=False)==ast.dump(expected,include_attributes=False)
    n=3
    values=dict(yaw=torch.randn(n),feet_local=torch.randn(n,4,3),upward=torch.rand(n,4),
        body_hit=torch.tensor([False,True,False]),long_hit=torch.tensor([False,False,True]),
        roll=torch.tensor([0.,.1,1.5]),pitch=torch.tensor([0.,-.1,-1.5]),
        d=SimpleNamespace(root_link_pos_w=torch.tensor([[1.,0.,.4],[1.,0.,2.4],[1.,0.,1.]])),
        route_h=torch.tensor([0.,2.,2.]),bounds=torch.tensor([True,True,False]),math=math,
        frame=dict(timeout=torch.tensor([False,True,False])))
    obs=dict(proprio=torch.randn(n,45),history=torch.randn(n,5,45),route_error=torch.tensor([[1.,0.,1.]]*n))
    obs['proprio'][:,6]=.3
    tensors={k:v for k,v in values.items() if isinstance(v,torch.Tensor)}
    tensors.update({'root_pos':values['d'].root_link_pos_w,'timeout':values['frame']['timeout'],**obs})
    before={k:v.clone() for k,v in tensors.items()};versions={k:v._version for k,v in tensors.items()}
    command=blind_yaw_command(obs['proprio'],obs['route_error']).clone()
    rng=(random.getstate(),copy.deepcopy(np.random.get_state()),torch.get_rng_state().clone())
    module=ast.fix_missing_locations(ast.Module(body=[nodes[0]],type_ignores=[]))
    exec(compile(module,'approved_recording_node','exec'),{},values)
    for k,v in tensors.items():
        assert torch.equal(v,before[k]),k
        assert v._version==versions[k],k
    assert torch.equal(command,blind_yaw_command(obs['proprio'],obs['route_error']))
    assert rng[0]==random.getstate()
    actual=np.random.get_state()
    assert rng[1][0]==actual[0] and np.array_equal(rng[1][1],actual[1]) and rng[1][2:]==actual[2:]
    assert torch.equal(rng[2],torch.get_rng_state())
    for key,original in [('yaw','yaw'),('foot_position_local','feet_local'),('top_upward_force','upward')]:
        assert values['frame'][key].data_ptr()!=values[original].data_ptr()
