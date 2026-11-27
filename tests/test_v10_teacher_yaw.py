"""Planner A1: exact deterministic observable-only blind yaw command, CPU only."""
import copy
import math
import pytest
import torch
from src.tasks.velocity.v10 import teachers as module
from src.tasks.velocity.rl.ideal_heightmap_actor_critic import FrozenETHNormalizer,mlp

@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
 monkeypatch.setattr(torch.cuda,'is_available',lambda:False)


def observation(vx=.6,y=0.,yaw=0.,n=1):
 p=torch.zeros(n,45);p[:,6]=vx
 route=torch.tensor([[y/.2,math.sin(yaw),math.cos(yaw)]],dtype=torch.float32).repeat(n,1)
 return dict(proprio=p,route_error=route,history=torch.arange(n*225,dtype=torch.float32).reshape(n,5,45),heightmap=torch.zeros(n,3,40,30))

@pytest.mark.parametrize('y,yaw,want',[(.2,0.,-.03328408188791190),(-.2,0.,.03328408188791190),(0.,.2,-.1),(0.,-.2,.1)])
def test_observable_lateral_and_yaw_error_correction_sign(y,yaw,want):
 o=observation(y=y,yaw=yaw)
 result=module.blind_yaw_command(o['proprio'],o['route_error'])
 assert torch.allclose(result,torch.tensor([[.6,0.,want]]),atol=2e-7,rtol=1e-6)

@pytest.mark.parametrize('y,want',[(.8,-.075),(-.8,.075)])
def test_desired_heading_saturates_before_yaw_rate(y,want):
 o=observation(vx=.3,y=y)
 actual=module.blind_yaw_command(o['proprio'],o['route_error'])
 assert abs(actual[0,2].item()-want)<2e-7

@pytest.mark.parametrize('yaw',[math.pi-1e-4,-math.pi+1e-4])
def test_heading_error_wraps_across_atan2_branch_cut(yaw):
 o=observation(y=.8,yaw=yaw)
 assert abs(module.blind_yaw_command(o['proprio'],o['route_error'])[0,2].item()-.15)<1e-7

@pytest.mark.parametrize('yaw',[math.pi,-math.pi])
def test_exact_pi_remains_finite_rate_bounded_without_imposing_extra_tie_break(yaw):
 o=observation(yaw=yaw)
 rate=module.blind_yaw_command(o['proprio'],o['route_error'])[0,2]
 assert torch.isfinite(rate) and abs(abs(rate.item())-.15)<1e-7

@pytest.mark.parametrize('vx',[0.,-.1,-1.])
def test_nonpositive_request_produces_all_zero_teacher_command(vx):
 o=observation(vx=vx,y=.8,yaw=2.)
 assert torch.equal(module.blind_yaw_command(o['proprio'],o['route_error']),torch.zeros(1,3))


def test_rate_clamp_then_speed_gate_and_ramp_to_zero():
 o=observation(yaw=.4,n=5);o['proprio'][:,6]=torch.tensor([.3,.15,.03,.003,0.])
 actual=module.blind_yaw_command(o['proprio'],o['route_error'])
 assert torch.allclose(actual[:,2],torch.tensor([-.15,-.075,-.015,-.0015,0.]),atol=1e-7)
 assert torch.equal(actual[:,0],o['proprio'][:,6]) and torch.equal(actual[:,1],torch.zeros(5))


def test_raw_eth_path_is_replaced_only_by_private_command_and_student_aliases_stay_equal():
 teacher=module.BlindTeacher();o=observation(y=.8,yaw=.2,n=2)
 o['proprio'][:,7]=.123;o['proprio'][:,8]=.456
 before={k:v.clone() for k,v in o.items()};weights=copy.deepcopy(teacher.state_dict())
 # The approved command is vx=.6, vy=0, wz=-.15: desired=-.15, error=-.35, rate saturates.
 private=o['proprio'].clone();private[:,6:9]=torch.tensor([.6,0.,-.15])
 actual=teacher(o);expected=teacher.mlp(teacher.normalizer(private))
 assert torch.equal(actual,expected)
 assert not torch.equal(actual,teacher.mlp(teacher.normalizer(o['proprio'])))
 assert torch.equal(actual,teacher(o))
 assert all(torch.equal(v,before[k]) for k,v in o.items())
 assert all(torch.equal(v,weights[k]) for k,v in teacher.state_dict().items())
 assert all(not p.requires_grad for p in teacher.parameters())


def test_real_eth_weights_agree_on_hand_checked_command_and_zero_correction_identity():
 state=torch.load('/home/xxs/dog/eth_a2_locomotion_controller/models/model_2000.pt',map_location='cpu',weights_only=False)['actor_state_dict']
 net=mlp([45,512,256,128,12]);norm=FrozenETHNormalizer()
 net.load_state_dict({k[4:]:v for k,v in state.items() if k.startswith('mlp.')})
 norm.load_state_dict({k[15:]:v for k,v in state.items() if k.startswith('obs_normalizer.')})
 teacher=module.BlindTeacher()
 for y,yaw,rate in [(0.,0.,0.),(.8,.2,-.15)]:
  o=observation(y=y,yaw=yaw);private=o['proprio'].clone();private[:,6:9]=torch.tensor([.6,0.,rate])
  assert torch.equal(teacher(o),net(norm(private)))


def test_mixed_routing_stairs_original_input_and_blind_hidden_metadata_invariance():
 class Stairs(torch.nn.Module):
  def forward(self,o):return o['proprio'][:,:12].clone()
 stairs=Stairs();teacher=module.DualTeacher(stairs);o=observation(y=.8,yaw=.2,n=4)
 o['proprio'][:,7]=.123;o['proprio'][:,8]=.456
 before={k:v.clone() for k,v in o.items()};mask=[True,False,True,False]
 actual=teacher.label(o,mask)
 assert torch.equal(actual[[0,2]],stairs(o)[[0,2]])
 private=o['proprio'].clone();private[:,6:9]=torch.tensor([.6,0.,-.15])
 assert torch.equal(actual[[1,3]],teacher.blind.mlp(teacher.blind.normalizer(private))[[1,3]])
 # Hidden labels cannot alter the adapter; F/P/posttop share one observable-only path.
 first=dict(o,task='F_WALK',phase='WALK',teacher_id='arbitrary-a')
 second=dict(o,task='P_STOP',phase='POST_WALK',teacher_id='arbitrary-b')
 assert torch.equal(teacher.blind(first),teacher.blind(second))
 assert all(torch.equal(v,before[k]) for k,v in o.items())


def test_small_positive_speed_uses_point_one_denominator_before_speed_gate():
 o=observation(vx=.03,y=.02)
 assert abs(module.blind_yaw_command(o['proprio'],o['route_error'])[0,2].item()-(-.001998934356164502))<1e-8


def test_proprio_history_shared_storage_remains_unmodified():
 teacher=module.BlindTeacher();o=observation(y=.8,yaw=.2)
 o['history']=o['proprio'][:,None,:].expand(-1,5,-1)
 before=o['proprio'].clone()
 teacher(o)
 assert torch.equal(o['proprio'],before) and torch.equal(o['history'],before[:,None,:].expand(-1,5,-1))


def test_teacher_identity_binds_weights_protocol_and_actual_adapter_parameters(monkeypatch):
 import hashlib,json
 from src.tasks.velocity.v10 import spec
 identity=module.teacher_identity();config=module.blind_adapter_configuration()
 assert identity['blind']==module.BLIND_SHA and identity['stairs']==spec.STAIRS_SHA256
 assert config['protocol']==module.BLIND_YAW_PROTOCOL
 encoded=json.dumps(config,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
 assert identity['blind_adapter']==hashlib.sha256(encoded).hexdigest()
 assert module.teacher_identity()==identity
 monkeypatch.setattr(module,'BLIND_YAW_HEADING_GAIN',.4)
 assert module.teacher_identity()['blind_adapter']!=identity['blind_adapter']
