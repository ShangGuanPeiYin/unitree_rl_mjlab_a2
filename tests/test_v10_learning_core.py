import copy
import torch
from src.tasks.velocity.v10.replay import TaskReservoir, TASKS
from src.tasks.velocity.v10.ppo import gae, epoch_batches
from src.tasks.velocity.v10.rewards import AirtimeTracker

def sample(n):
 return dict(obs={'proprio':torch.arange(n*45,dtype=torch.float32).reshape(n,45)}, actions=torch.zeros(n,12), task=[TASKS[i%6] for i in range(n)], teacher=['blind']*n, case=list(range(n)), episode=[0]*n, control_step=[0]*n)

def test_reservoir_restore_and_snapshot():
 r=TaskReservoir(capacity=5,seed=42); x=sample(60); r.add(x); before=copy.deepcopy(r.state_dict()); x['obs']['proprio'].zero_(); assert r.data[TASKS[0]]['obs']['proprio'].abs().sum()>0
 b=TaskReservoir(capacity=5,seed=1); b.load_state_dict(before)
 a=r.sample_per_task(3); c=b.sample_per_task(3)
 assert torch.equal(a['obs']['proprio'],c['obs']['proprio'])
 r.add(sample(120)); b.add(sample(120)); assert torch.equal(r.data[TASKS[2]]['obs']['proprio'],b.data[TASKS[2]]['obs']['proprio'])

def test_balanced_current_history():
 r=TaskReservoir(capacity=5); current=sample(60); b=r.sample_logical(current)
 assert len(b['actions'])==3072
 assert all(b['task'].count(k)==512 for k in TASKS)
 r.add(current); b=r.sample_logical(current); assert len(b['actions'])==3072

def test_gae_terminal_and_rollout_bootstrap():
 a,ret=gae(torch.ones(2,1),torch.zeros(2,1),torch.tensor([[True],[False]]),torch.tensor([10.]))
 assert a[0,0]==1; assert torch.allclose(ret[1],torch.tensor([10.9]))

def test_epoch_exact_once():
 g=torch.Generator().manual_seed(42)
 for batches in epoch_batches(48*512,generator=g):
  assert len(batches)==4
  assert torch.equal(torch.cat(batches).sort().values,torch.arange(48*512))

def test_airtime_uses_reward_edges_and_clears_stop():
 t=AirtimeTracker(1,'cpu'); C=torch.ones(1,4,dtype=torch.bool); move=torch.ones(1,dtype=torch.bool)
 assert t.step(C,move).item()==0
 for _ in range(10): assert t.step(~C,move).item()==0
 assert abs(t.step(C,move).item()-.4)<1e-6
 t.step(~C,move); t.step(C,~move); assert t.step(C,move).item()==0

def obs(n):
 return dict(proprio=torch.randn(n,45),history=torch.randn(n,5,45),heightmap=torch.randn(n,3,40,30),route_error=torch.randn(n,3),privileged=torch.randn(n,35))

def test_real_actor_identity_and_ppo_initialization():
 from src.tasks.velocity.v10.model import V10ActorCritic
 from src.tasks.velocity.v10.teachers import BlindTeacher
 from src.tasks.velocity.rl.ideal_heightmap_actor_critic import FrozenETHNormalizer, mlp
 torch.set_num_threads(1)
 model=V10ActorCritic.from_stairs(); x=obs(2)
 checkpoint=torch.load('logs/good/s2_1_latest.pt',map_location='cpu',weights_only=False)['model']
 assert all(torch.equal(v,checkpoint[k]) for k,v in model.state_dict().items())
 before=model(x).detach(); normalizer=copy.deepcopy(model.critic_normalizer.state_dict())
 rng=torch.get_rng_state().clone(); keep=model.initialize_ppo(); assert torch.equal(rng,torch.get_rng_state())
 assert torch.equal(before,model(x)); assert torch.equal(before,keep(x))
 assert torch.equal(model.evaluate(x),torch.zeros(2,1))
 assert all(torch.equal(v,normalizer[k]) for k,v in model.critic_normalizer.state_dict().items())
 assert all(not p.requires_grad for p in model.actor.parameters())
 assert not model.std.requires_grad
 model.set_phase('P1'); assert model.std.requires_grad
 with torch.no_grad(): model.std[:6]=2;model.std[6:]=-1
 model.constrain_std(); assert model.std.max()<=.30 and model.std.min()>=.05
 blind=BlindTeacher(); state=torch.load('/home/xxs/dog/eth_a2_locomotion_controller/models/model_2000.pt',map_location='cpu',weights_only=False)['actor_state_dict']
 net=mlp([45,512,256,128,12]); norm=FrozenETHNormalizer()
 net.load_state_dict({k[4:]:v for k,v in state.items() if k.startswith('mlp.')})
 norm.load_state_dict({k[15:]:v for k,v in state.items() if k.startswith('obs_normalizer.')})
 from src.tasks.velocity.v10.teachers import blind_yaw_command
 private=x['proprio'].clone();private[:,6:9]=blind_yaw_command(x['proprio'],x['route_error'])
 assert torch.equal(blind(x),net(norm(private)))
 neutral={k:v.clone() for k,v in x.items()}
 neutral['proprio'][:,6:9]=torch.tensor([.6,0.,0.]);neutral['route_error'][:]=torch.tensor([0.,0.,1.])
 assert torch.equal(blind(neutral),net(norm(neutral['proprio'])))

def test_dagger_microbatch_gradient_equivalence():
 from src.tasks.velocity.v10.dagger import DAggerLearner
 from src.tasks.velocity.v10 import spec
 class Linear(torch.nn.Module):
  def __init__(self): super().__init__();self.net=torch.nn.Linear(45,12)
  def forward(self,x): return self.net(x['proprio'])
 torch.set_num_threads(1); torch.manual_seed(2); a=Linear(); b=copy.deepcopy(a)
 current=sample(60); current['obs']['proprio']/=1000
 r=TaskReservoir(capacity=5,seed=42); r2=TaskReservoir(capacity=5,seed=42)
 old=spec.MICROBATCH
 try:
  spec.MICROBATCH=256; DAggerLearner(a).update(current,r,512)
  spec.MICROBATCH=3072; DAggerLearner(b).update(current,r2,512)
 finally: spec.MICROBATCH=old
 assert all(torch.allclose(x,y,atol=2e-6) for x,y in zip(a.parameters(),b.parameters()))

def reward_frame():
 return dict(body_velocity=torch.zeros(1,3),world_velocity=torch.zeros(1,3),angular_velocity=torch.zeros(1,3),projected_gravity=torch.tensor([[0.,0.,-1.]]),command=torch.zeros(1),g_level=torch.ones(1),base_z=torch.tensor([.4]),base_surface_height=torch.zeros(1),base_surface_valid=torch.ones(1,dtype=torch.bool),q=torch.zeros(1,12),q0=torch.zeros(12),qdot=torch.zeros(1,12),previous_qdot=torch.zeros(1,12),action=torch.zeros(1,12),previous_action=torch.zeros(1,12),q_applied=torch.zeros(1,12),hard_limits=torch.tensor([[-1.,1.]]*12),reward_contact=torch.ones(1,4,dtype=torch.bool),foot_velocity=torch.zeros(1,4,3),foot_site_z=torch.full((1,4),.032),foot_surface_height=torch.zeros(1,4),foot_surface_valid=torch.ones(1,4,dtype=torch.bool),foot_is_stairs=torch.zeros(1,4,dtype=torch.bool),airtime_reward=torch.zeros(1),local_y=torch.zeros(1),yaw_error=torch.zeros(1),heading_speed=torch.zeros(1),stable=torch.ones(1,dtype=torch.bool),cruise=torch.zeros(1,dtype=torch.bool),delta_x=torch.zeros(1),top=torch.zeros(1,dtype=torch.bool),success=torch.zeros(1,dtype=torch.bool),failure=torch.zeros(1,dtype=torch.bool))

def test_reward_hand_calculations_and_failure_priority():
 from src.tasks.velocity.v10.rewards import compute_reward
 x=reward_frame(); r,c=compute_reward(x)
 assert torch.allclose(r,torch.tensor([.02*(1.5+.8+1)]))
 x['stable'][:]=False; x['q'][0,0]=1.
 r,c=compute_reward(x); assert torch.allclose(c['raw']['soft_limits'],torch.tensor([.1]))
 x['base_surface_valid'][:]=False
 _,c=compute_reward(x); assert c['raw']['height'].item()==2.25
 x['top'][:]=True;x['success'][:]=True;x['failure'][:]=True
 _,c=compute_reward(x);assert c['weighted']['top'].item()==0 and c['weighted']['success'].item()==0 and c['weighted']['failure'].item()==-10

def test_all_reward_terms_hand_values():
 from src.tasks.velocity.v10.rewards import compute_reward
 x=reward_frame();x.update(body_velocity=torch.tensor([[.5,.2,.25]]),angular_velocity=torch.tensor([[.3,.4,.5]]),projected_gravity=torch.tensor([[.1,.2,-.97]]),command=torch.tensor([.3]),g_level=torch.tensor([.5]),q=torch.ones(1,12),qdot=torch.full((1,12),.2),action=torch.ones(1,12),local_y=torch.tensor([.4]),yaw_error=torch.tensor([torch.pi]),heading_speed=torch.tensor([.6]),foot_velocity=torch.ones(1,4,3),airtime_reward=torch.tensor([.3]),delta_x=torch.tensor([.2]),cruise=torch.ones(1,dtype=torch.bool))
 _,c=compute_reward(x);t=c['raw']
 expected=dict(velocity=torch.exp(torch.tensor(-.08/.25)).item(),yaw_velocity=torch.exp(torch.tensor(-1.)).item(),vertical_velocity=.01,roll=.04,pitch=.005,height=0,action_rate=12,posture=12,joint_acceleration=1200,joint_velocity=.48,angular_xy=.25,airtime=.3,slip=8,foot_clearance=0,tracking=12,lateral=4,heading=2,soft_limits=1.2,overspeed=.01125,stable=0)
 for k,v in expected.items():assert abs(t[k].item()-v)<1e-4,(k,t[k],v)
 assert abs(c['weighted']['displacement'].item()-.08)<1e-6
 x['reward_contact'][:]=False;_,c=compute_reward(x)
 assert abs(c['raw']['foot_clearance'].item()-4*(.075**2)*(2**.5))<1e-6

def test_teacher_construction_does_not_freeze_student():
 from src.tasks.velocity.v10.model import V10ActorCritic
 from src.tasks.velocity.v10.teachers import DualTeacher
 model=V10ActorCritic.from_stairs(); before=[p.requires_grad for p in model.parameters()]
 teachers=DualTeacher(model)
 assert [p.requires_grad for p in model.parameters()]==before
 assert all(not p.requires_grad for p in teachers.parameters())

def test_ppo_p0_actor_frozen_and_p1_keep_relabels():
 from src.tasks.velocity.v10.ppo import PPOLearner
 class Small(torch.nn.Module):
  def __init__(self):
   super().__init__();self.actor=torch.nn.Linear(45,12);self.critic=torch.nn.Linear(45,1);self.std=torch.nn.Parameter(torch.full((12,),.1))
  def forward(self,obs): return self.actor(obs['proprio'])
  def evaluate(self,obs):return self.critic(obs['proprio'])
  def action_distribution(self,obs):return torch.distributions.Normal(self(obs),self.std.clamp(.05,.30))
  def constrain_std(self):
   with torch.no_grad():self.std.clamp_(.05,.30)
  def set_phase(self,p):
   self.actor.requires_grad_(p=='P1');self.critic.requires_grad_(True);self.std.requires_grad_(p=='P1')
 torch.set_num_threads(1); torch.manual_seed(42);m=Small();k=copy.deepcopy(m).requires_grad_(False)
 r=TaskReservoir(capacity=5);old=sample(60);old['obs']['proprio']/=1000;old['actions'].fill_(1e5);r.add(old)
 o={'proprio':torch.randn(16,45)}
 with torch.no_grad():
  dist=m.action_distribution(o);actions=dist.sample();logp=dist.log_prob(actions).sum(-1);value=m.evaluate(o).flatten()
 b=dict(obs=o,actions=actions,log_prob=logp,values=value,returns=value+1,advantages=torch.randn(16))
 learner=PPOLearner(m);actor=copy.deepcopy(m.actor.state_dict());std=m.std.detach().clone();learner.update(b,k,r,'P0')
 assert all(torch.equal(t,actor[n]) for n,t in m.actor.state_dict().items());assert torch.equal(m.std,std)
 metrics=learner.update(b,k,r,'P1');assert metrics['keep']<1 and m.std.min()>=.05 and m.std.max()<=.30

def test_integer_metadata_roundtrips_without_float32_loss():
 r=TaskReservoir(capacity=10);b=sample(6);b['episode']=torch.arange(6,dtype=torch.int64)+2**40;r.add(b)
 s=r.sample_per_task(1);assert s['episode'].dtype==torch.int64
 assert sorted(s['episode'].tolist())==b['episode'].tolist()

def test_ppo_microbatch_full_policy_value_entropy_keep_equivalence():
 from src.tasks.velocity.v10.ppo import PPOLearner
 from src.tasks.velocity.v10 import spec
 class Small(torch.nn.Module):
  def __init__(self):
   super().__init__();self.actor=torch.nn.Linear(45,12);self.critic=torch.nn.Linear(45,1);self.std=torch.nn.Parameter(torch.full((12,),.1))
  def forward(self,o):return self.actor(o['proprio'])
  def evaluate(self,o):return self.critic(o['proprio'])
  def action_distribution(self,o):return torch.distributions.Normal(self(o),self.std.clamp(.05,.30))
  def constrain_std(self):
   with torch.no_grad():self.std.clamp_(.05,.30)
  def set_phase(self,p):self.requires_grad_(True)
 torch.set_num_threads(1);torch.manual_seed(5);a=Small();b=copy.deepcopy(a);keep=copy.deepcopy(a).requires_grad_(False)
 with torch.no_grad():keep.actor.bias.add_(.1)
 r=TaskReservoir(capacity=10,seed=7);old=sample(60);old['obs']['proprio']/=1000;r.add(old)
 r2=TaskReservoir(capacity=10);r2.load_state_dict(r.state_dict())
 o={'proprio':torch.randn(64,45)}
 with torch.no_grad():
  d=a.action_distribution(o);actions=d.sample();logp=d.log_prob(actions).sum(-1);values=a.evaluate(o).flatten()
 rollout=dict(obs=o,actions=actions,log_prob=logp,values=values,returns=values+torch.randn(64),advantages=torch.randn(64))
 orig=spec.MICROBATCH
 try:
  spec.MICROBATCH=3;PPOLearner(a,seed=42).update(rollout,keep,r,'P1')
  spec.MICROBATCH=1536;PPOLearner(b,seed=42).update(rollout,keep,r2,'P1')
 finally:spec.MICROBATCH=orig
 assert all(torch.allclose(x,y,atol=2e-6,rtol=2e-5) for x,y in zip(a.parameters(),b.parameters()))

def test_ppo_initialization_preserves_actual_cuda_rng():
 import pytest
 from src.tasks.velocity.v10.model import V10ActorCritic
 if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
 torch.cuda.manual_seed_all(123)
 before=torch.cuda.get_rng_state_all();m=V10ActorCritic.from_stairs();m.initialize_ppo()
 assert all(torch.equal(a,b) for a,b in zip(before,torch.cuda.get_rng_state_all()))
