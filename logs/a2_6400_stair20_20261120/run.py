import sys,json,hashlib,time,collections
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'scripts'))
from a2_v07_bootstrap import activate
activate()
import torch
from src.tasks.velocity.v07 import terrain,events,spec
from src.tasks.velocity.v07.env import V07Env
from src.tasks.velocity.v07.model import V07ActorCritic
out=Path(__file__).resolve().parent
checkpoint=Path('logs/a2_ideal_stairs_v07/continue_seed42_from004817_trial01/checkpoints/stop_006400.pt')
sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
assert sha=='30ed23ba16f7e490afca001fbbecadb4a11ff4b11733a5ae7ac7d0c841f9a8b2'
terrain.configure_playback_platform(tile_id=21,length=20.,width=4.)
model=V07ActorCritic().to('cuda:0')
model.load_state_dict(torch.load(checkpoint,map_location='cuda:0',weights_only=False)['model'],strict=True)
model.eval()
records=[]
for seed in range(1,21):
 torch.manual_seed(seed)
 env=V07Env(num_envs=1,device='cuda:0',seed=seed,phase='L',substage='L40',evaluation=True,fixed_vx=.6,frontier_tile_id=21)
 try:
  initial=env.episode_initial_pose[0].tolist()
  limit=int(events.timeout_steps(env.tile_ids,env.command[:,0])[0])
  start=time.perf_counter()
  with torch.inference_mode():
   for step in range(limit+1):
    env.command[:]=torch.tensor([.6,0.,0.],device='cuda:0')
    obs=env._observations(update=False)
    _,_,_,_,info=env.step(model.act(obs,deterministic=True))
    if info['episodes']:
     record=dict(info['episodes'][0],seed=seed,wall_seconds=time.perf_counter()-start,initial_pose_recorded=initial)
     records.append(record)
     with (out/'episodes.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
     print(json.dumps(record),flush=True)
     break
   else:raise RuntimeError('No original terminal event by timeout')
 finally:env.close()
summary=dict(checkpoint=str(checkpoint),checkpoint_sha256=sha,seeds=list(range(1,21)),num_envs=1,phase='L',substage='L40',tile_id=21,platform_length=20.,platform_width=4.,vx=.6,vy=0.,wz=0.,deterministic=True,spawn=dict(x=spec.SPAWN_X,y=spec.SPAWN_Y,yaw_degrees=spec.SPAWN_YAW_DEGREES),success=sum(r['success'] for r in records),failure=sum(r['failure'] for r in records),timeout=sum(r['timeout'] for r in records),total=len(records),reasons=dict(collections.Counter(r['reason'] for r in records)))
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary),flush=True)
