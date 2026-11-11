"""Export a trusted local checkpoint's actor; no simulator or training is started."""
import argparse
import json
from pathlib import Path
import torch
from src.tasks.pointcloud.model import PointCloudActorCritic
from src.tasks.pointcloud.export import export_actor

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--tolerance',type=float,default=1e-6)
    a=p.parse_args(); torch.manual_seed(a.seed); torch.set_num_threads(1)
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    model=PointCloudActorCritic(state['model']['critic.0.weight'].shape[1],use_points=state['config'].get('variant','full')!='blind')
    model.load_state_dict(state['model'],strict=True)
    inputs=dict(obs_history=torch.randn(a.batch_size,5,45),points=torch.randn(a.batch_size,5,70,3),mask=torch.rand(a.batch_size,5,70)>.5,ages=torch.tensor([0.,.02,.04,.06,.08]).expand(a.batch_size,-1))
    error=export_actor(model.actor,a.output,inputs,a.tolerance)
    contract=dict(max_abs_error=error,seed=a.seed,inputs={k:list(v.shape) for k,v in inputs.items()},proprio_history='oldest to newest',pointcloud_history='newest to oldest; own sampling body frame; ages include perception delay',voxel_m=.1,roi=[[-1.2,1.2],[-.6,.6],[-1,.5]],points_per_group=70,groups=5,preprocessing='dual lidar balanced 35 each with spare-capacity fill; no duplicate padding; invalid mask false',privileged_inputs=False)
    a.output.with_suffix('.json').write_text(json.dumps(contract,indent=2)+'\n')
    print(json.dumps(contract,indent=2))
if __name__=='__main__': main()
