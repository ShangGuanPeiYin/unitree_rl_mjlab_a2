"""Update-boundary checkpoints deliberately exclude simulator and sensory history."""
import random
from pathlib import Path
import numpy as np
import torch

def save_checkpoint(path,model,optimizer,curriculum,iteration,config,*,update_complete):
    if not update_complete: raise ValueError('Checkpoint requires completed PPO update')
    path=Path(path)
    if path.exists(): raise FileExistsError(path)
    state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),curriculum=curriculum,iteration=iteration,config=config,torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream: torch.save(state,stream)

def restore_checkpoint(path,model,optimizer,reset_all):
    state=torch.load(path,map_location='cpu',weights_only=False)
    model.load_state_dict(state['model'],strict=True)
    optimizer.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['torch_rng']); np.random.set_state(state['numpy_rng']); random.setstate(state['python_rng'])
    if state['cuda_rng'] is not None: torch.cuda.set_rng_state_all(state['cuda_rng'])
    reset_all()
    return {**state,'event':'resume_reset'}
