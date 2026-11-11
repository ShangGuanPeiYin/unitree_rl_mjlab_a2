"""Actor-only ONNX export and numerical verification."""
from pathlib import Path
import numpy as np
import torch

def export_actor(actor,path,inputs,tolerance=1e-6):
    import onnxruntime as ort
    path=Path(path)
    if path.exists(): raise FileExistsError(path)
    names=['obs_history','points','mask','ages']
    actor.eval()
    args=tuple(inputs[n] for n in names)
    torch.onnx.export(actor,args,str(path),input_names=names,output_names=['actions'],dynamic_axes={n:{0:'batch'} for n in names+['actions']},opset_version=17,dynamo=False)
    session=ort.InferenceSession(str(path),providers=['CPUExecutionProvider'])
    actual=session.run(None,{item.name:inputs[item.name].detach().cpu().numpy() for item in session.get_inputs()})[0]
    with torch.no_grad(): expected=actor(*args).cpu().numpy()
    error=float(np.max(np.abs(actual-expected)))
    if error>tolerance: raise ValueError(f'ONNX maximum action error {error} exceeds {tolerance}')
    return error
