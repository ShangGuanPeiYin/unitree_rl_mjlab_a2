"""v08 finite local panels and optional backend seam; remote ACK is separate."""
from copy import deepcopy
import json
from pathlib import Path
import time
from .recovery import jsonable

PANELS=('train','reward','ppo','ppo_grad','curriculum','gate','retention','resource')
def flatten_numeric(value,prefix=''):
    out={}
    if isinstance(value,dict):
        for key,item in value.items():out.update(flatten_numeric(item,f'{prefix}/{key}' if prefix else str(key)))
    elif isinstance(value,(float,int,bool)):out[prefix]=value
    return out

class WandbBackend:
    def __init__(self,project='a2-ideal-stairs-v08',entity=None,name=None,enabled=True):
        self.project,self.entity,self.name,self.enabled=project,entity,name,bool(enabled);self.run=None
    def start(self,run_id,config):
        if not self.enabled:return None
        import wandb
        self.run=wandb.init(project=self.project,entity=self.entity,name=self.name or run_id,id=run_id,
            group='v08',config=config,resume='allow')
        return self.run
    def log(self,record,step=None):
        if self.run is None:raise RuntimeError('W&B run has not started')
        self.run.log(record,step=step)
    def finish(self,summary):
        if self.run is not None:
            self.run.summary.update(summary);self.run.finish();self.run=None

class V08Telemetry:
    def __init__(self,directory=None,backend=None,*,clock=time.time):
        self.directory=Path(directory) if directory is not None else None
        if self.directory:self.directory.mkdir(parents=True,exist_ok=True)
        self.backend,self.clock=backend,clock;self.records=[];self.session=None;self.start_time=clock()
    def panels(self):return dict(panels=list(PANELS))
    def _emit(self,record):
        raw=json.dumps(jsonable(record),sort_keys=True,allow_nan=False)
        self.records.append(deepcopy(record))
        if self.directory:
            with (self.directory/'telemetry.jsonl').open('a') as stream:stream.write(raw+'\n');stream.flush()
        return flatten_numeric(jsonable(record))
    def emit(self,kind,payload):return self._emit(dict(kind=kind,payload=payload,online_status='disabled'))
    def start(self,run_id,config):
        self.session=dict(run_id=run_id,config=deepcopy(config),started=self.clock())
        if self.backend is not None:self.backend.start(run_id,config)
        self._emit(dict(kind='start',session=self.session,remote_ack=False))
        return self.session
    def update(self,payload):return self._emit(dict(kind='update',train=payload))
    def gate(self,payload):return self._emit(dict(kind='gate',gate=payload))
    def finish(self,payload):
        summary=dict(payload,duration_s=self.clock()-self.start_time)
        self._emit(dict(kind='finish',summary=summary))
        if self.directory:(self.directory/'summary.json').write_text(json.dumps(jsonable(summary),sort_keys=True,allow_nan=False)+'\n')
        return summary
