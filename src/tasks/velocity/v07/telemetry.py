"""v0.6 telemetry: W&B panels for training, reward terms, PPO group gradients,
curriculum, gates, retention sets and resources.

The telemetry layer is read-only: it never feeds a value back into the training
math and it never changes a counter.  When W&B is unavailable or deliberately
disabled the same records are written to a local JSON-lines file so evidence is
still complete (SDS 7.1 ``telemetry.py``/``connection.py`` row; the reliable
delivery transport itself is owned by the coordination module).
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import time

import torch

from . import model as model_module
from . import speed

#: Required dashboard namespaces (SDS 7.1 telemetry row).
PANELS = ('train', 'reward', 'ppo', 'ppo_grad', 'curriculum', 'gate', 'retention',
          'resource')
REWARD_TERMS = ('tracking_lin_vel', 'tracking_ang_vel', 'lin_vel_z', 'orientation_roll',
                'base_height', 'action_rate', 'similar_to_default', 'dof_acc', 'dof_vel',
                'ang_vel_xy', 'feet_air_time', 'foot_slip', 'foot_clearance',
                'joint_tracking', 'center', 'heading', 'joint_limits')
GRADIENT_GROUPS = model_module.GRADIENT_GROUPS


class TelemetryError(ValueError):
    """A metric is not finite/JSON-safe (never silently dropped)."""


def finite_metrics(metrics):
    """Validate and copy a metric mapping; NaN/Inf is a hard error."""
    if not isinstance(metrics, dict):
        raise TypeError('telemetry metrics must be a mapping')
    clean = {}
    for key, value in metrics.items():
        if torch.is_tensor(value):
            value = value.detach().to('cpu')
            if value.numel() != 1:
                raise TelemetryError(f'metric {key!r} is not scalar')
            value = float(value.reshape(-1)[0])
        if isinstance(value, bool):
            clean[str(key)] = bool(value)
            continue
        if isinstance(value, (int, float)):
            if isinstance(value, float) and not math.isfinite(value):
                raise TelemetryError(f'metric {key!r} is not finite')
            clean[str(key)] = float(value) if isinstance(value, float) else int(value)
            continue
        if isinstance(value, str) or value is None:
            clean[str(key)] = value
            continue
        if isinstance(value, dict):
            clean[str(key)] = finite_metrics(value)
            continue
        if isinstance(value, (list, tuple)):
            clean[str(key)] = [finite_metrics({'item': item})['item'] if isinstance(item, dict)
                               else item for item in value]
            continue
        raise TelemetryError(f'metric {key!r} has unsupported type {type(value).__name__}')
    return clean


def flatten_numeric(record, prefix=''):
    """Flatten nested numeric metrics into ``a/b/c`` names (panels use this)."""
    flat = {}
    for key, value in finite_metrics(record).items():
        name = f'{prefix}/{key}' if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_numeric(value, name))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            flat[name] = value
    return flat


def reward_panel(metrics):
    terms = dict(metrics.get('reward_terms') or {})
    return {f'reward/{name}': float(terms.get(name, 0.0)) for name in REWARD_TERMS
            if name in terms}


def gradient_panel(diagnostics):
    panel = {}
    for group in GRADIENT_GROUPS:
        block = diagnostics.get(f'grad_norm_{group}')
        if isinstance(block, dict):
            for statistic in ('min', 'mean', 'max'):
                if statistic in block:
                    panel[f'ppo_grad/{group}/{statistic}'] = float(block[statistic])
    if 'gradient_clip_ratio' in diagnostics:
        panel['ppo_grad/clip_ratio'] = float(diagnostics['gradient_clip_ratio'])
    return panel


class WandbBackend:
    """Minimal W&B adapter; ``wandb`` is imported lazily and optionally."""

    def __init__(self, project='a2-ideal-stairs-v07', entity=None, name=None, enabled=True):
        self.project = project
        self.entity = entity
        self.name = name
        self.enabled = bool(enabled)
        self.run = None
        self.reason = None

    def start(self, run_id, config):
        if not self.enabled:
            self.reason = 'disabled'
            return None
        try:
            import wandb
        except ImportError:
            self.enabled = False
            self.reason = 'wandb_not_installed'
            return None
        self.run = wandb.init(project=self.project, entity=self.entity, name=self.name or run_id,
                              id=run_id, config=config, resume='allow')
        return self.run

    def log(self, record, step=None):
        if self.run is None:
            return False
        self.run.log(record, step=step)
        return True

    def finish(self, summary):
        if self.run is not None:
            self.run.summary.update(summary)
            self.run.finish()
            self.run = None
            return True
        return False


class V07Telemetry:
    """Panels + local evidence; never modifies training state."""

    def __init__(self, directory=None, backend=None, *, clock=time.time):
        self.directory = Path(directory) if directory else None
        self.backend = backend
        self.clock = clock
        self.records = []
        self.start_time = float(clock())
        self.session = None

    def panels(self):
        return dict(panels=list(PANELS), reward_terms=list(REWARD_TERMS),
                    ppo_gradient_groups=list(GRADIENT_GROUPS),
                    identity=dict(task=model_module.TASK_ID,
                                  recipe=model_module.RECIPE_VERSION,
                                  code_version=model_module.CODE_VERSION,
                                  architecture=model_module.ARCHITECTURE),
                    gate=dict(interval=int(speed.GATE_INTERVAL),
                              cases=int(speed.GATE_CASES),
                              pass_min=int(speed.GATE_PASS_MIN)),
                    retention=dict(interval=int(speed.RETENTION_INTERVAL),
                                   cases=int(speed.RETENTION_CASES),
                                   fail_cases=int(speed.RETENTION_FAIL_CASES)),
                    resource=dict(limits=dict(speed_limits())))

    def start(self, run_id, config):
        self.session = dict(run_id=str(run_id), config=deepcopy(config),
                            started=float(self.clock()))
        if self.backend is not None:
            self.backend.start(run_id, config)
        return self.session

    def update(self, metrics, boundary=None):
        record = dict(kind='update', timestamp=float(self.clock()),
                      train=finite_metrics({key: value for key, value in metrics.items()
                                            if not isinstance(value, dict)}),
                      reward=reward_panel(metrics),
                      ppo_grad=gradient_panel(metrics),
                      curriculum=dict(update=metrics.get('completed_update'),
                                      phase=(boundary or {}).get('phase'),
                                      substage=(boundary or {}).get('substage'),
                                      actions=list((boundary or {}).get('actions', []))))
        self._emit(record)
        return record

    def gate(self, gate):
        record = dict(kind='gate', timestamp=float(self.clock()),
                      gate=finite_metrics({key: value for key, value in gate.items()
                                           if not isinstance(value, (dict, list))}))
        self._emit(record)
        return record

    def retention(self, retention):
        record = dict(kind='retention', timestamp=float(self.clock()),
                      retention=finite_metrics({key: value for key, value in retention.items()
                                                if not isinstance(value, (dict, list))}))
        self._emit(record)
        return record

    def resource(self, snapshot):
        record = dict(kind='resource', timestamp=float(self.clock()),
                      resource=finite_metrics({key: value for key, value in snapshot.items()
                                               if isinstance(value, (int, float))}))
        self._emit(record)
        return record

    def _emit(self, record):
        self.records.append(record)
        flat = flatten_numeric({key: value for key, value in record.items()
                                if key not in ('kind', 'timestamp')})
        if self.backend is not None and hasattr(self.backend, 'log'):
            self.backend.log(flat, step=record.get('train', {}).get('update'))
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / 'telemetry.jsonl'
            with path.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')
        return flat

    def finish(self, result):
        summary = dict(state=result.get('state'),
                       failure_class=result.get('failure_class'),
                       update=result.get('update'),
                       phase=result.get('phase'),
                       budget=result.get('budget'),
                       duration_s=float(self.clock() - self.start_time))
        if self.backend is not None:
            self.backend.finish(summary)
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)
            (self.directory / 'summary.json').write_text(
                json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        return summary


def speed_limits():
    from .orchestration import resource_limits
    return resource_limits()[0]


__all__ = ['GRADIENT_GROUPS', 'PANELS', 'REWARD_TERMS', 'TelemetryError', 'V07Telemetry',
           'WandbBackend', 'finite_metrics', 'flatten_numeric', 'gradient_panel',
           'reward_panel', 'speed_limits']
