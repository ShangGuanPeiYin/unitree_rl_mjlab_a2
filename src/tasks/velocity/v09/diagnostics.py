"""Offline paired S1.1 evidence on saved physical states and 6400 labels."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import math
from . import spec
import torch


def loss_report(model, folder, manifest, *, value_scale, device, code_sha256, dataset_checkpoint_sha256):
    """Evaluate a fixed saved state set without advancing a simulator."""
    if not isinstance(value_scale, (float, int)) or not math.isfinite(value_scale) or value_scale < 1:
        raise ValueError('finite value_scale >= 1 required')
    summary = json.loads((Path(folder) / 'summary.json').read_text())
    expected_identity = dict(sds_sha256=spec.SDS_SHA256, code_sha256=code_sha256,
        manifest_sha256=manifest['manifest_sha256'], policy_checkpoint_sha256=dataset_checkpoint_sha256,
        teacher_source_sha256=spec.SOURCE_SHA256, suite='S1_DIAG')
    if (summary.get('checkpoint_sha256') != dataset_checkpoint_sha256
            or summary.get('code_sha256') != code_sha256
            or summary.get('manifest_sha256') != manifest['manifest_sha256']):
        raise ValueError('diagnostic summary source identity mismatch')
    lookup = {case['case_sha256']: case for case in manifest['cases']}
    result = {}
    samples = 0
    seen_cases = set()
    model.eval()
    with torch.no_grad():
        for path in sorted(Path(folder).glob('batch_*.pt')):
            batch = torch.load(path, map_location='cpu', weights_only=False)
            if batch.get('identity') != expected_identity:
                raise ValueError('diagnostic batch source identity mismatch')
            identities = batch['case_sha256s']
            if len(identities) != len(set(identities)) or not set(identities) <= lookup.keys():
                raise ValueError('invalid batch case mapping')
            certificate = dict(batch['geometry'])
            checksum = certificate.pop('certificate_sha256')
            encoded = json.dumps(certificate, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
            if (checksum != hashlib.sha256(encoded).hexdigest()
                    or certificate.get('validated') is not True
                    or certificate.get('validator') != 'v07.terrain.index_geometry'
                    or certificate.get('code_sha256') != code_sha256
                    or certificate.get('sds_sha256') != spec.SDS_SHA256):
                raise ValueError('compiled geometry certificate invalid')
            for snapshot in batch['diagnostic_samples']:
                slots = snapshot['slots']
                if (slots.ndim != 1 or slots.dtype != torch.long or slots.numel() < 1
                        or slots.unique().numel() != slots.numel()
                        or int(slots.min()) < 0 or int(slots.max()) >= len(identities)):
                    raise ValueError('invalid diagnostic sample slots')
                size = slots.numel()
                if (snapshot['teacher_mean'].shape != (size, 12)
                        or snapshot['teacher_value'].shape != (size, 1)
                        or any(value.shape[0] != size for value in snapshot['student_observation'].values())):
                    raise ValueError('diagnostic labels/observations must match exact sample count')
                observations = {key: value.to(device) for key, value in snapshot['student_observation'].items()}
                mean = model(observations).cpu()
                value = model.evaluate(observations).cpu()
                if not all(torch.isfinite(tensor).all() for tensor in (mean, value, snapshot['teacher_mean'], snapshot['teacher_value'])):
                    raise FloatingPointError('nonfinite saved-state diagnostic outputs')
                action_loss = (mean - snapshot['teacher_mean']).square().mean(-1)
                value_loss = ((value - snapshot['teacher_value']) / value_scale).square().flatten()
                if not torch.isfinite(action_loss).all() or not torch.isfinite(value_loss).all():
                    raise FloatingPointError('nonfinite diagnostic loss')
                for index, slot in enumerate(snapshot['slots'].tolist()):
                    identity = batch['case_sha256s'][slot]
                    case = lookup[identity]
                    key = f"{case['task']}/{case['lesson']}/vx={case['speed']:.1f}/{case['stop_mode']}"
                    entry = result.setdefault(key, dict(samples=0, action_loss=0., value_loss=0.))
                    entry['samples'] += 1
                    entry['action_loss'] += float(action_loss[index])
                    entry['value_loss'] += float(value_loss[index])
                    samples += 1
                    seen_cases.add(identity)
    if seen_cases != lookup.keys() or not samples:
        raise ValueError('diagnostic states must cover every frozen case')
    for entry in result.values():
        entry['action_loss'] /= entry['samples']
        entry['value_loss'] /= entry['samples']
    return dict(samples=samples, case_count=len(seen_cases), groups=result)
