"""S2 resource admission: 1024 first, documented fallback to 512 (§2.3)."""
from __future__ import annotations

from dataclasses import dataclass

from . import s2_spec

PROBE_EVAL_TIMEOUT_STEPS = 9472  # batch hard-timeout inclusive; 9471 is last index
REQUIREMENTS = dict(gpu_peak_mib=7168, host_free_mib=4096,
                    warmups=5, measurements=50, restore_checks=1,
                    rollout_steps=48, adam_updates_per_rollout=20,
                    independent_copy=True, allow_oom=False,
                    allow_nonfinite=False, allow_contact_overflow=False,
                    evaluation_envs=100)


@dataclass(frozen=True)
class ProbeResult:
    num_envs: int
    peak_mib: float
    host_free_mib: float
    oom: bool
    nonfinite: bool
    contact_overflow: bool
    warmups: int
    measurements: int
    restore_checks: int
    independent_copy: bool
    evaluation_envs: int
    evidence_sha256: str | None = None

    def admitted(self):
        return all((
            self.peak_mib <= REQUIREMENTS['gpu_peak_mib'],
            self.host_free_mib >= REQUIREMENTS['host_free_mib'],
            not self.oom, not self.nonfinite, not self.contact_overflow,
            self.warmups >= REQUIREMENTS['warmups'],
            self.measurements >= REQUIREMENTS['measurements'],
            self.restore_checks >= REQUIREMENTS['restore_checks'],
            self.independent_copy,
            self.evaluation_envs == REQUIREMENTS['evaluation_envs'],
        ))

    def failures(self):
        reasons = []
        if self.peak_mib > REQUIREMENTS['gpu_peak_mib']:
            reasons.append('gpu_peak_mib')
        if self.host_free_mib < REQUIREMENTS['host_free_mib']:
            reasons.append('host_free_mib')
        if self.oom:
            reasons.append('oom')
        if self.nonfinite:
            reasons.append('nonfinite')
        if self.contact_overflow:
            reasons.append('contact_overflow')
        if self.warmups < REQUIREMENTS['warmups']:
            reasons.append('warmups')
        if self.measurements < REQUIREMENTS['measurements']:
            reasons.append('measurements')
        if self.restore_checks < REQUIREMENTS['restore_checks']:
            reasons.append('restore')
        if not self.independent_copy:
            reasons.append('independent_copy')
        if self.evaluation_envs != REQUIREMENTS['evaluation_envs']:
            reasons.append('evaluation_envs')
        return reasons


def probe_plan(num_envs, *, evaluation_timeout_steps=PROBE_EVAL_TIMEOUT_STEPS):
    if num_envs not in s2_spec.S2_ENV_CANDIDATES:
        raise ValueError('S2 probe is 1024 then 512 only')
    samples = num_envs * REQUIREMENTS['warmups'] * REQUIREMENTS['rollout_steps']
    samples += num_envs * REQUIREMENTS['measurements'] * REQUIREMENTS['rollout_steps']
    samples += num_envs * REQUIREMENTS['rollout_steps']  # restore rollout
    samples += REQUIREMENTS['evaluation_envs'] * int(evaluation_timeout_steps)
    samples += 1 * int(evaluation_timeout_steps)  # one-env video process
    return dict(num_envs=num_envs, requirements=dict(REQUIREMENTS),
                evaluation_timeout_steps=int(evaluation_timeout_steps),
                evaluation_samples=REQUIREMENTS['evaluation_envs'] * int(evaluation_timeout_steps),
                video_samples=int(evaluation_timeout_steps),
                control_samples=samples, physics_substeps=4 * samples,
                warmups=REQUIREMENTS['warmups'],
                measurements=REQUIREMENTS['measurements'],
                restore_checks=REQUIREMENTS['restore_checks'])


def choose_resource(probes, *, require_1024_failure_evidence=True):
    """Return 1024, or 512 only after an explicit 1024 failure record."""
    probes = list(probes)
    if require_1024_failure_evidence:
        if not probes or probes[0].num_envs != 1024:
            raise ValueError('512 fallback requires a 1024 probe first')
        if probes[0].admitted():
            raise ValueError('1024 already admitted; 512 fallback is not allowed')
    evidence = []
    for result in probes:
        evidence.append(dict(num_envs=result.num_envs, admitted=result.admitted(),
                             failures=result.failures(),
                             evidence_sha256=result.evidence_sha256))
        if result.admitted():
            return dict(selected=result.num_envs, fallback_evidence=evidence)
    return dict(selected=None, fallback_evidence=evidence,
                reason='both 1024 and 512 failed; stop and report')


def run_probe(*, num_envs, device='cuda:0', output_dir=None,
              independent_copy=True):
    """Measured §2.3 probe against the real S2TrainEnv (GPU call only).

    Five warmups, fifty measured full 48-step/20-Adam rollouts and one
    save→load restore.  The caller should run this only in an approved GPU
    window; no ledger write happens here.
    """
    import json
    import time
    from pathlib import Path
    import torch
    try:
        import psutil
        host_free = psutil.virtual_memory().available / 2 ** 20
    except Exception:
        host_free = None
    from . import s2_physics, s2_train, s2_train_real
    env = s2_train_real.S2TrainEnv(num_envs=num_envs, device=device)
    actor = s2_physics.load_s1_actor(device=device)
    trainer = s2_train.S2Trainer(model=actor['model'], env=env, device=device,
                                 attempt_id=f'probe-{num_envs}')
    timings = []
    oom = nonfinite = contact_overflow = False
    peak_mib = 0.0
    try:
        torch.cuda.reset_peak_memory_stats()
        for _ in range(REQUIREMENTS['warmups']):
            trainer.train_rollout()
        for _ in range(REQUIREMENTS['measurements']):
            started = time.perf_counter()
            trainer.train_rollout()
            timings.append(time.perf_counter() - started)
            peak_mib = max(peak_mib, torch.cuda.max_memory_allocated() / 2 ** 20)
        import tempfile
        env.close()
        env = None
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'probe.pt'
            trainer.save(path)
            restored = s2_train.S2Trainer(
                model=s2_physics.load_s1_actor(device=device)['model'],
                env=s2_train_real.S2TrainEnv(num_envs=num_envs, device=device),
                device=device, attempt_id=f'probe-{num_envs}-restore')
            restored.resume(path, device=device)
            restored.env.close()
            restore_checks = 1
        admitted = (peak_mib <= REQUIREMENTS['gpu_peak_mib']
                    and (host_free is None or host_free >= REQUIREMENTS['host_free_mib'])
                    and not oom and not nonfinite and not contact_overflow)
    except RuntimeError as error:
        oom = 'out of memory' in str(error).lower()
        nonfinite = 'nonfinite' in str(error).lower()
        contact_overflow = 'contact' in str(error).lower()
        admitted = False
        restore_checks = 0
        peak_mib = (torch.cuda.max_memory_allocated() / 2 ** 20
                    if torch.cuda.is_available() else 0.0)
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    report = dict(
        num_envs=num_envs, admitted=bool(admitted), peak_mib=peak_mib,
        host_free_mib=host_free, oom=oom, nonfinite=nonfinite,
        contact_overflow=contact_overflow,
        warmups=REQUIREMENTS['warmups'], measurements=len(timings),
        restore_checks=restore_checks, independent_copy=bool(independent_copy),
        evaluation_envs=0,
        elapsed_s=float(sum(timings)), timings_s=timings,
        requirements=dict(REQUIREMENTS),
        note='real S2TrainEnv probe; evaluation/video separate from full training')
    if output_dir is not None:
        path = Path(output_dir) / 'probe_report.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return report
